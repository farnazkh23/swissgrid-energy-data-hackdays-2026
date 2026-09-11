"""Build the separate G4 correlated-Gaussian Challenge 4 table.

This runner never writes ``edh.group_0.g4_submission_v1`` and never invokes
the official submission API.  Point forecasts use only target history at or
before the causal issue time.  Uncertainty is the repository's real G4 layer:
joint OOF residuals, covariance, bias, calibration, and deterministic draws.
"""

from collections import Counter
from datetime import datetime, timedelta, timezone
from math import sqrt
from statistics import median
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, TimestampType


REPO_ROOT = "/Workspace/Users/user32@swissgridlab.onmicrosoft.com/swissgrid-energy-data-hackdays-2026"
REPO_SRC = f"{REPO_ROOT}/src"
EDH_SOURCE_ROOT = "/Workspace/github/dp-light-edh/src"
for path in (REPO_SRC, EDH_SOURCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from edh2026.local_scoring import score_prediction_table  # noqa: E402
from swissgrid_forecaster.real_modeling_v1 import (  # noqa: E402
    _sample,
    _seasonal_persistence_value,
    build_hourly_targets,
)
from swissgrid_forecaster.real_uncertainty_backtest import fit_calibrated_champion  # noqa: E402
from swissgrid_forecaster.target_contract import TARGETS, validate_output_targets  # noqa: E402
from swissgrid_forecaster.time_contract import organizer_timestamp  # noqa: E402
from swissgrid_forecaster.uncertainty_model import (  # noqa: E402
    ResidualPanel,
    sample_distribution,
)


TABLE_NAME = "edh.group_0.g4_submission_correlated_gaussian_v1"
TARGET_COLUMNS = ("timestamp", *TARGETS)
FORECAST_START = datetime(2026, 9, 2, tzinfo=timezone.utc)
FORECAST_END = datetime(2026, 9, 8, 23, tzinfo=timezone.utc)
FORECAST_HOURS = 168
LONG_LEAD_LAGS = (504, 672, 840)
OOF_FOLDS = 12
FIT_WEEKS = 8
CALIBRATION_WEEKS = 2
CALIBRATION_COVERAGE = 0.8
G4_SEED = 2026
SAMPLES = 300


def _rows(dataframe):
    return tuple(
        {
            key: organizer_timestamp(value) if isinstance(value, datetime) else value
            for key, value in row.asDict(recursive=True).items()
        }
        for row in dataframe.toLocalIterator()
    )


def _long_lead_point(series, stamp, issue):
    point, label = _seasonal_persistence_value(series, stamp, issue)
    if label != "latest_causal":
        return point, label
    references = tuple(
        series[stamp - timedelta(hours=lag)]
        for lag in LONG_LEAD_LAGS
        if stamp - timedelta(hours=lag) <= issue
        and stamp - timedelta(hours=lag) in series
    )
    if references:
        return median(references), "long_lead_weekly_ensemble"
    return point, label


def _forecast_window():
    timestamps = tuple(FORECAST_START + timedelta(hours=index) for index in range(FORECAST_HOURS))
    if timestamps[-1] != FORECAST_END:
        raise ValueError("fixed forecast constants do not describe 168 hours")
    return timestamps


def _build_oof_panel(targets, target_series, issue_time):
    """Build 12 fixed-origin weekly OOF folds immediately before issue_time."""
    last_forecast_start = issue_time - timedelta(hours=FORECAST_HOURS - 1)
    first_forecast_start = last_forecast_start - timedelta(hours=(OOF_FOLDS - 1) * FORECAST_HOURS)
    rows = []
    point_forecasts = {}
    fold_starts = []
    for fold_index in range(OOF_FOLDS):
        forecast_start = first_forecast_start + timedelta(hours=fold_index * FORECAST_HOURS)
        fold_issue = forecast_start - timedelta(hours=1)
        fold_starts.append((fold_issue, forecast_start))
        for hour_index in range(FORECAST_HOURS):
            timestamp = forecast_start + timedelta(hours=hour_index)
            means = {}
            for target in TARGETS:
                point, _ = _long_lead_point(target_series[target], timestamp, fold_issue)
                means[target] = point
            point_forecasts[timestamp] = means
            actual = targets[timestamp]
            rows.append((fold_issue, timestamp, timestamp - fold_issue,
                         tuple(actual[target] - means[target] for target in TARGETS)))
    panel = ResidualPanel(TARGETS, tuple(rows))
    return panel, point_forecasts, tuple(fold_starts)


def _validate_submission(dataframe, *, label):
    if dataframe.columns != list(TARGET_COLUMNS):
        raise ValueError(f"{label}: unexpected columns {dataframe.columns}")
    if not isinstance(dataframe.schema["timestamp"].dataType, TimestampType):
        raise ValueError(f"{label}: timestamp must be timestamp type")
    for target in TARGETS:
        data_type = dataframe.schema[target].dataType
        if not isinstance(data_type, ArrayType) or not isinstance(data_type.elementType, IntegerType):
            raise ValueError(f"{label}: {target} must be array<int>")
    rows = tuple(sorted(
        (row.asDict(recursive=True) for row in dataframe.toLocalIterator()),
        key=lambda row: row["timestamp"],
    ))
    if len(rows) != FORECAST_HOURS:
        raise ValueError(f"{label}: expected 168 rows, got {len(rows)}")
    timestamps = tuple(row["timestamp"] for row in rows)
    expected = tuple((stamp + timedelta(hours=index)).replace(tzinfo=None)
                     for index in range(FORECAST_HOURS) for stamp in (FORECAST_START,))
    if timestamps != expected or len(set(timestamps)) != FORECAST_HOURS:
        raise ValueError(f"{label}: timestamps are not exactly Sep 2 00:00 through Sep 8 23:00")
    for row in rows:
        for target in TARGETS:
            values = row[target]
            if values is None or len(values) != SAMPLES:
                raise ValueError(f"{label}: {target} must contain 300 samples")
            if any(value is None or isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"{label}: {target} contains an invalid sample")
    return rows


def _rmse(samples, actuals):
    errors = []
    for timestamp, draws in samples.items():
        for target in TARGETS:
            mean_value = sum(draws[target]) / len(draws[target])
            errors.append((mean_value - actuals[timestamp][target]) ** 2)
    return sqrt(sum(errors) / len(errors))


def _comparison_rows(fold, point_forecasts, model, fit_residuals):
    g4_rows = []
    simple_rows = []
    actuals = {}
    for index, (_, timestamp, _, residuals) in enumerate(fold):
        means = point_forecasts[timestamp]
        actuals[timestamp] = {target: means[target] + residuals[target_index]
                              for target_index, target in enumerate(TARGETS)}
        joint = sample_distribution(model, means, seed=G4_SEED + index, n_samples=SAMPLES)
        simple = {
            target: _sample(means[target], fit_residuals[target],
                            seed=20260911 + index + sum(map(ord, target)) * 1009)
            for target in TARGETS
        }
        g4_rows.append((timestamp, *(list(joint[target]) for target in TARGETS)))
        simple_rows.append((timestamp, *(list(simple[target]) for target in TARGETS)))
    return tuple(g4_rows), tuple(simple_rows), actuals


def _score_comparison(spark, g4_rows, simple_rows, actuals):
    prediction_schema = "timestamp timestamp, CH array<int>, DE array<int>, FR array<int>, IT array<int>"
    realization_schema = "timestamp timestamp, CH double, DE double, FR double, IT double"
    actual_rows = tuple((timestamp, *(actuals[timestamp][target] for target in TARGETS))
                        for timestamp in actuals)
    spark.createDataFrame(g4_rows, schema=prediction_schema).createOrReplaceTempView("g4_cg_comparison_predictions")
    spark.createDataFrame(simple_rows, schema=prediction_schema).createOrReplaceTempView("g4_simple_comparison_predictions")
    spark.createDataFrame(actual_rows, schema=realization_schema).createOrReplaceTempView("g4_comparison_realizations")
    return {
        "g4_default": float(score_prediction_table("g4_cg_comparison_predictions", "g4_comparison_realizations")),
        "simple_default": float(score_prediction_table("g4_simple_comparison_predictions", "g4_comparison_realizations")),
        "g4_rmse_raw": _rmse({row[0]: {target: row[index + 1] for index, target in enumerate(TARGETS)}
                               for row in g4_rows}, actuals),
        "simple_rmse_raw": _rmse({row[0]: {target: row[index + 1] for index, target in enumerate(TARGETS)}
                                   for row in simple_rows}, actuals),
    }


def run_submission(spark):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    validate_output_targets(TARGETS)
    forecast_timestamps = _forecast_window()
    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", *TARGETS)
    targets = build_hourly_targets(_rows(net_dataframe))
    if not targets:
        raise ValueError("no complete causal target history")
    issue_time = max(targets)
    if issue_time >= FORECAST_START:
        raise ValueError("issue_time must precede the official forecast window")
    target_series = {target: {stamp: row[target] for stamp, row in targets.items()}
                     for target in TARGETS}

    panel, oof_points, fold_starts = _build_oof_panel(targets, target_series, issue_time)
    calibrated, demo_folds = fit_calibrated_champion(
        panel, "correlated_gaussian", fit_weeks=FIT_WEEKS,
        calibration_weeks=CALIBRATION_WEEKS, target_coverage=CALIBRATION_COVERAGE,
        seed=G4_SEED, n_samples=SAMPLES,
    )
    # The calibrated helper intentionally does not expose its fit partition;
    # reconstruct the same first-eight-week causal partition for comparison.
    ordered_folds = tuple(tuple(panel.rows[index:index + FORECAST_HOURS])
                          for index in range(0, len(panel.rows), FORECAST_HOURS))
    fit_rows = tuple(row for fold in ordered_folds[:FIT_WEEKS] for row in fold)
    fit_residuals = {
        target: tuple(row[3][TARGETS.index(target)] for row in fit_rows)
        for target in TARGETS
    }

    means_by_timestamp = {}
    fallback_counts = Counter()
    samples_by_timestamp = {}
    for index, timestamp in enumerate(forecast_timestamps):
        means = {}
        for target in TARGETS:
            point, label = _long_lead_point(target_series[target], timestamp, issue_time)
            means[target] = point
            if target == TARGETS[0]:
                fallback_counts[label] += 1
        means_by_timestamp[timestamp] = means
        draws = sample_distribution(calibrated, means, seed=G4_SEED + index, n_samples=SAMPLES)
        samples_by_timestamp[timestamp] = draws

    rows = tuple((timestamp, *(list(samples_by_timestamp[timestamp][target]) for target in TARGETS))
                 for timestamp in forecast_timestamps)
    schema = "timestamp timestamp, CH array<int>, DE array<int>, FR array<int>, IT array<int>"
    dataframe = spark.createDataFrame(rows, schema=schema).select(*TARGET_COLUMNS)
    validated = _validate_submission(dataframe, label="generated")
    for index, timestamp in enumerate(forecast_timestamps):
        repeat = sample_distribution(calibrated, means_by_timestamp[timestamp],
                                     seed=G4_SEED + index, n_samples=SAMPLES)
        if repeat != samples_by_timestamp[timestamp]:
            raise ValueError("G4 sampling is not deterministic")

    g4_rows, simple_rows, actuals = _comparison_rows(
        demo_folds[0], oof_points, calibrated, fit_residuals)
    scores = _score_comparison(spark, g4_rows, simple_rows, actuals)
    model_covariance = calibrated.covariance_for(None)
    print("G4 CORRELATED-GAUSSIAN PREFLIGHT")
    print(f"- latest actual / issue_time: {issue_time}")
    print(f"- forecast start/end: {FORECAST_START} / {FORECAST_END}")
    print(f"- OOF rows/folds: {len(panel.rows)} / {len(fold_starts)}")
    print(f"- OOF fold issue range: {fold_starts[0][0]} through {fold_starts[-1][0]}")
    print(f"- target order: {TARGETS}")
    print(f"- fallback counts: {dict(fallback_counts)}")
    print(f"- G4 fit/calibration weeks: {FIT_WEEKS} / {CALIBRATION_WEEKS}")
    print(f"- calibration coverage: {CALIBRATION_COVERAGE}")
    print(f"- G4 bias vector: {calibrated.bias_for(None)}")
    print(f"- G4 covariance matrix: {model_covariance}")
    print(f"- G4 calibration scale: {calibrated.scale_for(None)}")
    print(f"- local comparison scores: {scores}")
    for target in TARGETS:
        values = [value for timestamp in forecast_timestamps for value in samples_by_timestamp[timestamp][target]]
        print(f"- {target} sample mean/min/max: {sum(values) / len(values):.3f}/{min(values)}/{max(values)}")
    print(f"- generated validator: PASS ({len(validated)} rows, 300 joint samples each)")

    # This is deliberately a distinct table; the default G4 table is untouched.
    dataframe.write.mode("overwrite").saveAsTable(TABLE_NAME)
    stored = spark.table(TABLE_NAME).select(*TARGET_COLUMNS)
    _validate_submission(stored, label="stored")
    print(f"G4 TABLE READY: {TABLE_NAME}")
    return stored


submission_table = run_submission(spark)
