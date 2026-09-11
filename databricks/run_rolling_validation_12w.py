"""Databricks entrypoint for the 12-week organizer-style validation harness.

This runner reads the visible group-0 validation realizations, trains only on
history available before each fixed weekly issue time, writes rolling harness
artifacts, and never calls an official submission helper.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
from collections import defaultdict

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

REPO_ROOT = "/Workspace/Users/user32@swissgridlab.onmicrosoft.com/swissgrid-energy-data-hackdays-2026"
REPO_SRC = f"{REPO_ROOT}/src"
EDH_SOURCE_ROOT = "/Workspace/github/dp-light-edh/src"
for path in (REPO_SRC, EDH_SOURCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from swissgrid_forecaster.real_modeling_v1 import LSEGFeatures, _hourly_table, _run_scenario, build_hourly_targets
from swissgrid_forecaster.time_contract import organizer_timestamp, source_query_bounds
from swissgrid_forecaster.rolling_validation import (
    HOUR, POINT_MODELS, TRAIN_HOURS, VALIDATION_WEEKS, WARM_START_WEEKS, history_requirements,
    build_validation_folds, require_history, run_rolling_validation,
    validate_validation_folds,
)
from swissgrid_forecaster.target_contract import INPUT_COUNTRIES, TARGETS
from edh2026.local_scoring import score_prediction_table
from swissgrid_forecaster.uncertainty_model import ResidualPanel, fit_uncertainty_model, sample_distribution

UTC = timezone.utc
VALIDATION_TABLE = "edh.group_0.g4_validation_realizations"
GENERATION_ALLOWED_FIELD = "generation_forecast"
GENERATION_FORBIDDEN_FIELDS = frozenset({"actual_generation", "scheduled_consumption"})


def _utc(value):
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC) if isinstance(value, datetime) else value


def _rows(dataframe, *, timestamp_timezone="UTC"):
    return tuple({key: (organizer_timestamp(value) if isinstance(value, datetime) else value)
                  for key, value in row.asDict(recursive=True).items()}
                 for row in dataframe.toLocalIterator())


def _filtered_rows(spark, table_name, start, end_exclusive):
    dataframe = spark.table(table_name)
    numeric = [field.name for field in dataframe.schema.fields if isinstance(field.dataType, NumericType)]
    query_start, query_end = source_query_bounds(start, end_exclusive, "UTC")
    selected = [F.col("`Zeitstempel`").alias("Zeitstempel")]
    selected.extend(F.col(f"`{name}`").cast("double").alias(name) for name in numeric)
    return _rows(dataframe.select(*selected).where(
        (F.col("Zeitstempel") >= F.lit(query_start)) & (F.col("Zeitstempel") < F.lit(query_end))),
                 timestamp_timezone="UTC")


def _generation_rows(spark, start, end_exclusive):
    rows = []
    for entry in spark.sql("SHOW TABLES IN edh.input").collect():
        table = entry.tableName
        if not table.startswith("generation_forecast_"):
            continue
        dataframe = spark.table(f"edh.input.`{table}`")
        timestamp_name = "mtu_start" if "mtu_start" in dataframe.columns else "Zeitstempel"
        numeric = {field.name for field in dataframe.schema.fields if isinstance(field.dataType, NumericType)}
        selected_names = numeric & {GENERATION_ALLOWED_FIELD}
        assert not selected_names & GENERATION_FORBIDDEN_FIELDS
        if not selected_names:
            continue
        selected = [F.date_trunc("hour", F.col(f"`{timestamp_name}`")).alias("Zeitstempel")]
        selected.extend(F.col(f"`{name}`").cast("double").alias(f"{table}_{name}") for name in selected_names)
        aggregated = (dataframe.select(*selected)
                      .where((F.col("Zeitstempel") >= F.lit(start)) & (F.col("Zeitstempel") < F.lit(end_exclusive)))
                      .groupBy("Zeitstempel")
                      .agg(*[F.avg(f"{table}_{name}").alias(f"{table}_{name}") for name in selected_names]))
        rows.extend(_rows(aggregated, timestamp_timezone="UTC"))
    return tuple(rows)


def _hour_range(start, end):
    return tuple(start + timedelta(hours=index)
                 for index in range(int((end - start).total_seconds() // 3600) + 1))


def _print_gap_report(label, expected, available):
    missing = tuple(timestamp for timestamp in expected if timestamp not in available)
    print(f"  {label} missing timestamp count: {len(missing)}")
    print(f"  {label} first 20 missing timestamps: {list(missing[:20])}")


def _official_score(spark, predictions, actuals, timestamps):
    """Call the organizer's local evaluator on two temporary 168-row views."""
    prediction_rows = [(timestamp, *[predictions[timestamp][target] for target in TARGETS])
                       for timestamp in timestamps]
    realization_rows = [(timestamp, *[actuals[timestamp][target] for target in TARGETS])
                        for timestamp in timestamps]
    prediction_name = "rolling_validation_predictions"
    realization_name = "rolling_validation_realizations"
    prediction_schema = "timestamp timestamp, CH array<int>, DE array<int>, FR array<int>, IT array<int>"
    realization_schema = "timestamp timestamp, CH double, DE double, FR double, IT double"
    spark.createDataFrame(prediction_rows, schema=prediction_schema).createOrReplaceTempView(prediction_name)
    spark.createDataFrame(realization_rows, schema=realization_schema).createOrReplaceTempView(realization_name)
    return float(score_prediction_table(prediction_name, realization_name))


def run_12_week_validation(spark, *, output_dir):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    validation = _rows(spark.table(VALIDATION_TABLE).select("timestamp", "CH_actual", "DE_actual", "FR_actual", "IT_actual"))
    if len(validation) != VALIDATION_WEEKS * 168:
        raise ValueError("organizer validation table is not exactly 12 weeks")
    validation = tuple(sorted(validation, key=lambda row: row["timestamp"]))
    first = validation[0]["timestamp"]
    last = validation[-1]["timestamp"]
    requirements = history_requirements(first, warm_start_weeks=WARM_START_WEEKS)
    start = requirements["required_earliest"]
    end_exclusive = last + timedelta(hours=1)
    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", *INPUT_COUNTRIES)
    query_start, query_end = source_query_bounds(start, end_exclusive, "UTC")
    net = _rows(net_dataframe.where((F.col("Zeitstempel") >= F.lit(query_start)) & (F.col("Zeitstempel") < F.lit(query_end))),
                 timestamp_timezone="UTC")
    cross = _filtered_rows(spark, "edh.input.cross_border_exchanges", start, end_exclusive)
    ntc = _filtered_rows(spark, "edh.input.ntc_month", start, end_exclusive)
    generation = _generation_rows(spark, start, end_exclusive)
    hourly_targets = build_hourly_targets(net)
    print("Rolling validation history diagnostics:")
    print("  canonical source contract: organizer timestamp coordinate preserved; aware UTC label only")
    print("  first available hourly target:", min(hourly_targets) if hourly_targets else None)
    print("  last available hourly target:", max(hourly_targets) if hourly_targets else None)
    print("  required earliest timestamp:", requirements["required_earliest"])
    print("  Week 1 training start:", requirements["week1_train_start"])
    print("  Week 1 issue time:", requirements["week1_issue_time"])
    print("  historical warm-start weeks:", requirements["warm_start_weeks"])
    print("  historical warm-start hours:", requirements["warm_start_hours"])
    available_hours = set(hourly_targets)
    week1_train = _hour_range(requirements["week1_train_start"], requirements["week1_issue_time"])
    week1_forecast = _hour_range(first, first + timedelta(hours=167))
    _print_gap_report("Week 1 train", week1_train, available_hours)
    _print_gap_report("Week 1 forecast", week1_forecast, available_hours)
    for warm_index in range(requirements["warm_start_weeks"]):
        warm_start = requirements["warm_forecast_start"] + timedelta(hours=warm_index * 168)
        warm_issue = warm_start - timedelta(hours=1)
        warm_train = _hour_range(warm_issue - timedelta(hours=TRAIN_HOURS - 1), warm_issue)
        warm_forecast = _hour_range(warm_start, warm_start + timedelta(hours=167))
        _print_gap_report(f"Warm week {warm_index + 1} train", warm_train, available_hours)
        _print_gap_report(f"Warm week {warm_index + 1} forecast", warm_forecast, available_hours)
    require_history(hourly_targets, requirements)
    realizations = {row["timestamp"]: {target: float(row[f"{target}_actual"]) for target in TARGETS} for row in validation}
    tables = {
        "edh.input.cross_border_exchanges": _hourly_table(cross),
        "edh.input.ntc_month": _hourly_table(ntc),
    }
    if generation:
        tables["edh.input.generation_forecast"] = _hourly_table(generation)
    result = run_rolling_validation(hourly_targets=hourly_targets, realizations=realizations,
                                    db_tables=tables, output_dir=output_dir,
                                    validation_weeks=VALIDATION_WEEKS,
                                    official_score_fn=lambda predictions, actuals, timestamps:
                                    _official_score(spark, predictions, actuals, timestamps))
    status = {"validation_table": VALIDATION_TABLE, "targets": list(TARGETS),
              "validation_weeks": VALIDATION_WEEKS, "development_weeks": 11,
              "holdout_week": 12, "official_submission_called": False,
              "output_dir": str(output_dir), "net_rows_collected": len(net),
              "generation_forecast_rows_collected": len(generation),
              "summary": result["summary"]}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "run_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def run_preflight(spark, *, output_dir):
    """Fast live-data contract audit; does not fit models or submit anything."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    validation = _rows(spark.table(VALIDATION_TABLE).select("timestamp", "CH_actual", "DE_actual", "FR_actual", "IT_actual"))
    validation = tuple(sorted(validation, key=lambda row: row["timestamp"]))
    if len(validation) != VALIDATION_WEEKS * 168:
        raise ValueError("preflight: validation table must contain exactly 2016 rows")
    first, last = validation[0]["timestamp"], validation[-1]["timestamp"]
    requirements = history_requirements(first, warm_start_weeks=WARM_START_WEEKS)
    start, end_exclusive = requirements["required_earliest"], last + timedelta(hours=1)
    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", *INPUT_COUNTRIES)
    net = _rows(net_dataframe.where((F.col("Zeitstempel") >= F.lit(start)) & (F.col("Zeitstempel") < F.lit(end_exclusive))))
    hourly_targets = build_hourly_targets(net)
    require_history(hourly_targets, requirements)
    print("Preflight target range:", min(hourly_targets), "through", max(hourly_targets))
    print("Preflight required earliest:", requirements["required_earliest"])
    week1_train = _hour_range(requirements["week1_train_start"], requirements["week1_issue_time"])
    week1_forecast = _hour_range(first, first + timedelta(hours=167))
    _print_gap_report("Week 1 train", week1_train, set(hourly_targets))
    _print_gap_report("Week 1 forecast", week1_forecast, set(hourly_targets))
    validation_folds = build_validation_folds(tuple(row["timestamp"] for row in validation), hourly_targets)
    warm_start = requirements["warm_forecast_start"]
    warm_timestamps = tuple(warm_start + index * HOUR for index in range(WARM_START_WEEKS * 168))
    warm_folds = build_validation_folds(warm_timestamps, hourly_targets, weeks=WARM_START_WEEKS)
    validate_validation_folds((*warm_folds, *validation_folds))
    print("Preflight folds: warm", len(warm_folds), "validation", len(validation_folds), "holdout week 12")
    cross = _filtered_rows(spark, "edh.input.cross_border_exchanges", start, end_exclusive)
    ntc = _filtered_rows(spark, "edh.input.ntc_month", start, end_exclusive)
    generation = _generation_rows(spark, start, end_exclusive)
    if any(name in row for row in generation for name in ("actual_generation", "scheduled_consumption")):
        raise ValueError("preflight: forbidden generation fields were loaded")
    print("Preflight feature rows: cross_border", len(cross), "ntc", len(ntc), "generation_forecast", len(generation))
    if not callable(score_prediction_table):
        raise ValueError("preflight: organizer local scorer is not callable")
    timestamps = tuple(row["timestamp"] for row in validation[:168])
    prediction_rows = [(timestamp, [0] * 300, [0] * 300, [0] * 300, [0] * 300) for timestamp in timestamps]
    realization_rows = [(timestamp, 0.0, 0.0, 0.0, 0.0) for timestamp in timestamps]
    spark.createDataFrame(prediction_rows, schema="timestamp timestamp, CH array<int>, DE array<int>, FR array<int>, IT array<int>").createOrReplaceTempView("rolling_preflight_predictions")
    spark.createDataFrame(realization_rows, schema="timestamp timestamp, CH double, DE double, FR double, IT double").createOrReplaceTempView("rolling_preflight_realizations")
    scorer_result = score_prediction_table("rolling_preflight_predictions", "rolling_preflight_realizations")
    if scorer_result is None:
        raise ValueError("preflight: organizer local scorer returned no score")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("Preflight scorer score:", scorer_result)
    print("ROLLING VALIDATION PREFLIGHT: PASS")
    return {"preflight": "PASS", "validation_rows": len(validation), "warm_folds": len(warm_folds), "validation_folds": len(validation_folds), "required_earliest": str(start), "scorer_result": float(scorer_result)}


def run_smoke_test(spark, *, output_dir):
    """Fit all point candidates on the warm fold crossing the DST reference."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    validation = tuple(sorted(_rows(spark.table(VALIDATION_TABLE).select("timestamp", "CH_actual", "DE_actual", "FR_actual", "IT_actual")), key=lambda row: row["timestamp"]))
    first, last = validation[0]["timestamp"], validation[-1]["timestamp"]
    requirements = history_requirements(first, warm_start_weeks=WARM_START_WEEKS)
    start, end_exclusive = requirements["required_earliest"], last + timedelta(hours=1)
    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", *INPUT_COUNTRIES)
    query_start, query_end = source_query_bounds(start, end_exclusive, "UTC")
    net = _rows(net_dataframe.where((F.col("Zeitstempel") >= F.lit(query_start)) & (F.col("Zeitstempel") < F.lit(query_end))))
    hourly_targets = build_hourly_targets(net)
    require_history(hourly_targets, requirements)
    warm_timestamps = tuple(requirements["warm_forecast_start"] + index * HOUR for index in range(WARM_START_WEEKS * 168))
    warm_folds = build_validation_folds(warm_timestamps, hourly_targets, weeks=WARM_START_WEEKS)
    affected_index, affected = next(((index, fold) for index, fold in enumerate(warm_folds) if any(timestamp - timedelta(hours=168) not in hourly_targets for timestamp in fold.forecast_timestamps)), (len(warm_folds) - 1, warm_folds[-1]))
    cross = _filtered_rows(spark, "edh.input.cross_border_exchanges", start, end_exclusive)
    ntc = _filtered_rows(spark, "edh.input.ntc_month", start, end_exclusive)
    generation = _generation_rows(spark, start, end_exclusive)
    tables = {"edh.input.cross_border_exchanges": _hourly_table(cross), "edh.input.ntc_month": _hourly_table(ntc)}
    if generation:
        tables["edh.input.generation_forecast"] = _hourly_table(generation)
    pre_forecast_start = affected.forecast_start - timedelta(hours=336)
    pre_issue = pre_forecast_start - HOUR
    pre_train_required = tuple(pre_issue - timedelta(hours=TRAIN_HOURS - 1 - index)
                               for index in range(TRAIN_HOURS))
    pre_train = tuple(timestamp for timestamp in pre_train_required if timestamp in hourly_targets)
    pre_forecast = tuple(pre_forecast_start + index * HOUR for index in range(168))
    if len(pre_train) == 0 or any(timestamp not in hourly_targets for timestamp in pre_forecast):
        raise ValueError("smoke test lacks a complete causal pre-DST fold")
    pre_fold = type(affected)("smoke_pre_dst", pre_train, pre_forecast)
    smoke_folds = (pre_fold, affected)
    _, scores, predictions = _run_scenario(
        targets=hourly_targets, db_tables=tables, lseg=LSEGFeatures({}, {}, {}, ()),
        folds=smoke_folds, use_lseg=False, scenario="rolling_validation_smoke",
        seed=20260910, model_names=POINT_MODELS, collect_predictions=True)
    expected = len(POINT_MODELS) * len(TARGETS) * 168 * 2
    if len(predictions) != expected or len(scores) != len(POINT_MODELS) * len(TARGETS) * 2:
        raise ValueError("smoke test did not produce all point-model predictions")
    seasonal = {(row["fold_id"], datetime.fromisoformat(row["timestamp"]), row["target"]): row["pred"]
                for row in predictions if row["model"] == "seasonal_persistence"}
    prior_rows = []
    for fold in (pre_fold,):
        for timestamp in fold.forecast_timestamps:
            prior_rows.append((fold.issue_time, timestamp, timestamp - fold.issue_time,
                               tuple(hourly_targets[timestamp][target] - seasonal[(fold.fold_id, timestamp, target)]
                                     for target in TARGETS)))
    if len(prior_rows) < 2:
        raise ValueError("smoke test lacks prior residual rows for uncertainty fitting")
    uncertainty = fit_uncertainty_model(ResidualPanel(TARGETS, tuple(prior_rows)),
                                        method="correlated_gaussian", fit_cutoff=affected.issue_time)
    point_forecasts = {timestamp: {target: seasonal[(affected.fold_id, timestamp, target)] for target in TARGETS}
                       for timestamp in affected.forecast_timestamps}
    samples = {}
    for index, timestamp in enumerate(affected.forecast_timestamps):
        samples[timestamp] = sample_distribution(
            uncertainty, point_forecasts[timestamp], seed=20260910 + index,
            n_samples=300)
    if any(len(samples[timestamp][target]) != 300 or
           any(not isinstance(value, int) for value in samples[timestamp][target])
           for timestamp in samples for target in TARGETS):
        raise ValueError("smoke test did not produce 300 integer samples")
    actuals = {timestamp: {target: hourly_targets[timestamp][target] for target in TARGETS}
               for timestamp in affected.forecast_timestamps}
    score = _official_score(spark, samples, actuals, affected.forecast_timestamps)
    if score is None:
        raise ValueError("smoke test organizer scorer returned no score")
    print("Smoke fold:", affected.fold_id, "issue_time:", affected.issue_time)
    print("Smoke uncertainty: correlated_gaussian, organizer score:", score)
    print("ROLLING VALIDATION SMOKE TEST: PASS")
    return {"smoke": "PASS", "fold_id": affected.fold_id, "predictions": len(predictions), "models": list(POINT_MODELS), "uncertainty": "correlated_gaussian", "score": float(score)}


# Databricks execution cell.  Open this file and Run all.  Keep preflight on
# until its PASS result is visible before starting the expensive model run.
PREFLIGHT_ONLY = True
SMOKE_ONLY = False
VALIDATION_WEEKS = 12
OUTPUT_DIR = "/Workspace/Users/user32@swissgridlab.onmicrosoft.com/swissgrid_backtest_outputs/rolling_validation_12w"
if PREFLIGHT_ONLY:
    print("Starting rolling validation preflight")
    status = run_preflight(spark, output_dir=OUTPUT_DIR)
elif SMOKE_ONLY:
    print("Starting rolling validation smoke test; official submission disabled")
    status = run_smoke_test(spark, output_dir=OUTPUT_DIR)
else:
    print("Starting organizer-style rolling validation; official submission disabled")
    status = run_12_week_validation(spark, output_dir=OUTPUT_DIR)
print(json.dumps(status, indent=2, sort_keys=True, default=str))
if PREFLIGHT_ONLY and "dbutils" in globals():
    dbutils.notebook.exit(json.dumps({"result": "ROLLING VALIDATION PREFLIGHT: PASS", **status}, sort_keys=True, default=str))
if SMOKE_ONLY and "dbutils" in globals():
    dbutils.notebook.exit(json.dumps({"result": "ROLLING VALIDATION SMOKE TEST: PASS", **status}, sort_keys=True, default=str))
status
