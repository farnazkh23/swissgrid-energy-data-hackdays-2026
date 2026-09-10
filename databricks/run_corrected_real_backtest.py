# Databricks notebook source
"""Databricks-only entrypoint for the corrected Swissgrid backtest.

Run this from a Databricks cluster with the repository's ``src`` directory on
``sys.path``.  The expensive table reads, feature construction, model fits,
rolling folds, uncertainty fits, 300-sample generation, and local EDH score
all happen on the attached Databricks runtime.  This file is intentionally not
invoked by local development commands.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

# The challenge helper is provided by the EDH workspace checkout.
EDH_SOURCE_ROOT = "/Workspace/github/dp-light-edh/src"
if EDH_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, EDH_SOURCE_ROOT)

from edh2026.local_scoring import score_prediction_table
from swissgrid_forecaster.oof_handoff import write_oof_handoff
from swissgrid_forecaster.real_modeling_v1 import TARGETS, make_weekly_folds, run_v1
from swissgrid_forecaster.real_uncertainty_backtest import (
    METHODS, compare_methods, fit_calibrated_champion,
    load_point_forecasts, load_residual_panel, sample_folds,
)


GENERATION_ALLOWED_FIELD = "generation_forecast"
GENERATION_FORBIDDEN_FIELDS = frozenset({
    "actual_generation", "scheduled_consumption", "actual", "current",
    "day_ahead", "intraday",
})


def _rows(dataframe):
    return tuple(row.asDict(recursive=True) for row in dataframe.toLocalIterator())


def _filtered_table_rows(spark, table_name, start, end_exclusive):
    """Select only timestamp plus numeric measurements, then collect filtered rows."""
    dataframe = spark.table(table_name)
    numeric_names = [field.name for field in dataframe.schema.fields
                     if isinstance(field.dataType, NumericType)]
    selected = [F.col("`Zeitstempel`").alias("Zeitstempel")]
    selected.extend(F.col(f"`{name}`").cast("double").alias(name) for name in numeric_names)
    filtered = (dataframe.select(*selected)
                .where((F.col("Zeitstempel") >= F.lit(start))
                       & (F.col("Zeitstempel") < F.lit(end_exclusive))))
    rows = _rows(filtered)
    return rows, len(rows)


def _fold_bounds(spark, net_dataframe, fold_count):
    """Find the exact existing fold range from a one-column Spark completeness index."""
    completeness = (F.col("row_count") == 4) & (F.col("quarter_count") == 4)
    for target in TARGETS:
        completeness = completeness & (F.col(f"{target}_count") == 4)
    complete_hours = (net_dataframe
                      .where(F.minute("Zeitstempel").isin(0, 15, 30, 45))
                      .withColumn("hour", F.date_trunc("hour", "Zeitstempel"))
                      .groupBy("hour")
                      .agg(F.count("*").alias("row_count"),
                           F.countDistinct("Zeitstempel").alias("quarter_count"),
                           *[F.count(target).alias(f"{target}_count") for target in TARGETS])
                      .where(completeness)
                      .select("hour")
                      .orderBy("hour"))
    # Only the timestamp index is collected here; raw measurements are filtered
    # and aggregated before their separate, bounded collections below.
    index = tuple(row["hour"] for row in complete_hours.toLocalIterator())
    folds = make_weekly_folds(index, max_folds=fold_count)
    if len(folds) != fold_count:
        raise ValueError(f"expected {fold_count} complete folds, got {len(folds)}")
    return (folds[0].train_timestamps[0],
            folds[-1].forecast_timestamps[-1] + timedelta(hours=1),
            len(index))


def _generation_rows(spark, start, end_exclusive):
    """Use only issue-time forecast fields; never ingest actuals or consumption."""
    tables = [row.tableName for row in spark.sql("SHOW TABLES IN edh.input").collect()
              if row.tableName.startswith("generation_forecast_")]
    rows = []
    collected_count = 0
    for table in sorted(tables):
        dataframe = spark.table(f"edh.input.`{table}`")
        timestamp_name = "mtu_start" if "mtu_start" in dataframe.columns else "Zeitstempel"
        numeric_names = {field.name for field in dataframe.schema.fields
                         if isinstance(field.dataType, NumericType)}
        # Leakage rule: forecast features may use only the explicit
        # generation_forecast field. Actual generation is unavailable at issue
        # time for target/future timestamps, and scheduled consumption semantics
        # are not proven, so both are excluded from the clean benchmark.
        assert GENERATION_FORBIDDEN_FIELDS.isdisjoint({GENERATION_ALLOWED_FIELD})
        selected_names = numeric_names & {GENERATION_ALLOWED_FIELD}
        assert selected_names <= {GENERATION_ALLOWED_FIELD}
        assert not selected_names & GENERATION_FORBIDDEN_FIELDS
        if not selected_names:
            continue
        selected = [F.date_trunc("hour", F.col(f"`{timestamp_name}`")).alias("Zeitstempel")]
        selected.extend(F.col(f"`{name}`").cast("double").alias(f"{table}_{name}")
                       for name in sorted(selected_names))
        aggregated = (dataframe.select(*selected)
                      .where((F.col("Zeitstempel") >= F.lit(start))
                             & (F.col("Zeitstempel") < F.lit(end_exclusive)))
                      .groupBy("Zeitstempel")
                      .agg(*[F.avg(f"{table}_{name}").alias(f"{table}_{name}")
                             for name in sorted(selected_names)]))
        table_rows = _rows(aggregated)
        collected_count += len(table_rows)
        for row in table_rows:
            if row["Zeitstempel"] is not None:
                rows.append(row)
    return tuple(rows), collected_count


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def _write_model_artifacts(result, output_dir):
    _write_json(output_dir / "corrected_model_scoreboard.json", result["scoreboard"])
    _write_json(output_dir / "corrected_champions.json", result["champions"])
    _write_json(output_dir / "corrected_fold_manifest.json", result["fold_manifest"])
    rows = result["oof_predictions"]
    write_oof_handoff(rows, {target: (result["champions"][target], "real-v1")
                             for target in TARGETS}, output_dir)
    return rows


def _score_one_fold(spark, samples, rows, output_dir):
    """Score the first generated 168-hour fold through edh2026.local_scoring."""
    first = samples[:168]
    timestamps = [datetime.fromisoformat(entry["timestamp"]) for entry in first]
    prediction_rows = [(stamp, *[entry[f"{target}_samples"] for target in TARGETS])
                       for stamp, entry in zip(timestamps, first)]
    realization_by_time = {datetime.fromisoformat(row["timestamp"]): row for row in rows}
    realization_rows = [(stamp, *[realization_by_time[stamp][f"{target}_actual"] for target in TARGETS])
                        for stamp in timestamps]
    prediction_name = "corrected_backtest_predictions"
    realization_name = "corrected_backtest_realizations"
    prediction_schema = "timestamp timestamp, AT array<int>, DE array<int>, FR array<int>, IT array<int>"
    realization_schema = "timestamp timestamp, AT double, DE double, FR double, IT double"
    spark.createDataFrame(prediction_rows, schema=prediction_schema).createOrReplaceTempView(prediction_name)
    spark.createDataFrame(realization_rows, schema=realization_schema).createOrReplaceTempView(realization_name)
    raw_score = float(score_prediction_table(prediction_name, realization_name))
    _write_json(output_dir / "edh_local_score.json", {
        "raw_score": raw_score,
        "rows_scored": len(first),
        "targets": list(TARGETS),
        "official_submission_called": False,
        "scorer": "edh2026.local_scoring.score_prediction_table",
    })
    return raw_score


def run_corrected_real_backtest(spark, *, fold_count=1,
                                output_dir="/dbfs/FileStore/swissgrid-forecaster/artifacts/real_backtest",
                                seed=20260910):
    """Run one, three, or twelve corrected folds; never invoke submission APIs."""
    if fold_count not in (1, 3, 12):
        raise ValueError("fold_count must be one of 1, 3, or 12")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", *TARGETS, "CH")
    start, end_exclusive, indexed_hours = _fold_bounds(spark, net_dataframe, fold_count)
    net = _rows(net_dataframe.where((F.col("Zeitstempel") >= F.lit(start))
                                    & (F.col("Zeitstempel") < F.lit(end_exclusive))))
    cross, cross_count = _filtered_table_rows(spark, "edh.input.cross_border_exchanges",
                                               start, end_exclusive)
    ntc, ntc_count = _filtered_table_rows(spark, "edh.input.ntc_month", start, end_exclusive)
    generation, generation_count = _generation_rows(spark, start, end_exclusive)

    result = run_v1(net_rows=net, cross_rows=cross, ntc_rows=ntc,
                    generation_rows=generation, lseg_root=None,
                    max_folds=fold_count, seed=seed)
    oof_rows = _write_model_artifacts(result, output_dir)

    status = {"folds_completed": fold_count, "targets": list(TARGETS),
              "champions": result["champions"], "probability": "not_run",
              "driver_collection_counts": {
                  "complete_hour_index": indexed_hours,
                  "net_positions": len(net), "cross_border_exchanges": cross_count,
                  "ntc_month": ntc_count, "generation_forecast_aggregated": generation_count,
              },
              "generation_fields_used": [GENERATION_ALLOWED_FIELD],
              "generation_fields_excluded": sorted(GENERATION_FORBIDDEN_FIELDS)}
    if fold_count < 3:
        status["probability"] = "blocked: at least 3 weekly OOF folds are required for fit/calibration/demo"
        _write_json(output_dir / "run_status.json", status)
        return status

    panel = load_residual_panel(output_dir / "oof_predictions.csv")
    methods = dict(METHODS)
    ranked = compare_methods(panel, seed=seed, n_samples=300, methods=methods)
    _write_json(output_dir / "uncertainty_method_comparison.json",
                [row.to_row() for row in ranked])
    champion_method = ranked[0].method
    fit_weeks = min(8, fold_count - 2)
    calibration_weeks = min(2, fold_count - fit_weeks - 1)
    model, demo_folds = fit_calibrated_champion(
        panel, champion_method, fit_weeks=fit_weeks,
        calibration_weeks=calibration_weeks, target_coverage=0.8,
        seed=seed, n_samples=300,
        **methods[champion_method])
    point_forecasts = load_point_forecasts(output_dir / "oof_predictions.csv")
    first_demo = demo_folds[:1]
    samples = sample_folds(model, first_demo, point_forecasts, seed=seed, n_samples=300)
    (output_dir / "uncertainty_champion_samples.json").write_text(
        json.dumps(samples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_score = _score_one_fold(spark, samples, oof_rows, output_dir)
    status.update({"probability": "completed", "probability_champion": champion_method,
                   "edh_local_score_first_fold": raw_score})
    _write_json(output_dir / "run_status.json", status)
    return status
