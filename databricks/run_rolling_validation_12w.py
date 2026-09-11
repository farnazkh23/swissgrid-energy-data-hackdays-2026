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

from swissgrid_forecaster.real_modeling_v1 import _hourly_table, build_hourly_targets
from swissgrid_forecaster.rolling_validation import (
    TRAIN_HOURS, VALIDATION_WEEKS, WARM_START_WEEKS, history_requirements,
    require_history, run_rolling_validation,
)
from swissgrid_forecaster.target_contract import INPUT_COUNTRIES, TARGETS
from edh2026.local_scoring import score_prediction_table

UTC = timezone.utc
VALIDATION_TABLE = "edh.group_0.g4_validation_realizations"
GENERATION_ALLOWED_FIELD = "generation_forecast"
GENERATION_FORBIDDEN_FIELDS = frozenset({"actual_generation", "scheduled_consumption"})


def _utc(value):
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC) if isinstance(value, datetime) else value


def _rows(dataframe):
    return tuple({key: _utc(value) if isinstance(value, datetime) else value
                  for key, value in row.asDict(recursive=True).items()}
                 for row in dataframe.toLocalIterator())


def _filtered_rows(spark, table_name, start, end_exclusive):
    dataframe = spark.table(table_name)
    numeric = [field.name for field in dataframe.schema.fields if isinstance(field.dataType, NumericType)]
    selected = [F.col("`Zeitstempel`").alias("Zeitstempel")]
    selected.extend(F.col(f"`{name}`").cast("double").alias(name) for name in numeric)
    return _rows(dataframe.select(*selected).where(
        (F.col("Zeitstempel") >= F.lit(start)) & (F.col("Zeitstempel") < F.lit(end_exclusive))))


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
        rows.extend(_rows(aggregated))
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
    net = _rows(net_dataframe.where((F.col("Zeitstempel") >= F.lit(start)) & (F.col("Zeitstempel") < F.lit(end_exclusive))))
    cross = _filtered_rows(spark, "edh.input.cross_border_exchanges", start, end_exclusive)
    ntc = _filtered_rows(spark, "edh.input.ntc_month", start, end_exclusive)
    generation = _generation_rows(spark, start, end_exclusive)
    hourly_targets = build_hourly_targets(net)
    print("Rolling validation history diagnostics:")
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


# Databricks execution cell.  Open this file and Run all.
VALIDATION_WEEKS = 12
OUTPUT_DIR = "/Workspace/Users/user32@swissgridlab.onmicrosoft.com/swissgrid_backtest_outputs/rolling_validation_12w"
print("Starting organizer-style rolling validation; official submission disabled")
status = run_12_week_validation(spark, output_dir=OUTPUT_DIR)
print(json.dumps(status, indent=2, sort_keys=True, default=str))
status
