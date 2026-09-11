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
from swissgrid_forecaster.rolling_validation import TRAIN_HOURS, VALIDATION_WEEKS, run_rolling_validation
from swissgrid_forecaster.target_contract import INPUT_COUNTRIES, TARGETS

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


def run_12_week_validation(spark, *, output_dir):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    validation = _rows(spark.table(VALIDATION_TABLE).select("timestamp", "CH_actual", "DE_actual", "FR_actual", "IT_actual"))
    if len(validation) != VALIDATION_WEEKS * 168:
        raise ValueError("organizer validation table is not exactly 12 weeks")
    validation = tuple(sorted(validation, key=lambda row: row["timestamp"]))
    first = validation[0]["timestamp"]
    last = validation[-1]["timestamp"]
    start = first - timedelta(hours=TRAIN_HOURS + 168)
    end_exclusive = last + timedelta(hours=1)
    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", *INPUT_COUNTRIES)
    net = _rows(net_dataframe.where((F.col("Zeitstempel") >= F.lit(start)) & (F.col("Zeitstempel") < F.lit(end_exclusive))))
    cross = _filtered_rows(spark, "edh.input.cross_border_exchanges", start, end_exclusive)
    ntc = _filtered_rows(spark, "edh.input.ntc_month", start, end_exclusive)
    generation = _generation_rows(spark, start, end_exclusive)
    hourly_targets = build_hourly_targets(net)
    realizations = {row["timestamp"]: {target: float(row[f"{target}_actual"]) for target in TARGETS} for row in validation}
    tables = {
        "edh.input.cross_border_exchanges": _hourly_table(cross),
        "edh.input.ntc_month": _hourly_table(ntc),
    }
    if generation:
        tables["edh.input.generation_forecast"] = _hourly_table(generation)
    result = run_rolling_validation(hourly_targets=hourly_targets, realizations=realizations,
                                    db_tables=tables, output_dir=output_dir,
                                    validation_weeks=VALIDATION_WEEKS)
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
