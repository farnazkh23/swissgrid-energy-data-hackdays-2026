"""Databricks entrypoint for the single Challenge 4 submission table.

This is a fixed-origin production run only.  It does not run historical
validation and it never invokes an official submission helper.
"""

from datetime import datetime, timedelta, timezone
import sys

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, NumericType, TimestampType


REPO_ROOT = "/Workspace/Users/user32@swissgridlab.onmicrosoft.com/swissgrid-energy-data-hackdays-2026"
REPO_SRC = f"{REPO_ROOT}/src"
EDH_SOURCE_ROOT = "/Workspace/github/dp-light-edh/src"
for path in (REPO_SRC, EDH_SOURCE_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from swissgrid_forecaster.real_modeling_v1 import (  # noqa: E402
    _hourly_table,
    _sample,
    _seasonal_persistence_value,
    build_hourly_targets,
)
from swissgrid_forecaster.target_contract import TARGETS, validate_output_targets  # noqa: E402
from swissgrid_forecaster.time_contract import organizer_timestamp, source_query_bounds  # noqa: E402


TARGET_COLUMNS = ("timestamp", *TARGETS)
TABLE_NAME = "edh.group_0.g4_submission_v1"
FORECAST_START = datetime(2026, 9, 2, tzinfo=timezone.utc)
FORECAST_END = datetime(2026, 9, 8, 23, tzinfo=timezone.utc)
FORECAST_HOURS = 168
GENERATION_ALLOWED_FIELD = "generation_forecast"
GENERATION_FORBIDDEN_FIELDS = frozenset({"actual_generation", "scheduled_consumption"})


def _rows(dataframe):
    return tuple(
        {
            key: organizer_timestamp(value) if isinstance(value, datetime) else value
            for key, value in row.asDict(recursive=True).items()
        }
        for row in dataframe.toLocalIterator()
    )


def _source_rows(spark, table_name, start, end_exclusive):
    dataframe = spark.table(table_name)
    numeric = [field.name for field in dataframe.schema.fields if isinstance(field.dataType, NumericType)]
    query_start, query_end = source_query_bounds(start, end_exclusive, "UTC")
    selected = [F.col("`Zeitstempel`").alias("Zeitstempel")]
    selected.extend(F.col(f"`{name}`").cast("double").alias(name) for name in numeric)
    return _rows(
        dataframe.select(*selected).where(
            (F.col("Zeitstempel") >= F.lit(query_start))
            & (F.col("Zeitstempel") < F.lit(query_end))
        )
    )


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
        if selected_names & GENERATION_FORBIDDEN_FIELDS:
            raise ValueError("forbidden generation fields were selected")
        if not selected_names:
            continue
        selected = [F.date_trunc("hour", F.col(f"`{timestamp_name}`")).alias("Zeitstempel")]
        selected.extend(F.col(f"`{name}`").cast("double").alias(f"{table}_{name}") for name in selected_names)
        query = (dataframe.select(*selected)
                 .where((F.col("Zeitstempel") >= F.lit(start)) & (F.col("Zeitstempel") < F.lit(end_exclusive)))
                 .groupBy("Zeitstempel")
                 .agg(*[F.avg(f"{table}_{name}").alias(f"{table}_{name}") for name in selected_names]))
        rows.extend(_rows(query))
    return tuple(rows)


def _coverage(rows):
    """Summarize timestamp coverage without filling missing source values."""
    timestamps = tuple(sorted({row["Zeitstempel"] for row in rows if row.get("Zeitstempel") is not None}))
    hours = tuple(sorted({stamp.replace(minute=0, second=0, microsecond=0) for stamp in timestamps}))
    return {
        "rows": len(rows),
        "unique_hours": len(hours),
        "first": hours[0] if hours else None,
        "last": hours[-1] if hours else None,
    }


def _validate_submission(dataframe, *, label):
    expected = list(TARGET_COLUMNS)
    if dataframe.columns != expected:
        raise ValueError(f"{label}: columns must be {expected}, got {dataframe.columns}")
    if not isinstance(dataframe.schema["timestamp"].dataType, TimestampType):
        raise ValueError(f"{label}: timestamp must have timestamp type")
    for target in TARGETS:
        data_type = dataframe.schema[target].dataType
        if not isinstance(data_type, ArrayType) or not isinstance(data_type.elementType, IntegerType):
            raise ValueError(f"{label}: {target} must have array<int> type")
    rows = tuple(sorted(
        (row.asDict(recursive=True) for row in dataframe.toLocalIterator()),
        key=lambda row: row["timestamp"],
    ))
    if len(rows) != FORECAST_HOURS:
        raise ValueError(f"{label}: expected exactly {FORECAST_HOURS} rows, got {len(rows)}")
    timestamps = [row["timestamp"] for row in rows]
    if any(value is None for value in timestamps):
        raise ValueError(f"{label}: null timestamp")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{label}: timestamps are not unique")
    expected_timestamps = tuple(
        (FORECAST_START + timedelta(hours=index)).replace(tzinfo=None)
        for index in range(FORECAST_HOURS)
    )
    if tuple(timestamps) != expected_timestamps:
        raise ValueError(
            f"{label}: timestamps must be exactly {expected_timestamps[0]} through "
            f"{expected_timestamps[-1]}"
        )
    for row in rows:
        for target in TARGETS:
            values = row[target]
            if values is None or not isinstance(values, (list, tuple)):
                raise ValueError(f"{label}: {target} must be a non-null array")
            if len(values) != 300:
                raise ValueError(f"{label}: {target} array length is {len(values)}, not 300")
            if any(value is None for value in values):
                raise ValueError(f"{label}: {target} contains a null sample")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"{label}: {target} contains a non-integer sample")
    return rows


def _print_diagnostics(rows, dataframe):
    print("SUBMISSION PREFLIGHT")
    print(f"- row count: {len(rows)}")
    print(f"- first timestamp: {rows[0]['timestamp']}")
    print(f"- last timestamp: {rows[-1]['timestamp']}")
    print(f"- schema: {dataframe.schema.simpleString()}")
    for target in TARGETS:
        values = rows[0][target]
        all_values = [sample for row in rows for sample in row[target]]
        print(f"- {target} array length (first row): {len(values)}")
        print(f"- {target} first 3 samples: {values[:3]}")
        print(f"- {target} min/max/mean samples: {min(all_values)}/{max(all_values)}/{sum(all_values) / len(all_values):.3f}")


def run_submission(spark):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    validate_output_targets(TARGETS)

    net_dataframe = spark.table("edh.input.net_positions").select("Zeitstempel", "AT", *TARGETS)
    net_rows = _rows(net_dataframe)
    targets = build_hourly_targets(net_rows)
    if not targets:
        raise ValueError("no complete causal hourly target data")
    issue_time = max(targets)
    forecast_timestamps = tuple(
        FORECAST_START + timedelta(hours=index) for index in range(FORECAST_HOURS)
    )
    if forecast_timestamps[-1] != FORECAST_END:
        raise ValueError("fixed forecast constants do not describe 168 hourly timestamps")
    if issue_time >= FORECAST_START:
        raise ValueError(f"issue_time {issue_time} is not before fixed forecast start")
    train_timestamps = tuple(issue_time - timedelta(hours=index) for index in range(671, -1, -1))
    if any(timestamp not in targets for timestamp in train_timestamps):
        raise ValueError("latest causal history does not contain 672 complete hours")

    source_start = train_timestamps[0]
    source_end = forecast_timestamps[-1] + timedelta(hours=1)
    cross_rows = _source_rows(spark, "edh.input.cross_border_exchanges", source_start, source_end)
    ntc_rows = _source_rows(spark, "edh.input.ntc_month", source_start, source_end)
    generation_rows = _generation_rows(spark, source_start, source_end)
    # Materialize the same bounded source contract used by the production model.
    _hourly_table(cross_rows)
    _hourly_table(ntc_rows)
    _hourly_table(generation_rows)

    gap_hours = int((FORECAST_START - issue_time).total_seconds() // 3600)
    coverage = {
        "net_positions": _coverage(net_rows),
        "cross_border_exchanges": _coverage(cross_rows),
        "ntc_month": _coverage(ntc_rows),
        "generation_forecast_*": _coverage(generation_rows),
    }
    print("SUBMISSION CAUSAL AUDIT")
    print(f"- latest actual target timestamp: {issue_time}")
    print(f"- issue_time: {issue_time}")
    print(f"- forecast start: {FORECAST_START}")
    print(f"- forecast end: {FORECAST_END}")
    print(f"- gap from issue_time to forecast start: {gap_hours} hours")
    print("- gap handling: no target values are fabricated; causal seasonal persistence "
          "uses 168h/nearby/336h lags when available, then latest_causal fallback")
    print(f"- feature coverage: {coverage}")

    target_series = {target: {stamp: row[target] for stamp, row in targets.items()} for target in TARGETS}
    samples_by_target = {target: [] for target in TARGETS}
    for target in TARGETS:
        residuals = tuple(
            targets[stamp][target] - targets[stamp - timedelta(hours=168)][target]
            for stamp in train_timestamps
            if stamp - timedelta(hours=168) in targets
        )
        if not residuals:
            raise ValueError(f"no causal residuals available for {target}")
        for index, timestamp in enumerate(forecast_timestamps):
            point, _ = _seasonal_persistence_value(target_series[target], timestamp, issue_time)
            samples_by_target[target].append(list(_sample(point, residuals, seed=20260911 + index + sum(map(ord, target)) * 1009)))

    rows = tuple({"timestamp": timestamp, **{target: samples_by_target[target][index] for target in TARGETS}}
                 for index, timestamp in enumerate(forecast_timestamps))
    schema = "timestamp timestamp, CH array<int>, DE array<int>, FR array<int>, IT array<int>"
    dataframe = spark.createDataFrame(rows, schema=schema).select(*TARGET_COLUMNS)
    try:
        validated = _validate_submission(dataframe, label="generated")
        _print_diagnostics(validated, dataframe)
        print("- validator PASS")
    except Exception:
        print("- validator FAIL")
        raise

    # The generated table has passed the complete contract before this overwrite.
    dataframe.write.mode("overwrite").saveAsTable(TABLE_NAME)
    stored = spark.table(TABLE_NAME).select(*TARGET_COLUMNS)
    _validate_submission(stored, label="stored")
    print(f"SUBMISSION TABLE READY: {TABLE_NAME}")
    return stored


# Databricks execution cell. Run this file only for the one-table submission path.
submission_table = run_submission(spark)
