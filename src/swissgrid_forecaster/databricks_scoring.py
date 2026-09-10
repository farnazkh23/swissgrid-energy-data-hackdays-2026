"""Databricks adapter for the EDH 2026 local scorer.

The challenge helper is deliberately kept outside this package.  This module
only imports it inside Databricks, through the official ``local_scoring``
module, and never imports or calls ``trigger_submission``.
"""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
import importlib
import re
import sys
from typing import Iterable, Sequence


OFFICIAL_SOURCE_ROOT = "/Workspace/github/dp-light-edh/src"
OFFICIAL_SAMPLE_COUNT = 300
OFFICIAL_SUBMISSION_ROW_COUNT = 168
OFFICIAL_COLUMN_COUNT = 5
_VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class OfficialScorerUnavailable(RuntimeError):
    """Raised when this process is not running in a compatible Databricks runtime."""


def _official_local_scorer():
    """Load the official scorer without making it a package dependency locally."""
    if OFFICIAL_SOURCE_ROOT not in sys.path:
        sys.path.insert(0, OFFICIAL_SOURCE_ROOT)
    try:
        module = importlib.import_module("edh2026.local_scoring")
        scorer = getattr(module, "score_prediction_table")
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        raise OfficialScorerUnavailable(
            "The Databricks edh2026.local_scoring helper is unavailable. "
            "Run this adapter from a Databricks notebook with the EDH source path."
        ) from exc
    return scorer


def official_scorer_available() -> bool:
    """Return whether the official local scorer can be imported in this process."""
    try:
        _official_local_scorer()
        importlib.import_module("databricks.sdk.runtime")
    except OfficialScorerUnavailable:
        return False
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def _spark_session(spark=None):
    if spark is not None:
        return spark
    try:
        from pyspark.sql import SparkSession
    except (ImportError, ModuleNotFoundError) as exc:
        raise OfficialScorerUnavailable(
            "PySpark is unavailable; this operation requires a Databricks runtime."
        ) from exc
    return SparkSession.builder.getOrCreate()


def score_week_with_official_scorer(
    prediction_table_name: str,
    realization_table_name: str,
) -> float:
    """Score two Databricks tables with the official EDH local scorer.

    This function calls ``edh2026.local_scoring.score_prediction_table`` only.
    It does not discover jobs, call ``run_now``, enforce a cooldown, or write
    official submission results.
    """
    scorer = _official_local_scorer()
    spark = _spark_session()
    predictions = spark.table(prediction_table_name)
    realizations = spark.table(realization_table_name)
    validate_prediction_dataframe(predictions, submission_mode=True)
    validate_realization_dataframe(realizations)
    validate_timestamp_alignment(predictions, realizations)
    try:
        result = scorer(prediction_table_name, realization_table_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise OfficialScorerUnavailable(
            "The Databricks runtime dependencies for edh2026.local_scoring "
            "are unavailable."
        ) from exc
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not isfinite(result):
        raise ValueError("The official local scorer returned a non-finite numeric score.")
    return float(result)


def _validate_names(names: Sequence[str], *, label: str) -> tuple[str, ...]:
    names = tuple(names)
    if len(names) != OFFICIAL_COLUMN_COUNT - 1:
        raise ValueError(f"{label} must contain exactly four names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must not contain duplicate names")
    if any(name == "timestamp" or not isinstance(name, str) or not _VALID_NAME.fullmatch(name)
           for name in names):
        raise ValueError(
            f"{label} names must contain only letters, numbers, underscores, and dashes "
            "and must not be 'timestamp'"
        )
    return names


def _integer_samples(samples, *, target_name: str, rounding: str) -> tuple[int, ...]:
    values = tuple(samples) if samples is not None else ()
    if len(values) != OFFICIAL_SAMPLE_COUNT:
        raise ValueError(
            f"Target '{target_name}' must contain exactly {OFFICIAL_SAMPLE_COUNT} samples; "
            f"found {len(values)}"
        )
    if rounding not in {"nearest", "exact"}:
        raise ValueError("rounding must be 'nearest' or 'exact'")
    converted = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ValueError(f"Target '{target_name}' contains a non-finite numeric sample")
        if rounding == "exact" and float(value) != int(value):
            raise ValueError(f"Target '{target_name}' contains a non-integer sample")
        converted.append(int(round(float(value))))
    return tuple(converted)


def build_official_prediction_records(
    forecasts: Iterable,
    *,
    target_columns: Sequence[str],
    rounding: str = "nearest",
) -> tuple[tuple, ...]:
    """Convert ``ForecastOutput`` objects into exact official table records.

    ``forecasts`` must contain one output for each target and timestamp.  The
    target names are explicit because the official evaluator intentionally
    treats the four business columns positionally and does not publish fixed
    names in its helper source.
    """
    target_columns = _validate_names(target_columns, label="target_columns")
    by_timestamp = defaultdict(dict)
    for forecast in forecasts:
        timestamp = getattr(forecast, "target_time", None)
        target_name = getattr(forecast, "target_name", None)
        if timestamp is None or target_name not in target_columns:
            raise ValueError(
                "Each forecast must have a target_time and one of the configured target_columns"
            )
        if target_name in by_timestamp[timestamp]:
            raise ValueError(f"Duplicate forecast for timestamp {timestamp!r}, target '{target_name}'")
        by_timestamp[timestamp][target_name] = _integer_samples(
            getattr(forecast, "samples", None), target_name=target_name, rounding=rounding
        )

    if not by_timestamp:
        raise ValueError("At least one forecast row is required")
    records = []
    for timestamp in sorted(by_timestamp):
        row = by_timestamp[timestamp]
        missing = tuple(name for name in target_columns if name not in row)
        if missing:
            raise ValueError(f"Timestamp {timestamp!r} is missing targets: {', '.join(missing)}")
        records.append((timestamp, *(row[name] for name in target_columns)))
    return tuple(records)


def _require_spark_types():
    try:
        from pyspark.sql import functions as F
        from pyspark.sql.types import ArrayType, IntegerType, TimestampType
    except (ImportError, ModuleNotFoundError) as exc:
        raise OfficialScorerUnavailable(
            "PySpark is unavailable; table validation requires a Databricks runtime."
        ) from exc
    return F, ArrayType, IntegerType, TimestampType


def validate_prediction_dataframe(
    dataframe,
    *,
    submission_mode: bool = True,
    target_columns: Sequence[str] | None = None,
) -> None:
    """Validate prediction schema and values before writing or scoring."""
    F, ArrayType, IntegerType, TimestampType = _require_spark_types()
    columns = tuple(dataframe.columns)
    if len(columns) != OFFICIAL_COLUMN_COUNT:
        raise ValueError("The prediction table must contain exactly five columns")
    if len(set(columns)) != len(columns) or columns[0] != "timestamp":
        raise ValueError("The prediction table must start with a unique 'timestamp' column")
    if target_columns is not None and columns[1:] != _validate_names(target_columns, label="target_columns"):
        raise ValueError("Prediction target columns do not match target_columns")
    invalid_names = [name for name in columns if not _VALID_NAME.fullmatch(name)]
    if invalid_names:
        raise ValueError(f"Invalid prediction column names: {', '.join(invalid_names)}")
    fields = dataframe.schema.fields
    if not isinstance(fields[0].dataType, TimestampType):
        raise ValueError("The 'timestamp' column must have timestamp type")
    bad_arrays = [
        field.name for field in fields[1:]
        if not (isinstance(field.dataType, ArrayType)
                and isinstance(field.dataType.elementType, IntegerType))
    ]
    if bad_arrays:
        raise ValueError(f"Prediction columns must have array<int> type: {', '.join(bad_arrays)}")

    invalid = F.col("timestamp").isNull()
    for column in columns[1:]:
        array = F.col(column)
        invalid = invalid | array.isNull() | (F.size(array) != OFFICIAL_SAMPLE_COUNT) \
            | F.exists(array, lambda value: value.isNull())
    summary = dataframe.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.countDistinct("timestamp").alias("timestamp_count"),
        F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid_row_count"),
    ).first()
    if submission_mode and summary.row_count != OFFICIAL_SUBMISSION_ROW_COUNT:
        raise ValueError(
            f"Submission predictions must contain exactly {OFFICIAL_SUBMISSION_ROW_COUNT} rows; "
            f"found {summary.row_count}"
        )
    if not submission_mode and summary.row_count == 0:
        raise ValueError("Validation predictions must contain a non-zero number of rows")
    if summary.invalid_row_count:
        raise ValueError(
            "Every prediction timestamp must be present and every target array must "
            f"contain exactly {OFFICIAL_SAMPLE_COUNT} non-null integer values"
        )
    if summary.timestamp_count != summary.row_count:
        raise ValueError("Prediction timestamps must be unique")


def validate_realization_dataframe(
    dataframe,
    *,
    realization_columns: Sequence[str] | None = None,
) -> None:
    """Validate the realization shape used by the official evaluator.

    The current official source checks realization columns positionally, not by
    fixed business names.  ``realization_columns`` is optional and can be used
    by a team that has a protected table with known names.
    """
    _, _, _, TimestampType = _require_spark_types()
    columns = tuple(dataframe.columns)
    if len(columns) != OFFICIAL_COLUMN_COUNT or columns[0] != "timestamp":
        raise ValueError("The realization table must contain timestamp plus exactly four columns")
    if len(set(columns)) != len(columns):
        raise ValueError("The realization table must not contain duplicate columns")
    if realization_columns is not None and columns[1:] != _validate_names(
        realization_columns, label="realization_columns"
    ):
        raise ValueError("Realization columns do not match realization_columns")
    if not isinstance(dataframe.schema.fields[0].dataType, TimestampType):
        raise ValueError("The realization 'timestamp' column must have timestamp type")


def validate_timestamp_alignment(predictions, realizations) -> None:
    """Require both tables to have the same non-empty unique timestamp set."""
    pred_ts = predictions.select("timestamp")
    real_ts = realizations.select("timestamp")
    prediction_count = pred_ts.count()
    realization_count = real_ts.count()
    if prediction_count == 0 or realization_count == 0:
        raise ValueError("Prediction and realization tables must both contain rows")
    if pred_ts.join(real_ts, "timestamp", "left_anti").limit(1).count():
        raise ValueError("Prediction timestamps are missing from the realization table")
    if real_ts.join(pred_ts, "timestamp", "left_anti").limit(1).count():
        raise ValueError("Realization timestamps are missing from the prediction table")
    if prediction_count != realization_count:
        raise ValueError("Prediction and realization timestamps must align exactly")


def write_official_prediction_table(
    forecasts: Iterable,
    prediction_table_name: str,
    *,
    target_columns: Sequence[str],
    spark=None,
    submission_mode: bool = True,
    mode: str = "errorifexists",
    rounding: str = "nearest",
):
    """Write internal forecast outputs to a Delta table with official schema."""
    target_columns = _validate_names(target_columns, label="target_columns")
    records = build_official_prediction_records(
        forecasts, target_columns=target_columns, rounding=rounding
    )
    spark = _spark_session(spark)
    try:
        from pyspark.sql.types import ArrayType, IntegerType, StructField, StructType, TimestampType
    except (ImportError, ModuleNotFoundError) as exc:
        raise OfficialScorerUnavailable("PySpark is unavailable") from exc
    schema = StructType([
        StructField("timestamp", TimestampType(), nullable=False),
        *(StructField(name, ArrayType(IntegerType(), containsNull=False), nullable=False)
          for name in target_columns),
    ])
    dataframe = spark.createDataFrame(records, schema=schema)
    validate_prediction_dataframe(
        dataframe, submission_mode=submission_mode, target_columns=target_columns
    )
    dataframe.write.format("delta").mode(mode).saveAsTable(prediction_table_name)
    return dataframe
