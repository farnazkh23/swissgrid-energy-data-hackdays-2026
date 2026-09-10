# Databricks notebook source
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
EXPECTED_ARRAY_LENGTH = 300
EXPECTED_COLUMN_COUNT = 5
EXPECTED_ROWS_COUNT = 168
REALIZATIONS_TABLE_NAME = "edh.edh_admin.realizations"
RESULTS_TABLE_NAME = "edh.eval.submission_results"
SUBMISSION_AUTHORIZATIONS_TABLE_NAME = "edh.edh_admin.submission_authorizations"
SUBMISSION_CATALOG_NAME = "edh"
SUBMISSION_COOLDOWN_MINUTES = 30
CAPTURE_WEIGHT = 0.2
ERROR_WEIGHT = 0.45
SHARPNESS_WEIGHT = 0.35
KERNEL_BANDWIDTH = 3.0
PERFECT_SCORE = CAPTURE_WEIGHT + ERROR_WEIGHT + SHARPNESS_WEIGHT  # 1.0


def _widget(name: str) -> str:
    value = dbutils.widgets.get(name).strip()
    if not value:
        raise ValueError(f"The '{name}' parameter must not be empty.")
    return value


def evaluate_submission(
    submission_table: DataFrame,
    realizations_table: DataFrame,
) -> Any:
    """Compute the weighted forecast score against the realizations."""
    if submission_table is None:
        raise ValueError("The submission table does not exist.")

    columns = submission_table.columns
    if len(columns) != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            "The submission table must contain exactly five columns: a timestamp "
            "column followed by four array<int> columns."
        )
    if len(columns) != len(set(columns)):
        raise ValueError("The submission table must not contain duplicate columns.")
    if columns[0] != "timestamp":
        raise ValueError("The first column must be named 'timestamp'.")

    invalid_names = [name for name in columns if not VALID_NAME.fullmatch(name)]
    if invalid_names:
        raise ValueError(
            "Column names may contain only letters, numbers, underscores, and dashes. "
            f"Invalid columns: {', '.join(invalid_names)}."
        )

    fields = submission_table.schema.fields
    if not isinstance(fields[0].dataType, TimestampType):
        raise ValueError("The 'timestamp' column must have timestamp type.")

    invalid_array_columns = [
        field.name
        for field in fields[1:]
        if not (
            isinstance(field.dataType, ArrayType)
            and isinstance(field.dataType.elementType, IntegerType)
        )
    ]
    if invalid_array_columns:
        raise ValueError(
            "All columns after 'timestamp' must have array<int> type. "
            f"Invalid columns: {', '.join(invalid_array_columns)}."
        )

    array_columns = columns[1:]
    invalid_value = F.col("timestamp").isNull()
    forecast_means = []
    forecast_minima = []
    forecast_maxima = []
    forecast_standard_deviations = []
    for column_name in array_columns:
        array = F.col(column_name)
        invalid_value = (
            invalid_value
            | array.isNull()
            | (F.size(array) != EXPECTED_ARRAY_LENGTH)
            | F.exists(array, lambda value: value.isNull())
        )
        mean = (
            F.aggregate(
                array,
                F.lit(0.0).cast("double"),
                lambda total, value: total + value.cast("double"),
            )
            / EXPECTED_ARRAY_LENGTH
        )
        forecast_means.append(mean)
        forecast_minima.append(F.array_min(array).cast("double"))
        forecast_maxima.append(F.array_max(array).cast("double"))
        forecast_standard_deviations.append(
            F.sqrt(
                F.aggregate(
                    array,
                    F.lit(0.0).cast("double"),
                    lambda total, value: total
                    + (value.cast("double") - mean) * (value.cast("double") - mean),
                )
                / EXPECTED_ARRAY_LENGTH
            )
        )

    summary = submission_table.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.countDistinct("timestamp").alias("timestamp_count"),
        F.sum(F.when(invalid_value, 1).otherwise(0)).alias("invalid_row_count"),
    ).first()

    if summary.row_count != EXPECTED_ROWS_COUNT:
        raise ValueError(
            f"The submission table must contain exactly {EXPECTED_ROWS_COUNT} rows; "
            f"found {summary.row_count}."
        )
    if summary.invalid_row_count:
        raise ValueError(
            "Every timestamp must be present, and every array must contain exactly "
            f"{EXPECTED_ARRAY_LENGTH} non-null integer values."
        )
    if summary.timestamp_count != EXPECTED_ROWS_COUNT:
        raise ValueError("Every submission timestamp must be unique.")

    realization_columns = realizations_table.columns
    forecasts = submission_table.select(
        "timestamp",
        *[
            expression.alias(f"forecast_{statistic}_{index}")
            for statistic, expressions in (
                ("mean", forecast_means),
                ("minimum", forecast_minima),
                ("maximum", forecast_maxima),
                ("standard_deviation", forecast_standard_deviations),
            )
            for index, expression in enumerate(expressions)
        ],
    )
    realizations = realizations_table.select(
        "timestamp",
        *[
            F.col(column_name).cast("double").alias(f"realization_{index}")
            for index, column_name in enumerate(realization_columns[1:])
        ],
    )
    distribution_scores = []
    for index in range(EXPECTED_COLUMN_COUNT - 1):
        realization = F.col(f"realization_{index}")
        minimum = F.col(f"forecast_minimum_{index}")
        maximum = F.col(f"forecast_maximum_{index}")
        mean = F.col(f"forecast_mean_{index}")
        standard_deviation = F.col(f"forecast_standard_deviation_{index}")
        distribution_range = maximum - minimum
        captured = F.when(realization.between(minimum, maximum), F.lit(1.0)).otherwise(
            F.lit(0.0)
        )
        zero_spread = standard_deviation == F.lit(0.0)
        goodness = F.when(captured == F.lit(0.0), F.lit(0.0)).otherwise(
            F.when(
                zero_spread,
                F.when(realization == mean, F.lit(1.0)).otherwise(F.lit(0.0)),
            ).otherwise(
                F.greatest(
                    F.lit(0.0),
                    F.lit(1.0)
                    - F.abs(realization - mean)
                    / (F.lit(KERNEL_BANDWIDTH) * standard_deviation),
                )
            )
        )
        sharpness = (
            F.when(captured == F.lit(0.0), F.lit(0.0))
            .when(zero_spread, F.lit(1.0))
            .otherwise(
                distribution_range
                / (distribution_range + F.lit(2.0) * standard_deviation)
            )
        )
        distribution_scores.append(
            CAPTURE_WEIGHT * captured
            + ERROR_WEIGHT * goodness
            + SHARPNESS_WEIGHT * sharpness
        )
    row_score = sum(distribution_scores, F.lit(0.0)) / len(distribution_scores)
    score_summary = (
        forecasts.join(realizations, "timestamp", "inner")
        .agg(
            F.count(F.lit(1)).alias("row_count"),
            F.avg(row_score).alias("score"),
        )
        .first()
    )
    if score_summary.row_count != EXPECTED_ROWS_COUNT:
        raise ValueError("Submission and realization timestamps must match exactly.")

    return float(score_summary.score)


def normalize_score_to_percentage(score: float) -> float:
    """Map the raw score in [0, 1] to a percentage; a perfect prediction is 100%."""
    return min(100.0, max(0.0, 100.0 * score / PERFECT_SCORE))


def _load_submission_table(spark: SparkSession, full_table_name: str) -> DataFrame:
    if not spark.catalog.tableExists(full_table_name):
        raise ValueError(f"The submission table '{full_table_name}' does not exist.")
    return spark.table(full_table_name)


def _enforce_submission_authorization(
    spark: SparkSession, catalog_name: str, schema_name: str, user_id: str
) -> None:
    if catalog_name != SUBMISSION_CATALOG_NAME:
        raise ValueError(
            f"Submissions must be in the '{SUBMISSION_CATALOG_NAME}' catalog."
        )

    authorized_schemas = {
        row.schema_name
        for row in (
            spark.table(SUBMISSION_AUTHORIZATIONS_TABLE_NAME)
            .where(F.col("user_id") == user_id)
            .select("schema_name")
            .collect()
        )
    }
    if schema_name not in authorized_schemas:
        raise ValueError(
            f"The user ID '{user_id}' is not authorized to submit tables from "
            f"the '{schema_name}' schema."
        )


def _get_last_submission_timestamp(
    spark: SparkSession, group_name: str
) -> datetime | None:
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    if not spark.catalog.tableExists(RESULTS_TABLE_NAME):
        return None

    summary = (
        spark.table(RESULTS_TABLE_NAME)
        .where(F.col("group_name") == group_name)
        .agg(F.max("timestamp").alias("last_timestamp"))
        .first()
    )
    if summary is None:
        return None

    if summary.last_timestamp is None:
        return None

    last_timestamp = summary.last_timestamp
    if last_timestamp.tzinfo is None:
        return last_timestamp.replace(tzinfo=timezone.utc)
    return last_timestamp.astimezone(timezone.utc)


def _enforce_submission_cooldown(spark: SparkSession, group_name: str) -> None:
    last_submission_timestamp = _get_last_submission_timestamp(spark, group_name)
    if last_submission_timestamp is None:
        return

    cooldown_ends_at = last_submission_timestamp + timedelta(
        minutes=SUBMISSION_COOLDOWN_MINUTES
    )
    now = datetime.now(timezone.utc)
    if now < cooldown_ends_at:
        remaining_seconds = int((cooldown_ends_at - now).total_seconds())
        remaining_minutes = max(1, (remaining_seconds + 59) // 60)
        raise ValueError(
            f"The team '{group_name}' submitted too recently. Please wait "
            f"{remaining_minutes} more minute(s) before submitting again."
        )


def write_to_results_table(
    spark: SparkSession,
    result: float,
    identifier: str,
    group_name: str,
) -> None:
    """Append an EDH26 evaluation result to the shared results table."""
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    result_schema = StructType(
        [
            StructField("group_name", StringType(), nullable=False),
            StructField("result", DoubleType(), nullable=False),
            StructField("identifier", StringType(), nullable=False),
            StructField("timestamp", TimestampType(), nullable=False),
        ]
    )
    result_table = spark.createDataFrame(
        [
            (
                group_name,
                float(result),
                identifier,
                datetime.now(timezone.utc),
            )
        ],
        schema=result_schema,
    )
    result_table.write.mode("append").saveAsTable(RESULTS_TABLE_NAME)


def _run_notebook() -> None:
    catalog_name = _widget("catalog_name")
    schema_name = _widget("schema_name")
    table_name = _widget("table_name")
    identifier = _widget("identifier")
    user_id = _widget("user_id")

    for parameter_name, parameter_value in {
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "identifier": identifier,
    }.items():
        if not VALID_NAME.fullmatch(parameter_value):
            raise ValueError(
                f"The '{parameter_name}' parameter may contain only letters, "
                "numbers, and underscores."
            )

    spark = SparkSession.builder.getOrCreate()
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    _enforce_submission_authorization(spark, catalog_name, schema_name, user_id)
    _enforce_submission_cooldown(spark, schema_name)
    submission_table = _load_submission_table(spark, full_table_name)
    realizations_table = _load_submission_table(spark, REALIZATIONS_TABLE_NAME)
    result = evaluate_submission(submission_table, realizations_table)
    write_to_results_table(spark, result, identifier, schema_name)
    dbutils.notebook.exit(
        json.dumps(
            {
                "result": result,
                "table": full_table_name,
                "identifier": identifier,
            }
        )
    )


if "dbutils" in globals():
    try:
        _run_notebook()
    except Exception as error:
        print(f"Evaluation failed: {error}")
        raise
