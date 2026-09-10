from pyspark.sql import SparkSession


def score_prediction_table(
    prediction_table_name: str,
    realizations_table_name: str,
) -> float:
    """Score a prediction table locally against caller-provided realizations.

    This function does not submit a result, enforce the submission cooldown, or
    write to the official results table.
    """
    from databricks.sdk.runtime import dbutils

    spark = SparkSession.builder.getOrCreate()
    prediction_table = spark.table(prediction_table_name)
    realizations_table = spark.table(realizations_table_name)
    evaluator_notebook = dbutils.import_notebook("edh2026.evaluate_submission")
    return evaluator_notebook.evaluate_submission(prediction_table, realizations_table)
