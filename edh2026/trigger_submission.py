import json
import re
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import OperationFailed
from pyspark.sql import SparkSession

DEFAULT_JOB_NAME = "edh26_submission_evaluation"


def _find_job_id(workspace: WorkspaceClient, job_name: str) -> int:
    jobs = list(workspace.jobs.list(name=job_name))
    exact_matches = [
        job for job in jobs if job.settings and job.settings.name == job_name
    ]

    if len(exact_matches) != 1 or exact_matches[0].job_id is None:
        raise RuntimeError(
            f"Expected exactly one Databricks job named '{job_name}', "
            f"found {len(exact_matches)}. Ask an administrator to deploy the EDH26 "
            "submission bundle."
        )

    return exact_matches[0].job_id


def _parse_result(raw_result: str | None) -> dict[str, Any]:
    if not raw_result:
        raise RuntimeError("The evaluation job completed without returning a result.")

    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError as error:
        raise RuntimeError("The evaluation job returned an invalid result.") from error

    if "result" not in result:
        raise RuntimeError(
            result.get("error", "The evaluation job returned no result.")
        )

    return result


def _error_message(output: Any) -> str:
    error_trace = getattr(output, "error_trace", None)
    if error_trace:
        match = re.search(r"(?:ValueError|RuntimeError): (.+)", error_trace)
        if match:
            return match.group(1).strip()

    return getattr(output, "error", None) or "The evaluation job failed."


def _verify_submission_table_access(full_table_name: str) -> None:
    """Confirm that the calling notebook user can read the submitted table."""
    spark = SparkSession.builder.getOrCreate()
    try:
        spark.table(full_table_name).limit(1).collect()
    except Exception as error:
        raise PermissionError(
            f"You do not have read access to '{full_table_name}'. You need USE "
            "CATALOG, USE SCHEMA, and SELECT on the table. Maybe this table does "
            "not even exist!"
        ) from error


def _get_current_user_id(workspace: WorkspaceClient) -> str:
    user_id = workspace.current_user.me().id
    if user_id is None:
        raise RuntimeError("Could not determine the current Databricks user ID.")
    return str(user_id)


def submit_prediction_table(
    schema_name: str,
    table_name: str,
    identifier: str = "predictions",
    *,
    catalog_name: str = "edh",
    job_name: str = DEFAULT_JOB_NAME,
) -> Any:
    """Run the EDH26 evaluation job and return its result to the calling notebook."""
    workspace = WorkspaceClient()
    job_id = _find_job_id(workspace, job_name)
    user_id = _get_current_user_id(workspace)

    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    _verify_submission_table_access(full_table_name)
    print(f"Submitting {full_table_name} for evaluation!")
    run_waiter = workspace.jobs.run_now(
        job_id=job_id,
        notebook_params={
            "catalog_name": catalog_name,
            "schema_name": schema_name,
            "table_name": table_name,
            "identifier": identifier,
            "user_id": user_id,
        },
    )
    failed_run = False
    operation_error = None
    try:
        completed_run = run_waiter.result()
    except OperationFailed as error:
        completed_run = workspace.jobs.get_run(run_waiter.response.run_id)
        failed_run = True
        operation_error = error

    if not completed_run.tasks or completed_run.tasks[0].run_id is None:
        raise RuntimeError("The evaluation job did not create a task run.")

    task = completed_run.tasks[0]
    output = workspace.jobs.get_run_output(task.run_id)
    result_state = getattr(getattr(task, "state", None), "result_state", None)
    result_state_value = getattr(result_state, "value", result_state)
    if failed_run or result_state_value not in {None, "SUCCESS"}:
        message = _error_message(output)
        print(f"Evaluation failed: {message}")
        if failed_run:
            raise RuntimeError(message) from operation_error
        raise RuntimeError(message)

    result = _parse_result(
        output.notebook_output.result if output.notebook_output else None
    )
    print("Evaluation complete.")
    return result["result"]
