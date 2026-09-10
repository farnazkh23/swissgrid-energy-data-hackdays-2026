# Databricks local scoring runbook

This repository now has a thin adapter around the official EDH helper. It
imports only `edh2026.local_scoring.score_prediction_table`; it never imports
`edh2026.trigger_submission` and never starts the official evaluation job.

## Contract confirmed from the challenge source

The official evaluator requires the prediction table to have:

- exactly five columns;
- `timestamp` as the first column and Spark `timestamp`/`TimestampType`;
- four following columns of `array<int>`;
- exactly 300 non-null integer elements in every array;
- no null timestamps, unique timestamps, and exactly 168 rows for submission;
- exact timestamp alignment with the realization table.

The evaluator source does not enforce fixed names for the four business
columns. It uses the four columns positionally. It likewise reads realization
columns positionally after `timestamp`. The writer therefore requires the
team to pass the four names used by its realization table rather than silently
guessing a hidden protected-table schema. Names may contain only letters,
numbers, underscores, and dashes.

The challenge README says validation tables may contain any non-zero aligned
period. The currently deployed `evaluate_submission` source inspected at
`/Workspace/github/dp-light-edh/src/edh2026/evaluate_submission` still has a
hard-coded 168-row check in `evaluate_submission`. The adapter's standalone
validation accepts any non-zero unique period, but the official local scorer
can score it only if the deployed evaluator permits that period. With the
currently inspected source, use 168 aligned validation rows for scoring.

The current evaluator does not validate realization business-column names or
reject null realization values. The adapter checks the five-column shape and
timestamp type, and leaves value semantics to the official scorer.

## Notebook setup

Run this in a Python notebook attached to a Databricks cluster:

```python
import sys

# Official challenge helper; this is read-only and is not the submission helper.
sys.path.insert(0, "/Workspace/github/dp-light-edh/src")

# Make the checked-out Swissgrid package importable. Set this to the workspace
# path where this repository is synced for the team.
sys.path.insert(0, "/Workspace/Users/<your-user>/swissgrid-databricks-scorer/src")

from swissgrid_forecaster.databricks_scoring import (
    score_week_with_official_scorer,
    validate_prediction_dataframe,
    validate_realization_dataframe,
    validate_timestamp_alignment,
)
```

If the package is installed on the cluster, omit the second `sys.path.insert`.
Do not import `edh2026.trigger_submission`.

## Validate existing tables

```python
prediction_table = "edh.group_n.validation_predictions"
realization_table = "edh.group_n.validation_realizations"

predictions = spark.table(prediction_table)
realizations = spark.table(realization_table)

validate_prediction_dataframe(predictions, submission_mode=True)
validate_realization_dataframe(realizations)
validate_timestamp_alignment(predictions, realizations)
print(predictions.schema.simpleString())
print(f"prediction rows: {predictions.count()}")
print(f"realization rows: {realizations.count()}")
```

For a non-submission validation period, use `submission_mode=False` in the
first validation call. The period must still be non-empty, unique, and exactly
aligned with the realizations table. The deployed official scorer currently
requires 168 rows as noted above.

## Score locally with the official scorer

```python
score = score_week_with_official_scorer(
    prediction_table_name=prediction_table,
    realization_table_name=realization_table,
)
print(f"Official local raw score: {score:.8f}")
```

This is the complete scoring call. It reads both tables on the attached
cluster and invokes the local evaluator notebook through the official helper.
It does not call a Databricks job, consume a submission slot, write results,
or publish a score.

## Write internal forecast outputs as Delta

The internal pipeline emits one `ForecastOutput` per target and target time.
Each output must already contain exactly 300 samples. The writer converts
finite numeric samples to integer arrays using deterministic nearest-integer
rounding, then writes a Delta table with `timestamp` plus the four configured
target columns. Use `rounding="exact"` to fail instead of rounding.

```python
from swissgrid_forecaster.databricks_scoring import write_official_prediction_table

target_columns = ("CH-AT", "CH-DE", "CH-FR", "CH-IT")
# Replace `forecast_outputs` with the team's four-target ForecastOutput iterable.
write_official_prediction_table(
    forecast_outputs,
    "edh.group_n.validation_predictions",
    target_columns=target_columns,
    submission_mode=True,
    mode="overwrite",
    rounding="nearest",
)
```

The target names above are an example mapping, not a claim about protected
realization-table names; use the four names selected by the team and keep the
same order for every forecast row. The official evaluator is positional.

## Local fallback and tests

The adapter has no PySpark or Databricks SDK import at module import time.
Outside Databricks, `official_scorer_available()` returns `False`, and the
runtime probe test is skipped. Run the repository test suite locally with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```
