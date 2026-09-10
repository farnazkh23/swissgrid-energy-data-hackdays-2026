"""Pure-Python reimplementation of edh2026's official row-scoring formula.

`edh2026/evaluate_submission.py` runs on PySpark inside Databricks and cannot
be imported here (this project has no pyspark/databricks-sdk dependency, and
none is installed automatically). This module reproduces its documented
per-row math exactly, so the same score can be computed locally against any
samples/realizations pair before ever touching Databricks. Constants are
copied from `edh2026/evaluate_submission.py`; keep them in sync if the
challenge changes them.

Two distinct checks exist on purpose:
- `score_submission` / `score_row` mirror the row-score math used by both
  local validation (`edh2026.local_scoring.score_prediction_table`, any
  nonzero row count) and the official submission.
- `validate_submission_rows` mirrors the *submission-only* structural checks
  (exactly 168 rows, exactly 300 non-null integers per array). Local
  validation tables are not required to satisfy this.
"""
from math import isfinite, sqrt

EXPECTED_ARRAY_LENGTH = 300
EXPECTED_ROWS_COUNT = 168
CAPTURE_WEIGHT = 0.2
ERROR_WEIGHT = 0.45
SHARPNESS_WEIGHT = 0.35
KERNEL_BANDWIDTH = 3.0
PERFECT_SCORE = CAPTURE_WEIGHT + ERROR_WEIGHT + SHARPNESS_WEIGHT  # 1.0


def score_distribution(samples, realization: float) -> float:
    """One column's contribution to a row's score; mirrors evaluate_submission's per-column formula."""
    values = tuple(samples)
    if len(values) != EXPECTED_ARRAY_LENGTH:
        raise ValueError(f"samples must contain exactly {EXPECTED_ARRAY_LENGTH} values")
    if any(value is None or isinstance(value, bool) or not isfinite(value) for value in values):
        raise ValueError("samples must be complete, finite numbers")
    if isinstance(realization, bool) or not isfinite(realization):
        raise ValueError("realization must be finite")
    minimum, maximum = min(values), max(values)
    if not minimum <= realization <= maximum:
        return 0.0
    mean = sum(values) / len(values)
    std = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    if std == 0.0:
        goodness = 1.0 if realization == mean else 0.0
        sharpness = 1.0
    else:
        goodness = max(0.0, 1.0 - abs(realization - mean) / (KERNEL_BANDWIDTH * std))
        distribution_range = maximum - minimum
        sharpness = distribution_range / (distribution_range + 2.0 * std)
    return CAPTURE_WEIGHT * 1.0 + ERROR_WEIGHT * goodness + SHARPNESS_WEIGHT * sharpness


def score_row(samples_by_target: dict, realization_by_target: dict) -> float:
    """Average of `score_distribution` across all target columns for one timestamp."""
    if set(samples_by_target) != set(realization_by_target):
        raise ValueError("samples and realizations must cover the same targets")
    if not samples_by_target:
        raise ValueError("at least one target column is required")
    scores = [score_distribution(samples_by_target[name], realization_by_target[name])
             for name in samples_by_target]
    return sum(scores) / len(scores)


def score_submission(rows) -> float:
    """Mean of `score_row` over `(samples_by_target, realization_by_target)` rows.

    Mirrors `evaluate_submission`'s whole-table average. Any nonzero row count
    is accepted here (this is the local/validation path); use
    `validate_submission_rows` separately before an official submission.
    """
    rows = tuple(rows)
    if not rows:
        raise ValueError("at least one row is required")
    return sum(score_row(samples, realization) for samples, realization in rows) / len(rows)


def normalize_score_to_percentage(score: float) -> float:
    """Map the raw score in [0, 1] to a percentage; a perfect prediction is 100%."""
    return min(100.0, max(0.0, 100.0 * score / PERFECT_SCORE))


def validate_submission_rows(rows: dict, targets) -> None:
    """Fail closed on the official submission-only structural checks.

    `rows` maps timestamp -> {target: samples}. Row-count and array-length
    checks apply only to a final submission table, not to local validation.
    """
    targets = tuple(targets)
    if not targets:
        raise ValueError("at least one target column is required")
    if len(rows) != EXPECTED_ROWS_COUNT:
        raise ValueError(f"submission must contain exactly {EXPECTED_ROWS_COUNT} rows; found {len(rows)}")
    for timestamp, samples_by_target in rows.items():
        if set(samples_by_target) != set(targets):
            raise ValueError(f"row {timestamp} must contain exactly the target columns {targets}")
        for name in targets:
            values = samples_by_target[name]
            if len(values) != EXPECTED_ARRAY_LENGTH:
                raise ValueError(f"{name} at {timestamp} must contain exactly {EXPECTED_ARRAY_LENGTH} samples")
            if any(value is None or isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"{name} at {timestamp} must contain only non-null integers")
