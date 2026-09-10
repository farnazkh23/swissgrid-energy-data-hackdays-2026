"""Positional challenge submission table writer with target-order assertions.

The official evaluator maps realizations positionally, so column names alone
do not protect us. The internal mapping is fixed and asserted before any
write:

    target_0 -> AT
    target_1 -> DE
    target_2 -> FR
    target_3 -> IT

CH is never a submission target; it may only enter as an input/context
feature. Every builder call re-asserts the canonical target order from
``swissgrid_forecaster.target_contract``.
"""
from datetime import datetime, timedelta
from math import isfinite

from .availability import utc
from .target_contract import (InvalidOutputTargets, TARGETS, TARGET_POSITIONS,
                              validate_output_targets)

SUBMISSION_ROWS = 168
SAMPLES_PER_TARGET = 300
SUBMISSION_COLUMNS = ("timestamp", "target_0", "target_1", "target_2", "target_3")
SUBMISSION_HORIZON = timedelta(hours=1)


def submission_target_mapping() -> dict[str, str]:
    """Documented internal mapping from positional column to scored target."""
    return {f"target_{index}": entity for index, entity in enumerate(TARGETS)}


def _validate_samples(entity, values):
    if values is None or hasattr(values, "keys") or not hasattr(values, "__iter__"):
        raise ValueError(f"{entity} samples must be an ordered sequence")
    samples = tuple(values)
    if len(samples) != SAMPLES_PER_TARGET:
        raise ValueError(f"{entity} requires exactly {SAMPLES_PER_TARGET} samples per hour, got {len(samples)}")
    converted = []
    for value in samples:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{entity} samples must be finite numbers")
        converted.append(int(value) if isinstance(value, int) else int(round(float(value))))
    return tuple(converted)


def _normalize_timestamps(timestamps):
    if timestamps is None:
        raise ValueError("168 challenge-order timestamps are required")
    if isinstance(timestamps, (str, datetime)) or hasattr(timestamps, "keys") \
            or not hasattr(timestamps, "__iter__"):
        raise ValueError("timestamps must be an ordered sequence")
    sequence = tuple(timestamps)
    if len(sequence) != SUBMISSION_ROWS:
        raise ValueError(f"submission requires exactly {SUBMISSION_ROWS} timestamps, got {len(sequence)}")
    return tuple(utc(value, "timestamp") for value in sequence)


def build_submission_table(series_by_target, timestamps):
    """Build the positional submission rows from per-target sample series.

    ``series_by_target`` maps each scored entity (exactly AT, DE, FR, IT) to a
    sequence of 168 per-hour sample sequences of 300 values each. Floating
    point samples are rounded to integers because the challenge column type
    is ``array<int>``.
    """
    if not isinstance(series_by_target, dict):
        raise ValueError("series_by_target must be a mapping of entity -> sample series")
    if "CH" in series_by_target:
        raise InvalidOutputTargets("CH is an input/context country, not a submission target")
    if set(series_by_target) != set(TARGETS):
        raise InvalidOutputTargets(
            "submission requires exactly the four scored targets " + ", ".join(TARGETS) +
            "; got " + (", ".join(sorted(series_by_target)) or "none"))
    validate_output_targets(TARGETS)
    stamps = _normalize_timestamps(timestamps)
    series = {}
    for entity in TARGETS:
        values = series_by_target[entity]
        if values is None or hasattr(values, "keys") or not hasattr(values, "__iter__"):
            raise ValueError(f"{entity} series must be an ordered sequence of 168 hours")
        hours = tuple(values)
        if len(hours) != SUBMISSION_ROWS:
            raise ValueError(f"{entity} series requires exactly {SUBMISSION_ROWS} hours, got {len(hours)}")
        series[entity] = tuple(_validate_samples(entity, hour) for hour in hours)
    rows = []
    for index, stamp in enumerate(stamps):
        row = {"timestamp": stamp.isoformat()}
        for entity in TARGETS:
            row[f"target_{TARGET_POSITIONS[entity]}"] = list(series[entity][index])
        rows.append(row)
    return tuple(rows)


def validate_submission_table(rows):
    """Assert an already-built submission table preserves the target contract.

    Returns the documented positional mapping; raises on any violation of the
    168 x 4 x 300 integer contract or the AT/DE/FR/IT order.
    """
    if rows is None or hasattr(rows, "keys") or not hasattr(rows, "__iter__"):
        raise ValueError("submission table must be an ordered sequence of rows")
    sequence = tuple(rows)
    if len(sequence) != SUBMISSION_ROWS:
        raise ValueError(f"submission requires exactly {SUBMISSION_ROWS} rows, got {len(sequence)}")
    for row in sequence:
        if not isinstance(row, dict) or tuple(row) != SUBMISSION_COLUMNS:
            raise ValueError("submission row columns must be exactly " + ", ".join(SUBMISSION_COLUMNS))
        stamp = row["timestamp"]
        if isinstance(stamp, str):
            try:
                stamp = datetime.fromisoformat(stamp)
            except ValueError as exc:
                raise ValueError("submission timestamp must be an ISO-8601 string") from exc
        utc(stamp, "timestamp")
        for column in SUBMISSION_COLUMNS[1:]:
            _validate_samples(column, row[column])
    return submission_target_mapping()
