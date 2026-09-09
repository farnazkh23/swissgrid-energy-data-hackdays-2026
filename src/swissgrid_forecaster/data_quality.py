"""Diagnostic flags; invalid clocks are reported before Observation construction."""
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from .availability import is_available, utc
from .observation_schema import Observation


class QualityFlag(str, Enum):
    MISSING = 'missing'
    STALE = 'stale'
    REVISED = 'revised'
    DUPLICATE = 'duplicate'
    OUT_OF_RANGE = 'out_of_range'
    TIMEZONE_ISSUE = 'timezone_issue'
    UNAVAILABLE_AT_ISSUE = 'unavailable_at_issue'


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    max_age: timedelta | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if self.max_age is not None and (not isinstance(self.max_age, timedelta) or self.max_age < timedelta(0)):
            raise ValueError('max_age must be nonnegative')
        for value in (self.minimum, self.maximum):
            if value is not None and (type(value) not in (float, int) or not isfinite(value)):
                raise ValueError('range bounds must be finite numbers')
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError('minimum exceeds maximum')


def timestamp_flags(**timestamps: datetime | None) -> frozenset[QualityFlag]:
    """Pass required clocks only; omit unknown optional publication_time."""
    flags = set()
    for name, value in timestamps.items():
        if value is None:
            flags.update((QualityFlag.MISSING, QualityFlag.UNAVAILABLE_AT_ISSUE))
        else:
            try:
                utc(value, name)
            except ValueError:
                flags.add(QualityFlag.TIMEZONE_ISSUE)
    return frozenset(flags)


def assess(records: Iterable[Observation], *, issue_time: datetime,
           policy: QualityPolicy = QualityPolicy()) -> tuple[frozenset[QualityFlag], ...]:
    """Return flags aligned with input; stale means age of event_time at issue.

    Duplicate means repeated source/record/revision identity (all occurrences).
    As-of conflict detection remains the resolver's responsibility.
    """
    issue = utc(issue_time, 'issue_time')
    rows = tuple(records)
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        if not isinstance(row, Observation):
            raise ValueError('expected validated Observation')
        key = (*row.logical_key, row.revision_id)
        counts[key] = counts.get(key, 0) + 1
    results = []
    for row in rows:
        flags = set()
        if row.value is None:
            flags.add(QualityFlag.MISSING)
        if policy.max_age is not None and issue - row.event_time > policy.max_age:
            flags.add(QualityFlag.STALE)
        if row.supersedes_revision_id is not None:
            flags.add(QualityFlag.REVISED)
        if counts[(*row.logical_key, row.revision_id)] > 1:
            flags.add(QualityFlag.DUPLICATE)
        if type(row.value) in (int, float) and (
            (policy.minimum is not None and row.value < policy.minimum) or
            (policy.maximum is not None and row.value > policy.maximum)
        ):
            flags.add(QualityFlag.OUT_OF_RANGE)
        if not is_available(row.known_at, issue):
            flags.add(QualityFlag.UNAVAILABLE_AT_ISSUE)
        results.append(frozenset(flags))
    return tuple(results)
