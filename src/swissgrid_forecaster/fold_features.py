"""Pure elapsed-time transforms with explicit fold-local fit provenance."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from statistics import mean, pstdev
from typing import Protocol, Sequence
from .availability import nonempty, utc
from .feature_registry import identity


@dataclass(frozen=True)
class FeatureRow:
    row_id: str
    event_time: datetime
    known_at: datetime
    value: float | None
    source_version: str
    partition: str = 'train'

    def __post_init__(self):
        for name in ('row_id', 'source_version'):
            nonempty(getattr(self, name), name)
        for name in ('event_time', 'known_at'):
            object.__setattr__(self, name, utc(getattr(self, name), name))
        if self.partition not in ('train', 'validation', 'calibration', 'holdout'):
            raise ValueError('invalid partition')
        if self.value is not None and not isfinite(self.value):
            raise ValueError('nonfinite value')


@dataclass(frozen=True)
class TransformArtifact:
    fit_cutoff: datetime
    feature_version: str
    parent_source_versions: tuple[str, ...]
    training_row_ids: tuple[str, ...]


@dataclass(frozen=True)
class FeatureValue:
    values: tuple[float | None, ...]
    known_at: datetime
    parent_source_versions: tuple[str, ...]


class FoldTransform(Protocol):
    artifact: TransformArtifact | None
    def fit(self, train_rows: Sequence[FeatureRow], cutoff: datetime) -> 'FoldTransform': ...
    def transform(self, rows: Sequence[FeatureRow], issue_time: datetime) -> FeatureValue: ...


class TimeTransform:
    """One scalar series; callers supply PIT-resolved vintages, not mixed series.

    Windows are (issue-window, issue]; lag requires an exact elapsed-time match.
    Rolling std uses population convention; rate is change per elapsed second.
    Calendar emits UTC weekday/hour/month. No target calendar is guessed.
    """
    def __init__(self, name: str, *, window: timedelta | None = None,
                 missingness_policy: str = 'propagate', version: str = '1'):
        if name not in ('lag', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
                        'rate_of_change', 'missingness_flag', 'freshness', 'calendar'):
            raise ValueError('unknown transform')
        if name in ('lag', 'rate_of_change') or name.startswith('rolling_'):
            if not isinstance(window, timedelta) or window <= timedelta(0):
                raise ValueError('positive elapsed window required')
        if missingness_policy not in ('error', 'propagate'):
            raise ValueError('invalid missingness policy')
        self.name, self.window, self.missingness_policy = name, window, missingness_policy
        nonempty(version, 'transform version')
        self.feature_version = identity([name, str(window), missingness_policy, version])
        self.artifact = None

    def fit(self, train_rows, cutoff):
        self.artifact = None
        cutoff = utc(cutoff)
        rows = tuple(train_rows)
        if any(r.partition != 'train' or r.known_at > cutoff or r.event_time > cutoff for r in rows):
            raise ValueError('fit accepts only training evidence available at cutoff')
        if len({r.row_id for r in rows}) != len(rows):
            raise ValueError('duplicate training rows')
        self.artifact = TransformArtifact(cutoff, self.feature_version,
                                         tuple(sorted({r.source_version for r in rows})),
                                         tuple(sorted(r.row_id for r in rows)))
        return self

    def transform(self, rows, issue_time):
        issue = utc(issue_time)
        if self.artifact is None or self.artifact.fit_cutoff > issue:
            raise ValueError('transform must be fitted by issue time')
        eligible = sorted((r for r in rows if r.known_at <= issue and r.event_time <= issue),
                          key=lambda r: (r.event_time, r.row_id))
        if len({r.event_time for r in eligible}) != len(eligible):
            raise ValueError('resolve revisions and series before transforming')
        used = []
        if self.name == 'calendar':
            values = (float(issue.weekday()), float(issue.hour), float(issue.month))
        elif self.name in ('freshness', 'missingness_flag'):
            used = eligible[-1:]
            values = ((float((issue-used[0].event_time).total_seconds()) if used else None),) if self.name == 'freshness' else (float(not used or used[0].value is None),)
        elif self.name == 'lag':
            used = [r for r in eligible if r.event_time == issue-self.window]
            values = (used[0].value if used else None,)
        elif self.name == 'rate_of_change':
            used = [r for r in eligible if r.event_time in (issue-self.window, issue)]
            values = ((used[1].value-used[0].value)/self.window.total_seconds(),) if len(used) == 2 and all(r.value is not None for r in used) else (None,)
        else:
            used = [r for r in eligible if r.event_time > issue-self.window]
            data = [r.value for r in used]
            fn = {'rolling_mean': mean, 'rolling_std': pstdev, 'rolling_min': min, 'rolling_max': max}[self.name]
            values = (float(fn(data)),) if data and all(v is not None for v in data) else (None,)
        if None in values and self.missingness_policy == 'error':
            raise ValueError('missing transform input')
        return FeatureValue(values, max([self.artifact.fit_cutoff] + [r.known_at for r in used]),
                            tuple(sorted(set(self.artifact.parent_source_versions) | {r.source_version for r in used})))


class FoldStandardScaler(TimeTransform):
    """Optional generic preprocessing guard; no global statistics."""
    def __init__(self):
        super().__init__('lag', window=timedelta(seconds=1))
        self.feature_version = identity(['standard_scaler', '1'])

    def fit(self, train_rows, cutoff):
        rows = tuple(train_rows)
        super().fit(rows, cutoff)
        if not rows or any(r.value is None for r in rows):
            self.artifact = None
            raise ValueError('scaler requires complete training values')
        self.center = mean(r.value for r in rows)
        self.scale = pstdev(r.value for r in rows) or 1.0
        return self

    def transform(self, rows, issue_time):
        issue = utc(issue_time)
        if self.artifact is None or self.artifact.fit_cutoff > issue:
            raise ValueError('scaler not fitted by issue')
        rows = tuple(rows)
        if any(r.known_at > issue or r.event_time > issue for r in rows):
            raise ValueError('future scaler input')
        return FeatureValue(tuple(None if r.value is None else (r.value-self.center)/self.scale for r in rows),
                            max([self.artifact.fit_cutoff] + [r.known_at for r in rows]),
                            tuple(sorted(set(self.artifact.parent_source_versions) | {r.source_version for r in rows})))
