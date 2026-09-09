"""Elapsed-time rolling origins. All intervals are half-open, in UTC."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import isfinite
from .availability import utc, nonempty
from .feature_registry import identity


@dataclass(frozen=True)
class Sample:
    row_id: str
    issue_time: datetime
    target_time: datetime
    feature_known_at: datetime
    label_known_at: datetime
    features: tuple[float | None, ...]
    target: float | None
    partition: str = 'train'

    def __post_init__(self):
        nonempty(self.row_id, 'row_id')
        for name in ('issue_time', 'target_time', 'feature_known_at', 'label_known_at'):
            object.__setattr__(self, name, utc(getattr(self, name)))
        object.__setattr__(self, 'features', tuple(self.features))
        if self.target_time <= self.issue_time or self.label_known_at < self.target_time:
            raise ValueError('invalid target/label chronology')
        if self.partition not in ('train', 'validation', 'calibration', 'holdout'):
            raise ValueError('invalid partition')
        if any(v is not None and not isfinite(v) for v in (*self.features, self.target)):
            raise ValueError('nonfinite sample')

    def manifest_entry(self):
        return [self.row_id, self.issue_time.isoformat(), self.target_time.isoformat(),
                self.feature_known_at.isoformat(), self.label_known_at.isoformat(),
                self.features, self.target, self.partition]


def data_manifest(rows):
    rows = tuple(rows)
    if len({r.row_id for r in rows}) != len(rows):
        raise ValueError('duplicate sample IDs')
    return identity([r.manifest_entry() for r in sorted(rows, key=lambda r: r.row_id)])


@dataclass(frozen=True)
class Fold:
    fold_id: str
    fit_cutoff: datetime
    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    calibration: tuple[Sample, ...]
    holdout_start: datetime


@dataclass(frozen=True)
class RollingOrigin:
    first_origin: datetime
    holdout_start: datetime
    train_window: timedelta
    validation_window: timedelta
    calibration_window: timedelta
    step: timedelta
    purge: timedelta = timedelta(0)
    embargo: timedelta = timedelta(0)
    label_delay: timedelta = timedelta(0)

    def __post_init__(self):
        for name in ('first_origin', 'holdout_start'):
            object.__setattr__(self, name, utc(getattr(self, name)))
        for name in ('train_window', 'validation_window', 'calibration_window', 'step', 'purge', 'embargo', 'label_delay'):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value < timedelta(0) or (name.endswith('window') or name == 'step') and value == timedelta(0):
                raise ValueError('invalid duration')
        if self.first_origin >= self.holdout_start:
            raise ValueError('origin must precede holdout')

    def split(self, rows):
        rows = tuple(sorted(rows, key=lambda r: (r.issue_time, r.row_id)))
        if len({r.row_id for r in rows}) != len(rows):
            raise ValueError('duplicate sample IDs')
        if any(r.partition != 'train' for r in rows):
            raise ValueError('split input must be unassigned train samples')
        folds = []
        origin = self.first_origin
        while origin + self.validation_window + self.embargo + self.calibration_window <= self.holdout_start:
            train_end = origin - self.purge
            cutoff = train_end - self.label_delay
            val_end = origin + self.validation_window
            cal_start = val_end + self.embargo
            cal_end = cal_start + self.calibration_window
            def select(start, end, partition):
                return tuple(replace(r, partition=partition) for r in rows
                             if start <= r.issue_time < end and r.target_time < end
                             and r.target_time < self.holdout_start)
            train = tuple(r for r in select(train_end-self.train_window, train_end, 'train')
                          if r.label_known_at <= cutoff and r.feature_known_at <= r.issue_time)
            val = select(origin, val_end, 'validation')
            cal = select(cal_start, cal_end, 'calibration')
            spec = [origin.isoformat(), self.holdout_start.isoformat(),
                    *[str(getattr(self, n)) for n in ('train_window', 'validation_window', 'calibration_window', 'step', 'purge', 'embargo', 'label_delay')]]
            folds.append(Fold(identity(spec), cutoff, train, val, cal, self.holdout_start))
            # The next origin may expand history, but never train inside the
            # preceding calibration interval or its post-evaluation embargo.
            origin = max(origin + self.step, cal_end + self.embargo + self.purge + self.label_delay)
        return tuple(folds)

    def holdout(self, rows):
        """Separate access; never returned by split or consumed by OOF."""
        return tuple(replace(r, partition='holdout') for r in rows if r.issue_time >= self.holdout_start)
