"""Estimator-neutral contracts and shared fail-closed fit/predict guards."""
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Protocol, Sequence
from .availability import nonempty, utc
from .manifest import validate_digest
from .splits import Sample, data_manifest


@dataclass(frozen=True)
class FitContext:
    rows: tuple[Sample, ...]
    fit_cutoff: datetime
    feature_manifest_hash: str
    training_data_manifest_hash: str

    def __post_init__(self):
        object.__setattr__(self, 'rows', tuple(self.rows))
        object.__setattr__(self, 'fit_cutoff', utc(self.fit_cutoff))
        validate_digest(self.feature_manifest_hash)
        validate_digest(self.training_data_manifest_hash)
        if not self.rows or data_manifest(self.rows) != self.training_data_manifest_hash:
            raise ValueError('training manifest mismatch or empty training set')
        if any(r.partition != 'train' or r.label_known_at > self.fit_cutoff
               or r.target_time > self.fit_cutoff or r.issue_time > self.fit_cutoff
               or r.feature_known_at > r.issue_time or r.target is None for r in self.rows):
            raise ValueError('training partition, availability or cutoff violation')


@dataclass(frozen=True)
class PredictContext:
    rows: tuple[Sample, ...]
    feature_manifest_hash: str

    def __post_init__(self):
        object.__setattr__(self, 'rows', tuple(self.rows))
        validate_digest(self.feature_manifest_hash)
        if len({r.row_id for r in self.rows}) != len(self.rows):
            raise ValueError('duplicate prediction rows')
        if any(r.feature_known_at > r.issue_time for r in self.rows):
            raise ValueError('future feature knowledge')


@dataclass(frozen=True)
class Prediction:
    prediction_type: str
    values: tuple[float, ...]
    quantile_levels: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'values', tuple(self.values))
        object.__setattr__(self, 'quantile_levels', tuple(self.quantile_levels))
        if self.prediction_type not in ('point', 'quantile', 'distribution', 'probability'):
            raise ValueError('unknown prediction type')
        if not self.values or any(not isfinite(v) for v in self.values):
            raise ValueError('empty/nonfinite predictions')
        if self.prediction_type in ('point', 'probability') and len(self.values) != 1:
            raise ValueError('scalar prediction required')
        if self.prediction_type == 'probability' and not 0 <= self.values[0] <= 1:
            raise ValueError('invalid probability')
        if self.prediction_type == 'quantile':
            q = self.quantile_levels
            if len(q) != len(self.values) or any(not isfinite(v) or not 0 < v < 1 for v in q) or any(a >= b for a, b in zip(q, q[1:])):
                raise ValueError('invalid quantile levels')
            if any(a > b for a, b in zip(self.values, self.values[1:])):
                raise ValueError('crossed quantiles')
        elif self.quantile_levels:
            raise ValueError('levels only apply to quantiles')


class Model(Protocol):
    model_id: str
    version: str
    fit_cutoff: datetime | None
    feature_manifest_hash: str | None
    training_data_manifest_hash: str | None
    def fit(self, train_X, train_y, context: FitContext) -> 'Model': ...
    def predict(self, X, context: PredictContext) -> Sequence[Prediction]: ...


class GuardedModel:
    def __init__(self, model_id: str, version: str):
        nonempty(model_id, 'model_id')
        nonempty(version, 'version')
        self.model_id, self.version = model_id, version
        self.fit_cutoff = self.feature_manifest_hash = self.training_data_manifest_hash = None

    def _check_fit(self, X, y, context):
        # Invalidate a previous fit even when a refit fails.
        self.fit_cutoff = self.feature_manifest_hash = self.training_data_manifest_hash = None
        X, y = tuple(tuple(row) for row in X), tuple(y)
        if X != tuple(r.features for r in context.rows) or y != tuple(r.target for r in context.rows):
            raise ValueError('fit arrays do not match manifested training rows')
        if len({len(row) for row in X}) != 1:
            raise ValueError('inconsistent feature width')
        return X, y

    def _commit_fit(self, context):
        self.fit_cutoff = context.fit_cutoff
        self.feature_manifest_hash = context.feature_manifest_hash
        self.training_data_manifest_hash = context.training_data_manifest_hash
        self._width = len(context.rows[0].features)

    def _check_predict(self, X, context):
        X = tuple(tuple(row) for row in X)
        if self.fit_cutoff is None or context.feature_manifest_hash != self.feature_manifest_hash:
            raise ValueError('unfitted model or feature manifest mismatch')
        if any(r.issue_time < self.fit_cutoff for r in context.rows):
            raise ValueError('model fitted after prediction issue')
        if X != tuple(r.features for r in context.rows) or any(len(row) != self._width for row in X):
            raise ValueError('prediction arrays differ from context or fitted width')
        return X
