"""Immutable feature DAG definitions, separate from evaluation evidence."""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
from .availability import nonempty, utc


def identity(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    feature_family: str
    feature_name: str
    country_scope: tuple[str, ...]
    source: str
    known_at_rule: str
    horizon_available: tuple[timedelta, ...]
    transform_name: str
    transform_version: str
    parent_features: tuple[str, ...]
    units: str
    leakage_risk: str
    missingness_policy: str
    enabled: bool = True

    def __post_init__(self):
        for name in ('feature_id', 'feature_family', 'feature_name', 'source', 'known_at_rule',
                     'transform_name', 'transform_version', 'units', 'leakage_risk'):
            nonempty(getattr(self, name), name)
        for name in ('country_scope', 'parent_features', 'horizon_available'):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f'duplicate {name}')
            object.__setattr__(self, name, values)
        for value in self.country_scope + self.parent_features:
            nonempty(value, 'scope/parent')
        if not self.horizon_available or any(not isinstance(h, timedelta) or h <= timedelta(0) for h in self.horizon_available):
            raise ValueError('positive supported horizons required')
        if self.missingness_policy not in ('error', 'propagate'):
            raise ValueError('missingness_policy must be error or propagate')
        if type(self.enabled) is not bool:
            raise ValueError('enabled must be bool')

    @property
    def version(self) -> str:
        data = asdict(self)
        data['horizon_available'] = [str(h) for h in self.horizon_available]
        return identity(data)


@dataclass(frozen=True)
class FeatureAudit:
    feature_version: str
    evaluation_id: str
    pearson: float | None = None
    spearman: float | None = None
    mutual_information: float | None = None
    univariate_score: float | None = None
    oof_gain: float | None = None
    stability: float | None = None
    keep_drop: str = 'UNDECIDED'

    def __post_init__(self):
        nonempty(self.feature_version, 'feature_version')
        nonempty(self.evaluation_id, 'evaluation_id')
        if self.keep_drop not in ('KEEP', 'DROP', 'UNDECIDED'):
            raise ValueError('invalid keep_drop')
        for name in ('pearson', 'spearman', 'mutual_information', 'univariate_score', 'oof_gain', 'stability'):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError('nonfinite audit')


@dataclass(frozen=True)
class FeatureRegistry:
    definitions: tuple[FeatureDefinition, ...]

    def __post_init__(self):
        definitions = tuple(sorted(self.definitions, key=lambda f: f.feature_id))
        object.__setattr__(self, 'definitions', definitions)
        index = {f.feature_id: f for f in definitions}
        if len(index) != len(definitions):
            raise ValueError('duplicate feature ID')
        visiting, visited = set(), set()
        def visit(key):
            if key not in index:
                raise ValueError(f'unknown parent {key}')
            if key in visiting:
                raise ValueError('feature dependency cycle')
            if key in visited:
                return
            visiting.add(key)
            for parent in index[key].parent_features:
                visit(parent)
            visiting.remove(key)
            visited.add(key)
        for key in index:
            visit(key)

    @property
    def version(self) -> str:
        return identity([(f.feature_id, f.version) for f in self.definitions])

    def known_at(self, feature_id: str, own_known_at: dict[str, datetime]) -> datetime:
        index = {f.feature_id: f for f in self.definitions}
        def resolve(key):
            if key not in index or key not in own_known_at:
                raise ValueError(f'missing feature/knowledge {key}')
            return max([utc(own_known_at[key])] + [resolve(p) for p in index[key].parent_features])
        return resolve(feature_id)
