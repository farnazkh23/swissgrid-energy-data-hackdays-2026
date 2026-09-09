"""Explicit lifecycle metadata. G2 deliberately exposes no promotion operation."""
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from .availability import nonempty, utc
from .manifest import validate_digest
from .feature_registry import identity


class ModelState(str, Enum):
    CANDIDATE = 'CANDIDATE'
    CHAMPION = 'CHAMPION'
    REJECTED = 'REJECTED'
    FROZEN = 'FROZEN'


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    version: str
    fit_cutoff: datetime
    feature_manifest_hash: str
    training_data_manifest_hash: str
    state: ModelState = ModelState.CANDIDATE

    def __post_init__(self):
        nonempty(self.model_id, 'model_id')
        nonempty(self.version, 'version')
        object.__setattr__(self, 'fit_cutoff', utc(self.fit_cutoff))
        object.__setattr__(self, 'state', ModelState(self.state))
        validate_digest(self.feature_manifest_hash)
        validate_digest(self.training_data_manifest_hash)

    @property
    def manifest_hash(self):
        return identity([self.model_id, self.version, self.fit_cutoff.isoformat(),
                         self.feature_manifest_hash, self.training_data_manifest_hash])


class ModelRegistry:
    def __init__(self):
        self._entries = {}

    @property
    def entries(self):
        return tuple(self._entries[k] for k in sorted(self._entries))

    def register(self, entry):
        if entry.model_id in self._entries:
            raise ValueError('duplicate model ID, including alternate versions')
        if entry.state != ModelState.CANDIDATE:
            raise ValueError('new entries must be candidates')
        self._entries[entry.model_id] = entry

    def transition(self, model_id, state):
        state = ModelState(state)
        entry = self._entries[model_id]
        if state == ModelState.CHAMPION:
            raise NotImplementedError('Champion promotion is not enabled in G2')
        if entry.state != ModelState.CANDIDATE or state not in (ModelState.REJECTED, ModelState.FROZEN):
            raise ValueError('invalid state transition')
        self._entries[model_id] = replace(entry, state=state)
