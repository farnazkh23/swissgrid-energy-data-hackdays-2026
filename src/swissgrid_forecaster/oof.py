"""Rolling-origin execution and immutable, revalidatable OOF evidence."""
from dataclasses import dataclass, replace
from datetime import datetime
from .availability import nonempty, utc
from .manifest import validate_digest
from .model_contracts import FitContext, PredictContext, Prediction
from .splits import Fold, RollingOrigin, data_manifest


@dataclass(frozen=True)
class OOFRecord:
    fold_id: str
    row_id: str
    issue_time: datetime
    target_time: datetime
    model_id: str
    version: str
    feature_manifest_hash: str
    training_cutoff: datetime
    training_data_manifest_hash: str
    prediction: Prediction

    def __post_init__(self):
        for name in ('fold_id', 'row_id', 'model_id', 'version'):
            nonempty(getattr(self, name), name)
        for name in ('issue_time', 'target_time', 'training_cutoff'):
            object.__setattr__(self, name, utc(getattr(self, name)))
        for name in ('feature_manifest_hash', 'training_data_manifest_hash'):
            validate_digest(getattr(self, name))
        if not self.training_cutoff <= self.issue_time < self.target_time:
            raise ValueError('OOF chronology violation')
        if not isinstance(self.prediction, Prediction):
            raise ValueError('typed prediction required')

    @property
    def prediction_type(self):
        return self.prediction.prediction_type

    @property
    def prediction_values(self):
        return self.prediction.values


@dataclass(frozen=True)
class OOFResult:
    plan: RollingOrigin
    folds: tuple[Fold, ...]
    records: tuple[OOFRecord, ...]
    feature_manifest_hash: str
    candidate_identities: tuple[tuple[str, str], ...]

    def __post_init__(self):
        object.__setattr__(self, 'folds', tuple(self.folds))
        object.__setattr__(self, 'records', tuple(self.records))
        object.__setattr__(self, 'candidate_identities', tuple(tuple(k) for k in self.candidate_identities))
        self.validate()

    def validate(self):
        validate_digest(self.feature_manifest_hash)
        if not self.folds or not self.records:
            raise ValueError('empty OOF evidence')
        if not self.candidate_identities or any(len(k) != 2 for k in self.candidate_identities):
            raise ValueError('candidate identities required')
        if len({k[0] for k in self.candidate_identities}) != len(self.candidate_identities):
            raise ValueError('duplicate candidate IDs')
        source = {}
        for fold in self.folds:
            for row in (*fold.train, *fold.validation, *fold.calibration):
                normalized = replace(row, partition='train')
                if row.row_id in source and source[row.row_id] != normalized:
                    raise ValueError('inconsistent sample across folds')
                source[row.row_id] = normalized
        if self.plan.split(source.values()) != self.folds:
            raise ValueError('folds do not match rolling-origin plan')
        folds = {f.fold_id: f for f in self.folds}
        seen = set()
        versions = {}
        for record in self.records:
            if (record.model_id, record.version) not in self.candidate_identities:
                raise ValueError('unregistered OOF candidate')
            key = (record.fold_id, record.row_id, record.model_id, record.version)
            if key in seen:
                raise ValueError('duplicate OOF prediction')
            seen.add(key)
            previous = versions.setdefault(record.model_id, record.version)
            if previous != record.version:
                raise ValueError('model version changed across folds')
            if record.fold_id not in folds:
                raise ValueError('unknown OOF fold')
            fold = folds[record.fold_id]
            rows = {r.row_id: r for r in fold.validation}
            if record.row_id not in rows:
                raise ValueError('OOF must predict validation, never train/calibration/holdout')
            row = rows[record.row_id]
            if (record.issue_time, record.target_time) != (row.issue_time, row.target_time):
                raise ValueError('OOF sample timestamp mismatch')
            if record.training_cutoff != fold.fit_cutoff or record.training_data_manifest_hash != data_manifest(fold.train):
                raise ValueError('OOF training provenance mismatch')
            if record.feature_manifest_hash != self.feature_manifest_hash:
                raise ValueError('OOF feature manifest mismatch')
            FitContext(fold.train, fold.fit_cutoff, self.feature_manifest_hash, data_manifest(fold.train))
            PredictContext((row,), self.feature_manifest_hash)

    def truth(self, fold_id, row_id):
        for fold in self.folds:
            if fold.fold_id == fold_id:
                for row in fold.validation:
                    if row.row_id == row_id:
                        if row.target is None:
                            raise ValueError('OOF truth missing')
                        return row.target
        raise ValueError('unknown validation sample')


def run_oof(plan, rows, model_factories, feature_manifest_hash):
    """Fresh estimator per fold. Calibration and holdout never reach a model."""
    folds = plan.split(rows)
    factories = tuple(model_factories)
    if not factories:
        raise ValueError('no candidates')
    records, instances = [], []
    identities = None
    for fold in folds:
        if not fold.train or not fold.validation:
            raise ValueError('empty train/validation fold')
        current = []
        for factory in factories:
            model = factory()
            if any(model is old for old in instances) or model.fit_cutoff is not None:
                raise ValueError('each fold requires a fresh unfitted model')
            instances.append(model)
            key = (model.model_id, model.version)
            if any(k[0] == key[0] for k in current):
                raise ValueError('duplicate model ID')
            current.append(key)
            context = FitContext(fold.train, fold.fit_cutoff, feature_manifest_hash, data_manifest(fold.train))
            model.fit(tuple(r.features for r in fold.train), tuple(r.target for r in fold.train), context)
            if (model.model_id, model.version) != key or (model.fit_cutoff, model.feature_manifest_hash, model.training_data_manifest_hash) != (context.fit_cutoff, context.feature_manifest_hash, context.training_data_manifest_hash):
                raise ValueError('model fit provenance mismatch')
            # Do not expose validation truth or release times to predict().
            hidden = tuple(replace(r, target=None, label_known_at=r.target_time) for r in fold.validation)
            for row in hidden:
                # A batch spanning issue times could let an estimator consume
                # later features while predicting an earlier issue.
                predictions = tuple(model.predict((row.features,), PredictContext((row,), feature_manifest_hash)))
                if len(predictions) != 1:
                    raise ValueError('prediction count mismatch')
                prediction = predictions[0]
                records.append(OOFRecord(fold.fold_id, row.row_id, row.issue_time, row.target_time,
                                         *key, feature_manifest_hash, fold.fit_cutoff,
                                         context.training_data_manifest_hash, prediction))
        if identities is not None and identities != current:
            raise ValueError('candidate identities changed across folds')
        identities = current
    return OOFResult(plan, folds, tuple(records), feature_manifest_hash, tuple(identities or ()))
