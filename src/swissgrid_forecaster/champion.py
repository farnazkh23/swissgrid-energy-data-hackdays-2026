"""Initial candidates and rolling-OOF-only recommendation. No promotion."""
from datetime import timedelta
from statistics import mean
from .feature_registry import identity
from .model_contracts import GuardedModel, Prediction
from . import metrics


class Persistence(GuardedModel):
    """Latest training label; optional season requires an exact elapsed-time match."""
    def __init__(self, model_id='persistence', *, season=None):
        if season is not None and (not isinstance(season, timedelta) or season <= timedelta(0)):
            raise ValueError('positive season required')
        super().__init__(model_id, identity(['persistence', '1', str(season)]))
        self.season = season

    def fit(self, train_X, train_y, context):
        self._check_fit(train_X, train_y, context)
        if len({r.target_time for r in context.rows}) != len(context.rows):
            raise ValueError('persistence needs a single resolved target series')
        self.history = {r.target_time: r.target for r in context.rows}
        self._commit_fit(context)
        return self

    def predict(self, X, context):
        self._check_predict(X, context)
        result = []
        for row in context.rows:
            key = row.target_time-self.season if self.season else max(self.history)
            if key not in self.history:
                raise ValueError('seasonal history unavailable; no silent fallback')
            result.append(Prediction('point', (self.history[key],)))
        return tuple(result)


class SeasonalPersistence(Persistence):
    def __init__(self, season, model_id='seasonal_persistence'):
        super().__init__(model_id, season=season)


class HistoricalConditional(GuardedModel):
    """Equal-weight empirical samples, conditioned on explicit feature columns.

    Empty conditioning columns give an unconditional empirical distribution.
    Unseen groups fail; missing values are an explicit group value.
    """
    def __init__(self, condition_columns=(), model_id='historical_conditional'):
        columns = tuple(condition_columns)
        if len(set(columns)) != len(columns) or any(type(c) is not int or c < 0 for c in columns):
            raise ValueError('invalid condition columns')
        super().__init__(model_id, identity(['historical_conditional', '1', columns]))
        self.columns = columns

    def fit(self, train_X, train_y, context):
        X, y = self._check_fit(train_X, train_y, context)
        if any(c >= len(X[0]) for c in self.columns):
            raise ValueError('condition column out of range')
        self.groups = {}
        for row, value in zip(X, y):
            self.groups.setdefault(tuple(row[c] for c in self.columns), []).append(value)
        self._commit_fit(context)
        return self

    def predict(self, X, context):
        X = self._check_predict(X, context)
        result = []
        for row in X:
            key = tuple(row[c] for c in self.columns)
            if key not in self.groups:
                raise ValueError('unseen conditional group')
            result.append(Prediction('distribution', tuple(sorted(self.groups[key]))))
        return tuple(result)


class OptionalEstimator(GuardedModel):
    """Lazy sklearn adapter. Never installs dependencies; no normalization."""
    def __init__(self, kind, *, alpha=1.0, quantile=0.5, model_id=None):
        if kind not in ('ridge', 'quantile_boosting'):
            raise ValueError('unknown estimator')
        if not 0 < quantile < 1 or not 0 <= alpha < float('inf'):
            raise ValueError('invalid estimator parameters')
        super().__init__(model_id or kind, identity([kind, '1', alpha, quantile]))
        self.kind, self.alpha, self.quantile = kind, alpha, quantile

    def fit(self, train_X, train_y, context):
        X, y = self._check_fit(train_X, train_y, context)
        if not X[0] or any(v is None for row in X for v in row):
            raise ValueError('estimator requires explicit complete features')
        try:
            if self.kind == 'ridge':
                from sklearn.linear_model import Ridge
                estimator = Ridge(alpha=self.alpha)
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                estimator = GradientBoostingRegressor(loss='quantile', alpha=self.quantile, random_state=0)
        except ImportError as exc:
            raise RuntimeError('optional scikit-learn dependency unavailable; install explicitly') from exc
        estimator.fit(X, y)
        self.estimator = estimator
        self._commit_fit(context)
        return self

    def predict(self, X, context):
        X = self._check_predict(X, context)
        if any(v is None for row in X for v in row):
            raise ValueError('missing estimator input')
        values = self.estimator.predict(X)
        return tuple(Prediction('point', (float(v),)) if self.kind == 'ridge' else
                     Prediction('quantile', (float(v),), (self.quantile,)) for v in values)


def select_champion(evidence, *, metric, quantile=None):
    """Return a recommendation from complete, matched rolling OOF cases only.

    Pooled observations have equal weight. Ties break by model ID/version.
    Distributions map to their empirical mean only for point metrics.
    """
    from .oof import OOFResult
    if not isinstance(evidence, OOFResult):
        raise ValueError('validated rolling OOF evidence required')
    evidence.validate()
    if metric not in ('mae', 'rmse', 'pinball', 'brier'):
        raise ValueError('choose an implemented proper minimization objective')
    groups = {key: [] for key in evidence.candidate_identities}
    for record in evidence.records:
        groups.setdefault((record.model_id, record.version), []).append(record)
    expected = {(f.fold_id, r.row_id) for f in evidence.folds for r in f.validation}
    scores = {}
    for key, records in groups.items():
        if {(r.fold_id, r.row_id) for r in records} != expected:
            raise ValueError('candidates must cover identical complete OOF cases')
        truth, prediction = [], []
        for r in records:
            p = r.prediction
            truth.append(evidence.truth(r.fold_id, r.row_id))
            if metric == 'pinball':
                if p.prediction_type != 'quantile' or quantile not in p.quantile_levels:
                    raise ValueError('requested quantile unavailable')
                prediction.append(p.values[p.quantile_levels.index(quantile)])
            elif metric == 'brier':
                if p.prediction_type != 'probability':
                    raise ValueError('Brier requires probability predictions')
                prediction.append(p.values[0])
            else:
                if p.prediction_type not in ('point', 'distribution'):
                    raise ValueError('point objective requires point or empirical distribution')
                prediction.append(mean(p.values))
        scores[key] = getattr(metrics, metric)(truth, prediction, quantile) if metric == 'pinball' else getattr(metrics, metric)(truth, prediction)
    if not scores:
        raise ValueError('no OOF candidates')
    return min(scores, key=lambda k: (scores[k], k)), scores
