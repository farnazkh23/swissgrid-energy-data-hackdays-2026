"""OOF-only residual evidence, joint uncertainty models, and deterministic sampling.

Scope: this module owns Y = mu + epsilon uncertainty and cross-target dependency
for the final probabilistic submission layer (independent/correlated Gaussian,
horizon- or hour-of-day-conditioned covariance, empirical bootstrap, Student-t).
It does not fit point forecasts (Champion/specialist layer) and does not decide
which method is official (see `sampler_evaluation.evaluate_sampler`). All
covariance, bias, and calibration parameters must come from out-of-fold
residuals; nothing here reads training-fold or in-sample values.
"""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from math import isfinite, sqrt
from random import Random
from statistics import NormalDist, correlation, pstdev

from .availability import nonempty, utc
from .feature_registry import identity
from .histogram import empirical_quantile
from .manifest import validate_digest

METHODS = (
    "independent_gaussian",
    "correlated_gaussian",
    "horizon_covariance",
    "hour_of_day_covariance",
    "empirical_bootstrap",
    "student_t",
)


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_rows(vectors, minimum: int = 2) -> None:
    if len(vectors) < minimum:
        raise ValueError(f"bucket has fewer than {minimum} OOF residual rows")


def bucket_key_for(method: str, row):
    if method == "horizon_covariance":
        return row[2]
    if method == "hour_of_day_covariance":
        return row[1].hour
    return None


def _mean_vector(vectors) -> tuple[float, ...]:
    width = len(vectors[0])
    return tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(width))


def _covariance_matrix(vectors, mean=None) -> tuple[tuple[float, ...], ...]:
    _require_rows(vectors)
    width = len(vectors[0])
    mean = mean or _mean_vector(vectors)
    cov = [[0.0] * width for _ in range(width)]
    for v in vectors:
        for i in range(width):
            di = v[i] - mean[i]
            for j in range(width):
                cov[i][j] += di * (v[j] - mean[j])
    denominator = len(vectors) - 1
    return tuple(tuple(cov[i][j] / denominator for j in range(width)) for i in range(width))


def _diagonal_only(cov) -> tuple[tuple[float, ...], ...]:
    width = len(cov)
    return tuple(tuple(cov[i][j] if i == j else 0.0 for j in range(width)) for i in range(width))


def _cholesky(cov) -> tuple[tuple[float, ...], ...]:
    """Lower-triangular factor; fails closed if covariance is not positive semi-definite."""
    n = len(cov)
    factor = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(factor[i][k] * factor[j][k] for k in range(j))
            if i == j:
                value = cov[i][i] - total
                if value < -1e-9:
                    raise ValueError("covariance matrix is not positive semi-definite")
                factor[i][j] = sqrt(max(value, 0.0))
            else:
                factor[i][j] = (cov[i][j] - total) / factor[j][j] if factor[j][j] > 1e-12 else 0.0
    return tuple(tuple(row) for row in factor)


def _draw_joint(rng: Random, chol, dof: float | None = None) -> tuple[float, ...]:
    width = len(chol)
    z = [rng.gauss(0.0, 1.0) for _ in range(width)]
    if dof is not None:
        chi2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(max(int(dof), 1)))
        factor = sqrt(dof / chi2) if chi2 > 0 else 1.0
        z = [value * factor for value in z]
    return tuple(sum(chol[i][k] * z[k] for k in range(i + 1)) for i in range(width))


@dataclass(frozen=True, slots=True)
class ResidualObservation:
    """One OOF residual (truth - point forecast) for one target/timestamp/fold."""

    fold_id: str
    row_id: str
    target_name: str
    issue_time: datetime
    target_time: datetime
    horizon: timedelta
    residual: float

    def __post_init__(self) -> None:
        for name in ("fold_id", "row_id", "target_name"):
            nonempty(getattr(self, name), name)
        object.__setattr__(self, "issue_time", utc(self.issue_time, "issue_time"))
        object.__setattr__(self, "target_time", utc(self.target_time, "target_time"))
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        if self.target_time - self.issue_time != self.horizon:
            raise ValueError("horizon does not match issue/target times")
        object.__setattr__(self, "residual", _finite(self.residual, "residual"))

    @property
    def hour_of_day(self) -> int:
        return self.target_time.hour

    @classmethod
    def from_oof(cls, evidence, *, target_name: str) -> tuple["ResidualObservation", ...]:
        """Build residuals from a validated single-target OOFResult (point or empirical-mean)."""
        from .oof import OOFResult
        if not isinstance(evidence, OOFResult):
            raise ValueError("validated rolling OOF evidence required")
        evidence.validate()
        result = []
        for record in evidence.records:
            truth = evidence.truth(record.fold_id, record.row_id)
            prediction = record.prediction
            if prediction.prediction_type not in ("point", "distribution"):
                raise ValueError("residuals require point or empirical-distribution predictions")
            point = (prediction.values[0] if prediction.prediction_type == "point"
                     else sum(prediction.values) / len(prediction.values))
            result.append(cls(record.fold_id, record.row_id, target_name, record.issue_time,
                              record.target_time, record.target_time - record.issue_time, truth - point))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ResidualPanel:
    """Joint OOF residual vectors across targets, aligned by matching (issue_time, target_time)."""

    target_names: tuple[str, ...]
    rows: tuple[tuple[datetime, datetime, timedelta, tuple[float, ...]], ...]

    def __post_init__(self) -> None:
        names = tuple(self.target_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("target_names must be nonempty and unique")
        object.__setattr__(self, "target_names", names)
        cleaned, seen = [], set()
        for issue_time, target_time, horizon, residuals in self.rows:
            issue_time = utc(issue_time, "issue_time")
            target_time = utc(target_time, "target_time")
            if not isinstance(horizon, timedelta) or horizon <= timedelta(0) or target_time - issue_time != horizon:
                raise ValueError("invalid row chronology")
            residuals = tuple(_finite(v, "residual") for v in residuals)
            if len(residuals) != len(names):
                raise ValueError("residual vector must match target_names width")
            key = (issue_time, target_time)
            if key in seen:
                raise ValueError("duplicate joint timestamp")
            seen.add(key)
            cleaned.append((issue_time, target_time, horizon, residuals))
        if not cleaned:
            raise ValueError("panel requires at least one joint row")
        object.__setattr__(self, "rows", tuple(sorted(cleaned, key=lambda r: (r[0], r[1]))))

    @classmethod
    def from_target_residuals(cls, per_target: dict) -> "ResidualPanel":
        """Inner-join per-target OOF residuals on shared (issue_time, target_time) only."""
        names = tuple(sorted(per_target))
        if not names:
            raise ValueError("at least one target's residuals are required")
        indexed = {}
        for name in names:
            observations = tuple(per_target[name])
            if not observations:
                raise ValueError(f"no residuals supplied for target {name}")
            by_key = {}
            for observation in observations:
                if not isinstance(observation, ResidualObservation) or observation.target_name != name:
                    raise ValueError("residual/target mismatch")
                key = (observation.issue_time, observation.target_time)
                if key in by_key:
                    raise ValueError("duplicate residual for timestamp")
                by_key[key] = observation
            indexed[name] = by_key
        shared = set.intersection(*(set(by_key) for by_key in indexed.values()))
        if not shared:
            raise ValueError("no shared OOF timestamps across targets")
        rows = []
        for key in shared:
            horizon = indexed[names[0]][key].horizon
            if any(indexed[name][key].horizon != horizon for name in names):
                raise ValueError("horizon mismatch across targets at shared timestamp")
            rows.append((*key, horizon, tuple(indexed[name][key].residual for name in names)))
        return cls(names, tuple(rows))

    def matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(row[3] for row in self.rows)

    def identity_hash(self) -> str:
        return identity([list(self.target_names),
                        [[r[0].isoformat(), r[1].isoformat(), str(r[2]), list(r[3])] for r in self.rows]])


@dataclass(frozen=True, slots=True)
class UncertaintyModel:
    """Fitted bias/covariance (or empirical pool) per bucket key, plus optional calibration scale."""

    target_names: tuple[str, ...]
    method: str
    fit_cutoff: datetime
    residual_manifest_hash: str
    bias: tuple[tuple[object, tuple[float, ...]], ...]
    covariance: tuple[tuple[object, tuple[tuple[float, ...], ...]], ...]
    empirical_residuals: tuple[tuple[object, tuple[tuple[float, ...], ...]], ...] = ()
    degrees_of_freedom: float | None = None
    calibration_scale: tuple[tuple[object, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_names", tuple(self.target_names))
        if not self.target_names or len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must be nonempty and unique")
        if self.method not in METHODS:
            raise ValueError("unknown uncertainty method")
        object.__setattr__(self, "fit_cutoff", utc(self.fit_cutoff, "fit_cutoff"))
        validate_digest(self.residual_manifest_hash)
        object.__setattr__(self, "bias", tuple(self.bias))
        object.__setattr__(self, "covariance", tuple(self.covariance))
        object.__setattr__(self, "empirical_residuals", tuple(self.empirical_residuals))
        object.__setattr__(self, "calibration_scale", tuple(self.calibration_scale))
        bias_keys = tuple(key for key, _ in self.bias)
        if not bias_keys or len(set(bias_keys)) != len(bias_keys):
            raise ValueError("bias buckets must be nonempty and unique")
        if tuple(key for key, _ in self.covariance) != bias_keys:
            raise ValueError("covariance buckets must match bias buckets")
        width = len(self.target_names)
        for _, vector in self.bias:
            if len(vector) != width or any(not isfinite(v) for v in vector):
                raise ValueError("bias vector must be finite and match target width")
        for _, matrix in self.covariance:
            if len(matrix) != width or any(len(row) != width for row in matrix):
                raise ValueError("covariance matrix must be square and match target width")
            if any(not isfinite(v) for row in matrix for v in row):
                raise ValueError("covariance matrix must be finite")
            if any(abs(matrix[i][j] - matrix[j][i]) > 1e-9 for i in range(width) for j in range(width)):
                raise ValueError("covariance matrix must be symmetric")
            _cholesky(matrix)
        if self.method == "empirical_bootstrap":
            if tuple(key for key, _ in self.empirical_residuals) != bias_keys:
                raise ValueError("empirical residual buckets must match bias buckets")
            for _, vectors in self.empirical_residuals:
                if len(vectors) < 2 or any(len(v) != width for v in vectors):
                    raise ValueError("empirical residual pool must contain matching-width vectors")
        elif self.empirical_residuals:
            raise ValueError("empirical_residuals only applies to empirical_bootstrap")
        if self.method == "student_t":
            if self.degrees_of_freedom is None or not isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 2:
                raise ValueError("student_t requires finite degrees_of_freedom > 2")
        elif self.degrees_of_freedom is not None:
            raise ValueError("degrees_of_freedom only applies to student_t")
        for key, scale in self.calibration_scale:
            if key not in bias_keys:
                raise ValueError("calibration scale bucket must exist in the fitted model")
            if not isfinite(scale) or scale <= 0:
                raise ValueError("calibration scale must be positive and finite")

    def _lookup(self, table, key):
        mapping = dict(table)
        if key not in mapping:
            raise ValueError("unseen bucket key; this model was not fit for it")
        return mapping[key]

    def bias_for(self, key=None) -> tuple[float, ...]:
        return self._lookup(self.bias, key)

    def covariance_for(self, key=None) -> tuple[tuple[float, ...], ...]:
        return self._lookup(self.covariance, key)

    def empirical_for(self, key=None) -> tuple[tuple[float, ...], ...]:
        if self.method != "empirical_bootstrap":
            raise ValueError("empirical residuals only apply to empirical_bootstrap")
        return self._lookup(self.empirical_residuals, key)

    def scale_for(self, key=None) -> float:
        table = dict(self.calibration_scale)
        if not table:
            return 1.0
        if key not in table:
            raise ValueError("uncalibrated bucket key")
        return table[key]

    def with_calibration_scale(self, scale_by_bucket: dict) -> "UncertaintyModel":
        return replace(self, calibration_scale=tuple(scale_by_bucket.items()))


def fit_uncertainty_model(panel: ResidualPanel, *, method: str, fit_cutoff: datetime,
                          degrees_of_freedom: float | None = None) -> UncertaintyModel:
    """Fit bias/covariance (or empirical pool) per bucket from OOF residuals issued by fit_cutoff."""
    if not isinstance(panel, ResidualPanel):
        raise ValueError("ResidualPanel required")
    if method not in METHODS:
        raise ValueError("unknown uncertainty method")
    fit_cutoff = utc(fit_cutoff, "fit_cutoff")
    if any(row[0] > fit_cutoff for row in panel.rows):
        raise ValueError("residual panel contains OOF rows issued after fit_cutoff")
    if method == "student_t":
        if degrees_of_freedom is None or not isfinite(degrees_of_freedom) or degrees_of_freedom <= 2:
            raise ValueError("student_t requires finite degrees_of_freedom > 2")
    elif degrees_of_freedom is not None:
        raise ValueError("degrees_of_freedom only applies to student_t")
    buckets: dict = {}
    for row in panel.rows:
        buckets.setdefault(bucket_key_for(method, row), []).append(row[3])
    bias, covariance, empirical = [], [], []
    for key, vectors in buckets.items():
        mean = _mean_vector(vectors)
        cov = _covariance_matrix(vectors, mean)
        if method == "independent_gaussian":
            cov = _diagonal_only(cov)
        bias.append((key, mean))
        covariance.append((key, cov))
        if method == "empirical_bootstrap":
            empirical.append((key, tuple(vectors)))
    return UncertaintyModel(panel.target_names, method, fit_cutoff, panel.identity_hash(),
                            tuple(bias), tuple(covariance), tuple(empirical),
                            float(degrees_of_freedom) if method == "student_t" and degrees_of_freedom is not None else None)


def sample_distribution(model: UncertaintyModel, means: dict, *, bucket_key=None, seed: int,
                        n_samples: int = 300, round_to_int: bool = True) -> dict:
    """Draw n_samples joint vectors deterministically from a fitted model, added to `means`."""
    if not isinstance(model, UncertaintyModel):
        raise ValueError("fitted UncertaintyModel required")
    names = model.target_names
    if set(means) != set(names):
        raise ValueError("means must supply exactly the fitted target names")
    mean_vector = tuple(_finite(means[name], name) for name in names)
    if type(n_samples) is not int or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    rng = Random(seed)
    scale = sqrt(model.scale_for(bucket_key))
    if model.method == "empirical_bootstrap":
        pool = model.empirical_for(bucket_key)
        draws = [pool[rng.randrange(len(pool))] for _ in range(n_samples)]
    else:
        chol = _cholesky(model.covariance_for(bucket_key))
        bias = model.bias_for(bucket_key)
        dof = model.degrees_of_freedom if model.method == "student_t" else None
        draws = [tuple(b + scale * noise for b, noise in zip(bias, _draw_joint(rng, chol, dof)))
                 for _ in range(n_samples)]
    result = {}
    for index, name in enumerate(names):
        values = tuple(mean_vector[index] + draw[index] for draw in draws)
        if any(not isfinite(v) for v in values):
            raise ValueError("nonfinite sample generated")
        result[name] = tuple(int(round(v)) for v in values) if round_to_int else values
    return result


def calibrate_scale(model: UncertaintyModel, calibration_panel: ResidualPanel, *,
                    target_coverage: float = 0.5, seed: int = 0, n_samples: int = 300) -> UncertaintyModel:
    """Fit a per-bucket variance-scale multiplier on a distinct calibration-fold panel.

    Never reuse the rows used to fit `model` or later evaluation rows for this search.
    """
    if not isinstance(model, UncertaintyModel):
        raise ValueError("fitted UncertaintyModel required")
    if not isinstance(calibration_panel, ResidualPanel) or calibration_panel.target_names != model.target_names:
        raise ValueError("calibration panel must match the fitted target names")
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be inside (0, 1)")
    buckets: dict = {}
    for row in calibration_panel.rows:
        buckets.setdefault(bucket_key_for(model.method, row), []).append(row)
    scale_by_bucket = {}
    for key, rows in buckets.items():
        model.covariance_for(key)
        scale_by_bucket[key] = _search_scale(model, key, rows, target_coverage, seed, n_samples)
    return model.with_calibration_scale(scale_by_bucket)


def _empirical_coverage(model, key, rows, scale, target_coverage, seed, n_samples) -> float:
    lower_q, upper_q = (1 - target_coverage) / 2, 1 - (1 - target_coverage) / 2
    hits = total = 0
    trial = model.with_calibration_scale({key: scale})
    for index, row in enumerate(rows):
        means = {name: 0.0 for name in model.target_names}
        draws = sample_distribution(trial, means, bucket_key=key, seed=seed + index,
                                    n_samples=n_samples, round_to_int=False)
        for target_index in range(len(model.target_names)):
            values = sorted(draws[model.target_names[target_index]])
            lo, hi = empirical_quantile(values, lower_q), empirical_quantile(values, upper_q)
            hits += lo <= row[3][target_index] <= hi
            total += 1
    return hits / total


def _search_scale(model, key, rows, target_coverage, seed, n_samples,
                  low: float = 0.1, high: float = 5.0, iterations: int = 25) -> float:
    """Bisection on empirical coverage, which increases monotonically with scale."""
    for _ in range(iterations):
        mid = (low + high) / 2
        coverage = _empirical_coverage(model, key, rows, mid, target_coverage, seed, n_samples)
        if coverage < target_coverage:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def qq_diagnostic(values, *, points: int = 19) -> tuple[tuple[float, float], ...]:
    """Theoretical-vs-empirical normal quantile pairs for a residual sample."""
    values = tuple(sorted(_finite(v, "value") for v in values))
    _require_rows(values)
    normal = NormalDist()
    probabilities = tuple((i + 1) / (points + 1) for i in range(points))
    return tuple((normal.inv_cdf(p), empirical_quantile(values, p)) for p in probabilities)


def correlation_matrix(panel: ResidualPanel) -> tuple[tuple[float, ...], ...]:
    """Pearson correlation between each pair of targets' OOF residuals."""
    vectors = panel.matrix()
    _require_rows(vectors)
    width = len(panel.target_names)
    columns = [[v[i] for v in vectors] for i in range(width)]
    return tuple(tuple(1.0 if i == j else correlation(columns[i], columns[j]) for j in range(width))
                for i in range(width))


def residual_scale_by_bucket(panel: ResidualPanel, method: str) -> tuple[tuple[object, tuple[float, ...]], ...]:
    """Per-bucket residual standard deviation per target; diagnostic only, not used for sampling."""
    buckets: dict = {}
    for row in panel.rows:
        buckets.setdefault(bucket_key_for(method, row), []).append(row[3])
    result = []
    for key, vectors in buckets.items():
        _require_rows(vectors)
        width = len(vectors[0])
        result.append((key, tuple(pstdev(v[i] for v in vectors) for i in range(width))))
    return tuple(result)


def covariance_stability(panel: ResidualPanel, *, fold_size: int) -> float:
    """Max relative Frobenius deviation of fold-local covariance from the full-panel covariance."""
    vectors = panel.matrix()
    if fold_size <= 1 or fold_size >= len(vectors):
        raise ValueError("fold_size must be smaller than the full panel and greater than one")
    overall = _covariance_matrix(vectors)
    overall_norm = sqrt(sum(v * v for row in overall for v in row)) or 1.0
    worst = 0.0
    for start in range(0, len(vectors) - fold_size + 1, fold_size):
        chunk = vectors[start:start + fold_size]
        if len(chunk) < 2:
            continue
        cov = _covariance_matrix(chunk)
        deviation = sqrt(sum((cov[i][j] - overall[i][j]) ** 2 for i in range(len(cov)) for j in range(len(cov))))
        worst = max(worst, deviation / overall_norm)
    return worst
