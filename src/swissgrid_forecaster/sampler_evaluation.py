"""Weekly rolling backtests, capture/sharpness diagnostics, and method comparison.

This module scores probabilistic samplers built on `uncertainty_model` but does
not implement the official edh2026.local_scoring function. Pass it through
`score_fn` once available; otherwise a documented placeholder loss is used and
callers should treat the resulting numbers as directional only, not official.
"""
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from statistics import mean, median, pstdev

from .ablation import KEEP_DROP
from .histogram import empirical_quantile
from .uncertainty_model import (
    METHODS, ResidualPanel, UncertaintyModel, bucket_key_for,
    fit_uncertainty_model, sample_distribution,
)


def weekly_folds(panel: ResidualPanel, *, week_hours: int = 168) -> tuple[tuple, ...]:
    """Consecutive, non-overlapping chunks of `week_hours` rows in chronological order."""
    if type(week_hours) is not int or week_hours <= 0:
        raise ValueError("week_hours must be a positive integer")
    rows = panel.rows
    if len(rows) < week_hours:
        raise ValueError("panel does not contain a full week of rows")
    return tuple(rows[start:start + week_hours] for start in range(0, len(rows) - week_hours + 1, week_hours))


def _default_score(capture: float, sharpness: float, mae: float, *, target_coverage: float = 0.9) -> float:
    """Placeholder composite loss until edh2026.local_scoring is wired in. Lower is better."""
    capture_penalty = max(0.0, target_coverage - capture) * 10.0
    return mae + sharpness * 0.5 + capture_penalty


@dataclass(frozen=True, slots=True)
class WeeklyScore:
    week_start: datetime
    score: float
    capture: float
    sharpness: float
    mae: float
    bias: float
    per_target: tuple[tuple[str, float], ...]
    per_horizon: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for name in ("score", "capture", "sharpness", "mae", "bias"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "per_target", tuple(self.per_target))
        object.__setattr__(self, "per_horizon", tuple(self.per_horizon))


@dataclass(frozen=True, slots=True)
class SamplerEvaluation:
    method: str
    weekly_scores: tuple[WeeklyScore, ...]
    mean_score: float
    median_score: float
    worst_week_score: float
    score_stdev: float
    capture_rate: float
    mean_sharpness: float
    mean_bias: float
    mean_mae: float
    keep_drop: str = "UNDECIDED"
    keep_drop_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError("unknown uncertainty method")
        object.__setattr__(self, "weekly_scores", tuple(self.weekly_scores))
        if not self.weekly_scores:
            raise ValueError("evaluation requires at least one weekly fold")
        if self.keep_drop not in KEEP_DROP:
            raise ValueError("invalid keep_drop")
        evidence = tuple(self.keep_drop_evidence)
        if self.keep_drop != "UNDECIDED" and not evidence:
            raise ValueError("KEEP/DROP requires evidence")
        object.__setattr__(self, "keep_drop_evidence", evidence)

    def to_row(self) -> dict:
        return {"method": self.method, "mean_score": self.mean_score, "median_score": self.median_score,
                "worst_week": self.worst_week_score, "capture": self.capture_rate,
                "sharpness": self.mean_sharpness, "mae": self.mean_mae,
                "stability": self.score_stdev, "keep_drop": self.keep_drop}


def evaluate_sampler(panel: ResidualPanel, method: str, *, seed: int, n_samples: int = 300,
                     degrees_of_freedom: float | None = None, week_hours: int = 168,
                     target_coverage: float = 0.9, score_fn=None) -> SamplerEvaluation:
    """Expanding-window walk-forward backtest: each week is scored from a model fit only on earlier weeks.

    The first weekly fold is training-only history (no prior week exists to fit
    from), matching the hidden test being an unseen future week. Truth is the
    OOF residual itself (this layer is evaluated independently of which point
    forecast produced it), so `means` is the zero vector throughout.
    """
    if not isinstance(panel, ResidualPanel):
        raise ValueError("ResidualPanel required")
    folds = weekly_folds(panel, week_hours=week_hours)
    if len(folds) < 2:
        raise ValueError("evaluation requires at least two weekly folds: one to train, one to hold out")
    scorer = score_fn or (lambda capture, sharpness, mae, rows: _default_score(capture, sharpness, mae,
                                                                               target_coverage=target_coverage))
    weekly = []
    for held_out_index in range(1, len(folds)):
        held_out = folds[held_out_index]
        training_rows = tuple(row for fold in folds[:held_out_index] for row in fold)
        training_panel = ResidualPanel(panel.target_names, training_rows)
        model = fit_uncertainty_model(training_panel, method=method, fit_cutoff=held_out[0][0],
                                      degrees_of_freedom=degrees_of_freedom)
        hits = {name: 0 for name in panel.target_names}
        widths = {name: [] for name in panel.target_names}
        errors = {name: [] for name in panel.target_names}
        biases = {name: [] for name in panel.target_names}
        horizon_errors: dict = {}
        lower_q, upper_q = (1 - target_coverage) / 2, 1 - (1 - target_coverage) / 2
        for row_index, row in enumerate(held_out):
            _, _, horizon, truth = row
            key = bucket_key_for(method, row)
            means = {name: 0.0 for name in panel.target_names}
            draws = sample_distribution(model, means, bucket_key=key,
                                        seed=seed + held_out_index * 10_000 + row_index,
                                        n_samples=n_samples, round_to_int=False)
            for target_index, name in enumerate(panel.target_names):
                values = sorted(draws[name])
                lo, hi = empirical_quantile(values, lower_q), empirical_quantile(values, upper_q)
                truth_value = truth[target_index]
                hits[name] += lo <= truth_value <= hi
                widths[name].append(hi - lo)
                point = median(values)
                errors[name].append(abs(point - truth_value))
                biases[name].append(point - truth_value)
                horizon_errors.setdefault(str(horizon), []).append(abs(point - truth_value))
        n_rows = len(held_out)
        capture = mean(hits[name] / n_rows for name in panel.target_names)
        sharpness = mean(mean(widths[name]) for name in panel.target_names)
        mae_value = mean(mean(errors[name]) for name in panel.target_names)
        bias_value = mean(mean(biases[name]) for name in panel.target_names)
        per_target = tuple((name, mean(errors[name])) for name in panel.target_names)
        per_horizon = tuple((horizon_name, mean(values)) for horizon_name, values in horizon_errors.items())
        score = scorer(capture, sharpness, mae_value, held_out)
        if not isfinite(score):
            raise ValueError("sampler score must be finite")
        weekly.append(WeeklyScore(held_out[0][0], score, capture, sharpness, mae_value, bias_value,
                                  per_target, per_horizon))
    scores = tuple(week.score for week in weekly)
    return SamplerEvaluation(method, tuple(weekly), mean(scores), median(scores), max(scores),
                             pstdev(scores) if len(scores) > 1 else 0.0,
                             mean(week.capture for week in weekly), mean(week.sharpness for week in weekly),
                             mean(week.bias for week in weekly), mean(week.mae for week in weekly))


def compare_samplers(evaluations: dict, *, baseline: str = "independent_gaussian",
                     tolerance: float = 0.0) -> tuple[SamplerEvaluation, ...]:
    """Attach KEEP/DROP relative to `baseline` from paired weekly score deltas; OOF-only, no ties broken by intuition."""
    if baseline not in evaluations:
        raise ValueError("baseline method must be present in evaluations")
    base_scores = {week.week_start: week.score for week in evaluations[baseline].weekly_scores}
    result = []
    for method, evaluation in evaluations.items():
        if method == baseline:
            result.append(replace(evaluation, keep_drop="KEEP", keep_drop_evidence=("baseline method",)))
            continue
        shared = [(week.score, base_scores[week.week_start])
                  for week in evaluation.weekly_scores if week.week_start in base_scores]
        if len(shared) < 2:
            result.append(evaluation)
            continue
        deltas = [base_score - candidate_score for candidate_score, base_score in shared]
        wins = sum(delta > tolerance for delta in deltas)
        if wins > len(deltas) / 2:
            decision = "KEEP"
        elif wins < len(deltas) / 2:
            decision = "DROP"
        else:
            decision = "UNDECIDED"
        evidence = (f"beat baseline on {wins}/{len(deltas)} weekly folds",)
        result.append(replace(evaluation, keep_drop=decision, keep_drop_evidence=evidence))
    return tuple(sorted(result, key=lambda evaluation: evaluation.mean_score))


def capture_sharpness_curve(model: UncertaintyModel, panel: ResidualPanel, *, scales, seed: int,
                            n_samples: int = 300, target_coverage: float = 0.9,
                            bucket_key=None) -> tuple[tuple[float, float, float], ...]:
    """Sweep variance-scale multipliers to show the capture/sharpness tradeoff (deliverable #8)."""
    lower_q, upper_q = (1 - target_coverage) / 2, 1 - (1 - target_coverage) / 2
    result = []
    for scale in scales:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not isfinite(scale) or scale <= 0:
            raise ValueError("scale multiplier must be positive and finite")
        scaled = model.with_calibration_scale({bucket_key: float(scale)})
        hits = total = 0
        widths = []
        for index, row in enumerate(panel.rows):
            means = {name: 0.0 for name in panel.target_names}
            draws = sample_distribution(scaled, means, bucket_key=bucket_key, seed=seed + index,
                                        n_samples=n_samples, round_to_int=False)
            for target_index, name in enumerate(panel.target_names):
                values = sorted(draws[name])
                lo, hi = empirical_quantile(values, lower_q), empirical_quantile(values, upper_q)
                widths.append(hi - lo)
                hits += lo <= row[3][target_index] <= hi
                total += 1
        result.append((float(scale), hits / total, mean(widths)))
    return tuple(result)
