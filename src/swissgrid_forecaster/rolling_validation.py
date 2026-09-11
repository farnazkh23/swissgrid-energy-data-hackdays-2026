"""Organizer-style rolling validation for the visible Challenge-4 weeks.

This module is deliberately independent of Databricks and official submission
APIs.  It consumes an hourly target history plus the organizer's visible
realizations, creates twelve fixed-origin 168-hour folds, selects point and
uncertainty methods only from earlier folds, and writes debugging artifacts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import json
from math import sqrt
from pathlib import Path
from statistics import mean, median, pstdev

from .real_modeling_v1 import LSEGFeatures, WeeklyFold, _run_scenario
from .sampler_evaluation import evaluate_sampler
from .target_contract import TARGETS, validate_output_targets
from .uncertainty_model import (
    ResidualPanel,
    fit_uncertainty_model,
    sample_distribution,
)

HOUR = timedelta(hours=1)
WEEK_HOURS = 168
TRAIN_HOURS = 672
VALIDATION_WEEKS = 12
WARM_START_WEEKS = 8
SAMPLES_PER_TARGET = 300
POINT_MODELS = ("ridge", "seasonal_persistence", "hist_gradient_boosting")
UNCERTAINTY_METHODS = {
    "independent_gaussian": {},
    "correlated_gaussian": {},
    "empirical_bootstrap": {},
    "student_t": {"degrees_of_freedom": 6.0},
    "horizon_covariance": {},
}


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def build_validation_folds(
    validation_timestamps, available_timestamps, *, weeks: int = VALIDATION_WEEKS,
    train_hours: int = TRAIN_HOURS,
) -> tuple[WeeklyFold, ...]:
    """Build exactly consecutive fixed-origin folds from organizer timestamps."""
    stamps = tuple(validation_timestamps)
    if len(stamps) != weeks * WEEK_HOURS:
        raise ValueError(f"expected {weeks * WEEK_HOURS} validation timestamps")
    if len(set(stamps)) != len(stamps) or tuple(sorted(stamps)) != stamps:
        raise ValueError("validation timestamps must be unique and chronological")
    if any(right - left != HOUR for left, right in zip(stamps, stamps[1:])):
        raise ValueError("validation timestamps must be consecutive hourly rows")
    available = set(available_timestamps)
    folds = []
    for index in range(weeks):
        forecast = stamps[index * WEEK_HOURS:(index + 1) * WEEK_HOURS]
        issue = forecast[0] - HOUR
        train = tuple(issue - timedelta(hours=train_hours - 1 - offset)
                      for offset in range(train_hours))
        if any(timestamp not in available for timestamp in (*train, *forecast)):
            raise ValueError(f"fold {index + 1} has unavailable training or validation timestamps")
        if issue >= forecast[0] or any(issue >= timestamp for timestamp in forecast):
            raise ValueError(f"fold {index + 1} violates fixed-origin chronology")
        folds.append(WeeklyFold(f"validation_week_{index + 1:02d}", train, forecast))
    result = tuple(folds)
    validate_validation_folds(result)
    return result


def validate_validation_folds(folds) -> tuple[WeeklyFold, ...]:
    """Validate fixed-origin chronology and prevent future-week training."""
    folds = tuple(folds)
    if not folds:
        raise ValueError("at least one validation fold is required")
    for index, fold in enumerate(folds, start=1):
        if len(fold.forecast_timestamps) != WEEK_HOURS:
            raise ValueError(f"fold {index} must contain exactly 168 forecast rows")
        if fold.issue_time >= fold.forecast_start:
            raise ValueError(f"fold {index} issue time must precede forecast start")
        if any(timestamp - fold.issue_time != HOUR * (offset + 1)
               for offset, timestamp in enumerate(fold.forecast_timestamps)):
            raise ValueError(f"fold {index} horizons must be exactly 1..168")
        if any(timestamp > fold.issue_time for timestamp in fold.train_timestamps):
            raise ValueError(f"fold {index} trains on data after its issue time")
        if index > 1 and fold.forecast_start <= folds[index - 2].forecast_timestamps[-1]:
            raise ValueError("validation weeks must be strictly chronological")
    return folds


def _point_metrics(actual, prediction) -> dict:
    errors = [p - a for a, p in zip(actual, prediction)]
    return {
        "mae": mean(abs(error) for error in errors),
        "rmse": sqrt(mean(error * error for error in errors)),
        "bias": mean(errors),
    }


def _distribution_metrics(actual, samples) -> tuple[dict, list[float]]:
    captures = []
    sharpness = []
    row_scores = []
    for truth, values in zip(actual, samples):
        ordered = sorted(values)
        minimum, maximum = ordered[0], ordered[-1]
        average = mean(values)
        standard_deviation = pstdev(values)
        captured = float(minimum <= truth <= maximum)
        goodness = 0.0 if not captured else max(0.0, 1.0 - abs(truth - average) / (3.0 * standard_deviation)) if standard_deviation else float(truth == average)
        sharp = 0.0 if not captured else 1.0 if not standard_deviation else (maximum - minimum) / (maximum - minimum + 2.0 * standard_deviation)
        captures.append(captured)
        sharpness.append(sharp)
        row_scores.append(0.2 * captured + 0.45 * goodness + 0.35 * sharp)
    return {"capture": mean(captures), "sharpness": mean(sharpness)}, row_scores


def _validate_samples(samples_by_target):
    validate_output_targets(tuple(samples_by_target))
    for target in TARGETS:
        values = samples_by_target[target]
        if len(values) != SAMPLES_PER_TARGET or any(not isinstance(value, int) for value in values):
            raise ValueError(f"{target} requires exactly 300 integer samples")


def organizer_local_score(predictions, actuals, timestamps) -> float:
    """Exact pure-Python equivalent of edh2026.evaluate_submission scoring."""
    if len(timestamps) != WEEK_HOURS or len(predictions) != WEEK_HOURS:
        raise ValueError("organizer scoring requires exactly 168 rows")
    if tuple(predictions) != tuple(timestamps) or set(predictions) != set(timestamps):
        raise ValueError("prediction and realization timestamps are not exactly aligned")
    row_scores = []
    for target in TARGETS:
        actual = [actuals[timestamp][target] for timestamp in timestamps]
        distribution = [predictions[timestamp][target] for timestamp in timestamps]
        _, scores = _distribution_metrics(actual, distribution)
        row_scores.append(scores)
    return mean(sum(row_scores[target_index][row_index] for target_index in range(len(TARGETS))) / len(TARGETS)
                 for row_index in range(WEEK_HOURS))


def _score_week(predictions, actuals, timestamps, official_score_fn=None) -> tuple[dict, dict]:
    if len(timestamps) != WEEK_HOURS or len(predictions) != WEEK_HOURS:
        raise ValueError("organizer scoring requires exactly 168 rows")
    if tuple(predictions) != tuple(timestamps) or set(predictions) != set(timestamps):
        raise ValueError("prediction and realization timestamps are not exactly aligned")
    target_metrics = {}
    row_scores = []
    for target in TARGETS:
        actual = [actuals[timestamp][target] for timestamp in timestamps]
        point = [mean(predictions[timestamp][target]) for timestamp in timestamps]
        metrics = _point_metrics(actual, point)
        distribution = [predictions[timestamp][target] for timestamp in timestamps]
        distribution_metrics, scores = _distribution_metrics(actual, distribution)
        metrics.update(distribution_metrics)
        target_metrics[target] = metrics
        row_scores.append(scores)
    diagnostic_score = mean(sum(row_scores[target_index][row_index] for target_index in range(len(TARGETS))) / len(TARGETS)
                            for row_index in range(WEEK_HOURS))
    official_score = (official_score_fn(predictions, actuals, timestamps)
                      if official_score_fn is not None else organizer_local_score(predictions, actuals, timestamps))
    aggregate = {
        "official_local_score": official_score,
        "diagnostic_score": diagnostic_score,
        "capture": mean(target_metrics[target]["capture"] for target in TARGETS),
        "sharpness": mean(target_metrics[target]["sharpness"] for target in TARGETS),
        "MAE": mean(target_metrics[target]["mae"] for target in TARGETS),
    }
    return target_metrics, aggregate


def _choose_point_models(predictions, actuals, folds, week_index, point_models):
    selected = {}
    for target in TARGETS:
        if week_index == 0:
            selected[target] = "seasonal_persistence"
            continue
        scores = {}
        for model in point_models:
            errors = []
            for fold in folds[:week_index]:
                for timestamp in fold.forecast_timestamps:
                    row = predictions[(fold.fold_id, timestamp, target, model)]
                    errors.append(abs(row["pred"] - actuals[timestamp][target]))
            scores[model] = mean(errors)
        selected[target] = min(scores, key=lambda model: (scores[model], model))
    return selected


def _residual_panel(predictions, actuals, folds, selected_models_by_week) -> ResidualPanel:
    rows = []
    for fold_index, fold in enumerate(folds):
        selected_models = selected_models_by_week[fold_index]
        for timestamp in fold.forecast_timestamps:
            residuals = tuple(actuals[timestamp][target] - predictions[(fold.fold_id, timestamp, target, selected_models[target])]["pred"] for target in TARGETS)
            rows.append((fold.issue_time, timestamp, timestamp - fold.issue_time, residuals))
    return ResidualPanel(TARGETS, tuple(rows))


def _choose_uncertainty_method(prior_panel: ResidualPanel, prior_weeks: int, seed: int) -> str:
    if prior_weeks < 2:
        return "correlated_gaussian"
    scores = {}
    for method, kwargs in UNCERTAINTY_METHODS.items():
        try:
            evaluation = evaluate_sampler(prior_panel, method, seed=seed, n_samples=64, **kwargs)
        except ValueError:
            continue
        scores[method] = evaluation.mean_score
    return min(scores, key=lambda method: (scores[method], method)) if scores else "correlated_gaussian"


def _samples_for_week(method, prior_panel, point_forecasts, timestamps, seed):
    if prior_panel is None or not prior_panel.rows:
        return {timestamp: {target: tuple(int(round(point_forecasts[timestamp][target])) for _ in range(SAMPLES_PER_TARGET)) for target in TARGETS} for timestamp in timestamps}
    kwargs = UNCERTAINTY_METHODS[method]
    model = fit_uncertainty_model(prior_panel, method=method, fit_cutoff=timestamps[0] - HOUR, **kwargs)
    result = {}
    for index, timestamp in enumerate(timestamps):
        horizon = timestamp - (timestamps[0] - HOUR)
        bucket_key = horizon if method == "horizon_covariance" else None
        draws = sample_distribution(model, point_forecasts[timestamp], bucket_key=bucket_key,
                                    seed=seed + index, n_samples=SAMPLES_PER_TARGET)
        result[timestamp] = draws
    return result


def run_rolling_validation(*, hourly_targets, realizations, output_dir=None,
                           db_tables=None, lseg=None,
                           point_models=POINT_MODELS,
                           validation_weeks: int = VALIDATION_WEEKS,
                           warm_start_weeks: int = WARM_START_WEEKS,
                           official_score_fn=None,
                           seed: int = 20260910) -> dict:
    """Run the 12-week organizer-style validation harness without submission."""
    validate_output_targets(TARGETS)
    validation_timestamps = tuple(sorted(realizations))
    folds = build_validation_folds(validation_timestamps, hourly_targets, weeks=validation_weeks)
    warm_folds = ()
    if warm_start_weeks:
        warm_timestamps = tuple(
            folds[0].forecast_start - timedelta(hours=WEEK_HOURS * warm_start_weeks + 1)
            + index * HOUR for index in range(WEEK_HOURS * warm_start_weeks))
        warm_folds = build_validation_folds(warm_timestamps, hourly_targets, weeks=warm_start_weeks)
    all_folds = tuple(warm_folds) + tuple(folds)
    all_actuals = dict(realizations)
    for fold in warm_folds:
        for timestamp in fold.forecast_timestamps:
            all_actuals[timestamp] = {target: hourly_targets[timestamp][target] for target in TARGETS}
    lseg = lseg or LSEGFeatures({}, {}, {}, ())
    _, _, raw_predictions = _run_scenario(
        targets=hourly_targets, db_tables=db_tables or {}, lseg=lseg, folds=all_folds,
        use_lseg=False, scenario="rolling_validation", seed=seed,
        model_names=point_models, collect_predictions=True,
    )
    indexed = {(row["fold_id"], datetime.fromisoformat(row["timestamp"]), row["target"], row["model"]): row for row in raw_predictions}
    selected_by_week = []
    week_rows = []
    target_rows = defaultdict(list)
    prediction_artifacts = {}
    for global_index, fold in enumerate(all_folds):
        selected_models = _choose_point_models(indexed, all_actuals, all_folds, global_index, point_models)
        selected_by_week.append(selected_models)
        if global_index < warm_start_weeks:
            continue
        week_index = global_index - warm_start_weeks
        prior_panel = _residual_panel(indexed, all_actuals, all_folds[:global_index], selected_by_week[:global_index])
        method = _choose_uncertainty_method(prior_panel, global_index, seed)
        point_forecasts = {timestamp: {target: indexed[(fold.fold_id, timestamp, target, selected_models[target])]["pred"] for target in TARGETS} for timestamp in fold.forecast_timestamps}
        samples = _samples_for_week(method, prior_panel, point_forecasts, fold.forecast_timestamps, seed + week_index * 10000)
        target_metrics, aggregate = _score_week(samples, realizations, fold.forecast_timestamps, official_score_fn)
        prediction_artifacts[fold.fold_id] = tuple({"timestamp": _iso(timestamp), **{f"{target}_samples": list(samples[timestamp][target]) for target in TARGETS}} for timestamp in fold.forecast_timestamps)
        week_rows.append({"week_number": week_index + 1, "forecast_start": _iso(fold.forecast_start), "forecast_end": _iso(fold.forecast_timestamps[-1]), "issue_time": _iso(fold.issue_time), "score": aggregate["official_local_score"], "official_local_score": aggregate["official_local_score"], "diagnostic_score": aggregate["diagnostic_score"], "capture": aggregate["capture"], "sharpness": aggregate["sharpness"], "MAE": aggregate["MAE"], **target_metrics, "targets": target_metrics, "selected_point_model": selected_models, "selected_uncertainty_method": method, "calibration_parameters": {}})
        for target in TARGETS:
            target_rows[target].append(target_metrics[target])
    scores = [row["official_local_score"] for row in week_rows]
    development_scores = scores[:validation_weeks - 1]
    summary = {"weeks_completed": len(week_rows), "development_weeks": validation_weeks - 1, "holdout_week": validation_weeks, "mean_score_weeks_1_11": mean(development_scores), "median_score_weeks_1_11": median(development_scores), "worst_score_weeks_1_11": min(development_scores), "std_score_weeks_1_11": pstdev(development_scores), "week_12_holdout_score": scores[-1], "all_12_mean_score": mean(scores), "all_12_median_score": median(scores), "warm_start_weeks": warm_start_weeks, "historical_calibration_timestamps": [_iso(timestamp) for fold in warm_folds for timestamp in fold.forecast_timestamps]}
    by_target = {}
    for target in TARGETS:
        rows = target_rows[target]
        by_target[target] = {"mean_MAE": mean(row["mae"] for row in rows), "median_MAE": median(row["mae"] for row in rows), "worst_week_MAE": max(row["mae"] for row in rows), "mean_bias": mean(row["bias"] for row in rows), "mean_RMSE": mean(row["rmse"] for row in rows), "mean_capture": mean(row["capture"] for row in rows), "mean_sharpness": mean(row["sharpness"] for row in rows)}
    result = {"summary": summary, "by_week": week_rows, "by_target": by_target, "predictions": prediction_artifacts}
    if output_dir is not None:
        directory = Path(output_dir)
        (directory / "predictions").mkdir(parents=True, exist_ok=True)
        (directory / "rolling_validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (directory / "rolling_validation_by_week.json").write_text(json.dumps(week_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (directory / "rolling_validation_by_target.json").write_text(json.dumps(by_target, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for fold_id, rows in prediction_artifacts.items():
            (directory / "predictions" / f"{fold_id}.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
