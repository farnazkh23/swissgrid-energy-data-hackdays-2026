"""Baseline candidate execution and OOF-only scoreboard construction."""
from dataclasses import dataclass
from datetime import timedelta
from statistics import mean

from .champion import HistoricalConditional, OptionalEstimator, Persistence, SeasonalPersistence, select_champion
from .feature_registry import identity
from .metrics import bias, brier, coverage, mae, pinball, rmse, sharpness
from .oof import OOFResult, run_oof


def _empirical_quantile(values, probability):
    from .histogram import empirical_quantile
    return empirical_quantile(values, probability)


def _prediction_quantile(prediction, probability):
    if prediction.prediction_type == "distribution":
        return _empirical_quantile(prediction.values, probability)
    raise ValueError("distribution prediction required")


def _mean_prediction(prediction):
    return mean(prediction.values)


@dataclass(frozen=True, slots=True)
class BaselineRun:
    oof: OOFResult
    scoreboard: dict
    capabilities: dict
    selected_model: tuple[str, str]


def baseline_factories(*, include_ridge=True):
    """Return the eligible factories and capability records without installing anything."""
    factories = [lambda: Persistence(),
                 lambda: SeasonalPersistence(timedelta(hours=24)),
                 lambda: HistoricalConditional(())]
    capabilities = {
        "persistence": {"available": True, "reason": None},
        "seasonal_persistence": {"available": True, "reason": None},
        "historical_conditional": {"available": True, "reason": None},
    }
    if include_ridge:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            capabilities["ridge"] = {"available": False, "reason": "scikit-learn unavailable"}
        else:
            capabilities["ridge"] = {"available": True, "reason": None}
            factories.append(lambda: OptionalEstimator("ridge"))
    return tuple(factories), capabilities


def score_oof(oof: OOFResult) -> dict:
    """Score only records represented in the validated rolling OOF result."""
    scoreboard = {}
    for model_id, version in oof.candidate_identities:
        records = tuple(r for r in oof.records if (r.model_id, r.version) == (model_id, version))
        truth = [oof.truth(record.fold_id, record.row_id) for record in records]
        point = [_mean_prediction(record.prediction) for record in records]
        metrics = {"mae": mae(truth, point), "rmse": rmse(truth, point), "bias": bias(truth, point)}
        distributions = [r for r in records if r.prediction.prediction_type == "distribution"]
        if len(distributions) == len(records):
            for level in (0.10, 0.50, 0.90):
                predictions = [_prediction_quantile(r.prediction, level) for r in records]
                metrics[f"pinball_p{int(level * 100):02d}"] = pinball(truth, predictions, level)
            lower_10 = [_prediction_quantile(r.prediction, .10) for r in records]
            upper_90 = [_prediction_quantile(r.prediction, .90) for r in records]
            lower_25 = [_prediction_quantile(r.prediction, .25) for r in records]
            upper_75 = [_prediction_quantile(r.prediction, .75) for r in records]
            metrics["coverage_p10_p90"] = coverage(truth, lower_10, upper_90)
            metrics["sharpness_p10_p90"] = sharpness(lower_10, upper_90)
            metrics["coverage_p25_p75"] = coverage(truth, lower_25, upper_75)
            metrics["sharpness_p25_p75"] = sharpness(lower_25, upper_75)
            import_prob = [sum(v >= 0 for v in r.prediction.values) / len(r.prediction.values) for r in records]
            import_truth = [1 if value >= 0 else 0 for value in truth]
            metrics["brier_import"] = brier(import_truth, import_prob)
        scoreboard[model_id] = {
            "model_id": model_id, "model_version": version, "oof_rows": len(records),
            "metrics": metrics, "scored_from": "rolling_oof_predictions_only",
        }
    return scoreboard


def run_baselines(plan, rows, feature_manifest_hash, *, include_ridge=True) -> BaselineRun:
    factories, capabilities = baseline_factories(include_ridge=include_ridge)
    results = []
    for factory in factories:
        model_id = factory().model_id
        try:
            result = run_oof(plan, rows, (factory,), feature_manifest_hash)
            results.append(result)
            capabilities[model_id]["eligible"] = True
        except (ValueError, RuntimeError) as exc:
            capabilities[model_id] = {"available": True, "eligible": False,
                                      "reason": str(exc)}
    if not results:
        raise ValueError("no eligible baseline candidates produced complete OOF evidence")
    first = results[0]
    oof = OOFResult(plan, first.folds,
                    tuple(record for result in results for record in result.records),
                    feature_manifest_hash,
                    tuple(identity for result in results for identity in result.candidate_identities))
    scoreboard = score_oof(oof)
    selected, _ = select_champion(oof, metric="mae")
    capabilities["selection"] = {"metric": "mae", "selected_model_id": selected[0],
                                  "selected_model_version": selected[1]}
    return BaselineRun(oof, scoreboard, capabilities, selected)
