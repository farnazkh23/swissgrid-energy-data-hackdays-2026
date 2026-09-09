"""Validated forecast output and deterministic quantile/probability helpers."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite

from .availability import nonempty, utc
from .histogram import empirical_quantile
from .manifest import validate_digest


QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)


def repair_quantiles(values, levels=QUANTILE_LEVELS) -> tuple[tuple[float, ...], bool]:
    """Repair crossings with a deterministic cumulative maximum."""
    values, levels = tuple(float(value) for value in values), tuple(float(level) for level in levels)
    if len(values) != len(levels) or any(not isfinite(value) for value in values):
        raise ValueError("finite quantiles at all requested levels are required")
    if any(not 0 < level < 1 for level in levels) or any(a >= b for a, b in zip(levels, levels[1:])):
        raise ValueError("quantile levels must be strictly ordered inside (0, 1)")
    repaired = []
    for value in values:
        repaired.append(max(value, repaired[-1]) if repaired else value)
    result = tuple(repaired)
    return result, result != values


def probabilities_from_samples(samples):
    """Use the declared sign convention: nonnegative is import, negative export."""
    values = tuple(float(value) for value in samples)
    if not values or not all(isfinite(value) for value in values):
        raise ValueError("nonempty finite samples required")
    probability_import = sum(value >= 0 for value in values) / len(values)
    probability_export = sum(value < 0 for value in values) / len(values)
    return probability_import, probability_export


def _duration_text(value: timedelta) -> str:
    return str(value)


@dataclass(frozen=True, slots=True)
class ForecastOutput:
    forecast_id: str
    issue_time: datetime
    target_time: datetime
    horizon: timedelta
    target_name: str
    target_entity: str
    unit: str
    sign_convention: str
    selected_model_id: str
    selected_model_version: str
    feature_manifest_hash: str
    training_data_manifest_hash: str
    fit_cutoff: datetime
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    samples: tuple[float, ...] | None
    probability_import: float | None
    probability_export: float | None
    source_ids: tuple[str, ...]
    dataset_manifest_id: str
    feature_manifest_id: str
    fold_evaluation_summary: dict
    warnings: tuple[str, ...]
    fallback_abstention_state: dict
    quantile_repair_applied: bool = False

    def __post_init__(self):
        nonempty(self.forecast_id, "forecast_id")
        for name in ("issue_time", "target_time", "fit_cutoff"):
            object.__setattr__(self, name, utc(getattr(self, name), name))
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("positive horizon required")
        if self.target_time != self.issue_time + self.horizon or self.fit_cutoff > self.issue_time:
            raise ValueError("forecast chronology violation")
        for name in ("target_name", "target_entity", "unit", "sign_convention",
                     "selected_model_id", "selected_model_version"):
            nonempty(getattr(self, name), name)
        validate_digest(self.feature_manifest_hash)
        validate_digest(self.training_data_manifest_hash)
        validate_digest(self.dataset_manifest_id)
        validate_digest(self.feature_manifest_id)
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if not self.source_ids or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source identities are required")
        quantiles, repaired = repair_quantiles((self.p10, self.p25, self.p50, self.p75, self.p90))
        if quantiles != (self.p10, self.p25, self.p50, self.p75, self.p90):
            raise ValueError("quantiles must be repaired before constructing output")
        if type(self.quantile_repair_applied) is not bool or self.quantile_repair_applied != repaired:
            raise ValueError("quantile repair state is inconsistent")
        if not isinstance(self.fallback_abstention_state, dict) or any(
                key not in self.fallback_abstention_state for key in ("fallback_used", "abstained")):
            raise ValueError("explicit fallback/abstention state is required")
        if any(type(self.fallback_abstention_state[key]) is not bool
               for key in ("fallback_used", "abstained")):
            raise ValueError("fallback/abstention flags must be boolean")
        if self.samples is not None:
            object.__setattr__(self, "samples", tuple(float(value) for value in self.samples))
            if not self.samples or not all(isfinite(value) for value in self.samples):
                raise ValueError("samples must be nonempty and finite")
            expected = probabilities_from_samples(self.samples)
            if (self.probability_import, self.probability_export) != expected:
                raise ValueError("probabilities must be derived from samples")
        elif self.probability_import is not None or self.probability_export is not None:
            raise ValueError("probabilities require samples")
        if self.probability_import is not None and abs(self.probability_import + self.probability_export - 1) > 1e-12:
            raise ValueError("import/export probabilities must sum to one")

    @classmethod
    def from_samples(cls, *, forecast_id, issue_time, horizon, target_name,
                     target_entity, unit, selected_model_id, selected_model_version,
                     feature_manifest_hash, training_data_manifest_hash, fit_cutoff,
                     samples, source_ids, dataset_manifest_id, feature_manifest_id,
                     fold_evaluation_summary, warnings=(), fallback_abstention_state=None,
                     sign_convention="positive=import; negative=export"):
        values = tuple(float(value) for value in samples)
        raw = tuple(empirical_quantile(values, level) for level in QUANTILE_LEVELS)
        quantiles, repaired = repair_quantiles(raw)
        probability_import, probability_export = probabilities_from_samples(values)
        return cls(forecast_id, issue_time, utc(issue_time) + horizon, horizon,
                   target_name, target_entity, unit, sign_convention, selected_model_id,
                   selected_model_version, feature_manifest_hash,
                   training_data_manifest_hash, fit_cutoff, *quantiles, values,
                   probability_import, probability_export, tuple(source_ids),
                   dataset_manifest_id, feature_manifest_id, dict(fold_evaluation_summary),
                   tuple(warnings), dict(fallback_abstention_state or {
                       "fallback_used": False, "abstained": False, "reason": None}), repaired)

    def to_dict(self) -> dict:
        return {
            "forecast_id": self.forecast_id,
            "issue_time": self.issue_time.isoformat(), "target_time": self.target_time.isoformat(),
            "horizon": _duration_text(self.horizon), "target_name": self.target_name,
            "target_entity": self.target_entity, "unit": self.unit,
            "sign_convention": self.sign_convention,
            "model": {"selected_model_id": self.selected_model_id,
                      "selected_model_version": self.selected_model_version,
                      "feature_manifest_hash": self.feature_manifest_hash,
                      "training_data_manifest_hash": self.training_data_manifest_hash,
                      "fit_cutoff": self.fit_cutoff.isoformat()},
            "quantiles": {"p10": self.p10, "p25": self.p25, "p50": self.p50,
                          "p75": self.p75, "p90": self.p90,
                          "quantile_repair_applied": self.quantile_repair_applied},
            "samples": list(self.samples) if self.samples is not None else None,
            "probability_import": self.probability_import,
            "probability_export": self.probability_export,
            "provenance": {"source_ids": list(self.source_ids),
                           "dataset_manifest_id": self.dataset_manifest_id,
                           "feature_manifest_id": self.feature_manifest_id,
                           "fold_evaluation_summary": self.fold_evaluation_summary},
            "warnings": list(self.warnings),
            "fallback_abstention_state": self.fallback_abstention_state,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ForecastOutput":
        model, quantiles, provenance = payload["model"], payload["quantiles"], payload["provenance"]
        samples = None if payload.get("samples") is None else tuple(payload["samples"])
        return cls(payload["forecast_id"], datetime.fromisoformat(payload["issue_time"]),
                   datetime.fromisoformat(payload["target_time"]),
                   timedelta_from_text(payload["horizon"]), payload["target_name"],
                   payload["target_entity"], payload["unit"], payload["sign_convention"],
                   model["selected_model_id"], model["selected_model_version"],
                   model["feature_manifest_hash"], model["training_data_manifest_hash"],
                   datetime.fromisoformat(model["fit_cutoff"]), quantiles["p10"],
                   quantiles["p25"], quantiles["p50"], quantiles["p75"], quantiles["p90"],
                   samples, payload.get("probability_import"), payload.get("probability_export"),
                   tuple(provenance["source_ids"]), provenance["dataset_manifest_id"],
                   provenance["feature_manifest_id"], provenance["fold_evaluation_summary"],
                   tuple(payload.get("warnings", ())), payload["fallback_abstention_state"],
                   quantiles.get("quantile_repair_applied", False))


def timedelta_from_text(value: str) -> timedelta:
    try:
        days, clock = value.split(" day, ") if " day, " in value else ("0", value)
        hours, minutes, seconds = clock.split(":")
        return timedelta(days=int(days), hours=int(hours), minutes=int(minutes), seconds=float(seconds))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid timedelta transport value") from exc
