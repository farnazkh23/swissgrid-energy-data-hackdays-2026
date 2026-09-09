"""Framework-neutral, versioned response models for forecast serving.

The models in this module are transport models.  They intentionally contain
only values present in the persisted artifacts (plus serving state and stale
warnings derived from the supplied clock); they never calculate a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
from math import isfinite
from typing import Any, Mapping

from .forecast_output import ForecastOutput
from .histogram import HistogramPayload


API_SCHEMA_VERSION = "forecast-api.v1"


class ForecastState(str, Enum):
    NORMAL = "NORMAL"
    FALLBACK = "FALLBACK"
    ABSTAIN = "ABSTAIN"
    DEGRADED = "DEGRADED"


def _json_value(value: Any) -> Any:
    """Return a JSON-safe copy while rejecting non-transport values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("API payload contains a nonfinite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported API value: {type(value).__name__}")


def serialize_json(payload: Mapping[str, Any]) -> str:
    """Serialize a response deterministically for caches and API tests."""
    return json.dumps(_json_value(payload), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _state_for(forecast: ForecastOutput, *, stale: bool) -> ForecastState:
    flags = forecast.fallback_abstention_state
    if flags.get("abstained") is True:
        return ForecastState.ABSTAIN
    if flags.get("fallback_used") is True:
        return ForecastState.FALLBACK
    if stale or forecast.warnings:
        return ForecastState.DEGRADED
    return ForecastState.NORMAL


@dataclass(frozen=True, slots=True)
class ForecastResponse:
    """The complete frontend forecast response."""

    payload: dict

    @classmethod
    def from_artifacts(cls, forecast: ForecastOutput, histogram: HistogramPayload,
                       scoreboard: Mapping[str, Any], provenance: Mapping[str, Any],
                       *, stale: bool = False,
                       stale_warning: str | None = None) -> "ForecastResponse":
        if not isinstance(forecast, ForecastOutput) or not isinstance(histogram, HistogramPayload):
            raise ValueError("validated forecast and histogram are required")
        if not isinstance(scoreboard, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("scoreboard and provenance objects are required")

        selected_model = forecast.selected_model_id
        selected_version = forecast.selected_model_version
        candidates = scoreboard.get("candidates", {})
        candidate = candidates.get(selected_model, {}) if isinstance(candidates, Mapping) else {}
        metrics = candidate.get("metrics") if isinstance(candidate, Mapping) else None
        if metrics is not None and not isinstance(metrics, Mapping):
            raise ValueError("selected model metrics must be an object")

        warnings = list(forecast.warnings)
        if stale_warning and stale_warning not in warnings:
            warnings.append(stale_warning)

        quality = provenance.get("freshness", provenance.get("feature_quality_flags"))
        payload = {
            "schema_version": API_SCHEMA_VERSION,
            "forecast": {
                "forecast_id": forecast.forecast_id,
                "issue_time": forecast.issue_time.isoformat(),
                "target_time": forecast.target_time.isoformat(),
                "horizon": str(forecast.horizon),
                "target": {"name": forecast.target_name, "entity": forecast.target_entity},
                "unit": forecast.unit,
                "sign_convention": forecast.sign_convention,
            },
            "probability": {
                "p10": forecast.p10,
                "p25": forecast.p25,
                "p50": forecast.p50,
                "p75": forecast.p75,
                "p90": forecast.p90,
                "probability_import": forecast.probability_import,
                "probability_export": forecast.probability_export,
            },
            "visualization": {
                "histogram": histogram.to_dict(),
                "central_50_interval": list(histogram.central_50_interval),
                "central_90_interval": list(histogram.central_90_interval),
                "median": histogram.median,
            },
            "model": {
                "selected_champion": selected_model,
                "selected_model": selected_model,
                "model_version": selected_version,
                "oof_metrics": None if metrics is None else dict(metrics),
                "fit_cutoff": forecast.fit_cutoff.isoformat(),
            },
            "evidence": {
                "dataset_manifest": forecast.dataset_manifest_id,
                "feature_manifest": forecast.feature_manifest_id,
                "sources": list(forecast.source_ids),
                "freshness": quality,
                "warnings": warnings,
            },
            "state": _state_for(forecast, stale=stale).value,
        }
        return cls(_json_value(payload))

    def to_dict(self) -> dict:
        return _json_value(self.payload)

    def to_json(self) -> str:
        return serialize_json(self.payload)


def _require_mapping(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be an object")
    return payload


def validate_response(payload: Mapping[str, Any]) -> dict:
    """Validate the stable envelope without weakening artifact validation."""
    root = _require_mapping(payload, "response")
    if root.get("schema_version") != API_SCHEMA_VERSION:
        raise ValueError("unsupported forecast API schema")
    for key in ("forecast", "probability", "visualization", "model", "evidence", "state"):
        if key not in root:
            raise ValueError(f"missing API field: {key}")
    forecast = _require_mapping(root["forecast"], "forecast")
    probability = _require_mapping(root["probability"], "probability")
    for key in ("forecast_id", "issue_time", "target_time", "horizon", "target", "unit", "sign_convention"):
        if key not in forecast:
            raise ValueError(f"missing forecast field: {key}")
    for key in ("p10", "p25", "p50", "p75", "p90", "probability_import", "probability_export"):
        if key not in probability:
            raise ValueError(f"missing probability field: {key}")
    values = [probability[key] for key in ("p10", "p25", "p50", "p75", "p90")]
    if any(type(value) not in (int, float) or not isfinite(float(value)) for value in values):
        raise ValueError("quantiles must be finite numbers")
    if any(a > b for a, b in zip(values, values[1:])):
        raise ValueError("quantiles must be ordered")
    import_probability = probability["probability_import"]
    export_probability = probability["probability_export"]
    if (import_probability is None) != (export_probability is None):
        raise ValueError("import/export probabilities must both be present or absent")
    if import_probability is not None:
        if any(type(value) not in (int, float) or not 0 <= float(value) <= 1
               for value in (import_probability, export_probability)):
            raise ValueError("probabilities must be between zero and one")
        if abs(float(import_probability) + float(export_probability) - 1.0) > 1e-12:
            raise ValueError("import/export probabilities must sum to one")
    visualization = _require_mapping(root["visualization"], "visualization")
    histogram = _require_mapping(visualization.get("histogram"), "histogram")
    HistogramPayload.from_dict(dict(histogram))
    if root["state"] not in {state.value for state in ForecastState}:
        raise ValueError("unsupported forecast state")
    return dict(root)


def serialize_response(response: ForecastResponse | Mapping[str, Any]) -> str:
    payload = response.to_dict() if isinstance(response, ForecastResponse) else dict(response)
    validate_response(payload)
    return serialize_json(payload)


# A concise alias for callers that prefer the transport-model terminology.
ForecastAPIResponse = ForecastResponse
