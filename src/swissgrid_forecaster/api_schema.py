"""Transport-safe API payload and JSON schema for a future frontend."""
import json

from .forecast_output import ForecastOutput
from .histogram import HistogramPayload


API_SCHEMA_VERSION = "mock-forecast.v1"

FORECAST_API_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "forecast", "histogram", "warnings", "fallback_abstention_state"],
    "properties": {
        "schema_version": {"const": API_SCHEMA_VERSION},
        "forecast": {"type": "object", "required": ["forecast_id", "issue_time", "target_time", "quantiles"]},
        "histogram": {"type": "object", "required": ["bin_edges", "bin_centers", "probabilities", "total_probability"]},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "fallback_abstention_state": {"type": "object"},
    },
}
API_SCHEMA = FORECAST_API_SCHEMA


def forecast_api_schema() -> dict:
    """Return the schema object for registration with a future transport."""
    return FORECAST_API_SCHEMA


def build_api_payload(forecast: ForecastOutput, histogram: HistogramPayload) -> dict:
    if not isinstance(forecast, ForecastOutput) or not isinstance(histogram, HistogramPayload):
        raise ValueError("validated forecast and histogram are required")
    return {"schema_version": API_SCHEMA_VERSION, "forecast": forecast.to_dict(),
            "histogram": histogram.to_dict(), "warnings": list(forecast.warnings),
            "fallback_abstention_state": forecast.fallback_abstention_state}


def validate_api_payload(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("schema_version") != API_SCHEMA_VERSION:
        raise ValueError("unsupported forecast API schema")
    for key in ("forecast", "histogram", "warnings", "fallback_abstention_state"):
        if key not in payload:
            raise ValueError(f"missing API field: {key}")
    ForecastOutput.from_dict(payload["forecast"])
    HistogramPayload.from_dict(payload["histogram"])
    if not isinstance(payload["warnings"], list) or not all(isinstance(v, str) for v in payload["warnings"]):
        raise ValueError("warnings must be strings")
    if not isinstance(payload["fallback_abstention_state"], dict):
        raise ValueError("fallback state must be an object")
    return payload


def serialize_api_payload(forecast: ForecastOutput, histogram: HistogramPayload) -> str:
    return json.dumps(build_api_payload(forecast, histogram), sort_keys=True, separators=(",", ":"), allow_nan=False)
