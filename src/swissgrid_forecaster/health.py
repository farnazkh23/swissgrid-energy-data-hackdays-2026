"""Read-only health information for the forecast artifact repository."""

from __future__ import annotations

from .api_models import API_SCHEMA_VERSION
from .artifact_repository import ArtifactRepository, ArtifactRepositoryError


def check_health(repository: ArtifactRepository) -> dict:
    """Return a deterministic health payload; never expose a traceback."""
    try:
        bundle = repository.load_bundle()
    except ArtifactRepositoryError as exc:
        return {
            "schema_version": API_SCHEMA_VERSION,
            "status": "degraded",
            "artifacts": "unavailable",
            "detail": str(exc),
        }
    return {
        "schema_version": API_SCHEMA_VERSION,
        "status": "ok",
        "artifacts": "available",
        "latest_forecast_id": bundle.forecast.forecast_id,
    }


health = check_health
