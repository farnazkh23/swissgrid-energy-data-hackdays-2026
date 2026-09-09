"""Optional FastAPI transport for the framework-neutral forecast service.

FastAPI is deliberately imported inside ``create_app`` so the package remains
usable in environments that only need the repository and service layers.
"""

from __future__ import annotations

from .artifact_repository import (ArtifactRepositoryError, MalformedArtifactError,
                                  MissingArtifactError)
from .forecast_service import ForecastService
from .health import check_health


def create_app(service: ForecastService | None = None, *, repository=None):
    """Create the optional FastAPI application without starting a server."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("FastAPI is not installed; use ForecastService directly") from exc

    if service is None:
        if repository is None:
            raise ValueError("service or repository is required")
        service = ForecastService(repository)

    app = FastAPI(title="Swissgrid Forecast API", version="1.0.0")

    def call(operation):
        try:
            return operation()
        except MissingArtifactError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MalformedArtifactError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ArtifactRepositoryError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health_endpoint():
        return check_health(service.repository)

    @app.get("/forecast/latest")
    def latest_forecast_endpoint():
        return call(service.get_latest_forecast)

    @app.get("/forecast/{forecast_id}")
    def forecast_endpoint(forecast_id: str):
        return call(lambda: service.get_forecast(forecast_id))

    @app.get("/scoreboard")
    def scoreboard_endpoint():
        return call(service.get_scoreboard)

    @app.get("/histogram/latest")
    def histogram_endpoint():
        return call(service.get_histogram_latest)

    @app.get("/provenance/{forecast_id}")
    def provenance_endpoint(forecast_id: str):
        return call(lambda: service.get_provenance(forecast_id))

    return app
