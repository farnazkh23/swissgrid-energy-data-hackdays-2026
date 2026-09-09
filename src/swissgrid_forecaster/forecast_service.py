"""Serving orchestration over immutable forecast artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

from .api_models import API_SCHEMA_VERSION, ForecastResponse, serialize_json
from .artifact_repository import ArtifactBundle, ArtifactRepository


class ForecastService:
    """Expose API-shaped data while keeping model execution out of serving."""

    def __init__(self, repository: ArtifactRepository, *, now: Callable[[], datetime] | datetime | None = None,
                 clock: Callable[[], datetime] | datetime | None = None,
                 max_forecast_age: timedelta = timedelta(hours=24)):
        if not isinstance(repository, ArtifactRepository):
            raise TypeError("repository must be an ArtifactRepository")
        if not isinstance(max_forecast_age, timedelta) or max_forecast_age <= timedelta(0):
            raise ValueError("max_forecast_age must be positive")
        if now is not None and clock is not None:
            raise ValueError("provide only one of now or clock")
        now = clock if clock is not None else now
        self.repository = repository
        self.max_forecast_age = max_forecast_age
        if now is None:
            self._now = lambda: datetime.now(timezone.utc)
        elif isinstance(now, datetime):
            fixed = now
            self._now = lambda: fixed
        elif callable(now):
            self._now = now
        else:
            raise TypeError("now must be a datetime or callable")

    def _bundle_response(self, bundle: ArtifactBundle) -> dict:
        current = self._now()
        if not isinstance(current, datetime):
            raise ValueError("clock must return a datetime")
        current = current.astimezone(timezone.utc) if current.tzinfo else None
        if current is None:
            raise ValueError("clock must return a timezone-aware datetime")
        age = current - bundle.forecast.issue_time
        stale = age > self.max_forecast_age
        response = ForecastResponse.from_artifacts(
            bundle.forecast, bundle.histogram, bundle.scoreboard, bundle.provenance,
            stale=stale,
            stale_warning="forecast artifact is stale" if stale else None,
        )
        return response.to_dict()

    def get_latest_forecast(self) -> dict:
        return self._bundle_response(self.repository.load_bundle())

    def get_forecast(self, forecast_id: str) -> dict:
        return self._bundle_response(self.repository.load_forecast_by_id(forecast_id))

    def get_scoreboard(self) -> dict:
        bundle = self.repository.load_bundle()
        return {"schema_version": API_SCHEMA_VERSION, "scoreboard": bundle.scoreboard}

    def get_histogram_latest(self) -> dict:
        bundle = self.repository.load_bundle()
        return {"schema_version": API_SCHEMA_VERSION, "forecast_id": bundle.forecast.forecast_id,
                "histogram": bundle.histogram.to_dict()}

    def get_provenance(self, forecast_id: str) -> dict:
        bundle = self.repository.load_forecast_by_id(forecast_id)
        return {"schema_version": API_SCHEMA_VERSION, "forecast_id": forecast_id,
                "provenance": bundle.provenance}

    def serialize_latest_forecast(self) -> str:
        return serialize_json(self.get_latest_forecast())

    # Endpoint-oriented aliases keep the framework adapter thin.
    latest = get_latest_forecast
    forecast = get_forecast
    scoreboard = get_scoreboard
    histogram_latest = get_histogram_latest
    provenance = get_provenance
