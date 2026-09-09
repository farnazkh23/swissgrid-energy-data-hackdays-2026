"""Read-only repository for persisted forecast artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .forecast_output import ForecastOutput
from .histogram import HistogramPayload


ARTIFACT_NAMES = ("forecast.json", "histogram.json", "scoreboard.json",
                  "provenance.json", "run_manifest.json")


class ArtifactRepositoryError(ValueError):
    """Base error for an unreadable or invalid artifact bundle."""


class MissingArtifactError(ArtifactRepositoryError):
    """A required artifact does not exist."""


class MalformedArtifactError(ArtifactRepositoryError):
    """An artifact exists but is not valid JSON or the expected shape."""


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    forecast: ForecastOutput
    histogram: HistogramPayload
    scoreboard: dict
    provenance: dict
    run_manifest: dict


class ArtifactRepository:
    """Load artifacts without recomputing, mutating, or normalizing their data."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _read_json(self, filename: str) -> Any:
        path = self.root / filename
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise MissingArtifactError(f"missing artifact: {filename}") from exc
        except UnicodeError as exc:
            raise MalformedArtifactError(f"malformed text artifact: {filename}") from exc
        except OSError as exc:
            raise ArtifactRepositoryError(f"cannot read artifact: {filename}") from exc
        try:
            def reject_nonstandard_number(value):
                raise ValueError(f"non-finite JSON number: {value}")

            value = json.loads(raw, parse_constant=reject_nonstandard_number)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MalformedArtifactError(f"malformed JSON artifact: {filename}") from exc
        if not isinstance(value, dict):
            raise MalformedArtifactError(f"artifact must contain a JSON object: {filename}")
        return value

    def load_forecast(self) -> ForecastOutput:
        payload = self._read_json("forecast.json")
        try:
            return ForecastOutput.from_dict(payload)
        except Exception as exc:
            raise MalformedArtifactError("malformed forecast artifact") from exc

    def load_histogram(self) -> HistogramPayload:
        payload = self._read_json("histogram.json")
        try:
            return HistogramPayload.from_dict(payload)
        except Exception as exc:
            raise MalformedArtifactError("malformed histogram artifact") from exc

    def load_scoreboard(self) -> dict:
        payload = self._read_json("scoreboard.json")
        candidates = payload.get("candidates")
        selected = payload.get("selected_model")
        if not isinstance(candidates, dict) or not isinstance(selected, list) or len(selected) != 2:
            raise MalformedArtifactError("malformed scoreboard artifact")
        for model_id, candidate in candidates.items():
            if not isinstance(model_id, str) or not isinstance(candidate, dict):
                raise MalformedArtifactError("malformed scoreboard candidate")
            if "metrics" in candidate and not isinstance(candidate["metrics"], dict):
                raise MalformedArtifactError("malformed scoreboard metrics")
        return payload

    def load_provenance(self) -> dict:
        return self._read_json("provenance.json")

    def load_run_manifest(self) -> dict:
        payload = self._read_json("run_manifest.json")
        if "artifacts" in payload and not isinstance(payload["artifacts"], list):
            raise MalformedArtifactError("malformed run manifest artifact list")
        return payload

    def load_bundle(self) -> ArtifactBundle:
        bundle = ArtifactBundle(self.load_forecast(), self.load_histogram(),
                                self.load_scoreboard(), self.load_provenance(),
                                self.load_run_manifest())
        self._validate_identity(bundle)
        return bundle

    @staticmethod
    def _validate_identity(bundle: ArtifactBundle) -> None:
        selected = bundle.scoreboard.get("selected_model")
        expected = [bundle.forecast.selected_model_id, bundle.forecast.selected_model_version]
        if selected != expected:
            raise MalformedArtifactError("scoreboard selection does not match forecast model")
        provenance = bundle.provenance
        for key, expected_value in (("dataset_manifest_id", bundle.forecast.dataset_manifest_id),
                                    ("feature_manifest_id", bundle.forecast.feature_manifest_id),
                                    ("model_id", bundle.forecast.selected_model_id),
                                    ("model_version", bundle.forecast.selected_model_version)):
            if key in provenance and provenance[key] != expected_value:
                raise MalformedArtifactError(f"provenance mismatch: {key}")

    def load_forecast_by_id(self, forecast_id: str) -> ArtifactBundle:
        bundle = self.load_bundle()
        if bundle.forecast.forecast_id != forecast_id:
            raise MissingArtifactError(f"forecast not found: {forecast_id}")
        return bundle

    # Friendly aliases used by framework adapters and callers.
    latest = load_bundle
    by_id = load_forecast_by_id
