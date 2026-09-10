import json
import unittest
from datetime import datetime, timezone

from swissgrid_forecaster.api_models import (API_SCHEMA_VERSION, ForecastState,
                                             ForecastResponse, serialize_response,
                                             validate_response)
from swissgrid_forecaster.artifact_repository import ArtifactRepository


ROOT = "artifacts/mock_run"


class ApiModelTests(unittest.TestCase):
    def bundle(self):
        return ArtifactRepository(ROOT).load_bundle()

    def response(self, *, stale=False, stale_warning=None):
        bundle = self.bundle()
        return ForecastResponse.from_artifacts(
            bundle.forecast, bundle.histogram, bundle.scoreboard, bundle.provenance,
            stale=stale, stale_warning=stale_warning)

    def test_versioned_contract_contains_required_sections(self):
        payload = self.response().to_dict()
        self.assertEqual(payload["schema_version"], API_SCHEMA_VERSION)
        self.assertEqual(set(payload), {"schema_version", "forecast", "probability",
                                        "visualization", "model", "evidence", "state"})
        self.assertEqual(payload["forecast"]["target"], {"name": "net_position", "entity": "AT"})
        self.assertEqual(payload["model"]["selected_champion"], "seasonal_persistence")
        self.assertEqual(payload["probability"]["probability_import"] +
                         payload["probability"]["probability_export"], 1.0)

    def test_probability_validation(self):
        payload = self.response().to_dict()
        payload["probability"]["probability_import"] = 1.2
        with self.assertRaises(ValueError):
            validate_response(payload)
        payload = self.response().to_dict()
        payload["probability"]["p25"] = payload["probability"]["p10"] - 1
        with self.assertRaises(ValueError):
            validate_response(payload)

    def test_deterministic_serialization(self):
        response = self.response()
        first = serialize_response(response)
        second = serialize_response(ForecastResponse(response.to_dict()))
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), response.to_dict())

    def test_state_mapping(self):
        bundle = self.bundle()
        fallback = dict(bundle.forecast.to_dict())
        fallback["fallback_abstention_state"] = {"fallback_used": True, "abstained": False}
        from swissgrid_forecaster.forecast_output import ForecastOutput
        forecast = ForecastOutput.from_dict(fallback)
        response = ForecastResponse.from_artifacts(forecast, bundle.histogram,
                                                    bundle.scoreboard, bundle.provenance)
        self.assertEqual(response.to_dict()["state"], ForecastState.FALLBACK.value)

        abstain = dict(fallback)
        abstain["fallback_abstention_state"] = {"fallback_used": True, "abstained": True}
        response = ForecastResponse.from_artifacts(ForecastOutput.from_dict(abstain),
                                                    bundle.histogram, bundle.scoreboard,
                                                    bundle.provenance)
        self.assertEqual(response.to_dict()["state"], ForecastState.ABSTAIN.value)
