import json
import unittest
from datetime import datetime, timezone

from swissgrid_forecaster.api_schema import (API_SCHEMA_VERSION, build_api_payload,
                                             serialize_api_payload, validate_api_payload)
from swissgrid_forecaster.pipeline import PipelineConfig, run_mock_pipeline


class ApiSchemaTests(unittest.TestCase):
    def payload(self):
        result = run_mock_pipeline(PipelineConfig(datetime(2026, 1, 10, tzinfo=timezone.utc),
                                                  seed=41, include_ridge=False))
        return build_api_payload(result.forecast, result.histogram)

    def test_exposes_frontend_contract(self):
        payload = self.payload()
        self.assertEqual(payload["schema_version"], API_SCHEMA_VERSION)
        self.assertIn("quantiles", payload["forecast"])
        self.assertIn("probability_import", payload["forecast"])
        self.assertIn("probabilities", payload["histogram"])
        self.assertIn("provenance", payload["forecast"])
        self.assertIn("fallback_abstention_state", payload)
        self.assertEqual(validate_api_payload(payload), payload)

    def test_transport_json_is_stable(self):
        payload = self.payload()
        encoded = serialize_api_payload(
            __import__("swissgrid_forecaster.forecast_output", fromlist=["ForecastOutput"]).ForecastOutput.from_dict(payload["forecast"]),
            __import__("swissgrid_forecaster.histogram", fromlist=["HistogramPayload"]).HistogramPayload.from_dict(payload["histogram"]))
        self.assertEqual(json.loads(encoded), payload)

    def test_invalid_schema_rejected(self):
        payload = self.payload()
        payload["schema_version"] = "other"
        with self.assertRaises(ValueError):
            validate_api_payload(payload)

