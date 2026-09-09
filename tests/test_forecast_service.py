import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from swissgrid_forecaster.artifact_repository import ArtifactRepository
from swissgrid_forecaster.forecast_service import ForecastService


ROOT = Path(__file__).parents[1] / "artifacts" / "mock_run"
ISSUE = datetime(2026, 1, 10, tzinfo=timezone.utc)


class ForecastServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        for artifact in ROOT.glob("*.json"):
            (self.path / artifact.name).write_bytes(artifact.read_bytes())
        self.repository = ArtifactRepository(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def service(self, now=ISSUE + timedelta(hours=1)):
        return ForecastService(self.repository, now=now, max_forecast_age=timedelta(hours=24))

    def test_latest_and_id_endpoints_are_artifact_backed(self):
        forecast_id = self.repository.load_forecast().forecast_id
        latest = self.service().get_latest_forecast()
        by_id = self.service().get_forecast(forecast_id)
        self.assertEqual(latest, by_id)
        self.assertEqual(latest["forecast"]["forecast_id"], forecast_id)
        self.assertEqual(self.service().get_histogram_latest()["histogram"]["total_probability"], 1.0)

    def test_stale_forecast_is_degraded_and_values_are_unchanged(self):
        payload = self.service(now=ISSUE + timedelta(days=2)).get_latest_forecast()
        self.assertEqual(payload["state"], "DEGRADED")
        self.assertIn("forecast artifact is stale", payload["evidence"]["warnings"])
        self.assertEqual(payload["probability"]["p50"], -47.992912)

    def test_fallback_state(self):
        forecast_path = self.path / "forecast.json"
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        forecast["fallback_abstention_state"]["fallback_used"] = True
        forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
        self.assertEqual(self.service().get_latest_forecast()["state"], "FALLBACK")

    def test_provenance_endpoint_preserves_artifact(self):
        provenance_path = self.path / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["operator_note"] = {"text": "preserve me"}
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        forecast_id = self.repository.load_forecast().forecast_id
        self.assertEqual(self.service().get_provenance(forecast_id)["provenance"]["operator_note"],
                         {"text": "preserve me"})

    def test_scoreboard_is_persisted(self):
        result = self.service().get_scoreboard()
        self.assertEqual(result["scoreboard"]["selected_model"],
                         ["seasonal_persistence", "213c84ff226cbefda9f3af3beb1c2e3e42a651642f015367ac28143c6930f277"])
