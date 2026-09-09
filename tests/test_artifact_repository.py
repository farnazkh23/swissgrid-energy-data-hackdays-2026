import json
from pathlib import Path
import tempfile
import unittest

from swissgrid_forecaster.artifact_repository import (ArtifactRepository, MalformedArtifactError,
                                                      MissingArtifactError)


ROOT = Path(__file__).parents[1] / "artifacts" / "mock_run"


class ArtifactRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        for artifact in ROOT.glob("*.json"):
            (self.path / artifact.name).write_bytes(artifact.read_bytes())
        self.repository = ArtifactRepository(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_loads_existing_bundle_without_recomputation(self):
        bundle = self.repository.load_bundle()
        self.assertEqual(bundle.forecast.forecast_id,
                         "15f286c5b75a1ee8823d4982b632c34f21f9883a0c312304daf0b420b02cba0f")
        self.assertEqual(bundle.histogram.total_probability, 1.0)

    def test_missing_artifact(self):
        (self.path / "histogram.json").unlink()
        with self.assertRaises(MissingArtifactError):
            self.repository.load_bundle()

    def test_malformed_artifact_rejected(self):
        (self.path / "forecast.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(MalformedArtifactError):
            self.repository.load_bundle()

    def test_nonfinite_json_rejected(self):
        (self.path / "histogram.json").write_text('{"total_probability": NaN}', encoding="utf-8")
        with self.assertRaises(MalformedArtifactError):
            self.repository.load_bundle()

    def test_wrong_forecast_id_is_not_found(self):
        with self.assertRaises(MissingArtifactError):
            self.repository.load_forecast_by_id("does-not-exist")

    def test_provenance_extra_fields_are_preserved(self):
        provenance_path = self.path / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["future_receipt"] = {"source": "operator", "value": 7}
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
        self.assertEqual(self.repository.load_bundle().provenance["future_receipt"],
                         {"source": "operator", "value": 7})
