import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from swissgrid_forecaster.acceptance import AcceptanceStatus, evaluate_acceptance
from swissgrid_forecaster.run_manifest import build_run_manifest, file_hash, write_run_manifest


class AcceptanceTests(unittest.TestCase):
    def bundle(self):
        directory = Path(tempfile.mkdtemp())
        source = Path("artifacts/mock_run")
        for path in source.iterdir():
            shutil.copy2(path, directory / path.name)
        self.refresh_manifest(directory)
        return directory

    def refresh_manifest(self, directory):
        digest = "c" * 64
        artifacts = {path.name: file_hash(path) for path in directory.glob("*.json") if path.name != "run_manifest.json"}
        manifest = build_run_manifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), git_revision="d" * 40,
                                      config_hashes={"run": digest}, dataset_hash=digest, source_manifests=(),
                                      target_contract_hash=digest, feature_manifest_hash=digest,
                                      model_identities=(), fit_cutoffs=(), fold_ids=(), random_seed=1,
                                      artifact_hashes=artifacts, warnings=(), readiness_result={})
        write_run_manifest(manifest, directory / "run_manifest.json")

    def tearDown(self):
        for directory in getattr(self, "directories", ()):
            shutil.rmtree(directory, ignore_errors=True)

    def test_valid_bundle_passes_and_fallback_requires_permission(self):
        directory = self.bundle(); self.directories = (directory,)
        self.assertEqual(evaluate_acceptance(directory).status, AcceptanceStatus.PASS)
        forecast = json.loads((directory / "forecast.json").read_text())
        forecast["fallback_abstention_state"]["fallback_used"] = True
        (directory / "forecast.json").write_text(json.dumps(forecast))
        self.refresh_manifest(directory)
        self.assertEqual(evaluate_acceptance(directory).status, AcceptanceStatus.FAIL)
        self.assertEqual(evaluate_acceptance(directory, allow_fallback=True).status,
                         AcceptanceStatus.PASS_WITH_WARNINGS)

    def test_corruption_missing_provenance_and_histogram_mismatch_fail(self):
        directory = self.bundle(); self.directories = (directory,)
        (directory / "provenance.json").unlink()
        self.assertEqual(evaluate_acceptance(directory).status, AcceptanceStatus.FAIL)
        directory = self.bundle(); self.directories += (directory,)
        histogram = json.loads((directory / "histogram.json").read_text())
        histogram["total_probability"] = 0.5
        (directory / "histogram.json").write_text(json.dumps(histogram))
        self.assertEqual(evaluate_acceptance(directory).status, AcceptanceStatus.FAIL)
