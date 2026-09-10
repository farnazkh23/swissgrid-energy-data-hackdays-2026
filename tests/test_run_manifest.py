import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from swissgrid_forecaster.run_manifest import (build_run_manifest, canonical_hash, file_hash,
                                               write_run_manifest)


class RunManifestTests(unittest.TestCase):
    def make(self, config_hash):
        digest = "a" * 64
        return build_run_manifest(created_at=datetime(2026, 1, 1, tzinfo=timezone.utc), git_revision="b" * 40,
                                  config_hashes={"run": config_hash}, dataset_hash=digest, source_manifests=(),
                                  target_contract_hash=digest, feature_manifest_hash=digest,
                                  model_identities=({"model_id": "persistence", "version": "1"},),
                                  fit_cutoffs=(), fold_ids=("fold-1",), random_seed=7, artifact_hashes={},
                                  warnings=(), readiness_result={})

    def test_hashes_change_and_same_inputs_reuse_identity(self):
        first = self.make(canonical_hash({"x": 1}))
        same = self.make(canonical_hash({"x": 1}))
        changed = self.make(canonical_hash({"x": 2}))
        self.assertEqual(first.run_id, same.run_id)
        self.assertNotEqual(first.run_id, changed.run_id)

    def test_round_trip_and_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            manifest = self.make("a" * 64)
            write_run_manifest(manifest, path)
            self.assertEqual(file_hash(path), file_hash(path))
            from swissgrid_forecaster.run_manifest import RunManifest
            self.assertEqual(RunManifest.from_dict(__import__("json").loads(path.read_text())).run_id,
                             manifest.run_id)
