import json
import tempfile
import unittest
from pathlib import Path

from swissgrid_forecaster.cli import main, normalize_config


class CLITests(unittest.TestCase):
    def test_readiness_mock_and_nonzero_failure(self):
        self.assertEqual(main(["readiness", "--config", "configs/run.example.json"]), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps({"output_directory": directory}))
            self.assertNotEqual(main(["readiness", "--config", str(path)]), 0)

    def test_config_references_are_resolved_and_historical_paths_untouched(self):
        config = normalize_config("configs/run.example.json")
        self.assertIn("columns", config["column_mapping"])
        self.assertIn("target_contract", config)
        self.assertTrue(Path("docs/G0_G1_FOUNDATION.md").exists())
