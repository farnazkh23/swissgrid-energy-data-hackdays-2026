import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from swissgrid_forecaster.dry_run import run_and_write


class DryRunTests(unittest.TestCase):
    def test_writes_complete_artifact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_and_write(output_dir=directory, issue_time=datetime(2026, 1, 10, tzinfo=timezone.utc),
                                   seed=31, include_ridge=False)
            names = {path.name for path in Path(directory).iterdir()}
            self.assertEqual(names, {"forecast.json", "histogram.json", "scoreboard.json",
                                     "provenance.json", "run_manifest.json"})
            self.assertEqual(json.loads((Path(directory) / "forecast.json").read_text())["forecast_id"],
                             result.forecast.forecast_id)
            self.assertEqual(json.loads((Path(directory) / "run_manifest.json").read_text())["holdout_used"], False)

