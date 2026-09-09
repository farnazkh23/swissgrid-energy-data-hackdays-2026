from pathlib import Path
import tempfile
import unittest

from swissgrid_forecaster.artifact_repository import ArtifactRepository
from swissgrid_forecaster.health import check_health


ROOT = Path(__file__).parents[1] / "artifacts" / "mock_run"


class HealthTests(unittest.TestCase):
    def test_available_repository(self):
        result = check_health(ArtifactRepository(ROOT))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["artifacts"], "available")

    def test_missing_repository_is_degraded(self):
        with tempfile.TemporaryDirectory() as directory:
            result = check_health(ArtifactRepository(directory))
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["artifacts"], "unavailable")
