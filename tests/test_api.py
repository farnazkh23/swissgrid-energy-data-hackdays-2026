from pathlib import Path
import unittest

from swissgrid_forecaster.api import create_app
from swissgrid_forecaster.artifact_repository import ArtifactRepository
from swissgrid_forecaster.forecast_service import ForecastService


class ApiTests(unittest.TestCase):
    def test_fastapi_is_optional_and_no_server_starts(self):
        service = ForecastService(ArtifactRepository(Path("artifacts/mock_run")))
        try:
            import fastapi  # noqa: F401
        except ImportError:
            with self.assertRaises(RuntimeError):
                create_app(service)
        else:
            app = create_app(service)
            paths = {route.path for route in app.routes}
            self.assertTrue({"/health", "/forecast/latest", "/scoreboard",
                             "/histogram/latest"}.issubset(paths))
            self.assertIn("/forecast/{forecast_id}", paths)
            self.assertIn("/provenance/{forecast_id}", paths)
