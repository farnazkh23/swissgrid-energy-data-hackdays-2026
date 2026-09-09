from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from swissgrid_forecaster.target_config import TargetConfig
from swissgrid_forecaster.target_contract import TargetContract, ForecastType
from swissgrid_forecaster.mock_sources import mock_contract
from swissgrid_forecaster.source_registry import SourceRegistry


class TargetConfigTests(unittest.TestCase):
    def config(self, **changes):
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        contract = TargetContract('explicit-test-target', 'test-entity', 'MW', 'test-sign',
                                  timedelta(hours=1), t, t+timedelta(hours=2), timedelta(hours=2),
                                  'test-aggregation', 'test-label-rule', ForecastType.POINT, ('test-metric',))
        return TargetConfig(replace(contract, **changes), 'mock-load', 'synthetic-scalar')

    def test_reuses_explicit_contract(self):
        config = self.config()
        config.validate(SourceRegistry([mock_contract()]))
        self.assertEqual(config.contract.sign_convention, 'test-sign')
        self.assertIsNone(config.contract.country)

    def test_unit_mismatch(self):
        with self.assertRaises(ValueError):
            self.config(unit='MWh').validate(SourceRegistry([mock_contract()]))

    def test_no_implicit_target(self):
        with self.assertRaises(ValueError):
            TargetConfig(None, 'mock-load', 'record')
