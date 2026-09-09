from dataclasses import replace
from datetime import timedelta
import unittest
from swissgrid_forecaster.mock_sources import mock_contract
from swissgrid_forecaster.source_contracts import Domain, HorizonAvailability


class SourceContractTests(unittest.TestCase):
    def test_all_domains(self):
        self.assertEqual(len(Domain), 14)
        for domain in Domain:
            self.assertEqual(mock_contract(domain).domain, domain)

    def test_invalid_metadata(self):
        for changes in (dict(timezone='invalid/zone'), dict(units=()), dict(units=('MW', 'MW')),
                        dict(units='MW'), dict(update_cadence=timedelta(0)),
                        dict(publication_lag=timedelta(seconds=-1)), dict(provider=''),
                        dict(domain='unknown'), dict(horizon_availability='unknown')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(mock_contract(), **changes)

    def test_units_exact_and_configurable(self):
        source = mock_contract(unit='custom-unit')
        source.validate_unit('custom-unit')
        with self.assertRaises(ValueError):
            source.validate_unit('MW')

    def test_unknown_availability_fails_closed(self):
        for name in ('update_cadence', 'publication_lag', 'horizon_availability'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                replace(mock_contract(), **{name: None}).require_forecast_metadata()

    def test_reversed_horizons(self):
        with self.assertRaises(ValueError):
            HorizonAvailability(timedelta(hours=1), timedelta(0))
