import unittest
from swissgrid_forecaster.mock_sources import mock_contract
from swissgrid_forecaster.source_contracts import Domain
from swissgrid_forecaster.source_registry import SourceRegistry


class SourceRegistryTests(unittest.TestCase):
    def test_lookup_and_filter(self):
        registry = SourceRegistry(mock_contract(d) for d in reversed(tuple(Domain)))
        self.assertEqual(registry.get('mock-load'), mock_contract())
        self.assertEqual(registry.list('load'), (mock_contract(),))
        self.assertEqual([s.source_id for s in registry.list()], sorted(s.source_id for s in registry.list()))

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            SourceRegistry([mock_contract(), mock_contract()])

    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):
            SourceRegistry().get('Swissgrid')
