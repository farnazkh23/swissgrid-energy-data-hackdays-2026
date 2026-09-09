from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from swissgrid_forecaster.dataset_contracts import DatasetContract
from swissgrid_forecaster.mock_sources import MockAdapter, mock_contract, mock_record
from swissgrid_forecaster.source_contracts import HorizonAvailability
from swissgrid_forecaster.source_registry import SourceRegistry

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class DatasetContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = SourceRegistry([mock_contract()])
        self.dataset = DatasetContract('test', ('mock-load',))
        self.row = MockAdapter().normalize(mock_record(at=T), normalized_at=T)

    def test_asof_revisions_no_future_leak(self):
        revised = replace(self.row, revision_id='r1', revision_sequence=1,
                          supersedes_revision_id='r0', known_at=T+timedelta(hours=1), value=2)
        self.assertEqual(self.dataset.as_of([revised, self.row, self.row], self.registry, T), (self.row,))
        self.assertEqual(self.dataset.as_of([self.row, revised], self.registry, T+timedelta(hours=1)), (revised,))

    def test_conflict_fails(self):
        with self.assertRaises(ValueError):
            self.dataset.as_of([self.row, replace(self.row, value=3)], self.registry, T)

    def test_missing_metadata_even_empty(self):
        registry = SourceRegistry([replace(mock_contract(), horizon_availability=None)])
        with self.assertRaises(ValueError):
            self.dataset.as_of([], registry, T)

    def test_invalid_membership_and_records(self):
        for row in (None, replace(self.row, source_id='unknown'), replace(self.row, unit='MWh')):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.dataset.as_of([row], self.registry, T)

    def test_horizon_boundary(self):
        registry = SourceRegistry([replace(mock_contract(), horizon_availability=HorizonAvailability(timedelta(0), timedelta(0)))])
        self.assertEqual(self.dataset.as_of([self.row], registry, T), (self.row,))
        self.assertEqual(self.dataset.as_of([self.row], registry, T+timedelta(seconds=1)), ())

    def test_invalid_contract(self):
        for ids in ((), ('x', 'x'), 'x'):
            with self.assertRaises(ValueError):
                DatasetContract('test', ids)
