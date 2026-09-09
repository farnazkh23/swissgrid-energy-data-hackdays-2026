from dataclasses import replace
from datetime import datetime, timedelta, timezone
import tempfile
import unittest
from swissgrid_forecaster.ingestion import ingest
from swissgrid_forecaster.mock_sources import MockAdapter, mock_contract, mock_record
from swissgrid_forecaster.raw_store import RawStore
from swissgrid_forecaster.source_registry import SourceRegistry

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = RawStore(self.tmp.name)
        self.registry = SourceRegistry([mock_contract()])

    def run_ingest(self, received=None, adapter=None, **kwargs):
        return ingest(received or mock_record(at=T), adapter or MockAdapter(), self.registry,
                      self.store, normalized_at=T + timedelta(seconds=1), **kwargs)

    def test_preserves_evidence_and_clocks(self):
        received = mock_record(at=T, revision_sequence=1)
        result = self.run_ingest(received, for_forecasting=True)
        row = result.observation
        self.assertEqual(self.store.read(row.raw_sha256), received.payload)
        self.assertEqual(row.event_time, T)
        self.assertEqual(row.valid_time, T)
        self.assertEqual(row.publication_time, T)
        self.assertEqual(row.first_received_at, T)
        self.assertEqual(row.normalized_at, T + timedelta(seconds=1))
        self.assertEqual(row.known_at, row.normalized_at)
        self.assertEqual(row.supersedes_revision_id, 'r0')
        self.assertEqual(row.source_id, received.source_id)
        self.assertEqual(row.revision_id, received.revision_id)

    def test_repeated_payload_preserves_receipts(self):
        first, second = self.run_ingest(), self.run_ingest()
        self.assertEqual(first.observation, second.observation)
        self.assertTrue(second.raw.duplicate_payload)
        self.assertEqual(len(self.store.manifests()), 2)

    def test_wrong_unit_retains_raw(self):
        with self.assertRaises(ValueError):
            self.run_ingest(mock_record(at=T, unit='MWh'))
        self.assertEqual(len(self.store.manifests()), 1)

    def test_adapter_cannot_change_provenance(self):
        for changes in (dict(source_id='other'), dict(source_record_id='other'),
                        dict(revision_id='other'), dict(first_received_at=T-timedelta(seconds=1)),
                        dict(normalized_at=T), dict(raw_sha256='0'*64)):
            class BadAdapter:
                def normalize(self, received, *, normalized_at):
                    return replace(MockAdapter().normalize(received, normalized_at=normalized_at), **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_ingest(adapter=BadAdapter())

    def test_missing_availability_forecast_only(self):
        self.registry = SourceRegistry([replace(mock_contract(), publication_lag=None)])
        self.run_ingest()
        with self.assertRaises(ValueError):
            self.run_ingest(for_forecasting=True)

    def test_invalid_clocks(self):
        with self.assertRaises(ValueError):
            replace(mock_record(at=T), first_received_at=T.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            ingest(mock_record(at=T), MockAdapter(), self.registry, self.store,
                   normalized_at=T-timedelta(seconds=1))
