from datetime import datetime, timedelta, timezone
import unittest
from swissgrid_forecaster.mock_sources import MockAdapter, mock_contract, mock_record
from swissgrid_forecaster.source_contracts import Domain

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class MockSourceTests(unittest.TestCase):
    def test_deterministic_for_every_domain(self):
        for domain in Domain:
            with self.subTest(domain=domain):
                first, second = mock_record(at=T, domain=domain), mock_record(at=T, domain=domain)
                self.assertEqual(first, second)
                self.assertEqual(first.source_id, mock_contract(domain).source_id)
                self.assertEqual(MockAdapter().normalize(first, normalized_at=T), MockAdapter().normalize(second, normalized_at=T))

    def test_offset_normalization(self):
        self.assertEqual(mock_record(at=T), mock_record(at=T.astimezone(timezone(timedelta(hours=2)))))

    def test_revision_lineage(self):
        row = MockAdapter().normalize(mock_record(at=T, revision_sequence=3), normalized_at=T)
        self.assertEqual((row.revision_id, row.supersedes_revision_id, row.revision_sequence), ('r3', 'r2', 3))

    def test_bad_revision(self):
        with self.assertRaises(ValueError):
            mock_record(at=T, revision_sequence=-1)
