import unittest
from datetime import datetime, timezone, timedelta

from swissgrid_forecaster.column_mapping import ColumnMapping
from swissgrid_forecaster.real_source_adapter import PITMetadataError, RealSourceAdapter, RealSourceConfig


T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def mapping():
    return ColumnMapping.from_dict({"columns": {"source_id": "SRC", "country": "AREA", "domain": "KIND",
        "unit": "U", "value": "VAL", "event_time": "EVENT", "valid_time": "VALID", "known_at": "KNOWN",
        "first_received_at": "RECEIVED", "normalized_at": "NORMALIZED", "publication_time": "PUBLISHED",
        "revision_id": "REV", "revision_sequence": "SEQ", "supersedes_revision_id": "SUPER",
        "source_record_id": "REC", "issue_time": "ISSUE", "horizon": "HORIZON"}})


def row(**changes):
    result = {"SRC": "provider-load", "AREA": "CH", "KIND": "load", "U": "MW", "VAL": 10,
              "EVENT": T.isoformat(), "VALID": (T + timedelta(hours=1)).isoformat(), "KNOWN": T.isoformat(),
              "RECEIVED": T.isoformat(), "NORMALIZED": T.isoformat(), "PUBLISHED": T.isoformat(),
              "REV": "r1", "SEQ": 1, "SUPER": "r0", "REC": "record-1",
              "ISSUE": T.isoformat(), "HORIZON": 3600}
    result.update(changes)
    return result


def config(policy="require"):
    return RealSourceConfig("real-test", mapping(), "UTC", timedelta(hours=1), timedelta(minutes=5),
                            timedelta(0), timedelta(days=2), policy)


class RealSourceAdapterTests(unittest.TestCase):
    def test_preserves_observation_and_issue_metadata(self):
        observation, metadata = RealSourceAdapter(config()).adapt_row(row())
        self.assertEqual((observation.source_id, observation.unit, observation.revision_sequence),
                         ("provider-load", "MW", 1))
        self.assertEqual(observation.supersedes_revision_id, "r0")
        self.assertEqual(metadata.domain.value, "load")
        self.assertEqual(metadata.issue_time, T)
        self.assertEqual(metadata.horizon, timedelta(seconds=3600))

    def test_pit_metadata_is_required(self):
        missing = row(KNOWN=None, RECEIVED=None, NORMALIZED=None)
        with self.assertRaises(PITMetadataError):
            RealSourceAdapter(config()).adapt_row(missing)

    def test_documented_publication_lag_fallback(self):
        adapted = RealSourceAdapter(config("publication_plus_lag")).adapt_row(
            row(KNOWN=None, RECEIVED=None, NORMALIZED=None))[0]
        self.assertEqual(adapted.known_at, T + timedelta(minutes=5))

    def test_fallback_preserves_available_historical_known_at(self):
        adapted, metadata = RealSourceAdapter(config("publication_plus_lag")).adapt_row(
            row(KNOWN=(T + timedelta(minutes=1)).isoformat()))
        self.assertEqual(adapted.known_at, T + timedelta(minutes=1))
        self.assertFalse(metadata.known_at_reconstructed)

    def test_naive_clock_fails_closed(self):
        with self.assertRaises(ValueError):
            RealSourceAdapter(config()).adapt_row(row(EVENT="2026-01-01T00:00:00"))
