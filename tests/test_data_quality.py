from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest
from swissgrid_forecaster.data_quality import QualityFlag as F, QualityPolicy, assess, timestamp_flags
from swissgrid_forecaster.mock_sources import MockAdapter, mock_record

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class DataQualityTests(unittest.TestCase):
    def test_missing_stale_revised_duplicate_unavailable(self):
        row = MockAdapter().normalize(mock_record(at=T, value=None, revision_sequence=1),
                                      normalized_at=T+timedelta(hours=2))
        flags = assess([row, row], issue_time=T+timedelta(hours=1), policy=QualityPolicy(max_age=timedelta(0)))
        self.assertEqual(flags, (frozenset({F.MISSING, F.STALE, F.REVISED, F.DUPLICATE, F.UNAVAILABLE_AT_ISSUE}),)*2)

    def test_range_boundaries(self):
        row = MockAdapter().normalize(mock_record(at=T), normalized_at=T)
        self.assertEqual(assess([row], issue_time=T, policy=QualityPolicy(minimum=1, maximum=1)), (frozenset(),))
        self.assertIn(F.OUT_OF_RANGE, assess([replace(row, value=2)], issue_time=T, policy=QualityPolicy(maximum=1))[0])

    def test_timezone_and_missing_metadata(self):
        self.assertEqual(timestamp_flags(event_time=T.replace(tzinfo=None)), frozenset({F.TIMEZONE_ISSUE}))
        self.assertEqual(timestamp_flags(known_at=None), frozenset({F.MISSING, F.UNAVAILABLE_AT_ISSUE}))
        self.assertEqual(timestamp_flags(event_time=T), frozenset())

    def test_bad_policy(self):
        for changes in (dict(max_age=timedelta(seconds=-1)), dict(minimum=float('nan')), dict(minimum=2, maximum=1)):
            with self.assertRaises(ValueError):
                QualityPolicy(**changes)
