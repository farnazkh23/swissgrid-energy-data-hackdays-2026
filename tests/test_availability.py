from datetime import datetime, timedelta, timezone
import unittest
from swissgrid_forecaster.availability import earliest_known_at, is_available

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class AvailabilityTests(unittest.TestCase):
    def test_boundary_and_future(self):
        self.assertTrue(is_available(T, T))
        self.assertFalse(is_available(T+timedelta(microseconds=1), T))

    def test_invalid_clocks_fail_closed(self):
        for bad in (None, '2026-01-01', T.replace(tzinfo=None)):
            for known, issue in ((bad, T), (T, bad)):
                with self.subTest(known=known, issue=issue), self.assertRaises(ValueError):
                    is_available(known, issue)

    def test_policy_embargo_and_normalization(self):
        permitted = T+timedelta(hours=1)
        self.assertEqual(earliest_known_at(first_received_at=T, normalized_at=T,
                                          permitted_at=permitted), permitted)
        with self.assertRaises(ValueError):
            earliest_known_at(first_received_at=T, normalized_at=T-timedelta(seconds=1))
