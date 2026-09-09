from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
import unittest

from swissgrid_forecaster.observation_schema import Observation

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def observation(**changes):
    fields = dict(source_id='weather', source_record_id='forecast-run/CH/hour-1',
                  revision_id='v1', revision_sequence=1,
                  event_time=T+timedelta(hours=1), valid_time=T+timedelta(hours=2),
                  publication_time=T-timedelta(minutes=2), first_received_at=T-timedelta(minutes=1),
                  normalized_at=T, known_at=T, value=12.0, unit='degC')
    fields.update(changes)
    return Observation(**fields)


class ObservationTests(unittest.TestCase):
    def test_future_event_and_valid_time_allowed(self):
        record = observation()
        self.assertGreater(record.event_time, record.known_at)
        self.assertGreater(record.valid_time, record.known_at)

    def test_all_clocks_reject_naive(self):
        for field in ('event_time', 'valid_time', 'publication_time', 'first_received_at',
                      'normalized_at', 'known_at'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                observation(**{field: T.replace(tzinfo=None)})

    def test_missing_known_at(self):
        with self.assertRaises(ValueError):
            observation(known_at=None)

    def test_availability_lower_bounds(self):
        for changes in (dict(known_at=T-timedelta(seconds=1)),
                        dict(publication_time=T+timedelta(seconds=1)),
                        dict(first_received_at=T+timedelta(seconds=1))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                observation(**changes)

    def test_lineage_and_immutability(self):
        old = observation()
        new = replace(old, revision_id='v2', revision_sequence=2,
                      supersedes_revision_id=old.revision_id, known_at=T+timedelta(hours=1))
        self.assertEqual(new.event_time, old.event_time)
        self.assertEqual(new.supersedes_revision_id, old.revision_id)
        with self.assertRaises(FrozenInstanceError):
            old.value = 99

    def test_unknown_publication_is_allowed(self):
        self.assertIsNone(observation(publication_time=None).publication_time)
