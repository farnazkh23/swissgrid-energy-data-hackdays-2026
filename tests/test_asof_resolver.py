from dataclasses import replace
from datetime import timedelta
from itertools import permutations
import unittest
from swissgrid_forecaster.asof_resolver import resolve_as_of, resolve_sources_as_of
from test_observation_schema import T, observation


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.old = observation()
        self.new = replace(self.old, revision_id='v2', revision_sequence=2,
                           supersedes_revision_id='v1', known_at=T+timedelta(hours=1), value=13)

    def test_later_revision_invisible_and_boundary(self):
        self.assertEqual(resolve_as_of([self.new, self.old], T), (self.old,))
        self.assertEqual(resolve_as_of([self.old, self.new], self.new.known_at), (self.new,))

    def test_future_valid_forecast_is_available(self):
        self.assertEqual(resolve_as_of([self.old], T), (self.old,))

    def test_after_issue_excluded(self):
        self.assertEqual(resolve_as_of([self.new], T), ())

    def test_deterministic_across_sources_and_input_order(self):
        other = replace(self.old, source_id='outages')
        expected = (other, self.new)
        for records in permutations([self.old, self.new, other]):
            self.assertEqual(resolve_as_of(records, self.new.known_at), expected)
        self.assertEqual(resolve_sources_as_of({'weather': [self.new, self.old], 'outages': [other]},
                                              self.new.known_at), expected)

    def test_late_delivery_of_older_revision(self):
        late_old = replace(self.old, known_at=T+timedelta(hours=2))
        self.assertEqual(resolve_as_of([late_old, self.new], late_old.known_at), (self.new,))

    def test_conflicts_fail_closed_and_duplicates_idempotent(self):
        self.assertEqual(resolve_as_of([self.old, self.old], T), (self.old,))
        for conflict in (replace(self.old, value=99), replace(self.old, revision_id='different')):
            with self.assertRaises(ValueError):
                resolve_as_of([self.old, conflict], T)
        # An ineligible conflicting revision cannot affect an earlier result.
        conflict = replace(self.old, value=99, known_at=self.new.known_at)
        self.assertEqual(resolve_as_of([self.old, conflict], T), (self.old,))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            resolve_as_of([{'known_at': None}], T)
        with self.assertRaises(ValueError):
            resolve_as_of([], T.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            resolve_sources_as_of({'wrong': [self.old]}, T)
