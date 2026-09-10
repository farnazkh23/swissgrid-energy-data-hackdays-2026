import unittest
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from swissgrid_forecaster.databricks_scoring import (
    OFFICIAL_SAMPLE_COUNT,
    build_official_prediction_records,
    official_scorer_available,
)


TARGETS = ("CH-AT", "CH-DE", "CH-FR", "CH-IT")


def forecast(timestamp, target, samples=None):
    return SimpleNamespace(
        target_time=timestamp,
        target_name=target,
        samples=(1,) * OFFICIAL_SAMPLE_COUNT if samples is None else samples,
    )


class DatabricksScoringTests(unittest.TestCase):
    def rows(self, count=2):
        return [
            forecast(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index), target)
            for index in range(count)
            for target in TARGETS
        ]

    def test_internal_outputs_become_five_column_records(self):
        records = build_official_prediction_records(self.rows(), target_columns=TARGETS)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[0]), 5)
        self.assertEqual(records[0][1:], ((1,) * OFFICIAL_SAMPLE_COUNT,) * 4)

    def test_writer_rejects_missing_target(self):
        rows = self.rows(1)[:-1]
        with self.assertRaisesRegex(ValueError, "missing targets"):
            build_official_prediction_records(rows, target_columns=TARGETS)

    def test_writer_rejects_wrong_sample_count(self):
        rows = self.rows(1)
        rows[0].samples = (1,)
        with self.assertRaisesRegex(ValueError, "exactly 300"):
            build_official_prediction_records(rows, target_columns=TARGETS)

    def test_exact_rounding_policy_can_be_requested(self):
        rows = self.rows(1)
        rows[0].samples = (1.5,) * OFFICIAL_SAMPLE_COUNT
        with self.assertRaisesRegex(ValueError, "non-integer"):
            build_official_prediction_records(rows, target_columns=TARGETS, rounding="exact")

    @unittest.skipUnless(
        official_scorer_available(),
        "Databricks edh2026.local_scoring is unavailable outside a Databricks runtime",
    )
    def test_official_runtime_probe(self):
        self.assertTrue(official_scorer_available())

