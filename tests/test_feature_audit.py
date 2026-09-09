import unittest
from datetime import datetime, timezone, timedelta

from swissgrid_forecaster.feature_audit import audit_features, univariate_oof_score
from swissgrid_forecaster.splits import RollingOrigin, Sample


class FeatureAuditTests(unittest.TestCase):
    def test_statistics_and_no_correlation_only_selection(self):
        x = tuple(float(i) for i in range(20))
        y = tuple(2 * value + (value % 3) for value in x)
        report = audit_features({"x": x, "noise": tuple((i * 7) % 5 for i in range(20))}, y,
                                evaluation_id="eval-1", incremental_oof_gains={"x": .2, "noise": -.1},
                                fold_scores={"x": (1, 2, 1), "noise": (3, 5, 4)})
        self.assertEqual(len(report.records), 2)
        x_record = next(record for record in report.records if record.feature_id == "x")
        self.assertGreater(x_record.pearson, .9)
        self.assertIsNotNone(x_record.mutual_information)
        self.assertEqual(x_record.incremental_oof_gain, .2)
        self.assertEqual(report.records[0].keep_drop, "UNDECIDED")
        self.assertIn("correlation", report.selection_policy.lower())

    def test_fold_safe_univariate_score(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = tuple(Sample(str(i), base + timedelta(hours=i), base + timedelta(hours=i + 1),
                            base + timedelta(hours=i), base + timedelta(hours=i + 1), (float(i % 2),), float(i))
                     for i in range(30))
        plan = RollingOrigin(base + timedelta(hours=10), base + timedelta(hours=25),
                             timedelta(hours=8), timedelta(hours=3), timedelta(hours=1), timedelta(hours=6))
        self.assertGreaterEqual(univariate_oof_score(plan, rows, 0), 0)
