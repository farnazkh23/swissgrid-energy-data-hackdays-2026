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

    def test_keep_drop_requires_explicit_multi_signal_policy(self):
        report = audit_features(
            {"keep_me": (1, 2, 3, 4), "drop_me": (4, 3, 2, 1)},
            (1, 2, 3, 4), evaluation_id="eval-2",
            univariate_scores={"keep_me": 1.0, "drop_me": 3.0},
            incremental_oof_gains={"keep_me": .2, "drop_me": -.1},
            fold_scores={"keep_me": (1, 1, 1), "drop_me": (3, 5, 4)},
            recommendation_policy={"min_incremental_oof_gain": 0,
                                   "min_fold_stability": .5,
                                   "max_missingness": 0})
        decisions = {record.feature_id: record.keep_drop for record in report.records}
        self.assertEqual(decisions, {"keep_me": "KEEP", "drop_me": "DROP"})
