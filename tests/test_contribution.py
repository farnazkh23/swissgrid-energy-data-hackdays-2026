import unittest
from datetime import timedelta

from swissgrid_forecaster.contribution import ContributionRecord


H = timedelta(hours=1)


def record(**changes):
    fields = dict(baseline_model_id="champion", component_id="tail-1", component_kind="specialist",
                  metric_name="rmse", metric_delta=.2, oof_fold_ids=("fold-1",), horizon=H)
    fields.update(changes)
    return ContributionRecord(**fields)


class ContributionTests(unittest.TestCase):
    def test_specialist_contribution_is_explicit(self):
        value = record(regime="high-flow", dependency_group="tail", delta_uncertainty=.05,
                       keep_drop="KEEP", keep_drop_evidence=("stable OOF gain",))
        self.assertEqual(value.specialist_id, "tail-1")
        self.assertEqual(value.dependency_group, "tail")

    def test_model_contribution_has_no_specialist_alias(self):
        self.assertIsNone(record(component_id="ridge", component_kind="model").specialist_id)

    def test_in_sample_and_bad_uncertainty_fail(self):
        with self.assertRaises(ValueError):
            record(evaluation_scope="IN_SAMPLE")
        with self.assertRaises(ValueError):
            record(delta_uncertainty=float("inf"))

    def test_keep_drop_and_fold_validation(self):
        with self.assertRaises(ValueError):
            record(keep_drop="DROP")
        with self.assertRaises(ValueError):
            record(oof_fold_ids=("fold-1", "fold-1"))

    def test_horizon_and_component_kind_are_required(self):
        with self.assertRaises(ValueError):
            record(horizon=timedelta(0))
        with self.assertRaises(ValueError):
            record(component_kind="weight")
