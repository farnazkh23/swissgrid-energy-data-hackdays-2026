import unittest
from datetime import timedelta
from dataclasses import replace

from swissgrid_forecaster.ablation import AblationRecord


H = timedelta(hours=1)


def record(**changes):
    fields = dict(baseline_model_id="champion", removed_component_id="weather-1",
                  removed_component_kind="specialist", metric_name="mae", metric_delta=.4,
                  oof_fold_ids=("fold-1", "fold-2"), horizon=H)
    fields.update(changes)
    return AblationRecord(**fields)


class AblationTests(unittest.TestCase):
    def test_records_oof_comparison_and_optional_context(self):
        value = record(delta_uncertainty=.1, regime="winter", dependency_group="weather",
                       keep_drop="KEEP", keep_drop_evidence=("positive across folds",))
        self.assertEqual(value.removed_model_or_specialist, "weather-1")
        self.assertEqual(value.oof_fold_ids, ("fold-1", "fold-2"))

    def test_in_sample_claims_are_forbidden(self):
        with self.assertRaises(ValueError):
            record(evaluation_scope="IN_SAMPLE")

    def test_keep_drop_requires_evidence(self):
        with self.assertRaises(ValueError):
            record(keep_drop="DROP")
        with self.assertRaises(ValueError):
            record(keep_drop="MAYBE")

    def test_invalid_deltas_and_folds_fail(self):
        for change in (dict(metric_delta=float("nan")), dict(delta_uncertainty=-.1),
                       dict(oof_fold_ids=()), dict(oof_fold_ids=("fold-1", "fold-1"))):
            with self.assertRaises(ValueError):
                record(**change)

    def test_component_kind_is_explicit(self):
        with self.assertRaises(ValueError):
            record(removed_component_kind="ensemble_weight")
        self.assertEqual(replace(record(), removed_component_kind="model").removed_component_kind, "model")
