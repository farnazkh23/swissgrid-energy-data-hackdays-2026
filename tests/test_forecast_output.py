import unittest
from datetime import datetime, timezone, timedelta

from swissgrid_forecaster.forecast_output import (ForecastOutput, probabilities_from_samples,
                                                  repair_quantiles)


class ForecastOutputTests(unittest.TestCase):
    def output(self):
        return ForecastOutput.from_samples(
            forecast_id="f1", issue_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            horizon=timedelta(hours=1), target_name="net_position", target_entity="CH",
            unit="MW", selected_model_id="persistence", selected_model_version="a" * 64,
            feature_manifest_hash="b" * 64, training_data_manifest_hash="c" * 64,
            fit_cutoff=datetime(2026, 1, 1, tzinfo=timezone.utc), samples=(-2, -1, 1, 3),
            source_ids=("mock-net_position",), dataset_manifest_id="d" * 64,
            feature_manifest_id="e" * 64, fold_evaluation_summary={"holdout_used": False})

    def test_quantile_ordering_and_probabilities(self):
        output = self.output()
        self.assertLessEqual(output.p10, output.p25)
        self.assertLessEqual(output.p25, output.p50)
        self.assertEqual(probabilities_from_samples(output.samples), (.5, .5))
        self.assertEqual(output.probability_import + output.probability_export, 1.0)

    def test_crossing_repair_is_deterministic(self):
        self.assertEqual(repair_quantiles((3, 1, 2, 4, 0)), ((3, 3, 3, 4, 4), True))
        self.assertEqual(repair_quantiles((3, 1, 2, 4, 0)), repair_quantiles((3, 1, 2, 4, 0)))

    def test_forecast_json_round_trip(self):
        output = self.output()
        self.assertEqual(ForecastOutput.from_dict(output.to_dict()), output)

    def test_probabilities_require_samples(self):
        with self.assertRaises(ValueError):
            probabilities_from_samples(())

