import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from swissgrid_forecaster.real_modeling_v1 import WeeklyFold
from swissgrid_forecaster.rolling_validation import (
    HOUR, TARGETS, WEEK_HOURS, _score_week, build_validation_folds,
    organizer_local_score, run_rolling_validation, validate_validation_folds,
)


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def synthetic_data():
    warm_start_weeks = 2
    data_start = T0 - HOUR
    all_timestamps = tuple(data_start + timedelta(hours=index) for index in range(673 + warm_start_weeks * 168 + 12 * 168))
    validation = all_timestamps[673 + warm_start_weeks * 168:]
    targets = {
        timestamp: {"AT": float(index + 10), **{
            target: float(index + target_index * 100)
            for target_index, target in enumerate(TARGETS)
        }}
        for index, timestamp in enumerate(all_timestamps)
    }
    realizations = {timestamp: {target: targets[timestamp][target] for target in TARGETS}
                    for timestamp in validation}
    return all_timestamps, validation, targets, realizations


class RollingValidationTests(unittest.TestCase):
    def test_builds_twelve_fixed_origin_weeks_without_future_training(self):
        all_timestamps, validation, _, _ = synthetic_data()
        folds = build_validation_folds(validation, all_timestamps)
        self.assertEqual(len(folds), 12)
        for fold in folds:
            self.assertEqual(len(fold.forecast_timestamps), 168)
            self.assertEqual(fold.issue_time, fold.forecast_start - HOUR)
            self.assertEqual(tuple(timestamp - fold.issue_time for timestamp in fold.forecast_timestamps),
                             tuple(timedelta(hours=index) for index in range(1, 169)))
            self.assertLessEqual(max(fold.train_timestamps), fold.issue_time)

    def test_rejects_deliberate_future_week_training(self):
        all_timestamps, validation, _, _ = synthetic_data()
        good = build_validation_folds(validation, all_timestamps)
        bad = list(good)
        bad[0] = WeeklyFold("bad", (bad[1].forecast_start,), bad[0].forecast_timestamps)
        with self.assertRaises(ValueError):
            validate_validation_folds(bad)

    def test_scorer_rejects_partial_or_misaligned_week(self):
        all_timestamps, validation, targets, realizations = synthetic_data()
        predictions = {timestamp: {target: tuple(int(targets[timestamp][target]) for _ in range(300))
                                   for target in TARGETS} for timestamp in validation[:167]}
        with self.assertRaises(ValueError):
            _score_week(predictions, realizations, validation[:167])

    def test_organizer_scorer_parity_formula(self):
        timestamps = tuple(T0 + index * HOUR for index in range(168))
        actuals = {timestamp: {target: 10.0 for target in TARGETS} for timestamp in timestamps}
        predictions = {timestamp: {target: tuple([9, 10, 11] * 100) for target in TARGETS}
                       for timestamp in timestamps}
        standard_deviation = (2.0 / 3.0) ** 0.5
        sharpness = 2.0 / (2.0 + 2.0 * standard_deviation)
        expected = 0.2 + 0.45 + 0.35 * sharpness
        self.assertAlmostEqual(organizer_local_score(predictions, actuals, timestamps), expected, places=12)

    def test_end_to_end_artifacts_preserve_order_holdout_and_samples(self):
        all_timestamps, validation, targets, realizations = synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            result = run_rolling_validation(
                hourly_targets=targets, realizations=realizations,
                output_dir=directory, point_models=("seasonal_persistence",),
                warm_start_weeks=2,
                seed=7,
            )
            summary = result["summary"]
            self.assertEqual(summary["weeks_completed"], 12)
            self.assertEqual(summary["development_weeks"], 11)
            self.assertEqual(summary["holdout_week"], 12)
            self.assertGreater(summary["week_12_holdout_score"], 0.0)
            self.assertEqual(result["by_week"][-1]["week_number"], 12)
            self.assertEqual(result["by_week"][-1]["forecast_start"], "2026-04-30T00:00:00Z")
            self.assertEqual(len(result["summary"]["historical_calibration_timestamps"]), 2 * 168)
            self.assertTrue(all(timestamp < result["by_week"][0]["issue_time"]
                                for timestamp in result["summary"]["historical_calibration_timestamps"]))
            self.assertEqual(set(result["by_target"]), set(TARGETS))
            self.assertEqual(tuple(result["by_week"][0]["selected_point_model"]), TARGETS)
            first = result["predictions"]["validation_week_01"]
            self.assertEqual(len(first), 168)
            self.assertEqual(tuple(first[0])[:1], ("timestamp",))
            self.assertEqual(set(first[0]) - {"timestamp"}, {f"{target}_samples" for target in TARGETS})
            self.assertTrue(all(len(row[f"{target}_samples"]) == 300 for row in first for target in TARGETS))
            self.assertTrue(all(isinstance(value, int) for row in first for target in TARGETS for value in row[f"{target}_samples"]))
            self.assertTrue((Path(directory) / "rolling_validation_summary.json").is_file())
            self.assertEqual(json.loads((Path(directory) / "rolling_validation_summary.json").read_text())["holdout_week"], 12)


if __name__ == "__main__":
    unittest.main()
