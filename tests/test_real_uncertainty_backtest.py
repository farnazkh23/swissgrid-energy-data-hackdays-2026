import unittest
from pathlib import Path

from swissgrid_forecaster.real_uncertainty_backtest import (
    TARGET_NAMES, compare_methods, fit_calibrated_champion,
    load_point_forecasts, load_residual_panel, sample_folds,
)
from swissgrid_forecaster.uncertainty_model import ResidualPanel
from swissgrid_forecaster.sampler_evaluation import weekly_folds

CSV_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "real_backtest" / "oof_predictions.csv"


def small_panel(weeks: int = 4) -> ResidualPanel:
    """First few real weekly folds only, to keep tests fast."""
    panel = load_residual_panel(CSV_PATH)
    folds = weekly_folds(panel, week_hours=168)[:weeks]
    return ResidualPanel(panel.target_names, tuple(row for fold in folds for row in fold))


@unittest.skipUnless(CSV_PATH.is_file(), "real OOF backtest artifact not present")
class LoadResidualPanelTests(unittest.TestCase):
    def test_loads_all_rows_with_expected_targets(self):
        panel = load_residual_panel(CSV_PATH)
        self.assertEqual(panel.target_names, TARGET_NAMES)
        self.assertEqual(len(panel.rows), 2016)

    def test_residual_rows_are_finite_and_chronological(self):
        panel = load_residual_panel(CSV_PATH)
        times = [row[1] for row in panel.rows]
        self.assertEqual(times, sorted(times))

    def test_point_forecasts_cover_every_residual_timestamp(self):
        panel = load_residual_panel(CSV_PATH)
        forecasts = load_point_forecasts(CSV_PATH)
        self.assertTrue(all(row[1] in forecasts for row in panel.rows))
        self.assertEqual(set(forecasts[panel.rows[0][1]]), set(TARGET_NAMES))


@unittest.skipUnless(CSV_PATH.is_file(), "real OOF backtest artifact not present")
class CompareMethodsOnRealDataTests(unittest.TestCase):
    def test_all_four_methods_evaluate_and_rank(self):
        panel = small_panel(4)
        ranked = compare_methods(panel, seed=1, n_samples=20, week_hours=168)
        self.assertEqual({row.method for row in ranked},
                         {"independent_gaussian", "correlated_gaussian", "student_t", "empirical_bootstrap"})
        self.assertEqual(list(ranked), sorted(ranked, key=lambda r: r.mean_score))
        for row in ranked:
            self.assertGreater(len(row.weekly_scores), 0)


@unittest.skipUnless(CSV_PATH.is_file(), "real OOF backtest artifact not present")
class CalibratedChampionSamplingTests(unittest.TestCase):
    def test_fit_calibrate_and_sample_real_demo_week(self):
        panel = small_panel(6)
        calibrated, demo_folds = fit_calibrated_champion(
            panel, "correlated_gaussian", fit_weeks=3, calibration_weeks=1,
            target_coverage=0.8, seed=5, n_samples=30)
        self.assertEqual(len(demo_folds), 2)
        point_forecasts = load_point_forecasts(CSV_PATH)
        samples = sample_folds(calibrated, demo_folds, point_forecasts, seed=5, n_samples=30)
        self.assertEqual(len(samples), 2 * 168)
        first = samples[0]
        for name in TARGET_NAMES:
            values = first[f"{name}_samples"]
            self.assertEqual(len(values), 30)
            self.assertTrue(all(isinstance(v, int) for v in values))

    def test_rejects_when_not_enough_weeks_for_split(self):
        panel = small_panel(3)
        with self.assertRaises(ValueError):
            fit_calibrated_champion(panel, "correlated_gaussian", fit_weeks=2, calibration_weeks=1, seed=1)


if __name__ == "__main__":
    unittest.main()
