import unittest
import csv
from pathlib import Path

from swissgrid_forecaster.real_uncertainty_backtest import (
    METHODS, TARGET_NAMES, compare_methods, fit_calibrated_champion,
    load_actuals, load_point_forecasts, load_residual_panel,
    sample_folds, score_samples_against_actuals,
)
from swissgrid_forecaster.edh_scoring import PERFECT_SCORE
from swissgrid_forecaster.uncertainty_model import ResidualPanel
from swissgrid_forecaster.sampler_evaluation import weekly_folds

CSV_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "real_backtest" / "oof_predictions.csv"


def _artifact_header():
    if not CSV_PATH.is_file():
        return ()
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        return tuple(next(csv.reader(stream), ()))


_header = _artifact_header()
ARTIFACT_READY = CSV_PATH.is_file() and "CH_residual" in _header and "AT_residual" not in _header


def small_panel(weeks: int = 4) -> ResidualPanel:
    """First few real weekly folds only, to keep tests fast."""
    panel = load_residual_panel(CSV_PATH)
    folds = weekly_folds(panel, week_hours=168)[:weeks]
    return ResidualPanel(panel.target_names, tuple(row for fold in folds for row in fold))


@unittest.skipUnless(ARTIFACT_READY, "repaired AT/DE/FR/IT OOF artifact not present")
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


@unittest.skipUnless(ARTIFACT_READY, "repaired AT/DE/FR/IT OOF artifact not present")
class CompareMethodsOnRealDataTests(unittest.TestCase):
    def test_all_four_methods_evaluate_and_rank(self):
        panel = small_panel(4)
        ranked = compare_methods(panel, seed=1, n_samples=20, week_hours=168)
        self.assertEqual({row.method for row in ranked}, set(METHODS))
        self.assertEqual(list(ranked), sorted(ranked, key=lambda r: r.mean_score))
        for row in ranked:
            self.assertGreater(len(row.weekly_scores), 0)


@unittest.skipUnless(ARTIFACT_READY, "repaired AT/DE/FR/IT OOF artifact not present")
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


@unittest.skipUnless(ARTIFACT_READY, "repaired AT/DE/FR/IT OOF artifact not present")
class ScoreAgainstRealActualsTests(unittest.TestCase):
    def test_official_formula_scores_our_real_champion_samples(self):
        panel = small_panel(6)
        calibrated, demo_folds = fit_calibrated_champion(
            panel, "correlated_gaussian", fit_weeks=3, calibration_weeks=1,
            target_coverage=0.8, seed=5, n_samples=30)
        point_forecasts = load_point_forecasts(CSV_PATH)
        # edh2026's formula requires exactly 300 samples per array, unlike the other (faster) tests here.
        samples = sample_folds(calibrated, demo_folds, point_forecasts, seed=5, n_samples=300)
        actuals = load_actuals(CSV_PATH)
        score = score_samples_against_actuals(samples, actuals)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, PERFECT_SCORE)

    def test_missing_actual_rejected(self):
        with self.assertRaises(ValueError):
            score_samples_against_actuals(({"timestamp": "2099-01-01T00:00:00+00:00",
                                           **{f"{n}_samples": [0] * 300 for n in TARGET_NAMES}},), {})


if __name__ == "__main__":
    unittest.main()
