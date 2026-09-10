import unittest
from datetime import timedelta

from swissgrid_forecaster.sampler_evaluation import (
    SamplerEvaluation, WeeklyScore, capture_sharpness_curve,
    compare_samplers, evaluate_sampler, weekly_folds,
)
from swissgrid_forecaster.uncertainty_model import fit_uncertainty_model
from test_uncertainty_model import H, T0, synthetic_panel


class WeeklyFoldsTests(unittest.TestCase):
    def test_splits_into_consecutive_chunks(self):
        panel = synthetic_panel(100)
        folds = weekly_folds(panel, week_hours=20)
        self.assertEqual(len(folds), 5)
        self.assertTrue(all(len(fold) == 20 for fold in folds))

    def test_rejects_partial_week(self):
        with self.assertRaises(ValueError):
            weekly_folds(synthetic_panel(10), week_hours=20)

    def test_rejects_non_positive_week_hours(self):
        with self.assertRaises(ValueError):
            weekly_folds(synthetic_panel(50), week_hours=0)


class EvaluateSamplerTests(unittest.TestCase):
    def setUp(self):
        self.panel = synthetic_panel(200)

    def test_requires_at_least_two_folds(self):
        with self.assertRaises(ValueError):
            evaluate_sampler(self.panel, "correlated_gaussian", seed=1, week_hours=200)

    def test_produces_one_weekly_score_per_evaluated_fold(self):
        evaluation = evaluate_sampler(self.panel, "correlated_gaussian", seed=1, week_hours=20, n_samples=100)
        self.assertIsInstance(evaluation, SamplerEvaluation)
        self.assertEqual(len(evaluation.weekly_scores), 9)
        self.assertTrue(all(isinstance(week, WeeklyScore) for week in evaluation.weekly_scores))
        self.assertEqual(evaluation.worst_week_score, max(w.score for w in evaluation.weekly_scores))

    def test_student_t_requires_dof(self):
        with self.assertRaises(ValueError):
            evaluate_sampler(self.panel, "student_t", seed=1, week_hours=20, n_samples=100)
        evaluation = evaluate_sampler(self.panel, "student_t", seed=1, week_hours=20,
                                      n_samples=100, degrees_of_freedom=8)
        self.assertEqual(evaluation.method, "student_t")

    def test_custom_score_fn_is_used(self):
        calls = []

        def score_fn(capture, sharpness, mae, rows):
            calls.append((capture, sharpness, mae))
            return 1.0

        evaluation = evaluate_sampler(self.panel, "independent_gaussian", seed=1, week_hours=20,
                                      n_samples=50, score_fn=score_fn)
        self.assertTrue(calls)
        self.assertTrue(all(week.score == 1.0 for week in evaluation.weekly_scores))

    def test_independent_vs_correlated_are_both_valid_finite(self):
        for method in ("independent_gaussian", "correlated_gaussian", "empirical_bootstrap"):
            with self.subTest(method=method):
                evaluation = evaluate_sampler(self.panel, method, seed=3, week_hours=20, n_samples=80)
                self.assertTrue(all(w.score == w.score for w in evaluation.weekly_scores))  # not NaN


class CompareSamplersTests(unittest.TestCase):
    def test_baseline_always_keep_and_sorted_by_score(self):
        panel = synthetic_panel(200)
        independent = evaluate_sampler(panel, "independent_gaussian", seed=5, week_hours=20, n_samples=80)
        correlated = evaluate_sampler(panel, "correlated_gaussian", seed=5, week_hours=20, n_samples=80)
        ranked = compare_samplers({"independent_gaussian": independent, "correlated_gaussian": correlated},
                                  baseline="independent_gaussian")
        by_method = {r.method: r for r in ranked}
        self.assertEqual(by_method["independent_gaussian"].keep_drop, "KEEP")
        self.assertIn(by_method["correlated_gaussian"].keep_drop, ("KEEP", "DROP", "UNDECIDED"))
        self.assertEqual(list(ranked), sorted(ranked, key=lambda r: r.mean_score))

    def test_unknown_baseline_rejected(self):
        panel = synthetic_panel(200)
        evaluation = evaluate_sampler(panel, "independent_gaussian", seed=5, week_hours=20, n_samples=50)
        with self.assertRaises(ValueError):
            compare_samplers({"independent_gaussian": evaluation}, baseline="missing")


class CaptureSharpnessCurveTests(unittest.TestCase):
    def test_wider_scale_increases_capture_and_sharpness_width(self):
        panel = synthetic_panel(60)
        model = fit_uncertainty_model(panel, method="correlated_gaussian", fit_cutoff=T0 + 1000 * H)
        curve = capture_sharpness_curve(model, panel, scales=(0.2, 1.0, 4.0), seed=1, n_samples=200)
        self.assertEqual([c[0] for c in curve], [0.2, 1.0, 4.0])
        widths = [c[2] for c in curve]
        self.assertEqual(widths, sorted(widths))

    def test_rejects_nonpositive_scale(self):
        panel = synthetic_panel(60)
        model = fit_uncertainty_model(panel, method="correlated_gaussian", fit_cutoff=T0 + 1000 * H)
        with self.assertRaises(ValueError):
            capture_sharpness_curve(model, panel, scales=(0.0,), seed=1, n_samples=50)


if __name__ == "__main__":
    unittest.main()
