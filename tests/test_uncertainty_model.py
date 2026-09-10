import unittest
from datetime import datetime, timedelta, timezone
from random import Random
from statistics import correlation as stats_correlation

from swissgrid_forecaster.uncertainty_model import (
    METHODS, ResidualObservation, ResidualPanel, UncertaintyModel,
    calibrate_scale, correlation_matrix, covariance_stability,
    fit_uncertainty_model, qq_diagnostic, residual_scale_by_bucket, sample_distribution,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def synthetic_panel(n=200, *, seed=1, rho=0.8, scale=(2.0, 3.0), horizon=H, start=T0):
    rng = Random(seed)
    rows = []
    for i in range(n):
        issue = start + i * H
        shared = rng.gauss(0, 1)
        e1 = scale[0] * (rho * shared + (1 - rho ** 2) ** 0.5 * rng.gauss(0, 1))
        e2 = scale[1] * (rho * shared + (1 - rho ** 2) ** 0.5 * rng.gauss(0, 1))
        rows.append((issue, issue + horizon, horizon, (e1, e2)))
    return ResidualPanel(("border_a", "border_b"), tuple(rows))


class ResidualObservationTests(unittest.TestCase):
    def test_horizon_must_match_clocks(self):
        with self.assertRaises(ValueError):
            ResidualObservation("f", "r", "t", T0, T0 + H, 2 * H, 1.0)
        with self.assertRaises(ValueError):
            ResidualObservation("f", "r", "t", T0, T0 + H, timedelta(0), 1.0)

    def test_residual_must_be_finite(self):
        with self.assertRaises(ValueError):
            ResidualObservation("f", "r", "t", T0, T0 + H, H, float("nan"))

    def test_from_oof_bridges_existing_pipeline(self):
        from swissgrid_forecaster.oof import run_oof
        from swissgrid_forecaster.champion import Persistence
        from test_splits import samples, plan
        from test_model_contracts import HASH
        result = run_oof(plan(), samples(), [Persistence], HASH)
        residuals = ResidualObservation.from_oof(result, target_name="net_position")
        self.assertEqual(len(residuals), len(result.records))
        self.assertTrue(all(isinstance(r, ResidualObservation) for r in residuals))


class ResidualPanelTests(unittest.TestCase):
    def test_inner_join_on_shared_timestamps(self):
        a = (ResidualObservation("f", "0", "a", T0, T0 + H, H, 1.0),
             ResidualObservation("f", "1", "a", T0 + H, T0 + 2 * H, H, 2.0))
        b = (ResidualObservation("f", "0", "b", T0, T0 + H, H, 5.0),)
        panel = ResidualPanel.from_target_residuals({"a": a, "b": b})
        self.assertEqual(panel.target_names, ("a", "b"))
        self.assertEqual(len(panel.rows), 1)
        self.assertEqual(panel.rows[0][3], (1.0, 5.0))

    def test_no_shared_timestamps_rejected(self):
        a = (ResidualObservation("f", "0", "a", T0, T0 + H, H, 1.0),)
        b = (ResidualObservation("f", "0", "b", T0 + H, T0 + 2 * H, H, 5.0),)
        with self.assertRaises(ValueError):
            ResidualPanel.from_target_residuals({"a": a, "b": b})

    def test_horizon_mismatch_at_shared_timestamp_rejected(self):
        a = (ResidualObservation("f", "0", "a", T0, T0 + H, H, 1.0),)
        b = (ResidualObservation("f", "0", "b", T0, T0 + 2 * H, 2 * H, 5.0),)
        with self.assertRaises(ValueError):
            ResidualPanel.from_target_residuals({"a": a, "b": b})

    def test_duplicate_joint_timestamp_rejected(self):
        rows = ((T0, T0 + H, H, (1.0, 2.0)), (T0, T0 + H, H, (3.0, 4.0)))
        with self.assertRaises(ValueError):
            ResidualPanel(("a", "b"), rows)

    def test_width_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ResidualPanel(("a", "b"), ((T0, T0 + H, H, (1.0,)),))

    def test_identity_hash_stable(self):
        panel = synthetic_panel(10)
        self.assertEqual(panel.identity_hash(), synthetic_panel(10).identity_hash())
        self.assertNotEqual(panel.identity_hash(), synthetic_panel(10, seed=2).identity_hash())


class FitUncertaintyModelTests(unittest.TestCase):
    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            fit_uncertainty_model(synthetic_panel(20), method="bogus", fit_cutoff=T0 + 100 * H)

    def test_future_row_rejected_no_leakage(self):
        panel = synthetic_panel(20)
        with self.assertRaises(ValueError):
            fit_uncertainty_model(panel, method="correlated_gaussian", fit_cutoff=T0)

    def test_independent_gaussian_has_zero_off_diagonal(self):
        model = fit_uncertainty_model(synthetic_panel(50), method="independent_gaussian", fit_cutoff=T0 + 100 * H)
        cov = model.covariance_for(None)
        self.assertEqual(cov[0][1], 0.0)
        self.assertEqual(cov[1][0], 0.0)

    def test_correlated_gaussian_has_nonzero_off_diagonal(self):
        model = fit_uncertainty_model(synthetic_panel(50), method="correlated_gaussian", fit_cutoff=T0 + 100 * H)
        cov = model.covariance_for(None)
        self.assertNotEqual(cov[0][1], 0.0)
        self.assertAlmostEqual(cov[0][1], cov[1][0])

    def test_horizon_covariance_buckets_by_horizon(self):
        one_hour = synthetic_panel(40, horizon=H, start=T0)
        two_hour = synthetic_panel(40, horizon=2 * H, start=T0 + 200 * H, seed=2, scale=(6.0, 7.0))
        panel = ResidualPanel(("border_a", "border_b"), one_hour.rows + two_hour.rows)
        model = fit_uncertainty_model(panel, method="horizon_covariance", fit_cutoff=T0 + 1000 * H)
        self.assertEqual(set(k for k, _ in model.bias), {H, 2 * H})
        with self.assertRaises(ValueError):
            model.covariance_for(3 * H)

    def test_hour_of_day_covariance_buckets_by_hour(self):
        model = fit_uncertainty_model(synthetic_panel(200), method="hour_of_day_covariance", fit_cutoff=T0 + 1000 * H)
        self.assertEqual(set(k for k, _ in model.bias), set(range(24)))

    def test_empirical_bootstrap_stores_pool(self):
        model = fit_uncertainty_model(synthetic_panel(50), method="empirical_bootstrap", fit_cutoff=T0 + 100 * H)
        self.assertEqual(len(model.empirical_for(None)), 50)

    def test_student_t_requires_valid_dof(self):
        panel = synthetic_panel(50)
        with self.assertRaises(ValueError):
            fit_uncertainty_model(panel, method="student_t", fit_cutoff=T0 + 100 * H)
        with self.assertRaises(ValueError):
            fit_uncertainty_model(panel, method="student_t", fit_cutoff=T0 + 100 * H, degrees_of_freedom=2)
        model = fit_uncertainty_model(panel, method="student_t", fit_cutoff=T0 + 100 * H, degrees_of_freedom=5)
        self.assertEqual(model.degrees_of_freedom, 5.0)

    def test_degrees_of_freedom_only_for_student_t(self):
        with self.assertRaises(ValueError):
            fit_uncertainty_model(synthetic_panel(20), method="correlated_gaussian",
                                  fit_cutoff=T0 + 100 * H, degrees_of_freedom=5)

    def test_all_methods_produce_valid_models(self):
        panel = synthetic_panel(200)
        for method in METHODS:
            with self.subTest(method=method):
                dof = 6 if method == "student_t" else None
                model = fit_uncertainty_model(panel, method=method, fit_cutoff=T0 + 1000 * H, degrees_of_freedom=dof)
                self.assertIsInstance(model, UncertaintyModel)


class SampleDistributionTests(unittest.TestCase):
    def setUp(self):
        self.panel = synthetic_panel(200)
        self.model = fit_uncertainty_model(self.panel, method="correlated_gaussian", fit_cutoff=T0 + 1000 * H)

    def test_exactly_n_samples_finite_ints(self):
        draws = sample_distribution(self.model, {"border_a": 10.0, "border_b": -5.0}, seed=1, n_samples=300)
        self.assertEqual(set(draws), {"border_a", "border_b"})
        for values in draws.values():
            self.assertEqual(len(values), 300)
            self.assertTrue(all(isinstance(v, int) for v in values))

    def test_deterministic_with_same_seed(self):
        a = sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=42, n_samples=50)
        b = sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=42, n_samples=50)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        a = sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=1, n_samples=50)
        b = sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=2, n_samples=50)
        self.assertNotEqual(a, b)

    def test_joint_dependency_preserved(self):
        draws = sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=7,
                                    n_samples=3000, round_to_int=False)
        observed = stats_correlation(draws["border_a"], draws["border_b"])
        self.assertGreater(observed, 0.5)

    def test_means_must_match_target_names(self):
        with self.assertRaises(ValueError):
            sample_distribution(self.model, {"border_a": 0.0}, seed=1, n_samples=10)

    def test_n_samples_must_be_positive_int(self):
        with self.assertRaises(ValueError):
            sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=1, n_samples=0)
        with self.assertRaises(ValueError):
            sample_distribution(self.model, {"border_a": 0.0, "border_b": 0.0}, seed=1, n_samples=1.5)

    def test_unseen_bucket_key_rejected(self):
        horizon_model = fit_uncertainty_model(self.panel, method="horizon_covariance", fit_cutoff=T0 + 1000 * H)
        with self.assertRaises(ValueError):
            sample_distribution(horizon_model, {"border_a": 0.0, "border_b": 0.0},
                                bucket_key=timedelta(hours=99), seed=1, n_samples=10)

    def test_empirical_bootstrap_draws_from_pool(self):
        model = fit_uncertainty_model(self.panel, method="empirical_bootstrap", fit_cutoff=T0 + 1000 * H)
        draws = sample_distribution(model, {"border_a": 0.0, "border_b": 0.0}, seed=1, n_samples=300)
        self.assertEqual(len(draws["border_a"]), 300)


class CalibrateScaleTests(unittest.TestCase):
    def test_scale_search_moves_coverage_toward_target(self):
        fit_panel = synthetic_panel(300, seed=1, scale=(1.0, 1.0))
        calibration_panel = synthetic_panel(300, seed=2, scale=(4.0, 4.0), start=T0 + 1000 * H)
        model = fit_uncertainty_model(fit_panel, method="correlated_gaussian", fit_cutoff=T0 + 1000 * H)
        calibrated = calibrate_scale(model, calibration_panel, target_coverage=0.8, seed=3, n_samples=200)
        self.assertGreater(calibrated.scale_for(None), 1.0)

    def test_calibration_panel_must_match_targets(self):
        model = fit_uncertainty_model(synthetic_panel(50), method="correlated_gaussian", fit_cutoff=T0 + 1000 * H)
        mismatched = ResidualPanel(("x", "y"), synthetic_panel(50, seed=9).rows)
        with self.assertRaises(ValueError):
            calibrate_scale(model, mismatched)


class DiagnosticsTests(unittest.TestCase):
    def test_qq_diagnostic_is_monotonic(self):
        values = [Random(1).gauss(0, 1) for _ in range(100)]
        points = qq_diagnostic(values)
        theoretical = [p[0] for p in points]
        empirical = [p[1] for p in points]
        self.assertEqual(theoretical, sorted(theoretical))
        self.assertEqual(empirical, sorted(empirical))

    def test_correlation_matrix_diagonal_and_symmetric(self):
        matrix = correlation_matrix(synthetic_panel(100))
        self.assertEqual(matrix[0][0], 1.0)
        self.assertEqual(matrix[1][1], 1.0)
        self.assertAlmostEqual(matrix[0][1], matrix[1][0])
        self.assertGreater(matrix[0][1], 0.5)

    def test_residual_scale_by_bucket_matches_generating_scale(self):
        panel = synthetic_panel(300, scale=(2.0, 6.0))
        buckets = residual_scale_by_bucket(panel, "correlated_gaussian")
        self.assertEqual(len(buckets), 1)
        _, scales = buckets[0]
        self.assertAlmostEqual(scales[0], 2.0, delta=0.5)
        self.assertAlmostEqual(scales[1], 6.0, delta=1.0)

    def test_covariance_stability_rejects_bad_fold_size(self):
        panel = synthetic_panel(100)
        with self.assertRaises(ValueError):
            covariance_stability(panel, fold_size=1)
        with self.assertRaises(ValueError):
            covariance_stability(panel, fold_size=200)
        self.assertGreaterEqual(covariance_stability(panel, fold_size=20), 0.0)


if __name__ == "__main__":
    unittest.main()
