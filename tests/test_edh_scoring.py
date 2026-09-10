import unittest

from swissgrid_forecaster.edh_scoring import (
    CAPTURE_WEIGHT, ERROR_WEIGHT, KERNEL_BANDWIDTH, PERFECT_SCORE, SHARPNESS_WEIGHT,
    normalize_score_to_percentage, score_distribution, score_row, score_submission,
    validate_submission_rows,
)


def uniform_samples(low: int, high: int, count: int = 300) -> list:
    """`count` integers spread as evenly as possible across [low, high]."""
    if count == 1:
        return [low]
    step = (high - low) / (count - 1)
    return [round(low + i * step) for i in range(count)]


class ScoreDistributionTests(unittest.TestCase):
    def test_perfect_score_when_all_samples_equal_realization(self):
        samples = [10] * 300
        self.assertAlmostEqual(score_distribution(samples, 10), PERFECT_SCORE)

    def test_degenerate_samples_miss_realization(self):
        samples = [10] * 300
        self.assertEqual(score_distribution(samples, 11), 0.0)

    def test_realization_outside_range_scores_zero(self):
        samples = uniform_samples(0, 100)
        self.assertEqual(score_distribution(samples, 200), 0.0)
        self.assertEqual(score_distribution(samples, -1), 0.0)

    def test_realization_at_mean_maximizes_goodness(self):
        samples = uniform_samples(0, 100)
        mean = sum(samples) / len(samples)
        score_at_mean = score_distribution(samples, mean)
        score_off_mean = score_distribution(samples, mean + 10)
        self.assertGreater(score_at_mean, score_off_mean)

    def test_matches_hand_computed_formula(self):
        samples = uniform_samples(0, 10, count=300)
        realization = 5.0
        minimum, maximum = min(samples), max(samples)
        mean = sum(samples) / len(samples)
        std = (sum((v - mean) ** 2 for v in samples) / len(samples)) ** 0.5
        goodness = max(0.0, 1.0 - abs(realization - mean) / (KERNEL_BANDWIDTH * std))
        sharpness = (maximum - minimum) / ((maximum - minimum) + 2.0 * std)
        expected = CAPTURE_WEIGHT + ERROR_WEIGHT * goodness + SHARPNESS_WEIGHT * sharpness
        self.assertAlmostEqual(score_distribution(samples, realization), expected)

    def test_more_concentrated_distribution_scores_higher_for_same_range(self):
        # Same [min, max] and same mean, but concentrated has much lower std.
        concentrated = [50] * 298 + [0, 100]
        spread = uniform_samples(0, 100)
        self.assertGreater(score_distribution(concentrated, 50), score_distribution(spread, 50))

    def test_rejects_wrong_sample_count(self):
        with self.assertRaises(ValueError):
            score_distribution([1] * 299, 1)

    def test_rejects_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            score_distribution([float("nan")] * 300, 1)
        with self.assertRaises(ValueError):
            score_distribution(uniform_samples(0, 10), float("inf"))


class ScoreRowAndSubmissionTests(unittest.TestCase):
    def test_score_row_averages_across_targets(self):
        samples = uniform_samples(0, 10)
        row_samples = {"CH": samples, "DE": samples}
        realization = {"CH": 5.0, "DE": 5.0}
        expected = score_distribution(samples, 5.0)
        self.assertAlmostEqual(score_row(row_samples, realization), expected)

    def test_score_row_requires_matching_targets(self):
        with self.assertRaises(ValueError):
            score_row({"CH": uniform_samples(0, 10)}, {"DE": 5.0})

    def test_score_submission_averages_rows(self):
        good = uniform_samples(45, 55)
        bad = uniform_samples(0, 100)
        rows = [({"CH": good}, {"CH": 50.0}), ({"CH": bad}, {"CH": 200.0})]
        expected = (score_distribution(good, 50.0) + 0.0) / 2
        self.assertAlmostEqual(score_submission(rows), expected)

    def test_score_submission_requires_rows(self):
        with self.assertRaises(ValueError):
            score_submission([])

    def test_normalize_score_to_percentage(self):
        self.assertAlmostEqual(normalize_score_to_percentage(PERFECT_SCORE), 100.0)
        self.assertAlmostEqual(normalize_score_to_percentage(0.0), 0.0)
        self.assertEqual(normalize_score_to_percentage(-1.0), 0.0)
        self.assertEqual(normalize_score_to_percentage(PERFECT_SCORE * 2), 100.0)


class ValidateSubmissionRowsTests(unittest.TestCase):
    def _valid_rows(self, count=168):
        return {f"t{i}": {"CH": [0] * 300, "DE": [0] * 300} for i in range(count)}

    def test_valid_shape_passes(self):
        validate_submission_rows(self._valid_rows(), targets=("CH", "DE"))

    def test_wrong_row_count_rejected(self):
        with self.assertRaises(ValueError):
            validate_submission_rows(self._valid_rows(167), targets=("CH", "DE"))

    def test_wrong_array_length_rejected(self):
        rows = self._valid_rows()
        rows["t0"]["CH"] = [0] * 299
        with self.assertRaises(ValueError):
            validate_submission_rows(rows, targets=("CH", "DE"))

    def test_non_integer_value_rejected(self):
        rows = self._valid_rows()
        rows["t0"]["CH"][0] = 1.5
        with self.assertRaises(ValueError):
            validate_submission_rows(rows, targets=("CH", "DE"))

    def test_null_value_rejected(self):
        rows = self._valid_rows()
        rows["t0"]["CH"][0] = None
        with self.assertRaises(ValueError):
            validate_submission_rows(rows, targets=("CH", "DE"))

    def test_missing_target_column_rejected(self):
        rows = self._valid_rows()
        del rows["t0"]["DE"]
        with self.assertRaises(ValueError):
            validate_submission_rows(rows, targets=("CH", "DE"))


if __name__ == "__main__":
    unittest.main()
