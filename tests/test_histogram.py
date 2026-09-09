import unittest

from swissgrid_forecaster.histogram import HistogramPayload


class HistogramTests(unittest.TestCase):
    def test_probability_payload_sums_to_one(self):
        payload = HistogramPayload.from_samples((-2, -1, 0, 1, 2), bins=4)
        self.assertAlmostEqual(sum(payload.probabilities), 1.0)
        self.assertAlmostEqual(payload.total_probability, 1.0)
        self.assertEqual(len(payload.bin_centers), len(payload.probabilities))

    def test_constant_and_round_trip(self):
        payload = HistogramPayload.from_samples((4, 4, 4), bins=10)
        self.assertEqual(payload.probabilities, (1.0,))
        self.assertEqual(HistogramPayload.from_dict(payload.to_dict()), payload)

    def test_invalid_probability_is_rejected(self):
        payload = HistogramPayload.from_samples((1, 2, 3))
        broken = payload.to_dict()
        broken["total_probability"] = .5
        with self.assertRaises(ValueError):
            HistogramPayload.from_dict(broken)

