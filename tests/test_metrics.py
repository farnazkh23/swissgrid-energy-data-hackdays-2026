import unittest
from math import sqrt
from swissgrid_forecaster import metrics as m


class MetricTests(unittest.TestCase):
    def test_point_metrics(self):
        self.assertEqual(m.mae([0, 2], [1, 4]), 1.5)
        self.assertEqual(m.rmse([0, 2], [1, 4]), sqrt(2.5))
        self.assertEqual(m.bias([0, 2], [1, 4]), 1.5)
        self.assertEqual(m.bias([0, 2], [-1, 0]), -1.5)

    def test_probabilistic_metrics(self):
        self.assertEqual(m.pinball([0, 2], [1, 1], .25), .5)
        self.assertEqual(m.coverage([0, 2], [0, 0], [1, 1]), .5)
        self.assertEqual(m.sharpness([0, 1], [2, 4]), 2.5)
        self.assertEqual(m.brier([0, 1], [.5, 1]), .125)

    def test_invalid_inputs(self):
        for fn in (m.mae, m.rmse, m.bias, m.brier):
            for y, p in [([], []), ([1], [1, 2]), ([None], [1]), ([1], [float('nan')]), ([1], [float('inf')])]:
                with self.subTest(fn=fn, y=y, p=p), self.assertRaises(ValueError):
                    fn(y, p)
        for call in [lambda: m.pinball([1], [1], 0), lambda: m.pinball([1], [1], float('nan')),
                     lambda: m.coverage([1], [2], [0]), lambda: m.sharpness([2], [0]),
                     lambda: m.brier([2], [.5]), lambda: m.brier([1], [1.1])]:
            with self.assertRaises(ValueError):
                call()

    def test_explicit_scaffolds(self):
        for fn in (m.crps, m.wis):
            with self.assertRaises(NotImplementedError):
                fn([], [])
