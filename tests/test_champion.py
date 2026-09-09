import builtins
import unittest
from unittest.mock import patch
from swissgrid_forecaster.champion import Persistence, SeasonalPersistence, HistoricalConditional, OptionalEstimator, select_champion
from swissgrid_forecaster.model_contracts import PredictContext
from swissgrid_forecaster.oof import run_oof
from test_splits import samples, plan, H
from test_model_contracts import fit, HASH


class ChampionTests(unittest.TestCase):
    def test_seasonal_elapsed_time(self):
        model = fit(SeasonalPersistence(2*H))
        r = samples(7)[6]
        self.assertEqual(model.predict([r.features], PredictContext((r,), HASH))[0].values, (4,))
        r = samples(8)[7]
        with self.assertRaises(ValueError):
            model.predict([r.features], PredictContext((r,), HASH))

    def test_empirical_conditional(self):
        model = fit(HistoricalConditional((0,)))
        r = samples(7)[6]
        self.assertEqual(model.predict([r.features], PredictContext((r,), HASH))[0].values, (0, 2, 4))
        from dataclasses import replace
        r = replace(r, features=(9,))
        with self.assertRaises(ValueError):
            model.predict([r.features], PredictContext((r,), HASH))

    def test_selection_rolling_oof_only(self):
        result = run_oof(plan(), samples(), [Persistence, HistoricalConditional], HASH)
        winner, scores = select_champion(result, metric='mae')
        self.assertEqual(winner[0], 'persistence')
        self.assertEqual(len(scores), 2)
        with self.assertRaises(ValueError):
            select_champion(result.records, metric='mae')
        with self.assertRaises(ValueError):
            select_champion(result, metric='bias')

    def test_optional_missing_dependency(self):
        real_import = builtins.__import__
        def no_sklearn(name, *args, **kwargs):
            if name.startswith('sklearn'):
                raise ImportError('simulated absent dependency')
            return real_import(name, *args, **kwargs)
        for kind in ('ridge', 'quantile_boosting'):
            with patch('builtins.__import__', side_effect=no_sklearn), self.assertRaisesRegex(RuntimeError, 'unavailable'):
                fit(OptionalEstimator(kind))

    def test_optional_estimators_when_available(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest('optional scikit-learn is not installed')
        r = samples(7)[6]
        for kind, expected in [('ridge', 'point'), ('quantile_boosting', 'quantile')]:
            model = fit(OptionalEstimator(kind))
            prediction = model.predict([r.features], PredictContext((r,), HASH))[0]
            self.assertEqual(prediction.prediction_type, expected)

    def test_parameter_version_identity(self):
        self.assertNotEqual(Persistence().version, SeasonalPersistence(H).version)
        self.assertNotEqual(OptionalEstimator('ridge').version, OptionalEstimator('ridge', alpha=2).version)
        self.assertNotEqual(HistoricalConditional().version, HistoricalConditional((0,)).version)
