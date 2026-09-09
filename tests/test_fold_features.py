import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from swissgrid_forecaster.fold_features import FeatureRow, TimeTransform, FoldStandardScaler

T = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


def row(hour, value, **kwargs):
    return FeatureRow(str(hour), T+hour*H, T+hour*H, value, 'source-v1', **kwargs)


class FoldFeatureTests(unittest.TestCase):
    def transform(self, name, rows, window=3*H):
        return TimeTransform(name, window=window).fit([row(0, 0)], T).transform(rows, T+4*H).values

    def test_elapsed_windows_irregular_sampling(self):
        rows = [row(0, 100), row(1, 100), row(2, 2), row(4, 4)]
        for name, expected in [('rolling_mean', 3), ('rolling_std', 1), ('rolling_min', 2), ('rolling_max', 4)]:
            self.assertEqual(self.transform(name, rows), (expected,))
        self.assertEqual(self.transform('lag', rows, 2*H), (2,))
        self.assertEqual(self.transform('lag', rows, H), (None,))
        self.assertEqual(self.transform('rate_of_change', rows, 2*H), (2/7200,))

    def test_future_knowledge_and_events_excluded(self):
        rows = [row(2, 2), replace(row(3, 999), known_at=T+5*H), replace(row(5, 999), known_at=T)]
        self.assertEqual(self.transform('rolling_mean', rows), (2,))

    def test_missingness_age_calendar(self):
        self.assertEqual(self.transform('missingness_flag', []), (1,))
        self.assertEqual(self.transform('missingness_flag', [row(3, None)]), (1,))
        self.assertEqual(self.transform('freshness', [row(2, 3)]), (7200,))
        self.assertEqual(self.transform('freshness', []), (None,))
        self.assertEqual(self.transform('calendar', []), (3, 4, 1))
        self.assertEqual(self.transform('rolling_mean', [row(2, None), row(3, 3)]), (None,))
        with self.assertRaises(ValueError):
            TimeTransform('lag', window=H, missingness_policy='error').fit([], T).transform([], T)

    def test_fit_leakage_and_failed_refit(self):
        model = TimeTransform('lag', window=H).fit([row(0, 1)], T)
        for bad in [row(1, 2), row(0, 2, partition='validation'), row(0, 2, partition='holdout')]:
            with self.assertRaises(ValueError):
                model.fit([bad], T)
            with self.assertRaises(ValueError):
                model.transform([], T+H)
        with self.assertRaises(ValueError):
            model.fit([row(0, 1), row(0, 1)], T)

    def test_duplicate_vintages_and_fit_cutoff(self):
        model = TimeTransform('lag', window=H).fit([], T+H)
        with self.assertRaises(ValueError):
            model.transform([], T)
        with self.assertRaises(ValueError):
            model.transform([row(1, 1), replace(row(1, 2), row_id='other')], T+2*H)

    def test_scaler_training_only(self):
        model = FoldStandardScaler().fit([row(0, 0), row(1, 2)], T+H)
        result = model.transform([row(2, 101, partition='validation'), row(3, None)], T+3*H)
        self.assertEqual(result.values, (100, None))
        self.assertEqual(model.center, 1)
        with self.assertRaises(ValueError):
            model.fit([row(2, 100, partition='validation')], T+2*H)
