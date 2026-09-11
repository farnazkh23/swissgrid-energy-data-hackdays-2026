import unittest
from datetime import datetime, timedelta, timezone

from swissgrid_forecaster.real_modeling_v1 import (
    LSEGFeatures,
    _sample,
    build_hourly_targets,
    build_features,
    make_weekly_folds,
)


class RealModelingV1Tests(unittest.TestCase):
    def test_duplicate_quarter_is_not_emitted(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [{"Zeitstempel": start + timedelta(minutes=15 * index), **{target: index for target in ("AT", "CH", "DE", "FR", "IT")}}
                for index in (0, 1, 2, 3, 3)]
        self.assertEqual(len(build_hourly_targets(rows)), 0)

    def test_weekly_folds_are_168_hours(self):
        start = datetime(2024, 1, 1, 1, tzinfo=timezone.utc)
        timestamps = tuple(start + timedelta(hours=index) for index in range(672 + 168 * 3))
        folds = make_weekly_folds(timestamps, max_folds=2)
        self.assertEqual(len(folds), 2)
        self.assertTrue(all(len(forecast.forecast_timestamps) == 168 for forecast in folds))
        self.assertTrue(all(max(forecast.train_timestamps) < forecast.forecast_start for forecast in folds))

    def test_lseg_future_value_is_excluded_but_causal_lag_is_allowed(self):
        issue = datetime(2024, 1, 2, tzinfo=timezone.utc)
        future = issue + timedelta(hours=1)
        lseg = LSEGFeatures(
            {"lseg_fr_da_price": {issue: (50.0, issue)}, "lseg_it_da_price": {}, "lseg_it_demand": {}},
            {}, {}, (),
        )
        history = {issue: {target: 1.0 for target in ("AT", "CH", "DE", "FR", "IT")}}
        names, values = build_features(target_history=history, db_tables={}, lseg=lseg,
                                       stamp=future, issue=issue, use_lseg=True)
        self.assertEqual(dict(zip(names, values))["lseg_fr_price_lag_1h"], 50.0)
        self.assertIsNone(dict(zip(names, values))["lseg_fr_price_lag_24h"])
        names, values = build_features(target_history=history, db_tables={}, lseg=lseg,
                                       stamp=issue + timedelta(hours=1), issue=issue + timedelta(hours=1), use_lseg=True)
        self.assertEqual(dict(zip(names, values))["lseg_fr_price_lag_1h"], 50.0)

    def test_fixed_origin_features_cannot_see_later_forecast_hours(self):
        issue = datetime(2024, 1, 1, tzinfo=timezone.utc)
        history = {
            issue + timedelta(hours=index):
            {target: float(index) for target in ("AT", "CH", "DE", "FR", "IT")}
            for index in range(200)
        }
        forecast_hour_100 = issue + timedelta(hours=101)
        names, values = build_features(
            target_history=history, db_tables={"edh.input.cross_border_exchanges": {
                issue + timedelta(hours=index): {"AT-CH": float(index)} for index in range(200)
            }, "edh.input.generation_forecast": {
                issue + timedelta(hours=index): {"generation_forecast": float(index)} for index in range(200)
            }}, lseg=LSEGFeatures({}, {}, {}, ()), stamp=forecast_hour_100,
            issue=issue, use_lseg=False,
        )
        features = dict(zip(names, values))
        self.assertIsNone(features["target_AT_lag_1h"])
        self.assertIsNone(features["source_cross_border_exchanges_AT_CH_value"])
        self.assertIsNone(features["source_generation_forecast_generation_forecast_value"])

    def test_forbidden_generation_fields_are_rejected(self):
        issue = datetime(2024, 1, 1, tzinfo=timezone.utc)
        history = {issue: {target: 1.0 for target in ("AT", "CH", "DE", "FR", "IT")}}
        with self.assertRaises(ValueError):
            build_features(
                target_history=history,
                db_tables={"edh.input.generation_forecast": {
                    issue: {"actual_generation": 10.0}
                }},
                lseg=LSEGFeatures({}, {}, {}, ()), stamp=issue,
                issue=issue, use_lseg=False,
            )

    def test_integer_samples_are_deterministic_and_exactly_300(self):
        a = _sample(10.5, (-1.0, 2.0), 123)
        b = _sample(10.5, (-1.0, 2.0), 123)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 300)
        self.assertTrue(all(isinstance(value, int) for value in a))


if __name__ == "__main__":
    unittest.main()
