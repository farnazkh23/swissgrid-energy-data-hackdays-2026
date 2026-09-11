import unittest
from datetime import datetime, timedelta, timezone

from swissgrid_forecaster.real_modeling_v1 import (
    LSEGFeatures,
    _sample,
    build_hourly_targets,
    build_features,
    _seasonal_persistence_value,
    make_weekly_folds,
)
from swissgrid_forecaster.time_contract import canonical_utc, organizer_timestamp


class RealModelingV1Tests(unittest.TestCase):
    def test_duplicate_quarter_is_not_emitted(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [{"Zeitstempel": start + timedelta(minutes=15 * index), **{target: index for target in ("AT", "CH", "DE", "FR", "IT")}}
                for index in (0, 1, 2, 3, 3)]
        self.assertEqual(len(build_hourly_targets(rows)), 0)

    def test_missing_optional_at_does_not_drop_scored_target_hour(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = [{"Zeitstempel": start + timedelta(minutes=15 * index),
                 **{target: float(index) for target in ("CH", "DE", "FR", "IT")}}
                for index in range(4)]
        hourly = build_hourly_targets(rows)
        self.assertEqual(tuple(hourly), (start,))
        self.assertEqual(set(hourly[start]), {"CH", "DE", "FR", "IT"})
        self.assertNotIn("AT", hourly[start])

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
        self.assertIsNone(features["target_CH_lag_1h"])
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

    def test_europe_zurich_dst_wall_clock_contract(self):
        spring_valid = datetime(2026, 3, 29, 1, 45, tzinfo=timezone.utc).replace(tzinfo=None)
        spring_next = datetime(2026, 3, 29, 3, 0, tzinfo=timezone.utc).replace(tzinfo=None)
        self.assertEqual(canonical_utc(spring_valid, "Europe/Zurich"),
                         datetime(2026, 3, 29, 0, 45, tzinfo=timezone.utc))
        self.assertEqual(canonical_utc(spring_next, "Europe/Zurich"),
                         datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            canonical_utc(datetime(2026, 3, 29, 2, 0), "Europe/Zurich")
        with self.assertRaises(ValueError):
            canonical_utc(datetime(2025, 10, 26, 2, 0), "Europe/Zurich")
        self.assertEqual(organizer_timestamp(datetime(2026, 8, 20, 23, 0)),
                         datetime(2026, 8, 20, 23, 0, tzinfo=timezone.utc))

    def test_seasonal_persistence_falls_back_across_missing_dst_reference(self):
        issue = datetime(2026, 4, 5, 23, tzinfo=timezone.utc)
        target = datetime(2026, 4, 5, 2, tzinfo=timezone.utc)
        missing = datetime(2026, 3, 29, 2, tzinfo=timezone.utc)
        series = {issue - timedelta(hours=index): float(index) for index in range(672)}
        series.pop(missing, None)
        value, method = _seasonal_persistence_value(series, target, issue)
        self.assertEqual(method, "nearby_167h")
        self.assertEqual(value, series[target - timedelta(hours=167)])
        again = _seasonal_persistence_value(series, target, issue)
        self.assertEqual((value, method), again)

    def test_seasonal_persistence_uses_exact_lag_normally(self):
        issue = datetime(2026, 4, 5, 23, tzinfo=timezone.utc)
        target = datetime(2026, 4, 5, 2, tzinfo=timezone.utc)
        series = {target - timedelta(hours=168): 42.0, issue: 99.0}
        self.assertEqual(_seasonal_persistence_value(series, target, issue), (42.0, "exact_168h"))


if __name__ == "__main__":
    unittest.main()
