from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from swissgrid_forecaster.target_contract import ForecastType, TargetContract

T = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.target = TargetContract(
            target_name='configured_flow', target_entity='CH', country='CH',
            unit='MW', sign_convention='configured by target owner',
            resolution=timedelta(hours=1), forecast_issue_time=T,
            target_valid_time=T + timedelta(hours=1), horizon=timedelta(hours=1),
            aggregation_rule='interval mean', label_availability_rule='verified release',
            forecast_type=ForecastType.QUANTILE, quantile_levels=(.1, .5, .9),
            evaluation_metrics=('pinball',),
        )

    def test_valid_configurable_contract(self):
        self.assertEqual(self.target.country, 'CH')
        self.assertFalse(self.target.scenario_support)
        self.assertEqual(replace(self.target, sign_convention='opposite convention').sign_convention,
                         'opposite convention')

    def test_horizon_mismatch(self):
        with self.assertRaises(ValueError):
            replace(self.target, horizon=timedelta(hours=2))

    def test_target_must_follow_issue(self):
        for valid in (T, T - timedelta(hours=1)):
            with self.subTest(valid=valid), self.assertRaises(ValueError):
                replace(self.target, target_valid_time=valid, horizon=valid-T)

    def test_invalid_quantiles(self):
        for levels in ((.9, .1), (.5, .5), (0, .5), (.5, 1), (float('nan'),), (), (True,)):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                replace(self.target, quantile_levels=levels)

    def test_naive_timestamps(self):
        for field in ('forecast_issue_time', 'target_valid_time'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(self.target, **{field: T.replace(tzinfo=None)})

    def test_aware_offset_normalized(self):
        target = replace(self.target, forecast_issue_time=T.astimezone(timezone(timedelta(hours=2))))
        self.assertIs(target.forecast_issue_time.tzinfo, timezone.utc)
