import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from swissgrid_forecaster.readiness import ReadinessStatus, evaluate_readiness
from swissgrid_forecaster.real_dataset_loader import load_real_dataset
from test_run_real_champion import build_file, source_config


class ReadinessTests(unittest.TestCase):
    def config(self, **changes):
        value = {"target_contract": {"target_name": "net", "target_entity": "CH", "unit": "MW",
                  "sign_convention": "configured", "resolution_seconds": 3600,
                  "forecast_issue_time": "2026-01-01T19:00:00+00:00",
                  "target_valid_time": "2026-01-01T20:00:00+00:00", "horizon_seconds": 3600,
                  "aggregation_rule": "identity", "label_availability_rule": "known_at <= valid_time",
                  "forecast_type": "point", "evaluation_metrics": ["mae"]},
                  "column_mapping": {"columns": {"value": "v"}},
                  "target_source_id": "net", "expected_source_ids": ["net", "load"],
                  "feature_source_ids": ["load"], "minimum_label_count": 1,
                  "fold_config": {"first_origin": "2026-01-01T08:00:00+00:00",
                                   "holdout_start": "2026-01-01T19:00:00+00:00",
                                   "train_window_seconds": 28800},
                  "output_directory": "."}
        value.update(changes)
        return value

    def test_unresolved_target_sign_and_missing_dataset_block(self):
        config = self.config()
        config["target_contract"] = dict(config["target_contract"], target_name="<target>", sign_convention="")
        report = evaluate_readiness(config=config)
        self.assertEqual(report.status, ReadinessStatus.FAIL)
        self.assertFalse(report.ready)

    def test_complete_dataset_is_ready(self):
        with self.subTest("complete"):
            import tempfile
            with tempfile.TemporaryDirectory() as directory:
                dataset = load_real_dataset(build_file(directory), source_config=source_config())
                report = evaluate_readiness(dataset, self.config())
                self.assertTrue(report.ready)
                self.assertNotEqual(report.status, ReadinessStatus.FAIL)

    def test_missing_pit_and_fallback_policy(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_real_dataset(build_file(directory), source_config=source_config())
            failed = evaluate_readiness(replace(dataset, pit_ready=False), self.config())
            self.assertEqual(failed.status, ReadinessStatus.FAIL)
            fallback_config = self.config(publication_rules={"reconstruction_allowed": True})
            fallback = evaluate_readiness(replace(dataset, known_at_policy="publication_plus_lag"), fallback_config)
            self.assertEqual(fallback.status, ReadinessStatus.WARNING)
            check = next(item for item in fallback.checks if item.name == "documented_publication_fallback")
            self.assertEqual(check.status, ReadinessStatus.WARNING)

    def test_insufficient_history_missing_target_and_invalid_timezone(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_real_dataset(build_file(directory), source_config=source_config())
            config = self.config(fold_config={"first_origin": "2026-01-01T07:00:00+00:00",
                                             "holdout_start": "2026-01-01T19:00:00+00:00",
                                             "train_window_seconds": 28800})
            report = evaluate_readiness(dataset, config)
            self.assertEqual(next(item for item in report.checks if item.name == "sufficient_history").status,
                             ReadinessStatus.FAIL)
            missing = evaluate_readiness(dataset, dict(config, target_source_id="absent"))
            self.assertEqual(next(item for item in missing.checks if item.name == "sufficient_labels").status,
                             ReadinessStatus.FAIL)
        invalid = evaluate_readiness(config=dict(self.config(), timezone="Not/AZone"))
        self.assertEqual(next(item for item in invalid.checks if item.name == "timezone_dst_validity").status,
                         ReadinessStatus.FAIL)
