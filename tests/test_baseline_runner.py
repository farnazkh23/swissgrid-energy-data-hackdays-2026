import unittest
from datetime import datetime, timezone

from swissgrid_forecaster.baseline_runner import run_baselines
from swissgrid_forecaster.pipeline import PipelineConfig, build_feature_samples, ingest_mock_evidence


class BaselineRunnerTests(unittest.TestCase):
    def result(self):
        config = PipelineConfig(datetime(2026, 1, 10, tzinfo=timezone.utc), seed=23, include_ridge=False)
        evidence = ingest_mock_evidence(config)
        features = build_feature_samples(config, evidence)
        from swissgrid_forecaster.splits import RollingOrigin
        from datetime import timedelta
        plan = RollingOrigin(config.issue_time - timedelta(hours=72), config.issue_time,
                             timedelta(hours=48), timedelta(hours=12), timedelta(hours=6),
                             timedelta(hours=24), purge=timedelta(hours=1), embargo=timedelta(hours=1))
        return run_baselines(plan, features.samples, features.registry.manifest_hash, include_ridge=False)

    def test_scoreboard_uses_oof_predictions_only(self):
        result = self.result()
        self.assertTrue(all(entry["scored_from"] == "rolling_oof_predictions_only"
                            for entry in result.scoreboard.values()))
        for fold in result.oof.folds:
            train_ids = {row.row_id for row in fold.train}
            self.assertTrue(all(record.row_id not in train_ids for record in result.oof.records
                                if record.fold_id == fold.fold_id))

    def test_candidates_and_metrics_are_real(self):
        result = self.result()
        self.assertEqual(set(result.scoreboard), {"persistence", "seasonal_persistence", "historical_conditional"})
        self.assertIn("mae", result.scoreboard[result.selected_model[0]]["metrics"])
        self.assertNotIn("pinball_p50", result.scoreboard["persistence"]["metrics"])
        self.assertFalse(result.capabilities.get("ridge", {}).get("available", False))

