import unittest
from datetime import datetime, timezone, timedelta
from dataclasses import replace

from swissgrid_forecaster.asof_resolver import resolve_as_of
from swissgrid_forecaster.pipeline import (PipelineConfig, build_feature_samples,
                                           run_mock_pipeline, _source_id, TARGET_DOMAIN)


ISSUE = datetime(2026, 1, 10, tzinfo=timezone.utc)


class PipelineTests(unittest.TestCase):
    def config(self):
        return PipelineConfig(ISSUE, seed=17, include_ridge=False)

    def test_complete_mock_run_succeeds(self):
        result = run_mock_pipeline(self.config())
        self.assertEqual(result.forecast.target_time, ISSUE + timedelta(hours=1))
        self.assertTrue(result.baselines.oof.records)
        self.assertEqual(result.histogram.total_probability, 1.0)

    def test_same_seed_and_config_are_identical(self):
        left, right = run_mock_pipeline(self.config()), run_mock_pipeline(self.config())
        self.assertEqual(left.forecast.to_dict(), right.forecast.to_dict())
        self.assertEqual(left.histogram.to_dict(), right.histogram.to_dict())
        self.assertEqual(left.baselines.scoreboard, right.baselines.scoreboard)
        self.assertEqual(left.provenance, right.provenance)

    def test_future_known_data_never_enters_features(self):
        result = run_mock_pipeline(self.config())
        issue = ISSUE
        used = result.features.used_observation_ids[issue.isoformat()]
        used_rows = [row for row in result.evidence.observations
                     if f"{row.source_id}:{row.source_record_id}:{row.revision_id}" in used]
        self.assertTrue(all(row.event_time <= issue and row.known_at <= issue for row in used_rows))
        self.assertTrue(all(row.issue_time <= issue for row in result.features.samples))

    def test_revised_data_resolves_correctly_as_of_issue(self):
        result = run_mock_pipeline(self.config())
        load = _source_id(__import__("swissgrid_forecaster.source_contracts", fromlist=["Domain"]).Domain.LOAD)
        revised = next(row for row in result.evidence.observations
                       if row.source_id == load and row.revision_sequence == 1
                       and any(base.logical_key == row.logical_key
                               and base.revision_sequence == 0
                               and base.known_at < row.known_at
                               for base in result.evidence.observations))
        rows = [row for row in result.evidence.observations
                if row.logical_key == revised.logical_key]
        before = resolve_as_of(rows, revised.known_at - timedelta(seconds=1))
        after = resolve_as_of(rows, revised.known_at)
        self.assertEqual(before[0].revision_sequence, 0)
        self.assertEqual(after[0].revision_sequence, 1)

    def test_missing_and_stale_data_are_explicitly_flagged(self):
        result = run_mock_pipeline(self.config())
        all_flags = [flag for issue in result.features.flags_by_issue.values() for flag in issue.values()]
        self.assertTrue(any(flag["missing"] for flag in all_flags))
        self.assertTrue(any(flag["stale"] for flag in all_flags))
        self.assertTrue(any("stale" in warning for warning in result.forecast.warnings))

    def test_holdout_is_not_used(self):
        result = run_mock_pipeline(self.config())
        self.assertTrue(all(record.issue_time < ISSUE for record in result.baselines.oof.records))
        self.assertTrue(all(record.row_id not in {row.row_id for row in result.plan.holdout(result.features.samples)}
                            for record in result.baselines.oof.records))
