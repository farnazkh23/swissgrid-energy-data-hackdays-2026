import unittest
from datetime import datetime, timedelta, timezone

from swissgrid_forecaster.specialist_contracts import EvidenceBrief, TargetForecastBrief, EVIDENCE_BRIEF, TARGET_FORECAST_BRIEF
from swissgrid_forecaster.specialist_registry import SpecialistDefinition, SpecialistRegistry
from swissgrid_forecaster.specialist_runner import SpecialistRunContext, SpecialistRunner


T = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


class EvidenceSpecialist:
    specialist_id = "demand-1"

    def run(self, context):
        return EvidenceBrief(self.specialist_id, context.issue_time, (H,), "Demand",
                             {"state": "stable"}, known_at=context.issue_time)


class ForecastSpecialist:
    specialist_id = "model-1"

    def run(self, context):
        return TargetForecastBrief(self.specialist_id, context.issue_time, context.target_time,
                                   context.horizon, "model", "v1", context.issue_time,
                                   point_prediction=1, feature_manifest_id="features")


def registry():
    return SpecialistRegistry([
        SpecialistDefinition("demand-1", "Demand", "v1", (EVIDENCE_BRIEF,), (H,)),
        SpecialistDefinition("model-1", "Demand", "v1", (TARGET_FORECAST_BRIEF,), (H,)),
    ])


class SpecialistRunnerTests(unittest.TestCase):
    def test_valid_typed_briefs_run(self):
        runner = SpecialistRunner(registry())
        context = SpecialistRunContext(T, T+H, H, T)
        self.assertIsInstance(runner.run("demand-1", EvidenceSpecialist(), context), EvidenceBrief)
        self.assertIsInstance(runner.run("model-1", ForecastSpecialist(), context), TargetForecastBrief)

    def test_raw_transcript_and_wrong_identity_fail_closed(self):
        runner = SpecialistRunner(registry())
        context = SpecialistRunContext(T)
        with self.assertRaises(ValueError):
            runner.run("demand-1", lambda _: {"transcript": "raw"}, context)
        with self.assertRaises(ValueError):
            runner.run("demand-1", ForecastSpecialist(), context)

    def test_runner_checks_context_and_family(self):
        runner = SpecialistRunner(registry())
        context = SpecialistRunContext(T, known_at_cutoff=T)
        self.assertIsInstance(runner.run("demand-1", EvidenceSpecialist(), context), EvidenceBrief)

        class BadFamily(EvidenceSpecialist):
            def run(self, context):
                return EvidenceBrief(self.specialist_id, context.issue_time, (H,), "Market", {}, known_at=T)

        with self.assertRaises(ValueError):
            runner.run("demand-1", BadFamily(), context)

    def test_run_many_is_deterministic_and_rejects_empty(self):
        runner = SpecialistRunner(registry())
        context = SpecialistRunContext(T, T+H, H, T)
        result = runner.run_many({"model-1": ForecastSpecialist(), "demand-1": EvidenceSpecialist()}, context)
        self.assertEqual([brief.specialist_id for brief in result], ["demand-1", "model-1"])
        with self.assertRaises(ValueError):
            runner.run_many({}, context)

    def test_future_context_fails(self):
        with self.assertRaises(ValueError):
            SpecialistRunContext(T, known_at_cutoff=T+H)
