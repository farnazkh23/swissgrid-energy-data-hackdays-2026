import unittest
from datetime import datetime, timedelta, timezone
from dataclasses import replace

from swissgrid_forecaster.specialist_contracts import (
    CalibrationEvidence,
    EvidenceBrief,
    TargetForecastBrief,
)


T = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = timedelta(hours=1)


class SpecialistContractTests(unittest.TestCase):
    def evidence(self, **changes):
        fields = dict(specialist_id="demand-1", issue_time=T, horizon_applicability=(H,),
                      evidence_family="Demand", state_features={"load": 10.0}, direction="up",
                      uncertainty={"kind": "qualitative"}, freshness=timedelta(minutes=5),
                      source_ids=("src",), known_at=T)
        fields.update(changes)
        return EvidenceBrief(**fields)

    def forecast(self, **changes):
        fields = dict(specialist_id="demand-1", issue_time=T, target_time=T+H, horizon=H,
                      model_id="model", version="v1", fit_cutoff=T,
                      p10=1, p50=2, p90=3, feature_manifest_id="features-v1")
        fields.update(changes)
        return TargetForecastBrief(**fields)

    def test_evidence_brief_normalizes_and_requires_known_at(self):
        brief = self.evidence(state_features={"b": 2, "a": 1})
        self.assertEqual(brief.state_features, (("a", 1), ("b", 2)))
        self.assertEqual(brief.known_at, T)
        with self.assertRaises(ValueError):
            self.evidence(known_at=T+H)

    def test_forecast_has_explicit_representation_and_horizon(self):
        brief = self.forecast()
        self.assertEqual(brief.target_time - brief.issue_time, brief.horizon)
        with self.assertRaises(ValueError):
            self.forecast(horizon=2*H)
        with self.assertRaises(ValueError):
            self.forecast(p10=3, p50=2)

    def test_calibration_claim_requires_oof_evidence_known_in_time(self):
        with self.assertRaises(ValueError):
            self.forecast(calibration_claimed=True)
        evidence = CalibrationEvidence("eval", ("fold-1",), {"coverage": .8}, T)
        self.assertTrue(self.forecast(calibration_claimed=True, calibration_evidence=evidence).calibration_claimed)
        with self.assertRaises(ValueError):
            self.forecast(calibration_evidence=replace(evidence, known_at=T+H))

    def test_confidence_is_not_trust_or_weight(self):
        self.assertIsNotNone(self.evidence(uncertainty={"confidence": "low"}))
        self.assertEqual(self.forecast(uncertainty=.2).uncertainty, (("value", .2),))
        with self.assertRaises(ValueError):
            self.evidence(uncertainty={"trust_score": .9})
        with self.assertRaises(ValueError):
            self.forecast(uncertainty={"ensemble_weight": .5})
        with self.assertRaises(ValueError):
            self.forecast(uncertainty={"calibrated": True})
        with self.assertRaises(ValueError):
            self.evidence(quality_flags=("calibrated",))

    def test_forecast_needs_manifest_and_prediction(self):
        with self.assertRaises(ValueError):
            self.forecast(feature_manifest_id=None)
        with self.assertRaises(ValueError):
            self.forecast(p10=None, p50=None, p90=None, samples=())

    def test_malformed_briefs_fail_closed(self):
        with self.assertRaises(ValueError):
            self.evidence(horizon_applicability=())
        with self.assertRaises(ValueError):
            self.forecast(fit_cutoff=T+H)
        with self.assertRaises(ValueError):
            self.forecast(samples=(float("nan"),))
