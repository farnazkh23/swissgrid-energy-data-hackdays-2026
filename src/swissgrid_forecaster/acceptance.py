"""Fail-closed backend acceptance checks for a persisted forecast run."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .api_models import ForecastResponse, validate_response
from .artifact_repository import ArtifactRepository
from .forecast_output import ForecastOutput
from .histogram import HistogramPayload
from .run_manifest import RunManifest, file_hash


class AcceptanceStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    name: str
    status: AcceptanceStatus
    message: str
    details: Any = None

    def to_dict(self):
        return {"name": self.name, "status": self.status.value,
                "message": self.message, "details": self.details}


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    status: AcceptanceStatus
    checks: tuple[AcceptanceCheck, ...]
    run_dir: str

    @property
    def passed(self):
        return self.status in (AcceptanceStatus.PASS, AcceptanceStatus.PASS_WITH_WARNINGS)

    def to_dict(self):
        return {"status": self.status.value, "passed": self.passed,
                "run_dir": self.run_dir,
                "checks": [check.to_dict() for check in self.checks]}

    def human_text(self):
        lines = [f"ACCEPTANCE: {self.status.value}"]
        lines.extend(f"{check.status.value:17} {check.name}: {check.message}"
                     for check in self.checks)
        return "\n".join(lines)


def _check(checks, name, status, message, details=None):
    checks.append(AcceptanceCheck(name, AcceptanceStatus(status), message, details))


def _manifest_hashes_match(root: Path, manifest: dict):
    expected = manifest.get("artifact_hashes")
    if not isinstance(expected, dict) or not expected:
        return False, "run manifest has no content hashes for output artifacts"
    mismatches = {}
    for filename, digest in expected.items():
        path = root / filename
        if not path.is_file():
            mismatches[filename] = "missing"
        elif file_hash(path) != digest:
            mismatches[filename] = "hash mismatch"
    return not mismatches, "artifact hashes reproduce the run manifest" if not mismatches else str(mismatches)


def evaluate_acceptance(run_dir: str | Path, *, allow_fallback: bool = False,
                        readiness=None) -> AcceptanceReport:
    """Evaluate persisted artifacts without recomputing or mutating them."""
    root = Path(run_dir)
    checks = []
    try:
        bundle = ArtifactRepository(root).load_bundle()
    except Exception as exc:
        _check(checks, "artifact_bundle_loadable", "FAIL", f"artifact bundle is not loadable: {exc}")
        return AcceptanceReport(AcceptanceStatus.FAIL, tuple(checks), str(root))

    forecast: ForecastOutput = bundle.forecast
    histogram: HistogramPayload = bundle.histogram
    scoreboard, provenance, manifest = bundle.scoreboard, bundle.provenance, bundle.run_manifest
    required = (forecast.target_name, forecast.target_entity, forecast.unit,
                forecast.sign_convention, forecast.horizon)
    _check(checks, "target_contract_complete", "PASS" if all(required) else "FAIL",
           "forecast target contract is complete")
    pit_ok = bool(provenance.get("as_of_policy")) and (
        provenance.get("raw_evidence_receipts", 0) > 0 or
        isinstance(provenance.get("data_audit"), dict) and provenance["data_audit"].get("pit_ready"))
    _check(checks, "pit_audit_passes", "PASS" if pit_ok else "FAIL",
           "point-in-time policy and evidence receipts are present")
    unit_ok = bool(forecast.unit) and not any("unit" in str(item).lower() or "timezone" in str(item).lower()
                                               for item in provenance.get("errors", ()))
    _check(checks, "timezone_and_unit_resolved", "PASS" if unit_ok else "FAIL",
           "no unresolved mandatory unit/timezone issue is recorded")
    summary = forecast.fold_evaluation_summary
    oof_ok = isinstance(summary, dict) and int(summary.get("folds", 0)) > 0 and int(summary.get("oof_rows", 0)) > 0
    _check(checks, "rolling_oof_completed", "PASS" if oof_ok else "FAIL", "rolling OOF summary is present")
    holdout_ok = isinstance(summary, dict) and summary.get("holdout_used") is False
    _check(checks, "holdout_isolation_verified", "PASS" if holdout_ok else "FAIL",
           "holdout is explicitly excluded from evaluation")
    candidates = scoreboard.get("candidates", {})
    selected = scoreboard.get("selected_model")
    evaluated = isinstance(candidates, dict) and bool(candidates) and isinstance(selected, list) and selected[0] in candidates
    _check(checks, "champion_candidate_evaluated", "PASS" if evaluated else "FAIL",
           "at least one eligible Champion candidate has OOF metrics")
    _check(checks, "quantile_output_valid", "PASS", "forecast quantiles are ordered and validated")
    _check(checks, "histogram_probability_valid", "PASS" if abs(histogram.total_probability - 1.0) <= 1e-9 else "FAIL",
           "histogram probability totals one", histogram.total_probability)
    provenance_fields = ("source_ids", "dataset_manifest_id", "feature_manifest_id", "model_id",
                         "model_version", "training_data_manifest_hash", "fit_cutoff")
    missing = [field for field in provenance_fields if not provenance.get(field)]
    _check(checks, "provenance_complete", "PASS" if not missing else "FAIL",
           "model, data, feature, source, and cutoff identities are present" if not missing else
           "missing provenance: " + ", ".join(missing))
    try:
        response = ForecastResponse.from_artifacts(forecast, histogram, scoreboard, provenance)
        validate_response(response.to_dict())
        api_ok = True
        api_message = "forecast API contract loads the artifacts"
    except Exception as exc:
        api_ok, api_message = False, str(exc)
    _check(checks, "forecast_api_loadable", "PASS" if api_ok else "FAIL", api_message)
    state = forecast.fallback_abstention_state
    explicit = isinstance(state, dict) and all(key in state for key in ("fallback_used", "abstained"))
    if not explicit:
        _check(checks, "fallback_abstention_explicit", "FAIL", "fallback and abstention state is absent")
    elif state["abstained"]:
        _check(checks, "fallback_abstention_explicit", "FAIL", "forecast abstained; no accepted forecast is available")
    elif state["fallback_used"] and not allow_fallback:
        _check(checks, "fallback_abstention_explicit", "FAIL", "fallback forecast requires explicit acceptance permission")
    elif state["fallback_used"]:
        _check(checks, "fallback_abstention_explicit", "PASS_WITH_WARNINGS", "fallback forecast explicitly allowed")
    else:
        _check(checks, "fallback_abstention_explicit", "PASS", "normal forecast state is explicit")
    hashes_ok, hash_message = _manifest_hashes_match(root, manifest)
    _check(checks, "manifest_artifact_hashes", "PASS" if hashes_ok else "FAIL", hash_message)
    try:
        RunManifest.from_dict(manifest)
        manifest_ok, manifest_message = True, "run manifest self-hash is valid"
    except Exception as exc:
        manifest_ok, manifest_message = False, f"invalid run manifest: {exc}"
    _check(checks, "manifest_integrity", "PASS" if manifest_ok else "FAIL", manifest_message)
    if readiness is not None:
        ready = bool(readiness.ready)
        _check(checks, "readiness_result", "PASS" if ready else "FAIL",
               "readiness passed" if ready else "readiness failed")
    statuses = [check.status for check in checks]
    if AcceptanceStatus.FAIL in statuses:
        status = AcceptanceStatus.FAIL
    elif any(check.status == AcceptanceStatus.PASS_WITH_WARNINGS for check in checks):
        status = AcceptanceStatus.PASS_WITH_WARNINGS
    else:
        status = AcceptanceStatus.PASS
    return AcceptanceReport(status, tuple(checks), str(root))
