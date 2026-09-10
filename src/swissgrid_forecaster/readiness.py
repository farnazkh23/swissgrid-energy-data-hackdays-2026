"""Deterministic preflight checks for data, configuration, and artifact readiness."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import os
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .artifact_repository import ArtifactRepository
from .api_models import ForecastResponse, validate_response
from .baseline_runner import baseline_factories
from .data_audit import DataAuditReport, audit_dataset
from .real_dataset_loader import RealDataset


class ReadinessStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    status: ReadinessStatus
    mandatory: bool
    message: str
    details: Any = None

    def to_dict(self):
        return {"name": self.name, "status": self.status.value, "mandatory": self.mandatory,
                "message": self.message, "details": self.details}


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    status: ReadinessStatus
    checks: tuple[ReadinessCheck, ...]
    dataset_id: str | None = None

    @property
    def ready(self) -> bool:
        return self.status != ReadinessStatus.FAIL

    def to_dict(self):
        return {"status": self.status.value, "ready": self.ready,
                "dataset_id": self.dataset_id,
                "checks": [check.to_dict() for check in self.checks]}

    def human_text(self) -> str:
        lines = [f"READINESS: {self.status.value}"]
        for check in self.checks:
            suffix = " [mandatory]" if check.mandatory else ""
            lines.append(f"{check.status.value:7} {check.name}{suffix}: {check.message}")
        return "\n".join(lines)


def _check(checks, name, status, mandatory, message, details=None):
    checks.append(ReadinessCheck(name, ReadinessStatus(status), mandatory, message, details))


def _placeholder(value):
    return value is None or (isinstance(value, str) and (not value.strip() or value.strip().startswith("<") or value.strip().startswith("REPLACE_")))


def _get_target(config: Mapping):
    target = config.get("target_contract", config.get("target", {}))
    return target if isinstance(target, Mapping) else {}


def _get_mapping(config: Mapping):
    mapping = config.get("column_mapping", config.get("mapping", {}))
    return mapping if isinstance(mapping, Mapping) else {}


def _get_plan(config: Mapping):
    plan = config.get("fold_config", config.get("plan", {}))
    return plan if isinstance(plan, Mapping) else {}


def _seconds(value):
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _config_duration(plan, name, fallback=None):
    direct = plan.get(name)
    if direct is None:
        direct = plan.get(f"{name}_seconds", fallback)
    return _seconds(direct)


def _output_writable(path):
    path = Path(path)
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


def _artifact_compatibility(run_dir):
    try:
        bundle = ArtifactRepository(run_dir).load_bundle()
        response = ForecastResponse.from_artifacts(bundle.forecast, bundle.histogram,
                                                   bundle.scoreboard, bundle.provenance)
        validate_response(response.to_dict())
    except Exception as exc:
        return ReadinessStatus.FAIL, f"API artifact bundle is not loadable: {exc}"
    return ReadinessStatus.PASS, "forecast artifacts load through the serving API contract"


def _artifact_readiness(config, run_dir, checks):
    try:
        bundle = ArtifactRepository(run_dir).load_bundle()
    except Exception as exc:
        _check(checks, "api_artifact_compatibility", "FAIL", True, f"artifact bundle cannot be loaded: {exc}")
        return None
    forecast = bundle.forecast
    _check(checks, "target_contract_resolved", "PASS", True, "forecast artifact contains validated target identity")
    _check(checks, "target_column_exists", "PASS", True, "forecast artifact contains target output")
    _check(checks, "target_units_resolved", "PASS", True, f"forecast unit is {forecast.unit}")
    _check(checks, "sign_convention_resolved", "PASS", True, forecast.sign_convention)
    for name in ("timezone_dst_validity", "dataset_date_range", "cadence", "gaps", "duplicates",
                 "missingness", "source_coverage", "pit_known_at_completeness", "revisions",
                 "feature_availability_by_horizon", "sufficient_history", "sufficient_labels"):
        _check(checks, name, "WARNING", False, "raw dataset audit not available in artifact-only readiness mode")
    _check(checks, "champion_candidates_available", "PASS" if bundle.scoreboard.get("candidates") else "FAIL", True,
           "scoreboard contains evaluated candidates" if bundle.scoreboard.get("candidates") else "scoreboard has no candidates")
    output = config.get("output_directory", config.get("output_dir", run_dir))
    _check(checks, "output_directory_writable", "PASS" if _output_writable(output) else "FAIL", True,
           str(output))
    status, message = _artifact_compatibility(run_dir)
    _check(checks, "api_artifact_compatibility", status, True, message)
    return forecast


def evaluate_readiness(dataset: RealDataset | None = None, config: Mapping | None = None, *,
                       run_dir: str | Path | None = None, output_dir: str | Path | None = None) -> ReadinessReport:
    """Evaluate readiness without fitting or mutating model state."""
    if not isinstance(config, Mapping):
        raise ValueError("run configuration object is required")
    checks = []
    dataset_id = dataset.dataset_id if isinstance(dataset, RealDataset) else None
    if run_dir is not None:
        _artifact_readiness(config, run_dir, checks)
    target = _get_target(config)
    required_target = ("target_name", "target_entity", "unit", "sign_convention",
                       "resolution_seconds", "forecast_issue_time", "target_valid_time",
                       "horizon_seconds", "aggregation_rule", "label_availability_rule",
                       "forecast_type", "evaluation_metrics")
    target_missing = [field for field in required_target if _placeholder(target.get(field))]
    if run_dir is None:
        _check(checks, "target_contract_resolved", "FAIL" if target_missing else "PASS", True,
               "missing fields: " + ", ".join(target_missing) if target_missing else "explicit target contract fields are complete")
        mapping = _get_mapping(config)
        columns = mapping.get("columns", {}) if isinstance(mapping.get("columns", {}), Mapping) else {}
        value_column = columns.get("value") or mapping.get("target_column")
        present = isinstance(value_column, str) and isinstance(dataset, RealDataset) and all(value_column in row for row in dataset.raw_rows)
        _check(checks, "target_column_exists", "PASS" if present else "FAIL", True,
               value_column or "canonical value mapping is unresolved")
        unit = target.get("unit")
        _check(checks, "target_units_resolved", "PASS" if not _placeholder(unit) else "FAIL", True,
               str(unit) if unit else "target unit is unresolved")
        sign = target.get("sign_convention")
        _check(checks, "sign_convention_resolved", "PASS" if not _placeholder(sign) else "FAIL", True,
               str(sign) if sign else "sign convention is unresolved")
    if isinstance(dataset, RealDataset):
        issue_value = target.get("forecast_issue_time")
        try:
            issue = datetime.fromisoformat(issue_value.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            issue = None
        if issue is None:
            _check(checks, "timezone_dst_validity", "FAIL", True, "forecast issue time is invalid or timezone-naive")
        else:
            audit = audit_dataset(dataset, issue_time=issue)
            timezone = audit.timezone_dst_issues
            timezone_status = "FAIL" if timezone.get("unparseable_timestamp_values") else ("WARNING" if timezone.get("naive_timestamp_values") else "PASS")
            _check(checks, "timezone_dst_validity", timezone_status, True,
                   "raw timestamps are timezone-aware" if timezone_status == "PASS" else "timezone/DST review required", timezone)
            _check(checks, "dataset_date_range", "PASS" if audit.rows else "FAIL", True,
                   str(audit.date_range), audit.date_range)
            _check(checks, "cadence", "PASS" if any(value is not None for value in audit.cadence_seconds.values()) else "WARNING", False,
                   "cadence computed", audit.cadence_seconds)
            _check(checks, "gaps", "WARNING" if any(audit.gaps.values()) else "PASS", False,
                   "gaps require review" if any(audit.gaps.values()) else "no material gaps detected", audit.gaps)
            _check(checks, "duplicates", "WARNING" if audit.duplicates["duplicate_rows_after_first"] else "PASS", False,
                   str(audit.duplicates), audit.duplicates)
            _check(checks, "missingness", "WARNING" if any(item["missing_rows"] for item in audit.missingness.values()) else "PASS", False,
                   "missing values detected" if any(item["missing_rows"] for item in audit.missingness.values()) else "no missing values", audit.missingness)
            expected_sources = config.get("expected_source_ids", config.get("feature_source_ids", ()))
            present_sources = {row.source_id for row in dataset.observations}
            missing_sources = sorted(set(expected_sources) - present_sources)
            _check(checks, "source_coverage", "FAIL" if missing_sources else "PASS", True,
                   "missing sources: " + ", ".join(missing_sources) if missing_sources else "configured sources present")
            fallback_allowed = config.get("publication_rules", {}).get("reconstruction_allowed", False) if isinstance(config.get("publication_rules", {}), Mapping) else False
            if dataset.known_at_policy != "require" and not fallback_allowed:
                _check(checks, "documented_publication_fallback", "FAIL", True, "fallback used but reconstruction is not explicitly allowed")
            elif dataset.known_at_policy != "require":
                _check(checks, "documented_publication_fallback", "WARNING", True, "known_at reconstructed under explicit publication rule")
            else:
                _check(checks, "documented_publication_fallback", "PASS", False, "not required; known_at supplied")
            _check(checks, "pit_known_at_completeness", "PASS" if audit.pit_ready else "FAIL", True,
                   "known_at and receipt/normalization clocks are complete" if audit.pit_ready else "PIT metadata unresolved")
            _check(checks, "revisions", "PASS", False, "revision diagnostics computed", audit.revision_frequency)
            horizon = _seconds(target.get("horizon_seconds"))
            feature_horizon = audit.feature_availability_by_forecast_horizon
            _check(checks, "feature_availability_by_horizon", "PASS" if feature_horizon else "WARNING", False,
                   "horizon availability audited", feature_horizon)
            plan = _get_plan(config)
            first_origin = target_time = None
            try:
                first_origin = datetime.fromisoformat(plan["first_origin"].replace("Z", "+00:00"))
                target_time = datetime.fromisoformat(plan["holdout_start"].replace("Z", "+00:00"))
            except (KeyError, AttributeError, ValueError):
                pass
            train_window = _config_duration(plan, "train_window", 0)
            enough_history = bool(first_origin and audit.date_range[0] and train_window is not None and
                                  datetime.fromisoformat(audit.date_range[0]) <= first_origin - timedelta(seconds=train_window))
            _check(checks, "sufficient_history", "PASS" if enough_history else "FAIL", True,
                   "dataset begins before first rolling training window" if enough_history else "insufficient history for rolling folds")
            target_source = config.get("target_source_id")
            label_count = sum(row.source_id == target_source and row.value is not None for row in dataset.observations)
            minimum_labels = int(config.get("minimum_label_count", 1))
            _check(checks, "sufficient_labels", "PASS" if label_count >= minimum_labels else "FAIL", True,
                   f"{label_count} target observations available", {"count": label_count, "minimum": minimum_labels})
    elif run_dir is None:
        configured_timezone = config.get("timezone") or _get_target(config).get("timezone")
        if configured_timezone:
            try:
                ZoneInfo(configured_timezone)
                timezone_status, timezone_message = "PASS", f"configured timezone is valid: {configured_timezone}"
            except (ZoneInfoNotFoundError, ValueError):
                timezone_status, timezone_message = "FAIL", f"invalid IANA timezone: {configured_timezone}"
            _check(checks, "timezone_dst_validity", timezone_status, True, timezone_message)
        for name in ("timezone_dst_validity", "dataset_date_range", "cadence", "gaps", "duplicates", "missingness",
                     "source_coverage", "pit_known_at_completeness", "revisions", "feature_availability_by_horizon",
                     "sufficient_history", "sufficient_labels"):
            if name == "timezone_dst_validity" and configured_timezone:
                continue
            _check(checks, name, "FAIL", True, "dataset is unavailable")
    candidates = tuple(config.get("enabled_champion_candidates", ("persistence", "seasonal_persistence", "historical_conditional", "ridge")))
    _, capabilities = baseline_factories(include_ridge=True)
    available = [candidate for candidate in candidates if capabilities.get(candidate, {}).get("available")]
    _check(checks, "champion_candidates_available", "PASS" if available else "FAIL", True,
           ", ".join(available) if available else "no enabled Champion candidate is available")
    output = output_dir or config.get("output_directory", config.get("output_dir", "artifacts/real_run"))
    _check(checks, "output_directory_writable", "PASS" if _output_writable(output) else "FAIL", True, str(output))
    if run_dir is None:
        _check(checks, "api_artifact_compatibility", "WARNING", False, "checked after model artifacts are produced")
    statuses = [check.status for check in checks]
    status = ReadinessStatus.FAIL if any(status == ReadinessStatus.FAIL and check.mandatory for status, check in zip(statuses, checks)) else (
        ReadinessStatus.WARNING if ReadinessStatus.WARNING in statuses else ReadinessStatus.PASS)
    return ReadinessReport(status, tuple(checks), dataset_id)
