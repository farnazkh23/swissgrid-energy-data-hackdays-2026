"""Typed, compressed outputs for future specialist implementations.

This module deliberately contains schemas and validation only.  It does not
implement a power-system specialist, calibration, trust, or ensembling.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Mapping

from .availability import nonempty, utc


EVIDENCE_BRIEF = "EvidenceBrief"
TARGET_FORECAST_BRIEF = "TargetForecastBrief"


def _finite_number(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _scalar(value, name: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"{name} must not be blank")
        return
    _finite_number(value, name)


def _pairs(value, name: str, *, string_values: bool = False) -> tuple[tuple[str, object], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    try:
        pairs = tuple(tuple(pair) for pair in items)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain key/value pairs") from None
    result = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"{name} must contain key/value pairs")
        key, item = pair
        nonempty(key, f"{name} key")
        if string_values:
            nonempty(item, f"{name}[{key}]")
        else:
            _scalar(item, f"{name}[{key}]")
        result.append((key, item))
    if len({key for key, _ in result}) != len(result):
        raise ValueError(f"duplicate {name} keys")
    return tuple(sorted(result))


def _strings(value, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(value)
    for item in result:
        nonempty(item, name)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return result


def _durations(value, name: str, *, required: bool = False) -> tuple[timedelta, ...]:
    if value is None:
        value = ()
    result = tuple(value)
    if required and not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(item, timedelta) or item <= timedelta(0) for item in result):
        raise ValueError(f"{name} must contain positive durations")
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return tuple(sorted(result))


def _metric_pairs(value, name: str) -> tuple[tuple[str, float], ...]:
    pairs = _pairs(value, name)
    result = []
    for key, item in pairs:
        _finite_number(item, f"{name}[{key}]")
        result.append((key, float(item)))
    return tuple(result)


def _uncertainty_pairs(value) -> tuple[tuple[str, object], ...]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite_number(value, "uncertainty")
        return (("value", float(value)),)
    return _pairs(value, "uncertainty")


def _validate_uncertainty_metadata(pairs: tuple[tuple[str, object], ...], *, calibration_allowed: bool) -> bool:
    """Reserve trust/weight claims for future downstream components."""
    calibration_claim = False
    for key, value in pairs:
        normalized = key.casefold().replace("_", "").replace("-", "")
        if normalized in {"weight", "ensembleweight", "specialistweight", "trust", "trustscore"}:
            raise ValueError("specialist output cannot assign trust or ensemble weight")
        if normalized in {"calibrated", "calibrationclaim", "calibrationclaimed"}:
            calibration_claim = value is True or (isinstance(value, str) and value.casefold() in {
                "true", "calibrated", "supported"
            })
            if calibration_claim and not calibration_allowed:
                raise ValueError("calibration claim requires attached OOF evidence")
    return calibration_claim


def _validate_quality_flags(flags: tuple[str, ...]) -> None:
    for flag in flags:
        normalized = flag.casefold().replace("_", "").replace("-", "")
        if normalized in {"calibrated", "calibrationsupported", "calibrationclaim"}:
            raise ValueError("calibration claim requires attached OOF evidence")


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    """Attached evidence required before a forecast may claim calibration."""

    evaluation_id: str
    oof_fold_ids: tuple[str, ...]
    metrics: tuple[tuple[str, float], ...]
    known_at: datetime
    method: str = "OOF"

    def __post_init__(self) -> None:
        nonempty(self.evaluation_id, "evaluation_id")
        object.__setattr__(self, "oof_fold_ids", _strings(self.oof_fold_ids, "oof_fold_ids"))
        if not self.oof_fold_ids:
            raise ValueError("calibration evidence requires OOF folds")
        object.__setattr__(self, "metrics", _metric_pairs(self.metrics, "metrics"))
        if not self.metrics:
            raise ValueError("calibration evidence requires metrics")
        object.__setattr__(self, "known_at", utc(self.known_at, "known_at"))
        nonempty(self.method, "method")
        if self.method.upper() != "OOF":
            raise ValueError("calibration evidence must be OOF evidence")


@dataclass(frozen=True, slots=True)
class EvidenceBrief:
    """A compact state/evidence report, not necessarily a target forecast."""

    specialist_id: str
    issue_time: datetime
    horizon_applicability: tuple[timedelta, ...]
    evidence_family: str
    state_features: tuple[tuple[str, object], ...] = ()
    direction: str | None = None
    tendency: str | None = None
    uncertainty: tuple[tuple[str, object], ...] = ()
    freshness: timedelta | None = None
    missingness: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    known_at: datetime | None = None
    quality_flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        nonempty(self.specialist_id, "specialist_id")
        object.__setattr__(self, "issue_time", utc(self.issue_time, "issue_time"))
        object.__setattr__(self, "horizon_applicability",
                           _durations(self.horizon_applicability, "horizon_applicability", required=True))
        nonempty(self.evidence_family, "evidence_family")
        object.__setattr__(self, "state_features", _pairs(self.state_features, "state_features"))
        for name in ("direction", "tendency"):
            value = getattr(self, name)
            if value is not None:
                nonempty(value, name)
        uncertainty = _uncertainty_pairs(self.uncertainty)
        _validate_uncertainty_metadata(uncertainty, calibration_allowed=False)
        object.__setattr__(self, "uncertainty", uncertainty)
        if self.freshness is not None and (not isinstance(self.freshness, timedelta) or self.freshness < timedelta(0)):
            raise ValueError("freshness must be a nonnegative duration")
        object.__setattr__(self, "missingness", _strings(self.missingness, "missingness"))
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        cutoff = utc(self.known_at, "known_at") if self.known_at is not None else None
        if cutoff is None:
            raise ValueError("known_at cutoff is required")
        if cutoff > self.issue_time:
            raise ValueError("future-informed evidence brief")
        object.__setattr__(self, "known_at", cutoff)
        quality_flags = _strings(self.quality_flags, "quality_flags")
        _validate_quality_flags(quality_flags)
        object.__setattr__(self, "quality_flags", quality_flags)
        object.__setattr__(self, "warnings", _strings(self.warnings, "warnings"))
        object.__setattr__(self, "provenance", _pairs(self.provenance, "provenance", string_values=True))

    @property
    def brief_type(self) -> str:
        return EVIDENCE_BRIEF

    @property
    def features(self) -> tuple[tuple[str, object], ...]:
        return self.state_features

    @property
    def known_at_cutoff(self) -> datetime:
        return self.known_at


@dataclass(frozen=True, slots=True)
class TargetForecastBrief:
    """A compact target forecast with explicit point-in-time provenance."""

    specialist_id: str
    issue_time: datetime
    target_time: datetime
    horizon: timedelta
    model_id: str
    version: str
    fit_cutoff: datetime
    point_prediction: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    samples: tuple[float, ...] = ()
    uncertainty: tuple[tuple[str, object], ...] = ()
    source_manifest_ids: tuple[str, ...] = ()
    feature_manifest_id: str | None = None
    oof_metrics: tuple[tuple[str, float], ...] = ()
    calibration_claimed: bool = False
    calibration_evidence: CalibrationEvidence | None = None
    warnings: tuple[str, ...] = ()
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        nonempty(self.specialist_id, "specialist_id")
        object.__setattr__(self, "issue_time", utc(self.issue_time, "issue_time"))
        object.__setattr__(self, "target_time", utc(self.target_time, "target_time"))
        if self.target_time <= self.issue_time:
            raise ValueError("target_time must follow issue_time")
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        if self.target_time - self.issue_time != self.horizon:
            raise ValueError("horizon does not match issue/target times")
        nonempty(self.model_id, "model_id")
        nonempty(self.version, "version")
        object.__setattr__(self, "fit_cutoff", utc(self.fit_cutoff, "fit_cutoff"))
        if self.fit_cutoff > self.issue_time:
            raise ValueError("future fit cutoff")

        predictions = (self.point_prediction, self.p10, self.p25, self.p50, self.p75, self.p90)
        for name, value in zip(("point_prediction", "p10", "p25", "p50", "p75", "p90"), predictions):
            if value is not None:
                _finite_number(value, name)
        present_quantiles = [(level, value) for level, value in zip((.10, .25, .50, .75, .90), predictions[1:])
                             if value is not None]
        if any(left[1] > right[1] for left, right in zip(present_quantiles, present_quantiles[1:])):
            raise ValueError("crossed forecast quantiles")
        try:
            samples = tuple(self.samples)
        except TypeError:
            raise ValueError("samples must be a sequence of finite numbers") from None
        object.__setattr__(self, "samples", samples)
        if self.samples and any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
                                for value in self.samples):
            raise ValueError("samples must be finite numbers")
        if not any(value is not None for value in predictions) and not self.samples:
            raise ValueError("forecast brief requires a point, quantile, or samples")
        uncertainty = _uncertainty_pairs(self.uncertainty)
        metadata_calibration_claim = _validate_uncertainty_metadata(
            uncertainty, calibration_allowed=self.calibration_evidence is not None
        )
        if metadata_calibration_claim and not self.calibration_claimed:
            raise ValueError("use calibration_claimed for an explicit calibration claim")
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "source_manifest_ids", _strings(self.source_manifest_ids, "source_manifest_ids"))
        if self.feature_manifest_id is not None:
            nonempty(self.feature_manifest_id, "feature_manifest_id")
        if not self.source_manifest_ids and self.feature_manifest_id is None:
            raise ValueError("forecast provenance requires source or feature manifest IDs")
        object.__setattr__(self, "oof_metrics", _metric_pairs(self.oof_metrics, "oof_metrics"))
        if type(self.calibration_claimed) is not bool:
            raise ValueError("calibration_claimed must be bool")
        if self.calibration_claimed and self.calibration_evidence is None:
            raise ValueError("calibration claim requires attached OOF calibration evidence")
        if self.calibration_evidence is not None:
            if not isinstance(self.calibration_evidence, CalibrationEvidence):
                raise ValueError("invalid calibration evidence")
            if self.calibration_evidence.known_at > self.issue_time:
                raise ValueError("future-informed calibration evidence")
        object.__setattr__(self, "warnings", _strings(self.warnings, "warnings"))
        object.__setattr__(self, "provenance", _pairs(self.provenance, "provenance", string_values=True))

    @property
    def model_version(self) -> str:
        """Alias useful to consumers that name the version pair explicitly."""
        return self.version

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self.source_manifest_ids

    @property
    def brief_type(self) -> str:
        return TARGET_FORECAST_BRIEF
