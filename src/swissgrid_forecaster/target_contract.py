"""Configurable target definition without assumed grid sign semantics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from .availability import nonempty, utc


# Canonical challenge forecast-output target contract.
#
# The challenge provides five country net positions (AT, CH, DE, FR, IT) as
# inputs and asks for exactly four forecast arrays: the net positions of
# Switzerland's four neighbouring countries. Switzerland (CH) is NOT a scored
# forecast output; it remains valid as an input feature / system-state
# context country.
#
# The order below is canonical and stable repo-wide because the official
# evaluator maps realizations POSITIONALLY:
#
#     submission target_0 -> AT
#     submission target_1 -> DE
#     submission target_2 -> FR
#     submission target_3 -> IT
TARGETS: tuple[str, ...] = ("AT", "DE", "FR", "IT")
OUTPUT_TARGETS: tuple[str, ...] = TARGETS
TARGET_POSITIONS: dict[str, int] = {target: index for index, target in enumerate(TARGETS)}
POSITION_TARGETS: tuple[str, ...] = TARGETS
# Countries available as model inputs / context. CH is context-only and must
# never be selected as a forecast-output target.
INPUT_COUNTRIES: tuple[str, ...] = ("AT", "CH", "DE", "FR", "IT")
CONTEXT_ONLY_COUNTRIES: tuple[str, ...] = ("CH",)


class InvalidOutputTargets(ValueError):
    """A target set violates the canonical AT/DE/FR/IT output contract."""


def validate_output_targets(targets) -> tuple[str, ...]:
    """Validate a forecast-output target sequence against the canonical contract.

    Rejects any sequence that omits AT, includes CH, adds or drops a target,
    duplicates a target, or reorders the canonical positional order.
    """
    if isinstance(targets, str) or not hasattr(targets, "__iter__"):
        raise InvalidOutputTargets("output targets must be an ordered sequence")
    sequence = tuple(targets)
    if "CH" in sequence:
        raise InvalidOutputTargets("CH is an input/context country, not a forecast output target")
    if "AT" not in sequence:
        raise InvalidOutputTargets("AT is a required forecast output target")
    if sequence != TARGETS:
        detail = "/".join(sequence) if sequence else "empty sequence"
        raise InvalidOutputTargets(
            "output targets must be exactly " + "/".join(TARGETS) +
            " in that positional order; got " + detail)
    return sequence


def require_output_target(entity: str) -> str:
    """Assert a single entity is one of the four scored forecast-output targets."""
    nonempty(entity, "target_entity")
    if entity not in TARGETS:
        raise InvalidOutputTargets(
            f"target entity {entity!r} is not a scored forecast output target "
            f"(expected one of {', '.join(TARGETS)}); CH is input/context only")
    return entity


def submission_position(entity: str) -> int:
    """Positional index of a scored target in the submission table (0-based)."""
    require_output_target(entity)
    return TARGET_POSITIONS[entity]


class ForecastType(str, Enum):
    POINT = "point"
    QUANTILE = "quantile"
    DISTRIBUTION = "distribution"


@dataclass(frozen=True, slots=True)
class TargetContract:
    target_name: str
    target_entity: str
    unit: str
    sign_convention: str
    resolution: timedelta
    forecast_issue_time: datetime
    target_valid_time: datetime
    horizon: timedelta
    aggregation_rule: str
    label_availability_rule: str
    forecast_type: ForecastType
    evaluation_metrics: tuple[str, ...]
    country: str | None = None
    border: tuple[str, str] | None = None
    quantile_levels: tuple[float, ...] = ()
    scenario_support: bool = False

    def __post_init__(self) -> None:
        for name in ("target_name", "target_entity", "unit", "sign_convention",
                     "aggregation_rule", "label_availability_rule"):
            nonempty(getattr(self, name), name)
        for name in ("forecast_issue_time", "target_valid_time"):
            object.__setattr__(self, name, utc(getattr(self, name), name))
        if not isinstance(self.resolution, timedelta) or self.resolution <= timedelta(0):
            raise ValueError("resolution must be positive")
        delta = self.target_valid_time - self.forecast_issue_time
        if delta <= timedelta(0):
            raise ValueError("target_valid_time must follow forecast_issue_time")
        if not isinstance(self.horizon, timedelta) or self.horizon != delta:
            raise ValueError("horizon must match valid time minus issue time")
        object.__setattr__(self, "forecast_type", ForecastType(self.forecast_type))
        levels = tuple(self.quantile_levels)
        if any(isinstance(q, bool) or not isinstance(q, (float, int)) or
               not isfinite(q) or not 0 < q < 1 for q in levels):
            raise ValueError("quantiles must be finite and inside (0, 1)")
        if any(a >= b for a, b in zip(levels, levels[1:])):
            raise ValueError("quantiles must be strictly increasing")
        if self.forecast_type == ForecastType.QUANTILE and not levels:
            raise ValueError("quantile forecasts require levels")
        object.__setattr__(self, "quantile_levels", levels)
        metrics = tuple(self.evaluation_metrics)
        if not metrics:
            raise ValueError("evaluation_metrics must not be empty")
        for metric in metrics:
            nonempty(metric, "evaluation metric")
        if len(set(metrics)) != len(metrics):
            raise ValueError("duplicate evaluation metrics")
        object.__setattr__(self, "evaluation_metrics", metrics)
        if self.country is not None:
            nonempty(self.country, "country")
        if self.border is not None:
            border = tuple(self.border)
            if len(border) != 2 or border[0] == border[1]:
                raise ValueError("border requires two distinct endpoints")
            for endpoint in border:
                nonempty(endpoint, "border endpoint")
            object.__setattr__(self, "border", border)
        if not isinstance(self.scenario_support, bool):
            raise ValueError("scenario_support must be boolean")
