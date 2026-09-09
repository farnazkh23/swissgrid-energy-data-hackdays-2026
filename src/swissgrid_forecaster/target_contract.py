"""Configurable target definition without assumed grid sign semantics."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from .availability import nonempty, utc


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
