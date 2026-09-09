"""Generic out-of-fold contribution evidence contracts."""

from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

from .ablation import COMPONENT_KINDS, KEEP_DROP
from .availability import nonempty


@dataclass(frozen=True, slots=True)
class ContributionRecord:
    """One OOF comparison recording a component's contribution to a baseline."""

    baseline_model_id: str
    component_id: str
    component_kind: str
    metric_name: str
    metric_delta: float
    oof_fold_ids: tuple[str, ...]
    horizon: timedelta
    delta_uncertainty: float | None = None
    regime: str | None = None
    dependency_group: str | None = None
    keep_drop: str = "UNDECIDED"
    keep_drop_evidence: tuple[str, ...] = ()
    baseline_model_version: str | None = None
    evaluation_id: str | None = None
    evaluation_scope: str = "OOF"
    delta_definition: str = "comparison-specific"

    def __post_init__(self) -> None:
        nonempty(self.baseline_model_id, "baseline_model_id")
        nonempty(self.component_id, "component_id")
        if self.component_kind not in COMPONENT_KINDS:
            raise ValueError("component_kind must be model or specialist")
        nonempty(self.metric_name, "metric_name")
        if isinstance(self.metric_delta, bool) or not isinstance(self.metric_delta, (int, float)) or not isfinite(self.metric_delta):
            raise ValueError("metric_delta must be finite")
        folds = tuple(self.oof_fold_ids)
        if not folds or any(not isinstance(item, str) or not item.strip() for item in folds):
            raise ValueError("nonempty OOF fold IDs are required")
        if len(set(folds)) != len(folds):
            raise ValueError("duplicate OOF fold IDs")
        object.__setattr__(self, "oof_fold_ids", folds)
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        if self.delta_uncertainty is not None and (isinstance(self.delta_uncertainty, bool)
                                                   or not isinstance(self.delta_uncertainty, (int, float))
                                                   or not isfinite(self.delta_uncertainty)
                                                   or self.delta_uncertainty < 0):
            raise ValueError("delta_uncertainty must be a nonnegative finite number")
        for name in ("regime", "dependency_group", "baseline_model_version", "evaluation_id"):
            value = getattr(self, name)
            if value is not None:
                nonempty(value, name)
        if self.keep_drop not in KEEP_DROP:
            raise ValueError("invalid keep_drop")
        evidence = tuple(self.keep_drop_evidence)
        if any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValueError("keep/drop evidence must be nonblank text")
        if self.keep_drop != "UNDECIDED" and not evidence:
            raise ValueError("KEEP/DROP requires evidence")
        object.__setattr__(self, "keep_drop_evidence", evidence)
        if self.evaluation_scope != "OOF":
            raise ValueError("contribution claims must use OOF evidence; in-sample claims are forbidden")
        nonempty(self.delta_definition, "delta_definition")

    @property
    def specialist_id(self) -> str | None:
        return self.component_id if self.component_kind == "specialist" else None
