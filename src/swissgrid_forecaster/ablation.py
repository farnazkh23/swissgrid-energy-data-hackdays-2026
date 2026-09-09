"""Generic out-of-fold ablation evidence contracts."""

from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

from .availability import nonempty


KEEP_DROP = ("KEEP", "DROP", "UNDECIDED")
COMPONENT_KINDS = ("model", "specialist")


def _folds(value) -> tuple[str, ...]:
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError("nonempty OOF fold IDs are required")
    if len(set(result)) != len(result):
        raise ValueError("duplicate OOF fold IDs")
    return result


def _optional_text(value, name: str) -> None:
    if value is not None:
        nonempty(value, name)


@dataclass(frozen=True, slots=True)
class AblationRecord:
    """One baseline-versus-removed-component comparison from rolling OOF data."""

    baseline_model_id: str
    removed_component_id: str
    removed_component_kind: str
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
        nonempty(self.removed_component_id, "removed_component_id")
        if self.removed_component_kind not in COMPONENT_KINDS:
            raise ValueError("removed_component_kind must be model or specialist")
        nonempty(self.metric_name, "metric_name")
        if isinstance(self.metric_delta, bool) or not isinstance(self.metric_delta, (int, float)) or not isfinite(self.metric_delta):
            raise ValueError("metric_delta must be finite")
        object.__setattr__(self, "oof_fold_ids", _folds(self.oof_fold_ids))
        if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
            raise ValueError("horizon must be positive")
        if self.delta_uncertainty is not None and (isinstance(self.delta_uncertainty, bool)
                                                   or not isinstance(self.delta_uncertainty, (int, float))
                                                   or not isfinite(self.delta_uncertainty)
                                                   or self.delta_uncertainty < 0):
            raise ValueError("delta_uncertainty must be a nonnegative finite number")
        _optional_text(self.regime, "regime")
        _optional_text(self.dependency_group, "dependency_group")
        if self.keep_drop not in KEEP_DROP:
            raise ValueError("invalid keep_drop")
        evidence = tuple(self.keep_drop_evidence)
        if any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValueError("keep/drop evidence must be nonblank text")
        if self.keep_drop != "UNDECIDED" and not evidence:
            raise ValueError("KEEP/DROP requires evidence")
        object.__setattr__(self, "keep_drop_evidence", evidence)
        _optional_text(self.baseline_model_version, "baseline_model_version")
        _optional_text(self.evaluation_id, "evaluation_id")
        if self.evaluation_scope != "OOF":
            raise ValueError("ablation claims must use OOF evidence; in-sample claims are forbidden")
        nonempty(self.delta_definition, "delta_definition")

    @property
    def removed_model_or_specialist(self) -> str:
        return self.removed_component_id
