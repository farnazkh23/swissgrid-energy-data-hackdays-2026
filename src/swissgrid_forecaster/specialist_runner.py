"""Fail-closed execution boundary for typed specialist briefs."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from .availability import utc
from .specialist_contracts import EvidenceBrief, TargetForecastBrief
from .specialist_registry import SpecialistDefinition, SpecialistRegistry


@dataclass(frozen=True, slots=True)
class SpecialistRunContext:
    issue_time: datetime
    target_time: datetime | None = None
    horizon: timedelta | None = None
    known_at_cutoff: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_time", utc(self.issue_time, "issue_time"))
        if self.target_time is not None:
            object.__setattr__(self, "target_time", utc(self.target_time, "target_time"))
            if self.target_time <= self.issue_time:
                raise ValueError("target_time must follow issue_time")
            if self.horizon is None:
                object.__setattr__(self, "horizon", self.target_time - self.issue_time)
        if self.horizon is not None:
            if not isinstance(self.horizon, timedelta) or self.horizon <= timedelta(0):
                raise ValueError("horizon must be positive")
            if self.target_time is not None and self.target_time - self.issue_time != self.horizon:
                raise ValueError("context horizon does not match target_time")
        if self.known_at_cutoff is not None:
            object.__setattr__(self, "known_at_cutoff", utc(self.known_at_cutoff, "known_at_cutoff"))
            if self.known_at_cutoff > self.issue_time:
                raise ValueError("future known_at cutoff")


class SpecialistCallable(Protocol):
    def run(self, context: SpecialistRunContext): ...


class SpecialistRunner:
    """Invoke specialists and accept only validated compressed briefs."""

    def __init__(self, registry: SpecialistRegistry):
        if not isinstance(registry, SpecialistRegistry):
            raise ValueError("SpecialistRegistry required")
        self.registry = registry

    def run(self, specialist_id: str, specialist: SpecialistCallable, context: SpecialistRunContext):
        definition = self.registry.get(specialist_id)
        if not isinstance(context, SpecialistRunContext):
            raise ValueError("SpecialistRunContext required")
        if not definition.enabled:
            raise ValueError("disabled specialist cannot run")
        if getattr(specialist, "specialist_id", specialist_id) != specialist_id:
            raise ValueError("specialist identity mismatch")
        try:
            if hasattr(specialist, "run"):
                brief = specialist.run(context)
            elif hasattr(specialist, "produce"):
                brief = specialist.produce(context)
            elif callable(specialist):
                brief = specialist(context)
            else:
                raise ValueError("specialist must expose run/produce or be callable")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("specialist execution failed closed") from exc
        self._validate_brief(brief, definition, context)
        return brief

    def run_many(self, specialists: dict[str, SpecialistCallable], context: SpecialistRunContext):
        if not isinstance(specialists, dict) or not specialists:
            raise ValueError("nonempty specialist mapping required")
        if len(set(specialists)) != len(specialists):
            raise ValueError("duplicate specialist IDs")
        return tuple(self.run(specialist_id, specialists[specialist_id], context)
                     for specialist_id in sorted(specialists))

    @staticmethod
    def _validate_brief(brief, definition: SpecialistDefinition, context: SpecialistRunContext) -> None:
        if not isinstance(brief, (EvidenceBrief, TargetForecastBrief)):
            raise ValueError("specialists may return only compressed typed briefs")
        if brief.specialist_id != definition.specialist_id:
            raise ValueError("brief specialist ID mismatch")
        if brief.brief_type not in definition.output_types:
            raise ValueError("brief type is not registered for specialist")
        if brief.issue_time != context.issue_time:
            raise ValueError("brief issue_time mismatch")
        if isinstance(brief, EvidenceBrief):
            if brief.evidence_family != definition.family:
                raise ValueError("evidence family mismatch")
            if context.horizon is not None and context.horizon not in brief.horizon_applicability:
                raise ValueError("brief does not apply to requested horizon")
            if context.known_at_cutoff is not None and brief.known_at > context.known_at_cutoff:
                raise ValueError("brief exceeds known_at cutoff")
        else:
            if context.target_time is not None and brief.target_time != context.target_time:
                raise ValueError("brief target_time mismatch")
            if context.horizon is not None and brief.horizon != context.horizon:
                raise ValueError("brief horizon mismatch")
            if context.known_at_cutoff is not None and brief.fit_cutoff > context.known_at_cutoff:
                raise ValueError("brief exceeds fit cutoff")
