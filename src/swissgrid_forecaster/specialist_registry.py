"""Metadata registry for future specialist implementations."""

from dataclasses import dataclass
from datetime import timedelta

from .availability import nonempty
from .specialist_contracts import EVIDENCE_BRIEF, TARGET_FORECAST_BRIEF


SPECIALIST_FAMILIES = (
    "History/Analogue",
    "Demand",
    "Weather/Renewables",
    "Generation/Outage",
    "Hydro/Storage",
    "Neighbour/Flow",
    "Market",
    "News/Event",
    "Tail/Extreme",
)
SPECIALIST_OUTPUTS = (EVIDENCE_BRIEF, TARGET_FORECAST_BRIEF)


@dataclass(frozen=True, slots=True)
class SpecialistDefinition:
    specialist_id: str
    family: str
    version: str
    output_types: tuple[str, ...]
    horizon_applicability: tuple[timedelta, ...]
    enabled: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        nonempty(self.specialist_id, "specialist_id")
        if self.family not in SPECIALIST_FAMILIES:
            raise ValueError("unknown specialist family")
        nonempty(self.version, "version")
        outputs = tuple(self.output_types)
        if not outputs or any(output not in SPECIALIST_OUTPUTS for output in outputs):
            raise ValueError("output_types must contain known brief types")
        if len(set(outputs)) != len(outputs):
            raise ValueError("duplicate output type")
        object.__setattr__(self, "output_types", outputs)
        horizons = tuple(self.horizon_applicability)
        if not horizons or any(not isinstance(horizon, timedelta) or horizon <= timedelta(0) for horizon in horizons):
            raise ValueError("positive horizon applicability is required")
        if len(set(horizons)) != len(horizons):
            raise ValueError("duplicate horizon applicability")
        object.__setattr__(self, "horizon_applicability", tuple(sorted(horizons)))
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be bool")
        if not isinstance(self.description, str):
            raise ValueError("description must be text")


class SpecialistRegistry:
    """Explicit specialist metadata; no weights or promotion operations exist."""

    def __init__(self, definitions=()):
        self._entries: dict[str, SpecialistDefinition] = {}
        for definition in definitions:
            self.register(definition)

    @property
    def entries(self) -> tuple[SpecialistDefinition, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def register(self, definition: SpecialistDefinition) -> None:
        if not isinstance(definition, SpecialistDefinition):
            raise ValueError("typed SpecialistDefinition required")
        if definition.specialist_id in self._entries:
            raise ValueError("duplicate specialist ID, including alternate versions")
        self._entries[definition.specialist_id] = definition

    def get(self, specialist_id: str) -> SpecialistDefinition:
        nonempty(specialist_id, "specialist_id")
        try:
            return self._entries[specialist_id]
        except KeyError:
            raise ValueError("unknown specialist ID") from None

    def by_family(self, family: str) -> tuple[SpecialistDefinition, ...]:
        if family not in SPECIALIST_FAMILIES:
            raise ValueError("unknown specialist family")
        return tuple(entry for entry in self.entries if entry.family == family)
