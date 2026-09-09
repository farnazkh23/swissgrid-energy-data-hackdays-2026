"""Provider-neutral metadata; no production source semantics are preselected."""
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .availability import nonempty


class Domain(str, Enum):
    NET_POSITION = 'net_position'
    CROSS_BORDER_FLOW = 'cross_border_flow'
    LOAD = 'load'
    GENERATION = 'generation'
    WIND = 'wind'
    SOLAR = 'solar'
    HYDRO = 'hydro'
    STORAGE = 'storage'
    NUCLEAR = 'nuclear'
    OUTAGE = 'outage'
    WEATHER = 'weather'
    PRICE = 'price'
    CAPACITY = 'capacity'
    SCHEDULE = 'schedule'


@dataclass(frozen=True, slots=True)
class HorizonAvailability:
    """Declared valid_time - issue_time range, inclusive; not proof of receipt."""
    minimum: timedelta
    maximum: timedelta

    def __post_init__(self):
        if not all(isinstance(x, timedelta) for x in (self.minimum, self.maximum)):
            raise ValueError('horizons must be timedeltas')
        if self.minimum > self.maximum:
            raise ValueError('minimum horizon exceeds maximum')


@dataclass(frozen=True, slots=True)
class SourceContract:
    source_id: str
    country: str
    domain: Domain
    provider: str
    timezone: str
    units: tuple[str, ...]
    update_cadence: timedelta | None
    publication_lag: timedelta | None
    revision_policy: str
    horizon_availability: HorizonAvailability | None
    license_usage_notes: str

    def __post_init__(self):
        for name in ('source_id', 'country', 'provider', 'timezone', 'revision_policy',
                     'license_usage_notes'):
            nonempty(getattr(self, name), name)
        object.__setattr__(self, 'domain', Domain(self.domain))
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError('timezone must name an IANA zone') from exc
        if isinstance(self.units, str):
            raise ValueError('units must be a collection of exact unit symbols')
        units = tuple(self.units)
        if not units or len(set(units)) != len(units):
            raise ValueError('units must be nonempty and unique')
        for unit in units:
            nonempty(unit, 'unit')
        object.__setattr__(self, 'units', units)
        for name in ('update_cadence', 'publication_lag'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, timedelta) or value < timedelta(0)):
                raise ValueError(f'{name} must be a nonnegative timedelta or None')
        if self.update_cadence == timedelta(0):
            raise ValueError('update_cadence must be positive')
        if self.horizon_availability is not None and not isinstance(self.horizon_availability, HorizonAvailability):
            raise ValueError('invalid horizon_availability')

    def validate_unit(self, unit: str) -> None:
        if unit not in self.units:
            raise ValueError(f'unit {unit!r} is not declared for {self.source_id}')

    def require_forecast_metadata(self) -> None:
        if any(x is None for x in (self.update_cadence, self.publication_lag, self.horizon_availability)):
            raise ValueError('missing source availability metadata for forecasting')
