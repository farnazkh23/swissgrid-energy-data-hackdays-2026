"""Generic real-source adapter into existing Observation and source contracts."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections.abc import Mapping, Iterable
import json

from .availability import earliest_known_at, nonempty, utc
from .column_mapping import ColumnMapping
from .observation_schema import Observation
from .source_contracts import Domain, HorizonAvailability, SourceContract


class PITMetadataError(ValueError):
    """The dataset cannot support point-in-time forecasting under its policy."""


KNOWN_AT_POLICIES = frozenset({"require", "publication_plus_lag"})


def _parse_datetime(value, field: str) -> datetime:
    if isinstance(value, datetime):
        return utc(value, field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an aware ISO-8601 timestamp")
    try:
        return utc(datetime.fromisoformat(value.replace("Z", "+00:00")), field)
    except ValueError as exc:
        raise ValueError(f"{field} must be an aware ISO-8601 timestamp") from exc


def _parse_duration(value, field: str) -> timedelta:
    if isinstance(value, timedelta):
        result = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = timedelta(seconds=value)
    elif isinstance(value, str):
        try:
            result = timedelta(seconds=float(value))
        except ValueError as exc:
            raise ValueError(f"{field} must be seconds or timedelta") from exc
    else:
        raise ValueError(f"{field} must be seconds or timedelta")
    if result <= timedelta(0):
        raise ValueError(f"{field} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class RealSourceConfig:
    dataset_id: str
    mapping: ColumnMapping
    timezone: str
    update_cadence: timedelta
    publication_lag: timedelta
    horizon_minimum: timedelta
    horizon_maximum: timedelta
    known_at_policy: str = "require"
    source_id: str | None = None
    country: str | None = None
    domain: Domain | str | None = None
    unit: str | None = None
    revision_id_default: str | None = None
    revision_sequence_default: int | None = None

    def __post_init__(self):
        nonempty(self.dataset_id, "dataset_id")
        if not isinstance(self.mapping, ColumnMapping):
            raise ValueError("ColumnMapping is required")
        nonempty(self.timezone, "timezone")
        for name in ("update_cadence", "publication_lag"):
            if not isinstance(getattr(self, name), timedelta) or getattr(self, name) < timedelta(0):
                raise ValueError(f"invalid {name}")
        if self.update_cadence == timedelta(0):
            raise ValueError("update_cadence must be positive")
        if not isinstance(self.horizon_minimum, timedelta) or not isinstance(self.horizon_maximum, timedelta) or self.horizon_minimum > self.horizon_maximum:
            raise ValueError("invalid horizon range")
        if self.known_at_policy not in KNOWN_AT_POLICIES:
            raise ValueError("unknown known_at policy")
        for name in ("source_id", "country", "unit"):
            value = getattr(self, name)
            if value is not None:
                nonempty(value, name)
        if self.domain is not None:
            object.__setattr__(self, "domain", Domain(self.domain))
        if self.revision_sequence_default is not None and (type(self.revision_sequence_default) is not int or self.revision_sequence_default < 0):
            raise ValueError("revision_sequence_default must be nonnegative")

    @classmethod
    def from_dict(cls, payload: Mapping) -> "RealSourceConfig":
        if not isinstance(payload, Mapping):
            raise ValueError("source configuration must be an object")
        def duration(name, default=timedelta(0)):
            raw = payload.get(name, default)
            if raw in (0, 0.0, "0", "0.0", timedelta(0)):
                return timedelta(0)
            return _parse_duration(raw, name)
        horizons = payload.get("horizon", {})
        if not isinstance(horizons, Mapping):
            raise ValueError("horizon must be an object")
        return cls(payload["dataset_id"], ColumnMapping.from_dict(payload["mapping"]), payload["timezone"],
                   duration("update_cadence"), duration("publication_lag"),
                   _parse_duration(horizons.get("minimum", 0), "horizon.minimum") if horizons.get("minimum", 0) else timedelta(0),
                   _parse_duration(horizons.get("maximum", 1), "horizon.maximum"),
                   payload.get("known_at_policy", "require"), payload.get("source_id"),
                   payload.get("country"), payload.get("domain"), payload.get("unit"),
                   payload.get("revision_id_default"), payload.get("revision_sequence_default"))

    def contract(self, source_id: str, country: str, domain: Domain, units: Iterable[str]) -> SourceContract:
        return SourceContract(source_id, country, domain, "configured-real-dataset", self.timezone,
                              tuple(sorted(set(units))), self.update_cadence, self.publication_lag,
                              "mapping-provided revision sequence", HorizonAvailability(self.horizon_minimum, self.horizon_maximum),
                              "Caller-provided real dataset; license and publication rules remain under review")


@dataclass(frozen=True, slots=True)
class RealRecordMetadata:
    source_id: str
    country: str
    domain: Domain
    unit: str
    issue_time: datetime | None = None
    horizon: timedelta | None = None
    target_name: str | None = None
    target_entity: str | None = None
    sign_convention: str | None = None

    def to_dict(self):
        return {"source_id": self.source_id, "country": self.country, "domain": self.domain.value,
                "unit": self.unit, "issue_time": self.issue_time.isoformat() if self.issue_time else None,
                "horizon": str(self.horizon) if self.horizon else None, "target_name": self.target_name,
                "target_entity": self.target_entity, "sign_convention": self.sign_convention}


class RealSourceAdapter:
    """Decode mapped rows; target meaning is intentionally not inferred."""
    def __init__(self, config: RealSourceConfig):
        self.config = config

    def _value(self, row, field, *, default=None):
        return self.config.mapping.value(row, field, default=default)

    def adapt_row(self, row: Mapping, *, normalized_at: datetime | None = None) -> tuple[Observation, RealRecordMetadata]:
        if not isinstance(row, Mapping):
            raise ValueError("dataset rows must be mappings")
        mapping = self.config.mapping
        source_id = self._value(row, "source_id", default=self.config.source_id)
        country = self._value(row, "country", default=self.config.country)
        domain = self._value(row, "domain", default=self.config.domain)
        unit = self._value(row, "unit", default=self.config.unit)
        for name, value in (("source_id", source_id), ("country", country), ("domain", domain), ("unit", unit)):
            if value is None or value == "":
                raise ValueError(f"real source metadata is unresolved: {name}")
        source_id, country, unit = str(source_id), str(country), str(unit)
        domain = Domain(domain)
        event_time = _parse_datetime(mapping.value(row, "event_time"), "event_time")
        valid_time = _parse_datetime(mapping.value(row, "valid_time"), "valid_time")
        publication = mapping.value(row, "publication_time", default=None)
        publication_time = None if publication in (None, "") else _parse_datetime(publication, "publication_time")
        received_value = mapping.value(row, "first_received_at", default=None)
        first_received_at = None if received_value in (None, "") else _parse_datetime(received_value, "first_received_at")
        normalized_value = mapping.value(row, "normalized_at", default=normalized_at)
        normalized_clock = None if normalized_value in (None, "") else _parse_datetime(normalized_value, "normalized_at")
        known_value = mapping.value(row, "known_at", default=None)
        known_at = None if known_value in (None, "") else _parse_datetime(known_value, "known_at")
        if self.config.known_at_policy == "require":
            if known_at is None or first_received_at is None or normalized_clock is None:
                raise PITMetadataError("known_at, first_received_at, and normalized_at are required for forecasting")
        else:
            if publication_time is None:
                raise PITMetadataError("publication_time is required for publication_plus_lag reconstruction")
            if first_received_at is None:
                first_received_at = publication_time
            if normalized_clock is None:
                normalized_clock = first_received_at
            known_at = max(publication_time + self.config.publication_lag, first_received_at, normalized_clock)
        revision_id = mapping.value(row, "revision_id", default=self.config.revision_id_default)
        revision_sequence = mapping.value(row, "revision_sequence", default=self.config.revision_sequence_default)
        if revision_id is None or revision_sequence is None:
            raise PITMetadataError("revision identity and sequence are required or must be explicitly defaulted")
        source_record_id = mapping.value(row, "source_record_id", default=None)
        if source_record_id is None:
            raise ValueError("source_record_id is required")
        value = mapping.value(row, "value", default=None)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("real value must be numeric or null") from exc
        normalized_clock = normalized_clock or first_received_at
        supersedes = mapping.value(row, "supersedes_revision_id", default=None)
        if supersedes in (None, ""):
            supersedes = None
        observation = Observation(source_id, str(source_record_id), str(revision_id), int(revision_sequence),
                                  event_time, valid_time, publication_time, first_received_at,
                                  normalized_clock, known_at, value, unit,
                                  str(supersedes) if supersedes is not None else None)
        issue_value = mapping.value(row, "issue_time", default=None)
        issue_time = None if issue_value in (None, "") else _parse_datetime(issue_value, "issue_time")
        horizon_value = mapping.value(row, "horizon", default=None)
        horizon = None if horizon_value in (None, "") else _parse_duration(horizon_value, "horizon")
        metadata = RealRecordMetadata(source_id, country, domain, unit, issue_time, horizon,
                                      mapping.value(row, "target_name", default=None),
                                      mapping.value(row, "target_entity", default=None),
                                      mapping.value(row, "sign_convention", default=None))
        return observation, metadata

    def adapt_rows(self, rows: Iterable[Mapping], *, normalized_at: datetime | None = None):
        result = []
        for row in rows:
            result.append(self.adapt_row(row, normalized_at=normalized_at))
        return tuple(result)
