"""Ingest already-received bytes through pluggable, offline normalization adapters."""
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from .availability import utc
from .observation_schema import Observation
from .raw_store import RawStore, StoreResult
from .source_registry import SourceRegistry


@dataclass(frozen=True, slots=True)
class ReceivedRecord:
    """One independently revisable scalar payload; caller attests receipt time."""
    source_id: str
    source_record_id: str
    revision_id: str
    payload: bytes
    first_received_at: datetime

    def __post_init__(self):
        from .availability import nonempty
        for name in ('source_id', 'source_record_id', 'revision_id'):
            nonempty(getattr(self, name), name)
        if type(self.payload) is not bytes:
            raise ValueError('payload must be immutable bytes')
        object.__setattr__(self, 'first_received_at', utc(self.first_received_at))


class SourceAdapter(Protocol):
    def normalize(self, received: ReceivedRecord, *, normalized_at: datetime) -> Observation:
        """Preserve identity/clocks; supply explicit provider revision and availability."""
        ...


@dataclass(frozen=True, slots=True)
class IngestionResult:
    observation: Observation
    raw: StoreResult


def ingest(received: ReceivedRecord, adapter: SourceAdapter, registry: SourceRegistry,
           store: RawStore, *, normalized_at: datetime,
           for_forecasting: bool = False) -> IngestionResult:
    source = registry.get(received.source_id)
    if for_forecasting:
        source.require_forecast_metadata()
    normalized = utc(normalized_at, 'normalized_at')
    if normalized < received.first_received_at:
        raise ValueError('normalization precedes receipt')
    # Retain exact evidence even if parsing or unit validation subsequently fails.
    raw = store.put(received.payload, source_id=received.source_id,
                    source_record_id=received.source_record_id,
                    revision_id=received.revision_id,
                    first_received_at=received.first_received_at,
                    source_metadata={'provider': source.provider, 'country': source.country,
                                     'domain': source.domain.value, 'timezone': source.timezone,
                                     'license_usage_notes': source.license_usage_notes})
    record = adapter.normalize(received, normalized_at=normalized)
    if not isinstance(record, Observation):
        raise ValueError('adapter must return a validated Observation')
    for name in ('source_id', 'source_record_id', 'revision_id', 'first_received_at'):
        if getattr(record, name) != getattr(received, name):
            raise ValueError(f'adapter changed {name}')
    if record.normalized_at != normalized:
        raise ValueError('adapter changed normalized_at')
    source.validate_unit(record.unit)
    if record.raw_sha256 not in (None, raw.manifest.sha256):
        raise ValueError('adapter raw digest mismatch')
    return IngestionResult(replace(record, raw_sha256=raw.manifest.sha256), raw)
