"""Validated dataset membership and conservative point-in-time selection."""
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Iterable

from .availability import nonempty, utc
from .asof_resolver import resolve_as_of
from .observation_schema import Observation
from .source_registry import SourceRegistry


@dataclass(frozen=True, slots=True)
class DatasetContract:
    dataset_id: str
    source_ids: tuple[str, ...]

    def __post_init__(self):
        nonempty(self.dataset_id, 'dataset_id')
        if isinstance(self.source_ids, str):
            raise ValueError('source_ids must be a collection')
        ids = tuple(self.source_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('source_ids must be nonempty and unique')
        for source_id in ids:
            nonempty(source_id, 'source_id')
        object.__setattr__(self, 'source_ids', ids)

    def as_of(self, records: Iterable[Observation], registry: SourceRegistry,
              issue_time: datetime) -> tuple[Observation, ...]:
        issue = utc(issue_time, 'issue_time')
        for source_id in self.source_ids:
            registry.get(source_id).require_forecast_metadata()
        rows = tuple(records)
        for row in rows:
            if not isinstance(row, Observation):
                raise ValueError('missing validated observation/availability metadata')
            if row.source_id not in self.source_ids:
                raise ValueError('source not in dataset')
            registry.get(row.source_id).validate_unit(row.unit)
        selected = resolve_as_of(rows, issue)
        return tuple(row for row in selected
                     if registry.get(row.source_id).horizon_availability.minimum
                     <= row.valid_time - issue
                     <= registry.get(row.source_id).horizon_availability.maximum)
