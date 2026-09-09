"""Immutable scalar normalized observation and revision lineage."""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from .availability import earliest_known_at, nonempty, utc


@dataclass(frozen=True, slots=True)
class Observation:
    source_id: str
    source_record_id: str
    revision_id: str
    revision_sequence: int
    event_time: datetime
    valid_time: datetime
    publication_time: datetime | None
    first_received_at: datetime
    normalized_at: datetime
    known_at: datetime
    value: float | int | str | bool | None
    unit: str
    supersedes_revision_id: str | None = None
    raw_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "source_record_id", "revision_id", "unit"):
            nonempty(getattr(self, name), name)
        if type(self.revision_sequence) is not int or self.revision_sequence < 0:
            raise ValueError("revision_sequence must be a nonnegative integer")
        for name in ("event_time", "valid_time", "first_received_at", "normalized_at", "known_at"):
            object.__setattr__(self, name, utc(getattr(self, name), name))
        if self.publication_time is not None:
            object.__setattr__(self, "publication_time", utc(self.publication_time, "publication_time"))
        lower_bound = earliest_known_at(
            first_received_at=self.first_received_at, normalized_at=self.normalized_at,
            publication_time=self.publication_time,
        )
        if self.known_at < lower_bound:
            raise ValueError("known_at precedes receipt, normalization or publication")
        if self.supersedes_revision_id is not None:
            nonempty(self.supersedes_revision_id, "supersedes_revision_id")
            if self.supersedes_revision_id == self.revision_id:
                raise ValueError("revision cannot supersede itself")
        if self.raw_sha256 is not None:
            from .manifest import validate_digest
            validate_digest(self.raw_sha256)
        if type(self.value) not in (float, int, str, bool, type(None)):
            raise ValueError("value must be an immutable scalar")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("value must be finite")

    @property
    def logical_key(self) -> tuple[str, str]:
        return self.source_id, self.source_record_id
