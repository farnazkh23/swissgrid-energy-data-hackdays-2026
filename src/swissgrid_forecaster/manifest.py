"""Immutable per-receipt manifest schema."""

from dataclasses import dataclass
from datetime import datetime
import json
import re

from .availability import nonempty, utc


def validate_digest(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    receipt_id: str
    sha256: str
    byte_count: int
    source_id: str
    source_record_id: str
    revision_id: str
    first_received_at: datetime
    source_metadata: tuple[tuple[str, str], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_id, str) or re.fullmatch(r"[0-9a-f]{32}", self.receipt_id) is None:
            raise ValueError("invalid receipt_id")
        validate_digest(self.sha256)
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("invalid byte_count")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported manifest schema")
        for name in ("source_id", "source_record_id", "revision_id"):
            nonempty(getattr(self, name), name)
        object.__setattr__(self, "first_received_at", utc(self.first_received_at, "first_received_at"))
        metadata = tuple(tuple(pair) for pair in self.source_metadata)
        for pair in metadata:
            if len(pair) != 2:
                raise ValueError("metadata must contain key/value pairs")
            nonempty(pair[0], "metadata key")
            if not isinstance(pair[1], str):
                raise ValueError("metadata values must be strings")
        if len({pair[0] for pair in metadata}) != len(metadata):
            raise ValueError("duplicate metadata keys")
        object.__setattr__(self, "source_metadata", tuple(sorted(metadata)))

    def to_bytes(self) -> bytes:
        return json.dumps({
            "schema_version": self.schema_version, "receipt_id": self.receipt_id,
            "sha256": self.sha256, "byte_count": self.byte_count,
            "source_id": self.source_id, "source_record_id": self.source_record_id,
            "revision_id": self.revision_id,
            "first_received_at": self.first_received_at.isoformat(),
            "source_metadata": dict(self.source_metadata),
        }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ManifestEntry":
        fields = json.loads(payload)
        fields["first_received_at"] = datetime.fromisoformat(fields["first_received_at"])
        fields["source_metadata"] = tuple(fields["source_metadata"].items())
        return cls(**fields)
