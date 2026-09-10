"""Exact-byte content-addressed storage with atomic, no-clobber publication."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import tempfile
from uuid import uuid4

from .manifest import ManifestEntry, validate_digest


class IntegrityError(ValueError):
    """Persisted evidence differs from its claimed identity."""


def _fsync_directory(directory: Path) -> None:
    """Best-effort; Windows has no directory file descriptor to fsync."""
    if not hasattr(os, "O_DIRECTORY"):
        return
    handle = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _publish(path: Path, payload: bytes) -> bool:
    """fsync temp bytes then atomically link without replacing existing files."""
    fd, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(temporary_name)
    created = False
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
            created = True
        except FileExistsError:
            created = False
        return created
    finally:
        # Unlink the still-writable temp name first: hard links share one set of
        # attributes, so marking `path` read-only before this would block the
        # unlink on Windows.
        os.unlink(temporary)
        if created:
            os.chmod(path, 0o444)
        _fsync_directory(path.parent)


@dataclass(frozen=True, slots=True)
class StoreResult:
    manifest: ManifestEntry
    duplicate_payload: bool


class RawStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.manifest_dir = self.root / "manifests"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    def put(
        self, payload: bytes, *, source_id: str, source_record_id: str,
        revision_id: str, first_received_at: datetime,
        source_metadata: Mapping[str, str] | None = None,
    ) -> StoreResult:
        """Persist already-retrieved bytes; caller attests completed receipt time.

        Canonical representation is the original bytes, with no JSON rewriting.
        Hash once before persistence; verify existing bytes by exact comparison.
        """
        if type(payload) is not bytes:
            raise ValueError("payload must be immutable bytes from completed retrieval")
        digest = sha256(payload).hexdigest()
        entry = ManifestEntry(
            receipt_id=uuid4().hex, sha256=digest, byte_count=len(payload),
            source_id=source_id, source_record_id=source_record_id,
            revision_id=revision_id, first_received_at=first_received_at,
            source_metadata=tuple((source_metadata or {}).items()),
        )
        encoded_entry = entry.to_bytes()
        path = self.raw_dir / digest
        created = _publish(path, payload)
        if not created and path.read_bytes() != payload:
            raise IntegrityError("existing raw bytes differ from content address")
        if not _publish(self.manifest_dir / f"{entry.receipt_id}.json", encoded_entry):
            raise IntegrityError("receipt identity collision")
        return StoreResult(entry, duplicate_payload=not created)

    def retrieve(
        self, fetch: Callable[[], bytes], *, source_id: str,
        source_record_id: str, revision_id: str,
        source_metadata: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> StoreResult:
        """Fetch must fully read/close the external response before returning."""
        metadata = dict(source_metadata or {})
        payload = fetch()
        received = clock()  # Deliberately after retrieval completes successfully.
        return self.put(payload, source_id=source_id, source_record_id=source_record_id,
                        revision_id=revision_id, first_received_at=received,
                        source_metadata=metadata)

    def read(self, digest: str) -> bytes:
        validate_digest(digest)
        payload = (self.raw_dir / digest).read_bytes()
        if sha256(payload).hexdigest() != digest:
            raise IntegrityError("raw sha256 mismatch")
        return payload

    def read_receipt(self, entry: ManifestEntry) -> bytes:
        payload = self.read(entry.sha256)
        if len(payload) != entry.byte_count:
            raise IntegrityError("manifest byte count mismatch")
        return payload

    def manifests(self) -> tuple[ManifestEntry, ...]:
        entries = []
        for path in sorted(self.manifest_dir.glob("*.json")):
            entry = ManifestEntry.from_bytes(path.read_bytes())
            if path.stem != entry.receipt_id:
                raise IntegrityError("manifest receipt identity mismatch")
            entries.append(entry)
        return tuple(sorted(entries, key=lambda e: (e.first_received_at, e.receipt_id)))
