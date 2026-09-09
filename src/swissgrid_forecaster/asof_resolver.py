"""Pure revision resolution over explicit evidence, including multiple sources."""

from collections.abc import Iterable, Mapping
from datetime import datetime

from .availability import is_available, utc
from .observation_schema import Observation


def resolve_as_of(records: Iterable[Observation], issue_time: datetime) -> tuple[Observation, ...]:
    """Choose greatest eligible provider revision sequence per logical key.

    Opaque revision IDs are not lexically ordered. Conflicting eligible IDs or
    sequences fail closed; exact duplicate observations are idempotent.
    """
    issue = utc(issue_time, "issue_time")
    selected: dict[tuple[str, str], Observation] = {}
    identities: dict[tuple[str, str, str], Observation] = {}
    sequences: dict[tuple[str, str, int], Observation] = {}
    for record in records:
        if not isinstance(record, Observation):
            raise ValueError("records must be validated Observation instances")
        if not is_available(record.known_at, issue):
            continue
        identity = (*record.logical_key, record.revision_id)
        sequence = (*record.logical_key, record.revision_sequence)
        for table, key in ((identities, identity), (sequences, sequence)):
            if key in table and table[key] != record:
                raise ValueError("conflicting eligible revision identity or sequence")
            table[key] = record
        previous = selected.get(record.logical_key)
        if previous is None or record.revision_sequence > previous.revision_sequence:
            selected[record.logical_key] = record
    return tuple(selected[key] for key in sorted(selected))


def resolve_sources_as_of(
    sources: Mapping[str, Iterable[Observation]], issue_time: datetime,
) -> tuple[Observation, ...]:
    def observations() -> Iterable[Observation]:
        for source_id, records in sources.items():
            for record in records:
                if not isinstance(record, Observation) or record.source_id != source_id:
                    raise ValueError("source collection key does not match observation")
                yield record
    return resolve_as_of(observations(), issue_time)
