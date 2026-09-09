"""Deterministic synthetic fixtures for tests only; never production providers."""
from datetime import datetime, timedelta
import json

from .availability import earliest_known_at, utc
from .ingestion import ReceivedRecord
from .observation_schema import Observation
from .source_contracts import Domain, HorizonAvailability, SourceContract


def mock_contract(domain: Domain = Domain.LOAD, *, unit: str = 'MW') -> SourceContract:
    return SourceContract('mock-' + Domain(domain).value, 'TEST', domain, 'synthetic-test-only',
                          'UTC', (unit,), timedelta(hours=1), timedelta(0),
                          'explicit synthetic sequence and predecessor',
                          HorizonAvailability(timedelta(days=-365), timedelta(days=365)),
                          'Synthetic fixtures for tests only; no provider data')


def mock_record(*, at: datetime, domain: Domain = Domain.LOAD, unit: str = 'MW',
                value: float | None = 1.0, revision_sequence: int = 0) -> ReceivedRecord:
    at = utc(at)
    if type(revision_sequence) is not int or revision_sequence < 0:
        raise ValueError('revision_sequence must be nonnegative integer')
    body = dict(event_time=at.isoformat(), valid_time=at.isoformat(),
                publication_time=at.isoformat(), value=value, unit=unit,
                revision_sequence=revision_sequence,
                supersedes_revision_id=None if revision_sequence == 0 else f'r{revision_sequence - 1}')
    return ReceivedRecord('mock-' + Domain(domain).value, 'synthetic-scalar',
                          f'r{revision_sequence}',
                          json.dumps(body, sort_keys=True, allow_nan=False).encode(), at)


class MockAdapter:
    """Decode only the fixture format. No network, random state, or wall clock."""
    def normalize(self, received: ReceivedRecord, *, normalized_at: datetime) -> Observation:
        body = json.loads(received.payload)
        publication = (datetime.fromisoformat(body['publication_time'])
                       if body['publication_time'] is not None else None)
        return Observation(
            source_id=received.source_id, source_record_id=received.source_record_id,
            revision_id=received.revision_id, revision_sequence=body['revision_sequence'],
            event_time=datetime.fromisoformat(body['event_time']),
            valid_time=datetime.fromisoformat(body['valid_time']), publication_time=publication,
            first_received_at=received.first_received_at, normalized_at=normalized_at,
            known_at=earliest_known_at(first_received_at=received.first_received_at,
                                      normalized_at=normalized_at, publication_time=publication),
            value=body['value'], unit=body['unit'],
            supersedes_revision_id=body['supersedes_revision_id'])
