"""Explicit UTC clocks and conservative normalized-evidence availability."""

from datetime import datetime, timezone


def utc(value: datetime, name: str = "timestamp") -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def earliest_known_at(
    *, first_received_at: datetime, normalized_at: datetime,
    publication_time: datetime | None = None,
    permitted_at: datetime | None = None,
) -> datetime:
    """Maximum of receipt, normalization, known publication and policy embargo.

    Callers must supply actual completed receipt/normalization times. This pure
    function never infers historical access from provider publication alone.
    """
    received = utc(first_received_at, "first_received_at")
    normalized = utc(normalized_at, "normalized_at")
    if normalized < received:
        raise ValueError("normalized_at precedes completed receipt")
    times = [received, normalized]
    for name, value in (("publication_time", publication_time), ("permitted_at", permitted_at)):
        if value is not None:
            times.append(utc(value, name))
    return max(times)


def is_available(known_at: datetime, issue_time: datetime) -> bool:
    """Invalid or absent clocks raise; future knowledge is ineligible."""
    return utc(known_at, "known_at") <= utc(issue_time, "issue_time")
