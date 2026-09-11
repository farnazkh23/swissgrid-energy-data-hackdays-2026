"""Canonical timestamp conversion for provider and model clocks."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
EUROPE_ZURICH = ZoneInfo("Europe/Zurich")


def canonical_utc(value: datetime, source_timezone: str = "UTC") -> datetime:
    """Convert one source timestamp to aware UTC exactly once.

    Databricks net-position, exchange, and NTC tables expose local
    Europe/Zurich wall-clock timestamps.  A nonexistent spring-forward wall
    time is rejected, as is an ambiguous autumn wall time without an explicit
    fold, so DST data is never silently invented or collapsed.
    """
    if not isinstance(value, datetime):
        raise TypeError("timestamp must be a datetime")
    wall = value.replace(tzinfo=None)
    if source_timezone == "UTC":
        return wall.replace(tzinfo=UTC)
    if source_timezone != "Europe/Zurich":
        raise ValueError(f"unsupported source timezone: {source_timezone}")
    first = wall.replace(tzinfo=EUROPE_ZURICH, fold=0)
    second = wall.replace(tzinfo=EUROPE_ZURICH, fold=1)
    first_back = first.astimezone(EUROPE_ZURICH).replace(tzinfo=None)
    second_back = second.astimezone(EUROPE_ZURICH).replace(tzinfo=None)
    if first_back != wall and second_back != wall:
        raise ValueError(f"nonexistent Europe/Zurich local timestamp: {wall.isoformat()}")
    if first.utcoffset() != second.utcoffset():
        raise ValueError(f"ambiguous Europe/Zurich local timestamp requires fold: {wall.isoformat()}")
    return first.astimezone(UTC)


def source_query_bounds(start: datetime, end_exclusive: datetime, source_timezone: str):
    """Return naive Spark query bounds in the source table's wall-clock domain."""
    if source_timezone == "UTC":
        return start.astimezone(UTC).replace(tzinfo=None), end_exclusive.astimezone(UTC).replace(tzinfo=None)
    if source_timezone == "Europe/Zurich":
        return (start.astimezone(EUROPE_ZURICH).replace(tzinfo=None),
                end_exclusive.astimezone(EUROPE_ZURICH).replace(tzinfo=None))
    raise ValueError(f"unsupported source timezone: {source_timezone}")
