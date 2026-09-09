"""Pre-model data quality and point-in-time readiness audit."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from collections import Counter, defaultdict
import math

from .availability import is_available, utc
from .real_dataset_loader import RealDataset


@dataclass(frozen=True, slots=True)
class DataAuditReport:
    dataset_id: str
    issue_time: datetime
    rows: int
    date_range: tuple[str | None, str | None]
    cadence_seconds: dict
    missingness: dict
    duplicates: dict
    revision_frequency: dict
    timezone_dst_issues: dict
    unit_consistency: dict
    gaps: dict
    suspicious_future_availability: dict
    stale_periods: dict
    feature_availability_by_forecast_horizon: dict
    pit_ready: bool
    ready_for_forecast: bool
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    source_coverage: dict = field(default_factory=dict)
    revisions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id, "issue_time": self.issue_time.isoformat(),
            "rows": self.rows, "date_range": list(self.date_range),
            "cadence_seconds": self.cadence_seconds, "missingness": self.missingness,
            "duplicates": self.duplicates, "revision_frequency": self.revision_frequency,
            "timezone_dst_issues": self.timezone_dst_issues,
            "unit_consistency": self.unit_consistency, "gaps": self.gaps,
            "suspicious_future_availability": self.suspicious_future_availability,
            "stale_periods": self.stale_periods,
            "feature_availability_by_forecast_horizon": self.feature_availability_by_forecast_horizon,
            "pit_ready": self.pit_ready, "ready_for_forecast": self.ready_for_forecast,
            "source_coverage": self.source_coverage, "revisions": self.revisions,
            "warnings": list(self.warnings), "errors": list(self.errors),
        }


def _timestamp(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _per_source(observations):
    grouped = defaultdict(list)
    for row in observations:
        grouped[row.source_id].append(row)
    return {source: tuple(sorted(rows, key=lambda row: (row.event_time, row.source_record_id, row.revision_sequence)))
            for source, rows in grouped.items()}


def audit_dataset(dataset: RealDataset, *, issue_time: datetime,
                  forecast_horizons: tuple[timedelta, ...] = ()) -> DataAuditReport:
    if not isinstance(dataset, RealDataset):
        raise ValueError("RealDataset is required")
    issue = utc(issue_time, "issue_time")
    rows = tuple(dataset.observations)
    grouped = _per_source(rows)
    warnings, errors = [], []
    event_times = [row.event_time for row in rows]
    date_range = (min(event_times).isoformat(), max(event_times).isoformat()) if event_times else (None, None)
    cadence, gaps, stale = {}, {}, {}
    for source, source_rows in grouped.items():
        unique_events = sorted(set(row.event_time for row in source_rows))
        intervals = [(right - left).total_seconds() for left, right in zip(unique_events, unique_events[1:]) if right > left]
        typical = median(intervals) if intervals else None
        cadence[source] = typical
        source_gaps = []
        source_stale = []
        if typical:
            for left, right in zip(unique_events, unique_events[1:]):
                seconds = (right - left).total_seconds()
                if seconds > typical * 1.5:
                    source_gaps.append({"start": left.isoformat(), "end": right.isoformat(), "duration_seconds": seconds})
                if seconds > typical * 2:
                    source_stale.append({"start": left.isoformat(), "end": right.isoformat(),
                                         "duration_seconds": seconds, "kind": "historical_gap"})
            latest = max((row.event_time for row in source_rows if row.known_at <= issue), default=None)
            if latest is not None and (issue - latest).total_seconds() > typical * 2:
                source_stale.append({"start": latest.isoformat(), "end": issue.isoformat(),
                                     "duration_seconds": (issue - latest).total_seconds()})
        gaps[source], stale[source] = source_gaps, source_stale
    missingness = {}
    for source, source_rows in grouped.items():
        missing = sum(row.value is None for row in source_rows)
        missingness[source] = {"rows": len(source_rows), "missing_rows": missing,
                               "fraction": missing / len(source_rows) if source_rows else None}
    identities = Counter((row.source_id, row.source_record_id, row.revision_id) for row in rows)
    duplicate_rows = sum(count - 1 for count in identities.values() if count > 1)
    duplicates = {"duplicate_identity_groups": sum(count > 1 for count in identities.values()),
                  "duplicate_rows_after_first": duplicate_rows}
    revision_frequency = {}
    revisions = {}
    for source, source_rows in grouped.items():
        by_record = defaultdict(list)
        for row in source_rows:
            by_record[row.source_record_id].append(row)
        keys = set(by_record)
        revised_series = {key for key, values in by_record.items()
                          if len({row.revision_id for row in values}) > 1}
        revised = sum(len({row.revision_id for row in values}) - 1 for values in by_record.values()
                      if len({row.revision_id for row in values}) > 1)
        revision_frequency[source] = {"revision_rows": revised, "logical_series": len(keys),
                                      "fraction_of_rows": revised / len(source_rows) if source_rows else 0}
        revisions[source] = {"logical_series": len(keys), "revised_series": len(revised_series),
                             "revision_rows_after_first": revised,
                             "lineage_edges": sum(row.supersedes_revision_id is not None for row in source_rows),
                             "revision_ids": sorted({row.revision_id for row in source_rows})}
    units = defaultdict(set)
    for row in rows:
        units[row.source_id].add(row.unit)
    unit_consistency = {source: {"units": sorted(values), "consistent": len(values) == 1}
                        for source, values in sorted(units.items())}
    timezone_issues = {"configured_timezone": None, "naive_timestamp_values": 0,
                       "unparseable_timestamp_values": 0, "mixed_offsets": False,
                       "dst_review_required": False}
    # The raw envelope is retained by the loader so this audit can report
    # timezone problems even though the adapter rejects them before modeling.
    time_fields = ("event_time", "valid_time", "known_at", "first_received_at", "normalized_at", "publication_time", "issue_time")
    mapping = dataset.mapping
    timezone_issues["configured_timezone"] = next(
        (source.timezone for source in dataset.registry.list()), None)
    offsets = set()
    if dataset.raw_rows and dataset.metadata:
        for raw in dataset.raw_rows:
            for field in time_fields:
                column = mapping.column_for(field)
                value = raw.get(column) if column is not None else mapping.value(raw, field, default=None)
                if value in (None, ""):
                    continue
                parsed = _timestamp(value)
                if parsed is None:
                    timezone_issues["unparseable_timestamp_values"] += 1
                elif parsed.tzinfo is None or parsed.utcoffset() is None:
                    timezone_issues["naive_timestamp_values"] += 1
                else:
                    offsets.add(parsed.utcoffset())
    timezone_issues["mixed_offsets"] = len(offsets) > 1
    timezone_issues["dst_review_required"] = bool(
        timezone_issues["naive_timestamp_values"] or timezone_issues["mixed_offsets"] or
        (timezone_issues["configured_timezone"] not in (None, "UTC", "Etc/UTC")))
    future = {source: sum(row.known_at <= issue and row.event_time > issue for row in source_rows)
              for source, source_rows in sorted(grouped.items())}
    if any(future.values()):
        warnings.append("some future event times were known by issue; review forecast-versus-observation semantics")
    horizons = forecast_horizons or tuple(sorted({row.valid_time - issue for row in rows if row.valid_time > issue}))[:10]
    horizon_availability = {}
    for horizon in horizons:
        key = str(horizon)
        target_time = issue + horizon
        horizon_availability[key] = {source: any(row.known_at <= issue and row.valid_time == target_time
                                                for row in source_rows)
                                     for source, source_rows in sorted(grouped.items())}
    source_coverage = {}
    for source, source_rows in sorted(grouped.items()):
        eligible = tuple(row for row in source_rows if row.known_at <= issue)
        source_coverage[source] = {
            "rows": len(source_rows),
            "distinct_event_times": len({row.event_time for row in source_rows}),
            "distinct_valid_times": len({row.valid_time for row in source_rows}),
            "available_as_of_issue": len(eligible),
            "available_fraction": len(eligible) / len(source_rows) if source_rows else None,
            "countries": sorted({metadata.country for row, metadata in zip(rows, dataset.metadata)
                                  if row.source_id == source}),
            "domains": sorted({metadata.domain.value for row, metadata in zip(rows, dataset.metadata)
                                if row.source_id == source}),
            "units": sorted({row.unit for row in source_rows}),
            "date_range": [min(row.event_time for row in source_rows).isoformat(),
                           max(row.event_time for row in source_rows).isoformat()],
        }
    pit_ready = bool(dataset.pit_ready)
    if not pit_ready:
        errors.append("point-in-time metadata is unresolved")
    elif dataset.known_at_policy != "require":
        warnings.append("known_at was reconstructed from documented publication-lag policy")
    try:
        for contract in dataset.registry.list():
            contract.require_forecast_metadata()
    except ValueError as exc:
        errors.append(str(exc))
    if not rows:
        errors.append("dataset has no rows")
    if duplicate_rows:
        warnings.append("duplicate source/revision identities detected")
    if any(not report["consistent"] for report in unit_consistency.values()):
        errors.append("unit consistency is unresolved")
    ready = pit_ready and not errors
    return DataAuditReport(dataset.dataset_id, issue, len(rows), date_range, cadence, missingness,
                           duplicates, revision_frequency, timezone_issues, unit_consistency,
                           gaps, future, stale, horizon_availability, pit_ready, ready,
                           tuple(sorted(set(warnings))), tuple(sorted(set(errors))),
                           source_coverage, revisions)
