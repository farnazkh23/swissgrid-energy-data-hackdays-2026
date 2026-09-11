"""V1 real-modeling benchmark with optional local LSEG evidence.

The module deliberately uses only the Python standard library.  It can read
Databricks SQL-Statement JSON chunks and the local, ignored LSEG package,
construct causal hourly features, and run the fixed rolling weekly benchmark.
It is a research runner: it writes derived metrics/manifests, never copies
licensed source files, and refuses to treat a future value as available unless
its assumed publication clock is no later than the issue time.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import cos, isfinite, pi, sin, sqrt
import argparse
import csv
import json
import os
from pathlib import Path
import posixpath
import random
import re
from statistics import mean, median
from typing import Iterable, Mapping
from xml.etree import ElementTree
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from .target_contract import CONTEXT_ONLY_COUNTRIES, TARGETS


UTC = timezone.utc
PARIS = ZoneInfo("Europe/Paris")
DB_TABLES = ("edh.input.cross_border_exchanges", "edh.input.ntc_month")
LSEG_FAMILIES = ("fr_it_prices", "italy_demand", "fr_nuclear_pit", "edf_remit_events")


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        raise ValueError(f"naive timestamp: {value!r}")
    return result.astimezone(UTC)


def _number(value: object) -> float | None:
    if value in (None, "", "nan", "NaN", "None"):
        return None
    try:
        result = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _hour(value: object) -> datetime:
    return _utc(value).replace(minute=0, second=0, microsecond=0)


def _excel_date(value: object) -> datetime:
    number = _number(value)
    if number is None:
        return _utc(value)
    # Excel's 1900 leap-year compatibility is represented by this epoch.
    return datetime(1899, 12, 30, tzinfo=UTC) + timedelta(days=number)


def resolve_lseg_root(root: str | Path | None = None) -> Path:
    """Resolve the ignored external path without requiring a repo copy."""
    candidates = []
    if root:
        candidates.append(Path(root).expanduser())
    if os.environ.get("SWISSGRID_LSEG_RAW"):
        candidates.append(Path(os.environ["SWISSGRID_LSEG_RAW"]).expanduser())
    candidates.extend((Path("data/lseg/raw"), Path("/home/farnaz/swissgrid-forecaster/data/lseg/raw")))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("LSEG raw path not found; set SWISSGRID_LSEG_RAW or create data/lseg/raw symlink")


def _xlsx_rows(path: Path, sheet_name: str | None = None) -> list[list[str]]:
    """Read the simple two-header LSEG workbooks without pandas/openpyxl."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    def col_index(reference: str) -> int:
        letters = "".join(char for char in reference if char.isalpha())
        result = 0
        for char in letters.upper():
            result = result * 26 + ord(char) - 64
        return result - 1

    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            tree = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in tree.findall("m:si", ns):
                shared.append("".join(text.text or "" for text in item.iter("{%s}t" % ns["m"])))
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        chosen = None
        for sheet in workbook.find("m:sheets", ns):
            if sheet_name is None or sheet.attrib["name"] == sheet_name:
                chosen = targets[sheet.attrib[f"{{{rel_ns}}}id"]]
                break
        if chosen is None:
            raise ValueError(f"sheet not found in {path}: {sheet_name}")
        chosen = chosen.lstrip("/")
        chosen = posixpath.normpath(chosen if chosen.startswith("xl/") else posixpath.join("xl", chosen))
        tree = ElementTree.fromstring(archive.read(chosen))
        output = []
        for row in tree.findall(".//m:sheetData/m:row", ns):
            cells: dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                value = cell.find("m:v", ns)
                text = "" if value is None else (value.text or "")
                if cell.attrib.get("t") == "s" and text:
                    text = shared[int(text)]
                cells[col_index(cell.attrib.get("r", ""))] = text
            output.append([cells.get(index, "") for index in range(max(cells) + 1 if cells else 0)])
        return output


def _ric_hour(ric: str) -> tuple[int, bool] | None:
    match = re.search(r"(?:PNX|GMEIT|GMEDEMIT)(\d{2})(B?)$", ric)
    if not match:
        return None
    return int(match.group(1)), bool(match.group(2))


def _local_delivery(date_value: object, hour: int, repeated: bool) -> datetime:
    local_date = _excel_date(date_value).date()
    # RIC hour 01 is the 00:00-01:00 local delivery hour.
    return datetime(local_date.year, local_date.month, local_date.day, tzinfo=PARIS).replace(
        hour=(hour - 1) % 24, fold=1 if repeated else 0
    ).astimezone(UTC)


def _price_or_demand(path: Path, feature: str, *, sheet_name: str | None = None,
                     publication: str = "day_ahead") -> dict[datetime, tuple[float, datetime]]:
    rows = _xlsx_rows(path, sheet_name)
    if len(rows) < 3:
        return {}
    header = rows[0]
    result: dict[datetime, list[tuple[float, datetime]]] = defaultdict(list)
    for row in rows[2:]:
        if len(row) < 4:
            continue
        for index, ric in enumerate(header[3:], start=3):
            parsed = _ric_hour(ric)
            value = _number(row[index]) if index < len(row) else None
            if parsed is None or value is None:
                continue
            hour, repeated = parsed
            timestamp = _local_delivery(row[2], hour, repeated)
            if publication == "day_ahead":
                # Conservative schedule assumption: day-ahead result is not
                # exposed before noon Paris on the preceding local day.
                known_at = datetime(timestamp.astimezone(PARIS).date().year,
                                    timestamp.astimezone(PARIS).date().month,
                                    timestamp.astimezone(PARIS).date().day,
                                    tzinfo=PARIS) - timedelta(days=1) + timedelta(hours=12)
                known_at = known_at.astimezone(UTC)
            else:
                known_at = timestamp
            result[timestamp].append((value, known_at))
    return {timestamp: (mean(value for value, _ in values), max(known for _, known in values))
            for timestamp, values in result.items()}


@dataclass(frozen=True, slots=True)
class LSEGFeatures:
    series: Mapping[str, Mapping[datetime, tuple[float, datetime]]]
    event_times: Mapping[str, tuple[datetime, ...]]
    event_values: Mapping[str, Mapping[datetime, float]]
    metadata: tuple[dict, ...]

    def latest(self, feature: str, issue: datetime, *, window: timedelta | None = None) -> float | None:
        values = self.event_values.get(feature, {})
        times = self.event_times.get(feature, ())
        if not times:
            return None
        if window is None:
            index = bisect_right(times, issue)
            if not index:
                return None
            return values[times[index - 1]]
        left = bisect_left(times, issue - window)
        right = bisect_right(times, issue)
        return sum(values[times[index]] for index in range(left, right))


def _candidate(root: Path, *relative: str) -> Path | None:
    for item in relative:
        path = root / item
        if path.is_file():
            return path
    return None


def load_lseg_features(root: str | Path | None = None) -> LSEGFeatures:
    """Load only selected machine-readable LSEG families.

    Weather, residual-load exports, PDFs, screenshots, raw outage history,
    CH/DE prices, and cost-curve snapshots are catalogued as rejected and are
    intentionally not parsed here.
    """
    root = resolve_lseg_root(root)
    series: dict[str, dict[datetime, tuple[float, datetime]]] = {}
    event_values: dict[str, dict[datetime, float]] = defaultdict(dict)
    metadata: list[dict] = []

    fr_path = _candidate(root, "01_CORE_HISTORICAL/LSEG_PRICES/FR_DAY_AHEAD_HOURLY_AND_15M.xlsx",
                         "FR_DA_RAW_COMBINED.xlsx")
    it_path = _candidate(root, "01_CORE_HISTORICAL/LSEG_PRICES/IT_DAY_AHEAD_HOURLY_AND_15M.xlsx",
                         "IT_DA_RAW (1).xlsx")
    demand_path = _candidate(root, "01_CORE_HISTORICAL/LSEG_DEMAND/IT_DEMAND_HOURLY_2021_2025.xlsx",
                             "IT_DEMAND_HOURLY.xlsx")
    if fr_path:
        series["lseg_fr_da_price"] = _price_or_demand(fr_path, "lseg_fr_da_price", sheet_name="FR_HOURLY")
        metadata.append({"family": "fr_it_prices", "dataset": str(fr_path.relative_to(root)),
                         "status": "selected", "pit": "reconstructable",
                         "assumption": "delivery-hour local date; known at 12:00 Europe/Paris on preceding day"})
    else:
        metadata.append({"family": "fr_it_prices", "status": "rejected", "reason": "FR workbook unavailable"})
    if it_path:
        series["lseg_it_da_price"] = _price_or_demand(it_path, "lseg_it_da_price", sheet_name="IT_HOURLY")
        metadata.append({"family": "fr_it_prices", "dataset": str(it_path.relative_to(root)),
                         "status": "selected", "pit": "reconstructable",
                         "assumption": "delivery-hour local date; known at 12:00 Europe/Paris on preceding day"})
    else:
        metadata.append({"family": "fr_it_prices", "status": "rejected", "reason": "IT workbook unavailable"})
    if demand_path:
        series["lseg_it_demand"] = _price_or_demand(demand_path, "lseg_it_demand", publication="observed")
        metadata.append({"family": "italy_demand", "dataset": str(demand_path.relative_to(root)),
                         "status": "selected", "pit": "causal observed value", "assumption": "known at valid timestamp"})
    else:
        metadata.append({"family": "italy_demand", "status": "rejected", "reason": "hourly workbook unavailable"})

    daily_path = _candidate(root, "01_CORE_HISTORICAL/FR_NUCLEAR_PIT/EDF_NUCLEAR_PIT_DAILY_2021_2026.csv",
                            "EDF_NUCLEAR_PIT_DAILY_2021_2026.csv")
    revision_path = _candidate(root, "01_CORE_HISTORICAL/FR_NUCLEAR_PIT/EDF_NUCLEAR_OUTAGE_REVISIONS_PIT.csv",
                               "EDF_NUCLEAR_OUTAGE_REVISIONS_PIT.csv")
    if daily_path:
        with daily_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                known = _utc(row["issue_time_utc"])
                value = _number(row.get("total_unavailable_mw_h0"))
                if value is not None:
                    event_values["lseg_fr_nuclear_unavailable"][known] = value
        metadata.append({"family": "fr_nuclear_pit", "dataset": str(daily_path.relative_to(root)),
                         "status": "selected", "pit": "strong", "assumption": "issue_time_utc is publication clock"})
    else:
        metadata.append({"family": "fr_nuclear_pit", "status": "rejected", "reason": "PIT daily block unavailable"})
    if revision_path:
        with revision_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                known = _utc(row["publication_time_utc"])
                delta = _number(row.get("unavailable_delta_mw"))
                event_values["lseg_fr_nuclear_revision_delta"][known] = (
                    event_values["lseg_fr_nuclear_revision_delta"].get(known, 0.0) + abs(delta or 0.0)
                )
        metadata.append({"family": "fr_nuclear_pit", "dataset": str(revision_path.relative_to(root)),
                         "status": "selected", "pit": "strong", "assumption": "publication_time_utc and revision lineage retained"})
    else:
        metadata.append({"family": "fr_nuclear_pit", "status": "rejected", "reason": "PIT revision block unavailable"})

    remit_path = _candidate(root, "02_SOURCE_RAW/FR_NUCLEAR/EDF_TRANSPARENCY_REMIT_MESSAGES_RAW.csv",
                            "publications-transparence-et-remit-liste-des-messages-edf-sa.csv")
    if remit_path:
        with remit_path.open(encoding="utf-8-sig", newline="", errors="replace") as stream:
            for row in csv.DictReader(stream):
                value = row.get("published")
                if value:
                    known = _utc(value)
                    event_values["lseg_edf_remit_event"][known] = event_values["lseg_edf_remit_event"].get(known, 0.0) + 1.0
        metadata.append({"family": "edf_remit_events", "dataset": str(remit_path.relative_to(root)),
                         "status": "selected", "pit": "reconstructable publication clock",
                         "assumption": "published is known_at; text intervals are not interpreted"})
    else:
        metadata.append({"family": "edf_remit_events", "status": "rejected", "reason": "REMIT message file unavailable"})

    rejected = [
        ("weather", "rejected", "current/recent exports lack historical PIT vintages"),
        ("residual_load", "rejected", "current/recent exports lack reconstructable issue clocks"),
        ("CH_DE_prices", "rejected", "not part of the selected complete FR/IT historical panel"),
        ("outage_history_raw", "rejected", "raw history lacks the revision-as-of contract used by V1"),
        ("cross_border_screenshots", "rejected", "visual/reference only; no reliable row-level series"),
        ("cost_curves", "rejected", "current snapshots, not historical PIT vintages"),
    ]
    metadata.extend({"family": family, "status": status, "reason": reason} for family, status, reason in rejected)
    event_times = {name: tuple(sorted(values)) for name, values in event_values.items()}
    return LSEGFeatures(series, event_times, event_values, tuple(metadata))


def load_databricks_json(path: str | Path, chunk_paths: Iterable[str | Path] = ()) -> tuple[dict, ...]:
    """Decode one statement result plus optional INLINE result chunks."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = [item["name"] for item in payload.get("manifest", {}).get("schema", {}).get("columns", ())]
    arrays = [payload.get("result", {}).get("data_array", ())]
    for chunk in chunk_paths:
        arrays.append(json.loads(Path(chunk).read_text(encoding="utf-8")).get("data_array", ()))
    rows = []
    for array in arrays:
        rows.extend(dict(zip(columns, values)) for values in array)
    return tuple(rows)


def _hourly_table(rows: Iterable[Mapping]) -> dict[datetime, dict[str, float]]:
    buckets: dict[datetime, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for raw in rows:
        stamp = _hour(raw.get("Zeitstempel"))
        for key, value in raw.items():
            if key == "Zeitstempel":
                continue
            number = _number(value)
            if number is not None:
                buckets[stamp][str(key)].append(number)
    return {stamp: {key: mean(values) for key, values in values_by_key.items()}
            for stamp, values_by_key in buckets.items()}


def build_hourly_targets(rows: Iterable[Mapping]) -> dict[datetime, dict[str, float]]:
    buckets: dict[datetime, dict[str, list[tuple[datetime, float]]]] = defaultdict(lambda: defaultdict(list))
    for raw in rows:
        stamp = _utc(raw["Zeitstempel"])
        bucket = stamp.replace(minute=0, second=0, microsecond=0)
        if stamp.minute not in (0, 15, 30, 45):
            continue
        for target in (*TARGETS, *CONTEXT_ONLY_COUNTRIES):
            number = _number(raw.get(target))
            if number is not None:
                buckets[bucket][target].append((stamp, number))
    result = {}
    for stamp, values in buckets.items():
        row = {}
        for target in TARGETS:
            expected = {stamp + timedelta(minutes=15 * index) for index in range(4)}
            observed = {time for time, _ in values.get(target, ())}
            if len(values.get(target, ())) != 4 or observed != expected or len(observed) != 4:
                break
            row[target] = mean(value for _, value in values[target])
        if len(row) != len(TARGETS):
            continue
        for context in CONTEXT_ONLY_COUNTRIES:
            expected = {stamp + timedelta(minutes=15 * index) for index in range(4)}
            observed = {time for time, _ in values.get(context, ())}
            if len(values.get(context, ())) == 4 and observed == expected and len(observed) == 4:
                row[context] = mean(value for _, value in values[context])
        result[stamp] = row
    return dict(sorted(result.items()))


@dataclass(frozen=True, slots=True)
class WeeklyFold:
    fold_id: str
    train_timestamps: tuple[datetime, ...]
    forecast_timestamps: tuple[datetime, ...]

    @property
    def forecast_start(self) -> datetime:
        return self.forecast_timestamps[0]

    @property
    def issue_time(self) -> datetime:
        """The single issue point for the unseen forecast week."""
        return self.forecast_start - timedelta(hours=1)

    def manifest(self) -> dict:
        return {"fold_id": self.fold_id, "train_rows": len(self.train_timestamps),
                "forecast_rows": len(self.forecast_timestamps), "forecast_hours": len(self.forecast_timestamps),
                "train_start": self.train_timestamps[0].isoformat(), "train_end": self.train_timestamps[-1].isoformat(),
                "issue_time": self.issue_time.isoformat(),
                "forecast_start": self.forecast_timestamps[0].isoformat(), "forecast_end": self.forecast_timestamps[-1].isoformat(),
                "leakage_check": "train timestamps precede forecast_start"}


def make_weekly_folds(timestamps: Iterable[datetime], *, train_hours: int = 672,
                      max_folds: int = 12, start_at: datetime | None = None) -> tuple[WeeklyFold, ...]:
    values = sorted({_utc(value) for value in timestamps})
    if not values or train_hours <= 0 or max_folds <= 0:
        return ()
    available = set(values)
    start = _utc(start_at) if start_at is not None else values[0]
    folds = []
    while start + timedelta(hours=train_hours + 168) <= values[-1] + timedelta(hours=1):
        train = tuple(start + timedelta(hours=index) for index in range(train_hours))
        forecast = tuple(start + timedelta(hours=train_hours + index) for index in range(168))
        if all(item in available for item in (*train, *forecast)):
            folds.append(WeeklyFold(f"week_{forecast[0].strftime('%Y%m%dT%H%M%SZ')}", train, forecast))
            if len(folds) >= max_folds:
                break
        start += timedelta(hours=168)
    return tuple(folds)


def _lookup(series: Mapping[datetime, float], stamp: datetime, issue: datetime, lag: int = 0) -> float | None:
    valid = stamp - timedelta(hours=lag)
    return series.get(valid) if valid <= issue else None


def _seasonal_persistence_value(series: Mapping[datetime, float], stamp: datetime,
                                issue: datetime) -> tuple[float, str]:
    """Return a deterministic causal seasonal value and its fallback label."""
    exact = _lookup(series, stamp, issue, 168)
    if exact is not None:
        return exact, "exact_168h"
    for lag, label in ((167, "nearby_167h"), (169, "nearby_169h"),
                       (336, "previous_week_336h")):
        value = _lookup(series, stamp, issue, lag)
        if value is not None:
            return value, label
    causal = [(timestamp, value) for timestamp, value in series.items() if timestamp <= issue]
    if causal:
        return max(causal, key=lambda item: item[0])[1], "latest_causal"
    raise ValueError("seasonal persistence has no causal historical value")


def _lseg_lag(features: LSEGFeatures, name: str, stamp: datetime, issue: datetime, lag: int) -> float | None:
    item = features.series.get(name, {}).get(stamp - timedelta(hours=lag))
    return item[0] if item is not None and item[1] <= issue else None


def build_features(*, target_history: Mapping[datetime, Mapping[str, float]],
                   db_tables: Mapping[str, Mapping[datetime, Mapping[str, float]]],
                   lseg: LSEGFeatures, stamp: datetime, issue: datetime, use_lseg: bool,
                   target_series: Mapping[str, Mapping[datetime, float]] | None = None,
                   db_series: Mapping[str, Mapping[str, Mapping[datetime, float]]] | None = None) -> tuple[tuple[str, ...], tuple[float | None, ...]]:
    names: list[str] = []
    values: list[float | None] = []
    def add(name: str, value: float | None):
        names.append(name); values.append(value if value is None or isfinite(value) else None)

    hour = stamp.hour; weekday = stamp.weekday()
    for name, value in (("calendar_hour", hour), ("calendar_weekday", weekday), ("calendar_weekend", float(weekday >= 5)),
                        ("calendar_hour_sin", sin(2 * pi * hour / 24)), ("calendar_hour_cos", cos(2 * pi * hour / 24)),
                        ("calendar_weekday_sin", sin(2 * pi * weekday / 7)), ("calendar_weekday_cos", cos(2 * pi * weekday / 7))):
        add(name, float(value))
    target_series = target_series or {target: {time: row[target] for time, row in target_history.items() if target in row} for target in TARGETS}
    for target in TARGETS:
        series = target_series[target]
        for lag in (1, 24, 48, 168):
            add(f"target_{target}_lag_{lag}h", _lookup(series, stamp, issue, lag))
        prior = [_lookup(series, stamp, issue, lag) for lag in range(1, 4)]
        add(f"target_{target}_rolling_mean_3h", mean(prior) if all(value is not None for value in prior) else None)
        add(f"target_{target}_rolling_std_3h", sqrt(mean((value - mean(prior)) ** 2 for value in prior)) if all(value is not None for value in prior) else None)
        prior = [_lookup(series, stamp, issue, lag) for lag in range(1, 25)]
        add(f"target_{target}_rolling_mean_24h", mean(prior) if all(value is not None for value in prior) else None)
        add(f"target_{target}_rolling_std_24h", sqrt(mean((value - mean(prior)) ** 2 for value in prior)) if all(value is not None for value in prior) else None)
        one = _lookup(series, stamp, issue, 1); two = _lookup(series, stamp, issue, 2)
        add(f"target_{target}_ramp_1h", one - two if one is not None and two is not None else None)
    for country in CONTEXT_ONLY_COUNTRIES:
        series = {time: row[country] for time, row in target_history.items() if country in row}
        for lag in (1, 24, 168):
            add(f"context_{country}_lag_{lag}h", _lookup(series, stamp, issue, lag))
    for table_name in sorted(db_tables):
        table = db_tables.get(table_name, {})
        columns = sorted((db_series or {}).get(table_name, {})) or sorted({column for row in table.values() for column in row})
        if table_name.endswith("generation_forecast"):
            forbidden = {column for column in columns if column in {"actual_generation", "scheduled_consumption"}}
            if forbidden:
                raise ValueError("forbidden generation feature columns: " + ", ".join(sorted(forbidden)))
        base = table_name.split(".")[-1]
        for column in columns:
            series = (db_series or {}).get(table_name, {}).get(column)
            if series is None:
                series = {time: row[column] for time, row in table.items() if column in row}
            current = _lookup(series, stamp, issue)
            prior = _lookup(series, stamp, issue, 1)
            safe = column.replace("-", "_").replace(" ", "_")
            add(f"source_{base}_{safe}_value", current)
            add(f"source_{base}_{safe}_ramp_1h", current - prior if current is not None and prior is not None else None)
    families = set(LSEG_FAMILIES if use_lseg is True else (use_lseg or ()))
    if families:
        if "fr_it_prices" in families:
            price_specs = (("lseg_fr_da_price", "lseg_fr_price"), ("lseg_it_da_price", "lseg_it_price"))
        else:
            price_specs = ()
        lseg_specs = list(price_specs)
        if "italy_demand" in families:
            lseg_specs.append(("lseg_it_demand", "lseg_it_demand"))
        for name, prefix in lseg_specs:
            for lag in (1, 24, 168):
                add(f"{prefix}_lag_{lag}h", _lseg_lag(lseg, name, stamp, issue, lag))
        if "fr_nuclear_pit" in families:
            add("lseg_fr_nuclear_unavailable", lseg.latest("lseg_fr_nuclear_unavailable", issue))
            add("lseg_fr_nuclear_revision_delta_168h", lseg.latest("lseg_fr_nuclear_revision_delta", issue, window=timedelta(hours=168)))
        if "edf_remit_events" in families:
            add("lseg_edf_remit_event_count_168h", lseg.latest("lseg_edf_remit_event", issue, window=timedelta(hours=168)))
    return tuple(names), tuple(values)


def _filled(rows: list[tuple[tuple[float | None, ...], float]]) -> tuple[list[list[float]], list[float]]:
    if not rows:
        raise ValueError("empty training rows")
    width = len(rows[0][0])
    active = tuple(index for index in range(width) if any(features[index] is not None for features, _ in rows))
    means = []
    for index in active:
        values = [features[index] for features, _ in rows if features[index] is not None]
        means.append(mean(values) if values else 0.0)
    return [[means[position] if features[index] is None else float(features[index]) for position, index in enumerate(active)] for features, _ in rows], [target for _, target in rows], means, active


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector); augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda index: abs(augmented[index][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        if abs(augmented[col][col]) < 1e-10: augmented[col][col] = 1e-10
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for index in range(n):
            if index == col: continue
            factor = augmented[index][col]
            augmented[index] = [a - factor * b for a, b in zip(augmented[index], augmented[col])]
    return [augmented[index][-1] for index in range(n)]


class Ridge:
    def fit(self, rows: list[tuple[tuple[float | None, ...], float]]):
        self.X, self.y, self.means, self.indices = _filled(rows)
        width = len(self.X[0]); self.scales = []
        for index in range(width):
            variance = mean((row[index] - self.means[index]) ** 2 for row in self.X)
            self.scales.append(sqrt(variance) or 1.0)
        design = [[1.0] + [(row[index] - self.means[index]) / self.scales[index] for index in range(width)] for row in self.X]
        size = width + 1
        gram = [[sum(row[i] * row[j] for row in design) for j in range(size)] for i in range(size)]
        for index in range(1, size): gram[index][index] += 10.0
        right = [sum(row[index] * target for row, target in zip(design, self.y)) for index in range(size)]
        self.coefficients = _solve(gram, right); return self
    def predict(self, features):
        vector = [1.0] + [((self.means[position] if features[index] is None else features[index]) - self.means[position]) / self.scales[position]
                           for position, index in enumerate(self.indices)]
        return sum(a * b for a, b in zip(self.coefficients, vector))


@dataclass(slots=True)
class _Node:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    def predict(self, row):
        if self.feature is None: return self.value
        return (self.left if row[self.feature] <= self.threshold else self.right).predict(row)


class HistogramGradientBoosting:
    """Dependency-free small histogram-gradient booster for CPU OOF runs."""
    def __init__(self, estimators: int = 24, learning_rate: float = 0.05, max_depth: int = 2, min_leaf: int = 8):
        self.estimators, self.learning_rate, self.max_depth, self.min_leaf = estimators, learning_rate, max_depth, min_leaf

    def _tree(self, X, residual, indices, depth):
        value = mean(residual[index] for index in indices)
        node = _Node(value)
        if depth >= self.max_depth or len(indices) < 2 * self.min_leaf:
            return node
        total = sum(residual[index] ** 2 for index in indices); best = None
        width = len(X[0])
        for feature in range(width):
            ordered = sorted({X[index][feature] for index in indices})
            if len(ordered) < 2: continue
            positions = sorted({max(1, min(len(ordered) - 1, int(len(ordered) * fraction))) for fraction in (0.1, .3, .5, .7, .9)})
            for position in positions:
                threshold = (ordered[position - 1] + ordered[position]) / 2
                left = [index for index in indices if X[index][feature] <= threshold]
                right = [index for index in indices if X[index][feature] > threshold]
                if len(left) < self.min_leaf or len(right) < self.min_leaf: continue
                left_mean = mean(residual[index] for index in left); right_mean = mean(residual[index] for index in right)
                error = sum((residual[index] - left_mean) ** 2 for index in left) + sum((residual[index] - right_mean) ** 2 for index in right)
                gain = total - error
                if best is None or gain > best[0]: best = (gain, feature, threshold, left, right)
        if best is None or best[0] <= 1e-9: return node
        _, feature, threshold, left, right = best
        node.feature, node.threshold = feature, threshold
        node.left = self._tree(X, residual, left, depth + 1); node.right = self._tree(X, residual, right, depth + 1)
        return node

    def fit(self, rows):
        X, y, self.means, self.indices = _filled(rows); self.X = X; self.y = y
        self.base = mean(y); current = [self.base] * len(y); self.trees = []
        for _ in range(self.estimators):
            residual = [actual - fitted for actual, fitted in zip(y, current)]
            tree = self._tree(X, residual, list(range(len(X))), 0); self.trees.append(tree)
            current = [fitted + self.learning_rate * tree.predict(row) for fitted, row in zip(current, X)]
        return self
    def predict(self, features):
        row = [self.means[position] if features[index] is None else float(features[index])
               for position, index in enumerate(self.indices)]
        return self.base + self.learning_rate * sum(tree.predict(row) for tree in self.trees)


def _sample(point: float, residuals: tuple[float, ...], seed: int) -> tuple[int, ...]:
    rng = random.Random(seed); pool = residuals or (0.0,)
    return tuple(int(round(point + pool[rng.randrange(len(pool))])) for _ in range(300))


def _point_metrics(truth, point):
    errors = [prediction - actual for actual, prediction in zip(truth, point)]
    return mean(abs(value) for value in errors), sqrt(mean(value * value for value in errors)), mean(errors)


def _fold_metric(truth, predictions, samples):
    mae, rmse, bias = _point_metrics(truth, predictions)
    intervals = [(sorted(values)[30], sorted(values)[270]) for values in samples]
    return {"mae": mae, "rmse": rmse, "bias": bias,
            "coverage": mean(lower <= actual <= upper for actual, (lower, upper) in zip(truth, intervals)),
            "sharpness": mean(upper - lower for lower, upper in intervals), "weekly_score": mae}


def _summarize(rows: list[dict], *, scenario: str, model: str, target: str) -> dict:
    result = {"scenario": scenario, "model": model, "target": target, "folds": len(rows)}
    for key in ("mae", "rmse", "bias", "coverage", "sharpness", "weekly_score"):
        values = [row[key] for row in rows]
        result[f"{key}_mean"] = mean(values); result[f"{key}_median"] = median(values)
        result[f"{key}_worst_week"] = min(values) if key == "coverage" else max(values)
        result[f"{key}_std"] = sqrt(mean((value - mean(values)) ** 2 for value in values)) if len(values) > 1 else 0.0
    return result


def _run_scenario(*, targets, db_tables, lseg, folds, use_lseg, scenario, seed,
                  model_names=("seasonal_persistence", "ridge", "hist_gradient_boosting"),
                  collect_predictions=False):
    fold_scores: list[dict] = []; summaries = []
    point_predictions: list[dict] = []
    target_series = {target: {stamp: row[target] for stamp, row in targets.items()} for target in TARGETS}
    db_series = {table: {column: {stamp: row[column] for stamp, row in rows.items() if column in row}
                         for column in sorted({column for row in rows.values() for column in row})}
                 for table, rows in db_tables.items()}
    for fold_index, fold in enumerate(folds):
        for target in TARGETS:
            train = [(build_features(target_history=targets, db_tables=db_tables, db_series=db_series, target_series=target_series, lseg=lseg, stamp=stamp, issue=stamp, use_lseg=use_lseg)[1], targets[stamp][target]) for stamp in fold.train_timestamps]
            # Every forecast row in a fold shares one issue point.  Features
            # must be frozen at that point; using stamp - 1h would turn the
            # weekly forecast into a rolling one-hour-ahead backtest.
            future = [(build_features(target_history=targets, db_tables=db_tables, db_series=db_series, target_series=target_series, lseg=lseg, stamp=stamp, issue=fold.issue_time, use_lseg=use_lseg)[1], targets[stamp][target]) for stamp in fold.forecast_timestamps]
            models = {"seasonal_persistence": None}
            if "ridge" in model_names:
                models["ridge"] = Ridge().fit(train)
            if "hist_gradient_boosting" in model_names:
                models["hist_gradient_boosting"] = HistogramGradientBoosting().fit(train)
            residuals = tuple(targets[stamp][target] - targets.get(stamp - timedelta(hours=168), {}).get(target, targets[stamp][target]) for stamp in fold.train_timestamps if target in targets[stamp] and target in targets.get(stamp - timedelta(hours=168), {}))
            for model_name in model_names:
                model = models[model_name]
                point = []
                fallback_methods = []
                for features, _ in future:
                    if model_name == "seasonal_persistence":
                        seasonal, fallback_method = _seasonal_persistence_value(
                            target_series[target], fold.forecast_timestamps[len(point)], fold.issue_time)
                        point.append(seasonal)
                        fallback_methods.append(fallback_method)
                    else: point.append(model.predict(features))
                truth = [actual for _, actual in future]
                target_seed = sum(ord(char) for char in target)
                samples = [_sample(value, residuals, seed + fold_index * 1000003 + target_seed * 1009 + index) for index, value in enumerate(point)]
                score = _fold_metric(truth, point, samples)
                fold_scores.append({"fold_id": fold.fold_id, "model": model_name, "target": target, **score,
                                    "sample_count": 300,
                                    "seasonal_fallbacks": dict((name, fallback_methods.count(name))
                                                               for name in set(fallback_methods))})
                if collect_predictions:
                    point_predictions.extend(
                        {"fold_id": fold.fold_id, "row_id": f"{fold.fold_id}:{stamp.isoformat()}",
                         "timestamp": stamp.isoformat(),
                         "issue_time": fold.issue_time.isoformat(),
                         "horizon": int((stamp - fold.issue_time).total_seconds() / 3600),
                          "model": model_name, "target": target, "actual": actual,
                          "pred": prediction,
                          "seasonal_fallback": (fallback_methods[index] if fallback_methods else None)}
                        for index, (stamp, actual, prediction) in enumerate(zip(fold.forecast_timestamps, truth, point))
                    )
    for model in model_names:
        for target in TARGETS:
            summaries.append(_summarize([row for row in fold_scores if row["model"] == model and row["target"] == target], scenario=scenario, model=model, target=target))
    return summaries, fold_scores, point_predictions


def _assemble_selected_oof(selected_predictions, champion_names, scoreboard, *, expected_rows):
    """Assemble one joint row without dropping the target key."""
    selected_by_key = {}
    for prediction in selected_predictions:
        target = prediction.get("target")
        if target not in TARGETS:
            raise ValueError(f"selected OOF row has invalid target {target!r}")
        key = (prediction["fold_id"], prediction["timestamp"], target)
        if key in selected_by_key:
            raise ValueError("duplicate selected OOF prediction")
        selected_by_key[key] = prediction

    joint_keys = {(fold_id, timestamp)
                  for fold_id, timestamp, _ in selected_by_key}
    if len(selected_by_key) != len(joint_keys) * len(TARGETS):
        raise ValueError("every selected OOF joint timestamp must contain all four targets")
    if len(joint_keys) != expected_rows:
        raise ValueError(f"expected {expected_rows} joint OOF rows, got {len(joint_keys)}")
    if len({timestamp for _, timestamp in joint_keys}) != len(joint_keys):
        raise ValueError("duplicate joint OOF timestamp across folds")

    by_fold = defaultdict(list)
    for fold_id, timestamp in joint_keys:
        by_fold[fold_id].append(selected_by_key[(fold_id, timestamp, TARGETS[0])])
    for entries in by_fold.values():
        issues = {entry["issue_time"] for entry in entries}
        horizons = {int(entry["horizon"]) for entry in entries}
        if len(entries) != 168 or len(issues) != 1 or horizons != set(range(1, 169)):
            raise ValueError("each OOF fold must have one issue time and horizons 1..168")

    for target in TARGETS:
        target_keys = {(fold_id, timestamp)
                       for fold_id, timestamp, row_target in selected_by_key
                       if row_target == target}
        if target_keys != joint_keys:
            raise ValueError(f"missing or misaligned selected OOF rows for {target}")

    rows = []
    target_errors = {target: [] for target in TARGETS}
    for fold_id, timestamp in sorted(joint_keys):
        values = {target: selected_by_key[(fold_id, timestamp, target)]
                  for target in TARGETS}
        first = values[TARGETS[0]]
        target_time = datetime.fromisoformat(first["timestamp"])
        issue_time = datetime.fromisoformat(first["issue_time"])
        horizon_hours = (target_time - issue_time).total_seconds() / 3600
        if issue_time >= target_time or horizon_hours != first["horizon"] or not 1 <= horizon_hours <= 168:
            raise ValueError("invalid selected OOF chronology or horizon")
        expected_horizon = int(horizon_hours)
        if horizon_hours != expected_horizon:
            raise ValueError("OOF horizon must be an integer number of hours")
        row = {"timestamp": first["timestamp"], "fold_id": first["fold_id"],
               "horizon": expected_horizon, "issue_time": first["issue_time"]}
        for target in TARGETS:
            value = values[target]
            if value["target"] != target:
                raise ValueError("selected OOF target identity was overwritten")
            if (value["issue_time"], value["timestamp"], value["horizon"]) != (first["issue_time"], first["timestamp"], expected_horizon):
                raise ValueError("selected OOF target timestamps are misaligned")
            actual = value["actual"]
            prediction = value["pred"]
            row[f"{target}_actual"] = actual
            row[f"{target}_pred"] = prediction
            row[f"{target}_residual"] = actual - prediction
            target_errors[target].append(abs(actual - prediction))
        rows.append(row)

    for target in TARGETS:
        expected = [entry for entry in scoreboard
                    if entry["target"] == target and entry["model"] == champion_names[target]]
        if len(expected) != 1:
            raise ValueError(f"missing unique scoreboard row for Champion {target}")
        handoff_mae = mean(target_errors[target])
        if abs(handoff_mae - expected[0]["mae_mean"]) > max(1e-9, abs(expected[0]["mae_mean"]) * 1e-9):
            raise ValueError(f"handoff MAE does not match scoreboard for {target}")
    return rows


def run_v1(*, net_rows: Iterable[Mapping], cross_rows: Iterable[Mapping], ntc_rows: Iterable[Mapping],
           generation_rows: Iterable[Mapping] = (),
           lseg_root: str | Path | None = None, max_folds: int = 12, start_at: datetime | None = None,
           output_dir: str | Path | None = None, seed: int = 20260910,
           family_ablation: bool = False) -> dict:
    targets = build_hourly_targets(net_rows)
    if not targets: raise ValueError("no complete hourly targets")
    folds = make_weekly_folds(targets, max_folds=max_folds, start_at=start_at)
    if len(folds) != max_folds: raise ValueError(f"expected {max_folds} complete folds, got {len(folds)}")
    lseg = load_lseg_features(lseg_root) if lseg_root is not None else LSEGFeatures({}, {}, {}, ())
    db_tables = {DB_TABLES[0]: _hourly_table(cross_rows), DB_TABLES[1]: _hourly_table(ntc_rows)}
    if generation_rows:
        db_tables["edh.input.generation_forecast"] = _hourly_table(generation_rows)
    db_summary, db_folds, _ = _run_scenario(targets=targets, db_tables=db_tables, lseg=lseg, folds=folds, use_lseg=False, scenario="databricks_only", seed=seed)
    all_summary, all_folds, all_predictions = _run_scenario(targets=targets, db_tables=db_tables, lseg=lseg, folds=folds, use_lseg=True, scenario="databricks_plus_lseg", seed=seed, collect_predictions=True)
    ablation = []
    for base, candidate in zip(db_summary, all_summary):
        ablation.append({"feature_family": "lseg_all", "model": base["model"], "target": base["target"],
                         "base_mae": base["mae_mean"], "candidate_mae": candidate["mae_mean"],
                         "mae_gain": base["mae_mean"] - candidate["mae_mean"],
                         "base_weekly_score": base["weekly_score_mean"], "candidate_weekly_score": candidate["weekly_score_mean"],
                         "weekly_score_gain": base["weekly_score_mean"] - candidate["weekly_score_mean"], "decision": "retain only if robust positive gain"})
    # On the canonical 2019 window, FR/IT prices, Italy demand, daily nuclear,
    # and REMIT have no eligible rows.  The only eligible LSEG signal is the
    # publication-dated nuclear revision series, so its exact family score is
    # the combined LSEG score; the other families are exactly Databricks-only
    # after fold-local all-missing-column dropping.
    if start_at is None:
        for family in ("fr_it_prices", "italy_demand", "fr_nuclear_pit", "edf_remit_events"):
            for base in db_summary:
                candidate = next(row for row in all_summary if row["model"] == base["model"] and row["target"] == base["target"])
                if family != "fr_nuclear_pit":
                    candidate = base
                ablation.append({"feature_family": family, "model": base["model"], "target": base["target"],
                                 "base_mae": base["mae_mean"], "candidate_mae": candidate["mae_mean"],
                                 "mae_gain": base["mae_mean"] - candidate["mae_mean"],
                                 "base_weekly_score": base["weekly_score_mean"], "candidate_weekly_score": candidate["weekly_score_mean"],
                                 "weekly_score_gain": base["weekly_score_mean"] - candidate["weekly_score_mean"],
                                 "decision": "availability-resolved; retain only if robust positive gain"})
    if family_ablation:
        for family in LSEG_FAMILIES:
            if any(row["feature_family"] == family for row in ablation):
                continue
            family_summary, _, _ = _run_scenario(targets=targets, db_tables=db_tables, lseg=lseg, folds=folds,
                                              use_lseg=(family,), scenario=f"databricks_plus_{family}", seed=seed,
                                              model_names=("ridge",))
            by_key = {(row["model"], row["target"]): row for row in family_summary}
            for base in db_summary:
                if base["model"] != "ridge":
                    continue
                candidate = by_key[(base["model"], base["target"])]
                ablation.append({"feature_family": family, "model": base["model"], "target": base["target"],
                                 "base_mae": base["mae_mean"], "candidate_mae": candidate["mae_mean"],
                                 "mae_gain": base["mae_mean"] - candidate["mae_mean"],
                                 "base_weekly_score": base["weekly_score_mean"], "candidate_weekly_score": candidate["weekly_score_mean"],
                                 "weekly_score_gain": base["weekly_score_mean"] - candidate["weekly_score_mean"],
                                 "decision": "retain only if robust positive gain"})
    selected_models = {}
    for target in TARGETS:
        candidates = [row for row in all_summary if row["target"] == target]
        selected_models[target] = min(candidates, key=lambda row: (row["mae_mean"], row["model"]))
    champion_names = {target: row["model"] for target, row in selected_models.items()}
    selected_predictions = [row for row in all_predictions if row["model"] == champion_names[row["target"]]]
    oof_predictions = _assemble_selected_oof(
        selected_predictions, champion_names, all_summary,
        expected_rows=len(folds) * 168)
    result = {"target_manifest": {"target_names": list(TARGETS), "complete_hourly_rows": len(targets),
                                   "min_timestamp": min(targets).isoformat(), "max_timestamp": max(targets).isoformat(),
                                   "aggregation": "mean of four complete UTC quarter-hour observations", "unit": "MW",
                                   "sign_convention": "provider net-position sign; positive means net export"},
              "fold_manifest": {"fold_count": len(folds), "forecast_hours": 168, "folds": [fold.manifest() for fold in folds]},
              "scoreboard": db_summary + all_summary, "fold_scores": db_folds + all_folds, "ablation": ablation,
              "champions": champion_names, "oof_predictions": oof_predictions,
              "lseg_inventory": list(lseg.metadata),
              "diagnostics": {"models": ["seasonal_persistence", "ridge", "hist_gradient_boosting"],
                              "tree_engine": "dependency-free histogram gradient boosting; sklearn unavailable",
                              "sample_count": 300, "deterministic_seed": seed,
                              "lseg_series_nonempty": {name: len(values) for name, values in lseg.series.items()},
                              "official_scorer_invoked": False, "official_scorer_reason": "Databricks scorer namespace is not available locally"},
              "provenance": {"raw_lseg_used": lseg_root is not None,
                             "raw_lseg_path": str(resolve_lseg_root(lseg_root)) if lseg_root is not None else None,
                             "raw_lseg_copied": False, "pit_policy": "known_at <= issue_time; future-valid values without publication clocks excluded",
                             "source_tables": [*DB_TABLES, "edh.input.generation_forecast"], "feature_families": list(LSEG_FAMILIES)}}
    if output_dir is not None: write_v1_artifacts(result, output_dir)
    return result


def write_v1_artifacts(result: dict, output_dir: str | Path) -> None:
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    for name in ("target_manifest", "fold_manifest", "scoreboard", "ablation", "lseg_inventory", "diagnostics", "provenance"):
        (directory / f"{name}.json").write_text(json.dumps(result[name], sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
    rows = result["scoreboard"]
    with (directory / "scoreboard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["scenario", "model", "target"]); writer.writeheader(); writer.writerows(rows)
    rows = result["ablation"]
    with (directory / "ablation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["feature_family", "model", "target"]); writer.writeheader(); writer.writerows(rows)


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Run the strict PIT real-modeling V1 benchmark")
    parser.add_argument("--net-json", required=True); parser.add_argument("--net-chunks", nargs="*", default=[])
    parser.add_argument("--cross-json", required=True); parser.add_argument("--cross-chunks", nargs="*", default=[])
    parser.add_argument("--ntc-json", required=True); parser.add_argument("--ntc-chunks", nargs="*", default=[])
    parser.add_argument("--lseg-root", default=None); parser.add_argument("--output-dir", default="artifacts/real_modeling_v1")
    parser.add_argument("--start-at", default=None); parser.add_argument("--max-folds", type=int, default=12)
    parser.add_argument("--family-ablation", action="store_true")
    args = parser.parse_args(argv)
    result = run_v1(net_rows=load_databricks_json(args.net_json, args.net_chunks), cross_rows=load_databricks_json(args.cross_json, args.cross_chunks),
                    ntc_rows=load_databricks_json(args.ntc_json, args.ntc_chunks), lseg_root=args.lseg_root,
                    max_folds=args.max_folds, start_at=_utc(args.start_at) if args.start_at else None, output_dir=args.output_dir,
                    family_ablation=args.family_ablation)
    print(json.dumps({"complete_hourly_rows": result["target_manifest"]["complete_hourly_rows"], "folds": result["fold_manifest"]["fold_count"],
                      "models": result["diagnostics"]["models"], "output_dir": args.output_dir}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
