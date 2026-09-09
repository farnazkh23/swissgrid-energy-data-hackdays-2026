"""Format-neutral loading and adaptation of CSV, JSON, and optional Parquet files."""
from dataclasses import dataclass
import csv
import json
from pathlib import Path
from collections.abc import Mapping

from .column_mapping import ColumnMapping
from .dataset_contracts import DatasetContract
from .real_source_adapter import RealRecordMetadata, RealSourceAdapter, RealSourceConfig
from .source_registry import SourceRegistry


class DatasetFormatError(ValueError):
    """The file format or its tabular envelope is invalid."""


@dataclass(frozen=True, slots=True)
class RealDataset:
    dataset_id: str
    observations: tuple
    metadata: tuple[RealRecordMetadata, ...]
    raw_rows: tuple[dict, ...]
    registry: SourceRegistry
    contract: DatasetContract
    mapping: ColumnMapping
    pit_ready: bool
    known_at_policy: str

    def __post_init__(self):
        if len(self.observations) != len(self.metadata) or len(self.raw_rows) != len(self.observations):
            raise ValueError("observation metadata and raw rows must be aligned")


def detect_format(path: str | Path, format: str | None = None) -> str:
    selected = (format or Path(path).suffix.lstrip(".")).lower()
    if selected == "ndjson":
        selected = "json"
    if selected not in ("csv", "json", "parquet"):
        raise DatasetFormatError("supported formats are csv, json, and parquet")
    return selected


def load_rows(path: str | Path, *, format: str | None = None) -> tuple[dict, ...]:
    path = Path(path)
    selected = detect_format(path, format)
    if selected == "csv":
        with path.open(newline="", encoding="utf-8") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))
    if selected == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records", payload.get("data")), list):
            rows = payload.get("records", payload.get("data"))
        else:
            raise DatasetFormatError("JSON must contain a list or records/data list")
        if not all(isinstance(row, Mapping) for row in rows):
            raise DatasetFormatError("JSON records must be objects")
        return tuple(dict(row) for row in rows)
    try:
        import pandas as pd
    except ImportError:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("Parquet loading requires pandas or pyarrow; install explicitly") from exc
        table = parquet.read_table(path)
        return tuple(dict(row) for row in table.to_pylist())
    frame = pd.read_parquet(path)
    return tuple(dict(row) for row in frame.to_dict(orient="records"))


def load_real_dataset(path: str | Path, *, source_config: RealSourceConfig,
                      format: str | None = None) -> RealDataset:
    rows = load_rows(path, format=format)
    adapter = RealSourceAdapter(source_config)
    adapted = adapter.adapt_rows(rows)
    observations = tuple(item[0] for item in adapted)
    metadata = tuple(item[1] for item in adapted)
    contracts = {}
    units = {}
    countries = {}
    domains = {}
    for item in metadata:
        contracts.setdefault(item.source_id, None)
        units.setdefault(item.source_id, set()).add(item.unit)
        countries.setdefault(item.source_id, item.country)
        domains.setdefault(item.source_id, item.domain)
        if countries[item.source_id] != item.country or domains[item.source_id] != item.domain:
            raise ValueError("source metadata changes within one source_id")
    source_contracts = tuple(source_config.contract(source_id, countries[source_id], domains[source_id], units[source_id])
                            for source_id in sorted(contracts))
    registry = SourceRegistry(source_contracts)
    contract = DatasetContract(source_config.dataset_id, tuple(sorted(contracts)))
    # Adaptation has already enforced the required clocks. Keep this explicit
    # so downstream runners can refuse a manually constructed non-PIT dataset.
    pit_ready = source_config.known_at_policy == "require" or all(row.known_at is not None for row in observations)
    return RealDataset(source_config.dataset_id, observations, metadata, rows, registry, contract,
                       source_config.mapping, pit_ready, source_config.known_at_policy)


def load_configured_dataset(path: str | Path, config: Mapping) -> RealDataset:
    return load_real_dataset(path, source_config=RealSourceConfig.from_dict(config))
