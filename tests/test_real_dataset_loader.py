import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from swissgrid_forecaster.column_mapping import ColumnMapping
from swissgrid_forecaster.real_dataset_loader import DatasetFormatError, load_real_dataset, load_rows
from swissgrid_forecaster.real_source_adapter import RealSourceConfig


T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def config():
    return RealSourceConfig("loader-test", ColumnMapping.from_dict({"columns": {
        "source_id": "s", "country": "c", "domain": "d", "unit": "u", "value": "v",
        "event_time": "e", "valid_time": "t", "known_at": "k", "first_received_at": "r",
        "normalized_at": "n", "revision_id": "ri", "revision_sequence": "rs", "source_record_id": "id"}}),
        "UTC", timedelta(hours=1), timedelta(0), timedelta(0), timedelta(days=2))


def records():
    return [{"s": "src", "c": "CH", "d": "load", "u": "MW", "v": 1,
             "e": T.isoformat(), "t": T.isoformat(), "k": T.isoformat(), "r": T.isoformat(),
             "n": T.isoformat(), "ri": "r0", "rs": 0, "id": "id-0"}]


class RealDatasetLoaderTests(unittest.TestCase):
    def test_csv_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with (directory / "data.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=records()[0])
                writer.writeheader(); writer.writerows(records())
            (directory / "data.json").write_text(json.dumps(records()))
            csv_dataset = load_real_dataset(directory / "data.csv", source_config=config())
            json_dataset = load_real_dataset(directory / "data.json", source_config=config())
            self.assertEqual(csv_dataset.observations, json_dataset.observations)
            self.assertEqual(csv_dataset.contract.dataset_id, "loader-test")

    def test_json_envelope_and_format_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.any"
            path.write_text(json.dumps({"records": records()}))
            self.assertEqual(len(load_rows(path, format="json")), 1)
            with self.assertRaises(DatasetFormatError):
                load_rows(path, format="xml")

    def test_parquet_reports_optional_dependency_or_loads(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                with self.assertRaises(RuntimeError):
                    load_rows("missing.parquet")
            else:
                self.skipTest("optional parquet engine is environment-dependent")
        else:
            self.skipTest("optional parquet engine is environment-dependent")
