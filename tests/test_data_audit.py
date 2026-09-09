import csv
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from swissgrid_forecaster.column_mapping import ColumnMapping
from swissgrid_forecaster.data_audit import audit_dataset
from swissgrid_forecaster.real_dataset_loader import load_real_dataset
from swissgrid_forecaster.real_source_adapter import RealSourceConfig


T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_dataset(rows):
    mapping = ColumnMapping.from_dict({"columns": {"source_id": "s", "country": "c", "domain": "d", "unit": "u",
        "value": "v", "event_time": "e", "valid_time": "t", "known_at": "k", "first_received_at": "r",
        "normalized_at": "n", "revision_id": "ri", "revision_sequence": "rs", "source_record_id": "id"}})
    source = RealSourceConfig("audit", mapping, "UTC", timedelta(hours=1), timedelta(0), timedelta(0), timedelta(days=2))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
        # The loader materializes the dataset before the temporary path is removed.
        return load_real_dataset(path, source_config=source)


def row(index, *, value=1, revision=0, known=None, record=None):
    event = T + timedelta(hours=index)
    known = known or event
    return {"s": "src", "c": "CH", "d": "load", "u": "MW", "v": value,
            "e": event.isoformat(), "t": event.isoformat(), "k": known.isoformat(),
            "r": known.isoformat(), "n": known.isoformat(), "ri": f"r{revision}", "rs": revision,
            "id": record or f"id-{index}"}


class DataAuditTests(unittest.TestCase):
    def test_reports_quality_dimensions(self):
        rows = [row(0), row(1), row(2), row(5, value=None),
                row(5, value=2, revision=1, known=T + timedelta(hours=5))]
        report = audit_dataset(make_dataset(rows), issue_time=T + timedelta(hours=2),
                               forecast_horizons=(timedelta(hours=1),))
        self.assertEqual(report.rows, 5)
        self.assertTrue(report.gaps["src"])
        self.assertGreater(report.missingness["src"]["missing_rows"], 0)
        self.assertGreater(report.revision_frequency["src"]["revision_rows"], 0)
        self.assertTrue(report.ready_for_forecast)

    def test_suspicious_future_availability_is_flagged(self):
        future = row(5, known=T)
        report = audit_dataset(make_dataset([row(0), future]), issue_time=T + timedelta(hours=1))
        self.assertEqual(report.suspicious_future_availability["src"], 1)
        self.assertTrue(report.warnings)

    def test_duplicate_and_unit_diagnostics(self):
        duplicate = row(0)
        report = audit_dataset(make_dataset([duplicate, duplicate]), issue_time=T)
        self.assertEqual(report.duplicates["duplicate_rows_after_first"], 1)
