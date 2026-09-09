import csv
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from swissgrid_forecaster.column_mapping import ColumnMapping
from dataclasses import replace
from swissgrid_forecaster.real_dataset_loader import load_real_dataset
from swissgrid_forecaster.real_source_adapter import PITMetadataError, RealSourceConfig
from swissgrid_forecaster.run_real_champion import (RealChampionConfig, RealChampionRefusal,
                                                    run_real_champion)
from swissgrid_forecaster.splits import RollingOrigin
from swissgrid_forecaster.target_contract import ForecastType, TargetContract


T = datetime(2026, 1, 1, tzinfo=timezone.utc)


def source_config(policy="require"):
    mapping = ColumnMapping.from_dict({"columns": {"source_id": "s", "country": "c", "domain": "d", "unit": "u",
        "value": "v", "event_time": "e", "valid_time": "t", "known_at": "k", "first_received_at": "r",
        "normalized_at": "n", "revision_id": "ri", "revision_sequence": "rs", "source_record_id": "id"}})
    return RealSourceConfig("champion", mapping, "UTC", timedelta(hours=1), timedelta(0), timedelta(0), timedelta(days=2), policy)


def build_file(directory, *, pit=True):
    path = Path(directory) / "real.csv"
    fields = ["s", "c", "d", "u", "v", "e", "t", "k", "r", "n", "ri", "rs", "id"]
    rows = []
    for index in range(22):
        event = T + timedelta(hours=index)
        for source, domain, value in (("load", "load", 10 + index), ("net", "net_position", index - 10)):
            rows.append({"s": source, "c": "CH", "d": domain, "u": "MW", "v": value,
                         "e": event.isoformat(), "t": event.isoformat(),
                         "k": event.isoformat() if pit else "", "r": event.isoformat() if pit else "",
                         "n": event.isoformat() if pit else "", "ri": "r0", "rs": 0,
                         "id": f"{source}-{index}"})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return path


def champion_config():
    final_issue = T + timedelta(hours=19)
    target = TargetContract("explicit-real-target", "CH", "MW", "configured sign", timedelta(hours=1),
                            final_issue, final_issue + timedelta(hours=1), timedelta(hours=1),
                            "identity", "known_at <= target valid time", ForecastType.POINT, ("mae",), country="CH")
    plan = RollingOrigin(T + timedelta(hours=8), final_issue, timedelta(hours=8), timedelta(hours=3),
                         timedelta(hours=1), timedelta(hours=5), purge=timedelta(hours=1), embargo=timedelta(hours=1))
    return RealChampionConfig(target, "net", "net-*", tuple(T + timedelta(hours=i) for i in range(1, 20)),
                              plan, ("load",), include_ridge=False)


class RealChampionTests(unittest.TestCase):
    def test_audit_then_oof_scoreboard(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_real_dataset(build_file(directory), source_config=source_config())
            output = Path(directory) / "artifacts"
            result = run_real_champion(dataset, champion_config(), output_dir=output)
            self.assertTrue(result.data_audit.ready_for_forecast)
            self.assertTrue(result.baselines.oof.records)
            self.assertIn("persistence", result.baselines.scoreboard)
            self.assertFalse(result.diagnostics["holdout_used"])
            self.assertEqual({path.name for path in output.iterdir()},
                             {"data_audit.json", "feature_audit.json", "scoreboard.json", "diagnostics.json"})

    def test_unresolved_pit_refuses_before_model_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PITMetadataError):
                load_real_dataset(build_file(directory, pit=False), source_config=source_config())

    def test_non_pit_dataset_refuses_at_runner_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_real_dataset(build_file(directory), source_config=source_config())
            with self.assertRaises(RealChampionRefusal):
                run_real_champion(replace(dataset, pit_ready=False), champion_config())

    def test_target_contract_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_real_dataset(build_file(directory), source_config=source_config())
            bad = champion_config()
            with self.assertRaises(RealChampionRefusal):
                run_real_champion(dataset, RealChampionConfig(bad.target_contract, "unknown", "net-*",
                                                               bad.issue_times, bad.plan, ("load",), False))
