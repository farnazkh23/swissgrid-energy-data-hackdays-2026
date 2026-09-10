import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from swissgrid_forecaster.edh_scoring import PERFECT_SCORE
from swissgrid_forecaster.oof_handoff import write_oof_handoff
from swissgrid_forecaster.real_modeling_v1 import make_weekly_folds
from swissgrid_forecaster.real_uncertainty_backtest import (
    METHODS, compare_methods, fit_calibrated_champion, load_actuals,
    load_point_forecasts, load_residual_panel, sample_folds,
    score_samples_against_actuals,
)
from swissgrid_forecaster.target_contract import TARGETS, validate_output_targets
from swissgrid_forecaster.uncertainty_model import ResidualObservation, ResidualPanel


T0 = datetime(2019, 1, 1, tzinfo=timezone.utc)
HANDOFF_COLUMNS = (
    "timestamp", "fold_id", "horizon", "issue_time",
    "AT_actual", "AT_pred", "AT_residual",
    "DE_actual", "DE_pred", "DE_residual",
    "FR_actual", "FR_pred", "FR_residual",
    "IT_actual", "IT_pred", "IT_residual",
)


def synthetic_oof_rows(folds=3):
    timestamps = tuple(T0 + timedelta(hours=index)
                       for index in range(672 + 168 * folds))
    weekly = make_weekly_folds(timestamps, max_folds=folds)
    rows = []
    for fold_index, fold in enumerate(weekly):
        for hour_index, target_time in enumerate(fold.forecast_timestamps):
            issue_time = target_time - timedelta(hours=1)
            row = {"timestamp": target_time.isoformat(), "fold_id": fold.fold_id,
                   "horizon": 1, "issue_time": issue_time.isoformat()}
            for target_index, target in enumerate(TARGETS):
                actual = float(1000 * target_index + fold_index * 10 + hour_index)
                residual = float(((hour_index + fold_index) % 17) - 8 + target_index)
                row[f"{target}_actual"] = actual
                row[f"{target}_pred"] = actual - residual
                row[f"{target}_residual"] = residual
            rows.append(row)
    return tuple(rows), weekly


class CorrectedContractPathTests(unittest.TestCase):
    def test_three_fold_end_to_end_oof_uncertainty_and_score(self):
        rows, folds = synthetic_oof_rows(3)
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(len(forecast.forecast_timestamps) == 168 for forecast in folds))
        with tempfile.TemporaryDirectory() as directory:
            write_oof_handoff(rows, {target: ("ridge", "v1") for target in TARGETS}, directory)
            path = Path(directory) / "oof_predictions.csv"
            with path.open(newline="") as stream:
                self.assertEqual(tuple(csv.reader(stream).__next__()), HANDOFF_COLUMNS)
            panel = load_residual_panel(path)
            self.assertEqual(panel.target_names, TARGETS)
            self.assertEqual(len(panel.rows), 3 * 168)
            ranked = compare_methods(panel, seed=7, n_samples=8, methods=METHODS)
            self.assertEqual({row.method for row in ranked}, set(METHODS))
            model, demo = fit_calibrated_champion(
                panel, ranked[0].method, fit_weeks=1, calibration_weeks=1,
                seed=7, n_samples=8,
                **METHODS[ranked[0].method],
            )
            self.assertEqual(len(demo), 1)
            self.assertGreater(demo[0][0][0], panel.rows[167][0])
            samples = sample_folds(model, demo, load_point_forecasts(path), seed=7, n_samples=300)
            self.assertEqual(len(samples), 168)
            self.assertTrue(all(len(entry[f"{target}_samples"]) == 300 for entry in samples for target in TARGETS))
            self.assertTrue(all(isinstance(value, int)
                                for entry in samples for target in TARGETS
                                for value in entry[f"{target}_samples"]))
            score = score_samples_against_actuals(samples, load_actuals(path))
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, PERFECT_SCORE)

    def test_bad_chronology_duplicate_and_seconds_horizon_are_rejected(self):
        bad_equal = ((T0, T0, timedelta(hours=1), (0.0, 0.0, 0.0, 0.0)),)
        bad_negative = ((T0, T0 - timedelta(hours=1), timedelta(hours=-1), (0.0, 0.0, 0.0, 0.0)),)
        duplicate = ((T0, T0 + timedelta(hours=1), timedelta(hours=1), (0.0, 0.0, 0.0, 0.0)),) * 2
        for rows in (bad_equal, bad_negative, duplicate):
            with self.assertRaises(ValueError):
                ResidualPanel(TARGETS, rows)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", newline="", delete=False) as stream:
            writer = csv.DictWriter(stream, fieldnames=HANDOFF_COLUMNS)
            writer.writeheader()
            row = {"timestamp": (T0 + timedelta(hours=1)).isoformat(),
                   "issue_time": T0.isoformat(), "fold_id": "fold", "horizon": 3600}
            for target in TARGETS:
                row.update({f"{target}_actual": 1, f"{target}_pred": 0, f"{target}_residual": 1})
            writer.writerow(row)
            path = stream.name
        try:
            with self.assertRaises(ValueError):
                load_residual_panel(path)
        finally:
            Path(path).unlink()

    def test_cross_target_misalignment_insufficient_folds_and_target_order(self):
        observation = lambda target, offset=0: ResidualObservation(
            "fold", "row", target, T0, T0 + timedelta(hours=1 + offset),
            timedelta(hours=1 + offset), 1.0)
        per_target = {target: (observation(target),) for target in TARGETS}
        per_target["IT"] = (observation("IT", 1),)
        with self.assertRaises(ValueError):
            ResidualPanel.from_target_residuals(per_target)
        rows, _ = synthetic_oof_rows(1)
        panel = ResidualPanel(TARGETS, tuple(
            (datetime.fromisoformat(row["issue_time"]), datetime.fromisoformat(row["timestamp"]),
             timedelta(hours=1), tuple(row[f"{target}_residual"] for target in TARGETS))
            for row in rows))
        with self.assertRaises(ValueError):
            compare_methods(panel, seed=1, n_samples=4, methods=METHODS)
        with self.assertRaises(ValueError):
            validate_output_targets(("DE", "AT", "FR", "IT"))

    def test_generation_leakage_contract_is_explicit(self):
        source = (Path(__file__).resolve().parents[1] /
                  "databricks" / "run_corrected_real_backtest.py").read_text()
        self.assertIn('GENERATION_ALLOWED_FIELD = "generation_forecast"', source)
        self.assertIn('"actual_generation", "scheduled_consumption"', source)
        self.assertIn("assert not selected_names & GENERATION_FORBIDDEN_FIELDS", source)


if __name__ == "__main__":
    unittest.main()
