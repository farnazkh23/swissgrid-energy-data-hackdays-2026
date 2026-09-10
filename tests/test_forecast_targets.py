"""Regression tests: the scored forecast-output target contract is AT/DE/FR/IT.

These tests exist to prevent the historical mistake of modelling CH as one of
the four forecast outputs from ever returning. CH is an input/context country
only. The canonical order is positional because the official evaluator maps
realizations positionally (target_0->AT, target_1->DE, target_2->FR,
target_3->IT).
"""
import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from swissgrid_forecaster.champion import HistoricalConditional, Persistence, select_champion
from swissgrid_forecaster.oof import run_oof
from swissgrid_forecaster.oof_handoff import (OOF_HANDOFF_COLUMNS, RESIDUAL_VECTOR_ORDER,
                                              build_oof_handoff, oof_handoff_column_order,
                                              residual_summary, write_oof_handoff)
from swissgrid_forecaster.pipeline import PipelineConfig, run_mock_pipeline
from swissgrid_forecaster.submission import (SAMPLES_PER_TARGET, SUBMISSION_COLUMNS, SUBMISSION_ROWS,
                                             build_submission_table, submission_target_mapping,
                                             validate_submission_table)
from swissgrid_forecaster.target_contract import (InvalidOutputTargets, OUTPUT_TARGETS, TARGETS,
                                                  TARGET_POSITIONS, require_output_target,
                                                  submission_position, validate_output_targets)
from test_model_contracts import HASH
from test_splits import H, plan, samples

ISSUE = datetime(2026, 1, 10, tzinfo=timezone.utc)


class CanonicalTargetContractTests(unittest.TestCase):
    def test_targets_are_exactly_at_de_fr_it_in_order(self):
        self.assertEqual(TARGETS, ("AT", "DE", "FR", "IT"))
        self.assertIs(OUTPUT_TARGETS, TARGETS)

    def test_ch_is_not_a_forecast_output_and_at_is_required(self):
        self.assertNotIn("CH", TARGETS)
        self.assertIn("AT", TARGETS)
        self.assertEqual(TARGET_POSITIONS, {"AT": 0, "DE": 1, "FR": 2, "IT": 3})

    def test_validate_output_targets_accepts_only_canonical_order(self):
        self.assertEqual(validate_output_targets(TARGETS), TARGETS)
        self.assertEqual(validate_output_targets(list(TARGETS)), TARGETS)
        for bad in [("CH", "DE", "FR", "IT"), ("AT", "CH", "FR", "IT"),
                    ("DE", "FR", "IT"), ("AT", "DE", "FR"),
                    ("DE", "AT", "FR", "IT"), ("AT", "DE", "FR", "IT", "CH"),
                    ("AT", "AT", "FR", "IT"), ()]:
            with self.subTest(bad=bad), self.assertRaises(InvalidOutputTargets):
                validate_output_targets(bad)
        with self.assertRaises(InvalidOutputTargets):
            validate_output_targets("AT")

    def test_require_output_target_rejects_ch(self):
        self.assertEqual(require_output_target("AT"), "AT")
        for entity in ("CH", "", "ch", "XX"):
            with self.subTest(entity=entity), self.assertRaises(ValueError):
                require_output_target(entity)

    def test_submission_positions_are_stable(self):
        self.assertEqual(submission_position("AT"), 0)
        self.assertEqual(submission_position("DE"), 1)
        self.assertEqual(submission_position("FR"), 2)
        self.assertEqual(submission_position("IT"), 3)
        self.assertEqual(submission_target_mapping(),
                         {"target_0": "AT", "target_1": "DE", "target_2": "FR", "target_3": "IT"})

    def test_mock_pipeline_forecasts_a_scored_target_not_ch(self):
        result = run_mock_pipeline(PipelineConfig(ISSUE, seed=17, include_ridge=False))
        self.assertEqual(result.forecast.target_entity, "AT")
        self.assertNotEqual(result.forecast.target_entity, "CH")
        require_output_target(result.forecast.target_entity)


class SubmissionWriterTests(unittest.TestCase):
    def stamps(self):
        return tuple(ISSUE + timedelta(hours=index) for index in range(SUBMISSION_ROWS))

    def series(self, entity, offset=0.0):
        return tuple(tuple(offset + index * 300 + sample for sample in range(SAMPLES_PER_TARGET))
                     for index in range(SUBMISSION_ROWS))

    def table(self, **changes):
        series = {entity: self.series(entity, index * 1000.0)
                  for index, entity in enumerate(TARGETS)}
        series.update(changes)
        return build_submission_table(series, self.stamps())

    def test_submission_writer_preserves_target_order(self):
        rows = self.table()
        self.assertEqual(len(rows), SUBMISSION_ROWS)
        for index, row in enumerate(rows):
            self.assertEqual(tuple(row), SUBMISSION_COLUMNS)
            # positional mapping: target_0 holds AT samples, target_3 holds IT
            self.assertEqual(row["target_0"][0], index * 300)
            self.assertEqual(row["target_3"][0], 3000 + index * 300)
            self.assertEqual(len(row["target_1"]), SAMPLES_PER_TARGET)
        self.assertEqual(validate_submission_table(rows), submission_target_mapping())

    def test_submission_writer_rejects_ch_as_output(self):
        with self.assertRaises(InvalidOutputTargets):
            self.table(**{"CH": self.series("CH")})
        series = {entity: self.series(entity) for entity in TARGETS}
        series["CH"] = series["AT"]
        with self.assertRaises(InvalidOutputTargets):
            build_submission_table(series, self.stamps())

    def test_submission_writer_requires_all_four_targets(self):
        series = {entity: self.series(entity) for entity in TARGETS}
        del series["AT"]
        with self.assertRaises(InvalidOutputTargets):
            build_submission_table(series, self.stamps())

    def test_submission_writer_enforces_168_times_300_integers(self):
        series = {entity: self.series(entity) for entity in TARGETS}
        with self.assertRaises(ValueError):
            build_submission_table(series, self.stamps()[:-1])
        with self.assertRaises(ValueError):
            build_submission_table(series, self.stamps()[1:])
        short = {entity: tuple(hour[:-1] for hour in self.series(entity)) for entity in TARGETS}
        with self.assertRaises(ValueError):
            build_submission_table(short, self.stamps())
        nulled = {entity: self.series(entity) for entity in TARGETS}
        nulled["FR"] = tuple((None,) * SAMPLES_PER_TARGET if index == 0 else hour
                             for index, hour in enumerate(self.series("FR")))
        with self.assertRaises(ValueError):
            build_submission_table(nulled, self.stamps())
        boolean = {entity: self.series(entity) for entity in TARGETS}
        boolean["IT"] = tuple((True,) * SAMPLES_PER_TARGET if index == 0 else hour
                              for index, hour in enumerate(self.series("IT")))
        with self.assertRaises(ValueError):
            build_submission_table(boolean, self.stamps())

    def test_validate_submission_table_detects_shape_corruption(self):
        rows = self.table()
        with self.assertRaises(ValueError):
            validate_submission_table(rows[:167])
        with self.assertRaises(ValueError):
            validate_submission_table(tuple({**row, "target_2": row["target_2"][:-1]} for row in rows))
        with self.assertRaises(ValueError):
            validate_submission_table(tuple({k: v for k, v in row.items() if k != "target_3"}
                                            for row in rows))


class OOFHandoffTests(unittest.TestCase):
    def oof(self, offset=0.0, count=50):
        rows = tuple(replace(row, target=row.target + offset) for row in samples(count))
        return run_oof(plan(), rows, [Persistence, HistoricalConditional], HASH)

    def handoff(self):
        oof_by_target = {entity: self.oof(index * 10.0) for index, entity in enumerate(TARGETS)}
        champions = {}
        for entity in TARGETS:
            champions[entity], _ = select_champion(oof_by_target[entity], metric="mae")
        return build_oof_handoff(oof_by_target, champions), champions

    def test_handoff_columns_are_exactly_the_at_de_fr_it_contract(self):
        self.assertEqual(OOF_HANDOFF_COLUMNS,
                         ("timestamp", "fold_id", "horizon", "issue_time",
                          "AT_actual", "AT_pred", "AT_residual",
                          "DE_actual", "DE_pred", "DE_residual",
                          "FR_actual", "FR_pred", "FR_residual",
                          "IT_actual", "IT_pred", "IT_residual"))
        self.assertEqual(RESIDUAL_VECTOR_ORDER, ("AT", "DE", "FR", "IT"))
        rows, _ = self.handoff()
        self.assertTrue(rows)
        self.assertEqual(oof_handoff_column_order(rows), OOF_HANDOFF_COLUMNS)

    def test_handoff_uses_true_oof_predictions_and_residual_definition(self):
        rows, champions = self.handoff()
        expected_horizon = int((samples()[0].target_time - samples()[0].issue_time).total_seconds())
        for row in rows:
            for entity in TARGETS:
                self.assertAlmostEqual(row[f"{entity}_residual"],
                                       row[f"{entity}_actual"] - row[f"{entity}_pred"], places=12)
            self.assertEqual(row["horizon"], expected_horizon)
        # per-target champions may differ; each must be a real OOF candidate
        for entity in TARGETS:
            self.assertIn(champions[entity], self.oof().candidate_identities)

    def test_handoff_rejects_ch_and_missing_targets(self):
        rows, champions = self.handoff()
        oof_by_target = {entity: self.oof() for entity in TARGETS}
        with_ch = dict(oof_by_target, **{"CH": self.oof()})
        with self.assertRaises(InvalidOutputTargets):
            build_oof_handoff(with_ch, champions)
        with_ch_champions = dict(champions, **{"CH": champions["AT"]})
        with self.assertRaises(InvalidOutputTargets):
            build_oof_handoff(oof_by_target, with_ch_champions)
        incomplete = {entity: self.oof() for entity in TARGETS if entity != "AT"}
        with self.assertRaises(InvalidOutputTargets):
            build_oof_handoff(incomplete, {k: v for k, v in champions.items() if k != "AT"})

    def test_handoff_requires_aligned_folds_across_targets(self):
        oof_by_target = {entity: self.oof() for entity in TARGETS}
        champions = {}
        for entity in TARGETS:
            champions[entity], _ = select_champion(oof_by_target[entity], metric="mae")
        renamed = tuple(replace(row, row_id="it-" + row.row_id) for row in samples())
        misaligned = dict(oof_by_target, **{"IT": run_oof(plan(), renamed, [Persistence], HASH)})
        with self.assertRaises(ValueError):
            build_oof_handoff(misaligned, champions)

    def test_written_artifacts_preserve_target_order(self):
        rows, champions = self.handoff()
        with tempfile.TemporaryDirectory() as directory:
            predictions_path, summary_path = write_oof_handoff(rows, champions, directory)
            with predictions_path.open(newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
            self.assertEqual(tuple(header), OOF_HANDOFF_COLUMNS)
            summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["target_order"], ["AT", "DE", "FR", "IT"])
        self.assertEqual(summary["residual_vector_order"], ["AT", "DE", "FR", "IT"])
        self.assertEqual(set(summary["per_target"]), set(TARGETS))
        self.assertNotIn("CH", summary["per_target"])
        self.assertEqual(summary["champions"]["AT"]["model_id"], champions["AT"][0])
        for entity in TARGETS:
            stats = summary["per_target"][entity]
            self.assertEqual(stats["count"], len(rows))
            self.assertAlmostEqual(stats["bias"], -stats["residual_mean"], places=9)
            self.assertGreaterEqual(stats["mae"], 0.0)

    def test_residual_summary_rejects_ch(self):
        rows, champions = self.handoff()
        with self.assertRaises(InvalidOutputTargets):
            residual_summary(rows, dict(champions, **{"CH": champions["AT"]}))


if __name__ == "__main__":
    unittest.main()
