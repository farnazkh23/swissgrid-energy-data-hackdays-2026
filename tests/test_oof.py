import unittest
from dataclasses import replace
from swissgrid_forecaster.oof import run_oof
from swissgrid_forecaster.champion import Persistence, HistoricalConditional, select_champion
from test_splits import samples, plan, H
from test_model_contracts import HASH


class OOFTests(unittest.TestCase):
    def result(self):
        return run_oof(plan(), samples(), [Persistence, HistoricalConditional], HASH)

    def test_integrity_and_determinism(self):
        result = self.result()
        self.assertEqual(result, self.result())
        self.assertEqual(len(result.records), 24)
        self.assertEqual(result.records[0].prediction_type, 'point')
        self.assertEqual(result.records[0].prediction_values, (7,))

    def test_tampered_provenance(self):
        result = self.result()
        record = result.records[0]
        for changes in [dict(fold_id='unknown'), dict(row_id='0'), dict(row_id='40'),
                        dict(feature_manifest_hash='b'*64), dict(training_data_manifest_hash='c'*64),
                        dict(training_cutoff=record.training_cutoff-H), dict(target_time=record.target_time+H)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(result, records=(replace(record, **changes), *result.records[1:]))
        with self.assertRaises(ValueError):
            replace(result, records=(*result.records, record))
        with self.assertRaises(ValueError):
            replace(record, training_cutoff=record.issue_time+H)

    def test_incomplete_candidate_rejected(self):
        result = self.result()
        incomplete = replace(result, records=result.records[1:])
        with self.assertRaises(ValueError):
            select_champion(incomplete, metric='mae')
        missing = replace(result, records=tuple(r for r in result.records if r.model_id == 'persistence'))
        with self.assertRaises(ValueError):
            select_champion(missing, metric='mae')

    def test_holdout_cannot_affect_predictions(self):
        rows = tuple(replace(r, target=-999) if r.issue_time >= plan().holdout_start else r for r in samples())
        self.assertEqual(self.result(), run_oof(plan(), rows, [Persistence, HistoricalConditional], HASH))

    def test_predict_cannot_read_truth(self):
        class Spy(Persistence):
            def predict(self, X, context):
                if any(r.target is not None or r.partition != 'validation' for r in context.rows):
                    raise AssertionError('truth or wrong partition exposed')
                if len(context.rows) != 1:
                    raise AssertionError('later issue features exposed in batch')
                return super().predict(X, context)
        run_oof(plan(), samples(), [Spy], HASH)

    def test_duplicate_ids_reused_instance_wrong_count(self):
        with self.assertRaises(ValueError):
            run_oof(plan(), samples(), [Persistence, Persistence], HASH)
        model = Persistence()
        with self.assertRaises(ValueError):
            run_oof(plan(), samples(), [lambda: model], HASH)
        class Bad(Persistence):
            def predict(self, X, context):
                return ()
        with self.assertRaises(ValueError):
            run_oof(plan(), samples(), [Bad], HASH)

    def test_corrupt_fold_rejected(self):
        result = self.result()
        f = result.folds[0]
        for changes in [dict(train=(*f.train, replace(f.validation[0], partition='train'))),
                        dict(holdout_start=f.holdout_start+H), dict(fold_id='made-up')]:
            with self.assertRaises(ValueError):
                replace(result, folds=(replace(f, **changes), *result.folds[1:]))

    def test_future_validation_feature_rejected(self):
        rows = samples()
        rows = (*rows[:10], replace(rows[10], feature_known_at=rows[10].issue_time+H), *rows[11:])
        with self.assertRaises(ValueError):
            run_oof(plan(), rows, [Persistence], HASH)

    def test_estimator_provenance_is_checked(self):
        class Bad(Persistence):
            def fit(self, X, y, context):
                super().fit(X, y, context)
                self.training_data_manifest_hash = 'b'*64
                return self
        with self.assertRaises(ValueError):
            run_oof(plan(), samples(), [Bad], HASH)
