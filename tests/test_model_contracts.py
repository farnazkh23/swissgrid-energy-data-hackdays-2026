import unittest
from dataclasses import replace
from swissgrid_forecaster.model_contracts import FitContext, PredictContext, Prediction
from swissgrid_forecaster.champion import Persistence
from swissgrid_forecaster.splits import data_manifest
from test_splits import samples, T, H

HASH = 'a'*64


def fit_context(rows=None, cutoff=T+5*H):
    rows = samples(5) if rows is None else tuple(rows)
    return FitContext(rows, cutoff, HASH, data_manifest(rows))


def fit(model, context=None):
    context = context or fit_context()
    return model.fit([r.features for r in context.rows], [r.target for r in context.rows], context)


class ContractTests(unittest.TestCase):
    def test_metadata_and_fit_cutoff(self):
        model = fit(Persistence())
        self.assertEqual(model.fit_cutoff, T+5*H)
        self.assertEqual(model.feature_manifest_hash, HASH)
        self.assertEqual(model.training_data_manifest_hash, data_manifest(samples(5)))
        r = samples(7)[6]
        self.assertEqual(model.predict([r.features], PredictContext((r,), HASH))[0].values, (4,))
        r = samples(3)[2]
        with self.assertRaises(ValueError):
            model.predict([r.features], PredictContext((r,), HASH))

    def test_fit_partition_and_knowledge_guards(self):
        for partition in ('validation', 'calibration', 'holdout'):
            with self.assertRaises(ValueError):
                fit_context([replace(samples()[0], partition=partition)])
        for change in [dict(label_known_at=T+6*H), dict(feature_known_at=T+H)]:
            with self.assertRaises(ValueError):
                fit_context([replace(samples()[0], **change)])

    def test_manifest_identity(self):
        ctx = fit_context()
        with self.assertRaises(ValueError):
            replace(ctx, rows=(replace(ctx.rows[0], target=999), *ctx.rows[1:]))
        with self.assertRaises(ValueError):
            replace(ctx, feature_manifest_hash='bad')
        model = fit(Persistence())
        r = samples(7)[6]
        with self.assertRaises(ValueError):
            model.predict([r.features], PredictContext((r,), 'b'*64))
        self.assertNotEqual(data_manifest(ctx.rows), data_manifest((replace(ctx.rows[0], features=(8,)), *ctx.rows[1:])))

    def test_array_alignment_failed_refit(self):
        model = fit(Persistence())
        with self.assertRaises(ValueError):
            model.fit([[999]], [0], fit_context())
        self.assertIsNone(model.fit_cutoff)
        with self.assertRaises(ValueError):
            PredictContext((replace(samples()[0], feature_known_at=T+H),), HASH)

    def test_prediction_validation(self):
        for args in [('point', (1, 2)), ('quantile', (2, 1), (.1, .9)),
                     ('quantile', (1,), (.0,)), ('probability', (1.1,)),
                     ('distribution', (float('nan'),)), ('distribution', ())]:
            with self.assertRaises(ValueError):
                Prediction(*args)
