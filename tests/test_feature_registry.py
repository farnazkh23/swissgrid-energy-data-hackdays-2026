import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from swissgrid_forecaster.feature_registry import FeatureDefinition, FeatureRegistry, FeatureAudit


def feature(key='x', parents=()):
    return FeatureDefinition(key, 'weather', key, ('CH',), 'configured', 'receipt',
                             (timedelta(hours=1),), 'identity', '1', parents,
                             'configured', 'review', 'propagate')


class FeatureRegistryTests(unittest.TestCase):
    def test_identity_and_audit_separation(self):
        f = feature()
        audit = FeatureAudit(f.version, 'evaluation', pearson=.5, spearman=.4,
                             mutual_information=.1, univariate_score=2, oof_gain=.3,
                             stability=.9, keep_drop='KEEP')
        self.assertEqual(audit.feature_version, f.version)
        for field, value in [('units', 'other'), ('transform_version', '2'), ('enabled', False),
                             ('known_at_rule', 'later'), ('missingness_policy', 'error')]:
            self.assertNotEqual(f.version, replace(f, **{field: value}).version)

    def test_duplicate_unknown_cycle(self):
        for definitions in [(feature(), feature()), (feature(parents=('unknown',)),),
                            (feature('a', ('b',)), feature('b', ('a',))), (feature('a', ('a',)),)]:
            with self.subTest(definitions=definitions), self.assertRaises(ValueError):
                FeatureRegistry(definitions)

    def test_dag_and_order_independent_registry(self):
        a, b, c = feature('a'), feature('b', ('a',)), feature('c', ('a', 'b'))
        registry = FeatureRegistry((c, b, a))
        self.assertEqual(registry.topological_order(), ('a', 'b', 'c'))
        self.assertEqual(registry.manifest_hash, FeatureRegistry((a, b, c)).manifest_hash)
        t = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(registry.known_at('c', {'a': t+timedelta(hours=2), 'b': t, 'c': t}), t+timedelta(hours=2))
        with self.assertRaises(ValueError):
            registry.known_at('c', {'c': t})

    def test_invalid_metadata(self):
        for change in [dict(horizon_available=()), dict(missingness_policy='fill_global'),
                       dict(enabled=1), dict(parent_features=('a', 'a'))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(feature(), **change)
        with self.assertRaises(ValueError):
            FeatureAudit('v', 'e', pearson=float('nan'))
