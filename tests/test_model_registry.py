import unittest
from dataclasses import replace
from swissgrid_forecaster.model_registry import ModelEntry, ModelRegistry, ModelState
from test_splits import T, H
from test_model_contracts import HASH


class RegistryTests(unittest.TestCase):
    def entry(self):
        return ModelEntry('persistence', 'v1', T, HASH, 'b'*64)

    def test_duplicates_including_versions(self):
        registry = ModelRegistry()
        registry.register(self.entry())
        for entry in (self.entry(), replace(self.entry(), version='v2')):
            with self.assertRaises(ValueError):
                registry.register(entry)

    def test_lifecycle_no_promotion(self):
        registry = ModelRegistry()
        registry.register(self.entry())
        with self.assertRaises(NotImplementedError):
            registry.transition('persistence', ModelState.CHAMPION)
        self.assertEqual(registry.entries[0].state, ModelState.CANDIDATE)
        registry.transition('persistence', ModelState.FROZEN)
        with self.assertRaises(ValueError):
            registry.transition('persistence', ModelState.REJECTED)
        other = ModelRegistry()
        other.register(self.entry())
        other.transition('persistence', ModelState.REJECTED)
        self.assertEqual(other.entries[0].state, ModelState.REJECTED)
        with self.assertRaises(ValueError):
            ModelRegistry().register(replace(self.entry(), state=ModelState.CHAMPION))

    def test_manifest_identity(self):
        e = self.entry()
        for changes in [dict(version='v2'), dict(fit_cutoff=T+H), dict(feature_manifest_hash='c'*64),
                        dict(training_data_manifest_hash='d'*64)]:
            self.assertNotEqual(e.manifest_hash, replace(e, **changes).manifest_hash)
        self.assertEqual(e.manifest_hash, replace(e, state=ModelState.FROZEN).manifest_hash)
        with self.assertRaises(ValueError):
            replace(e, feature_manifest_hash='invalid')
