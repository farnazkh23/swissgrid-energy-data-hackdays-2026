import unittest
from datetime import timedelta
from dataclasses import replace

from swissgrid_forecaster.specialist_contracts import EVIDENCE_BRIEF, TARGET_FORECAST_BRIEF
from swissgrid_forecaster.specialist_registry import (
    SPECIALIST_FAMILIES,
    SpecialistDefinition,
    SpecialistRegistry,
)


H = timedelta(hours=1)


def definition(specialist_id="demand-1", family="Demand"):
    return SpecialistDefinition(specialist_id, family, "v1", (EVIDENCE_BRIEF, TARGET_FORECAST_BRIEF), (H,))


class SpecialistRegistryTests(unittest.TestCase):
    def test_all_future_families_are_representable(self):
        registry = SpecialistRegistry(definition(f"s-{i}", family) for i, family in enumerate(SPECIALIST_FAMILIES))
        self.assertEqual(tuple(entry.family for entry in registry.entries), SPECIALIST_FAMILIES)

    def test_duplicate_ids_include_alternate_versions(self):
        registry = SpecialistRegistry()
        registry.register(definition())
        with self.assertRaises(ValueError):
            registry.register(replace(definition(), version="v2"))

    def test_registry_has_no_weight_or_promotion_surface(self):
        entry = definition()
        self.assertFalse(hasattr(entry, "ensemble_weight"))
        self.assertEqual(registry := SpecialistRegistry([entry]).get("demand-1"), entry)
        self.assertEqual(registry, SpecialistRegistry([entry]).by_family("Demand")[0])

    def test_invalid_metadata_fails(self):
        for change in (dict(family="Unknown"), dict(output_types=()), dict(horizon_applicability=())):
            with self.assertRaises(ValueError):
                replace(definition(), **change)

    def test_unknown_lookup_and_bad_type_fail(self):
        with self.assertRaises(ValueError):
            SpecialistRegistry().get("missing")
        with self.assertRaises(ValueError):
            SpecialistRegistry().register("not-a-definition")
