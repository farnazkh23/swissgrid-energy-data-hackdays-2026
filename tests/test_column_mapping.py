import unittest
from swissgrid_forecaster.column_mapping import ColumnMapping


class ColumnMappingTests(unittest.TestCase):
    def test_arbitrary_columns_and_constants(self):
        mapping = ColumnMapping.from_dict({"columns": {"value": "MW_VALUE", "event_time": "WHEN"},
                                           "constants": {"unit": "MW", "country": "CH"}})
        self.assertEqual(mapping.value({"MW_VALUE": 3, "WHEN": "x"}, "value"), 3)
        self.assertEqual(mapping.value({}, "unit"), "MW")
        self.assertEqual(mapping.to_dict()["columns"]["event_time"], "WHEN")

    def test_missing_and_duplicate_mapping_fails(self):
        with self.assertRaises(ValueError):
            ColumnMapping.from_dict({"columns": {"value": "x", "event_time": "x"}})
        mapping = ColumnMapping.from_dict({"columns": {"value": "x"}})
        with self.assertRaises(ValueError):
            mapping.value({}, "value")
        with self.assertRaises(ValueError):
            mapping.require("known_at")

    def test_unknown_canonical_field_fails(self):
        with self.assertRaises(ValueError):
            ColumnMapping.from_dict({"columns": {"target_meaning": "x"}})
