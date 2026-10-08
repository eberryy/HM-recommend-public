from __future__ import annotations

import unittest

from hm_recsys.m54 import LINEAGES, NO_DECAY_TARGET_FEATURES, validate_lineage


class M54Tests(unittest.TestCase):
    def test_all_registered_lineages_are_forward_safe(self) -> None:
        for cutoff, lineage in LINEAGES.items():
            result = validate_lineage(cutoff, lineage)
            self.assertTrue(result["safe"])

    def test_first_cutoff_is_explicitly_missing_not_self_fit(self) -> None:
        result = validate_lineage("2019-11-27", None)
        self.assertFalse(result["available"])
        self.assertIn("missing", result["reason"])

    def test_no_decay_contract_excludes_decay_columns(self) -> None:
        self.assertTrue(NO_DECAY_TARGET_FEATURES)
        self.assertFalse(any("decay" in name for name in NO_DECAY_TARGET_FEATURES))
