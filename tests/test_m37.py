from __future__ import annotations

import unittest

from hm_recsys.m34 import OUTER_WINDOWS
from hm_recsys.m37 import CONFIG, SOURCE_VARIANTS, VARIANT_BUDGETS, validate_protocol


class M37ProtocolTests(unittest.TestCase):
    def test_protocol_is_development_only_and_factorial(self) -> None:
        validate_protocol()
        self.assertTrue(all(cutoff < "2020-09-16" for cutoff in OUTER_WINDOWS.values()))
        self.assertEqual(CONFIG["shallow_neighbor_depth"], 100)
        self.assertEqual(CONFIG["deep_neighbor_depth"], 500)
        self.assertEqual(
            set(SOURCE_VARIANTS),
            {
                "hard_shallow_split",
                "soft_shallow_split",
                "hard_deep_split",
                "soft_deep_split",
            },
        )

    def test_split_and_candidate_budgets_are_frozen(self) -> None:
        self.assertEqual(CONFIG["personalized_route_k"], 50)
        self.assertEqual(CONFIG["global_route_k"], 50)
        self.assertEqual(CONFIG["season_rrf_weight"], 0.5)
        self.assertEqual(VARIANT_BUDGETS["fixed_soft_deep_split_300"], 300)
        self.assertEqual(VARIANT_BUDGETS["expanded_soft_deep_split_400"], 400)


if __name__ == "__main__":
    unittest.main()
