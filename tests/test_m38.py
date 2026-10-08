from __future__ import annotations

import unittest

from hm_recsys.m38 import BUDGETS, IMAGE_SOURCE_FEATURES, VARIANTS, validate_protocol


class M38ProtocolTests(unittest.TestCase):
    def test_m38_protocol_and_factorial_budgets(self) -> None:
        validate_protocol()
        self.assertEqual(
            VARIANTS,
            (
                "base_historical_400",
                "base_soft_shallow_400",
                "base_historical_soft_500",
            ),
        )
        self.assertEqual(BUDGETS["base_300"], 300)
        self.assertEqual(BUDGETS[VARIANTS[0]], BUDGETS[VARIANTS[1]])
        self.assertEqual(BUDGETS[VARIANTS[1]], 400)
        self.assertEqual(BUDGETS[VARIANTS[2]], 500)

    def test_m38_source_features_are_unique_and_explain_routes(self) -> None:
        self.assertEqual(len(IMAGE_SOURCE_FEATURES), len(set(IMAGE_SOURCE_FEATURES)))
        self.assertIn("historical_present", IMAGE_SOURCE_FEATURES)
        self.assertIn("soft_personalized_rank", IMAGE_SOURCE_FEATURES)
        self.assertIn("soft_global_rank", IMAGE_SOURCE_FEATURES)
        self.assertIn("historical_soft_overlap", IMAGE_SOURCE_FEATURES)
