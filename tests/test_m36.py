from __future__ import annotations

import unittest

import pandas as pd

from hm_recsys.m34 import OUTER_WINDOWS
from hm_recsys.m36 import CONFIG, _rank_bucket, classify_loss_reason, validate_protocol


class M36ProtocolTests(unittest.TestCase):
    def test_protocol_is_diagnostic_and_excludes_final_week(self) -> None:
        validate_protocol()
        self.assertEqual(CONFIG["diagnostic_neighbor_depth"], 500)
        self.assertEqual(CONFIG["focus_window"], "spring_20200318")
        self.assertTrue(all(cutoff < "2020-09-16" for cutoff in OUTER_WINDOWS.values()))

    def test_visual_rank_buckets_are_fixed(self) -> None:
        self.assertEqual(_rank_bucket(90, None), "le_100")
        self.assertEqual(_rank_bucket(150, 240), "101_200")
        self.assertEqual(_rank_bucket(450, None), "201_500")
        self.assertEqual(_rank_bucket(None, None), ">500_or_unreached")

    def test_loss_reason_separates_hard_gate_depth_and_fusion(self) -> None:
        base = {
            "image_selected_hit": False,
            "image_source_hit": False,
            "image_covered": True,
            "personalized_visual_min_rank": 80,
            "global_visual_min_rank": None,
            "cold_scored_eligible": False,
            "user_has_seed": True,
        }
        self.assertEqual(classify_loss_reason(pd.Series(base)), "seasonal_hard_gate")
        deep = dict(base, personalized_visual_min_rank=180, cold_scored_eligible=True)
        self.assertEqual(classify_loss_reason(pd.Series(deep)), "neighbor_depth_101_500")
        selected = dict(base, image_selected_hit=True, image_source_hit=True)
        self.assertEqual(classify_loss_reason(pd.Series(selected)), "selected_top50")
        fusion = dict(base, image_source_hit=True, cold_scored_eligible=True)
        self.assertEqual(classify_loss_reason(pd.Series(fusion)), "top50_fusion_pruning")


if __name__ == "__main__":
    unittest.main()
