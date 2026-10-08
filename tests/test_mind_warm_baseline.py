from __future__ import annotations

import unittest

import pandas as pd

from hm_recsys.mind_warm_baseline import _replay_order, reconstruct_full_baseline


class MindWarmBaselineTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for user, events in (("active", 3), ("inactive", 0)):
            for rank in range(1, 51):
                rows.append({"customer_id": user, "article_id": f"{user}_{rank}",
                             "candidate_rank": rank, "rf": rank if events else 51-rank,
                             "ap_rf": rank, "user_history_events_12w": events,
                             "target": int(rank in (8, 13)), "truth_count": 2})
        self.ranks = pd.DataFrame(rows)
        self.swaps = pd.DataFrame([
            {"customer_id": "active", "challenger_article_id": "active_13", "victim_article_id": "active_8",
             "challenger_rank": 13, "victim_rank": 8},
            {"customer_id": "active", "challenger_article_id": "active_14", "victim_article_id": "active_12",
             "challenger_rank": 14, "victim_rank": 12}])

    def test_inactive_fallback_and_two_disjoint_swaps_replay(self):
        result = _replay_order(self.ranks, self.swaps)
        inactive = result[result.customer_id == "inactive"]
        self.assertEqual(inactive.champion_rank.tolist(), inactive.candidate_rank.tolist())
        self.assertTrue((inactive.model_rf != inactive.rf).all())
        lookup = result.set_index(["customer_id", "article_id"]).champion_rank
        self.assertEqual(lookup.loc[("active", "active_13")], 8)
        self.assertEqual(lookup.loc[("active", "active_8")], 13)
        self.assertEqual(lookup.loc[("active", "active_14")], 12)
        self.assertEqual(lookup.loc[("active", "active_12")], 14)
        self.assertTrue((result.loc[result.rf <= 7, "rf"] == result.loc[result.rf <= 7, "champion_rank"]).all())

    def test_inactive_swap_rejected(self):
        swaps = self.swaps.iloc[:1].copy()
        for name in ("customer_id", "challenger_article_id", "victim_article_id"):
            swaps[name] = swaps[name].str.replace("active", "inactive", regex=False)
        with self.assertRaisesRegex(ValueError, "inactive-fallback"):
            _replay_order(self.ranks, swaps)

    def test_saved_rank_mismatch_rejected(self):
        swaps = self.swaps.copy()
        swaps.loc[0, "victim_rank"] = 9
        with self.assertRaisesRegex(ValueError, "item/rank"):
            _replay_order(self.ranks, swaps)

    def test_missing_baseline_item_rejected(self):
        with self.assertRaisesRegex(ValueError, "fifty"):
            _replay_order(self.ranks.iloc[1:], self.swaps)

    def test_final_week_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "final week"):
            reconstruct_full_baseline("2020-09-16")

    def test_inner_cutoff_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, "full-cohort outer"):
            reconstruct_full_baseline("2019-12-25")


if __name__ == "__main__":
    unittest.main()
