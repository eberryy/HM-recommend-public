import unittest

import pandas as pd

from hm_recsys.warm_v3_target_aware_admission import TARGET_FEATURES, augment_pairs


class TargetAwareAdmissionTests(unittest.TestCase):
    def test_augments_both_items_and_signed_difference(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u"],
                "challenger_article_id": ["c"],
                "victim_article_id": ["v"],
                **{name: [0.0] for name in []},
            }
        )
        for name in (
            "log_history_events", "log_unique_items", "recency_days", "victim_rank",
            "challenger_rank", "baseline_gap", "candidate_rank_advantage", "e0_rank_advantage",
            "e1_rank_advantage", "base_score_delta", "bpr_score_delta", "latent_score_delta",
            "rrf_score_delta", "challenger_candidate_rank", "victim_candidate_rank",
            "challenger_e0_rank", "victim_e0_rank", "challenger_e1_rank", "victim_e1_rank",
            "challenger_bpr_missing", "victim_bpr_missing", "expert_preference_count",
            "rank_disagreement_delta",
        ):
            pairs[name] = 0.0
        target = pd.DataFrame(
            {
                "customer_id": ["u", "u"],
                "article_id": ["c", "v"],
                **{name: [2.0, 0.5] for name in TARGET_FEATURES},
            }
        )
        result = augment_pairs(pairs, target)
        for name in TARGET_FEATURES:
            self.assertEqual(result[f"challenger_{name}"].iloc[0], 2.0)
            self.assertEqual(result[f"victim_{name}"].iloc[0], 0.5)
            self.assertEqual(result[f"delta_{name}"].iloc[0], 1.5)


if __name__ == "__main__":
    unittest.main()
