import unittest

import numpy as np
import pandas as pd

from hm_recsys.warm_v3_clean_model_replay import choose_pair_actions, fused_candidate_scores


class CleanModelReplayTest(unittest.TestCase):
    def test_pair_actions_are_disjoint_and_do_not_require_labels(self):
        proposals = pd.DataFrame(
            {
                "customer_id": ["u", "u", "u"],
                "challenger_article_id": ["c1", "c1", "c2"],
                "victim_article_id": ["v1", "v2", "v2"],
                "challenger_rank": [13, 13, 14],
                "victim_rank": [12, 11, 11],
            }
        )
        chosen = choose_pair_actions(proposals, np.array([0.9, 0.8, 0.7]))
        self.assertEqual(len(chosen), 2)
        self.assertEqual(chosen.challenger_article_id.nunique(), 2)
        self.assertEqual(chosen.victim_article_id.nunique(), 2)

    def test_rrf_fusion_is_invariant_to_score_scale(self):
        candidates = pd.DataFrame(
            {
                "customer_id": ["u", "u", "u"],
                "article_id": ["a", "b", "c"],
                "rf": [1, 2, 3],
            }
        )
        first = fused_candidate_scores(candidates, np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0]))
        second = fused_candidate_scores(candidates, np.array([30.0, 20.0, 10.0]), np.array([10.0, 30.0, 20.0]))
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
