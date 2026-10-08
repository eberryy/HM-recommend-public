import unittest

import numpy as np
import pandas as pd

from hm_recsys.warm_v3_residual_admission import THRESHOLD, choose_decisions


class ResidualAdmissionTests(unittest.TestCase):
    def test_selects_at_most_one_positive_margin_pair_per_user(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2"],
                "challenger_article_id": ["c1", "c2", "c3"],
                "victim_article_id": ["v1", "v2", "v3"],
                "challenger_target": [1, 0, 0],
                "victim_target": [0, 0, 1],
                "challenger_rank": [13, 14, 13],
                "victim_rank": [12, 11, 12],
                "unit_gain": [0.1, 0.2, 0.1],
            }
        )
        chosen = choose_decisions(pairs, np.array([0.8, 0.6, 0.4]))
        self.assertEqual(chosen.customer_id.tolist(), ["u1"])
        self.assertEqual(chosen.challenger_article_id.tolist(), ["c1"])
        self.assertGreater(chosen.actual_delta.iloc[0], 0)

    def test_threshold_is_strict_and_tie_prefers_safer_tail_victim(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2"],
                "challenger_article_id": ["c1", "c1", "c2"],
                "victim_article_id": ["v8", "v12", "v2"],
                "challenger_target": [1, 1, 1],
                "victim_target": [0, 0, 0],
                "challenger_rank": [13, 13, 13],
                "victim_rank": [8, 12, 12],
                "unit_gain": [0.1, 0.1, 0.1],
            }
        )
        chosen = choose_decisions(pairs, np.array([0.6, 0.6, THRESHOLD]))
        self.assertEqual(chosen.victim_article_id.tolist(), ["v12"])


if __name__ == "__main__":
    unittest.main()
