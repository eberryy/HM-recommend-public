import unittest

import pandas as pd

from hm_recsys.warm_v3_pointwise_propensity import choose_propensity_decisions


class PointwisePropensityTests(unittest.TestCase):
    def test_selects_best_positive_expected_swap_and_preserves_noop(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["a", "a", "b"],
                "challenger_article_id": ["c1", "c2", "c3"],
                "victim_article_id": ["v1", "v1", "v2"],
                "challenger_target": [1, 0, 0],
                "victim_target": [0, 0, 0],
                "challenger_rank": [13, 14, 13],
                "victim_rank": [12, 12, 12],
                "unit_gain": [0.1, 0.2, 0.1],
            }
        )
        probabilities = pd.DataFrame(
            {
                "customer_id": ["a", "a", "a", "b", "b"],
                "article_id": ["c1", "c2", "v1", "c3", "v2"],
                "purchase_probability": [0.4, 0.5, 0.2, 0.1, 0.2],
            }
        )
        chosen = choose_propensity_decisions(pairs, probabilities)
        self.assertEqual(chosen.customer_id.tolist(), ["a"])
        self.assertEqual(chosen.challenger_article_id.tolist(), ["c2"])
        self.assertAlmostEqual(chosen.expected_delta.iloc[0], 0.06)


if __name__ == "__main__":
    unittest.main()
