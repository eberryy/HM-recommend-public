import unittest

import pandas as pd

from hm_recsys.warm_v3_pair_gate_audit import best_exact_gain, top_pair_proposals


class PairGateAuditTest(unittest.TestCase):
    def test_top_pair_proposals_uses_positive_expected_gain_and_limit(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u"] * 3,
                "challenger_article_id": ["c1", "c2", "c3"],
                "victim_article_id": ["v", "v", "v"],
                "challenger_target": [1, 0, 0],
                "victim_target": [0, 0, 0],
                "challenger_rank": [13, 14, 15],
                "victim_rank": [12, 12, 12],
                "unit_gain": [0.1, 0.1, 0.1],
            }
        )
        probabilities = pd.DataFrame(
            {
                "customer_id": ["u"] * 4,
                "article_id": ["c1", "c2", "c3", "v"],
                "purchase_probability": [0.8, 0.7, 0.1, 0.2],
            }
        )
        result = top_pair_proposals(pairs, probabilities, top_pairs=2)
        self.assertEqual(result.challenger_article_id.tolist(), ["c1", "c2"])
        self.assertEqual(result.proposal_rank.tolist(), [1, 2])

    def test_best_exact_gain_can_choose_two_disjoint_swaps(self):
        base = pd.DataFrame(
            {
                "rf": list(range(1, 15)),
                "target": [0] * 12 + [1, 1],
                "truth_count": [2] * 14,
            }
        )
        proposals = pd.DataFrame(
            {
                "challenger_article_id": ["c13", "c14"],
                "victim_article_id": ["v12", "v11"],
                "challenger_rank": [13, 14],
                "victim_rank": [12, 11],
            }
        )
        gain, count = best_exact_gain(base, proposals)
        self.assertGreater(gain, 0)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
