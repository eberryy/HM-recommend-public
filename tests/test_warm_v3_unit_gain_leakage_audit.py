import unittest

import pandas as pd

from hm_recsys.warm_v3_unit_gain_leakage_audit import choose_two_swaps_label_free


class UnitGainLeakageAuditTest(unittest.TestCase):
    def test_label_free_choice_ignores_target_dependent_unit_gain(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u", "u"],
                "challenger_article_id": ["c1", "c2"],
                "victim_article_id": ["v", "v"],
                "challenger_rank": [13, 14],
                "victim_rank": [12, 12],
                "unit_gain": [0.01, 100.0],
            }
        )
        scores = pd.DataFrame(
            {
                "customer_id": ["u", "u", "u"],
                "article_id": ["c1", "c2", "v"],
                "ranking_score": [0.9, 0.7, 0.1],
            }
        )
        chosen = choose_two_swaps_label_free(pairs, scores)
        self.assertEqual(chosen.iloc[0].challenger_article_id, "c1")


if __name__ == "__main__":
    unittest.main()
