import unittest

import pandas as pd

from hm_recsys.warm_v3_bpr_rich_admission import select_admissions


class BprRichAdmissionTest(unittest.TestCase):
    def test_uses_only_best_candidate_and_requires_score_improvement(self):
        candidates = pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2"],
                "article_id": ["a", "b", "c"],
                "expanded_rank": [4, 2, 1],
                "challenger_score": [0.9, 0.8, 0.2],
            }
        )
        victims = pd.DataFrame(
            {
                "customer_id": ["u1", "u2"],
                "victim_article_id": ["v1", "v2"],
                "victim_score": [0.85, 0.3],
            }
        )
        selected = select_admissions(candidates, victims)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0].customer_id, "u1")
        self.assertEqual(selected.iloc[0].article_id, "a")

    def test_rejects_label_polluted_candidate_input(self):
        candidates = pd.DataFrame(
            {
                "customer_id": ["u"],
                "article_id": ["a"],
                "expanded_rank": [1],
                "challenger_score": [0.9],
                "target": [1],
            }
        )
        victims = pd.DataFrame(
            {"customer_id": ["u"], "victim_article_id": ["v"], "victim_score": [0.1]}
        )
        with self.assertRaises(ValueError):
            select_admissions(candidates, victims)


if __name__ == "__main__":
    unittest.main()
