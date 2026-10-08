import unittest

import numpy as np
import pandas as pd

from hm_recsys.final_f70_explainability_audit import rank_positions, subgroup_recall


class TestF70ExplainabilityAudit(unittest.TestCase):
    def test_rank_positions_are_per_user_and_stable(self):
        frame = pd.DataFrame({"user_index": [0, 0, 0, 1, 1]})
        score = np.array([0.2, 0.4, 0.4, -1.0, 2.0])
        self.assertEqual(rank_positions(frame, score).tolist(), [3, 1, 2, 2, 1])

    def test_subgroup_recall_uses_positive_pairs_as_denominator(self):
        frame = pd.DataFrame({
            "target": [1, 1, 0, 1],
            "strict_cold_flag": [1, 1, 1, 0],
        })
        result = subgroup_recall(frame, np.array([1, 7, 2, 1]), "strict_cold_flag")
        self.assertEqual(result["positive_pair_denominator"], 2)
        self.assertEqual(result["hits"], {"1": 1, "5": 1})
        self.assertEqual(result["recall"], {"1": 0.5, "5": 0.5})


if __name__ == "__main__":
    unittest.main()
