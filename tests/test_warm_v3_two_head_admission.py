import unittest

import numpy as np
import pandas as pd

from hm_recsys.warm_v3_two_head_admission import choose_expected_decisions


class TwoHeadAdmissionTests(unittest.TestCase):
    def test_actionability_changes_pair_priority(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u1", "u1"],
                "challenger_article_id": ["c1", "c2"],
                "victim_article_id": ["v1", "v2"],
                "challenger_target": [0, 1],
                "victim_target": [0, 0],
                "challenger_rank": [13, 14],
                "victim_rank": [12, 11],
                "unit_gain": [0.2, 0.1],
            }
        )
        chosen = choose_expected_decisions(
            pairs,
            preference_probability=np.array([0.9, 0.7]),
            actionability_probability=np.array([0.01, 0.9]),
        )
        self.assertEqual(chosen.challenger_article_id.tolist(), ["c2"])
        self.assertGreater(chosen.actual_delta.iloc[0], 0)

    def test_nonpositive_expected_gain_is_noop(self):
        pairs = pd.DataFrame(
            {
                "customer_id": ["u1"],
                "challenger_article_id": ["c1"],
                "victim_article_id": ["v1"],
                "challenger_target": [1],
                "victim_target": [0],
                "challenger_rank": [13],
                "victim_rank": [12],
                "unit_gain": [0.1],
            }
        )
        chosen = choose_expected_decisions(pairs, np.array([0.5]), np.array([0.9]))
        self.assertTrue(chosen.empty)


if __name__ == "__main__":
    unittest.main()
