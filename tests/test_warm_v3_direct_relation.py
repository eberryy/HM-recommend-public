import unittest

import numpy as np
import pandas as pd

from hm_recsys.warm_v3_direct_relation import choose_direct_actions, relation_label


class DirectRelationTest(unittest.TestCase):
    def test_relation_labels(self):
        frame = pd.DataFrame(
            {"challenger_target": [0, 0, 1, 1], "victim_target": [1, 0, 0, 1]}
        )
        np.testing.assert_array_equal(relation_label(frame), np.array([0, 1, 2, 1]))

    def test_action_selector_is_label_free_and_disjoint(self):
        proposals = pd.DataFrame(
            {
                "customer_id": ["u", "u", "u"],
                "challenger_article_id": ["c1", "c1", "c2"],
                "victim_article_id": ["v1", "v2", "v2"],
                "challenger_rank": [13, 13, 14],
                "victim_rank": [12, 11, 11],
            }
        )
        chosen = choose_direct_actions(proposals, np.ones(3), np.array([0.9, 0.8, 0.7]))
        self.assertEqual(len(chosen), 2)
        self.assertNotIn("target", chosen)
        self.assertNotIn("unit_gain", chosen)


if __name__ == "__main__":
    unittest.main()
