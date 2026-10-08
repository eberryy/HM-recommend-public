import unittest

import numpy as np
import pandas as pd

from hm_recsys.mind_warm_relation_audit import KEYS, action_overlap, ap_matrix, constrained_oracle


class IndependentRelationAuditTests(unittest.TestCase):
    def test_ap_uses_complete_truth_count_not_candidate_hits(self):
        labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.int8)
        self.assertAlmostEqual(ap_matrix(labels, np.array([3]))[0], (1 + 2 / 9) / 3)
        self.assertAlmostEqual(ap_matrix(labels, np.array([30]))[0], (1 + 2 / 9) / 12)

    def test_oracle_accounts_for_effect_on_later_positive_precision(self):
        labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.int8)
        # Adding a positive at rank8 also changes the precision of the rank9 hit.
        expected = (2 / 8 + 1 / 9) / 3
        self.assertAlmostEqual(constrained_oracle(labels, np.array([3]), np.array([True]))[0], expected)

    def test_oracle_requires_available_candidate_and_allows_rejection(self):
        labels = np.zeros((2, 12), dtype=np.int8)
        result = constrained_oracle(labels, np.array([1, 1]), np.array([False, True]))
        np.testing.assert_allclose(result, [0, 1 / 8])
        saturated = np.ones((1, 12), dtype=np.int8)
        np.testing.assert_array_equal(constrained_oracle(saturated, np.array([13]), np.array([True])), [0])

    def test_user_intersection_is_not_exact_action_intersection(self):
        left = pd.DataFrame([["u", "a", "v", 8]], columns=KEYS)
        right = pd.DataFrame([["u", "b", "v", 8]], columns=KEYS)
        result = action_overlap(left, right)
        self.assertEqual(result["exact_action_intersection"], 0)
        self.assertEqual(result["selected_user_intersection"], 1)
        self.assertEqual(result["both_select_same_user_but_different_action"], 1)


if __name__ == "__main__":
    unittest.main()
