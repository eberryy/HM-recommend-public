import unittest

import pandas as pd

from hm_recsys.warm_v3_label_free_repair import (
    PROPOSAL_COLUMNS,
    choose_label_free_swaps,
    decision_invariance_audit,
    unlabeled_proposals,
)


class LabelFreeRepairTest(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame(
            {
                "customer_id": ["u", "u", "u"],
                "challenger_article_id": ["c1", "c1", "c2"],
                "victim_article_id": ["v1", "v2", "v2"],
                "challenger_rank": [13, 13, 14],
                "victim_rank": [12, 11, 11],
                "challenger_target": [0, 1, 0],
                "victim_target": [1, 0, 1],
                "unit_gain": [1000.0, 0.0001, 500.0],
            }
        )
        self.scores = pd.DataFrame(
            {
                "customer_id": ["u", "u", "u", "u"],
                "article_id": ["c1", "c2", "v1", "v2"],
                "ranking_score": [0.9, 0.8, 0.1, 0.2],
            }
        )

    def test_projection_removes_all_labels_and_gain(self):
        proposals = unlabeled_proposals(self.pairs)
        self.assertEqual(list(proposals.columns), PROPOSAL_COLUMNS)
        self.assertNotIn("unit_gain", proposals)
        self.assertNotIn("challenger_target", proposals)

    def test_selects_two_disjoint_swaps_by_score_difference(self):
        chosen = choose_label_free_swaps(unlabeled_proposals(self.pairs), self.scores)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(chosen.challenger_article_id.nunique(), 2)
        self.assertEqual(chosen.victim_article_id.nunique(), 2)
        self.assertEqual(chosen.iloc[0].challenger_article_id, "c1")
        self.assertEqual(chosen.iloc[0].victim_article_id, "v1")

    def test_label_mutation_does_not_change_choices(self):
        result = decision_invariance_audit(self.pairs, self.scores)
        self.assertTrue(result["passed"])
        self.assertTrue(result["choices_identical_after_label_mutation"])
        self.assertTrue(result["forbidden_columns_absent"])

    def test_selector_rejects_polluted_decision_frame(self):
        polluted = unlabeled_proposals(self.pairs).assign(unit_gain=1.0)
        with self.assertRaises(ValueError):
            choose_label_free_swaps(polluted, self.scores)


if __name__ == "__main__":
    unittest.main()
