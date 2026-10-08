import unittest

import numpy as np
import pandas as pd

from hm_recsys import mind_warm_candidate_ranker as r


class CandidateRankerTests(unittest.TestCase):
    def test_sql_thread_budget(self):
        with r.connection() as db:
            self.assertEqual(db.execute("SELECT current_setting('threads')").fetchone()[0], 2)

    def scores(self):
        return pd.DataFrame([["u", "b", 2.], ["u", "a", 2.], ["u", "c", 1.],
                             ["v", "d", 5.], ["v", "e", 4.], ["w", "f", 0.]], columns=r.KEYS + ["score"])

    def test_order_label_free_and_deterministic(self):
        ordered = r.order_candidates(self.scores())
        self.assertEqual(ordered.article_id.tolist(), ["a", "b", "c", "d", "e", "f"])
        self.assertEqual(ordered["rank"].tolist(), [1, 2, 3, 1, 2, 1])
        for bad in [self.scores().assign(target=1), self.scores().assign(score=np.nan),
                    pd.concat([self.scores(), self.scores().iloc[:1]])]:
            with self.assertRaises(AssertionError):
                r.order_candidates(bad)

    def test_metrics_denominators(self):
        order = r.order_candidates(self.scores())
        labels = order[r.KEYS].copy()
        labels["target"] = [0, 1, 1, 1, 0, 0]
        result = r.candidate_metrics(order, labels, 4)
        self.assertEqual(result["positive_candidate_users"], 2)
        self.assertEqual(result["conditional_hit_at_1"], .5)
        self.assertEqual(result["all_user_hit_at_1"], .25)
        self.assertAlmostEqual(result["positive_pair_recall_at_1"], 1/3)
        self.assertEqual(result["conditional_hit_at_5"], 1.)
        self.assertEqual(result["conditional_MRR"], .75)
        self.assertEqual(result["all_user_MRR"], .375)

    def test_identity_and_rank_rejections(self):
        order = r.order_candidates(self.scores())
        labels = order[r.KEYS].assign(target=1)
        with self.assertRaises(AssertionError):
            r.candidate_metrics(order.iloc[1:], labels, 4)
        with self.assertRaises(AssertionError):
            r.candidate_metrics(order.assign(rank=1), labels, 4)

    def test_no_victim_or_label_features(self):
        self.assertEqual(len(r.FEATURES), 36)
        self.assertEqual(len(set(r.FEATURES)), 36)
        self.assertFalse(any(f.startswith("v_") for f in r.FEATURES))
        self.assertFalse(set(r.FEATURES) & {"target", "truth_count", "champion_rank", "relation_label", "mind_rank"})

    def test_gate_both_controls_and_all_windows(self):
        rows = {str(i): {"candidate_ranker": {"conditional_hit_at_1": .2},
                         "raw_mind": {"conditional_hit_at_1": .1},
                         "rank002_favorite": {"conditional_hit_at_1": .15}} for i in range(3)}
        self.assertTrue(r.internal_gate(rows)["passed"])
        rows["0"]["raw_mind"]["conditional_hit_at_1"] = .21
        self.assertFalse(r.internal_gate(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
