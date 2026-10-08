import unittest

import numpy as np
import pandas as pd

from hm_recsys import mind_warm_relation as r


class RelationTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame([
            ["u", "c2", "v8", 8, .01, .96, .03],
            ["u", "c1", "v9", 9, .01, .95, .04],
            ["v", "c3", "v12", 12, .05, .94, .01],
            ["w", "c4", "v8", 8, .02, .96, .02],
        ], columns=r.KEYS + r.PROBS)

    def test_classes(self):
        np.testing.assert_array_equal(r.relation_labels(np.array([0, 0, 1, 1]), np.array([1, 0, 1, 0])), [0, 1, 1, 2])
        with self.assertRaises(ValueError):
            r.relation_labels(np.array([2]), np.array([0]))

    def test_action_refusal_and_one_per_user(self):
        actions = r.select_actions(self.sample())
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions.iloc[0].challenger_article_id, "c1")
        self.assertEqual(actions.iloc[0].champion_rank, 9)

    def test_selector_rejects_labels_head_and_nonfinite(self):
        for frame in [self.sample().assign(target=0), self.sample().assign(champion_rank=7),
                      self.sample().assign(p_benefit=np.nan), self.sample().assign(p_harm=1.2)]:
            with self.assertRaises(ValueError):
                r.select_actions(frame)

    def test_tie_and_chunk_invariance(self):
        frame = self.sample()
        duplicated = pd.concat([frame, frame.assign(challenger_article_id="a")], ignore_index=True)
        whole = r.select_actions(duplicated)
        chunks = pd.concat([r.select_actions(frame), r.select_actions(duplicated.iloc[4:])])
        combined = r.select_actions(chunks[r.KEYS + r.PROBS])
        pd.testing.assert_frame_equal(whole, combined)

    def test_temporal(self):
        r.check_temporal(["2019-12-25"], "2020-01-01")
        for training, evaluation in [(["2020-01-01"], "2020-01-01"),
                                     (["2019-12-25"], "2019-12-31"),
                                     (["2019-12-25"], "2020-09-16")]:
            with self.assertRaises(ValueError):
                r.check_temporal(training, evaluation)

    def test_features_no_label_or_source_rank(self):
        for columns in r.FEATURES.values():
            self.assertEqual(len(columns), len(set(columns)))
            self.assertFalse(set(columns) & {"target", "truth_count", "relation_label", "mind_rank", "candidate_rank", "mind_best_interest", "item2vec_vocab_count"})

    def test_exact_ap_replacement(self):
        baseline = pd.DataFrame({"customer_id": ["u"] * 12, "article_id": [f"v{i}" for i in range(1, 13)],
                                 "champion_rank": list(range(1, 13)), "target": [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], "truth_count": [3] * 12})
        labels = pd.concat([baseline[["customer_id", "article_id", "target"]], pd.DataFrame([["u", "c", 1]], columns=["customer_id", "article_id", "target"])])
        actions = pd.DataFrame([["u", "c", "v8", 8]], columns=r.KEYS)
        evaluated = r.old.exact_action_deltas(baseline, actions, labels)
        changed = baseline.copy()
        changed.loc[7, "target"] = 1
        self.assertAlmostEqual(evaluated.actual_delta.sum(), r.old._mean_ap(changed, 1) - r.old._mean_ap(baseline, 1), places=14)

    def test_gate_no_actions_not_success(self):
        windows = {str(i): {"dense_primary": {"delta_vs_baseline": 0, "MAP@12": .03, "selected_users": 0},
                            "context_control": {"MAP@12": .029}} for i in range(3)}
        result = r.gate(windows, "inner")
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["active_admission_windows"])


if __name__ == "__main__":
    unittest.main()
