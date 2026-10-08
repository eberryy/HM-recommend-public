from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from hm_recsys import mind_warm_ranking as ranking


class MindWarmRankingTests(unittest.TestCase):
    def test_final_week_guard_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "final week"):
            ranking._guard("2020-09-16")

    def test_registered_control_hides_every_mind_source_feature(self) -> None:
        columns = ["customer_id", "article_id", "candidate_rank", *ranking.MIND_FEATURES, "target"]
        primary = [name for name in columns if name not in ranking.NON_FEATURE_COLUMNS]
        control = [name for name in primary if name not in ranking.MIND_FEATURES]
        self.assertTrue(set(ranking.MIND_FEATURES).issubset(primary))
        self.assertFalse(set(ranking.MIND_FEATURES) & set(control))
        self.assertIn("candidate_rank", control)

    def test_action_selection_uses_no_labels_and_keeps_one_pair_per_user(self) -> None:
        challengers = pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2"],
                "article_id": ["c1", "c2", "c3"],
                "mind_rank": [2, 1, 1],
                "score": [0.8, 0.8, 0.1],
            }
        )
        victims = pd.DataFrame(
            {
                "customer_id": ["u1", "u1", "u2"],
                "article_id": ["v8", "v12", "v9"],
                "champion_rank": [8, 12, 9],
                "score": [0.4, 0.4, 0.2],
            }
        )
        selected = ranking.select_actions(challengers, victims, "score")
        self.assertEqual(selected["customer_id"].tolist(), ["u1"])
        self.assertEqual(selected.iloc[0]["challenger_article_id"], "c2")
        self.assertEqual(selected.iloc[0]["victim_article_id"], "v12")
        self.assertTrue(selected["customer_id"].is_unique)

    def test_action_selection_rejects_label_columns(self) -> None:
        challengers = pd.DataFrame(
            {"customer_id": ["u"], "article_id": ["c"], "mind_rank": [1], "score": [1.0], "target": [1]}
        )
        victims = pd.DataFrame(
            {"customer_id": ["u"], "article_id": ["v"], "champion_rank": [12], "score": [0.0]}
        )
        with self.assertRaisesRegex(ValueError, "labels"):
            ranking.select_actions(challengers, victims, "score")

    def test_exact_ap_recomputes_later_precision_effect(self) -> None:
        before = ranking._apk(np.array([0, 1, 0, 1], dtype=np.uint8), truth_count=2)
        after = ranking._apk(np.array([1, 1, 0, 1], dtype=np.uint8), truth_count=3)
        self.assertAlmostEqual(before, (1 / 2 + 2 / 4) / 2)
        self.assertAlmostEqual(after, (1 + 2 / 2 + 3 / 4) / 3)

    def test_matrix_missing_policy_is_finite(self) -> None:
        frame = pd.DataFrame({"x": [np.nan, np.inf, -np.inf, 1.0]})
        values = ranking._matrix(frame, ["x"])
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[:, 0].tolist(), [0.0, 1e6, -1e6, 1.0])

    def test_gate_requires_positive_primary_and_control_increment(self) -> None:
        contract = {
            "inner_gate": {
                "primary_mean_map12_delta_vs_frozen_baseline_min_exclusive": 0,
                "primary_nondegrade_windows_min": 3,
                "primary_worst_window_delta_min": -0.0002,
                "primary_mean_delta_vs_same_pool_control_min_exclusive": 0,
            }
        }
        windows = {
            str(index): {
                "policies": {
                    "same_pool_control": {"MAP@12": 0.02},
                    "mind_aware": {
                        "delta_vs_frozen_baseline": value,
                        "delta_vs_same_pool_control": 0.00001,
                    },
                }
            }
            for index, value in enumerate([0.0001, 0.0001, 0.0001, -0.0001])
        }
        with mock.patch.object(ranking, "_read_json", return_value=contract):
            self.assertTrue(ranking._gate(windows, "inner_gate")["passed"])


if __name__ == "__main__":
    unittest.main()
