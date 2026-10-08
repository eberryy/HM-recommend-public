import unittest

import pandas as pd

from hm_recsys.m21 import (
    inactive_fallback_order_expression,
    prepare_ranking_frame,
    summarize_development,
)


def _ordering(overall_map: float, inactive_map: float) -> dict:
    return {
        "segments": {"overall": {"map@12": overall_map}},
        "activity_segments": [
            {"activity_segment": "inactive_12w", "map@12": inactive_map},
            {"activity_segment": "high_21_plus", "map@12": overall_map},
        ],
    }


class M21Tests(unittest.TestCase):
    def test_ranking_group_key_keeps_same_customer_cutoffs_separate(self):
        frame = pd.DataFrame(
            {
                "target_cutoff": [
                    "2020-05-27",
                    "2020-05-27",
                    "2020-06-24",
                    "2020-06-24",
                ],
                "customer_id": ["u1", "u1", "u1", "u1"],
                "article_id": ["a2", "a1", "a4", "a3"],
                "candidate_rank": [2, 1, 2, 1],
                "target": [0, 1, 0, 0],
            }
        )
        ordered, sizes, evidence = prepare_ranking_frame(frame, candidate_k=2)
        self.assertEqual(sizes, [2, 2])
        self.assertEqual(evidence["groups"], 2)
        self.assertEqual(evidence["positive_groups"], 1)
        self.assertEqual(evidence["zero_positive_groups"], 1)
        self.assertEqual(
            ordered[["target_cutoff", "candidate_rank"]].values.tolist(),
            [
                ["2020-05-27", 1],
                ["2020-05-27", 2],
                ["2020-06-24", 1],
                ["2020-06-24", 2],
            ],
        )

    def test_ranking_group_rejects_wrong_candidate_count(self):
        frame = pd.DataFrame(
            {
                "target_cutoff": ["2020-05-27"],
                "customer_id": ["u1"],
                "article_id": ["a1"],
                "candidate_rank": [1],
                "target": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            prepare_ranking_frame(frame, candidate_k=2)

    def test_inactive_fallback_expression_is_fixed_and_validated(self):
        self.assertEqual(
            inactive_fallback_order_expression("score_model"),
            "CASE WHEN user_history_events_12w = 0 "
            "THEN -candidate_rank ELSE score_model END DESC",
        )
        with self.assertRaises(ValueError):
            inactive_fallback_order_expression("score; DROP TABLE")

    def test_summary_accepts_consistent_inactive_improvement(self):
        variants = ["model"]
        development = {
            "dev_a": {
                "evaluation": {
                    "orderings": {
                        "rrf": _ordering(0.010, 0.008),
                        "model": _ordering(0.020, 0.006),
                        "model__inactive_rrf": _ordering(0.021, 0.008),
                    }
                }
            },
            "dev_b": {
                "evaluation": {
                    "orderings": {
                        "rrf": _ordering(0.012, 0.009),
                        "model": _ordering(0.022, 0.007),
                        "model__inactive_rrf": _ordering(0.023, 0.009),
                    }
                }
            },
        }
        summary = summarize_development(development, variants, metric_k=12)
        self.assertTrue(summary["fallback_gates"]["model"]["accepted"])
        self.assertEqual(
            summary["selected_development_candidate"], "model__inactive_rrf"
        )
        self.assertEqual(summary["final_confirmation_run"], "not_run")


if __name__ == "__main__":
    unittest.main()
