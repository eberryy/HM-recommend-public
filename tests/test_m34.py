import unittest
from pathlib import Path

from hm_recsys.m34 import (
    GAP_WINDOWS,
    OUTER_WINDOWS,
    SOURCE_CONFIG,
    VARIANT_BUDGETS,
    summarize_development,
    validate_protocol,
)


def _metric(recall, budget):
    return {
        f"candidate_recall@{budget}": recall,
        f"candidate_hit_rate@{budget}": recall,
        "oracle_map@12": recall,
        "map@12": 0.0,
        "users": 10,
        "truth_pairs": 20,
    }


class M34ProtocolTests(unittest.TestCase):
    def test_protocol_is_fixed_and_excludes_final_week(self):
        validate_protocol()
        self.assertEqual(len(OUTER_WINDOWS), 4)
        self.assertTrue(all(cutoff < "2020-09-16" for cutoff in OUTER_WINDOWS.values()))
        self.assertEqual(SOURCE_CONFIG["source_k"], 100)
        self.assertEqual(VARIANT_BUDGETS["expanded_350"], 350)

    def test_gate_uses_expanded_pool_and_all_pre_registered_checks(self):
        development = {}
        for window in OUTER_WINDOWS:
            evaluations = {}
            for name, budget in VARIANT_BUDGETS.items():
                base = 0.10
                delta = 0.004 if name == "expanded_350" else 0.0
                evaluations[name] = {
                    "overall": _metric(base + delta, budget),
                    "season_sensitive": _metric(base + delta, budget),
                }
            development[window] = {
                "evaluations": evaluations,
                "source_overlap": {"marginal_season_sensitive_truth_pairs": 25},
            }
        summary = summarize_development(development)
        self.assertTrue(summary["retrieval_gate"]["passed"])
        self.assertEqual(
            set(summary["retrieval_gate"]["checks"]),
            {
                "gap_windows_season_sensitive_recall_strictly_improve",
                "mean_season_sensitive_recall_delta_at_least_0_002",
                "gap_windows_at_least_20_marginal_season_truth_pairs",
                "mean_overall_recall_delta_at_least_0_001",
            },
        )

    def test_winter_does_not_replace_the_three_known_gap_windows(self):
        self.assertNotIn("winter_20200122", GAP_WINDOWS)
        self.assertEqual(len(GAP_WINDOWS), 3)

    def test_overlap_audit_never_uses_join_only_using_syntax_in_exists(self):
        source = Path("src/hm_recsys/m34.py").read_text(encoding="utf-8")
        self.assertNotIn("FROM base b USING(customer_id,article_id)", source)

    def test_evaluation_database_is_closed_before_identity_hash(self):
        source = Path("src/hm_recsys/m34.py").read_text(encoding="utf-8")
        close_at = source.index('connection.execute("CHECKPOINT")')
        hash_at = source.index('"evaluation_db": _file_identity(database_path)', close_at)
        self.assertLess(close_at, hash_at)


if __name__ == "__main__":
    unittest.main()
