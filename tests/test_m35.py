import unittest
from pathlib import Path

from hm_recsys.m34 import OUTER_WINDOWS
from hm_recsys.m35 import CONFIG, VARIANT_BUDGETS, summarize, validate_protocol


def _metrics(value, budget):
    return {f"candidate_recall@{budget}": value, f"candidate_hit_rate@{budget}": value, "oracle_map@12": value, "map@12": 0.0, "users": 10, "truth_pairs": 20}


class M35ProtocolTests(unittest.TestCase):
    def test_protocol_excludes_final_week_and_is_bounded(self):
        validate_protocol()
        self.assertTrue(all(cutoff < "2020-09-16" for cutoff in OUTER_WINDOWS.values()))
        self.assertEqual(CONFIG["image_seed_k"], 5)
        self.assertEqual(VARIANT_BUDGETS["expanded_combined_400"], 400)

    def test_gate_requires_both_sources_and_all_cold_windows(self):
        development = {}
        for window in OUTER_WINDOWS:
            evaluations = {}
            for name, budget in VARIANT_BUDGETS.items():
                delta = 0.01 if name == "expanded_combined_400" else 0.0
                evaluations[name] = {"overall": _metrics(0.10 + delta, budget), "cold": _metrics(0.02 + delta, budget)}
            development[window] = {"evaluations": evaluations, "marginal_cold_truth": {"attribute": 2, "image": 3, "combined": 12}}
        self.assertTrue(summarize(development)["retrieval_gate"]["passed"])

    def test_combined_pool_has_independent_fifty_item_route_caps(self):
        source = Path("src/hm_recsys/m35.py").read_text(encoding="utf-8")
        self.assertIn("FROM attr_new WHERE new_rank<=50", source)
        self.assertIn("FROM filtered QUALIFY selected_rank<=50", source)


if __name__ == "__main__":
    unittest.main()
