from __future__ import annotations

import unittest

from hm_recsys.mind_warm_audit import WINDOWS, decide, dynamic_interest_count


class MindWarmAuditTests(unittest.TestCase):
    def test_final_week_is_not_an_audit_window(self) -> None:
        self.assertNotIn("2020-09-16", WINDOWS.values())

    def test_dynamic_interest_count_is_bounded(self) -> None:
        self.assertEqual(dynamic_interest_count(0, 3), 1)
        self.assertEqual(dynamic_interest_count(1, 3), 1)
        self.assertEqual(dynamic_interest_count(4, 3), 2)
        self.assertEqual(dynamic_interest_count(8, 3), 3)
        self.assertEqual(dynamic_interest_count(100, 3), 3)

    def test_decision_uses_all_preregistered_dimensions(self) -> None:
        rules = {
            "go": {
                "clear_multi_interest_truth_share_mean_min": 0.30,
                "secondary_mode_share_of_current_misses_mean_min": 0.20,
                "multi_profile_marginal_recall_mean_min": 0.01,
                "multi_profile_marginal_recall_positive_windows_min": 4,
                "multi_profile_unique_over_single_mean_min": 0.002,
            },
            "conditional_go": {
                "clear_multi_interest_truth_share_mean_min": 0.20,
                "secondary_mode_share_of_current_misses_mean_min": 0.10,
                "multi_profile_marginal_recall_mean_min": 0.003,
                "multi_profile_marginal_recall_positive_windows_min": 3,
                "multi_profile_unique_over_single_mean_min": 0.001,
            },
        }
        go = {
            "clear_multi_interest_truth_share_mean": 0.31,
            "secondary_mode_share_of_current_misses_mean": 0.21,
            "multi_profile_marginal_recall_mean": 0.011,
            "multi_profile_positive_windows": 4,
            "multi_profile_unique_over_single_mean": 0.0021,
        }
        self.assertEqual(decide(go, rules), "GO")
        conditional = dict(go, multi_profile_marginal_recall_mean=0.005)
        self.assertEqual(decide(conditional, rules), "CONDITIONAL GO")
        no_go = dict(conditional, multi_profile_positive_windows=2)
        self.assertEqual(decide(no_go, rules), "NO-GO")


if __name__ == "__main__":
    unittest.main()
