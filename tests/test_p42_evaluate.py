import copy
import unittest

import numpy as np
import pandas as pd

from hm_recsys.metrics import apk
from hm_recsys.p41a_stats import LCM
from hm_recsys.p42_evaluate import aggregate_windows, evaluate_window


def synthetic_data():
    users = np.array(["u", "v"])
    lists = np.array([[f"{u}-w{i}" for i in range(12)] for u in users], dtype=object)
    truths = {"u": {"u-c0", "u-c1", "u-w0"}, "v": {"v-w0", "v-cold-missed"}}
    counts = {"u-c0": 0, "u-c1": 3, "u-w0": 25, "v-w0": 25, "v-cold-missed": 0}
    warm = pd.DataFrame({"customer_id": np.repeat(users, 12), "article_id": lists.ravel(),
        "warm_rank": np.tile(np.arange(1, 13), 2), "target": [int(i in truths[u]) for u in users for i in lists[list(users).index(u)]],
        "interaction_count_before_cutoff": [counts.get(i, 25) for i in lists.ravel()]})
    cold = pd.DataFrame({"customer_id": ["u", "u", "u"], "article_id": ["u-c0", "u-c1", "u-w11"],
        "cold_rank": [1, 2, 3], "target": [1, 1, 0], "interaction_count_before_cutoff": [0, 3, 0],
        "source_branch": ["cold_only", "cold_only", "warm_and_cold"]})
    truth = pd.DataFrame([{"customer_id": u, "article_id": i, "interaction_count_before_cutoff": counts[i]}
        for u in users for i in sorted(truths[u])])
    ap = np.array([apk(list(truths[u]), list(items)) for u, items in zip(users, lists)])
    return {"cutoff": "2020-01-22", "users": users, "warm_lists": lists, "cold": cold, "warm": warm,
        "truthsets": truths, "truth": truth, "baseline_ap": ap, "baseline_map": float(ap.mean()),
        "denominator": np.array([3*LCM, 2*LCM])}


class P42EvaluationTest(unittest.TestCase):
    def aggregation_contract(self):
        return {"windows": {f"w{i}": f"2020-0{i+1}-01" for i in range(4)},
            "selection": {"near_equal_abs_tolerance": 1e-12},
            "gates": {"overall_mean_delta_min": 0., "overall_nondegrade_min": 3,
                "overall_worst_delta_min": -.0002, "warm_mean_delta_min": -.0001,
                "warm_window_delta_min": -.0002, "warm_protected_windows_min": 3,
                "cold_nondegrade_min": 3, "cold_only_positive_windows_min": 2}}

    def calibrations(self, contract, ratio=1.):
        return {w: {side: {"predicted_to_observed_rate_ratio": ratio, "roc_auc": .7, "warnings": []}
            for side in ("qC", "qW")} for w in contract["windows"]}

    def test_multi_admission_overlap_denominator_and_accounting(self):
        data = synthetic_data()
        cold_q = np.array([.8, .7, .99])
        warm_q = np.tile(np.linspace(.99, .01, 12), 2)
        result = evaluate_window(data, cold_q, warm_q)
        self.assertEqual(result["utility"].shape, (2, 12))
        self.assertEqual(result["metrics"]["pair_audit"]["overlap_excluded_cold_rows"], 1)
        self.assertEqual(result["metrics"]["pair_audit"]["pair_rows"], 24)
        a = result["metrics"]["variants"]["A_tau0"]
        self.assertEqual(a["admission"]["total_replacements"], 2)
        self.assertEqual(a["admission"]["max_replacements"], 2)
        self.assertEqual(a["admission"]["admission_user_share"], .5)
        self.assertEqual(a["admission"]["inserted_cold_positive_pairs"], 2)
        self.assertEqual(a["admission"]["removed_warm_positive_pairs"], 0)
        self.assertEqual(a["admission"]["strict_cold_positive_insertions"], 1)
        self.assertEqual(a["admission"]["sparse1_5_positive_insertions"], 1)
        self.assertEqual(a["segments"]["strict_cold"]["truth_users"], 2)
        self.assertEqual(a["admission_buckets"]["0"]["users"], 1)
        self.assertEqual(a["admission_buckets"]["2"]["users"], 1)
        np.testing.assert_array_equal(result["recommendations"]["A_tau0"][1], data["warm_lists"][1])
        rows = result["executed"].query("variant == 'A_tau0'")
        self.assertEqual(rows.cold_row_index.tolist(), [0, 1])
        self.assertTrue((rows.utility > rows.tau).all())

    def test_equal_probability_no_edge_exact_w0(self):
        data = synthetic_data()
        result = evaluate_window(data, np.full(3, .1), np.full(24, .1))
        self.assertEqual(len(result["executed"]), 0)
        for name, pred in result["recommendations"].items():
            np.testing.assert_array_equal(pred, data["warm_lists"])
            self.assertEqual(result["metrics"]["variants"][name]["delta_vs_w0"], 0)

    def test_invalid_probability_or_final_cutoff_fail_closed(self):
        data = synthetic_data()
        with self.assertRaises(ValueError):
            evaluate_window(data, [np.nan, .1, .2], np.full(24, .1))
        data["cutoff"] = "2020-09-16"
        with self.assertRaises(ValueError):
            evaluate_window(data, np.full(3, .1), np.full(24, .1))

    def test_aggregate_selects_conservative_on_exact_safe_tie(self):
        data = synthetic_data()
        result = evaluate_window(data, [.8, .7, .99], np.tile(np.linspace(.99, .01, 12), 2))
        contract = self.aggregation_contract()
        windows = {w: copy.deepcopy(result["metrics"]) for w in contract["windows"]}
        # This fixture makes all thresholds identical and therefore tests only
        # the preregistered selection rule, not tied matching assignments.
        for metrics in windows.values():
            for variant in ("M_tau_ln2", "C_tau_ln4"):
                metrics["variants"][variant] = copy.deepcopy(metrics["variants"]["A_tau0"])
        aggregate = aggregate_windows(windows, self.calibrations(contract), contract)
        self.assertEqual(aggregate["decision"], "promote_p4_2_conservative")
        self.assertEqual(aggregate["selected_variant"], "C_tau_ln4")
        self.assertEqual(len(aggregate["passing_variants"]), 3)
        self.assertTrue(aggregate["summary"]["C_tau_ln4"]["gates"]["all_pass"])
        # Promotion has priority over the evaluation-only severe diagnostic.
        warning = aggregate_windows(windows, self.calibrations(contract, 11.), contract)
        self.assertEqual(warning["decision"], "promote_p4_2_conservative")

    def test_no_gain_and_severe_calibration_decision_precedence(self):
        data = synthetic_data()
        result = evaluate_window(data, np.full(3, .1), np.full(24, .1))
        contract = self.aggregation_contract()
        windows = {w: copy.deepcopy(result["metrics"]) for w in contract["windows"]}
        no_gain = aggregate_windows(windows, self.calibrations(contract), contract)
        self.assertEqual(no_gain["decision"], "cold_gain_not_recovered")
        self.assertEqual(no_gain["selected_variant"], "W0")
        failure = aggregate_windows(windows, self.calibrations(contract, .09), contract)
        self.assertEqual(failure["decision"], "propensity_calibration_failure")
        self.assertEqual(failure["severe_calibration_counts"]["qC"]["extreme_rate_ratio_windows"], 4)
        del windows["w3"]
        with self.assertRaises(ValueError):
            aggregate_windows(windows, self.calibrations(contract), contract)


if __name__ == "__main__":
    unittest.main()
