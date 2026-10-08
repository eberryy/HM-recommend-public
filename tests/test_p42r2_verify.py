import copy
import unittest

import numpy as np

from hm_recsys.p42_contract import WINDOWS, TAUS
from hm_recsys.p42_evaluate import aggregate_windows, evaluate_window
from hm_recsys.p42r2_verify import (
    INVARIANTS, independent_calibration_gate, independent_promotion,
    verify_admission_bundle, verify_global_sequence,
)
import test_p42_evaluate as fixtures


class P42R2VerifyTest(unittest.TestCase):
    def _attempts(self):
        return [{"window": window, "side": "qW", "repair": "R2", "status": "converged", "convergence_warnings": []} for window in WINDOWS]

    def _bundle(self, tie=False):
        data = fixtures.synthetic_data()
        qc = np.full(3, .1) if tie else np.array([.8, .7, .99])
        qw = np.full(24, .1) if tie else np.tile(np.linspace(.99, .01, 12), 2)
        result = evaluate_window(data, qc, qw)
        cold, warm = data["cold"].assign(propensity=qc), data["warm"].assign(propensity=qw)
        return {"data": data, "cold": cold, "warm": warm, **{k: result[k] for k in (
            "executed", "user_audit", "utility", "recommendations", "metrics")}, "eligible": result["eligible_cold"]}

    def test_28_unique_invariants(self):
        self.assertEqual(len(INVARIANTS), 28)
        self.assertEqual(len(set(INVARIANTS)), 28)

    def test_all_chains_global_R2_and_stop_prefix(self):
        rows = self._attempts()
        self.assertTrue(verify_global_sequence(rows, {}, {}, "engineering_failure")["all_four_converged"])
        rows[2].update(status="non_converged", convergence_warnings=["fixed budget exhausted"])
        receipt = verify_global_sequence(rows[:3], {}, {}, "qW_global_R2_convergence_failure")
        self.assertEqual(receipt["attempts"], 3)
        self.assertFalse(receipt["all_four_converged"])
        with self.assertRaises(AssertionError):
            verify_global_sequence(rows, {}, {}, "qW_global_R2_convergence_failure")

    def test_no_per_window_R1_rescue_or_reordered_retry(self):
        rows = self._attempts()
        rows[2]["repair"] = "R1"
        with self.assertRaises(AssertionError):
            verify_global_sequence(rows, {}, {}, "engineering_failure")
        with self.assertRaises(AssertionError):
            verify_global_sequence(self._attempts()[::-1], {}, {}, "engineering_failure")
        with self.assertRaises(AssertionError):
            verify_global_sequence(self._attempts()[:2], {"some": {}}, {}, "engineering_failure")

    def test_calibration_gate_is_three_window_any_same_side(self):
        calibration = {w: {s: {"predicted_to_observed_rate_ratio": 1., "roc_auc": .7, "warnings": []} for s in ("qC", "qW")} for w in WINDOWS}
        self.assertTrue(independent_calibration_gate(calibration)["pass"])
        for w in list(WINDOWS)[:2]:
            calibration[w]["qC"]["predicted_to_observed_rate_ratio"] = 11.
        self.assertTrue(independent_calibration_gate(calibration)["pass"])
        calibration[list(WINDOWS)[2]]["qC"]["predicted_to_observed_rate_ratio"] = 11.
        self.assertFalse(independent_calibration_gate(calibration)["pass"])
        with self.assertRaises(AssertionError):
            independent_calibration_gate({})

    def test_independent_matching_risk_segment_and_full_denominator(self):
        bundle = self._bundle()
        derived, receipt = verify_admission_bundle(**bundle)
        self.assertTrue(receipt["pass"])
        self.assertEqual(receipt["maximum_admissions"], 2)
        self.assertEqual(receipt["segment_denominators"], 16)
        self.assertEqual(receipt["admission_buckets"], 20)
        self.assertEqual(derived["variants"]["A_tau0"]["admission"]["inserted_cold_positive_pairs"], 2)
        np.testing.assert_array_equal(bundle["recommendations"]["A_tau0"][1], bundle["data"]["warm_lists"][1])

    def test_zero_edges_replays_W0_for_all_variants(self):
        _, receipt = verify_admission_bundle(**self._bundle(tie=True))
        self.assertEqual(receipt["executed_replacements"], 0)
        self.assertEqual(receipt["maximum_admissions"], 0)

    def test_wrong_probability_or_utility_or_matching_rejected(self):
        for kind in ("q", "utility", "matching"):
            bundle = self._bundle()
            if kind == "q":
                bundle["cold"].loc[0, "propensity"] = np.nan
            elif kind == "utility":
                bundle["utility"][0, 0] += 1e-8
            else:
                bundle["executed"] = bundle["executed"].iloc[1:].copy()
            with self.assertRaises(AssertionError):
                verify_admission_bundle(**bundle)

    def test_wrong_risk_bucket_or_denominator_rejected(self):
        for kind in ("risk", "bucket", "segment", "user_audit"):
            bundle = self._bundle()
            a = bundle["metrics"]["variants"]["A_tau0"]
            if kind == "risk":
                a["admission"]["removed_warm_positive_pairs"] += 1
            elif kind == "bucket":
                a["admission_buckets"]["0"]["users"] -= 1
            elif kind == "segment":
                a["segments"]["strict_cold"]["truth_users"] -= 1
            else:
                bundle["user_audit"].loc[0, "ap"] = .9
            with self.assertRaises(AssertionError):
                verify_admission_bundle(**bundle)

    def test_independent_promotion_matches_original_frozen_gates(self):
        helper = fixtures.P42EvaluationTest()
        contract = helper.aggregation_contract()
        for tie in (True, False):
            bundle = self._bundle(tie=tie)
            windowmetrics = {w: copy.deepcopy(bundle["metrics"]) for w in contract["windows"]}
            actual = aggregate_windows(windowmetrics, helper.calibrations(contract), contract)
            expected = independent_promotion(windowmetrics, contract)
            self.assertEqual(expected["decision"], actual["decision"])
            self.assertEqual(expected["selected_variant"], actual["selected_variant"])
            self.assertEqual(expected["passing_variants"], actual["passing_variants"])
            self.assertEqual(expected["summary"], actual["summary"])


if __name__ == "__main__":
    unittest.main()
