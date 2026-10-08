"""P4.2R contract/decision checks; no real assets generated or models fitted."""
from __future__ import annotations

import ast
import copy
import inspect
import json
from datetime import date, timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

from hm_recsys.p42_contract import FEATURE_SPEC, TAUS, WINDOWS, branch_guard, guard_cutoff
from hm_recsys.p42_propensity import MODEL_PARAMS
from hm_recsys.p42r import calibration_gate, run
from hm_recsys.p42r_contract import LOG1P_COUNTS, RUN_ID
from hm_recsys.p42r_propensity import R2_COUNT_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def good_calibration():
    return {
        window: {
            side: {
                "predicted_to_observed_rate_ratio": 1.0,
                "roc_auc": 0.7,
                "warnings": [],
            }
            for side in ("qC", "qW")
        }
        for window in WINDOWS
    }


@unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
class P42RContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = ROOT / "reports" / "phase4"
        cls.contract = json.loads((report / "P4_2R_EXPERIMENT_CONTRACT.json").read_text(encoding="utf-8"))
        cls.original = json.loads((report / "P4_2_EXPERIMENT_CONTRACT.json").read_text(encoding="utf-8"))

    def test_original_utility_gate_and_feature_contract_values_unchanged(self):
        self.assertEqual(self.contract["run_id"], RUN_ID)
        self.assertEqual(self.contract["status"], "preregistered_before_formal_computation")
        for key in ("feature_spec", "epsilon", "taus", "utility", "pairing", "gates", "selection", "metrics_contract"):
            self.assertEqual(self.contract[key], self.original[key], key)
        self.assertEqual(self.contract["feature_spec"], FEATURE_SPEC)
        self.assertEqual(self.contract["taus"], TAUS)
        self.assertEqual({key: self.contract["model"][key] for key in MODEL_PARAMS}, MODEL_PARAMS)
        self.assertEqual(self.contract["qW_repair"]["model_params"], MODEL_PARAMS)

    def test_shared_historical_time_pools_and_no_b0_cutoff_excluded_both_sides(self):
        expected = {
            "winter_20200122": ["2019-12-25"],
            "spring_20200318": ["2019-12-25", "2020-01-22", "2020-02-19"],
            "early_summer_20200624": ["2019-12-25", "2020-01-22", "2020-02-19", "2020-03-18", "2020-04-29", "2020-05-27"],
            "late_summer_20200819": ["2019-12-25", "2020-01-22", "2020-02-19", "2020-03-18", "2020-04-29", "2020-05-27", "2020-06-24", "2020-07-22"],
        }
        for window, dates in expected.items():
            for side in ("qC", "qW"):
                self.assertEqual(self.contract["historical_pools"][window][side], dates)
            for cutoff in dates:
                self.assertLess(date.fromisoformat(cutoff) + timedelta(days=7), date.fromisoformat(WINDOWS[window]))
                self.assertTrue(self.contract["inputs"][cutoff]["b0_lineage"]["available"])
        self.assertEqual(self.contract["excluded_cutoffs"], [
            {"cutoff": "2019-11-27", "reason": "no_safe_B0", "excluded_sides": ["qC", "qW"]}
        ])

    def test_exact_r2_list_and_old_control_scope(self):
        self.assertEqual(len(LOG1P_COUNTS), 19)
        self.assertEqual(list(R2_COUNT_FEATURES), LOG1P_COUNTS)
        self.assertEqual(self.contract["qW_repair"]["R2"]["log1p_columns"], LOG1P_COUNTS)
        self.assertEqual(self.contract["old_qC_control"]["window"], "winter_20200122")
        self.assertFalse(self.contract["old_qC_control"]["used_for_utility"])
        self.assertIn("not_available", self.contract["old_qC_control"]["other_windows"])

    def test_branch_guard_never_switches_to_main(self):
        with patch("hm_recsys.p42_contract.subprocess.check_output", return_value="codex/warm\n") as probe:
            with self.assertRaisesRegex(ValueError, "main-only"):
                branch_guard(ROOT)
            probe.assert_called_once_with(["git", "branch", "--show-current"], cwd=ROOT, text=True)

    def test_final_week_and_overlapping_truth_rejected(self):
        for cutoff in ("2020-09-16", "2020-09-17", "2020-09-10"):
            with self.assertRaises(ValueError):
                guard_cutoff(cutoff)
        self.assertEqual(guard_cutoff("2020-08-19"), "2020-08-19")

    def test_calibration_gate_requires_exact_four_windows(self):
        rows = good_calibration()
        self.assertTrue(calibration_gate(rows)["pass"])
        del rows[next(iter(WINDOWS))]
        with self.assertRaises(ValueError):
            calibration_gate(rows)

    def test_extreme_ratio_three_window_failure_and_inclusive_boundaries(self):
        rows = good_calibration()
        keys = list(WINDOWS)
        for window in keys[:2]:
            rows[window]["qC"]["predicted_to_observed_rate_ratio"] = 10.1
        self.assertTrue(calibration_gate(rows)["pass"])
        rows[keys[2]]["qC"]["predicted_to_observed_rate_ratio"] = 0.099
        self.assertFalse(calibration_gate(rows)["pass"])
        rows[keys[0]]["qC"]["predicted_to_observed_rate_ratio"] = 0.1
        rows[keys[1]]["qC"]["predicted_to_observed_rate_ratio"] = 10.0
        self.assertTrue(calibration_gate(rows)["pass"])

    def test_roc_and_constant_failures_are_per_side_per_reason(self):
        rows = good_calibration()
        keys = list(WINDOWS)
        for window in keys[:3]:
            rows[window]["qW"]["roc_auc"] = 0.5
        self.assertFalse(calibration_gate(rows)["pass"])
        rows = good_calibration()
        for window in keys[:3]:
            rows[window]["qC"]["warnings"] = ["constant_prediction"]
        self.assertFalse(calibration_gate(rows)["pass"])
        # Two extreme windows plus one different warning are not three of
        # the same preregistered severe reason.
        rows = good_calibration()
        for window in keys[:2]:
            rows[window]["qC"]["predicted_to_observed_rate_ratio"] = 11.0
        rows[keys[2]]["qC"]["roc_auc"] = 0.5
        self.assertTrue(calibration_gate(rows)["pass"])

    def test_calibration_does_not_mutate_inputs(self):
        rows = good_calibration()
        original = copy.deepcopy(rows)
        calibration_gate(rows)
        self.assertEqual(rows, original)

    def test_driver_calibration_gate_precedes_admission_call(self):
        # Static call-graph chronology complements gate unit tests without
        # running expensive reconstruction or fitting any model.
        tree = ast.parse(inspect.getsource(run))
        calls = {
            name: [node.lineno for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]
            for name in ("calibration_gate", "evaluate_window", "aggregate_windows")
        }
        self.assertEqual(len(calls["calibration_gate"]), 1)
        self.assertEqual(len(calls["evaluate_window"]), 1)
        self.assertLess(calls["calibration_gate"][0], calls["evaluate_window"][0])
        self.assertLess(calls["evaluate_window"][0], calls["aggregate_windows"][0])
        source = inspect.getsource(run)
        gate = source.index('if not state["calibration_gate"]["pass"]:')
        stop = source.index('raise RepairStop("propensity_calibration_failure"', gate)
        admission = source.index("result = evaluate_window(")
        self.assertLess(stop, admission)


if __name__ == "__main__":
    unittest.main()
