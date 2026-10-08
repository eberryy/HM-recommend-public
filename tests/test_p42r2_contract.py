"""P4.2R2 preregistration and driver safeguards; no real fits or asset writes."""
from __future__ import annotations

import ast
import copy
import inspect
import json
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hm_recsys.p42_contract import FEATURE_SPEC, TAUS, WINDOWS
from hm_recsys.p42_propensity import MODEL_PARAMS
from hm_recsys.p42r_contract import LOG1P_COUNTS
from hm_recsys.p42r2_contract import FROZEN_KEYS, RUN_ID, preregister, validate_contract
from hm_recsys.p42r2 import convergence_gate, run


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
class P42R2ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report = ROOT / "reports" / "phase4"
        cls.previous = json.loads((report / "P4_2R_EXPERIMENT_CONTRACT.json").read_text(encoding="utf-8"))
        cls.previous_metrics = json.loads((report / "P4_2R_metrics.json").read_text(encoding="utf-8"))

    def make_contract(self):
        previous = copy.deepcopy(self.previous)
        prior = copy.deepcopy(self.previous_metrics)
        reads = {
            "P4_2R_EXPERIMENT_CONTRACT.json": previous,
            "P4_2R_metrics.json": prior,
            "P4_2R_VERIFICATION.json": {"status": "pass"},
        }
        captured = []
        with tempfile.TemporaryDirectory(prefix="hm-p42r2-contract-") as directory:
            def save(path, value):
                captured.append((Path(path), copy.deepcopy(value)))

            with patch("hm_recsys.p42r2_contract.branch_guard"), \
                 patch("hm_recsys.p42r2_contract.read_json", side_effect=lambda path: reads[Path(path).name]), \
                 patch("hm_recsys.p42r2_contract.write_json", side_effect=save), \
                 patch("hm_recsys.p42r2_contract.identity", side_effect=lambda path: {"path": str(path), "bytes": 1, "sha256": "synthetic"}), \
                 patch("hm_recsys.p42r2_contract.subprocess.check_output", return_value="synthetic-commit\n"):
                result = preregister(directory)
        self.assertEqual(previous, self.previous)
        self.assertEqual(prior, self.previous_metrics)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0].name, "P4_2R2_EXPERIMENT_CONTRACT.json")
        self.assertEqual(captured[0][1], result)
        return result

    def test_preregistration_preserves_old_contract_and_only_writes_new_stage(self):
        contract = self.make_contract()
        self.assertEqual(contract["run_id"], RUN_ID)
        self.assertEqual(contract["stage"], "P4.2R2")
        for key in FROZEN_KEYS:
            self.assertEqual(contract[key], self.previous[key], key)
        self.assertEqual(contract["feature_spec"], FEATURE_SPEC)
        self.assertEqual(contract["taus"], TAUS)
        self.assertNotIn("old_qC_control", contract)
        self.assertEqual(contract["final_week"], "not_run")

    def test_only_qw_global_r2_changes_with_fixed_19_count_fields(self):
        contract = self.make_contract()
        repair = contract["qW_repair"]
        self.assertEqual(repair["all_chains_repair"], "R2")
        self.assertEqual(repair["R1"], self.previous["qW_repair"]["R1"])
        self.assertEqual(repair["R2"]["log1p_columns"], LOG1P_COUNTS)
        self.assertEqual(len(LOG1P_COUNTS), 19)
        self.assertEqual(repair["R2"]["allowed_only_if"], "unconditional for every formal qW chain")
        self.assertEqual({key: contract["model"][key] for key in MODEL_PARAMS}, MODEL_PARAMS)
        self.assertEqual(repair["model_params"], MODEL_PARAMS)
        self.assertEqual(contract["budget"]["maximum_formal_propensity_fit_attempts"], 4)

    def test_qc_and_prepared_rows_are_exact_reuse_without_refit_or_reconstruction(self):
        contract = self.make_contract()
        self.assertEqual(contract["qC_reuse"]["new_fits"], 0)
        self.assertEqual(contract["budget"]["qC_fit_attempts"], 0)
        self.assertFalse(contract["budget"]["candidate_generation"])
        self.assertFalse(contract["budget"]["upstream_training"])
        self.assertEqual(contract["prepared_reuse"], self.previous_metrics["prepared"])
        for window in WINDOWS:
            self.assertEqual(contract["qC_reuse"]["windows"][window], self.previous_metrics["training"][window]["qC"])

    def test_convergence_gate_is_all_four_ordered_before_calibration_without_retry(self):
        contract = self.make_contract()
        gate = contract["qW_convergence_gate"]
        self.assertEqual(gate["order"], list(WINDOWS))
        self.assertEqual(gate["required_converged_windows"], 4)
        self.assertEqual(gate["failure_decision"], "qW_global_R2_convergence_failure")
        self.assertIn("no retries", gate["failure_action"])
        self.assertIn("any outer propensity prediction or calibration", gate["pass_before"])
        self.assertIn("qW_global_R2_convergence_failure", contract["decisions"]["precedence"])

    def test_temporal_pools_share_actual_h_support_and_strict_label_end(self):
        contract = self.make_contract()
        all_dates = set()
        for window, outer in WINDOWS.items():
            pool = contract["historical_pools"][window]
            self.assertEqual(pool["qC"], pool["qW"])
            for cutoff in pool["qC"]:
                all_dates.add(cutoff)
                self.assertNotEqual(cutoff, "2019-11-27")
                self.assertLess(date.fromisoformat(cutoff) + timedelta(days=7), date.fromisoformat(outer))
                self.assertIn(cutoff, contract["prepared_reuse"])
        self.assertEqual(len(all_dates), 8)

    def test_validation_rejects_refit_local_repair_or_gate_changes(self):
        contract = self.make_contract()
        for change in (lambda c: c["qC_reuse"].update(new_fits=1),
                       lambda c: c["qW_repair"].update(all_chains_repair="R1"),
                       lambda c: c["qW_convergence_gate"].update(required_converged_windows=3),
                       lambda c: c.update(epsilon=1e-5),
                       lambda c: c.update(final_week="run")):
            changed = copy.deepcopy(contract)
            change(changed)
            with self.assertRaises(AssertionError):
                validate_contract(changed, self.previous)

    def test_runtime_convergence_gate_requires_exact_four_global_r2_in_order(self):
        rows = [{"window": w, "side": "qW", "repair": "R2", "status": "converged"} for w in WINDOWS]
        self.assertTrue(convergence_gate(rows)["pass"])
        for invalid in (rows[:3], rows[::-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                convergence_gate(invalid)
        for key, value in (("repair", "R1"), ("side", "qC"), ("status", "non_converged")):
            invalid = copy.deepcopy(rows)
            invalid[-1][key] = value
            self.assertFalse(convergence_gate(invalid)["pass"])

    def test_driver_gates_chronologically_precede_all_outer_work(self):
        tree = ast.parse(inspect.getsource(run))
        def calls(name):
            return sorted(node.lineno for node in ast.walk(tree)
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name)

        for name in ("fit_repaired_propensity", "convergence_gate", "calibration_gate", "evaluate_window", "aggregate_windows"):
            self.assertEqual(len(calls(name)), 1, name)
        self.assertEqual(len(calls("load_cutoff")), 2)
        self.assertLess(calls("fit_repaired_propensity")[0], calls("convergence_gate")[0])
        self.assertLess(calls("convergence_gate")[0], min(calls("load_cutoff")))
        self.assertLess(calls("calibration_gate")[0], calls("evaluate_window")[0])
        self.assertLess(calls("evaluate_window")[0], calls("aggregate_windows")[0])
        self.assertEqual(calls("fit_propensity"), [])
        self.assertEqual(calls("prepare_cutoff"), [])
        fit_call = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "fit_repaired_propensity")
        self.assertEqual(next(k.value.value for k in fit_call.keywords if k.arg == "repair"), "R2")

    def test_driver_nonconvergence_and_calibration_failure_raise_before_next_stage(self):
        source = inspect.getsource(run)
        failure_check = source.index('if a["status"] != "converged":')
        failure_stop = source.index('raise RepairStop("qW_global_R2_convergence_failure"', failure_check)
        convergence_check = source.index('state["qW_convergence_gate"] = convergence_gate(')
        self.assertLess(failure_stop, convergence_check)
        calibration_check = source.index('if not state["calibration_gate"]["pass"]:')
        calibration_stop = source.index('raise RepairStop("propensity_calibration_failure"', calibration_check)
        admission_start = source.index("result = evaluate_window(")
        self.assertLess(calibration_stop, admission_start)
        # A terminal attempt is persisted before raising, with no surrounding
        # while/retry loop that could silently fit a different recipe.
        self.assertLess(source.index('state["attempts"].append(a)'), failure_check)
        self.assertFalse(any(isinstance(node, ast.While) for node in ast.walk(ast.parse(source))))


if __name__ == "__main__":
    unittest.main()
