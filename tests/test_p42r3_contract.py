"""P4.2R3 contract/gate unit checks; no real model fits or artifact writes."""
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hm_recsys.p42_propensity import MODEL_PARAMS
from hm_recsys.p42_contract import WINDOWS
from hm_recsys.p42r3_contract import RUN_ID, QW_MODEL_PARAMS, preregister, validate_contract
from hm_recsys.p42r3 import convergence_gate, run


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
class P42R3ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous = json.loads((ROOT/"reports/phase4/P4_2R2_EXPERIMENT_CONTRACT.json").read_text(encoding="utf-8"))

    def contract(self):
        previous = copy.deepcopy(self.previous)
        reads = {"P4_2R2_EXPERIMENT_CONTRACT.json": previous,
                 "P4_2R2_metrics.json": {"decision": "qW_global_R2_convergence_failure"},
                 "P4_2R2_VERIFICATION.json": {"status": "pass"}}
        saved = []
        with tempfile.TemporaryDirectory(prefix="hm-r3-contract-") as folder:
            with patch("hm_recsys.p42r3_contract.branch_guard"), \
                 patch("hm_recsys.p42r3_contract.read_json", side_effect=lambda p: reads[Path(p).name]), \
                 patch("hm_recsys.p42r3_contract.write_json", side_effect=lambda p,c: saved.append((Path(p),copy.deepcopy(c)))), \
                 patch("hm_recsys.p42r3_contract.identity", side_effect=lambda p: {"path":str(p), "bytes":1, "sha256":"synthetic"}), \
                 patch("hm_recsys.p42r3_contract.subprocess.check_output", return_value="synthetic-sha"):
                result = preregister(folder)
        self.assertEqual(previous, self.previous)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0].name, "P4_2R3_EXPERIMENT_CONTRACT.json")
        return result

    def test_only_qw_iteration_budget_changes_not_shared_qc_params(self):
        c = self.contract()
        self.assertEqual(c["run_id"], RUN_ID)
        self.assertEqual(c["qW_model_params"], {**MODEL_PARAMS,"max_iter":1200})
        self.assertEqual(c["qW_repair"]["model_params"], QW_MODEL_PARAMS)
        self.assertEqual(c["model"], self.previous["model"])
        self.assertEqual(c["model"]["max_iter"], 1000)
        self.assertEqual(c["qC_reuse"], self.previous["qC_reuse"])
        self.assertEqual(c["qC_reuse"]["new_fits"], 0)

    def test_statistical_preprocessing_pools_and_utility_gates_frozen(self):
        c = self.contract()
        for key in ("population","historical_pools","excluded_cutoffs","feature_spec","prepared_reuse",
                    "epsilon","taus","utility","pairing","calibration","metrics_contract","gates","selection"):
            self.assertEqual(c[key], self.previous[key], key)
        for key in ("R1","R2","all_chains_repair"):
            self.assertEqual(c["qW_repair"][key], self.previous["qW_repair"][key])
        self.assertEqual(c["final_week"], "not_run")

    def test_contract_rejects_any_other_solver_change_or_qc_budget_change(self):
        c = self.contract()
        for change in (lambda x:x["qW_model_params"].update(tol=1e-6),
                       lambda x:x["qW_model_params"].update(max_iter=1500),
                       lambda x:x["model"].update(max_iter=1200),
                       lambda x:x["qC_reuse"].update(new_fits=1),
                       lambda x:x.update(epsilon=1e-5)):
            bad=copy.deepcopy(c); change(bad)
            with self.assertRaises(AssertionError): validate_contract(bad,self.previous)

    def attempts(self):
        return [{"window":w,"side":"qW","repair":"R2","status":"converged",
                 "convergence_warnings":[],"solver_result":{"success":True,"status":0,"nit":1001}}
                for w in WINDOWS]

    def test_runtime_requires_four_actual_solver_successes_not_low_gradient(self):
        rows=self.attempts()
        self.assertTrue(convergence_gate(rows)["pass"])
        for change in (lambda a:a["solver_result"].update(success=False,status=1),
                       lambda a:a["solver_result"].update(nit=1200),
                       lambda a:a.update(convergence_warnings=["warning"]),
                       lambda a:a.update(status="non_converged")):
            bad=copy.deepcopy(rows); change(bad[-1]); bad[-1]["gradient_infinity_norm"]=1e-20
            self.assertFalse(convergence_gate(bad)["pass"])
        with self.assertRaises(ValueError): convergence_gate(rows[:3])
        with self.assertRaises(ValueError): convergence_gate(rows[::-1])

    def test_driver_fit_gate_calibration_gate_admission_chronology(self):
        tree=ast.parse(inspect.getsource(run))
        calls=lambda name: sorted(n.lineno for n in ast.walk(tree)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name)
        for name in ("fit_boundary_propensity","convergence_gate","calibration_gate","evaluate_window"):
            self.assertEqual(len(calls(name)),1,name)
        self.assertLess(calls("fit_boundary_propensity")[0],calls("convergence_gate")[0])
        self.assertLess(calls("convergence_gate")[0],min(calls("load_cutoff")))
        self.assertLess(calls("calibration_gate")[0],calls("evaluate_window")[0])
        self.assertFalse(calls("fit_propensity"))
        self.assertFalse(calls("prepare_cutoff"))
        source=inspect.getsource(run)
        self.assertLess(source.index('raise RepairStop("qW_solver_boundary_repair_failure"'),source.index('state["qW_convergence_gate"] ='))
        self.assertLess(source.index('raise RepairStop("propensity_calibration_failure"'),source.index('result = evaluate_window('))
        self.assertFalse(any(isinstance(n,ast.While) for n in ast.walk(tree)))


if __name__=="__main__": unittest.main()
