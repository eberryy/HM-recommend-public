"""Preregister the sole qW optimization-budget change, 1000 to 1200."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess

from .p41a_contract import identity, read_json, write_json
from .p42 import now
from .p42_contract import WINDOWS, FEATURE_SPEC, TAUS, branch_guard
from .p42_propensity import MODEL_PARAMS
from .p42r_contract import LOG1P_COUNTS
from .p42r2_contract import FROZEN_KEYS

RUN_ID = "p4-2r3-v1-lbfgs-solver-boundary-repair"
QW_MODEL_PARAMS = {**MODEL_PARAMS, "max_iter": 1200}
FAILURES = ["engineering_failure", "qW_solver_boundary_repair_failure", "propensity_calibration_failure"]


def validate_contract(c, previous):
    assert c["stage"] == "P4.2R3" and c["run_id"] == RUN_ID
    assert c["status"] == "preregistered_before_formal_computation"
    assert c["windows"] == WINDOWS and c["taus"] == TAUS and c["feature_spec"] == FEATURE_SPEC
    for key in (*FROZEN_KEYS, "qC_reuse", "prepared_reuse", "model", "software_versions", "budget"):
        assert c[key] == previous[key], key
    for key in ("R1", "R2", "all_chains_repair"):
        assert c["qW_repair"][key] == previous["qW_repair"][key], key
    assert c["qW_model_params"] == QW_MODEL_PARAMS == c["qW_repair"]["model_params"]
    assert {k: v for k, v in c["qW_model_params"].items() if k != "max_iter"} == {k: v for k, v in MODEL_PARAMS.items() if k != "max_iter"}
    assert c["model"]["max_iter"] == 1000 and c["qC_reuse"]["new_fits"] == 0
    assert c["qW_repair"]["R2"]["log1p_columns"] == LOG1P_COUNTS
    assert c["qW_convergence_gate"]["failure_decision"] == "qW_solver_boundary_repair_failure"
    assert c["qW_convergence_gate"]["required_converged_windows"] == 4
    assert c["final_week"] == "not_run"


def preregister(repo):
    repo = Path(repo).resolve()
    branch_guard(repo)
    report = repo / "reports/phase4"
    target = report / "P4_2R3_EXPERIMENT_CONTRACT.json"
    previous = read_json(report / "P4_2R2_EXPERIMENT_CONTRACT.json")
    if target.exists():
        c = read_json(target); validate_contract(c, previous); return c
    if (report / "P4_2R3_metrics.json").exists():
        raise ValueError("results exist before preregistration")
    assert read_json(report / "P4_2R2_metrics.json")["decision"] == "qW_global_R2_convergence_failure"
    assert read_json(report / "P4_2R2_VERIFICATION.json")["status"] == "pass"
    required = ["docs/ROADMAP_PHASE4.zh-CN.md", *["reports/phase4/" + p for p in (
        "P4_2_FINAL.md", "P4_2R_FINAL.md", "P4_2R2_FINAL.md",
        "P4_2_EXPERIMENT_CONTRACT.json", "P4_2R_EXPERIMENT_CONTRACT.json", "P4_2R2_EXPERIMENT_CONTRACT.json",
        "P4_2_OUTPUT_MANIFEST.json", "P4_2R_OUTPUT_MANIFEST.json", "P4_2R2_OUTPUT_MANIFEST.json",
        "P4_2R2_metrics.json", "P4_2R2_VERIFICATION.json", "p4_2r2_qw_global_R2_transform.json",
        "p4_2r2_convergence_audit.json", "p4_2r2_r1_vs_r2_numerics.json", "p4_2r2_solver_stop_audit.json")]]
    c = deepcopy(previous)
    c.update({"schema_version": "p4.2r3-contract-v1", "stage": "P4.2R3", "run_id": RUN_ID,
        "created_at_utc": now(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "authoritative_inputs": {p: identity(repo / p) for p in required},
        "prior_P4_2R2_manifest": identity(report / "P4_2R2_OUTPUT_MANIFEST.json"),
        "prior_P4_2R2_metrics": identity(report / "P4_2R2_metrics.json"),
        "prior_P4_2R2_run_root": str(repo / "artifacts/phase4" / previous["run_id"]),
        "qW_model_params": QW_MODEL_PARAMS,
        "only_authorized_training_change": "qW max_iter 1000 ->1200, all four chains refit from unchanged initialization; qC exact original reuse with max_iter1000",
        "solver_capture": "scoped read-only wrapper of sklearn.linear_model._logistic._check_optimize_result records actual SciPy OptimizeResult, invokes original unchanged, and restores on exit; no optimizer/gradient/iteration/callback change",
        "boundary_diagnostics": "compare each new qW with its P4.2R2 saved model on identical full training rows; record exact early-window parity, coefficients/intercept/log-loss and prediction deltas; diagnostic only, never model selection",
        "repair_failure_states": FAILURES})
    c["qW_repair"]["model_params"] = dict(QW_MODEL_PARAMS)
    c["qW_repair"]["attribution_boundary"] = "same rows, labels, preprocessing, fixed C/tol/solver; only qW max_iter differs; assess actual solver result and parameter/prediction deltas without assuming additional iteration count"
    c["qW_convergence_gate"].update({"failure_decision": "qW_solver_boundary_repair_failure",
        "status_definition": "actual solver success true and status0, no ConvergenceWarning, finite parameters, n_iter strictly below1200; independent gradient cannot override solver failure",
        "failure_action": "any non-success or n_iter>=1200 immediately stops, no retries or downstream calibration/admission"})
    c["feature_contract"].update({"stage": "P4.2R3", "run_id": RUN_ID})
    c["decisions"]["precedence"] = FAILURES + ["passing_tau_promotion", "pareto_frontier_supported_but_no_safe_operating_point", "warm_risk_uncontrolled", "cold_gain_not_recovered"]
    validate_contract(c, previous)
    write_json(target, c)
    return c


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--repo", default=".")
    print(preregister(p.parse_args().repo)["created_at_utc"])
