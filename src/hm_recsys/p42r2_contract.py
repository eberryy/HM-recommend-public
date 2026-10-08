"""P4.2R2: a new global preprocessing contract, not a retry of P4.2R."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess

from .p41a_contract import identity, read_json, write_json
from .p42 import now
from .p42_contract import WINDOWS, FEATURE_SPEC, TAUS, branch_guard
from .p42_propensity import MODEL_PARAMS
from .p42r_contract import LOG1P_COUNTS

RUN_ID = "p4-2r2-v1-global-heavy-tail-numerical-repair"
FAILURES = ["engineering_failure", "qW_global_R2_convergence_failure", "propensity_calibration_failure"]
FROZEN_KEYS = ("population", "historical_pools", "excluded_cutoffs", "feature_spec", "epsilon",
               "utility", "taus", "pairing", "calibration", "metrics_contract", "gates", "selection")


def validate_contract(c, previous):
    assert c["run_id"] == RUN_ID and c["stage"] == "P4.2R2"
    assert c["status"] == "preregistered_before_formal_computation"
    assert c["windows"] == WINDOWS and c["feature_spec"] == FEATURE_SPEC and c["taus"] == TAUS
    for key in FROZEN_KEYS:
        assert c[key] == previous[key], key
    assert c["qW_repair"]["R1"] == previous["qW_repair"]["R1"]
    assert c["qW_repair"]["R2"]["log1p_columns"] == LOG1P_COUNTS
    assert c["qW_repair"]["R2"]["allowed_only_if"] == "unconditional for every formal qW chain"
    assert c["qW_repair"]["all_chains_repair"] == "R2"
    assert {k: c["model"][k] for k in MODEL_PARAMS} == MODEL_PARAMS
    assert c["qC_reuse"]["new_fits"] == 0
    assert c["qW_convergence_gate"]["required_converged_windows"] == 4
    assert c["qW_convergence_gate"]["failure_decision"] == "qW_global_R2_convergence_failure"
    assert c["final_week"] == "not_run"


def preregister(repo):
    repo = Path(repo).resolve()
    branch_guard(repo)
    report = repo / "reports/phase4"
    target = report / "P4_2R2_EXPERIMENT_CONTRACT.json"
    previous = read_json(report / "P4_2R_EXPERIMENT_CONTRACT.json")
    if target.exists():
        c = read_json(target)
        validate_contract(c, previous)
        return c
    if (report / "P4_2R2_metrics.json").exists():
        raise ValueError("results exist before preregistration")
    prior = read_json(report / "P4_2R_metrics.json")
    assert prior["decision"] == "engineering_failure"
    assert read_json(report / "P4_2R_VERIFICATION.json")["status"] == "pass"
    assert all(prior["training"][w]["qC"]["status"] == "converged" for w in WINDOWS)
    required = ["docs/ROADMAP_PHASE4.zh-CN.md", *["reports/phase4/" + name for name in (
        "P4_2_FINAL.md", "P4_2_EXPERIMENT_CONTRACT.json", "P4_2_OUTPUT_MANIFEST.json",
        "P4_2R_FINAL.md", "P4_2R_EXPERIMENT_CONTRACT.json", "P4_2R_OUTPUT_MANIFEST.json",
        "P4_2R_metrics.json", "P4_2R_VERIFICATION.json", "p4_2r_shared_population_audit.json",
        "p4_2r_qc_population_comparison.json", "p4_2r_qw_R1_cleanup.json",
        "p4_2r_qw_R2_transform.json", "p4_2r_convergence_audit.json")]]
    c = deepcopy(previous)
    c.update({"schema_version": "p4.2r2-contract-v1", "stage": "P4.2R2", "run_id": RUN_ID,
        "created_at_utc": now(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "authoritative_inputs": {p: identity(repo / p) for p in required},
        "prior_P4_2R_manifest": identity(report / "P4_2R_OUTPUT_MANIFEST.json"),
        "prior_P4_2R_metrics": identity(report / "P4_2R_metrics.json"),
        "prior_run_root": str(repo / "artifacts/phase4" / previous["run_id"]),
        "qC_reuse": {"new_fits": 0, "policy": "reuse exact verified P4.2R model/preprocessing and common H_t assets, no new cohorts/features/fit",
                     "windows": {w: deepcopy(prior["training"][w]["qC"]) for w in WINDOWS}},
        "prepared_reuse": deepcopy(prior["prepared"]),
        "qW_convergence_gate": {"required_converged_windows": 4, "order": list(WINDOWS),
            "failure_decision": "qW_global_R2_convergence_failure", "failure_action": "stop immediately; no retries, outer predictions, calibration, or admission",
            "pass_before": "any outer propensity prediction or calibration", "status_definition": "same P4.2R finite parameters and absence of sklearn ConvergenceWarning"},
        "budget": {"estimated_compute_and_verify_minutes": [5, 15], "formal_soft_stop_seconds": 1800,
            "gpu": "none", "fitting": "CPU,4 BLAS threads", "maximum_formal_propensity_fit_attempts": 4,
            "qC_fit_attempts": 0, "candidate_generation": False, "upstream_training": False,
            "fallback": "W0; stop after closure, no P4.3/Warm-v2/final week", "no_sweeps": True},
        "repair_failure_states": FAILURES,
    })
    c["qW_repair"]["R2"]["allowed_only_if"] = "unconditional for every formal qW chain"
    c["qW_repair"].update({"all_chains_repair": "R2", "branch": "all qW use R1 cleanup plus global R2; any nonconvergence stops immediately",
        "attribution_boundary": "same P4.2R rows, labels, R1 cleanup, fixed solver and regularization; only count transform changes. Geometry comparison does not identify a unique causal feature."})
    c["qW_repair"].pop("winter_both_fail", None)
    c.pop("old_qC_control", None)
    c["model"]["fits"] = "four qW global R2 fits only; qC exact P4.2R reuse; no retry"
    c["feature_contract"].update({"stage": "P4.2R2", "run_id": RUN_ID,
        "preprocessing": "qC exact P4.2R reuse; qW training-only median -> R1 cleanup -> log1p retained fixed19 counts -> training-only numeric StandardScaler"})
    c["decisions"]["precedence"] = FAILURES + ["passing_tau_promotion", "pareto_frontier_supported_but_no_safe_operating_point", "warm_risk_uncontrolled", "cold_gain_not_recovered"]
    validate_contract(c, previous)
    write_json(target, c)
    return c


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    print(preregister(p.parse_args().repo)["created_at_utc"])
