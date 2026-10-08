"""Solver-budget boundary repair, then gated frozen admission."""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .p41a_contract import check_identity, identity, read_json, write_json
from .p41a_data import parquet, save_frame
from .p42 import now, preflight, artifact_record
from .p42_contract import WINDOWS, FEATURE_SPEC, branch_guard, guard_cutoff
from .p42_data import load_cutoff
from .p42_propensity import predict_propensity, propensity_calibration
from .p42r import RepairStop, calibration_gate
from .p42r_propensity import predict_repaired_propensity, R2_COUNT_FEATURES
from .p42r3_propensity import fit_boundary_propensity, parameter_delta, QW_MODEL_PARAMS
from .p42r3_contract import RUN_ID, LOG1P_COUNTS, preregister


def preserve_previous(c, root):
    """Trusted prior manifests answer whether historical assets changed."""
    checked = {}
    for key in ("original_P4_2_manifest", "prior_P4_2R_manifest", "prior_P4_2R2_manifest"):
        manifest = c[key]
        check_identity(manifest)
        m = read_json(Path(manifest["path"]))
        for group in ("reports", "artifacts", "sources"):
            for record in m[group]:
                path = record["path"]
                if path in checked:
                    assert checked[path]["sha256"] == record["sha256"], path
                else:
                    check_identity(record)
                    checked[path] = record
    receipt = {"checked_at_utc": now(), "all_match": True, "files": len(checked),
        "historical_decisions": {"P4.2": "engineering_failure", "P4.2R": "engineering_failure", "P4.2R2": "qW_global_R2_convergence_failure"},
        "trusted_manifests": [c["original_P4_2_manifest"], c["prior_P4_2R_manifest"], c["prior_P4_2R2_manifest"]]}
    write_json(root / "prior-preservation.json", receipt)
    return receipt


def convergence_gate(attempts):
    if len(attempts) != 4 or [a["window"] for a in attempts] != list(WINDOWS):
        raise ValueError("four ordered qW attempts required before calibration")
    passed = all(a["side"] == "qW" and a["repair"] == "R2" and a["status"] == "converged"
        and a["solver_result"]["success"] and a["solver_result"]["status"] == 0
        and a["solver_result"]["nit"] < 1200 and not a["convergence_warnings"] for a in attempts)
    return {"pass": passed, "required": 4, "converged": sum(a["status"] == "converged" for a in attempts),
        "checked_at_utc": now(), "checked_before_any_outer_prediction": True}


def boundary_audit(window, frame, old, current):
    """Read-only parameter replay; no early model reuse or optimizer calls."""
    def fitted(a):
        return {**{k: read_json(Path(a[k]["path"])) for k in ("model", "preprocessing")},
                "diagnostics": a["diagnostics"]}
    old_fit, new_fit = fitted(old), fitted(current)
    old_iter = old_fit["model"]["n_iter"][0]
    new_iter = new_fit["model"]["n_iter"][0]
    def clean(p):
        return {k: v for k, v in p.items() if k != "lineage"}
    parity = {
        "preprocessing": clean(old_fit["preprocessing"]) == clean(new_fit["preprocessing"]),
        "coefficients": old_fit["model"]["coefficient"] == new_fit["model"]["coefficient"],
        "intercept": old_fit["model"]["intercept"] == new_fit["model"]["intercept"],
        "n_iter": old_iter == new_iter,
    }
    if not parity["preprocessing"]:
        raise ValueError("R3 preprocessing changed despite max_iter-only authorization")
    return {"old_stage": "P4.2R2", "new_stage": "P4.2R3", "old_iterations": old_iter,
        "new_iterations": new_iter, "additional_iterations_beyond_old_cap": max(new_iter - 1000, 0),
        "parameter_delta": parameter_delta(frame, old_fit, new_fit),
        "old_diagnostics": old["diagnostics"], "new_diagnostics": current["diagnostics"],
        "solver_result": current["solver_result"],
        "early_window_exact_parity": parity,
        "early_window": window != "late_summer_20200819"}


def save_diagnostics(report, state):
    common = {"stage": "P4.2R3", "run_id": RUN_ID, "final_week": "not_run"}
    values = {
        "solver_boundary_audit": {"exact_count_list": LOG1P_COUNTS, "attempts": state["attempts"], "windows": state["boundary_comparison"]},
        "convergence_audit": {"attempts": state["attempts"], "gate": state["qW_convergence_gate"], "qC_reuse": state["qc_reuse"]},
        "late_parameter_delta": {"windows": {w: x for w, x in state["boundary_comparison"].items() if w == "late_summer_20200819"}},
        "propensity_calibration": {"windows": state["calibration"], "full_S_t_qW": state["full_calibration"], "gate": state["calibration_gate"]},
        "pair_utility_audit": {"windows": {w: x["pair_audit"] for w, x in state["windows"].items()}},
        "matching_audit": {"windows": {w: x["matching"] for w, x in state["windows"].items()}},
        "admission_risk": {"windows": {w: {v: {k: x[k] for k in ("admission", "admission_buckets")} for v, x in m["variants"].items()} for w, m in state["windows"].items()}},
        "segment_metrics": {"windows": {w: {v: x["segments"] for v, x in m["variants"].items()} for w, m in state["windows"].items()}},
    }
    for name, value in values.items():
        observed = value.get("windows", value.get("attempts", {}))
        write_json(report / f"p4_2r3_{name}.json", {**common, "status": "measured" if observed else "not_run", **value})


def peak_memory():
    """Windows API with explicit HANDLE signatures and success checking."""
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                *[(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        info = Counters(); info.cb = ctypes.sizeof(info)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb) or info.PeakWorkingSetSize <= 0:
            return None
        return info.PeakWorkingSetSize / 2**30
    except Exception:
        return None


def run(repo):
    from .p42_evaluate import evaluate_window, aggregate_windows
    repo = Path(repo).resolve()
    branch_guard(repo)
    c = preregister(repo)
    report = repo / "reports/phase4"
    target = report / "P4_2R3_metrics.json"
    root = repo / "artifacts/phase4" / RUN_ID
    if target.exists() or (root / "EXECUTION_START.json").exists():
        raise ValueError("P4.2R3 already attempted; no retry or overwrite")
    assert list(R2_COUNT_FEATURES) == LOG1P_COUNTS == c["qW_repair"]["R2"]["log1p_columns"]
    assert c["qW_model_params"] == QW_MODEL_PARAMS
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    execution = {"stage": "P4.2R3", "run_id": RUN_ID, "started_at_utc": now(),
        "contract": identity(report / "P4_2R3_EXPERIMENT_CONTRACT.json"),
        "implementation_before_formal": [identity(p) for p in sorted((repo / "src/hm_recsys").glob("p42*.py")) if not p.stem.endswith(("_report", "_verify"))], "final_week": "not_run"}
    write_json(root / "EXECUTION_START.json", execution)
    state = {"prepared": {}, "population": {}, "qc_reuse": {}, "attempts": [], "training": {},
        "boundary_comparison": {}, "selected_qW_repair": "R2", "qW_convergence_gate": None,
        "calibration": {}, "full_calibration": {}, "calibration_gate": None, "windows": {}, "artifacts": [], "execution": execution}
    decision, reason, combined = "engineering_failure", None, {}

    def budget():
        if time.perf_counter() - started > c["budget"]["formal_soft_stop_seconds"]:
            raise RepairStop("engineering_failure", "preregistered compute budget exhausted")

    try:
        state["original_preservation"] = preserve_previous(c, root)
        state["input_verification"] = preflight(c, repo, root)
        prior = read_json(Path(c["prior_P4_2R_metrics"]["path"]))
        state["prepared"] = c["prepared_reuse"]
        state["population"] = prior["population"]
        write_json(root / "PREPARED_MANIFEST.json", state["prepared"])
        fitted = {}
        for window in WINDOWS:
            a = c["qC_reuse"]["windows"][window]
            for k in ("model", "preprocessing"):
                check_identity(a[k])
            fitted[window] = {"qC": {k: read_json(Path(a[k]["path"])) for k in ("model", "preprocessing")}}
            state["training"][window] = {"qC": a}
            state["qc_reuse"][window] = {"status": "reused_exact", "new_fit": False,
                "model": a["model"], "preprocessing": a["preprocessing"], "training_cutoffs": a["training_cutoffs"],
                "source_stage": "P4.2R", "source_status": a["status"]}
        prior_r2 = read_json(Path(c["prior_P4_2R2_metrics"]["path"]))
        old_qw = {a["window"]: a for a in prior_r2["attempts"] if a["side"] == "qW" and a["repair"] == "R2"}
        spec = FEATURE_SPEC["qW"]
        for window, cutoff in WINDOWS.items():
            budget()
            guard_cutoff(cutoff)
            cutoffs = c["historical_pools"][window]["qW"]
            assert cutoffs == c["historical_pools"][window]["qC"]
            pool = []
            for t in cutoffs:
                guard_cutoff(t)
                e = state["prepared"][t]["features"]["qW"]
                assert e["label_end"] < cutoff
                check_identity(e)
                frame = parquet(e["path"])
                assert frame.target_cutoff.eq(t).all() and len(frame) == e["row_count"]
                assert int(frame.target.sum()) == e["positive_rows"]
                pool.append(frame[spec["numeric"] + spec["binary"] + ["target"]])
            train = pd.concat(pool, ignore_index=True)
            del pool, frame
            print(f"fit {window}/qW/R2-maxiter1200: {len(train)} rows,{int(train.target.sum())} positives", flush=True)
            fit_start = now()
            result = fit_boundary_propensity(train, spec)
            line = {"window": window, "side": "qW", "repair": "R2", "outer_cutoff": cutoff,
                "training_cutoffs": cutoffs, "latest_training_label_end": max(state["prepared"][t]["features"]["qW"]["label_end"] for t in cutoffs),
                "started_at_utc": fit_start, "finished_at_utc": now(), "train_population": "H_t", "preprocessing_training_only": True}
            folder = root / "models" / window / "qW-R2"
            write_json(folder / "model.json", {**result["model"], "lineage": line})
            write_json(folder / "preprocessing.json", {**result["preprocessing"], "lineage": line})
            a = {**result["audit"], **line, "cleanup": result["cleanup"], "diagnostics": result["diagnostics"],
                "model": identity(folder / "model.json"), "preprocessing": identity(folder / "preprocessing.json")}
            state["attempts"].append(a)
            state["training"][window]["qW"] = a
            write_json(folder / "training-audit.json", a)
            fit = {k: read_json(folder / f"{k}.json") for k in ("model", "preprocessing")}
            np.testing.assert_array_equal(predict_repaired_propensity(result, train.drop(columns="target").iloc[:1000]), predict_repaired_propensity(fit, train.drop(columns="target").iloc[:1000]))
            previous = old_qw[window]
            assert previous["cleanup"] == a["cleanup"]
            assert (previous["training_rows"], previous["positives"]) == (a["training_rows"], a["positives"])
            state["boundary_comparison"][window] = boundary_audit(window, train, previous, a)
            fitted[window]["qW"] = fit
            save_diagnostics(report, state)
            print(json.dumps({"window": window, "status": a["status"], "iterations": fit["model"]["n_iter"], "solver": a["solver_result"]["message"]}), flush=True)
            del train, result
            gc.collect()
            if a["status"] != "converged" or not a["solver_result"]["success"] or a["solver_result"]["nit"] >= 1200:
                raise RepairStop("qW_solver_boundary_repair_failure", f"{window} R2 with max_iter1200 did not return solver success below the iteration cap; stop without later chains or calibration")
        state["qW_convergence_gate"] = convergence_gate(state["attempts"])
        assert state["qW_convergence_gate"]["pass"]
        write_json(root / "QW_CONVERGENCE_PASSED_BEFORE_CALIBRATION.json", state["qW_convergence_gate"])
        for window, cutoff in WINDOWS.items():
            budget()
            print(f"calibration only {window}", flush=True)
            data = load_cutoff(c, cutoff)
            probs = {"qC": predict_propensity(fitted[window]["qC"], data["cold"]), "qW": predict_repaired_propensity(fitted[window]["qW"], data["warm"])}
            h = set(data["cold"].customer_id.astype(str))
            mask_w = data["warm"].customer_id.astype(str).isin(h).to_numpy()
            assert set(data["warm"].loc[mask_w, "customer_id"].astype(str)) == h
            state["calibration"][window] = {}
            for side, key in (("qC", "cold"), ("qW", "warm")):
                mask = np.ones(len(data[key]), dtype=bool) if side == "qC" else mask_w
                cal = propensity_calibration(data[key].target.to_numpy()[mask], probs[side][mask])
                cal.update({"training_base_rate": state["training"][window][side]["base_rate"], "user_population": "H_t", "prediction_semantics": c["population"]["estimand"]})
                state["calibration"][window][side] = cal
                frame = data[key].copy(); frame["propensity"] = probs[side]
                state["artifacts"].append(save_frame(frame, root / "outer" / window / f"{side}-predictions.parquet", cutoff, state["training"][window][side]))
            state["full_calibration"][window] = {**propensity_calibration(data["warm"].target, probs["qW"]), "user_population": "full S_t; additional diagnostic only"}
            state["artifacts"].append(save_frame(data["truth"], root / "outer" / window / "truth.parquet", cutoff, {"audit_only": True}))
            save_diagnostics(report, state)
            del data, probs, frame
            gc.collect()
        state["calibration_gate"] = calibration_gate(state["calibration"])
        save_diagnostics(report, state)
        if not state["calibration_gate"]["pass"]:
            raise RepairStop("propensity_calibration_failure", "primary four-window H_t severe calibration gate failed; admission not run")
        write_json(root / "CALIBRATION_PASSED_BEFORE_ADMISSION.json", state["calibration_gate"])
        for window, cutoff in WINDOWS.items():
            budget()
            print(f"admission original three thresholds {window}", flush=True)
            data = load_cutoff(c, cutoff)
            probs = {}
            for side, key in (("qC", "cold"), ("qW", "warm")):
                frame = parquet(root / "outer" / window / f"{side}-predictions.parquet")
                pd.testing.assert_frame_equal(frame[["customer_id", "article_id"]], data[key][["customer_id", "article_id"]], check_dtype=False)
                probs[side] = frame.propensity.to_numpy()
            result = evaluate_window(data, probs["qC"], probs["qW"])
            state["windows"][window] = {**result["metrics"], "parity": data["parity"]}
            folder = root / "outer" / window
            for key in ("executed", "eligible_cold", "user_audit"):
                state["artifacts"].append(save_frame(result[key], folder / f"{key}.parquet", cutoff, {"unchanged_P4_2_pairing": True}))
            np.save(folder / "pair-utility.float64.npy", result["utility"], allow_pickle=False)
            state["artifacts"].append(artifact_record(folder / "pair-utility.float64.npy", cutoff, {"unchanged_P4_2_utility": True}, len(result["utility"]), result["utility"].shape))
            for variant, lists in result["recommendations"].items():
                frame = pd.DataFrame({"customer_id": np.repeat(data["users"], 12), "article_id": lists.reshape(-1), "rank": np.tile(np.arange(1, 13), len(data["users"]))})
                state["artifacts"].append(save_frame(frame, folder / f"{variant}-top12.parquet", cutoff, {"variant": variant}))
            write_json(folder / "window-metrics.json", state["windows"][window])
            save_diagnostics(report, state)
            del data, result, frame
            gc.collect()
        combined = aggregate_windows(state["windows"], state["calibration"], c)
        decision = combined["decision"]
    except Exception as exc:
        decision = exc.decision if isinstance(exc, RepairStop) else "engineering_failure"
        reason = str(exc)
        failure = {"stage": "P4.2R3", "decision": decision, "reason": reason, "failed_at_utc": now(), "traceback": traceback.format_exc(), "final_week": "not_run"}
        write_json(root / "FAILURE.json", failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
    finally:
        save_diagnostics(report, state)
        metrics = {"stage": "P4.2R3", "run_id": RUN_ID, "status": "measured_pending_independent_verification", "finished_at_utc": now(),
            **state, **combined, "decision": decision, "failure_reason": reason, "selected_variant": combined.get("selected_variant", "W0"),
            "resources": {"formal_seconds": time.perf_counter() - started, "peak_process_working_set_gib": peak_memory(),
                "propensity_fit_attempts": len(state["attempts"]), "qC_new_fits": 0, "device": "CPU only", "upstream_training": False},
            "final_week": "not_run", "Warm_v2_integrated": False, "P4_3_started": False}
        write_json(target, metrics)
        print(json.dumps({"decision": decision, "reason": reason, "resources": metrics["resources"]}, ensure_ascii=False), flush=True)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("run", "verify", "report", "preregister"))
    p.add_argument("--repo", default=".")
    args = p.parse_args()
    if args.action == "preregister":
        print(preregister(args.repo)["created_at_utc"])
    elif args.action == "run":
        result = run(args.repo)
        if result["decision"] in ("engineering_failure", "qW_solver_boundary_repair_failure"):
            raise SystemExit(2)
    elif args.action == "verify":
        from .p42r3_verify import verify
        print(json.dumps(verify(Path(args.repo).resolve()), ensure_ascii=False))
    else:
        from .p42r3_report import report
        report(Path(args.repo).resolve())
