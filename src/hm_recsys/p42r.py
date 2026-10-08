"""P4.2R execution: lazy repair fits, calibration gate, unchanged admission."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from .p41a_contract import check_identity, identity, read_json, write_json
from .p41a_data import parquet, save_frame
from .p42 import now, preflight, peak_memory_gib, artifact_record
from .p42_contract import branch_guard, guard_cutoff, WINDOWS, TAUS, FEATURE_SPEC
from .p42_data import load_cutoff
from .p42_propensity import fit_propensity, predict_propensity, propensity_calibration
from .p42r_contract import preregister, RUN_ID, LOG1P_COUNTS


class RepairStop(RuntimeError):
    def __init__(self, decision, reason):
        super().__init__(reason)
        self.decision = decision


def calibration_gate(windows):
    if set(windows) != set(WINDOWS):
        raise ValueError("calibration gate requires all four windows")
    counts = {}
    for side in ("qC", "qW"):
        rows = [windows[w][side] for w in WINDOWS]
        counts[side] = {
            "extreme_rate_ratio_windows": sum(r["predicted_to_observed_rate_ratio"] is not None and
                not .1 <= r["predicted_to_observed_rate_ratio"] <= 10 for r in rows),
            "roc_not_above_chance_windows": sum(r["roc_auc"] is not None and r["roc_auc"] <= .5 for r in rows),
            "constant_prediction_windows": sum("constant_prediction" in r.get("warnings", []) for r in rows),
        }
    return {"pass": not any(v >= 3 for row in counts.values() for v in row.values()),
            "counts": counts, "scope": "primary common H_t candidate populations",
            "checked_at_utc": now(), "checked_before_any_admission": True}


def preserve_original(c, root):
    """Compare old failures to their existing trusted manifest, never replace."""
    check_identity(c["original_P4_2_manifest"])
    old = read_json(Path(c["original_P4_2_manifest"]["path"]))
    records = [r for group in ("reports", "artifacts", "sources") for r in old[group]]
    for r in records:
        check_identity(r)
    receipt = {"checked_at_utc": now(), "files": len(records), "all_match": True,
               "trusted_manifest": c["original_P4_2_manifest"], "historical_decision": "engineering_failure"}
    write_json(root / "original-P4.2-preservation.json", receipt)
    return receipt


def save_diagnostics(report, state):
    common = {"stage": "P4.2R", "run_id": RUN_ID, "final_week": "not_run"}
    for file, value in (
        ("shared_population_audit", {"cutoffs": state["population"], "scope": "approved H_t fitted support, full S_t outer evaluation"}),
        ("qc_population_comparison", {"cutoffs": state["comparison"], "old_qC_outer_control": state["old_control"]}),
        ("qw_R1_cleanup", {"attempts": [a for a in state["attempts"] if a["side"] == "qW" and a["repair"] == "R1"]}),
        ("qw_R2_transform", {"exact_count_list": LOG1P_COUNTS, "attempts": [a for a in state["attempts"] if a["side"] == "qW" and a["repair"] == "R2"]}),
        ("convergence_audit", {"attempts": state["attempts"], "selected_qW_repair": state["selected_qW_repair"]}),
        ("propensity_calibration", {"windows": state["calibration"], "full_S_t_qW": state["full_calibration"], "gate": state["calibration_gate"], "old_qC_control": state["old_control"]}),
        ("pair_utility_audit", {"windows": {w: x["pair_audit"] for w, x in state["windows"].items()}}),
        ("matching_audit", {"windows": {w: x["matching"] for w, x in state["windows"].items()}}),
        ("admission_risk", {"windows": {w: {v: {k: row[k] for k in ("admission", "admission_buckets")} for v, row in x["variants"].items()} for w, x in state["windows"].items()}}),
    ):
        observations = value.get("windows", value.get("attempts", value.get("cutoffs", {})))
        write_json(report / f"p4_2r_{file}.json", {**common, "status": "measured" if observations else "not_run", **value})


def run(repo, device="cuda", resume_bootstrap=False):
    from .p42r_data import prepare_cutoff
    from .p42r_propensity import fit_repaired_propensity, predict_repaired_propensity, R2_COUNT_FEATURES
    from .p42_evaluate import evaluate_window, aggregate_windows
    repo = Path(repo).resolve()
    c = preregister(repo)
    branch_guard(repo)
    assert list(R2_COUNT_FEATURES) == c["qW_repair"]["R2"]["log1p_columns"] == LOG1P_COUNTS
    report = repo / "reports/phase4"
    target = report / "P4_2R_metrics.json"
    root = repo / "artifacts/phase4" / RUN_ID
    if target.exists() or (root / "EXECUTION_START.json").exists():
        prior = read_json(target) if target.exists() else None
        if not (resume_bootstrap and prior and prior["decision"] == "engineering_failure"
                and not prior["attempts"] and not prior["prepared"] and not (root / "prepared").exists()):
            raise ValueError("P4.2R already attempted: cannot retry numerical or candidate-computation outcomes")
        archive = root / "bootstrap-attempt-01"
        archive.mkdir(exist_ok=True)
        if (archive / "EXECUTION_START.json").exists():
            raise ValueError("bootstrap failure already archived; no repeated restart loop")
        for file in (root / "EXECUTION_START.json", root / "FAILURE.json", target):
            shutil.copy2(file, archive / file.name)
        for file in report.glob("p4_2r_*.json"):
            shutil.copy2(file, archive / file.name)
        # Restore only the prior preflight audit authority that the empty
        # bootstrap closure had replaced; the empty failure is archived above.
        shutil.copy2(root / "PRIOR_PREFLIGHT_AUDIT.json", report / "p4_2r_shared_population_audit.json")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + c["budget"]["formal_soft_stop_seconds"]
    execution = {"stage": "P4.2R", "run_id": RUN_ID, "started_at_utc": now(),
        "contract": identity(report / "P4_2R_EXPERIMENT_CONTRACT.json"),
        "implementation_before_formal": [identity(p) for p in sorted((repo / "src/hm_recsys").glob("p42r*.py"))
            if not p.stem.endswith(("_verify", "_report"))],
        "final_week": "not_run"}
    write_json(root / "EXECUTION_START.json", execution)
    state = {"prepared": {}, "population": {}, "comparison": {}, "attempts": [], "training": {},
        "selected_qW_repair": None, "calibration": {}, "full_calibration": {}, "calibration_gate": None,
        "old_control": {}, "windows": {}, "artifacts": [], "execution": execution}
    decision, reason, combined = "engineering_failure", None, {}

    def budget():
        if time.perf_counter() > deadline:
            raise RepairStop("engineering_failure", "preregistered formal compute budget exhausted")

    def train_one(window, side, repair, frame, cutoffs):
        budget()
        print(f"fit {window}/{side}/{repair}: {len(frame)} rows, {int(frame.target.sum())} positives", flush=True)
        spec = FEATURE_SPEC[side]
        start = now()
        try:
            result = (fit_propensity(frame, spec) if side == "qC" else
                      fit_repaired_propensity(frame, spec, repair=repair))
        except Exception as exc:
            fail = "qC_population_repair_failure" if side == "qC" else "engineering_failure"
            raise RepairStop(fail, f"{window}/{side}/{repair}: {exc}") from exc
        lineage = {"side": side, "window": window, "outer_cutoff": WINDOWS[window],
            "repair": repair, "training_cutoffs": cutoffs,
            "latest_training_label_end": max(state["prepared"][t]["features"][side]["label_end"] for t in cutoffs),
            "started_at_utc": start, "finished_at_utc": now(), "train_population": "H_t", "preprocessing_training_only": True}
        folder = root / "models" / window / f"{side}-{repair}"
        write_json(folder / "model.json", {**result["model"], "lineage": lineage})
        write_json(folder / "preprocessing.json", {**result["preprocessing"], "lineage": lineage})
        fitted = {"model": read_json(folder / "model.json"), "preprocessing": read_json(folder / "preprocessing.json")}
        audit = {**result["audit"], **lineage, "cleanup": result.get("cleanup"),
            "diagnostics": result.get("diagnostics"), "model": identity(folder / "model.json"),
            "preprocessing": identity(folder / "preprocessing.json")}
        state["attempts"].append(audit)
        write_json(folder / "training-audit.json", audit)
        # JSON replay with labels removed; does not update preprocessing.
        pred = predict_propensity if side == "qC" else predict_repaired_propensity
        probe = frame.drop(columns="target").iloc[:1000]
        np.testing.assert_array_equal(pred(result, probe), pred(fitted, probe))
        save_diagnostics(report, state)
        return fitted, audit

    try:
        # Snapshot the preceding preflight before publishing the measured audit.
        shutil.copy2(report / "p4_2r_shared_population_audit.json", root / "PRIOR_PREFLIGHT_AUDIT.json")
        state["original_preservation"] = preserve_original(c, root)
        state["input_verification"] = preflight(c, repo, root)
        fitted = {}
        for window, cutoff in WINDOWS.items():
            fitted[window], state["training"][window] = {}, {}
            cutoffs = c["historical_pools"][window]["qC"]
            assert cutoffs == c["historical_pools"][window]["qW"] and "2019-11-27" not in cutoffs
            for t in cutoffs:
                guard_cutoff(t)
                if t in state["prepared"]:
                    continue
                budget()
                print(f"prepare shared historical population {t}", flush=True)
                try:
                    data = prepare_cutoff(repo, c, t, root, device=device)
                except ValueError as exc:
                    if "shared_population_contract_failure" in str(exc):
                        raise RepairStop("shared_population_contract_failure", str(exc)) from exc
                    raise
                h = set(map(str, data["history_users"]))
                if set(data["cold"].customer_id.astype(str)) != h or set(data["warm"].customer_id.astype(str)) != h:
                    raise RepairStop("shared_population_contract_failure", f"actual candidate-row user sets differ at {t}")
                entries = {}
                for side, key in (("qC", "cold"), ("qW", "warm")):
                    spec = FEATURE_SPEC[side]
                    meta = ["customer_id", "article_id", "target"] + (["b0_score_available"] if side == "qC" else [])
                    cols = list(dict.fromkeys(meta + spec["numeric"] + spec["binary"]))
                    frame = data[key][cols].copy()
                    frame["target_cutoff"] = t
                    entry = save_frame(frame, root / "prepared" / t / f"{side}-training-features.parquet", t, {"population": "H_t", "feature_spec": spec})
                    from datetime import date, timedelta
                    entry.update({"users": len(h), "positive_rows": int(frame.target.sum()), "label_end": (date.fromisoformat(t)+timedelta(days=7)).isoformat()})
                    entries[side] = entry
                state["prepared"][t] = {"features": entries, "audit": data["audit"]}
                state["population"][t] = data["audit"]
                state["comparison"][t] = data["comparison"]
                for name, record in data.get("artifacts", {}).items():
                    path = record["path"] if isinstance(record, dict) else record
                    state["artifacts"].append(artifact_record(Path(path), t, {"kind": name, "population": "H_t"}))
                write_json(root / "PREPARED_MANIFEST.json", state["prepared"])
                save_diagnostics(report, state)
                del data, frame
                gc.collect()
            for side in ("qC", "qW"):
                spec = FEATURE_SPEC[side]
                cols = spec["numeric"] + spec["binary"] + ["target"]
                train = pd.concat([parquet(state["prepared"][t]["features"][side]["path"])[cols] for t in cutoffs], ignore_index=True)
                repair = "original" if side == "qC" else state["selected_qW_repair"] or "R1"
                model, audit = train_one(window, side, repair, train, cutoffs)
                if audit["status"] != "converged":
                    if side == "qC":
                        raise RepairStop("qC_population_repair_failure", f"{window} qC failed frozen convergence budget")
                    if window != next(iter(WINDOWS)) or repair != "R1":
                        raise RepairStop("engineering_failure", f"{window} chosen qW repair failed; no per-window alternative")
                    model, audit = train_one(window, side, "R2", train, cutoffs)
                    if audit["status"] != "converged":
                        raise RepairStop("qW_R1_and_R2_convergence_failure", "winter qW R1 and R2 both failed frozen budget")
                    repair = "R2"
                if side == "qW" and state["selected_qW_repair"] is None:
                    state["selected_qW_repair"] = repair
                fitted[window][side] = model
                state["training"][window][side] = audit
                del train
                gc.collect()
        # Stage boundary: all fits complete BEFORE outer calibration, then all
        # calibration complete BEFORE any admission. No MAP-based repair choice.
        for window, cutoff in WINDOWS.items():
            budget()
            print(f"calibrate (no admission yet) {window}", flush=True)
            data = load_cutoff(c, cutoff)
            probs = {"qC": predict_propensity(fitted[window]["qC"], data["cold"]),
                     "qW": predict_repaired_propensity(fitted[window]["qW"], data["warm"])}
            h = set(data["cold"].customer_id.astype(str))
            warm_h = data["warm"].customer_id.astype(str).isin(h).to_numpy()
            state["calibration"][window] = {}
            for side, key in (("qC", "cold"), ("qW", "warm")):
                mask = np.ones(len(data[key]), dtype=bool) if side == "qC" else warm_h
                cal = propensity_calibration(data[key].target.to_numpy()[mask], probs[side][mask])
                cal.update({"training_base_rate": state["training"][window][side]["base_rate"], "user_population": "H_t", "prediction_semantics": c["population"]["estimand"]})
                state["calibration"][window][side] = cal
                frame = data[key].copy()
                frame["propensity"] = probs[side]
                state["artifacts"].append(save_frame(frame, root / "outer" / window / f"{side}-predictions.parquet", cutoff, state["training"][window][side]))
            state["full_calibration"][window] = propensity_calibration(data["warm"].target, probs["qW"])
            state["full_calibration"][window]["user_population"] = "full S_t; additional diagnostic, includes noCold users outside fitted support"
            if window == c["old_qC_control"]["window"]:
                old = {k: read_json(Path(c["old_qC_control"][k])) for k in ("model", "preprocessing")}
                state["old_control"][window] = {"old": propensity_calibration(data["cold"].target, predict_propensity(old, data["cold"])),
                    "new": state["calibration"][window]["qC"], "same_outer_rows": len(data["cold"]), "used_for_utility": False}
            save_diagnostics(report, state)
            del data, probs, frame
            gc.collect()
        state["calibration_gate"] = calibration_gate(state["calibration"])
        save_diagnostics(report, state)
        if not state["calibration_gate"]["pass"]:
            raise RepairStop("propensity_calibration_failure", "primary four-window H_t calibration failed preregistered severe gate; admission not run")
        write_json(root / "CALIBRATION_PASSED_BEFORE_ADMISSION.json", state["calibration_gate"])
        for window, cutoff in WINDOWS.items():
            budget()
            print(f"admission original three thresholds {window}", flush=True)
            data = load_cutoff(c, cutoff)
            probs = {side: parquet(root / "outer" / window / f"{side}-predictions.parquet").propensity.to_numpy() for side in ("qC", "qW")}
            result = evaluate_window(data, probs["qC"], probs["qW"])
            state["windows"][window] = {**result["metrics"], "parity": data["parity"]}
            folder = root / "outer" / window
            for key in ("executed", "eligible_cold", "user_audit"):
                state["artifacts"].append(save_frame(result[key], folder / f"{key}.parquet", cutoff, {"unchanged_P4_2_pairing": True}))
            np.save(folder / "pair-utility.float64.npy", result["utility"], allow_pickle=False)
            state["artifacts"].append(artifact_record(folder / "pair-utility.float64.npy", cutoff,
                {"unchanged_P4_2_utility": True}, len(result["utility"]), result["utility"].shape))
            for variant, lists in result["recommendations"].items():
                frame = pd.DataFrame({"customer_id": np.repeat(data["users"], 12), "article_id": lists.reshape(-1), "rank": np.tile(np.arange(1, 13), len(data["users"]))})
                state["artifacts"].append(save_frame(frame, folder / f"{variant}-top12.parquet", cutoff, {"variant": variant}))
            write_json(folder / "window-metrics.json", state["windows"][window])
            save_diagnostics(report, state)
            del data, result
            gc.collect()
        combined = aggregate_windows(state["windows"], state["calibration"], c)
        decision = combined["decision"]
    except Exception as exc:
        decision = exc.decision if isinstance(exc, RepairStop) else "engineering_failure"
        reason = str(exc)
        failure = {"stage": "P4.2R", "decision": decision, "reason": reason, "failed_at_utc": now(), "traceback": traceback.format_exc(), "final_week": "not_run"}
        write_json(root / "FAILURE.json", failure)
        print(json.dumps(failure, ensure_ascii=False), flush=True)
    finally:
        save_diagnostics(report, state)
        metrics = {"stage": "P4.2R", "run_id": RUN_ID, "status": "measured_pending_independent_verification",
            "finished_at_utc": now(), **state, **combined, "decision": decision, "failure_reason": reason,
            "selected_variant": combined.get("selected_variant", "W0"),
            "resources": {"formal_seconds": time.perf_counter()-started, "peak_process_working_set_gib": peak_memory_gib(),
                "propensity_fit_attempts": len(state["attempts"]), "device": device, "upstream_training": False},
            "final_week": "not_run", "Warm_v2_integrated": False, "P4_3_started": False}
        write_json(target, metrics)
        print(json.dumps({"decision": decision, "reason": reason, "resources": metrics["resources"]}, ensure_ascii=False), flush=True)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".")
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--resume-bootstrap", action="store_true", help="only a single archived error before any candidate computation or fit")
    args = p.parse_args()
    result = run(args.repo, args.device, args.resume_bootstrap)
    if result["decision"] in ("engineering_failure", "shared_population_contract_failure", "qC_population_repair_failure", "qW_R1_and_R2_convergence_failure"):
        raise SystemExit(2)
