"""P4.2 fixed experiment driver; no Warm/Cold upstream training entry points."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import shutil
import subprocess
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .p41a_contract import check_identity, identity, read_json, write_json
from .p41a_data import identities, parquet, save_frame
from .p42_contract import RUN_ID, WINDOWS, TAUS, branch_guard, preregister
from .p42_data import load_cutoff
from .p42_propensity import MODEL_PARAMS, EPSILON, THREAD_LIMIT, fit_propensity, predict_propensity, propensity_calibration


def now():
    return datetime.now(timezone.utc).isoformat()


def guard_runtime(contract, repo):
    branch_guard(repo)
    assert contract["status"] == "preregistered_before_formal_computation"
    assert contract["taus"] == TAUS and contract["epsilon"] == EPSILON
    assert contract["windows"] == WINDOWS and contract["model"]["thread_limit"] == THREAD_LIMIT
    assert {k:contract["model"][k] for k in MODEL_PARAMS} == MODEL_PARAMS
    for name,version in contract["software_versions"].items():
        assert importlib.metadata.version(name) == version, f"dependency drift: {name}"


def artifact_record(path, cutoff, lineage, rows=None, shape=None):
    out = {**identity(path), "cutoff":cutoff, "source_lineage":lineage}
    if rows is not None:
        out["row_count"] = int(rows)
    if shape is not None:
        out["shape"] = list(shape)
    return out


def preflight(contract, repo, root):
    """Explicit integrity contract: compare consumed assets to historical SHA."""
    guard_runtime(contract, repo)
    records = {}
    for tree in (contract["inputs"], contract["transactions"], contract["catalog"]):
        for record in identities(tree):
            records[str(Path(record["path"]).resolve())] = record
    for record in records.values():
        check_identity(record)
    # Original authority documents are snapshotted before the authorized
    # append-only closure update. Their original SHA stays in preregistration.
    authorities = {}
    for i,record in enumerate(contract["authoritative_inputs"].values()):
        check_identity(record)
        snapshot = root / "authority-snapshots" / f"{i:02d}-{Path(record['path']).name}"
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(record["path"], snapshot)
        authorities[record["path"]] = {**record, "snapshot_path":str(snapshot)}
    receipt = {"stage":"P4.2","checked_at_utc":now(),"input_files":len(records),
        "input_bytes":sum(r["bytes"] for r in records.values()),"all_frozen_sha_pass":True,
        "integrity_question":"did migrated/cached frozen candidates, score arrays, model lineage and PIT features retain their trusted prior manifest identity?",
        "trusted_comparisons":list(records.values()),"authority_snapshots":authorities,"final_week":"not_run"}
    write_json(root / "input-verification.json",receipt)
    return receipt


def prepare_inputs(contract, root):
    prepared = {}
    for cutoff in contract["existing_cutoff_sequence"]:
        print(f"prepare frozen cutoff {cutoff}",flush=True)
        data = load_cutoff(contract,cutoff)
        folder = root / "prepared" / cutoff
        entries = {}
        for side,key in (("qC","cold"),("qW","warm")):
            spec = contract["feature_spec"][side]
            meta = ["customer_id","article_id","target","interaction_count_before_cutoff"]
            if side == "qC":
                meta += ["b0_score_available"]
            columns = list(dict.fromkeys(meta + spec["numeric"] + spec["binary"]))
            frame = data[key][columns].copy()
            frame["target_cutoff"] = cutoff
            entry = save_frame(frame,folder / f"{side}-features.parquet",cutoff,{"frozen_input":contract["inputs"][cutoff],"feature_contract":contract["feature_spec"][side]})
            entry.update({"positive_rows":int(frame.target.sum()),"users":int(frame.customer_id.nunique()),
                          "label_end":(date.fromisoformat(cutoff)+timedelta(days=7)).isoformat()})
            entries[side] = entry
        prepared[cutoff] = {"features":entries,"parity":data["parity"]}
        write_json(folder / "PREPARED_MANIFEST.json",prepared[cutoff])
        del data
        gc.collect()
    write_json(root / "PREPARED_MANIFEST.json",prepared)
    return prepared


def fit_models(contract, prepared, root, deadline):
    """Finish all eight fits before evaluating any admission operating point."""
    fitted, training = {}, {}
    for window,outer in WINDOWS.items():
        fitted[window],training[window] = {}, {}
        for side in ("qC","qW"):
            if time.perf_counter() > deadline:
                raise RuntimeError("P4.2 preregistered compute budget exhausted before next fit")
            cutoffs = contract["historical_pools"][window][side]
            assert cutoffs and all(date.fromisoformat(c)+timedelta(days=7) < date.fromisoformat(outer) for c in cutoffs)
            pool = []
            selected = []
            for c in cutoffs:
                entry = prepared[c]["features"][side]
                frame = parquet(entry["path"])
                if side == "qC":
                    # All registered qC cutoffs have safe B0, no row-based
                    # label/score threshold is used to select train rows.
                    assert frame.b0_score_available.eq(1).all()
                assert frame.target_cutoff.eq(c).all()
                selected.append({"cutoff":c,"label_end":entry["label_end"],"rows":len(frame),"positive_rows":int(frame.target.sum()),"artifact":entry})
                pool.append(frame)
            train = pd.concat(pool,ignore_index=True)
            del pool
            spec = contract["feature_spec"][side]
            train = train[[*spec["numeric"],*spec["binary"],"target"]]
            print(f"fit {window} {side}: {len(train)} full rows, {int(train.target.sum())} positives",flush=True)
            started_at = now()
            result = fit_propensity(train,spec)
            end = max(x["label_end"] for x in selected)
            line = {"side":side,"outer_cutoff":outer,"training_cutoffs":cutoffs,"latest_training_label_end":end,
                    "fit_started_at_utc":started_at,"fit_finished_at_utc":now(),"training_only_preprocessing":True}
            folder = root / "models" / window
            model_path,pre_path = folder / f"{side}-model.json",folder / f"{side}-preprocessing.json"
            write_json(model_path,{**result["model"],"lineage":line})
            write_json(pre_path,{**result["preprocessing"],"lineage":line})
            model_entry=artifact_record(model_path,outer,line,len(train))
            pre_entry=artifact_record(pre_path,outer,line,len(train))
            audit={**result["audit"],**line,"cutoff_details":selected,"model":model_entry,"preprocessing":pre_entry}
            write_json(folder / f"{side}-training-audit.json",audit)
            if audit["status"] != "converged":
                raise RuntimeError(f"P4.2 fixed solver did not converge for {window}/{side}; no retry or retune")
            training[window][side]=audit
            fitted[window][side]={"model":read_json(model_path),"preprocessing":read_json(pre_path)}
            # Model/scaler serialized replay uses no target and no fit path.
            probe=train.drop(columns="target").iloc[:min(1000,len(train))]
            np.testing.assert_array_equal(predict_propensity(result,probe),predict_propensity(fitted[window][side],probe))
            del train,result
            gc.collect()
    return fitted,training


def peak_memory_gib():
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                *[(name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]]
        record=Counters(); record.cb=ctypes.sizeof(record)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(record),record.cb)
        return record.PeakWorkingSetSize/2**30
    except Exception:
        return None


def run(repo):
    from .p42_evaluate import evaluate_window,aggregate_windows
    repo=Path(repo).resolve()
    contract=preregister(repo)
    report=repo / "reports/phase4"
    target=report / "P4_2_metrics.json"
    if target.exists():
        raise ValueError("P4.2 measured results already exist; use verify, never overwrite")
    root=repo / "artifacts/phase4" / RUN_ID
    root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    deadline=started+contract["budget"]["formal_soft_stop_seconds"]
    execution={"stage":"P4.2","run_id":RUN_ID,"started_at_utc":now(),"contract":identity(report / "P4_2_EXPERIMENT_CONTRACT.json")}
    source_paths=list((repo / "src/hm_recsys").glob("p42*.py"))
    execution["implementation_before_formal"]=[identity(p) for p in source_paths]
    write_json(root / "EXECUTION_START.json",execution)
    try:
        receipt=preflight(contract,repo,root)
        prepared=prepare_inputs(contract,root)
        fitted,training=fit_models(contract,prepared,root,deadline)
        write_json(report / "p4_2_training_audit.json",{"stage":"P4.2","windows":training,"prepared_cutoffs":prepared,"final_week":"not_run"})
        windows,calibration,artifacts={},{},[]
        for window,cutoff in WINDOWS.items():
            print(f"evaluate fixed three thresholds: {window}",flush=True)
            data=load_cutoff(contract,cutoff)
            probs={side:predict_propensity(fitted[window][side],data[key]) for side,key in (("qC","cold"),("qW","warm"))}
            calibration[window]={side:propensity_calibration(data[key].target,probs[side]) for side,key in (("qC","cold"),("qW","warm"))}
            result=evaluate_window(data,probs["qC"],probs["qW"])
            windows[window]={**result["metrics"],"parity":data["parity"]}
            folder=root / "outer" / window
            line={"cutoff":cutoff,"model":{side:training[window][side]["model"] for side in ("qC","qW")},"preprocessing":{side:training[window][side]["preprocessing"] for side in ("qC","qW")}}
            for side,key in (("qC","cold"),("qW","warm")):
                frame=data[key].copy()
                frame["propensity"]=probs[side]
                artifacts.append(save_frame(frame,folder / f"{side}-predictions.parquet",cutoff,line))
            artifacts.append(save_frame(data["truth"],folder / "truth.parquet",cutoff,{"audit_only":True,"transactions":contract["transactions"]}))
            for key in ("executed","eligible_cold","user_audit"):
                artifacts.append(save_frame(result[key],folder / f"{key}.parquet",cutoff,line))
            upath=folder / "pair-utility.float64.npy"
            np.save(upath,result["utility"],allow_pickle=False)
            artifacts.append(artifact_record(upath,cutoff,line,len(result["utility"]),result["utility"].shape))
            for variant,lists in result["recommendations"].items():
                pred=pd.DataFrame({"customer_id":np.repeat(data["users"],12),"article_id":lists.reshape(-1),"rank":np.tile(np.arange(1,13),len(data["users"]))})
                artifacts.append(save_frame(pred,folder / f"{variant}-top12.parquet",cutoff,line))
            write_json(folder / "window-metrics.json",windows[window])
            del data,result
            gc.collect()
        combined=aggregate_windows(windows,calibration,contract)
        metrics={"stage":"P4.2","run_id":RUN_ID,"status":"measured_pending_independent_verification",
            "created_at_utc":now(),"contract":execution["contract"],"windows":windows,**combined,
            "resources":{"formal_seconds":time.perf_counter()-started,"peak_process_working_set_gib":peak_memory_gib(),"device":"CPU only","model_fits":8},
            "execution":execution,"artifacts":artifacts,"input_integrity":{"files":receipt["input_files"],"bytes":receipt["input_bytes"],"pass":True},
            "historical_boundaries":contract["historical_boundaries"],"final_week":"not_run"}
        write_json(target,metrics)
        write_json(report / "p4_2_propensity_calibration.json",{"stage":"P4.2","windows":calibration,"final_week":"not_run"})
        for filename,key in (("pair_utility_audit","pair_audit"),("matching_audit","matching")):
            write_json(report / f"p4_2_{filename}.json",{"stage":"P4.2","windows":{w:m[key] for w,m in windows.items()},"final_week":"not_run"})
        write_json(report / "p4_2_admission_risk.json",{"stage":"P4.2","windows":{w:{v:{k:x[k] for k in ("admission","admission_buckets")} for v,x in m["variants"].items()} for w,m in windows.items()},"final_week":"not_run"})
        write_json(report / "p4_2_segment_metrics.json",{"stage":"P4.2","windows":{w:{v:x["segments"] for v,x in m["variants"].items()} for w,m in windows.items()},"final_week":"not_run"})
        print(json.dumps({"decision":metrics["decision"],"resources":metrics["resources"]},ensure_ascii=False),flush=True)
        return metrics
    except Exception as exc:
        failure={**execution,"failed_at_utc":now(),"status":"engineering_failure","reason":str(exc),"traceback":traceback.format_exc(),"elapsed_seconds":time.perf_counter()-started,"final_week":"not_run"}
        write_json(root / f"FAILURE-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json",failure)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("run","verify","report"))
    parser.add_argument("--repo",default=".")
    args=parser.parse_args()
    repo=Path(args.repo).resolve()
    if args.action=="run":
        run(repo)
    elif args.action=="verify":
        from .p42_verify import verify
        print(json.dumps(verify(repo),ensure_ascii=False))
    else:
        from .p42_report import report
        report(repo)


if __name__=="__main__":
    main()
