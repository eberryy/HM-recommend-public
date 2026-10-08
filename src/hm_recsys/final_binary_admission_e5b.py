"""FINAL-E5B: freeze E5A winner and replay it on four development windows."""
from pathlib import Path
import gc
import json
import shutil
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import evaluate_rule, segment_metrics, stable_percentile
from .final_binary_admission_e5 import PARAMS, pseudo_training, one_per_user
from .final_candidate_e2 import WINDOWS, build_relation, M4COL
from .final_oracle_audit import dump
from .p42f_contract import USER, earlier
from .p42f_data import save_frame
from .p43a_run import guard


RUN = Path("artifacts/final/binary-admission-e5b-v1")
REPORT = Path("reports/final")
PREPARED = Path("artifacts/final/integration-10pct-v1/prepared")
ACTION = Path("artifacts/final/integration-10pct-v1/models")
E2 = Path("artifacts/final/candidate-e2-v1")
FEATURES = M4COL + USER
SELECTED = dict(arm="PB41", candidate="b0_best", quota=0.02, slot_policy="model_any")


def relation_file(repo, cutoff):
    prior = repo / E2 / cutoff / "m4-relation.npy"
    return prior if prior.exists() else repo / RUN / "relations" / cutoff / "m4-relation.npy"


def cold_matrix(repo, cutoff):
    data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
    rel = np.load(relation_file(repo, cutoff))
    idx = data["cold"].user_index.to_numpy(int)
    x = np.concatenate([rel, data["state"][USER].to_numpy(float)[idx]], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (len(data["cold"]), len(FEATURES))
    return data, x


def prepare_missing_relation(repo, cutoff):
    if relation_file(repo, cutoff).exists():
        return "reused E2 direct relation"
    catalog = pd.read_csv(repo / "artifacts/m4/m4-v1-supervised-cold-representation/student-v1/static_catalog/catalog_items.csv",
                          dtype={"article_id": str})
    articles = pd.read_csv(repo / "data/raw/articles.csv", dtype={"article_id": str},
                           usecols=["article_id", "product_type_no", "garment_group_no"]).set_index("article_id")
    aligned = articles.reindex(catalog.article_id)
    if aligned.isna().any().any():
        raise ValueError("catalog attribute alignment failure")
    target_root = repo / RUN / "relations"
    build_relation(repo, target_root, cutoff, catalog, {a:i for i,a in enumerate(catalog.article_id)},
                   aligned.product_type_no.to_numpy(np.int32), aligned.garment_group_no.to_numpy(np.int32),
                   torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return "built once under E5B with cutoff-safe E2 function"


def register(repo):
    source = json.loads((repo / REPORT / "BINARY_ADMISSION_E5A.json").read_text(encoding="utf-8"))
    if not source["admission_gate_pass"] or source["selected_admission"] != "PB41|b0_best|q=0.02|model_any":
        raise ValueError("E5A frozen winner/gate mismatch")
    contract = dict(status="preregistered_before_four_window_E5B_labels_and_fits",
        stage="FINAL-E5B frozen binary-admission four-window replay", source="BINARY_ADMISSION_E5A.json",
        selected=SELECTED, historical_metrics=source["admission_aggregate"][source["selected_admission"]],
        windows=WINDOWS, training={w: earlier(t) for w,t in WINDOWS.items()},
        training_population="pseudo-only E3 materializations at strictly earlier cutoffs; exact E5A PB41 sampling and model params",
        candidate="best available original B0 candidate after Warm overlap; no classifier reranking",
        admission="top2% candidate-bearing users by PB41 binary score on their B0-best candidate, stable within-window percentile",
        position="frozen P4.3A action model chooses best of original positions1-12 for that fixed candidate",
        gates="existing expansion gate: mean delta>0; >=3/4 delta>=-1e-5; worst>=-5e-5; Warm mean>=-2e-5; all-Cold/sparse mean>=0; inserted>=1; inserted>=removed",
        if_pass="fullscale allowed; no new tuning", if_fail="WV3-741 retained and binary/content route stopped",
        final_week="not_run", no_commit_push=True, budget=dict(max_seconds=7200, threads=4, min_disk_gib=15),
        resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30),
        git_sha=subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip())
    dump(repo / REPORT / "BINARY_ADMISSION_E5B_CONTRACT.json", contract)
    return contract


def run(repo):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E5B evidence")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo); started = time.perf_counter(); deadline = time.time()+7200
    relation_receipts = {}
    for cutoff in WINDOWS.values():
        guard(repo, deadline); relation_receipts[cutoff] = prepare_missing_relation(repo, cutoff)
    windows = {}
    for window, cutoff in WINDOWS.items():
        guard(repo, deadline); xs=[]; ys=[]; ws=[]
        for train_cutoff in earlier(cutoff):
            x,y,w = pseudo_training(repo, train_cutoff, "PB41"); xs.append(x);ys.append(y);ws.append(w)
        x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(ws)
        model=lgb.LGBMClassifier(**PARAMS)
        with threadpool_limits(limits=4):model.fit(x,y,sample_weight=weight,feature_name=FEATURES)
        model.booster_.save_model(str(repo/RUN/f"{window}-PB41.txt"))
        fit=dict(rows=len(y),positives=int(y.sum()),train_dates=earlier(cutoff))
        del x,y,weight,xs,ys,ws;gc.collect()
        data,vx=cold_matrix(repo,cutoff);score=model.predict_proba(vx,num_threads=4)[:,1]
        np.save(repo/RUN/f"{window}-PB41-scores.npy",score)
        chosen=one_per_user(data,-data["cold"].b0_rank.to_numpy(float))
        action=np.load(repo/ACTION/window/"scores.npy")
        result,lists,events=evaluate_rule(data,chosen,action,stable_percentile(score[chosen]),0.02,"model_any")
        result["segments"]=segment_metrics(data,lists);result["fit"]=fit
        result["candidate_users"]=len(chosen);result["selected_rule"]=SELECTED
        save_frame(events,repo/RUN/f"{window}-events.parquet")
        windows[window]=result
        print("E5B",window,json.dumps(result),flush=True)
        del model,data,vx,score,action,lists,events;gc.collect()
    delta=np.array([x["delta_map"] for x in windows.values()]);warm=np.mean([x["segments"]["warm_21_plus"] for x in windows.values()])
    cold=np.mean([x["segments"]["all_cold_sparse"] for x in windows.values()]);ins=sum(x["inserted"] for x in windows.values());rem=sum(x["removed"] for x in windows.values())
    gates=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),
               worst=bool(delta.min()>=-5e-5),warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),
               inserted_positive=bool(ins>=1),efficiency=bool(ins>=rem))
    passed=all(gates.values())
    output=dict(status="completed_10pct",stage=contract["stage"],windows=windows,mean_delta=float(delta.mean()),
                mean_warm_delta=float(warm),mean_cold_sparse_delta=float(cold),inserted=ins,removed=rem,
                gates=gates,fullscale_allowed=passed,fullscale_status="not_run",relation_receipts=relation_receipts,
                fallback="challenger" if passed else "WV3-741",final_week="not_run",seconds=time.perf_counter()-started)
    dump(repo/REPORT/"BINARY_ADMISSION_E5B.json",output)
    print(json.dumps(output,indent=2),flush=True)


if __name__=="__main__":
    try:run(Path.cwd())
    except Exception:
        dump(REPORT/f"BINARY_ADMISSION_E5B_FAILURE_{time.time_ns()}.json",dict(error=traceback.format_exc(),final_week="not_run"))
        raise
