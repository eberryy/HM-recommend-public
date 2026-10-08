"""FINAL-E8: full-cohort real-Cold supervision transferred to final Cold50."""
from pathlib import Path
import gc
import json
import math
import shutil
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import stable_percentile, segment_metrics
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import DATES, VALID, WINDOWS
from .final_oracle_audit import dump
from .final_relation_pilot import identify
from .p42f_core import batches
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN=Path("artifacts/final/full-cold-e8-v2")
REPORT=Path("reports/final")
P37=Path("artifacts/phase3/phase3-p3.7b-v1-time-aware-hybrid")
COARSE=Path("artifacts/phase3/phase3-v1-candidate-aware-history/coarse-v1")
PREPARED=Path("artifacts/final/integration-10pct-v1/prepared")
REL=Path("artifacts/final/candidate-e2-v1")
REL_LATE=Path("artifacts/final/binary-admission-e5b-v1/relations")
HIST_ACTION=Path("artifacts/final/relation-pilot-v1")
DEV_ACTION=Path("artifacts/final/integration-10pct-v1/models")
# P3.7B only materialized full-cohort training assets for these cutoffs.  The
# 2020-03-18 final-integration cutoff has no corresponding full P3.7B asset and
# therefore must not be silently treated as an available training source.
FULL_DATES=["2019-11-27","2019-12-25","2020-01-22","2020-02-19","2020-04-29","2020-05-27","2020-06-24","2020-07-22"]
FEATURES=[f"m4_{i}" for i in range(24)]+["history_count_0_7","history_count_8_28","history_count_29_84","history_count_over_84","days_since_last_purchase","recent_0_28_purchase_share"]+["m4_coarse_score","m4_coarse_rank","interaction_count_before_cutoff","strict_cold_flag","sparse1_5_flag"]
ALPHAS=(.05,.10,.25,.50)
QUOTAS=(.001,.005,.01,.02)
MODES=("b0_best","classifier_best","rrf025_best")
PARAMS=dict(objective="binary",learning_rate=.05,n_estimators=400,num_leaves=31,max_depth=7,min_child_samples=100,
            subsample=1.,colsample_bytree=1.,reg_lambda=1.,reg_alpha=0.,random_state=20260914,n_jobs=4,
            verbosity=-1,deterministic=True,force_col_wise=True)


def earlier_full(cutoff):
    from datetime import date,timedelta
    return [t for t in FULL_DATES if date.fromisoformat(t)+timedelta(days=7)<date.fromisoformat(cutoff)]


def full_paths(repo,cutoff):
    feature=repo/P37/"features-v1"/"training"/cutoff
    coarse=repo/COARSE/"training"/cutoff/"candidates_top200.npz"
    return feature,coarse


def sampled_full(repo,cutoff):
    feature,coarse=full_paths(repo,cutoff)
    with np.load(coarse,mmap_mode="r") as c:
        user=c["user_index"];item=c["catalog_row"];target=c["target"].astype(np.int8)
        seed=np.uint64(int(cutoff.replace("-","")));mix=(user.astype(np.uint64)*np.uint64(11400714819323198485))^(item.astype(np.uint64)*np.uint64(14029467366897019727))^seed
        keep=(target>0)|((mix%np.uint64(100))==0);u=user[keep];y=target[keep]
    rel=np.load(feature/"m4_relation.float32.npy",mmap_mode="r")[keep].reshape(-1,24)
    state=np.load(feature/"user_state.float32.npy",mmap_mode="r")[u,:6]
    candidate=np.load(feature/"candidate_state.float32.npy",mmap_mode="r")[keep]
    x=np.concatenate([rel,state,candidate],axis=1).astype(np.float32);x[~np.isfinite(x)]=np.nan
    return x,y,dict(cutoff=cutoff,full_rows=len(target),retained_rows=len(y),positive_rows=int(y.sum()),negative_sampling="1% stable row mix")


def fit(repo,cutoff,folder):
    xs=[];ys=[];audit=[]
    for t in earlier_full(cutoff):x,y,a=sampled_full(repo,t);xs.append(x);ys.append(y);audit.append(a)
    x=np.concatenate(xs);y=np.concatenate(ys);model=lgb.LGBMClassifier(**PARAMS)
    with threadpool_limits(limits=4):model.fit(x,y,feature_name=FEATURES)
    model.booster_.save_model(str(folder/f"{cutoff}-F35.txt"));info=dict(rows=len(y),positives=int(y.sum()),training=earlier_full(cutoff),sources=audit)
    del x,y,xs,ys;gc.collect();return model,info


def relation_path(repo,cutoff):
    p=repo/REL/cutoff/"m4-relation.npy"
    return p if p.exists() else repo/REL_LATE/cutoff/"m4-relation.npy"


def final_matrix(repo,cutoff,data):
    cold=data["cold"];rel=np.load(relation_path(repo,cutoff));state=cold[["history_count_0_7","history_count_8_28","history_count_29_84","history_count_over_84","days_since_last_purchase","recent_0_28_purchase_share"]].to_numpy(float)
    candidate=cold[["m4_coarse_score","m4_coarse_rank","interaction_count_before_cutoff","strict_cold_flag","sparse1_5_flag"]].to_numpy(float)
    x=np.concatenate([rel,state,candidate],axis=1).astype(np.float32);x[~np.isfinite(x)]=np.nan;assert x.shape==(len(cold),len(FEATURES));return x


def ranks_for(data,scores):
    rank=np.zeros(len(scores),np.int32)
    for idx in data["cold"].groupby("user_index",sort=False).indices.values():
        idx=np.asarray(idx);order=np.argsort(-scores[idx],kind="stable");rank[idx[order]]=np.arange(1,len(idx)+1)
    return rank


def rrf(data,scores,alpha):return 1/(60+data["cold"].b0_rank.to_numpy(float))+alpha/(60+ranks_for(data,scores))


def action_scores(repo,cutoff,data,historical):
    if not historical:
        w=next(w for w,t in WINDOWS.items() if t==cutoff);return np.load(repo/DEV_ACTION/w/"scores.npy")
    model=lgb.Booster(model_file=str(repo/HIST_ACTION/cutoff/"base70.txt"));out=np.empty((len(data["cold"]),12),float)
    for lo,cc,x,_ in batches(data,labels=False):out[lo:lo+len(cc)]=model.predict(x,num_threads=4).reshape(-1,12)
    return out


def select_candidate(data,score,mode):
    if mode=="b0_best":order=-data["cold"].b0_rank.to_numpy(float)
    elif mode=="classifier_best":order=score
    else:order=rrf(data,score,.25)
    return one_per_user(data,order)


def evaluate(data,score,actions,mode,quota):
    chosen=select_candidate(data,score,mode);admission=stable_percentile(score[chosen]);count=max(1,int(math.ceil(quota*len(chosen))));customer=data["cold"].iloc[chosen].customer_id.to_numpy(str);selected=np.lexsort((customer,-admission))[:count]
    lists=data["warm_lists"].copy();events=[]
    for local in selected:
        ci=int(chosen[local]);row=data["cold"].iloc[ci];ui=int(row.user_index);slot=int(np.argmax(actions[ci]));victim=lists[ui,slot];lists[ui,slot]=row.article_id
        delta=exact_ap(lists[ui],data["truthsets"][data["users"][ui]])-data["baseline_ap"][ui]
        events.append(dict(customer_id=data["users"][ui],candidate=row.article_id,victim=victim,slot=slot+1,delta=float(delta),inserted=int(row.target),removed=int(victim in data["truthsets"][data["users"][ui]]),score=float(score[ci])))
    e=pd.DataFrame(events);return dict(delta_map=float(e.delta.sum()/len(data["users"])),admissions=len(e),beneficial=int(e.delta.gt(0).sum()),harmful=int(e.delta.lt(0).sum()),neutral=int(e.delta.eq(0).sum()),inserted=int(e.inserted.sum()),removed=int(e.removed.sum())),lists,e


def register(repo):
    c=dict(status="preregistered_before_E8_v2_fit",stage="FINAL-E8-v2 full-cohort real-Cold classifier",
      evidence="Pseudo-cold candidate/action models failed development transfer. P3.7B already materialized million-row actual Cold Top200 training assets with hundreds-to-thousands of positives, enabling scale-up without server recomputation.",
      recovery_note="E8-v1 stopped after the first historical window because 2020-03-18 was incorrectly assumed to have a full P3.7B asset. V2 preserves that failure, uses only verified available full-cohort cutoffs, and keeps all model/rule hyperparameters unchanged.",
      training_population="all P3.7B actual Cold/sparse Top200 rows at strictly earlier historical cutoffs; every positive plus deterministic1% negatives",
      compatibility="use shared24 M4 relation + first6 temporal user state +5 candidate state fields; fields have identical definitions in full P3.7B and final Cold50",
      features=FEATURES,params=PARAMS,training_dates=FULL_DATES,historical_validation=VALID,development_windows=WINDOWS,
      candidates={"b0_best":"freeze best B0 item","classifier_best":"classifier chooses among Cold50","rrf025_best":"B0/F35 rank fusion alpha0.25"},
      candidate_diagnostics="raw F35 and RRF alpha0.05/0.10/0.25/0.50 against B0 Recall@1/@5",
      admission="rank candidate-bearing users by F35 score of selected item; top0.1%,0.5%,1%,2%; original action model chooses position",
      historical_selection="highest mean exact delta_MAP; tie fewer removals/admissions, b0_best",historical_gate="mean>0, >=2/3 nonnegative, inserted>=2 and inserted>=removed",
      development_gate="existing expansion gate unchanged",if_pass="fullscale allowed",if_fail="WV3 retained; stop Cold scale-up route",
      final_week="not_run",no_commit_push=True,budget=dict(max_seconds=10800,threads=4,min_disk_gib=15),resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30),git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    dump(repo/REPORT/"FULL_COLD_E8_V2_CONTRACT.json",c);return c


def run(repo):
    repo=Path(repo)
    if (repo/RUN).exists():raise FileExistsError("Do not overwrite E8 v2")
    (repo/RUN).mkdir(parents=True);contract=register(repo);start=time.perf_counter();historical={};aggregate={}
    for cutoff in VALID:
        data=joblib.load(repo/PREPARED/cutoff/"data.joblib");model,fit_info=fit(repo,cutoff,repo/RUN);x=final_matrix(repo,cutoff,data);score=model.predict_proba(x,num_threads=4)[:,1];np.save(repo/RUN/f"{cutoff}-F35-scores.npy",score);actions=action_scores(repo,cutoff,data,True)
        historical[cutoff]={"fit":fit_info,"B0":identify(data,-data["cold"].b0_rank.to_numpy(float)),"F35":identify(data,score),"rules":{}}
        for alpha in ALPHAS:historical[cutoff][f"F35-rrf-{alpha:g}"]=identify(data,rrf(data,score,alpha))
        for mode in MODES:
            for quota in QUOTAS:
                key=f"{mode}|q={quota:g}";r,_,_=evaluate(data,score,actions,mode,quota);historical[cutoff]["rules"][key]=r
        print("E8",cutoff,historical[cutoff]["F35"],flush=True);del model,data,x,score,actions;gc.collect()
    for mode in MODES:
        for quota in QUOTAS:
            key=f"{mode}|q={quota:g}";rows=[historical[t]["rules"][key] for t in VALID];aggregate[key]=dict(mean_delta=float(np.mean([r["delta_map"] for r in rows])),nonnegative=sum(r["delta_map"]>=0 for r in rows),admissions=sum(r["admissions"] for r in rows),inserted=sum(r["inserted"] for r in rows),removed=sum(r["removed"] for r in rows),beneficial=sum(r["beneficial"] for r in rows),harmful=sum(r["harmful"] for r in rows),mode=mode,quota=quota)
    eligible=[(k,v) for k,v in aggregate.items() if v["mean_delta"]>0 and v["nonnegative"]>=2 and v["inserted"]>=2 and v["inserted"]>=v["removed"]]
    eligible.sort(key=lambda kv:(-kv[1]["mean_delta"],kv[1]["removed"],kv[1]["admissions"],MODES.index(kv[1]["mode"])));selected=eligible[0][0] if eligible else None
    development={};gates={"not_run":"historical gate failed"};passed=False
    if selected:
        spec=aggregate[selected]
        for window,cutoff in WINDOWS.items():
            data=joblib.load(repo/PREPARED/cutoff/"data.joblib");model,fit_info=fit(repo,cutoff,repo/RUN);x=final_matrix(repo,cutoff,data);score=model.predict_proba(x,num_threads=4)[:,1];actions=action_scores(repo,cutoff,data,False);result,lists,events=evaluate(data,score,actions,spec["mode"],spec["quota"]);result["segments"]=segment_metrics(data,lists);result["fit"]=fit_info;development[window]=result;save_frame(events,repo/RUN/f"{window}-events.parquet");del data,model,x,score,actions,lists,events;gc.collect()
        delta=np.array([r["delta_map"] for r in development.values()]);warm=np.mean([r["segments"]["warm_21_plus"] for r in development.values()]);cold=np.mean([r["segments"]["all_cold_sparse"] for r in development.values()]);ins=sum(r["inserted"] for r in development.values());rem=sum(r["removed"] for r in development.values())
        gates=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),worst=bool(delta.min()>=-5e-5),warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),inserted_positive=bool(ins>=1),efficiency=bool(ins>=rem));passed=all(gates.values())
    out=dict(status="completed",stage=contract["stage"],historical=historical,historical_aggregate=aggregate,selected=selected,historical_gate_pass=selected is not None,development=development,development_gates=gates,fullscale_allowed=passed,fullscale_status="not_run",fallback="challenger" if passed else "WV3-741",final_week="not_run",seconds=time.perf_counter()-start)
    dump(repo/REPORT/"FULL_COLD_E8_V2.json",out);print(json.dumps(dict(selected=selected,selected_historical=aggregate.get(selected),development=development,gates=gates,fullscale_allowed=passed,seconds=out["seconds"]),indent=2),flush=True)


if __name__=="__main__":
    try:run(Path.cwd())
    except Exception:
        dump(REPORT/f"FULL_COLD_E8_V2_FAILURE_{time.time_ns()}.json",dict(error=traceback.format_exc(),final_week="not_run"));raise
