"""FINAL-E6: pseudo-cold augmented joint candidate-victim action utility."""
from pathlib import Path
import gc
import hashlib
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

from .final_b0_admission_e4 import segment_metrics
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import DATES, VALID, WINDOWS, M4COL
from .final_oracle_audit import dump
from .final_relation_pilot import identify
from .p42f_contract import WARM, USER, earlier
from .p42d_stats import single_ap_delta
from .p42f_data import frame, save_frame
from .p43a_policy import exact_ap
from .p43a_run import guard


RUN=Path("artifacts/final/pseudo-action-e6-v1")
REPORT=Path("reports/final")
PREPARED=Path("artifacts/final/integration-10pct-v1/prepared")
E2=Path("artifacts/final/candidate-e2-v1")
E3=Path("artifacts/final/pseudocold-e3-v1")
E5B=Path("artifacts/final/binary-admission-e5b-v1")
FEATURES=M4COL+WARM+USER+["candidate_source_rank_pct","warm_slot_pct"]
ARMS=("PA_pseudo","PAC_pseudo_real")
MODES=("b0_fixed","model_full")
QUOTAS=(.001,.005,.01,.02)
PARAMS=dict(objective="regression_l2",learning_rate=.05,n_estimators=300,num_leaves=15,max_depth=6,
            min_child_samples=100,subsample=1.,colsample_bytree=1.,reg_lambda=1.,reg_alpha=0.,
            random_state=20260914,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)


def sampled_edge_mask(cutoff, candidates, y):
    """Vectorized stable 2% neutral sampler; every nonzero AP delta is kept."""
    base=pd.util.hash_pandas_object(candidates[["customer_id","article_id"]],index=False).to_numpy(np.uint64)
    seed=np.uint64(int.from_bytes(hashlib.sha256(cutoff.encode()).digest()[:8],"big"))
    slots=np.arange(1,13,dtype=np.uint64)*np.uint64(0x9E3779B185EBCA87)
    neutral=((base[:,None]^seed^slots[None,:])%np.uint64(50))==0
    keep=(y!=0)|neutral
    return keep,np.where(y[keep]==0,50.,1.)


def relation_path(repo,cutoff):
    p=repo/E2/cutoff/"m4-relation.npy"
    if p.exists():return p
    p=repo/E5B/"relations"/cutoff/"m4-relation.npy"
    if p.exists():return p
    raise FileNotFoundError("No time-safe direct relation for "+cutoff)


def candidate_population(repo,cutoff,kind,data):
    if kind=="pseudo":
        candidates=frame(repo/E3/cutoff/"pseudo.parquet");relation=np.load(repo/E3/cutoff/"pseudo-relation.npy",mmap_mode="r")
        warm=pd.MultiIndex.from_frame(data["warm"][["customer_id","article_id"]])
        keep=~pd.MultiIndex.from_frame(candidates[["customer_id","article_id"]]).isin(warm)
        candidates=candidates.loc[keep].reset_index(drop=True);relation=np.asarray(relation[keep])
        rank=candidates.ap_rf.to_numpy(float)/50.
    else:
        candidates=data["cold"].reset_index(drop=True);relation=np.load(relation_path(repo,cutoff),mmap_mode="r")
        rank=candidates.b0_rank_pct.to_numpy(float)
    assert len(candidates)==len(relation) and not candidates.duplicated(["customer_id","article_id"]).any()
    return candidates,relation,rank


def sampled_actions(data,candidates,relation,rank,cutoff,domain_weight):
    warm=data["warm"][WARM].to_numpy(float).reshape(-1,12,len(WARM));state=data["state"][USER].to_numpy(float)
    xs=[];ys=[];ws=[];tot=np.zeros(3,np.int64);kept=np.zeros(3,np.int64)
    for start in range(0,len(candidates),4096):
        cc=candidates.iloc[start:start+4096];idx=cc.user_index.to_numpy(int);n=len(cc)
        y=single_ap_delta(data["relevance"][idx],cc.target.to_numpy(int),data["truth_count"][idx])
        keep,weight=sampled_edge_mask(cutoff,cc,y)
        xc=np.repeat(relation[start:start+n],12,axis=0);xw=warm[idx].reshape(-1,len(WARM));xu=np.repeat(state[idx],12,axis=0)
        rp=np.repeat(rank[start:start+n],12)[:,None];slot=np.tile(np.arange(1,13)/12.,n)[:,None]
        x=np.concatenate([xc,xw,xu,rp,slot],axis=1).astype(np.float32);x[~np.isfinite(x)]=np.nan
        yy=y[keep];xs.append(x[keep.ravel()]);ys.append(yy);ws.append(weight*domain_weight)
        tot+=np.array([(y>0).sum(),(y<0).sum(),(y==0).sum()]);kept+=np.array([(yy>0).sum(),(yy<0).sum(),(yy==0).sum()])
    x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(ws)
    assert x.shape[1]==len(FEATURES) and np.array_equal(tot[:2],kept[:2])
    return x,y,weight,dict(full_beneficial_harmful_neutral=tot.tolist(),retained=kept.tolist(),rows=len(y))


def prepare(repo,cutoff):
    folder=repo/RUN/"prepared"/cutoff;folder.mkdir(parents=True)
    data=joblib.load(repo/PREPARED/cutoff/"data.joblib");audit={}
    for kind,domain_weight in [("pseudo",1.),("real",5.)]:
        candidates,relation,rank=candidate_population(repo,cutoff,kind,data)
        x,y,w,info=sampled_actions(data,candidates,relation,rank,cutoff,domain_weight)
        np.save(folder/f"{kind}-X.npy",x);np.save(folder/f"{kind}-y.npy",y);np.save(folder/f"{kind}-w.npy",w)
        info.update(candidate_rows=len(candidates),positive_candidates=int(candidates.target.sum()),domain_weight=domain_weight,
                    pseudo_warm_overlap_removed=(int(len(frame(repo/E3/cutoff/"pseudo.parquet"))-len(candidates)) if kind=="pseudo" else 0))
        audit[kind]=info;print("E6 PREP",cutoff,kind,info,flush=True)
        del x,y,w,candidates,relation;gc.collect()
    dump(folder/"READY.json",audit);return audit


def fit_model(repo,cutoff,arm,folder):
    xs=[];ys=[];ws=[]
    for t in earlier(cutoff):
        kinds=["pseudo"] if arm=="PA_pseudo" else ["pseudo","real"]
        for kind in kinds:
            xs.append(np.load(repo/RUN/"prepared"/t/f"{kind}-X.npy"));ys.append(np.load(repo/RUN/"prepared"/t/f"{kind}-y.npy"));ws.append(np.load(repo/RUN/"prepared"/t/f"{kind}-w.npy"))
    x=np.concatenate(xs);y=np.concatenate(ys);weight=np.concatenate(ws);assert len(x)==len(y)==len(weight)
    model=lgb.LGBMRegressor(**PARAMS)
    with threadpool_limits(limits=4):model.fit(x,y,sample_weight=weight,feature_name=FEATURES)
    model.booster_.save_model(str(folder/f"{cutoff}-{arm}.txt"));info=dict(rows=len(y),beneficial=int((y>0).sum()),harmful=int((y<0).sum()),train_dates=earlier(cutoff),weighted_rows=float(weight.sum()))
    del x,y,weight,xs,ys,ws;gc.collect();return model,info


def score_real(repo,cutoff,data,model):
    candidates,relation,rank=candidate_population(repo,cutoff,"real",data);warm=data["warm"][WARM].to_numpy(float).reshape(-1,12,len(WARM));state=data["state"][USER].to_numpy(float)
    scores=np.empty((len(candidates),12),float)
    for start in range(0,len(candidates),4096):
        cc=candidates.iloc[start:start+4096];idx=cc.user_index.to_numpy(int);n=len(cc)
        x=np.concatenate([np.repeat(relation[start:start+n],12,axis=0),warm[idx].reshape(-1,len(WARM)),np.repeat(state[idx],12,axis=0),
                          np.repeat(rank[start:start+n],12)[:,None],np.tile(np.arange(1,13)/12.,n)[:,None]],axis=1).astype(np.float32);x[~np.isfinite(x)]=np.nan
        scores[start:start+n]=model.predict(x,num_threads=4).reshape(-1,12)
    return scores


def selected_edges(data,scores,mode):
    cold=data["cold"]
    if mode=="b0_fixed":
        chosen=one_per_user(data,-cold.b0_rank.to_numpy(float));slot=np.argmax(scores[chosen],axis=1);edge=chosen*12+slot
    else:
        edge=[]
        for idx in cold.groupby("user_index",sort=False).indices.values():
            idx=np.asarray(idx,int);local=int(np.argmax(scores[idx].ravel()));edge.append(int(idx[local//12]*12+local%12))
        edge=np.asarray(edge,int)
    return edge


def evaluate(data,scores,mode,quota):
    edges=selected_edges(data,scores,mode);values=scores.ravel()[edges];ci=edges//12
    customer=data["cold"].iloc[ci].customer_id.to_numpy(str);count=max(1,int(math.ceil(quota*len(edges))));order=np.lexsort((customer,-values))[:count]
    lists=data["warm_lists"].copy();events=[]
    for local in order:
        e=int(edges[local]);candidate,slot=divmod(e,12);row=data["cold"].iloc[candidate];ui=int(row.user_index);old=lists[ui,slot];lists[ui,slot]=row.article_id
        delta=exact_ap(lists[ui],data["truthsets"][data["users"][ui]])-data["baseline_ap"][ui]
        events.append(dict(customer_id=data["users"][ui],candidate=row.article_id,removed=old,slot=slot+1,score=float(values[local]),delta=float(delta),inserted=int(row.target),removed_positive=int(old in data["truthsets"][data["users"][ui]])))
    f=pd.DataFrame(events)
    return dict(delta_map=float(f.delta.sum()/len(data["users"])),admissions=len(f),beneficial=int(f.delta.gt(0).sum()),harmful=int(f.delta.lt(0).sum()),neutral=int(f.delta.eq(0).sum()),inserted=int(f.inserted.sum()),removed=int(f.removed_positive.sum())),lists,f


def register(repo):
    c=dict(status="preregistered_before_E6_preparation",stage="FINAL-E6 pseudo-cold augmented joint action utility",
      hypothesis="E5 binary qC found some Cold positives but separate victim models failed. Train one utility model on pseudo-cold candidate × current Warm12 actions so candidate and victim compete under the exact AP-delta target.",
      training_dates=DATES,historical_validation=VALID,development_windows=WINDOWS,
      pseudo_population="E3 user-unseen globally-warm candidates; remove any item already in reconstructed WV3 WarmTop12 without refill; no future-conditioned filtering",
      real_population="time-safe actual Cold50 after Warm overlap",target="exact single replacement AP@12 delta for candidate×12 positions",
      features=FEATURES,arms={"PA_pseudo":"pseudo actions only","PAC_pseudo_real":"pseudo plus real-Cold actions; real retained rows weight5"},
      sampling="retain all beneficial/harmful actions; neutral keep when (stable pandas user-item hash XOR SHA256 cutoff seed XOR fixed slot mix) mod50=0; retained neutral inverse weight50",
      params=PARAMS,candidate_modes={"b0_fixed":"freeze best available B0 item, model chooses position","model_full":"model chooses item and position"},quotas=QUOTAS,
      historical_selection="highest mean exact delta_MAP; tie fewer admissions, b0_fixed, pseudo+real",historical_gate="mean>0, >=2/3 nonnegative, inserted>=2 and inserted>=removed",
      development_gate="existing expansion gate unchanged",if_pass="fullscale allowed without retuning",if_fail="WV3 retained; stop pseudo action route",
      final_week="not_run",no_commit_push=True,budget=dict(max_seconds=10800,threads=4,min_disk_gib=15),resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30),git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    dump(repo/REPORT/"PSEUDO_ACTION_E6_CONTRACT.json",c);return c


def run(repo):
    repo=Path(repo)
    if (repo/RUN).exists():
        contract=json.loads((repo/REPORT/"PSEUDO_ACTION_E6_CONTRACT.json").read_text(encoding="utf-8"))
        dump(repo/REPORT/"PSEUDO_ACTION_E6_RECOVERY.json",dict(reason="local function name was shadowed before first model fit",prepared_reused=True,models_existed=False,labels_or_features_changed=False,final_week="not_run"))
    else:
        (repo/RUN).mkdir(parents=True);contract=register(repo)
    start=time.perf_counter();deadline=time.time()+10800;preparation={}
    for t in DATES:
        guard(repo,deadline);ready=repo/RUN/"prepared"/t/"READY.json"
        preparation[t]=json.loads(ready.read_text(encoding="utf-8")) if ready.exists() else prepare(repo,t)
    historical={};aggregate={};scores_cache={}
    for cutoff in VALID:
        data=joblib.load(repo/PREPARED/cutoff/"data.joblib");historical[cutoff]={};scores_cache[cutoff]={}
        historical[cutoff]["B0_candidate"]=identify(data,-data["cold"].b0_rank.to_numpy(float))
        for arm in ARMS:
            model,fit_info=fit_model(repo,cutoff,arm,repo/RUN);scores=score_real(repo,cutoff,data,model);np.save(repo/RUN/f"{cutoff}-{arm}-scores.npy",scores);scores_cache[cutoff][arm]=scores
            historical[cutoff][arm]={"fit":fit_info,"candidate":identify(data,scores.max(axis=1)),"rules":{}}
            for mode in MODES:
                for quota in QUOTAS:
                    key=f"{mode}|q={quota:g}";result,_,_=evaluate(data,scores,mode,quota);historical[cutoff][arm]["rules"][key]=result
            print("E6 FIT",cutoff,arm,fit_info,flush=True);del model;gc.collect()
    for arm in ARMS:
        for mode in MODES:
            for quota in QUOTAS:
                key=f"{arm}|{mode}|q={quota:g}";rows=[historical[t][arm]["rules"][f"{mode}|q={quota:g}"] for t in VALID]
                aggregate[key]=dict(mean_delta=float(np.mean([r["delta_map"] for r in rows])),nonnegative=sum(r["delta_map"]>=0 for r in rows),admissions=sum(r["admissions"] for r in rows),inserted=sum(r["inserted"] for r in rows),removed=sum(r["removed"] for r in rows),beneficial=sum(r["beneficial"] for r in rows),harmful=sum(r["harmful"] for r in rows),arm=arm,mode=mode,quota=quota)
    eligible=[(k,v) for k,v in aggregate.items() if v["mean_delta"]>0 and v["nonnegative"]>=2 and v["inserted"]>=2 and v["inserted"]>=v["removed"]]
    eligible.sort(key=lambda kv:(-kv[1]["mean_delta"],kv[1]["admissions"],kv[1]["mode"]!="b0_fixed",kv[1]["arm"]!="PAC_pseudo_real"));selected=eligible[0][0] if eligible else None
    development={};gates={"not_run":"historical gate failed"};passed=False
    if selected:
        spec=aggregate[selected]
        for window,cutoff in WINDOWS.items():
            guard(repo,deadline);data=joblib.load(repo/PREPARED/cutoff/"data.joblib");model,fit_info=fit_model(repo,cutoff,spec["arm"],repo/RUN);scores=score_real(repo,cutoff,data,model);result,lists,events=evaluate(data,scores,spec["mode"],spec["quota"]);result["segments"]=segment_metrics(data,lists);result["fit"]=fit_info;development[window]=result;save_frame(events,repo/RUN/f"{window}-events.parquet");del model,data,scores,lists,events;gc.collect()
        delta=np.array([r["delta_map"] for r in development.values()]);warm=np.mean([r["segments"]["warm_21_plus"] for r in development.values()]);cold=np.mean([r["segments"]["all_cold_sparse"] for r in development.values()]);ins=sum(r["inserted"] for r in development.values());rem=sum(r["removed"] for r in development.values())
        gates=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),worst=bool(delta.min()>=-5e-5),warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),inserted_positive=bool(ins>=1),efficiency=bool(ins>=rem));passed=all(gates.values())
    out=dict(status="completed",stage=contract["stage"],preparation=preparation,historical=historical,historical_aggregate=aggregate,selected=selected,historical_gate_pass=selected is not None,development=development,development_gates=gates,fullscale_allowed=passed,fullscale_status="not_run",fallback="challenger" if passed else "WV3-741",final_week="not_run",seconds=time.perf_counter()-start)
    dump(repo/REPORT/"PSEUDO_ACTION_E6.json",out);print(json.dumps(dict(selected=selected,selected_historical=aggregate.get(selected),development=development,gates=gates,fullscale_allowed=passed,seconds=out["seconds"]),indent=2),flush=True)


if __name__=="__main__":
    try:run(Path.cwd())
    except Exception:
        dump(REPORT/f"PSEUDO_ACTION_E6_FAILURE_{time.time_ns()}.json",dict(error=traceback.format_exc(),final_week="not_run"));raise
