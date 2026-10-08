"""FINAL-E5C: third-ranker victim selection for the frozen E5 admission rule."""
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
from .final_binary_admission_e5 import one_per_user, PARAMS
from .final_candidate_e2 import VALID, WINDOWS
from .final_oracle_audit import dump
from .p42f_contract import WARM, USER, earlier
from .p42f_data import save_frame
from .p43a_policy import exact_ap
from .p43a_run import guard


RUN = Path("artifacts/final/victim-risk-e5c-v1")
REPORT = Path("reports/final")
PREPARED = Path("artifacts/final/integration-10pct-v1/prepared")
HIST_ACTION = Path("artifacts/final/relation-pilot-v1")
DEV_ACTION = Path("artifacts/final/integration-10pct-v1/models")
HIST_PB = Path("artifacts/final/binary-admission-e5a-v1")
DEV_PB = Path("artifacts/final/binary-admission-e5b-v1")
POLICIES = ("action_any", "fixed_rank12", "qW_min_any", "risk_min_any", "risk_min_tail")
RISK_FEATURES = WARM + USER


def risk_matrix(data):
    warm = data["warm"]
    user = data["state"][USER].to_numpy(float)
    x = np.concatenate([warm[WARM].to_numpy(float), np.repeat(user, 12, axis=0)], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    y = warm.target.to_numpy(np.int32)
    assert x.shape == (len(data["users"])*12, len(RISK_FEATURES)) and len(y)==len(x)
    return x,y


def action_scores(repo, cutoff, data, historical):
    if not historical:
        window=next(w for w,t in WINDOWS.items() if t==cutoff)
        return np.load(repo/DEV_ACTION/window/"scores.npy")
    from .p42f_core import batches
    model=lgb.Booster(model_file=str(repo/HIST_ACTION/cutoff/"base70.txt"))
    scores=np.empty((len(data["cold"]),12),float)
    for lo,cc,x,_ in batches(data,labels=False):scores[lo:lo+len(cc)]=model.predict(x,num_threads=4).reshape(-1,12)
    return scores


def pb_scores(repo,cutoff,historical):
    if historical:return np.load(repo/HIST_PB/f"{cutoff}-PB41-scores.npy")
    window=next(w for w,t in WINDOWS.items() if t==cutoff)
    return np.load(repo/DEV_PB/f"{window}-PB41-scores.npy")


def evaluate(data,chosen,admission_score,actions,risk,policy):
    count=max(1,int(math.ceil(.02*len(chosen))))
    customer=data["cold"].iloc[chosen].customer_id.to_numpy(str)
    selected=np.lexsort((customer,-stable_percentile(admission_score[chosen])))[:count]
    lists=data["warm_lists"].copy();events=[]
    qw=data["warm"].qW_within_user_percentile.to_numpy(float).reshape(-1,12)
    risk=risk.reshape(-1,12)
    for local in selected:
        ci=int(chosen[local]);row=data["cold"].iloc[ci];ui=int(row.user_index)
        if policy=="action_any":slot=int(np.argmax(actions[ci]))
        elif policy=="fixed_rank12":slot=11
        elif policy=="qW_min_any":
            values=np.nan_to_num(qw[ui],nan=np.inf);slot=int(np.argmin(values)) if np.isfinite(values).any() else 11
        elif policy=="risk_min_any":slot=int(np.argmin(risk[ui]))
        elif policy=="risk_min_tail":slot=7+int(np.argmin(risk[ui,7:]))
        else:raise ValueError(policy)
        old=lists[ui,slot];lists[ui,slot]=row.article_id
        delta=exact_ap(lists[ui],data["truthsets"][data["users"][ui]])-data["baseline_ap"][ui]
        events.append(dict(customer_id=data["users"][ui],candidate=row.article_id,removed=old,slot=slot+1,
                           delta=float(delta),inserted=int(row.target),removed_positive=int(old in data["truthsets"][data["users"][ui]]),
                           admission_score=float(admission_score[ci]),victim_risk=float(risk[ui,slot])))
    e=pd.DataFrame(events);delta=float(e.delta.sum()/len(data["users"]))
    return dict(delta_map=delta,admissions=len(e),beneficial=int(e.delta.gt(0).sum()),harmful=int(e.delta.lt(0).sum()),
                neutral=int(e.delta.eq(0).sum()),inserted=int(e.inserted.sum()),removed=int(e.removed_positive.sum())),lists,e


def fit_risk(repo,cutoff,folder):
    xs=[];ys=[]
    for t in earlier(cutoff):
        d=joblib.load(repo/PREPARED/t/"data.joblib");x,y=risk_matrix(d);xs.append(x);ys.append(y)
    x=np.concatenate(xs);y=np.concatenate(ys)
    model=lgb.LGBMClassifier(**PARAMS)
    with threadpool_limits(limits=4):model.fit(x,y,feature_name=RISK_FEATURES)
    model.booster_.save_model(str(folder/f"{cutoff}-warm-risk.txt"))
    info=dict(rows=len(y),positives=int(y.sum()),train_dates=earlier(cutoff))
    del x,y,xs,ys;gc.collect();return model,info


def register(repo):
    e5a=json.loads((repo/REPORT/"BINARY_ADMISSION_E5A.json").read_text(encoding="utf-8"))
    e5b=json.loads((repo/REPORT/"BINARY_ADMISSION_E5B.json").read_text(encoding="utf-8"))
    if e5a["selected_admission"]!="PB41|b0_best|q=0.02|model_any" or e5b["fullscale_allowed"]:
        raise ValueError("E5 provenance mismatch")
    c=dict(status="preregistered_before_E5C_fit",stage="FINAL-E5C third-ranker victim risk",
      bottleneck="Frozen E5B admitted2 Cold positives but removed6 Warm positives; candidate/admission has some signal, eviction risk dominates net MAP.",
      fixed="B0-best candidate and PB41 top2% cross-user admission are unchanged from E5A/E5B",
      historical_validation=VALID,development_windows=WINDOWS,
      risk_target="whether each reconstructed WV3 WarmTop12 item is purchased by that user in next7days",
      risk_features=RISK_FEATURES,risk_training="all12 Warm rows from strictly earlier cutoffs; binary LightGBM; no Cold candidate feature",
      policies={"action_any":"old action model chooses any position","fixed_rank12":"always evict current rank12",
                "qW_min_any":"evict lowest existing qW within-user percentile, missing fallback rank12",
                "risk_min_any":"evict lowest learned Warm-purchase probability among12",
                "risk_min_tail":"same but only positions8-12"},
      historical_selection="highest mean exact delta_MAP; tie fewer removals then learned risk then fixed rank12 then control",
      historical_gate="mean>0, >=2/3 nonnegative, inserted>=2, inserted>=removed",
      development_gate="existing expansion gate unchanged",if_pass="fullscale allowed without further tuning",if_fail="WV3 retained; stop E5 route",
      final_week="not_run",no_commit_push=True,budget=dict(max_seconds=7200,threads=4,min_disk_gib=15),
      resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30),git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    dump(repo/REPORT/"VICTIM_RISK_E5C_CONTRACT.json",c);return c


def run(repo):
    repo=Path(repo)
    if (repo/RUN).exists():raise FileExistsError("Do not overwrite E5C evidence")
    (repo/RUN).mkdir(parents=True);contract=register(repo);start=time.perf_counter();deadline=time.time()+7200
    history={};agg={}
    for cutoff in VALID:
        guard(repo,deadline);data=joblib.load(repo/PREPARED/cutoff/"data.joblib");pb=pb_scores(repo,cutoff,True)
        chosen=one_per_user(data,-data["cold"].b0_rank.to_numpy(float));actions=action_scores(repo,cutoff,data,True)
        model,fit=fit_risk(repo,cutoff,repo/RUN);x,_=risk_matrix(data);risk=model.predict_proba(x,num_threads=4)[:,1]
        history[cutoff]={"fit":fit,"policies":{}}
        for policy in POLICIES:
            result,_,_=evaluate(data,chosen,pb,actions,risk,policy);history[cutoff]["policies"][policy]=result
        del data,pb,chosen,actions,model,x,risk;gc.collect()
    for policy in POLICIES:
        rows=[history[t]["policies"][policy] for t in VALID]
        agg[policy]=dict(mean_delta=float(np.mean([r["delta_map"] for r in rows])),nonnegative=sum(r["delta_map"]>=0 for r in rows),
                         inserted=sum(r["inserted"] for r in rows),removed=sum(r["removed"] for r in rows),
                         beneficial=sum(r["beneficial"] for r in rows),harmful=sum(r["harmful"] for r in rows))
    eligible=[(k,v) for k,v in agg.items() if v["mean_delta"]>0 and v["nonnegative"]>=2 and v["inserted"]>=2 and v["inserted"]>=v["removed"]]
    preference={"risk_min_any":0,"risk_min_tail":1,"fixed_rank12":2,"qW_min_any":3,"action_any":4}
    eligible.sort(key=lambda kv:(-kv[1]["mean_delta"],kv[1]["removed"],preference[kv[0]]));selected=eligible[0][0] if eligible else None
    development={};gates={"not_run":"historical gate failed"};passed=False
    if selected:
        for window,cutoff in WINDOWS.items():
            guard(repo,deadline);data=joblib.load(repo/PREPARED/cutoff/"data.joblib");pb=pb_scores(repo,cutoff,False)
            chosen=one_per_user(data,-data["cold"].b0_rank.to_numpy(float));actions=action_scores(repo,cutoff,data,False)
            model,fit=fit_risk(repo,cutoff,repo/RUN);x,_=risk_matrix(data);risk=model.predict_proba(x,num_threads=4)[:,1]
            result,lists,events=evaluate(data,chosen,pb,actions,risk,selected);result["segments"]=segment_metrics(data,lists);result["fit"]=fit
            save_frame(events,repo/RUN/f"{window}-events.parquet");development[window]=result
            del data,pb,chosen,actions,model,x,risk,lists,events;gc.collect()
        delta=np.array([x["delta_map"] for x in development.values()]);warm=np.mean([x["segments"]["warm_21_plus"] for x in development.values()]);cold=np.mean([x["segments"]["all_cold_sparse"] for x in development.values()]);ins=sum(x["inserted"] for x in development.values());rem=sum(x["removed"] for x in development.values())
        gates=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),worst=bool(delta.min()>=-5e-5),
                   warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),inserted_positive=bool(ins>=1),efficiency=bool(ins>=rem))
        passed=all(gates.values())
    out=dict(status="completed",stage=contract["stage"],historical=history,historical_aggregate=agg,selected=selected,
             historical_gate_pass=selected is not None,development=development,development_gates=gates,
             fullscale_allowed=passed,fullscale_status="not_run",fallback="challenger" if passed else "WV3-741",
             final_week="not_run",seconds=time.perf_counter()-start)
    dump(repo/REPORT/"VICTIM_RISK_E5C.json",out);print(json.dumps(out,indent=2),flush=True)


if __name__=="__main__":
    try:run(Path.cwd())
    except Exception:
        dump(REPORT/f"VICTIM_RISK_E5C_FAILURE_{time.time_ns()}.json",dict(error=traceback.format_exc(),final_week="not_run"));raise
