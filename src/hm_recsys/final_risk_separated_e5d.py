"""FINAL-E5D: combine Cold hit evidence with action safety and user propensity."""
from pathlib import Path
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

from .final_b0_admission_e4 import stable_percentile, segment_metrics
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import VALID, WINDOWS
from .final_oracle_audit import dump
from .p42f_core import batches
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN=Path("artifacts/final/risk-separated-e5d-v1")
REPORT=Path("reports/final")
PREPARED=Path("artifacts/final/integration-10pct-v1/prepared")
HIST_ACTION=Path("artifacts/final/relation-pilot-v1")
DEV_ACTION=Path("artifacts/final/integration-10pct-v1/models")
HIST_PB=Path("artifacts/final/binary-admission-e5a-v1")
DEV_PB=Path("artifacts/final/binary-admission-e5b-v1")
QUOTAS=(.005,.01,.02,.05)
SIGNALS=("pb","pb_action","pb_novel","pb_lowpop","pb_propensity","pb_action_novel",
         "pb_action_safe","pb_novel_safe","pb_all")


def action_scores(repo,cutoff,data,historical):
    if not historical:
        w=next(w for w,t in WINDOWS.items() if t==cutoff);return np.load(repo/DEV_ACTION/w/"scores.npy")
    model=lgb.Booster(model_file=str(repo/HIST_ACTION/cutoff/"base70.txt"));out=np.empty((len(data["cold"]),12),float)
    for lo,cc,x,_ in batches(data,labels=False):out[lo:lo+len(cc)]=model.predict(x,num_threads=4).reshape(-1,12)
    return out


def pb_scores(repo,cutoff,historical):
    if historical:return np.load(repo/HIST_PB/f"{cutoff}-PB41-scores.npy")
    w=next(w for w,t in WINDOWS.items() if t==cutoff);return np.load(repo/DEV_PB/f"{w}-PB41-scores.npy")


def signals(data,chosen,pb,actions):
    user=data["cold"].iloc[chosen].user_index.to_numpy(int)
    slot=np.argmax(actions[chosen],axis=1)
    qw=data["warm"].qW_within_user_percentile.to_numpy(float).reshape(-1,12)[user,slot]
    raw=dict(pb=pb[chosen],action=actions[chosen,slot],
             novel=data["state"].novel_purchase_share_0_5_recent84d.to_numpy(float)[user],
             lowpop=data["state"].low_pop_purchase_share_0_20.to_numpy(float)[user],
             safe=1-np.nan_to_num(qw,nan=1.0))
    p={k:stable_percentile(np.nan_to_num(v,nan=-np.inf)) for k,v in raw.items()}
    p["pb_action"]=np.mean([p["pb"],p["action"]],axis=0)
    p["pb_novel"]=np.mean([p["pb"],p["novel"]],axis=0)
    p["pb_lowpop"]=np.mean([p["pb"],p["lowpop"]],axis=0)
    p["pb_propensity"]=np.mean([p["pb"],p["novel"],p["lowpop"]],axis=0)
    p["pb_action_novel"]=np.mean([p["pb"],p["action"],p["novel"]],axis=0)
    p["pb_action_safe"]=np.mean([p["pb"],p["action"],p["safe"]],axis=0)
    p["pb_novel_safe"]=np.mean([p["pb"],p["novel"],p["safe"]],axis=0)
    p["pb_all"]=np.mean([p["pb"],p["action"],p["novel"],p["lowpop"],p["safe"]],axis=0)
    return {k:p[k] for k in SIGNALS}


def evaluate(data,chosen,actions,signal,quota):
    count=max(1,int(math.ceil(quota*len(chosen))));customer=data["cold"].iloc[chosen].customer_id.to_numpy(str)
    selected=np.lexsort((customer,-signal))[:count];lists=data["warm_lists"].copy();events=[]
    for local in selected:
        ci=int(chosen[local]);row=data["cold"].iloc[ci];ui=int(row.user_index);slot=int(np.argmax(actions[ci]));old=lists[ui,slot]
        lists[ui,slot]=row.article_id;delta=exact_ap(lists[ui],data["truthsets"][data["users"][ui]])-data["baseline_ap"][ui]
        events.append(dict(customer_id=data["users"][ui],candidate=row.article_id,removed=old,slot=slot+1,delta=float(delta),
                           inserted=int(row.target),removed_positive=int(old in data["truthsets"][data["users"][ui]]),signal=float(signal[local])))
    e=pd.DataFrame(events)
    return dict(delta_map=float(e.delta.sum()/len(data["users"])),admissions=len(e),beneficial=int(e.delta.gt(0).sum()),
                harmful=int(e.delta.lt(0).sum()),neutral=int(e.delta.eq(0).sum()),inserted=int(e.inserted.sum()),
                removed=int(e.removed_positive.sum())),lists,e


def register(repo):
    c=dict(status="preregistered_before_E5D_computation",stage="FINAL-E5D risk-separated admission evidence",
      bottleneck="E5B found2 Cold positives and removed6 Warm positives; E5C victim reranker failed, so preserve the original action position and change only which users are admitted.",
      fixed="B0-best candidate; PB41 model fits/scores from E5A/E5B; original action model best position; no retraining",
      base_signals={"pb":"pseudo-cold binary score","action":"frozen action-model max score","novel":"user recent84d share of purchases of items with global count0-5 at purchase time","lowpop":"user all-history share of purchases with global count0-20","safe":"1 minus qW within-user percentile of action-selected Warm victim; missing is unsafe0"},
      composites="unweighted mean of within-window stable percentiles named by each signal",signals=SIGNALS,quotas=QUOTAS,
      historical_validation=VALID,development_windows=WINDOWS,
      historical_selection="highest mean exact delta_MAP; tie fewer admissions, simpler signal, lower quota",
      historical_gate="mean>0, >=2/3 nonnegative, inserted>=2 and inserted>=removed",
      development_gate="existing expansion gate unchanged",if_pass="fullscale allowed without further tuning",if_fail="WV3 retained and stop E5 route",
      final_week="not_run",no_commit_push=True,budget=dict(max_seconds=3600,threads=4,min_disk_gib=15),
      resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30),git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    dump(repo/REPORT/"RISK_SEPARATED_E5D_CONTRACT.json",c);return c


def load(repo,cutoff,historical):
    data=joblib.load(repo/PREPARED/cutoff/"data.joblib");pb=pb_scores(repo,cutoff,historical);actions=action_scores(repo,cutoff,data,historical)
    chosen=one_per_user(data,-data["cold"].b0_rank.to_numpy(float));return data,chosen,actions,signals(data,chosen,pb,actions)


def run(repo):
    repo=Path(repo)
    if (repo/RUN).exists():raise FileExistsError("Do not overwrite E5D")
    (repo/RUN).mkdir(parents=True);contract=register(repo);start=time.perf_counter();historical={};cache={}
    for cutoff in VALID:
        data,chosen,actions,s=load(repo,cutoff,True);cache[cutoff]=(data,chosen,actions,s);historical[cutoff]={}
    aggregate={}
    for name in SIGNALS:
        for quota in QUOTAS:
            key=f"{name}|q={quota:g}";rows=[]
            for cutoff in VALID:
                data,chosen,actions,s=cache[cutoff];result,_,_=evaluate(data,chosen,actions,s[name],quota);historical[cutoff][key]=result;rows.append(result)
            aggregate[key]=dict(mean_delta=float(np.mean([r["delta_map"] for r in rows])),nonnegative=sum(r["delta_map"]>=0 for r in rows),
                                admissions=sum(r["admissions"] for r in rows),inserted=sum(r["inserted"] for r in rows),removed=sum(r["removed"] for r in rows),
                                beneficial=sum(r["beneficial"] for r in rows),harmful=sum(r["harmful"] for r in rows),signal=name,quota=quota)
    eligible=[(k,v) for k,v in aggregate.items() if v["mean_delta"]>0 and v["nonnegative"]>=2 and v["inserted"]>=2 and v["inserted"]>=v["removed"]]
    eligible.sort(key=lambda kv:(-kv[1]["mean_delta"],kv[1]["admissions"],SIGNALS.index(kv[1]["signal"]),kv[1]["quota"]));selected=eligible[0][0] if eligible else None
    development={};gates={"not_run":"historical gate failed"};passed=False
    if selected:
        spec=aggregate[selected]
        for window,cutoff in WINDOWS.items():
            data,chosen,actions,s=load(repo,cutoff,False);result,lists,events=evaluate(data,chosen,actions,s[spec["signal"]],spec["quota"]);result["segments"]=segment_metrics(data,lists);development[window]=result;save_frame(events,repo/RUN/f"{window}-events.parquet")
        delta=np.array([r["delta_map"] for r in development.values()]);warm=np.mean([r["segments"]["warm_21_plus"] for r in development.values()]);cold=np.mean([r["segments"]["all_cold_sparse"] for r in development.values()]);ins=sum(r["inserted"] for r in development.values());rem=sum(r["removed"] for r in development.values())
        gates=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),worst=bool(delta.min()>=-5e-5),warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),inserted_positive=bool(ins>=1),efficiency=bool(ins>=rem));passed=all(gates.values())
    out=dict(status="completed",stage=contract["stage"],historical=historical,historical_aggregate=aggregate,selected=selected,
             historical_gate_pass=selected is not None,development=development,development_gates=gates,
             fullscale_allowed=passed,fullscale_status="not_run",fallback="challenger" if passed else "WV3-741",final_week="not_run",seconds=time.perf_counter()-start)
    dump(repo/REPORT/"RISK_SEPARATED_E5D.json",out);print(json.dumps(dict(selected=selected,selected_historical=aggregate.get(selected),development=development,gates=gates,fullscale_allowed=passed,seconds=out["seconds"]),indent=2),flush=True)


if __name__=="__main__":
    try:run(Path.cwd())
    except Exception:
        dump(REPORT/f"RISK_SEPARATED_E5D_FAILURE_{time.time_ns()}.json",dict(error=traceback.format_exc(),final_week="not_run"));raise
