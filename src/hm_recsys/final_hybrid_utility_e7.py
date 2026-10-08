"""FINAL-E7: fuse binary Cold admission with pseudo-action utility evidence."""
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

from .final_b0_admission_e4 import stable_percentile, segment_metrics
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import VALID, WINDOWS
from .final_oracle_audit import dump
from .final_pseudo_action_e6 import fit_model, score_real
from .p42f_core import batches
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN=Path("artifacts/final/hybrid-utility-e7-v1")
REPORT=Path("reports/final")
PREPARED=Path("artifacts/final/integration-10pct-v1/prepared")
PB_HIST=Path("artifacts/final/binary-admission-e5a-v1")
PB_DEV=Path("artifacts/final/binary-admission-e5b-v1")
PAC_HIST=Path("artifacts/final/pseudo-action-e6-v1")
OLD_HIST=Path("artifacts/final/relation-pilot-v1")
OLD_DEV=Path("artifacts/final/integration-10pct-v1/models")
SIGNALS=("pb","pac","pb_pac_mean","pb_pac_min","pb_pac_old_mean","pb_pac_old_min")
POSITIONS=("old_action","pac_action")
QUOTAS=(.005,.01,.02)


def old_scores(repo,cutoff,data,historical):
    if not historical:
        w=next(w for w,t in WINDOWS.items() if t==cutoff);return np.load(repo/OLD_DEV/w/"scores.npy")
    model=lgb.Booster(model_file=str(repo/OLD_HIST/cutoff/"base70.txt"));out=np.empty((len(data["cold"]),12),float)
    for lo,cc,x,_ in batches(data,labels=False):out[lo:lo+len(cc)]=model.predict(x,num_threads=4).reshape(-1,12)
    return out


def pb_scores(repo,cutoff,historical):
    if historical:return np.load(repo/PB_HIST/f"{cutoff}-PB41-scores.npy")
    w=next(w for w,t in WINDOWS.items() if t==cutoff);return np.load(repo/PB_DEV/f"{w}-PB41-scores.npy")


def signal_table(chosen,pb,pac,old):
    pp=stable_percentile(pb[chosen]);pa=stable_percentile(pac[chosen].max(axis=1));po=stable_percentile(old[chosen].max(axis=1))
    return dict(pb=pp,pac=pa,pb_pac_mean=(pp+pa)/2,pb_pac_min=np.minimum(pp,pa),
                pb_pac_old_mean=(pp+pa+po)/3,pb_pac_old_min=np.minimum(np.minimum(pp,pa),po))


def evaluate(data,chosen,pac,old,signal,quota,position):
    count=max(1,int(math.ceil(quota*len(chosen))));customer=data["cold"].iloc[chosen].customer_id.to_numpy(str);selected=np.lexsort((customer,-signal))[:count]
    lists=data["warm_lists"].copy();events=[]
    for local in selected:
        ci=int(chosen[local]);row=data["cold"].iloc[ci];ui=int(row.user_index);slot=int(np.argmax(old[ci] if position=="old_action" else pac[ci]));victim=lists[ui,slot];lists[ui,slot]=row.article_id
        delta=exact_ap(lists[ui],data["truthsets"][data["users"][ui]])-data["baseline_ap"][ui]
        events.append(dict(customer_id=data["users"][ui],candidate=row.article_id,victim=victim,slot=slot+1,delta=float(delta),inserted=int(row.target),removed=int(victim in data["truthsets"][data["users"][ui]]),signal=float(signal[local])))
    e=pd.DataFrame(events);return dict(delta_map=float(e.delta.sum()/len(data["users"])),admissions=len(e),beneficial=int(e.delta.gt(0).sum()),harmful=int(e.delta.lt(0).sum()),neutral=int(e.delta.eq(0).sum()),inserted=int(e.inserted.sum()),removed=int(e.removed.sum())),lists,e


def register(repo):
    e5=json.loads((repo/REPORT/"BINARY_ADMISSION_E5A.json").read_text(encoding="utf-8"));e6=json.loads((repo/REPORT/"PSEUDO_ACTION_E6.json").read_text(encoding="utf-8"))
    c=dict(status="preregistered_before_E7_computation",stage="FINAL-E7 hybrid Cold admission and action utility",
      evidence="E5 PB41 B0-fixed found2 positives/2 removals with small positive historical mean; E6 PAC B0-fixed q1% found2 positives/5 removals but much larger positive historical mean. Their errors may be complementary.",
      fixed="best available B0 candidate; reuse historical PB/PAC scores; development refits exact frozen PB/PAC models only if historical gate passes",
      signals={"pb":"binary pseudo-cold candidate score percentile","pac":"pseudo+real joint action max-score percentile","pb_pac_mean":"unweighted mean","pb_pac_min":"conjunction by minimum","pb_pac_old_mean":"mean plus original action max percentile","pb_pac_old_min":"three-way conjunction"},
      positions={"old_action":"original action model position","pac_action":"pseudo-action model position"},quotas=QUOTAS,
      historical_validation=VALID,development_windows=WINDOWS,historical_selection="highest mean exact delta_MAP; tie fewer removals/admissions, old position, simpler signal",
      historical_gate="mean>0, >=2/3 nonnegative, inserted>=2 and inserted>=removed",development_gate="existing expansion gate unchanged",
      if_pass="fullscale allowed without retuning",if_fail="WV3 retained; stop hybrid route",final_week="not_run",no_commit_push=True,
      budget=dict(max_seconds=7200,threads=4,min_disk_gib=15),resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30),
      git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),source_summaries={"E5":e5["selected_admission"],"E6":e6["fullscale_allowed"]})
    dump(repo/REPORT/"HYBRID_UTILITY_E7_CONTRACT.json",c);return c


def run(repo):
    repo=Path(repo)
    if (repo/RUN).exists():raise FileExistsError("Do not overwrite E7")
    (repo/RUN).mkdir(parents=True);contract=register(repo);start=time.perf_counter();historical={};cache={}
    for cutoff in VALID:
        data=joblib.load(repo/PREPARED/cutoff/"data.joblib");chosen=one_per_user(data,-data["cold"].b0_rank.to_numpy(float));pb=pb_scores(repo,cutoff,True);pac=np.load(repo/PAC_HIST/f"{cutoff}-PAC_pseudo_real-scores.npy");old=old_scores(repo,cutoff,data,True);cache[cutoff]=(data,chosen,pac,old,signal_table(chosen,pb,pac,old));historical[cutoff]={}
    aggregate={}
    for name in SIGNALS:
        for quota in QUOTAS:
            for position in POSITIONS:
                key=f"{name}|q={quota:g}|{position}";rows=[]
                for cutoff in VALID:
                    data,chosen,pac,old,s=cache[cutoff];r,_,_=evaluate(data,chosen,pac,old,s[name],quota,position);historical[cutoff][key]=r;rows.append(r)
                aggregate[key]=dict(mean_delta=float(np.mean([r["delta_map"] for r in rows])),nonnegative=sum(r["delta_map"]>=0 for r in rows),admissions=sum(r["admissions"] for r in rows),inserted=sum(r["inserted"] for r in rows),removed=sum(r["removed"] for r in rows),beneficial=sum(r["beneficial"] for r in rows),harmful=sum(r["harmful"] for r in rows),signal=name,quota=quota,position=position)
    eligible=[(k,v) for k,v in aggregate.items() if v["mean_delta"]>0 and v["nonnegative"]>=2 and v["inserted"]>=2 and v["inserted"]>=v["removed"]]
    eligible.sort(key=lambda kv:(-kv[1]["mean_delta"],kv[1]["removed"],kv[1]["admissions"],kv[1]["position"]!="old_action",SIGNALS.index(kv[1]["signal"])));selected=eligible[0][0] if eligible else None
    development={};gates={"not_run":"historical gate failed"};passed=False
    if selected:
        spec=aggregate[selected]
        for window,cutoff in WINDOWS.items():
            data=joblib.load(repo/PREPARED/cutoff/"data.joblib");chosen=one_per_user(data,-data["cold"].b0_rank.to_numpy(float));pb=pb_scores(repo,cutoff,False);old=old_scores(repo,cutoff,data,False)
            model,fit_info=fit_model(repo,cutoff,"PAC_pseudo_real",repo/RUN);pac=score_real(repo,cutoff,data,model);s=signal_table(chosen,pb,pac,old);result,lists,events=evaluate(data,chosen,pac,old,s[spec["signal"]],spec["quota"],spec["position"]);result["segments"]=segment_metrics(data,lists);result["fit"]=fit_info;development[window]=result;save_frame(events,repo/RUN/f"{window}-events.parquet");del model,data,pac,old,pb,lists,events;gc.collect()
        delta=np.array([r["delta_map"] for r in development.values()]);warm=np.mean([r["segments"]["warm_21_plus"] for r in development.values()]);cold=np.mean([r["segments"]["all_cold_sparse"] for r in development.values()]);ins=sum(r["inserted"] for r in development.values());rem=sum(r["removed"] for r in development.values())
        gates=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),worst=bool(delta.min()>=-5e-5),warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),inserted_positive=bool(ins>=1),efficiency=bool(ins>=rem));passed=all(gates.values())
    out=dict(status="completed",stage=contract["stage"],historical=historical,historical_aggregate=aggregate,selected=selected,historical_gate_pass=selected is not None,development=development,development_gates=gates,fullscale_allowed=passed,fullscale_status="not_run",fallback="challenger" if passed else "WV3-741",final_week="not_run",seconds=time.perf_counter()-start)
    dump(repo/REPORT/"HYBRID_UTILITY_E7.json",out);print(json.dumps(dict(selected=selected,selected_historical=aggregate.get(selected),development=development,gates=gates,fullscale_allowed=passed,seconds=out["seconds"]),indent=2),flush=True)


if __name__=="__main__":
    try:run(Path.cwd())
    except Exception:
        dump(REPORT/f"HYBRID_UTILITY_E7_FAILURE_{time.time_ns()}.json",dict(error=traceback.format_exc(),final_week="not_run"));raise
