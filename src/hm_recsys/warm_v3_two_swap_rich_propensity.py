"""WV3-691: reuse frozen rich propensity scores for at most two protected tail swaps."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import INNER, OUTER, pair_frame, source
from .warm_v3_residual_admission_audit import apk_from_targets
from .warm_v3_rich_propensity import FEATURES, rich_candidate_frame


TRIAL="WV3-691"
CONTRACT=common.REPORT/"WV3-691_TWO_SWAP_RICH_CONTRACT.json"
SCREEN_REPORT=common.REPORT/"WV3-691_SCREEN.json"
SCREEN_MARKDOWN=common.REPORT/"WV3-691_SCREEN.md"
OUTER_REPORT=common.REPORT/"WV3-691_OUTER.json"
MODEL_PATH=common.ART/"WV3-661"/"MODEL.txt"
ART_ROOT=common.ART/TRIAL


def choose_two_swaps(pairs,scores):
    challenger=scores.rename(columns={"article_id":"challenger_article_id","ranking_score":"challenger_score"});victim=scores.rename(columns={"article_id":"victim_article_id","ranking_score":"victim_score"})
    cols=["customer_id","challenger_article_id","victim_article_id","challenger_rank","victim_rank","unit_gain"]
    ranked=pairs[cols].merge(challenger,on=["customer_id","challenger_article_id"],how="left",validate="many_to_one").merge(victim,on=["customer_id","victim_article_id"],how="left",validate="many_to_one")
    assert ranked[["challenger_score","victim_score"]].notna().all().all();ranked["score_difference"]=ranked.challenger_score-ranked.victim_score;ranked["predicted_gain"]=ranked.score_difference*ranked.unit_gain;ranked=ranked[ranked.predicted_gain>0].sort_values(["customer_id","predicted_gain","score_difference","victim_rank","challenger_rank","challenger_article_id","victim_article_id"],ascending=[True,False,False,False,True,True,True],kind="mergesort")
    selected=[]
    for _,group in ranked.groupby("customer_id",sort=False):
        used_c=set();used_v=set();order=0
        for row in group.itertuples(index=False):
            if row.challenger_article_id in used_c or row.victim_article_id in used_v:continue
            order+=1;record=row._asdict();record["swap_order"]=order;selected.append(record);used_c.add(row.challenger_article_id);used_v.add(row.victim_article_id)
            if order==2:break
    return pd.DataFrame(selected,columns=list(ranked.columns)+["swap_order"])


def exact_user_deltas(candidates,swaps):
    rows=[]
    candidate_groups={customer:g.sort_values("rf",kind="mergesort") for customer,g in candidates.groupby("customer_id",sort=False)}
    for customer,group in swaps.groupby("customer_id",sort=False):
        base=candidate_groups[customer];values=base.target.to_numpy(np.uint8);truth_count=int(base.truth_count.iloc[0]);before=apk_from_targets(values,truth_count)
        changed=values.copy()
        for row in group.sort_values("swap_order").itertuples(index=False):changed[int(row.victim_rank)-1],changed[int(row.challenger_rank)-1]=changed[int(row.challenger_rank)-1],changed[int(row.victim_rank)-1]
        after=apk_from_targets(changed,truth_count);rows.append({"customer_id":customer,"swap_count":len(group),"baseline_ap":before,"reranked_ap":after,"actual_delta":after-before})
    return pd.DataFrame(rows)


def evaluate(folder,name,model,stage):
    path,meta=source(folder);started=time.perf_counter();candidates,details=rich_candidate_frame(meta["cutoff"],rank_min=1);values=model.predict(candidates[FEATURES].to_numpy(np.float32),num_threads=4);scores=candidates[["customer_id","article_id"]].copy();scores["ranking_score"]=values;pairs=pair_frame(path,discordant_only=False);swaps=choose_two_swaps(pairs,scores);users=exact_user_deltas(candidates,swaps);beneficial=users.actual_delta>1e-15;harmful=users.actual_delta< -1e-15;neutral=~(beneficial|harmful);delta=float(users.actual_delta.sum()/meta["total_users"]);baseline=float(meta["baseline_map_population_component"])
    result={"window":name,"cutoff":meta["cutoff"],"role":stage,"total_users_denominator":meta["total_users"],"selected_users":len(users),"selected_swap_rows":len(swaps),"users_with_two_swaps":int((users.swap_count==2).sum()),"beneficial_selected_users":int(beneficial.sum()),"harmful_selected_users":int(harmful.sum()),"neutral_selected_users":int(neutral.sum()),"selected_neutral_share":float(neutral.mean()) if len(users) else 0.,"baseline_MAP@12":baseline,"MAP@12":baseline+delta,"population_delta":delta,"protected_ranks1_7_changes":0,"maximum_swaps_per_user":2,"exact_AP_recomputed_after_all_swaps":True,"candidate_input":details,"runtime_seconds":time.perf_counter()-started,"final_week":"not_run"};return result,swaps,users


def contract():
    return {"created_at":now(),"experiment_id":TRIAL,"parent":"WV3-661","parent_diagnostic":"WV3-690","architecture_family":"frozen_rich_propensity_maximum_two_swaps","hypothesis":"The frozen WV3-661 candidate scores can convert the material second-swap oracle headroom without retraining or destabilizing the protected head.","model":"exact frozen WV3-661 MODEL.txt; no training","decision":"greedily select at most two positive predicted-gain pairs with unique challenger and victim items, then recompute exact AP@12","candidate_pool":"unchanged Top50; ranks1-7 protected, challengers13-50, victims8-12","inner_gate":{"standard_vs_WV2_601":"mean>=+0.0001, >=3 positive, worst>=-0.0005","incremental_vs_WV3_661":"mean>0, >=3 nondegrade, worst>=-0.0002"},"expected_minutes":15,"outer_policy":"one frozen policy exposure only if both inner gates pass; no swap threshold, count or ordering rescue","fallback":"retain one-swap WV3-661 and close action-count expansion","final_week":"2020-09-16 not_run"}


def register():
    common.setup();common.budget(15);assert read(common.REPORT/"WV3-690_TWO_SWAP_ORACLE_AUDIT.json")["two_swap_model_authorized"] and MODEL_PATH.is_file()
    if not CONTRACT.exists():write(CONTRACT,contract())
    registry=read(common.REGISTRY)
    if not any(row["experiment_id"]==TRIAL for row in registry["trials"]):common.register(TRIAL,"frozen_rich_propensity_maximum_two_swaps",contract()["hypothesis"],candidate_protocol=contract()["candidate_pool"],training_protocol="none; reuse frozen WV3-661 model",features=FEATURES,params={"decision":contract()["decision"],"inner_gate":contract()["inner_gate"],"final_week":"not_run"},expected_minutes=15)


def incremental_gate(windows):
    prior=read(common.REPORT/"WV3-661_SCREEN.json")["windows"];delta={n:r["MAP@12"]-prior[n]["MAP@12"] for n,r in windows.items()};values=list(delta.values());return {"per_window_delta_vs_WV3_661":delta,"mean_delta_vs_WV3_661":float(np.mean(values)),"nondegrade_windows_vs_WV3_661":sum(v>=0 for v in values),"worst_delta_vs_WV3_661":min(values),"passed":bool(np.mean(values)>0 and sum(v>=0 for v in values)>=3 and min(values)>=-.0002)}


def render(result):
    s=result["standard_screening_vs_WV2_601"];i=result["incremental_screening_vs_WV3_661"];lines=["# WV3-691 丰富特征最多两次尾部准入","","## 结论","",f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {s['mean_population_delta']:+.9f}；相对一次换位 WV3-661 平均增加 {i['mean_delta_vs_WV3_661']:+.9f}。","","## 术语与精确评测","","- 最多两次准入（本项目自定义）：按冻结 WV3-661 分数差乘单次位置权重排序，贪心选择最多两个挑战商品和两个被替换商品；同一商品不能重复参与。","- 贪心（行业通用算法概念）：每次取当前预测收益最高且不与已选商品冲突的换位；本实验固定最多两次，不搜索次数。","- 精确 AP 重算：两次换位可能互相影响后续正例的累计精度，因此逐用户重建换位后的完整 Top12，再计算 AP@12；不把两个单换位增益相加。","- 中性用户（本项目自定义）：完成一或两次换位后，该用户精确 AP@12 与基线相同；占比分母为执行至少一次换位的用户。","- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。","","| 内层窗口 | 选择用户 | 使用两次 | 有益 / 有害 / 中性用户 | 相对 WV2-601 | 相对 WV3-661 |","|---|---:|---:|---:|---:|---:|"]
    for n,r in result["windows"].items():lines.append(f"| {n} | {r['selected_users']:,} | {r['users_with_two_swaps']:,} | {r['beneficial_selected_users']:,} / {r['harmful_selected_users']:,} / {r['neutral_selected_users']:,} | {r['population_delta']:+.9f} | {i['per_window_delta_vs_WV3_661'][n]:+.9f} |")
    lines += ["","模型未重训，候选池未变化。只有两个内层门槛同时通过才读取一次外层；失败后不尝试三至五次或调整准入阈值。最终周 `2020-09-16` 保持 `not_run`。",""];return "\n".join(lines)


def screen():
    common.setup();common.budget(15);entry=next(r for r in read(common.REGISTRY)["trials"] if r["experiment_id"]==TRIAL);assert entry["decision"]=="preregistered" and entry["outer_exposures"]==0;model=lgb.Booster(model_file=str(MODEL_PATH));started=time.perf_counter();windows={}
    for n,f in INNER:
        row,swaps,users=evaluate(f,n,model,"frozen_rich_two_swap_inner");windows[n]=row;root=ART_ROOT/n/"inner";root.mkdir(parents=True,exist_ok=True);save_parquet(swaps,root/"swaps.parquet");save_parquet(users,root/"users.parquet");write(root/"REVIEW.json",row);print({"two_swap_inner":n,"delta":row["population_delta"],"two":row["users_with_two_swaps"]},flush=True)
    standard=screening_gate(r["population_delta"] for r in windows.values());incremental=incremental_gate(windows);passed=standard["passed"] and incremental["passed"];result={"created_at":now(),"experiment_id":TRIAL,"architecture":"frozen_rich_propensity_maximum_two_swaps","windows":windows,"standard_screening_vs_WV2_601":standard,"incremental_screening_vs_WV3_661":incremental,"passed":passed,"new_training":False,"candidate_pool_changed":False,"runtime_seconds":time.perf_counter()-started,"final_week":"not_run"};write(SCREEN_REPORT,result);SCREEN_MARKDOWN.write_text(render(result),encoding="utf-8");common.update(TRIAL,decision="inner_pass" if passed else "reject_inner",inner_evidence={"standard":standard,"incremental":incremental,"passed":passed},runtime=result["runtime_seconds"],artifact_paths=[str(SCREEN_REPORT),str(SCREEN_MARKDOWN)])
    common.log(f"{TRIAL} inner screen","WV3-690 found 0.000698 mean exact oracle headroom from a second protected tail swap.",contract()["hypothesis"],f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",f"standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_661']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_661']}/4; pass={passed}.","Expose the frozen policy once." if passed else "Reject two swaps and retain WV3-661.","Run one outer confirmation if both gates pass; otherwise close action-count expansion.",alternatives="No retraining or score tuning; the only changed variable is maximum disjoint swaps one to two.",experiment="Reuse frozen WV3-661 scores, select two disjoint predicted-gain swaps, recompute exact AP.",reflection="Exact recomputation prevents invalid additive assumptions for interacting swaps.")
    print({"trial":TRIAL,"standard":standard,"incremental":incremental,"passed":passed},flush=True);return result


def confirm():
    common.setup();common.budget(15);entry=next(r for r in read(common.REGISTRY)["trials"] if r["experiment_id"]==TRIAL);assert entry["decision"]=="inner_pass" and entry["outer_exposures"]==1;assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists():return read(OUTER_REPORT)
    model=lgb.Booster(model_file=str(MODEL_PATH));started=time.perf_counter();windows={}
    for n,f in OUTER:
        row,swaps,users=evaluate(f,n,model,"frozen_rich_two_swap_outer");windows[n]=row;root=ART_ROOT/n/"outer";root.mkdir(parents=True,exist_ok=True);save_parquet(swaps,root/"swaps.parquet");save_parquet(users,root/"users.parquet");write(root/"REVIEW.json",row);print({"two_swap_outer":n,"delta":row["population_delta"],"two":row["users_with_two_swaps"]},flush=True)
    standard=common.summary({n:r["MAP@12"] for n,r in windows.items()},2020);prior=read(common.REPORT/"WV3-661_OUTER.json");delta={n:windows[n]["MAP@12"]-prior["per_window_MAP"][n] for n in windows};values=list(delta.values());incremental={"per_window_delta_vs_WV3_661":delta,"mean_delta_vs_WV3_661":float(np.mean(values)),"nondegrade_windows_vs_WV3_661":sum(v>=0 for v in values),"worst_delta_vs_WV3_661":min(values)};better=bool(standard["stable"] and np.mean(values)>0 and sum(v>=0 for v in values)>=3 and min(values)>=-.0002);result={"created_at":now(),"experiment_id":TRIAL,"architecture":"frozen_rich_propensity_maximum_two_swaps","windows":windows,**standard,"incremental_vs_WV3_661":incremental,"better_than_current_champion":better,"new_training":False,"candidate_pool_changed":False,"runtime_seconds":time.perf_counter()-started,"final_week":"not_run"};write(OUTER_REPORT,result);common.update(TRIAL,decision="promote_candidate" if better else "reject_outer",outer_MAP_by_window=standard["per_window_MAP"],mean_MAP=standard["mean_MAP"],delta_vs_WV2_601=standard["delta_vs_WV2_601"],nondegrade_windows=standard["nondegrade_windows"],worst_delta=standard["worst_delta"],runtime=result["runtime_seconds"],artifact_paths=entry["artifact_paths"]+[str(OUTER_REPORT)])
    common.log(f"{TRIAL} outer closure","The frozen two-swap policy passed both inner gates without retraining.",contract()["hypothesis"],str(OUTER_REPORT),f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-661={incremental['mean_delta_vs_WV3_661']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_661']}/4; better={better}.","Promote maximum-two-swap policy." if better else "Reject and retain one-swap WV3-661.","Close action-count expansion and proceed to final synthesis.",alternatives="No outer-driven threshold or three-swap rescue.",experiment="One outer exposure of the frozen WV3-661 model with exact two-swap evaluation.",reflection="This tests action capacity independently of representation and training.")
    print({"trial":TRIAL,"standard":standard,"incremental":incremental,"better":better},flush=True);return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["register","screen","confirm"]);globals()[parser.parse_args().command]()
