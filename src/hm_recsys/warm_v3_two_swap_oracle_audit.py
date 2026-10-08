"""WV3-690: read-only marginal oracle audit for a second protected tail swap."""
from __future__ import annotations

import time

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal
from .warm_v3_residual_admission import INNER, source
from .warm_v3_residual_admission_audit import apk_from_targets


TRIAL="WV3-690"
CONTRACT=common.REPORT/"WV3-690_TWO_SWAP_ORACLE_AUDIT_CONTRACT.json"
OUTPUT=common.REPORT/"WV3-690_TWO_SWAP_ORACLE_AUDIT.json"
MARKDOWN=common.REPORT/"WV3-690_TWO_SWAP_ORACLE_AUDIT.md"


def oracle_delta(targets,truth_count,max_swaps):
    values=np.asarray(targets,dtype=np.uint8); baseline=apk_from_targets(values,truth_count)
    victims=(np.flatnonzero(values[7:12]==0)+7).tolist(); challengers=(np.flatnonzero(values[12:50]==1)+12).tolist()
    count=min(max_swaps,len(victims),len(challengers)); changed=values.copy()
    for victim,challenger in zip(victims[:count],challengers[:count]): changed[victim],changed[challenger]=changed[challenger],changed[victim]
    return apk_from_targets(changed,truth_count)-baseline,count


def contract():
    return {"created_at":now(),"experiment_id":TRIAL,"role":"read_only_second_swap_marginal_oracle","hypothesis":"A second disjoint admission from ranks13-50 into ranks8-12 adds enough cross-window AP headroom to justify testing the already frozen rich propensity scorer with maximum two swaps.","comparison":"exact no-op-or-up-to-one versus no-op-or-up-to-two swaps; ranks1-7 fixed","gate":{"mean_incremental_oracle_delta_min":.0003,"each_window_incremental_oracle_delta_min":.0001,"each_window_users_using_second_swap_min":50},"training":"none","new_outer_exposure":0,"expected_minutes":5,"fallback":"keep maximum one swap and close action-count expansion","final_week":"2020-09-16 not_run"}


def register():
    common.setup();common.budget(5)
    if not CONTRACT.exists():write(CONTRACT,contract())
    registry=read(common.REGISTRY)
    if not any(row["experiment_id"]==TRIAL for row in registry["trials"]): common.register(TRIAL,"second_swap_marginal_oracle_audit",contract()["hypothesis"],candidate_protocol="unchanged Top50; exact inner oracle, ranks1-7 fixed",training_protocol="none; no outer labels",params={"gate":contract()["gate"],"final_week":"not_run"},expected_minutes=5)


def run():
    common.setup();common.budget(5);entry=next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"]==TRIAL);assert entry["decision"]=="preregistered" and entry["outer_exposures"]==0
    started=time.perf_counter();windows={}
    for name,folder in INNER:
        path,meta=source(folder)
        with connection() as con: frame=con.execute(f"SELECT customer_id,ap_rf,target,truth_count,user_history_events_12w FROM read_parquet({literal(path)}) WHERE ap_rf<=50 ORDER BY customer_id,ap_rf,article_id").fetchdf()
        one=two=0.;second_users=0
        for _,g in frame.groupby("customer_id",sort=False):
            if int(g.user_history_events_12w.iloc[0])<=0:continue
            values=g.target.to_numpy(np.uint8);truth_count=int(g.truth_count.iloc[0]);d1,_=oracle_delta(values,truth_count,1);d2,count=oracle_delta(values,truth_count,2);one+=d1;two+=d2;second_users+=count>=2
        windows[name]={"window":name,"cutoff":meta["cutoff"],"total_users_denominator":meta["total_users"],"one_swap_oracle_delta":one/meta["total_users"],"two_swap_oracle_delta":two/meta["total_users"],"incremental_second_swap_oracle_delta":(two-one)/meta["total_users"],"users_whose_oracle_uses_second_swap":second_users,"protected_ranks1_7_changes":0,"final_week":"not_run"};print({"two_swap_oracle":name,"incremental":windows[name]["incremental_second_swap_oracle_delta"],"users":second_users},flush=True)
    values=[r["incremental_second_swap_oracle_delta"] for r in windows.values()];gate=contract()["gate"];passed=bool(np.mean(values)>=gate["mean_incremental_oracle_delta_min"] and all(v>=gate["each_window_incremental_oracle_delta_min"] for v in values) and all(r["users_whose_oracle_uses_second_swap"]>=gate["each_window_users_using_second_swap_min"] for r in windows.values()))
    result={"created_at":now(),"experiment_id":TRIAL,"windows":windows,"mean_incremental_second_swap_oracle_delta":float(np.mean(values)),"minimum_incremental_second_swap_oracle_delta":min(values),"two_swap_model_authorized":passed,"runtime_seconds":time.perf_counter()-started,"new_training":False,"new_outer_exposure":0,"final_week":"not_run"};write(OUTPUT,result)
    lines=["# WV3-690 第二次尾部换位边际上限审计","","## 结论","",f"最多两次换位实验授权门槛：**{'pass' if passed else 'fail'}**。相对最多一次换位，第二次换位的四窗平均额外 Oracle MAP@12 上限为 {result['mean_incremental_second_swap_oracle_delta']:+.9f}。","","## 术语与口径","","- 最多两次换位（本项目自定义）：第1–7名不变；从第13–50名最多取两件商品，分别替换第8–12名中的不同商品；不足两次有益机会时允许一次或不操作。","- 第二次换位边际 Oracle（本项目自定义理想上限）：知道真值后，最多两次换位的最佳 MAP 增益减最多一次换位的最佳 MAP 增益；分母为窗口全部真值用户。","- 使用第二次换位的用户：Oracle 至少找到两件第13–50名真值商品以及两件第8–12名非真值商品；人数按窗口独立计数。","","| 内层窗口 | 一次换位上限 | 两次换位上限 | 第二次额外上限 | 使用第二次的用户 |","|---|---:|---:|---:|---:|"]
    for name,row in windows.items():lines.append(f"| {name} | {row['one_swap_oracle_delta']:+.9f} | {row['two_swap_oracle_delta']:+.9f} | {row['incremental_second_swap_oracle_delta']:+.9f} | {row['users_whose_oracle_uses_second_swap']:,} |")
    lines += ["","本审计只判断动作空间是否值得扩大，没有训练、没有读取外层。若通过，只允许复用已冻结 WV3-661 分数测试固定最多两次换位；失败后不尝试三至五次。最终周 `2020-09-16` 保持 `not_run`。",""];MARKDOWN.write_text("\n".join(lines),encoding="utf-8")
    common.update(TRIAL,decision="diagnostic_supports_two_swap" if passed else "diagnostic_rejects_two_swap",inner_evidence={"mean_incremental":result["mean_incremental_second_swap_oracle_delta"],"minimum_incremental":result["minimum_incremental_second_swap_oracle_delta"],"authorized":passed},runtime=result["runtime_seconds"],artifact_paths=[str(OUTPUT),str(MARKDOWN)])
    common.log(f"{TRIAL} second-swap oracle audit","The stable rich propensity scorer is restricted to one tail replacement, so action capacity remained a possible bottleneck.",contract()["hypothesis"],str(OUTPUT),f"mean incremental oracle={result['mean_incremental_second_swap_oracle_delta']:+.9f}; minimum={result['minimum_incremental_second_swap_oracle_delta']:+.9f}; authorized={passed}.","Preregister one maximum-two-swap policy." if passed else "Keep one swap and close action-count expansion.","Reuse the frozen scorer only if exact extra headroom clears the fixed gate.",alternatives="Top100 increased reach but converted poorly; this audit changes action count inside the stronger Top50 representation instead.",experiment="Read-only exact one-swap versus two-swap oracle on four inner windows.",reflection="The discrete next action count is tested before any model scoring or outer access.")
    print({"trial":TRIAL,"mean_incremental":result["mean_incremental_second_swap_oracle_delta"],"authorized":passed},flush=True);return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["register","run"]);globals()[parser.parse_args().command]()
