"""WV3-680: rich-feature within-user LambdaRank for protected one-swap admission."""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES, historical_training_data as rich_historical_data, rich_candidate_frame


TRIAL = "WV3-680"
CONTRACT = common.REPORT / "WV3-680_RICH_LAMBDARANK_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-680_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-680_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-680_OUTER.json"
MODEL_ROOT = common.ART / TRIAL
PARAMS = {
    "objective": "lambdarank",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 500,
    "lambda_l2": 10.0,
    "lambdarank_truncation_level": 12,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 20260910,
    "feature_fraction_seed": 20260910,
    "bagging_seed": 20260910,
    "verbosity": -1,
}
ROUNDS = 100


def effective_training_data() -> tuple[pd.DataFrame, np.ndarray, dict]:
    training, source_audit = rich_historical_data()
    keys = [training.source_window, training.customer_id]
    group_positive = training.groupby(keys, sort=False).target.transform("sum")
    group_size = training.groupby(keys, sort=False).target.transform("size")
    effective = training[(group_positive > 0) & (group_positive < group_size)].copy()
    ids = (effective.source_window.astype(str) + "|" + effective.customer_id.astype(str)).to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    groups = np.diff(boundaries).astype(np.int32)
    audit = {
        "role": "historical_within_user_ranking_supervision",
        "source_rows": len(training),
        "effective_rows": len(effective),
        "effective_groups": len(groups),
        "positive_rows": int(effective.target.sum()),
        "negative_rows": int((effective.target == 0).sum()),
        "minimum_group_size": int(groups.min()) if len(groups) else 0,
        "maximum_group_size": int(groups.max()) if len(groups) else 0,
        "group_rows_conserved": int(groups.sum()) == len(effective),
        "all_groups_have_positive_and_negative": bool(
            effective.groupby(["source_window", "customer_id"], sort=False).target.agg(["sum", "size"]).eval("sum>0 and sum<size").all()
        ),
        "features": FEATURES,
        "target_excluded": "target" not in FEATURES,
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "final_week": "not_run",
        "rich_source_audit": source_audit,
    }
    audit["passed"] = bool(audit["effective_groups"] >= 3000 and audit["group_rows_conserved"] and audit["all_groups_have_positive_and_negative"] and audit["target_excluded"])
    return effective, groups, audit


def fit_model(training: pd.DataFrame, groups: np.ndarray) -> lgb.Booster:
    dataset = lgb.Dataset(training[FEATURES].to_numpy(np.float32), label=training.target.to_numpy(np.uint8), group=groups, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def choose_rank_decisions(pairs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    challenger = scores.rename(columns={"article_id": "challenger_article_id", "ranking_score": "challenger_score"})
    victim = scores.rename(columns={"article_id": "victim_article_id", "ranking_score": "victim_score"})
    columns = ["customer_id", "challenger_article_id", "victim_article_id", "challenger_target", "victim_target", "challenger_rank", "victim_rank", "unit_gain"]
    chosen = pairs[columns].merge(challenger, on=["customer_id", "challenger_article_id"], how="left", validate="many_to_one").merge(victim, on=["customer_id", "victim_article_id"], how="left", validate="many_to_one")
    assert chosen[["challenger_score", "victim_score"]].notna().all().all()
    chosen["score_difference"] = chosen.challenger_score - chosen.victim_score
    chosen["expected_ordering_gain"] = chosen.score_difference * chosen.unit_gain
    chosen = chosen[chosen.expected_ordering_gain > 0]
    chosen = chosen.sort_values(["customer_id", "expected_ordering_gain", "score_difference", "victim_rank", "challenger_rank", "challenger_article_id", "victim_article_id"], ascending=[True, False, False, False, True, True, True], kind="mergesort").drop_duplicates("customer_id", keep="first")
    chosen["actual_delta"] = (chosen.challenger_target.astype(np.int8) - chosen.victim_target.astype(np.int8)) * chosen.unit_gain
    return chosen.sort_values("customer_id", kind="mergesort").reset_index(drop=True)


def evaluate(folder: str, name: str, model: lgb.Booster, stage: str):
    path, meta = source(folder)
    started = time.perf_counter()
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=8)
    scores = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    score_frame = candidates[["customer_id", "article_id"]].copy()
    score_frame["ranking_score"] = scores
    pairs = pair_frame(path, discordant_only=False)
    chosen = choose_rank_decisions(pairs, score_frame)
    beneficial = chosen.actual_delta > 1e-15
    harmful = chosen.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    delta = float(chosen.actual_delta.sum() / meta["total_users"])
    baseline = float(meta["baseline_map_population_component"])
    return {"window": name, "cutoff": meta["cutoff"], "role": stage, "total_users_denominator": meta["total_users"], "candidate_rows": len(candidates), "pair_rows": len(pairs), "selected_users": len(chosen), "beneficial_selected_users": int(beneficial.sum()), "harmful_selected_users": int(harmful.sum()), "neutral_selected_users": int(neutral.sum()), "selected_neutral_share": float(neutral.mean()) if len(chosen) else 0.0, "gross_positive_MAP": float(chosen.loc[beneficial, "actual_delta"].sum() / meta["total_users"]), "gross_negative_MAP": float(chosen.loc[harmful, "actual_delta"].sum() / meta["total_users"]), "baseline_MAP@12": baseline, "MAP@12": baseline + delta, "population_delta": delta, "score_quantiles": np.quantile(scores, [0, 0.1, 0.5, 0.9, 1]).tolist(), "protected_ranks1_7_changes": 0, "maximum_swaps_per_user": 1, "candidate_input": details, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}, chosen


def experiment_contract() -> dict:
    return {
        "created_at": now(), "experiment_id": TRIAL, "parent": "WV3-661", "architecture_family": "rich_feature_within_user_lambdarank_one_swap",
        "hypothesis": "Optimizing within-user candidate ordering on effective positive-negative groups will use sparse labels more efficiently than global binary cross-entropy and improve the same rich one-swap policy.",
        "training": "four 2019 sources; keep only user-window groups containing at least one positive and one negative Top50 candidate; LambdaRank over fixed99 features",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "score_use": "within-user challenger-minus-victim LambdaRank score times exact position unit_gain; score is not a calibrated probability",
        "candidate_pool": "WV3-661 Top50; ranks1-7 protected, challengers13-50, victims8-12, maximum one swap",
        "input_gate": "at least3000 effective user-window groups, each with positive and negative, group rows conserved, target excluded",
        "inner_gate": {"standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005", "incremental_vs_WV3_661": "mean>0, >=3 nondegrade, worst>=-0.0002"},
        "expected_minutes": 30,
        "outer_policy": "one frozen exposure only if input and both inner gates pass; no group filter, objective, threshold, feature or tree rescue",
        "distinction_from_WV3_381": "WV3-381 used 11 rank-output features and broadly reranked a Top12 union; WV3-680 uses99 candidate features and permits only one protected tail swap",
        "fallback": "retain WV3-661; close rich LambdaRank objective if it fails",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup(); common.budget(30)
    if not CONTRACT.exists(): write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(TRIAL, "rich_feature_within_user_lambdarank_one_swap", experiment_contract()["hypothesis"], candidate_protocol=experiment_contract()["candidate_pool"], training_protocol=experiment_contract()["training"], features=FEATURES, params={"model": experiment_contract()["model"], "inner_gate": experiment_contract()["inner_gate"], "final_week": "not_run"}, expected_minutes=30)


def incremental_gate(windows: dict) -> dict:
    prior = read(common.REPORT / "WV3-661_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}; values = list(delta.values())
    return {"per_window_delta_vs_WV3_661": delta, "mean_delta_vs_WV3_661": float(np.mean(values)), "nondegrade_windows_vs_WV3_661": sum(value >= 0 for value in values), "worst_delta_vs_WV3_661": min(values), "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002)}


def render_screen(result: dict) -> str:
    s=result["standard_screening_vs_WV2_601"]; i=result["incremental_screening_vs_WV3_661"]; a=result["training_audit"]
    lines=["# WV3-680 丰富特征组内 LambdaRank 局部准入","","## 结论","",f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {s['mean_population_delta']:+.9f}；相对 WV3-661 平均增加 {i['mean_delta_vs_WV3_661']:+.9f}。","","## 术语与监督口径","","- LambdaRank（行业通用学习排序方法）：按同一用户的候选组构造梯度，直接学习正例在负例之前；模型输出只表示组内排序强弱，不解释为购买概率。","- 有效用户组（本项目自定义）：同一历史窗口、同一用户的 Top50 候选中至少一件未来购买正例且至少一件负例；全负组没有组内排序梯度，因此训练前固定排除。","- 组内分数差准入（本项目自定义）：挑战商品分数减被替换商品分数，再乘位置 AP 权重；只执行最佳正值的一次换位，第1–7名不变。","- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。","",f"历史输入 {a['source_rows']:,} 行；有效监督 {a['effective_groups']:,} 个用户—窗口组、{a['effective_rows']:,} 行，其中正例 {a['positive_rows']:,}。","","| 内层窗口 | 选择用户数 | 有益 / 有害 / 中性 | 中性占比 | 相对 WV2-601 | 相对 WV3-661 |","|---|---:|---:|---:|---:|---:|"]
    for name,row in result["windows"].items(): lines.append(f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | {row['selected_neutral_share']:.2%} | {row['population_delta']:+.9f} | {i['per_window_delta_vs_WV3_661'][name]:+.9f} |")
    lines += ["","只有两个内层门槛同时通过才允许一次外层确认；失败后不改变有效组定义、分数阈值或树参数。最终周 `2020-09-16` 保持 `not_run`。",""]
    return "\n".join(lines)


def screen() -> dict:
    common.setup(); common.budget(30)
    entry=next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"]==TRIAL); assert entry["decision"]=="preregistered" and entry["outer_exposures"]==0
    started=time.perf_counter(); training,groups,audit=effective_training_data(); MODEL_ROOT.mkdir(parents=True,exist_ok=True); write(MODEL_ROOT/"TRAINING_AUDIT.json",audit)
    if not audit["passed"]:
        common.update(TRIAL,decision="reject_input_audit",inner_evidence=audit,runtime=time.perf_counter()-started,artifact_paths=[str(MODEL_ROOT/"TRAINING_AUDIT.json")]); raise RuntimeError("WV3-680 input gate failed")
    model=fit_model(training,groups); del training,groups; gc.collect(); model_path=MODEL_ROOT/"MODEL.txt"; model.save_model(str(model_path)); write(MODEL_ROOT/"MODEL.json",{"experiment_id":TRIAL,"model":evidence_id(model_path,reason="explicit_registry_evidence"),"features":FEATURES,"params":PARAMS,"rounds":ROUNDS,"training_audit":str(MODEL_ROOT/"TRAINING_AUDIT.json"),"outer_labels_used":False,"final_week":"not_run"})
    windows={}
    for name,folder in INNER:
        row,chosen=evaluate(folder,name,model,"rich_lambdarank_inner_screen"); windows[name]=row; root=MODEL_ROOT/name/"inner"; root.mkdir(parents=True,exist_ok=True); save_parquet(chosen,root/"decisions.parquet"); write(root/"REVIEW.json",row); print({"rich_lambdarank_inner":name,"delta":row["population_delta"],"selected":row["selected_users"]},flush=True)
    standard=screening_gate(row["population_delta"] for row in windows.values()); incremental=incremental_gate(windows); passed=standard["passed"] and incremental["passed"]
    result={"created_at":now(),"experiment_id":TRIAL,"architecture":"rich_feature_within_user_lambdarank_one_swap","training_audit":audit,"windows":windows,"standard_screening_vs_WV2_601":standard,"incremental_screening_vs_WV3_661":incremental,"passed":passed,"candidate_pool_changed":False,"runtime_seconds":time.perf_counter()-started,"final_week":"not_run"}; write(SCREEN_REPORT,result); SCREEN_MARKDOWN.write_text(render_screen(result),encoding="utf-8")
    common.update(TRIAL,decision="inner_pass" if passed else "reject_inner",inner_evidence={"input":audit,"standard":standard,"incremental":incremental,"passed":passed},runtime=result["runtime_seconds"],artifact_paths=[str(SCREEN_REPORT),str(SCREEN_MARKDOWN),str(MODEL_ROOT/"MODEL.json")])
    common.log(f"{TRIAL} inner screen","The stable rich binary model optimizes a global sparse classification loss although deployment only compares candidates within a user.",experiment_contract()["hypothesis"],f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",f"input pass={audit['passed']}; effective groups={audit['effective_groups']}; standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_661']:+.9f}; pass={passed}.","Freeze and expose once." if passed else "Reject the group-ranking objective without rescue.","Run one outer confirmation if both gates pass; otherwise retain WV3-661.",alternatives="This is not WV3-381: representation and protected action differ while only the objective changes versus WV3-661.",experiment="Train fixed rich LambdaRank on historical effective user groups and apply one score-difference swap.",reflection="The input gate ensures sparse global labels still form enough within-user ranking groups.")
    print({"trial":TRIAL,"standard":standard,"incremental":incremental,"passed":passed,"seconds":result["runtime_seconds"]},flush=True); return result


def confirm() -> dict:
    common.setup(); common.budget(20)
    entry=next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"]==TRIAL); assert entry["decision"]=="inner_pass" and entry["outer_exposures"]==1; assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists(): return read(OUTER_REPORT)
    model=lgb.Booster(model_file=str(MODEL_ROOT/"MODEL.txt")); started=time.perf_counter(); windows={}
    for name,folder in OUTER:
        row,chosen=evaluate(folder,name,model,"frozen_rich_lambdarank_outer"); windows[name]=row; root=MODEL_ROOT/name/"outer"; root.mkdir(parents=True,exist_ok=True); save_parquet(chosen,root/"decisions.parquet"); write(root/"REVIEW.json",row); print({"rich_lambdarank_outer":name,"delta":row["population_delta"],"selected":row["selected_users"]},flush=True)
    standard=common.summary({name:row["MAP@12"] for name,row in windows.items()},2020); prior=read(common.REPORT/"WV3-661_OUTER.json"); delta={name:windows[name]["MAP@12"]-prior["per_window_MAP"][name] for name in windows}; values=list(delta.values()); incremental={"per_window_delta_vs_WV3_661":delta,"mean_delta_vs_WV3_661":float(np.mean(values)),"nondegrade_windows_vs_WV3_661":sum(value>=0 for value in values),"worst_delta_vs_WV3_661":min(values)}; better=bool(standard["stable"] and np.mean(values)>0 and sum(value>=0 for value in values)>=3 and min(values)>=-.0002)
    result={"created_at":now(),"experiment_id":TRIAL,"architecture":"rich_feature_within_user_lambdarank_one_swap","windows":windows,**standard,"incremental_vs_WV3_661":incremental,"better_than_current_champion":better,"candidate_pool_changed":False,"runtime_seconds":time.perf_counter()-started,"final_week":"not_run"}; write(OUTER_REPORT,result); common.update(TRIAL,decision="promote_candidate" if better else "reject_outer",outer_MAP_by_window=standard["per_window_MAP"],mean_MAP=standard["mean_MAP"],delta_vs_WV2_601=standard["delta_vs_WV2_601"],nondegrade_windows=standard["nondegrade_windows"],worst_delta=standard["worst_delta"],runtime=result["runtime_seconds"],artifact_paths=entry["artifact_paths"]+[str(OUTER_REPORT)])
    common.log(f"{TRIAL} outer closure","The within-user rich ranker passed its input and both inner gates before freezing.",experiment_contract()["hypothesis"],str(OUTER_REPORT),f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-661={incremental['mean_delta_vs_WV3_661']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_661']}/4; better={better}.","Promote the rich LambdaRank admission." if better else "Reject and retain WV3-661.","Continue only if another evidence-backed bounded mechanism remains within budget.",alternatives="No outer-driven group filter, score threshold, feature or tree rescue is allowed.",experiment="One outer exposure of the exact frozen group-ranking model.",reflection="This tests whether the objective-level improvement transfers temporally.")
    print({"trial":TRIAL,"standard":standard,"incremental":incremental,"better":better},flush=True); return result


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("command",choices=["register","screen","confirm"]); globals()[parser.parse_args().command]()
