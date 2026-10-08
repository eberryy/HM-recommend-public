"""WV3-760: audit Top600 recent-popularity headroom before a two-stage cascade."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal, load_parquet
from .warm_v3_bpr_safe_admission import replace_rank12
from .warm_v3_residual_admission import INNER, source
from .warm_v3_rich_propensity import rich_candidate_frame


TRIAL = "WV3-760"
CONTRACT = common.REPORT / "WV3-760_POPULAR_CASCADE_AUDIT_CONTRACT.json"
REPORT = common.REPORT / "WV3-760_POPULAR_CASCADE_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-760_POPULAR_CASCADE_AUDIT.md"
REFERENCE_URL = "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/hello-world-2nd-place-solution"


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "inner_only_recent_popularity_Top600_candidate_headroom_and_density_audit",
        "hypothesis": (
            "A winner-style wide popularity pool is worth a two-stage cascade only if recent-popularity ranks101-600 "
            "add stable truth and constrained rank-12 oracle value beyond the current pool."
        ),
        "candidate_definition": "global article event count in [cutoff-7days, cutoff), deterministic count-desc/article-id tie order, Top600",
        "marginal_definition": "active Warm user x popular item pair absent from the frozen original candidate pool",
        "bands": {"top1_100": [1, 100], "top101_300": [101, 300], "top301_600": [301, 600]},
        "gates": {
            "marginal_truth_pairs_in_ranks101_600_each_window_min": 50,
            "rank12_constrained_oracle_mean_min": 0.0005,
            "rank12_constrained_oracle_each_window_min": 0.0002,
        },
        "training": "none",
        "outer": "not_run",
        "expected_minutes": 15,
        "success_action": "design a resource-bounded stage1 filter audit before any full two-stage training",
        "failure_action": "do not construct a Top600 cascade; retain WV3-741",
        "source_inspiration": REFERENCE_URL,
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(15)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            "recent_popularity_Top600_cascade_headroom_audit",
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_definition"],
            training_protocol="none; four 2020 inner windows only",
            params={"bands": contract()["bands"], "gates": contract()["gates"], "final_week": "not_run"},
            expected_minutes=15,
        )
    return read(CONTRACT)


def current_values(name: str, cutoff: str) -> dict[str, dict]:
    candidates, _ = rich_candidate_frame(cutoff, rank_min=1)
    swaps = load_parquet(common.ART / "WV3-740" / name / "pointwise_lambdarank_RRF60" / "swaps.parquet")
    swap_groups = {customer: group for customer, group in swaps.groupby("customer_id", sort=False)}
    states = {}
    for customer, group in candidates.groupby("customer_id", sort=False):
        ordered = group.sort_values("rf", kind="mergesort")
        values = ordered.target.to_numpy(np.uint8)
        selected = swap_groups.get(customer)
        if selected is not None:
            for row in selected.sort_values("swap_order", kind="mergesort").itertuples(index=False):
                left = int(row.victim_rank) - 1
                right = int(row.challenger_rank) - 1
                values[left], values[right] = values[right], values[left]
        states[customer] = {
            "values": values,
            "truth_count": int(ordered.truth_count.iloc[0]),
        }
    return states


def audit_window(name: str, folder: str) -> dict:
    ranks_path, meta = source(folder)
    cutoff = meta["cutoff"]
    with connection() as con:
        con.execute(
            f"""CREATE TEMP TABLE current_pool AS
            SELECT DISTINCT customer_id,article_id
            FROM read_parquet({literal(ranks_path)}) WHERE user_history_events_12w>0"""
        )
        con.execute("CREATE TEMP TABLE eligible_users AS SELECT DISTINCT customer_id FROM current_pool")
        con.execute(
            f"""CREATE TEMP TABLE popular AS WITH counts AS (
                SELECT article_id,count(*) events
                FROM read_parquet({literal(common.TX)})
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL 7 DAY AND t_dat<DATE '{cutoff}'
                GROUP BY article_id
            ) SELECT article_id,events,row_number() OVER(ORDER BY events DESC,article_id) popularity_rank
              FROM counts QUALIFY popularity_rank<=600"""
        )
        con.execute(
            f"""CREATE TEMP TABLE truth AS
            SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
        )
        con.execute(
            """CREATE TEMP TABLE marginal AS
            SELECT u.customer_id,p.article_id,p.popularity_rank,
                CAST(t.article_id IS NOT NULL AS INTEGER) AS target_flag
            FROM eligible_users u CROSS JOIN popular p
            LEFT JOIN current_pool c USING(customer_id,article_id)
            LEFT JOIN truth t USING(customer_id,article_id)
            WHERE c.article_id IS NULL"""
        )
        bands = con.execute(
            """SELECT CASE WHEN popularity_rank<=100 THEN 'top1_100'
                       WHEN popularity_rank<=300 THEN 'top101_300' ELSE 'top301_600' END band,
                count(*) candidate_pairs,count(DISTINCT customer_id) users,
                sum(target_flag) truth_pairs,count(DISTINCT customer_id) FILTER(WHERE target_flag=1) truth_users
            FROM marginal GROUP BY band ORDER BY min(popularity_rank)"""
        ).fetchdf()
        truth_users = con.execute(
            "SELECT customer_id,min(popularity_rank) best_truth_rank FROM marginal WHERE target_flag=1 GROUP BY customer_id"
        ).fetchdf()
        eligible_users = con.execute("SELECT count(*) FROM eligible_users").fetchone()[0]
        current_rows = con.execute("SELECT count(*) FROM current_pool").fetchone()[0]
    states = current_values(name, cutoff)
    gains = []
    missing_state = 0
    for row in truth_users.itertuples(index=False):
        state = states.get(row.customer_id)
        if state is None:
            missing_state += 1
            continue
        gains.append(max(0.0, replace_rank12(state["values"], 1, state["truth_count"])))
    oracle = float(np.sum(gains) / meta["total_users"])
    band_rows = {}
    for row in bands.itertuples(index=False):
        band_rows[row.band] = {
            "candidate_pairs": int(row.candidate_pairs),
            "users": int(row.users),
            "marginal_truth_pairs": int(row.truth_pairs),
            "marginal_truth_users": int(row.truth_users),
            "positive_density": float(row.truth_pairs / row.candidate_pairs),
        }
    tail_truth = band_rows["top101_300"]["marginal_truth_pairs"] + band_rows["top301_600"]["marginal_truth_pairs"]
    return {
        "window": name,
        "cutoff": cutoff,
        "total_users_denominator": meta["total_users"],
        "eligible_active_Warm_users": int(eligible_users),
        "current_pool_pairs": int(current_rows),
        "bands": band_rows,
        "marginal_truth_pairs_ranks101_600": tail_truth,
        "marginal_truth_users_Top600": len(truth_users),
        "rank12_constrained_oracle_population_delta": oracle,
        "truth_users_without_current_state": missing_state,
        "candidate_pool_changed": True,
        "new_training": False,
        "outer": "not_run",
        "final_week": "not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-760：最近热度 Top600 两阶段级联前置审计",
        "",
        "## 结论",
        "",
        f"继续设计两阶段级联：**{'通过' if result['cascade_design_authorized'] else '未通过'}**。本轮没有训练模型，也没有读取外层。",
        "",
        "## 背景与术语",
        "",
        f"H&M 第2名方案先给每位用户约600件热门商品，用第一阶段 LightGBM 缩到约130件，再补历史购买商品并训练第二阶段排序器。参考：{REFERENCE_URL}",
        "",
        "- 最近热度 Top600（本项目自定义）：按截止日前7天全体交易事件数给商品排序，事件数相同按商品编号；取前600件。",
        "- 边际候选（推荐系统常用概念）：热门商品与活跃 Warm 用户做笛卡尔积后，排除已经在冻结原候选池中的用户—商品对；统计单位是去重用户—商品对。",
        "- 正例密度（推荐训练常用诊断）：某热度名次段内，边际候选中恰好在下一周被该用户购买的比例；分母是该段全部边际用户—商品对。",
        "- 第12名受限 Oracle（本项目诊断上界）：若预知下一周标签，每用户至多挑一件边际真值替换 WV3-741 当前第12名，并允许不动作；以窗口全部真值用户为分母，不可部署。",
        "- 两阶段级联（推荐系统常见架构）：第一阶段从很宽候选池筛到较小集合，第二阶段用更丰富特征做最终 Top12 排序；可控制宽池带来的负例量和算力。",
        "",
        "| 内层窗口 | 1–100 真值/密度 | 101–300 真值/密度 | 301–600 真值/密度 | 101–600 边际真值 | 第12名 Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        bands = row["bands"]
        lines.append(
            f"| {name} | {bands['top1_100']['marginal_truth_pairs']:,} / {bands['top1_100']['positive_density']:.6%} | "
            f"{bands['top101_300']['marginal_truth_pairs']:,} / {bands['top101_300']['positive_density']:.6%} | "
            f"{bands['top301_600']['marginal_truth_pairs']:,} / {bands['top301_600']['positive_density']:.6%} | "
            f"{row['marginal_truth_pairs_ranks101_600']:,} | {row['rank12_constrained_oracle_population_delta']:+.9f} |"
        )
    lines += [
        "",
        "Oracle 只回答宽热度池是否包含可兑现目标；极低正例密度意味着不能把600件直接交给现有末级排序器，"
        "必须先证明第一阶段过滤能在严格历史监督下显著提纯。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-760 must start preregistered and unexposed")
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        windows[name] = audit_window(name, folder)
        print(
            {
                "popular_cascade_audit": name,
                "tail_truth": windows[name]["marginal_truth_pairs_ranks101_600"],
                "oracle": windows[name]["rank12_constrained_oracle_population_delta"],
            },
            flush=True,
        )
    rules = contract()["gates"]
    oracle_values = [row["rank12_constrained_oracle_population_delta"] for row in windows.values()]
    passed = bool(
        all(row["marginal_truth_pairs_ranks101_600"] >= rules["marginal_truth_pairs_in_ranks101_600_each_window_min"] for row in windows.values())
        and float(np.mean(oracle_values)) >= rules["rank12_constrained_oracle_mean_min"]
        and min(oracle_values) >= rules["rank12_constrained_oracle_each_window_min"]
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "gates": rules,
        "mean_rank12_constrained_oracle_population_delta": float(np.mean(oracle_values)),
        "minimum_rank12_constrained_oracle_population_delta": min(oracle_values),
        "cascade_design_authorized": passed,
        "new_training": False,
        "outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "source_inspiration": REFERENCE_URL,
        "final_week": "not_run",
    }
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="diagnostic_supports_popular_cascade" if passed else "diagnostic_rejects_popular_cascade",
        inner_evidence={
            "mean_oracle": result["mean_rank12_constrained_oracle_population_delta"],
            "minimum_oracle": result["minimum_rank12_constrained_oracle_population_delta"],
            "cascade_design_authorized": passed,
        },
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)],
        validity="valid_inner_candidate_headroom_audit",
    )
    common.log(
        f"{TRIAL} popular Top600 cascade audit",
        "The H&M second-place solution used an approximately 600-popular-item first stage, while our BPR expansion showed headroom but unusably low precision under direct admission.",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"mean rank12 oracle={result['mean_rank12_constrained_oracle_population_delta']:+.9f}; minimum={result['minimum_rank12_constrained_oracle_population_delta']:+.9f}; authorized={passed}.",
        "Design a bounded first-stage purity audit." if passed else "Do not build a Top600 cascade; retain WV3-741.",
        "If authorized, first test whether historical features can retain truth while reducing 600 to roughly 130; do not train the final ranker yet.",
        alternatives="Directly appending 600 candidates would reproduce the low-density failure already observed for BPR expansion.",
        experiment="Build recent-7-day global Top600 on each inner cutoff, exclude current-pool pairs, and measure truth by popularity band plus one-slot oracle.",
        reflection="Candidate breadth is useful only if it adds stable truth and an intermediate filter can control the resulting negative volume.",
    )
    return result


if __name__ == "__main__":
    register()
    run()
