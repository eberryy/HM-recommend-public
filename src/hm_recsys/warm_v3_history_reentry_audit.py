"""WV3-770: audit all-history re-entry candidates omitted by the frozen pool."""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal
from .warm_v3_bpr_safe_admission import replace_rank12
from .warm_v3_popular_cascade_audit import current_values
from .warm_v3_residual_admission import INNER, source


TRIAL = "WV3-770"
CONTRACT = common.REPORT / "WV3-770_HISTORY_REENTRY_AUDIT_CONTRACT.json"
REPORT = common.REPORT / "WV3-770_HISTORY_REENTRY_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-770_HISTORY_REENTRY_AUDIT.md"
REFERENCE_URL = "https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/hello-world-2nd-place-solution"


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "inner_only_all_history_reentry_candidate_audit",
        "hypothesis": (
            "Distinct items previously purchased before cutoff but omitted by the frozen pool form a denser and "
            "more rankable marginal source than the global Top600 popularity tail."
        ),
        "candidate_definition": (
            "for each active Warm user, every distinct pre-cutoff purchased article absent from the frozen candidate pool"
        ),
        "recency_bands_days": {"0_28": [0, 28], "29_84": [29, 84], "85_180": [85, 180], "181_plus": [181, None]},
        "gates": {
            "marginal_truth_pairs_each_window_min": 50,
            "positive_density_lift_vs_Top600_each_window_min": 2.0,
            "rank12_constrained_oracle_mean_min": 0.0005,
            "rank12_constrained_oracle_each_window_min": 0.0002,
        },
        "training": "none",
        "outer": "not_run",
        "expected_minutes": 10,
        "success_action": "audit source-specific supervision and feature availability before training",
        "failure_action": "close all-history re-entry as an expansion source and retain WV3-741",
        "source_inspiration": REFERENCE_URL,
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(10)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            "all_history_reentry_candidate_audit",
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_definition"],
            training_protocol="none; four 2020 inner windows only",
            params={"recency_bands_days": contract()["recency_bands_days"], "gates": contract()["gates"], "final_week": "not_run"},
            expected_minutes=10,
        )
    return read(CONTRACT)


def audit_window(name: str, folder: str, top600_density: float) -> dict:
    ranks_path, meta = source(folder)
    cutoff = meta["cutoff"]
    with connection() as con:
        con.execute(
            f"""CREATE TEMP TABLE current_pool AS SELECT DISTINCT customer_id,article_id
            FROM read_parquet({literal(ranks_path)}) WHERE user_history_events_12w>0"""
        )
        con.execute("CREATE TEMP TABLE eligible_users AS SELECT DISTINCT customer_id FROM current_pool")
        con.execute(
            f"""CREATE TEMP TABLE truth AS SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
        )
        con.execute(
            f"""CREATE TEMP TABLE marginal AS WITH purchased AS (
                SELECT t.customer_id,t.article_id,count(*) history_events,
                    date_diff('day',max(t_dat),DATE '{cutoff}') last_purchase_age_days
                FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
                WHERE t_dat<DATE '{cutoff}' GROUP BY t.customer_id,t.article_id
            ) SELECT p.*,CAST(t.article_id IS NOT NULL AS INTEGER) target_flag
            FROM purchased p LEFT JOIN current_pool c USING(customer_id,article_id)
            LEFT JOIN truth t USING(customer_id,article_id) WHERE c.article_id IS NULL"""
        )
        totals = con.execute(
            """SELECT count(*) AS pair_count,count(DISTINCT customer_id) AS user_count,
                sum(target_flag) AS truth_pair_count,
                count(DISTINCT customer_id) FILTER(WHERE target_flag=1) AS truth_user_count
            FROM marginal"""
        ).fetchone()
        bands = con.execute(
            """SELECT CASE WHEN last_purchase_age_days<=28 THEN '0_28'
                       WHEN last_purchase_age_days<=84 THEN '29_84'
                       WHEN last_purchase_age_days<=180 THEN '85_180' ELSE '181_plus' END AS recency_band,
                count(*) AS pair_count,sum(target_flag) AS truth_pair_count,
                count(DISTINCT customer_id) FILTER(WHERE target_flag=1) AS truth_user_count
            FROM marginal GROUP BY recency_band ORDER BY min(last_purchase_age_days)"""
        ).fetchdf()
        truth_users = con.execute(
            "SELECT customer_id,min(last_purchase_age_days) AS best_truth_recency FROM marginal WHERE target_flag=1 GROUP BY customer_id"
        ).fetchdf()
    states = current_values(name, cutoff)
    gains = []
    missing_state = 0
    for row in truth_users.itertuples(index=False):
        state = states.get(row.customer_id)
        if state is None:
            missing_state += 1
            continue
        gains.append(max(0.0, replace_rank12(state["values"], 1, state["truth_count"])))
    density = float(totals[2] / totals[0]) if totals[0] else 0.0
    band_rows = {}
    for row in bands.itertuples(index=False):
        band_rows[row.recency_band] = {
            "candidate_pairs": int(row.pair_count),
            "marginal_truth_pairs": int(row.truth_pair_count),
            "marginal_truth_users": int(row.truth_user_count),
            "positive_density": float(row.truth_pair_count / row.pair_count),
        }
    return {
        "window": name,
        "cutoff": cutoff,
        "total_users_denominator": int(meta["total_users"]),
        "candidate_pairs": int(totals[0]),
        "candidate_users": int(totals[1]),
        "marginal_truth_pairs": int(totals[2]),
        "marginal_truth_users": int(totals[3]),
        "positive_density": density,
        "Top600_positive_density_reference": top600_density,
        "positive_density_lift_vs_Top600": float(density / top600_density),
        "recency_bands": band_rows,
        "rank12_constrained_oracle_population_delta": float(np.sum(gains) / meta["total_users"]),
        "truth_users_without_current_state": missing_state,
        "candidate_pool_changed": True,
        "new_training": False,
        "outer": "not_run",
        "final_week": "not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-770：历史购买商品回补候选审计",
        "",
        "## 结论",
        "",
        f"授权继续：**{'通过' if result['training_design_authorized'] else '未通过'}**。本轮没有训练模型，也没有读取外层。",
        "",
        "## 术语",
        "",
        "- 历史购买商品回补（业界常见候选策略）：对每位有近期行为的 Warm 用户，取截止日前曾购买、但冻结候选池未包含的去重商品；统计单位为用户—商品对。",
        "- 边际真值（本项目诊断）：上述新增用户—商品对中，用户在截止日后7天真实再次购买的商品对；分母是本策略新增的全部用户—商品对。",
        "- 正例密度倍数（本项目诊断）：历史回补候选的边际真值比例 / 同窗口最近热度 Top600 边际候选的真值比例。",
        "- 第12名受限 Oracle（本项目不可部署上界）：预知标签后，每位用户最多拿一件历史回补真值替换 WV3-741 当前第12名；以窗口全部真值用户数为分母。",
        "- 最近购买年龄（推荐系统常见时效量）：候选商品在截止日前最后一次被该用户购买距截止日的天数；按0–28、29–84、85–180、181天以上分组。",
        "",
        "| 内层窗口 | 新增候选 | 边际真值 | 正例密度 | 相对 Top600 密度 | 第12名 Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['candidate_pairs']:,} | {row['marginal_truth_pairs']:,} | "
            f"{row['positive_density']:.6%} | {row['positive_density_lift_vs_Top600']:.2f}x | "
            f"{row['rank12_constrained_oracle_population_delta']:+.9f} |"
        )
    lines += [
        "",
        "若失败，则不能因为冠军方案使用过历史回补就直接照搬：当前冻结池已覆盖一部分复购商品，真正需要评估的是剩余边际部分。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-770 must start preregistered and unexposed")
    top600 = read(common.REPORT / "WV3-761_SCREEN.json")["windows"]
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        windows[name] = audit_window(name, folder, top600[name]["full_positive_density"])
        print({"history_reentry_audit": name, "truth": windows[name]["marginal_truth_pairs"], "density_lift": windows[name]["positive_density_lift_vs_Top600"], "oracle": windows[name]["rank12_constrained_oracle_population_delta"]}, flush=True)
    rules = contract()["gates"]
    oracle = [row["rank12_constrained_oracle_population_delta"] for row in windows.values()]
    passed = bool(
        min(row["marginal_truth_pairs"] for row in windows.values()) >= rules["marginal_truth_pairs_each_window_min"]
        and min(row["positive_density_lift_vs_Top600"] for row in windows.values()) >= rules["positive_density_lift_vs_Top600_each_window_min"]
        and float(np.mean(oracle)) >= rules["rank12_constrained_oracle_mean_min"]
        and min(oracle) >= rules["rank12_constrained_oracle_each_window_min"]
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "mean_rank12_constrained_oracle_population_delta": float(np.mean(oracle)),
        "minimum_rank12_constrained_oracle_population_delta": min(oracle),
        "gates": rules,
        "training_design_authorized": passed,
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
        decision="diagnostic_supports_history_reentry" if passed else "diagnostic_rejects_history_reentry",
        inner_evidence={"mean_oracle": result["mean_rank12_constrained_oracle_population_delta"], "minimum_oracle": result["minimum_rank12_constrained_oracle_population_delta"], "authorized": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)],
        validity="valid_cutoff_safe_inner_candidate_audit",
    )
    common.log(
        f"{TRIAL} all-history re-entry audit",
        "WV3-761 could not retain enough Top600 truth at Top130, while the H&M second-place cascade explicitly appended user purchase history after first-stage filtering.",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"mean rank12 oracle={result['mean_rank12_constrained_oracle_population_delta']:+.9f}; minimum={result['minimum_rank12_constrained_oracle_population_delta']:+.9f}; authorized={passed}.",
        "Audit source-specific supervision before training." if passed else "Close all-history re-entry and retain WV3-741.",
        "Do not train unless all four density and oracle gates pass; do not inspect outer.",
        alternatives="This is a distinct personalized repeat source, not a Top600 filter rescue or parameter change.",
        experiment="Measure all pre-cutoff purchased items absent the frozen pool, their future repeat density by recency band, and one-slot oracle on four inner windows.",
        reflection="A published architecture component is reusable only if its marginal contribution survives the current pool definition.",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
