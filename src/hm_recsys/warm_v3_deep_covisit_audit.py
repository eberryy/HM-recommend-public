"""WV3-780: audit deeper same-day co-visitation candidates beyond the frozen source."""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import literal
from .warm_v3_bpr_safe_admission import replace_rank12
from .warm_v3_pool import connection
from .warm_v3_popular_cascade_audit import current_values
from .warm_v3_residual_admission import INNER, source


TRIAL = "WV3-780"
CONTRACT = common.REPORT / "WV3-780_DEEP_COVISIT_AUDIT_CONTRACT.json"
REPORT = common.REPORT / "WV3-780_DEEP_COVISIT_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-780_DEEP_COVISIT_AUDIT.md"
NEIGHBOR_K = 200
SOURCE_K = 200


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "inner_only_deep_same_day_covisit_candidate_audit",
        "hypothesis": (
            "The frozen same-day co-visitation source truncates both item neighbors and per-user output at 50; "
            "ranks 51-200 may contain a denser, more personalized marginal source than global Top600 popularity."
        ),
        "candidate_definition": (
            "same original 28-day same-user/same-day graph, 2-20 distinct items per day, minimum two pair-days, "
            "cosine support normalization, 200 neighbors per seed and 200 aggregated candidates per active Warm user"
        ),
        "candidate_rank_bands": {"1_50": [1, 50], "51_100": [51, 100], "101_200": [101, 200]},
        "gates": {
            "marginal_truth_pairs_ranks51_200_each_window_min": 50,
            "positive_density_lift_vs_Top600_each_window_min": 1.5,
            "rank12_constrained_oracle_mean_min": 0.0005,
            "rank12_constrained_oracle_each_window_min": 0.0002,
        },
        "training": "none",
        "outer": "not_run",
        "expected_minutes": 20,
        "success_action": "audit deep-co-vis feature reconstruction and source-specific supervision before training",
        "failure_action": "close deeper co-visitation expansion and retain WV3-741",
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(20)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            "deep_same_day_covisit_candidate_audit",
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_definition"],
            training_protocol="none; four 2020 inner windows only",
            params={"neighbor_k": NEIGHBOR_K, "source_k": SOURCE_K, "gates": contract()["gates"], "final_week": "not_run"},
            expected_minutes=20,
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
            f"""CREATE TEMP TABLE valid_user_days AS SELECT customer_id,t_dat,count(DISTINCT article_id) distinct_items
            FROM read_parquet({literal(common.TX)})
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t_dat<DATE '{cutoff}'
            GROUP BY customer_id,t_dat HAVING distinct_items BETWEEN 2 AND 20"""
        )
        con.execute(
            f"""CREATE TEMP TABLE user_day_items AS SELECT DISTINCT t.customer_id,t.t_dat,t.article_id
            FROM read_parquet({literal(common.TX)}) t SEMI JOIN valid_user_days d
            ON t.customer_id=d.customer_id AND t.t_dat=d.t_dat
            WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t.t_dat<DATE '{cutoff}'"""
        )
        con.execute("CREATE TEMP TABLE item_support AS SELECT article_id,count(*) item_days FROM user_day_items GROUP BY article_id")
        con.execute(
            f"""CREATE TEMP TABLE neighbors AS WITH pairs AS (
                SELECT l.article_id left_id,r.article_id right_id,count(*) pair_days
                FROM user_day_items l JOIN user_day_items r
                ON l.customer_id=r.customer_id AND l.t_dat=r.t_dat AND l.article_id<r.article_id
                GROUP BY l.article_id,r.article_id HAVING pair_days>=2
            ), directed AS (
                SELECT left_id seed_article_id,right_id candidate_article_id,pair_days FROM pairs
                UNION ALL SELECT right_id,left_id,pair_days FROM pairs
            ), scored AS (
                SELECT d.*,d.pair_days/sqrt(s.item_days*c.item_days) similarity
                FROM directed d JOIN item_support s ON d.seed_article_id=s.article_id
                JOIN item_support c ON d.candidate_article_id=c.article_id
            ) SELECT *,row_number() OVER(PARTITION BY seed_article_id
                ORDER BY similarity DESC,pair_days DESC,candidate_article_id) neighbor_rank
            FROM scored QUALIFY neighbor_rank<={NEIGHBOR_K}"""
        )
        con.execute(
            f"""CREATE TEMP TABLE candidates AS WITH seeds AS (
                SELECT t.customer_id,t.article_id seed_article_id,
                    pow(2.0,-date_diff('day',max(t_dat),DATE '{cutoff}')/14.0)*(1.0+ln(count(*))) seed_score
                FROM read_parquet({literal(common.TX)}) t SEMI JOIN eligible_users u USING(customer_id)
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL 12 WEEK AND t_dat<DATE '{cutoff}'
                GROUP BY t.customer_id,t.article_id
            ), scored AS (
                SELECT s.customer_id,n.candidate_article_id article_id,
                    sum(s.seed_score*n.similarity) source_score,max(n.pair_days) max_pair_days
                FROM seeds s JOIN neighbors n USING(seed_article_id)
                GROUP BY s.customer_id,n.candidate_article_id
            ) SELECT *,row_number() OVER(PARTITION BY customer_id
                ORDER BY source_score DESC,max_pair_days DESC,article_id) source_rank
            FROM scored QUALIFY source_rank<={SOURCE_K}"""
        )
        con.execute(
            f"""CREATE TEMP TABLE truth AS SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
        )
        con.execute(
            """CREATE TEMP TABLE marginal AS SELECT c.*,
                CAST(t.article_id IS NOT NULL AS INTEGER) target_flag
            FROM candidates c LEFT JOIN current_pool p USING(customer_id,article_id)
            LEFT JOIN truth t USING(customer_id,article_id) WHERE p.article_id IS NULL"""
        )
        totals = con.execute(
            """SELECT count(*) pair_count,count(DISTINCT customer_id) user_count,sum(target_flag) truth_pair_count,
                count(DISTINCT customer_id) FILTER(WHERE target_flag=1) truth_user_count FROM marginal"""
        ).fetchone()
        bands = con.execute(
            """SELECT CASE WHEN source_rank<=50 THEN '1_50' WHEN source_rank<=100 THEN '51_100' ELSE '101_200' END rank_band,
                count(*) pair_count,sum(target_flag) truth_pair_count,
                count(DISTINCT customer_id) FILTER(WHERE target_flag=1) truth_user_count
            FROM marginal GROUP BY rank_band ORDER BY min(source_rank)"""
        ).fetchdf()
        truth_users = con.execute("SELECT DISTINCT customer_id FROM marginal WHERE target_flag=1").fetchdf()
        graph_stats = con.execute("SELECT count(*) directed_edges,count(DISTINCT seed_article_id) seed_items FROM neighbors").fetchone()
    states = current_values(name, cutoff)
    gains = []
    missing_state = 0
    for customer in truth_users.customer_id:
        state = states.get(customer)
        if state is None:
            missing_state += 1
            continue
        gains.append(max(0.0, replace_rank12(state["values"], 1, state["truth_count"])))
    density = float(totals[2] / totals[0]) if totals[0] else 0.0
    band_rows = {}
    for row in bands.itertuples(index=False):
        band_rows[row.rank_band] = {
            "candidate_pairs": int(row.pair_count),
            "marginal_truth_pairs": int(row.truth_pair_count),
            "marginal_truth_users": int(row.truth_user_count),
            "positive_density": float(row.truth_pair_count / row.pair_count),
        }
    for band in ("1_50", "51_100", "101_200"):
        band_rows.setdefault(band, {"candidate_pairs": 0, "marginal_truth_pairs": 0, "marginal_truth_users": 0, "positive_density": 0.0})
    return {
        "window": name,
        "cutoff": cutoff,
        "total_users_denominator": int(meta["total_users"]),
        "candidate_pairs": int(totals[0]),
        "candidate_users": int(totals[1]),
        "marginal_truth_pairs": int(totals[2]),
        "marginal_truth_users": int(totals[3]),
        "marginal_truth_pairs_ranks51_200": band_rows["51_100"]["marginal_truth_pairs"] + band_rows["101_200"]["marginal_truth_pairs"],
        "positive_density": density,
        "Top600_positive_density_reference": top600_density,
        "positive_density_lift_vs_Top600": float(density / top600_density),
        "rank_bands": band_rows,
        "directed_graph_edges_after_Top200": int(graph_stats[0]),
        "graph_seed_items": int(graph_stats[1]),
        "rank12_constrained_oracle_population_delta": float(np.sum(gains) / meta["total_users"]),
        "truth_users_without_current_state": missing_state,
        "candidate_pool_changed": True,
        "new_training": False,
        "outer": "not_run",
        "final_week": "not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-780：深层同日共现候选审计",
        "",
        "## 结论",
        "",
        f"授权继续：**{'通过' if result['training_design_authorized'] else '未通过'}**。本轮没有训练模型，也没有读取外层。",
        "",
        "## 术语",
        "",
        "- 同日共现（推荐系统常用近似）：两个商品在同一用户同一天的交易中共同出现；H&M 没有订单编号，因此这只是购物篮近似，不等同真实订单共购。",
        "- 余弦支持归一化（图相似度常用方法）：共同出现天数除以两个商品各自出现天数乘积的平方根，用于降低纯热门商品对分数的支配。",
        "- 深层同日共现候选（本项目自定义）：把冻结召回的每个种子 Top50 邻居和每用户 Top50 输出都扩到 Top200，再排除当前冻结池已有的用户—商品对。",
        "- 正例密度倍数（本项目诊断）：深层共现边际候选的下一周购买比例 / 同窗口最近热度 Top600 边际候选购买比例。",
        "- 第12名受限 Oracle（本项目不可部署上界）：预知标签后，每用户最多拿一件共现边际真值替换 WV3-741 当前第12名；分母是该窗口全部真值用户。",
        "",
        "| 内层窗口 | 新增候选/真值 | 51–200名真值 | 正例密度 | 相对 Top600 密度 | 第12名 Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['candidate_pairs']:,} / {row['marginal_truth_pairs']:,} | "
            f"{row['marginal_truth_pairs_ranks51_200']:,} | {row['positive_density']:.6%} | "
            f"{row['positive_density_lift_vs_Top600']:.2f}x | {row['rank12_constrained_oracle_population_delta']:+.9f} |"
        )
    lines += [
        "",
        "本轮只判断被原 Top50 截断的共现图是否还有值得建模的边际候选，不根据结果改邻居数、输出数或图时间窗。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-780 must start preregistered and unexposed")
    top600 = read(common.REPORT / "WV3-761_SCREEN.json")["windows"]
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        windows[name] = audit_window(name, folder, top600[name]["full_positive_density"])
        print({"deep_covisit_audit": name, "tail_truth": windows[name]["marginal_truth_pairs_ranks51_200"], "density_lift": windows[name]["positive_density_lift_vs_Top600"], "oracle": windows[name]["rank12_constrained_oracle_population_delta"]}, flush=True)
    rules = contract()["gates"]
    oracle = [row["rank12_constrained_oracle_population_delta"] for row in windows.values()]
    passed = bool(
        min(row["marginal_truth_pairs_ranks51_200"] for row in windows.values()) >= rules["marginal_truth_pairs_ranks51_200_each_window_min"]
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
        "final_week": "not_run",
    }
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="diagnostic_supports_deep_covisit" if passed else "diagnostic_rejects_deep_covisit",
        inner_evidence={"mean_oracle": result["mean_rank12_constrained_oracle_population_delta"], "minimum_oracle": result["minimum_rank12_constrained_oracle_population_delta"], "authorized": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)],
        validity="valid_cutoff_safe_inner_candidate_audit",
    )
    common.log(
        f"{TRIAL} deeper same-day co-visitation audit",
        "WV3-761 and WV3-770 rejected a lightweight global-popularity cascade and omitted all-history re-entry; the frozen co-visitation path still truncates neighbors and user output at 50.",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"mean rank12 oracle={result['mean_rank12_constrained_oracle_population_delta']:+.9f}; minimum={result['minimum_rank12_constrained_oracle_population_delta']:+.9f}; authorized={passed}.",
        "Audit feature reconstruction and supervision before training." if passed else "Close deeper co-visitation and retain WV3-741.",
        "Do not change graph depth or time windows after observing this screen; do not inspect outer.",
        alternatives="Unlike WV3-761 this source is personalized by each user's purchased seeds and product-product co-occurrence, not global popularity filtering.",
        experiment="Rebuild the original cutoff-safe co-vis graph at fixed Top200 depth and measure marginal truth density and one-slot oracle on four inner windows.",
        reflection="A deeper graph is worthwhile only if the truncated tail adds stable, denser positives beyond the current mixed pool.",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
