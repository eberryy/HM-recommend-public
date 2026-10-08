"""WV3-810: audit consensus candidates shared by recent popularity and deep co-visitation."""
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


TRIAL = "WV3-810"
CONTRACT = common.REPORT / "WV3-810_POP_COVISIT_CONSENSUS_AUDIT_CONTRACT.json"
REPORT = common.REPORT / "WV3-810_POP_COVISIT_CONSENSUS_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-810_POP_COVISIT_CONSENSUS_AUDIT.md"


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "inner_only_cross_source_consensus_candidate_audit",
        "hypothesis": (
            "New items independently supported by recent global Top600 popularity and per-user deep Top200 co-visitation "
            "will be materially denser than either broad source alone while retaining stable rank-12 oracle headroom."
        ),
        "candidate_definition": "intersection of WV3-760 Top600 and WV3-780 per-user Top200 co-visitation, excluding the frozen pool",
        "ranking": "sum of reciprocal popularity rank and reciprocal co-visitation source rank; diagnostic only",
        "gates": {
            "marginal_truth_pairs_each_window_min": 30,
            "positive_density_lift_vs_Top600_each_window_min": 2.0,
            "rank12_constrained_oracle_mean_min": 0.00025,
            "rank12_constrained_oracle_each_window_min": 0.0001,
        },
        "training": "none",
        "outer": "not_run",
        "expected_minutes": 20,
        "success_action": "freeze consensus source and audit source-specific ranking supervision; do not expose outer",
        "failure_action": "close cross-source consensus expansion and retain WV3-741",
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(20)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(TRIAL, "recent_popularity_and_deep_covisit_consensus_audit", contract()["hypothesis"], candidate_protocol=contract()["candidate_definition"], training_protocol="none; four 2020 inner windows only", params={"ranking": contract()["ranking"], "gates": contract()["gates"], "final_week": "not_run"}, expected_minutes=20)
    return read(CONTRACT)


def audit_window(name: str, folder: str, top600_density: float) -> dict:
    ranks_path, meta = source(folder)
    cutoff = meta["cutoff"]
    with connection() as con:
        con.execute(f"CREATE TEMP TABLE current_pool AS SELECT DISTINCT customer_id,article_id FROM read_parquet({literal(ranks_path)}) WHERE user_history_events_12w>0")
        con.execute("CREATE TEMP TABLE eligible_users AS SELECT DISTINCT customer_id FROM current_pool")
        con.execute(f"""CREATE TEMP TABLE popular AS WITH counts AS (
            SELECT article_id,count(*) events_7d FROM read_parquet({literal(common.TX)})
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 7 DAY AND t_dat<DATE '{cutoff}' GROUP BY article_id
        ) SELECT article_id,events_7d,row_number() OVER(ORDER BY events_7d DESC,article_id) popularity_rank
        FROM counts QUALIFY popularity_rank<=600""")
        con.execute(f"""CREATE TEMP TABLE valid_user_days AS SELECT customer_id,t_dat,count(DISTINCT article_id) distinct_items
            FROM read_parquet({literal(common.TX)}) WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t_dat<DATE '{cutoff}'
            GROUP BY customer_id,t_dat HAVING distinct_items BETWEEN 2 AND 20""")
        con.execute(f"""CREATE TEMP TABLE user_day_items AS SELECT DISTINCT t.customer_id,t.t_dat,t.article_id
            FROM read_parquet({literal(common.TX)}) t SEMI JOIN valid_user_days d ON t.customer_id=d.customer_id AND t.t_dat=d.t_dat
            WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t.t_dat<DATE '{cutoff}'""")
        con.execute("CREATE TEMP TABLE item_support AS SELECT article_id,count(*) item_days FROM user_day_items GROUP BY article_id")
        con.execute("""CREATE TEMP TABLE neighbors AS WITH pairs AS (
            SELECT l.article_id left_id,r.article_id right_id,count(*) pair_days FROM user_day_items l JOIN user_day_items r
            ON l.customer_id=r.customer_id AND l.t_dat=r.t_dat AND l.article_id<r.article_id
            GROUP BY l.article_id,r.article_id HAVING pair_days>=2
        ), directed AS (
            SELECT left_id seed_article_id,right_id candidate_article_id,pair_days FROM pairs
            UNION ALL SELECT right_id,left_id,pair_days FROM pairs
        ), scored AS (
            SELECT d.*,d.pair_days/sqrt(s.item_days*c.item_days) similarity FROM directed d
            JOIN item_support s ON d.seed_article_id=s.article_id JOIN item_support c ON d.candidate_article_id=c.article_id
        ) SELECT *,row_number() OVER(PARTITION BY seed_article_id ORDER BY similarity DESC,pair_days DESC,candidate_article_id) neighbor_rank
        FROM scored QUALIFY neighbor_rank<=200""")
        con.execute(f"""CREATE TEMP TABLE covisit AS WITH seeds AS (
            SELECT t.customer_id,t.article_id seed_article_id,
                pow(2.0,-date_diff('day',max(t_dat),DATE '{cutoff}')/14.0)*(1.0+ln(count(*))) seed_score
            FROM read_parquet({literal(common.TX)}) t SEMI JOIN eligible_users u USING(customer_id)
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 12 WEEK AND t_dat<DATE '{cutoff}' GROUP BY t.customer_id,t.article_id
        ), scored AS (
            SELECT s.customer_id,n.candidate_article_id article_id,sum(s.seed_score*n.similarity) source_score,max(n.pair_days) max_pair_days
            FROM seeds s JOIN neighbors n USING(seed_article_id) GROUP BY s.customer_id,n.candidate_article_id
        ) SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY source_score DESC,max_pair_days DESC,article_id) covisit_rank
        FROM scored QUALIFY covisit_rank<=200""")
        con.execute(f"""CREATE TEMP TABLE truth AS SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY""")
        con.execute("""CREATE TEMP TABLE consensus AS SELECT c.customer_id,c.article_id,c.covisit_rank,p.popularity_rank,
            1.0/(60+c.covisit_rank)+1.0/(60+p.popularity_rank) consensus_score,
            CAST(t.article_id IS NOT NULL AS INTEGER) target_flag
            FROM covisit c JOIN popular p USING(article_id)
            LEFT JOIN current_pool f USING(customer_id,article_id) LEFT JOIN truth t USING(customer_id,article_id)
            WHERE f.article_id IS NULL""")
        totals = con.execute("""SELECT count(*) pair_count,count(DISTINCT customer_id) user_count,sum(target_flag) truth_pair_count,
            count(DISTINCT customer_id) FILTER(WHERE target_flag=1) truth_user_count FROM consensus""").fetchone()
        truth_users = con.execute("SELECT DISTINCT customer_id FROM consensus WHERE target_flag=1").fetchdf()
        rank_stats = con.execute("SELECT min(covisit_rank),median(covisit_rank),max(covisit_rank),min(popularity_rank),median(popularity_rank),max(popularity_rank) FROM consensus WHERE target_flag=1").fetchone()
    states = current_values(name, cutoff)
    gains = []
    for customer in truth_users.customer_id:
        state = states[customer]
        gains.append(max(0.0, replace_rank12(state["values"], 1, state["truth_count"])))
    density = float(totals[2] / totals[0]) if totals[0] else 0.0
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
        "truth_rank_summary": {"covisit_min_median_max": [float(rank_stats[0]), float(rank_stats[1]), float(rank_stats[2])], "popularity_min_median_max": [float(rank_stats[3]), float(rank_stats[4]), float(rank_stats[5])]},
        "rank12_constrained_oracle_population_delta": float(np.sum(gains) / meta["total_users"]),
        "candidate_pool_changed": True,
        "new_training": False,
        "outer": "not_run",
        "final_week": "not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-810：近期热度与深层共现一致性候选审计",
        "",
        "## 结论",
        "",
        f"授权继续：**{'通过' if result['ranking_design_authorized'] else '未通过'}**。本轮没有训练模型，也没有读取外层。",
        "",
        "## 术语",
        "",
        "- 跨来源一致性候选（推荐系统常见思想，本项目固定实现）：同一用户—商品同时出现在最近热度 Top600 与该用户深层同日共现 Top200 中，并且不在冻结候选池；统计单位为用户—商品对。",
        "- 独立支持（本报告语义）：最近热度来自全体用户近7天交易，共现来自用户历史种子沿商品图扩展；两者计算路径不同，但并非统计独立性证明。",
        "- 正例密度倍数（本项目诊断）：一致性候选下一周购买比例 / 同窗口最近热度 Top600 边际候选购买比例。",
        "- 第12名受限 Oracle（本项目不可部署上界）：预知标签后，每用户最多取一件一致性真值替换 WV3-741 当前第12名；分母为窗口全部真值用户。",
        "",
        "| 内层窗口 | 新增候选/真值 | 正例密度 | 相对 Top600 密度 | 第12名 Oracle |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(f"| {name} | {row['candidate_pairs']:,}/{row['marginal_truth_pairs']:,} | {row['positive_density']:.6%} | {row['positive_density_lift_vs_Top600']:.2f}x | {row['rank12_constrained_oracle_population_delta']:+.9f} |")
    lines += ["", "只在四窗全部达到密度和 Oracle 门槛时，才允许另立监督审计；不按结果调整 Top600、Top200 或图时间窗。最终周 `2020-09-16` 保持 `not_run`。", ""]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-810 must start preregistered and unexposed")
    top600 = read(common.REPORT / "WV3-761_SCREEN.json")["windows"]
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        windows[name] = audit_window(name, folder, top600[name]["full_positive_density"])
        print({"pop_covisit_consensus": name, "truth": windows[name]["marginal_truth_pairs"], "density_lift": windows[name]["positive_density_lift_vs_Top600"], "oracle": windows[name]["rank12_constrained_oracle_population_delta"]}, flush=True)
    rules = contract()["gates"]
    oracle = [row["rank12_constrained_oracle_population_delta"] for row in windows.values()]
    passed = bool(min(row["marginal_truth_pairs"] for row in windows.values()) >= rules["marginal_truth_pairs_each_window_min"] and min(row["positive_density_lift_vs_Top600"] for row in windows.values()) >= rules["positive_density_lift_vs_Top600_each_window_min"] and float(np.mean(oracle)) >= rules["rank12_constrained_oracle_mean_min"] and min(oracle) >= rules["rank12_constrained_oracle_each_window_min"])
    result = {"created_at": now(), "experiment_id": TRIAL, "windows": windows, "mean_rank12_constrained_oracle_population_delta": float(np.mean(oracle)), "minimum_rank12_constrained_oracle_population_delta": min(oracle), "gates": rules, "ranking_design_authorized": passed, "new_training": False, "outer_exposure": 0, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(TRIAL, decision="diagnostic_supports_cross_source_consensus" if passed else "diagnostic_rejects_cross_source_consensus", inner_evidence={"mean_oracle": result["mean_rank12_constrained_oracle_population_delta"], "minimum_oracle": result["minimum_rank12_constrained_oracle_population_delta"], "authorized": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)], validity="valid_cutoff_safe_inner_candidate_audit")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
