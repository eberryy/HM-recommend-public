"""WV3-761: first-stage purity test for a recent-popularity Top600 cascade."""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import literal, save_parquet
from .warm_v3_bpr_safe_admission import replace_rank12
from .warm_v3_pool import connection
from .warm_v3_popular_cascade_audit import current_values
from .warm_v3_residual_admission import HISTORICAL, INNER, source


TRIAL = "WV3-761"
CONTRACT = common.REPORT / "WV3-761_POPULAR_STAGE1_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-761_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-761_SCREEN.md"
MODEL_ROOT = common.ART / TRIAL
ARTICLES = common.SHARED / "data/raw/articles.csv"
TOP_K = 600
FILTER_K = 130
NEGATIVE_RATIO = 30

FEATURES = [
    "popularity_rank_fraction",
    "log_popularity_events_7d",
    "log_item_events_28d",
    "item_trend_7d_vs_28d",
    "log_user_events_12w",
    "log_user_unique_items_12w",
    "log_user_recency_days",
    "log_user_item_events_12w",
    "log_user_item_recency_days",
    "user_product_type_share_12w",
    "user_department_share_12w",
    "user_garment_share_12w",
    "user_colour_share_12w",
    "user_index_group_share_12w",
    "user_item_price_gap",
]
PARAMS = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 200,
    "lambda_l2": 10.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 20260910,
    "feature_fraction_seed": 20260910,
    "bagging_seed": 20260910,
    "verbosity": -1,
}
ROUNDS = 150


def create_relations(con, cutoff: str, ranks_path: Path) -> None:
    con.execute(
        f"""CREATE TEMP TABLE current_pool AS SELECT DISTINCT customer_id,article_id
        FROM read_parquet({literal(ranks_path)}) WHERE user_history_events_12w>0"""
    )
    con.execute("CREATE TEMP TABLE eligible_users AS SELECT DISTINCT customer_id FROM current_pool")
    con.execute(
        f"""CREATE TEMP TABLE article_dim AS SELECT article_id kpr,
            try_cast(product_type_no AS INTEGER) product_type_no,
            try_cast(department_no AS INTEGER) department_no,
            try_cast(garment_group_no AS INTEGER) garment_group_no,
            try_cast(colour_group_code AS INTEGER) colour_group_code,
            try_cast(index_group_no AS INTEGER) index_group_no
        FROM read_csv({literal(ARTICLES)},header=true,all_varchar=true)"""
    )
    con.execute(
        f"""CREATE TEMP TABLE popular AS WITH counts AS (
            SELECT article_id,count(*) events_7d FROM read_parquet({literal(common.TX)})
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 7 DAY AND t_dat<DATE '{cutoff}' GROUP BY article_id
        ) SELECT article_id,events_7d,row_number() OVER(ORDER BY events_7d DESC,article_id) popularity_rank
          FROM counts QUALIFY popularity_rank<={TOP_K}"""
    )
    con.execute(
        f"""CREATE TEMP TABLE history AS SELECT t.*
        FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
        WHERE t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t_dat<DATE '{cutoff}'"""
    )
    con.execute(
        """CREATE TEMP TABLE history_attr AS SELECT h.*,a.product_type_no,a.department_no,
            a.garment_group_no,a.colour_group_code,a.index_group_no
        FROM history h JOIN article_dim a ON h.article_id=a.kpr"""
    )
    con.execute(
        f"""CREATE TEMP TABLE truth AS SELECT DISTINCT t.customer_id,t.article_id
        FROM read_parquet({literal(common.TX)}) t JOIN eligible_users u USING(customer_id)
        WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
    )
    con.execute(
        f"""CREATE TEMP VIEW candidate_features AS WITH
        user_stats AS (
            SELECT customer_id,count(*) user_events,count(DISTINCT article_id) user_unique,
                date_diff('day',max(t_dat),DATE '{cutoff}') user_recency,avg(price) user_avg_price
            FROM history GROUP BY customer_id
        ), item_stats AS (
            SELECT article_id,count(*) item_events_28d,
                count(*) FILTER(WHERE t_dat>=DATE '{cutoff}'-INTERVAL 7 DAY) item_events_7d,
                avg(price) item_avg_price
            FROM history WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY GROUP BY article_id
        ), user_item AS (
            SELECT customer_id,article_id,count(*) user_item_events,
                date_diff('day',max(t_dat),DATE '{cutoff}') user_item_recency
            FROM history GROUP BY customer_id,article_id
        ), product_type AS (
            SELECT customer_id,product_type_no,count(*) n FROM history_attr GROUP BY customer_id,product_type_no
        ), department AS (
            SELECT customer_id,department_no,count(*) n FROM history_attr GROUP BY customer_id,department_no
        ), garment AS (
            SELECT customer_id,garment_group_no,count(*) n FROM history_attr GROUP BY customer_id,garment_group_no
        ), colour AS (
            SELECT customer_id,colour_group_code,count(*) n FROM history_attr GROUP BY customer_id,colour_group_code
        ), index_group AS (
            SELECT customer_id,index_group_no,count(*) n FROM history_attr GROUP BY customer_id,index_group_no
        )
        SELECT u.customer_id,p.article_id,p.popularity_rank,
            p.popularity_rank*1.0/{TOP_K} popularity_rank_fraction,
            ln(1+p.events_7d) log_popularity_events_7d,
            ln(1+coalesce(i.item_events_28d,0)) log_item_events_28d,
            coalesce(4.0*i.item_events_7d/greatest(i.item_events_28d,1),0) item_trend_7d_vs_28d,
            ln(1+s.user_events) log_user_events_12w,
            ln(1+s.user_unique) log_user_unique_items_12w,
            ln(1+least(s.user_recency,10000)) log_user_recency_days,
            ln(1+coalesce(ui.user_item_events,0)) log_user_item_events_12w,
            ln(1+least(coalesce(ui.user_item_recency,10000),10000)) log_user_item_recency_days,
            coalesce(pt.n*1.0/s.user_events,0) user_product_type_share_12w,
            coalesce(d.n*1.0/s.user_events,0) user_department_share_12w,
            coalesce(g.n*1.0/s.user_events,0) user_garment_share_12w,
            coalesce(c.n*1.0/s.user_events,0) user_colour_share_12w,
            coalesce(ix.n*1.0/s.user_events,0) user_index_group_share_12w,
            abs(coalesce(i.item_avg_price,s.user_avg_price)-s.user_avg_price) user_item_price_gap,
            CAST(t.article_id IS NOT NULL AS INTEGER) target_flag
        FROM eligible_users u CROSS JOIN popular p
        JOIN user_stats s USING(customer_id)
        JOIN article_dim a ON p.article_id=a.kpr
        LEFT JOIN current_pool old USING(customer_id,article_id)
        LEFT JOIN truth t USING(customer_id,article_id)
        LEFT JOIN item_stats i USING(article_id)
        LEFT JOIN user_item ui USING(customer_id,article_id)
        LEFT JOIN product_type pt USING(customer_id,product_type_no)
        LEFT JOIN department d USING(customer_id,department_no)
        LEFT JOIN garment g USING(customer_id,garment_group_no)
        LEFT JOIN colour c USING(customer_id,colour_group_code)
        LEFT JOIN index_group ix USING(customer_id,index_group_no)
        WHERE old.article_id IS NULL"""
    )


def build_frame(name: str, folder: str, role: str) -> tuple[pd.DataFrame, dict]:
    ranks_path, meta = source(folder)
    cutoff = meta["cutoff"]
    root = MODEL_ROOT / "data" / cutoff
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{role}.parquet"
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and metadata_path.exists():
        return load_frame(destination), read(metadata_path)
    started = time.perf_counter()
    with connection() as con:
        create_relations(con, cutoff, ranks_path)
        if role == "sample":
            query = f"""WITH positive_users AS (
                SELECT customer_id,sum(target_flag) positives FROM candidate_features
                GROUP BY customer_id HAVING positives>0
            ), negatives AS (
                SELECT c.*,row_number() OVER(PARTITION BY c.customer_id
                    ORDER BY hash(c.customer_id,c.article_id,20260910),c.popularity_rank,c.article_id) negative_rank
                FROM candidate_features c JOIN positive_users p USING(customer_id) WHERE target_flag=0
            ), positives AS (
                SELECT c.*,0::BIGINT negative_rank FROM candidate_features c
                JOIN positive_users p USING(customer_id) WHERE target_flag=1
            ) SELECT * EXCLUDE(negative_rank) FROM (
                SELECT * FROM positives UNION ALL SELECT n.* FROM negatives n
                JOIN positive_users p USING(customer_id) WHERE negative_rank<={NEGATIVE_RATIO}*p.positives
            ) ORDER BY customer_id,target_flag DESC,popularity_rank,article_id"""
        elif role == "inner":
            query = "SELECT * FROM candidate_features ORDER BY customer_id,popularity_rank,article_id"
        else:
            raise ValueError(role)
        con.execute(f"COPY ({query}) TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD)")
        stats = con.execute(
            f"""SELECT count(*) AS row_count,count(DISTINCT customer_id) AS user_count,
                sum(target_flag) AS positive_count,min(popularity_rank) AS min_rank,
                max(popularity_rank) AS max_rank
            FROM read_parquet({literal(destination)})"""
        ).fetchone()
    result = {
        "window": name,
        "cutoff": cutoff,
        "role": role,
        "rows": int(stats[0]),
        "users": int(stats[1]),
        "positive_rows": int(stats[2]),
        "positive_rate": float(stats[2] / stats[0]),
        "min_popularity_rank": int(stats[3]),
        "max_popularity_rank": int(stats[4]),
        "features": FEATURES,
        "artifact": str(destination),
        "runtime_seconds": time.perf_counter() - started,
        "cutoff_safe": True,
        "final_week": "not_run",
    }
    write(metadata_path, result)
    return load_frame(destination), result


def load_frame(path: Path) -> pd.DataFrame:
    with connection() as con:
        return con.execute(f"SELECT * FROM read_parquet({literal(path)})").fetchdf()


def contract() -> dict:
    prior = read(common.REPORT / "WV3-760_POPULAR_CASCADE_AUDIT.json")
    if not prior["cascade_design_authorized"]:
        raise AssertionError("WV3-760 did not authorize stage-1 cascade testing")
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-741",
        "architecture_family": "recent_popularity_Top600_to_Top130_first_stage_filter",
        "hypothesis": (
            "A lightweight historical pointwise filter using recency, popularity and user-attribute affinity can "
            "retain substantially more than 130/600 of marginal Top600 truth while increasing candidate density."
        ),
        "candidate_pool": "per active Warm user, recent-7-day global Top600 items absent from the original pool",
        "features": FEATURES,
        "training": "four strict 2019 windows; all positives plus deterministic 30:1 negatives from positive-bearing users",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "selection": "per user Top130 by model score desc, popularity rank asc, article id asc",
        "inner_gate": {
            "mean_truth_retention_min": 0.45,
            "truth_retention_each_window_min": 0.35,
            "mean_selected_rank12_oracle_min": 0.001,
            "selected_rank12_oracle_each_window_min": 0.0008,
            "positive_density_lift_each_window_min": 1.5,
        },
        "expected_minutes": 35,
        "success_action": "freeze the 600-to-130 filter and audit a final-stage ranker; do not yet expose outer",
        "failure_action": "retain WV3-741 and close this lightweight cascade; no TopK, negative ratio or tree rescue",
        "outer": "not_run",
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(35)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            contract()["architecture_family"],
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_pool"],
            training_protocol=contract()["training"],
            features=FEATURES,
            params={"model": contract()["model"], "selection": contract()["selection"], "inner_gate": contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=35,
        )
    return read(CONTRACT)


def fit(training: pd.DataFrame) -> lgb.Booster:
    labels = training.target_flag.to_numpy(np.uint8)
    if not (0 < labels.sum() < len(labels)):
        raise AssertionError("Stage-1 training requires positive and negative examples")
    dataset = lgb.Dataset(training[FEATURES].to_numpy(np.float32), label=labels, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def evaluate(name: str, folder: str, model: lgb.Booster) -> dict:
    frame, data = build_frame(name, folder, "inner")
    scores = model.predict(frame[FEATURES].to_numpy(np.float32), num_threads=4)
    choices = frame[["customer_id", "article_id", "popularity_rank"]].copy()
    choices["stage1_score"] = scores
    choices = choices.sort_values(
        ["customer_id", "stage1_score", "popularity_rank", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).groupby("customer_id", sort=False).head(FILTER_K)
    selected_keys = choices[["customer_id", "article_id"]]
    labels = frame[["customer_id", "article_id", "target_flag"]]
    selected = selected_keys.merge(labels, on=["customer_id", "article_id"], validate="one_to_one")
    total_positive = int(frame.target_flag.sum())
    selected_positive = int(selected.target_flag.sum())
    full_density = float(frame.target_flag.mean())
    selected_density = float(selected.target_flag.mean())
    users_with_selected_truth = selected.loc[selected.target_flag == 1, "customer_id"].drop_duplicates()
    states = current_values(name, data["cutoff"])
    gains = []
    for customer in users_with_selected_truth:
        state = states.get(customer)
        if state is not None:
            gains.append(max(0.0, replace_rank12(state["values"], 1, state["truth_count"])))
    denominator = read(common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json")["windows"][name]["total_users_denominator"]
    oracle = float(np.sum(gains) / denominator)
    root = MODEL_ROOT / name / "inner"
    root.mkdir(parents=True, exist_ok=True)
    save_parquet(choices, root / "top130_scores.parquet")
    return {
        "window": name,
        "cutoff": data["cutoff"],
        "full_marginal_candidate_rows": len(frame),
        "selected_candidate_rows": len(selected),
        "full_marginal_truth_pairs": total_positive,
        "selected_truth_pairs": selected_positive,
        "truth_retention": float(selected_positive / total_positive),
        "full_positive_density": full_density,
        "selected_positive_density": selected_density,
        "positive_density_lift": float(selected_density / full_density),
        "selected_truth_users": len(users_with_selected_truth),
        "selected_rank12_constrained_oracle_population_delta": oracle,
        "total_users_denominator": denominator,
        "target_excluded_from_features": "target_flag" not in FEATURES,
        "selection_uses_target": False,
        "outer": "not_run",
        "final_week": "not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-761：最近热度 Top600 到 Top130 第一阶段过滤",
        "",
        "## 结论",
        "",
        f"第一阶段保真门槛：**{'通过' if result['passed'] else '未通过'}**；四窗平均真值保留率 `{result['mean_truth_retention']:.2%}`，"
        f"平均第12名受限 Oracle `{result['mean_selected_rank12_oracle_population_delta']:+.9f}`。本轮未读外层。",
        "",
        "## 术语",
        "",
        "- 第一阶段过滤（推荐级联常见）：在便宜特征上给每位用户约600件边际热门商品打分，只保留130件供后续复杂排序；本轮只验证过滤，不训练最终排序器。",
        "- 真值保留率（本项目诊断）：Top130 内下一周真实购买的边际商品对数量 / Top600 全部边际真值商品对数量；分母按窗口分别计算。",
        "- 正例密度提升倍数（本项目诊断）：Top130 正例比例 / Top600 正例比例；候选数减少约78%，若过滤有效该比值应明显大于1。",
        "- 用户属性偏好（推荐系统常见交叉特征）：用户最近12周在候选商品品类、部门、服装组、颜色和索引组上的购买事件占比；全部在截止日前计算。",
        "- 第12名受限 Oracle：仅对 Top130 中存在边际真值的用户，预知标签后最多拿一件替换 WV3-741 当前第12名；以窗口全部真值用户为分母，不可部署。",
        "",
        "| 内层窗口 | Top600 行/真值 | Top130 行/真值 | 真值保留率 | 密度提升 | Top130 Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['full_marginal_candidate_rows']:,} / {row['full_marginal_truth_pairs']:,} | "
            f"{row['selected_candidate_rows']:,} / {row['selected_truth_pairs']:,} | {row['truth_retention']:.2%} | "
            f"{row['positive_density_lift']:.2f}x | {row['selected_rank12_constrained_oracle_population_delta']:+.9f} |"
        )
    lines += [
        "",
        "训练只使用四个2019窗口；验证候选的 `target_flag` 不进入特征或 Top130 选择。失败后不调整130、30:1或树参数。最终周保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(35)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-761 must start preregistered and unexposed")
    started = time.perf_counter()
    samples = []
    source_audit = []
    for name, folder in HISTORICAL:
        frame, audit = build_frame(name, folder, "sample")
        samples.append(frame)
        source_audit.append(audit)
        print({"popular_stage1_sample": name, "rows": len(frame), "positives": int(frame.target_flag.sum())}, flush=True)
    training = pd.concat(samples, ignore_index=True)
    if len(training) < 100000 or int(training.target_flag.sum()) < 10000 or not np.isfinite(training[FEATURES].to_numpy(np.float64)).all():
        raise RuntimeError("WV3-761 historical input gate failed before fit")
    model = fit(training)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(
        MODEL_ROOT / "MODEL.json",
        {
            "experiment_id": TRIAL,
            "model": evidence_id(model_path, reason="explicit_registry_evidence"),
            "features": FEATURES,
            "params": PARAMS,
            "rounds": ROUNDS,
            "training_rows": len(training),
            "training_positive_rows": int(training.target_flag.sum()),
            "source_audit": source_audit,
            "outer_labels_used": False,
            "final_week": "not_run",
        },
    )
    del samples, training
    gc.collect()
    windows = {}
    for name, folder in INNER:
        windows[name] = evaluate(name, folder, model)
        print({"popular_stage1_inner": name, "retention": windows[name]["truth_retention"], "lift": windows[name]["positive_density_lift"], "oracle": windows[name]["selected_rank12_constrained_oracle_population_delta"]}, flush=True)
        gc.collect()
    rules = contract()["inner_gate"]
    retention = [row["truth_retention"] for row in windows.values()]
    oracle = [row["selected_rank12_constrained_oracle_population_delta"] for row in windows.values()]
    lift = [row["positive_density_lift"] for row in windows.values()]
    passed = bool(
        np.mean(retention) >= rules["mean_truth_retention_min"]
        and min(retention) >= rules["truth_retention_each_window_min"]
        and np.mean(oracle) >= rules["mean_selected_rank12_oracle_min"]
        and min(oracle) >= rules["selected_rank12_oracle_each_window_min"]
        and min(lift) >= rules["positive_density_lift_each_window_min"]
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "mean_truth_retention": float(np.mean(retention)),
        "minimum_truth_retention": min(retention),
        "mean_positive_density_lift": float(np.mean(lift)),
        "minimum_positive_density_lift": min(lift),
        "mean_selected_rank12_oracle_population_delta": float(np.mean(oracle)),
        "minimum_selected_rank12_oracle_population_delta": min(oracle),
        "gates": rules,
        "passed": passed,
        "new_training": True,
        "outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="stage1_pass_final_ranker_audit_required" if passed else "reject_inner_stage1",
        inner_evidence={
            "mean_truth_retention": result["mean_truth_retention"],
            "mean_density_lift": result["mean_positive_density_lift"],
            "mean_oracle": result["mean_selected_rank12_oracle_population_delta"],
            "passed": passed,
        },
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(MODEL_ROOT / "MODEL.json")],
        validity="valid_cutoff_safe_inner_stage1_screen",
    )
    common.log(
        f"{TRIAL} popular cascade stage1 screen",
        "WV3-760 found enough Top600 marginal truth and one-slot oracle to bridge the remaining target gap, but candidate density declines sharply in ranks301-600.",
        contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"mean retention={result['mean_truth_retention']:.2%}; minimum={result['minimum_truth_retention']:.2%}; mean density lift={result['mean_positive_density_lift']:.2f}x; mean selected oracle={result['mean_selected_rank12_oracle_population_delta']:+.9f}; pass={passed}.",
        "Freeze Top600-to-Top130 filter and audit final-ranker supervision/cost." if passed else "Reject lightweight popular cascade and retain WV3-741.",
        "Do not expose outer or train the final ranker until a separate candidate identity and supervision audit passes.",
        alternatives="Direct Top600 mixing is rejected by density; changing 130, negative ratio or trees after this screen is prohibited.",
        experiment="Train a lightweight cutoff-safe pointwise filter on strict 2019 windows and measure Top130 truth retention, density lift and one-slot oracle on four inner windows.",
        reflection="This isolates the first-stage recall-versus-purity tradeoff before paying for a complex final ranker.",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
