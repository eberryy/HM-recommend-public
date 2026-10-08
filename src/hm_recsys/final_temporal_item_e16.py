"""FINAL-E16: cutoff-safe item freshness/trend features for Cold admission."""
from pathlib import Path
import gc
import json
import math
import shutil
import subprocess
import time
import traceback

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import segment_metrics
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import VALID, WINDOWS
from .final_full_cold_e8 import PARAMS, PREPARED, earlier_full
from .final_multiview_cold_e9 import (
    FEATURES as BASE_FEATURES, candidate_path, feature_arrays, feature_dir,
    final_matrix as base_final_matrix,
)
from .final_negative_scale_e13 import E10
from .final_oracle_audit import dump
from .final_real_action_e10 import fit as fit_action
from .final_pseudo_action_e6 import score_real
from .final_relation_pilot import identify
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/temporal-item-e16-v1")
REPORT = Path("reports/final")
TEMPORAL = Path("artifacts/final/temporal-item-features-e16-v1")
CATALOG = Path("artifacts/m4/m4-v1-supervised-cold-representation/student-v1/static_catalog/catalog_items.csv")
TRANSACTIONS = Path("data/interim/audit/transactions.parquet")
TEMPORAL_NAMES = [
    "item_events_1d", "item_events_3d", "item_events_7d", "item_events_14d",
    "item_events_28d", "item_events_84d", "item_customers_7d",
    "item_customers_28d", "days_since_last_sale", "days_since_first_sale",
    "item_velocity_3d_vs_28d", "item_velocity_7d_vs_28d",
    "item_velocity_14d_vs_84d",
]
FEATURES = BASE_FEATURES + TEMPORAL_NAMES
VICTIMS = ("rank12", "real_risk")
QUOTAS = (.005, .01, .02, .05)


def item_features(repo: Path, cutoff: str):
    folder = repo / TEMPORAL / cutoff
    path = folder / "item-temporal.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    folder.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(repo / CATALOG, dtype={"article_id": str}).sort_values("catalog_row")
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
        frame = db.execute("""
            SELECT article_id,
              count(*) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 1 DAY)::BIGINT events_1d,
              count(*) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 3 DAY)::BIGINT events_3d,
              count(*) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 7 DAY)::BIGINT events_7d,
              count(*) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 14 DAY)::BIGINT events_14d,
              count(*) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 28 DAY)::BIGINT events_28d,
              count(*) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 84 DAY)::BIGINT events_84d,
              count(DISTINCT customer_id) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 7 DAY)::BIGINT customers_7d,
              count(DISTINCT customer_id) FILTER (WHERE t_dat >= ?::DATE - INTERVAL 28 DAY)::BIGINT customers_28d,
              date_diff('day', max(t_dat), ?::DATE)::BIGINT days_since_last_sale,
              date_diff('day', min(t_dat), ?::DATE)::BIGINT days_since_first_sale
            FROM read_parquet(?)
            WHERE t_dat < ?::DATE
            GROUP BY article_id
        """, [cutoff] * 10 + [str(repo / TRANSACTIONS), cutoff]).fetchdf()
    merged = catalog[["article_id", "catalog_row"]].merge(frame, on="article_id", how="left", validate="one_to_one")
    count_columns = [c for c in frame.columns if c.startswith(("events_", "customers_"))]
    merged[count_columns] = merged[count_columns].fillna(0)
    merged[["days_since_last_sale", "days_since_first_sale"]] = merged[[
        "days_since_last_sale", "days_since_first_sale"
    ]].fillna(9999)
    events28 = merged.events_28d.to_numpy(float)
    events84 = merged.events_84d.to_numpy(float)
    matrix = np.column_stack([
        merged.events_1d, merged.events_3d, merged.events_7d, merged.events_14d,
        merged.events_28d, merged.events_84d, merged.customers_7d,
        merged.customers_28d, merged.days_since_last_sale, merged.days_since_first_sale,
        (merged.events_3d.to_numpy(float) / 3 + 1e-3) / (events28 / 28 + 1e-3),
        (merged.events_7d.to_numpy(float) / 7 + 1e-3) / (events28 / 28 + 1e-3),
        (merged.events_14d.to_numpy(float) / 14 + 1e-3) / (events84 / 84 + 1e-3),
    ]).astype(np.float32)
    np.save(path, matrix)
    dump(folder / "AUDIT.json", {
        "cutoff": cutoff, "catalog_items": int(len(matrix)),
        "items_seen_before_cutoff": int((merged.days_since_last_sale < 9999).sum()),
        "latest_behavior_exclusive": cutoff, "feature_names": TEMPORAL_NAMES,
        "labels_used": False, "final_week": "not_run",
    })
    return np.load(path, mmap_mode="r")


def sampled_full(repo: Path, cutoff: str):
    with np.load(candidate_path(repo, cutoff), mmap_mode="r") as candidates:
        user = candidates["user_index"]
        item = candidates["catalog_row"]
        target = candidates["target"].astype(np.int8)
        seed = np.uint64(int(cutoff.replace("-", "")))
        mix = ((user.astype(np.uint64) * np.uint64(11400714819323198485)) ^
               (item.astype(np.uint64) * np.uint64(14029467366897019727)) ^ seed)
        keep = (target > 0) | ((mix % np.uint64(10)) == 0)
        users = user[keep]
        items = item[keep]
        y = target[keep]
        full_rows = len(target)
    m4, p33_rel, p33_global, user_state, candidate_state = feature_arrays(feature_dir(repo, cutoff))
    temporal = item_features(repo, cutoff)
    x = np.concatenate([
        m4[keep], p33_rel[keep], p33_global[keep], user_state[users],
        candidate_state[keep], temporal[items],
    ], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (len(y), len(FEATURES))
    return x, y, {
        "cutoff": cutoff, "full_rows": int(full_rows), "retained_rows": int(len(y)),
        "positive_rows": int(y.sum()), "negative_sampling": "稳定散列10%负例，全部正例",
    }


def fit(repo: Path, cutoff: str, folder: Path):
    xs, ys, sources = [], [], []
    for date in earlier_full(cutoff):
        x, y, audit = sampled_full(repo, date)
        xs.append(x)
        ys.append(y)
        sources.append(audit)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = lgb.LGBMClassifier(**PARAMS)
    with threadpool_limits(limits=4):
        model.fit(x, y, feature_name=FEATURES)
    model.booster_.save_model(str(folder / f"{cutoff}-F70.txt"))
    info = {
        "rows": int(len(y)), "positives": int(y.sum()),
        "sampled_positive_rate": float(y.mean()), "training": earlier_full(cutoff),
        "sources": sources,
    }
    del x, y, xs, ys
    gc.collect()
    return model, info


def final_matrix(repo: Path, cutoff: str, data: dict):
    base, audit = base_final_matrix(repo, cutoff, data)
    catalog = pd.read_csv(repo / CATALOG, dtype={"article_id": str}).sort_values("catalog_row")
    lookup = pd.Series(catalog.catalog_row.to_numpy(np.int64), index=catalog.article_id)
    rows = lookup.reindex(data["cold"].article_id).to_numpy()
    if np.any(pd.isna(rows)):
        raise RuntimeError(f"Cold article missing from catalog at {cutoff}")
    temporal = np.asarray(item_features(repo, cutoff)[rows.astype(np.int64)])
    x = np.concatenate([base, temporal], axis=1).astype(np.float32)
    assert x.shape == (len(data["cold"]), len(FEATURES))
    audit["temporal_width"] = len(TEMPORAL_NAMES)
    return x, audit


def evaluate(data: dict, score: np.ndarray, victim_score: np.ndarray | None,
             victim: str, quota: float):
    chosen = one_per_user(data, score)
    slots = np.full(len(chosen), 11, dtype=np.int8) if victim == "rank12" else np.argmax(victim_score[chosen], axis=1)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(chosen))))
    selected = np.lexsort((customer, -score[chosen]))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in selected:
        candidate = int(chosen[local])
        slot = int(slots[local])
        row = data["cold"].iloc[candidate]
        user = int(row.user_index)
        removed = lists[user, slot]
        lists[user, slot] = row.article_id
        delta = exact_ap(lists[user], data["truthsets"][data["users"][user]]) - data["baseline_ap"][user]
        events.append({
            "customer_id": data["users"][user], "candidate": row.article_id,
            "victim": removed, "slot": slot + 1, "score": float(score[candidate]),
            "delta": float(delta), "inserted": int(row.target),
            "removed": int(removed in data["truthsets"][data["users"][user]]),
        })
    events = pd.DataFrame(events)
    return {
        "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
    }, lists, events


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E16_feature_build",
        "stage": "FINAL-E16 Cold 商品新鲜度与短期趋势",
        "observed_bottleneck": "N10 已提高三窗候选 Recall@1/5，但未来正例未稳定集中在跨用户高分头部；现有57维特征仅有累计交互次数，不能区分刚上新起势与长期滞销的同计数商品。",
        "hypothesis": "严格截止日前的1/3/7/14/28/84日销量、独立顾客数、首末销售距今和短长周期速度比，可提升低频商品的时效辨识与准入精度。",
        "features": {
            "base": "E13 的57维多视角用户—商品特征。",
            "item_temporal": TEMPORAL_NAMES,
            "unit": "每个截止日、每个 catalog 商品一行；所有统计只使用 t_dat < cutoff 的交易。",
        },
        "training": "全部正例加稳定散列10%负例；LightGBM 参数与 E13 完全相同。",
        "candidate_diagnostic": "对比 B0 和 E13 N10 的 Cold50 Recall@1/5、Precision@1 与 MRR。",
        "integration": {
            "rank12": "F70 用户内第一名固定替换 WV3 第12位。",
            "real_risk": "F70 用户内第一名由 E10 真实风险模型选择被替换位置。",
            "admission": "按 F70 分数执行候选用户头部0.5%、1%、2%、5%。",
        },
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一规则，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；停止本轮 Cold 最后探索。",
        "cost_limit": "本机 CPU；特征快照预计10分钟内，模型与评测预计30分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "TEMPORAL_ITEM_E16_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E16")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        model, fit_info = fit(repo, cutoff, repo / RUN)
        x, mapping = final_matrix(repo, cutoff, data)
        score = model.booster_.predict(x, num_threads=4)
        np.save(repo / RUN / f"{cutoff}-F70-scores.npy", score)
        historical[cutoff] = {
            "fit": fit_info, "mapping": mapping,
            "B0": identify(data, -data["cold"].b0_rank.to_numpy(float)),
            "F70": identify(data, score), "rules": {},
        }
        for victim in VICTIMS:
            if victim == "rank12":
                victim_score = None
            else:
                victim_score = np.load(repo / E10 / f"{cutoff}-real_risk_separated-scores.npy")
            for quota in QUOTAS:
                key = f"{victim}|q={quota:g}"
                historical[cutoff]["rules"][key], _, _ = evaluate(
                    data, score, victim_score, victim, quota,
                )
        print("E16", cutoff, historical[cutoff]["F70"], flush=True)
        del data, model, x, score
        gc.collect()
    for victim in VICTIMS:
        for quota in QUOTAS:
            key = f"{victim}|q={quota:g}"
            rows = [historical[t]["rules"][key] for t in VALID]
            aggregate[key] = {
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "beneficial": int(sum(r["beneficial"] for r in rows)),
                "harmful": int(sum(r["harmful"] for r in rows)),
                "victim": victim, "quota": quota,
            }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], VICTIMS.index(pair[1]["victim"])))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            model, fit_info = fit(repo, cutoff, repo / RUN)
            x, mapping = final_matrix(repo, cutoff, data)
            score = model.booster_.predict(x, num_threads=4)
            if spec["victim"] == "rank12":
                victim_score, victim_info = None, None
            else:
                action_model, victim_info = fit_action(repo, cutoff, "real_risk_separated", repo / RUN)
                victim_score = score_real(repo, cutoff, data, action_model)
                del action_model
            result, lists, events = evaluate(
                data, score, victim_score, spec["victim"], spec["quota"],
            )
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = {"candidate": fit_info, "victim": victim_info, "mapping": mapping}
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, model, x, score, victim_score, lists, events
            gc.collect()
        delta = np.array([r["delta_map"] for r in development.values()])
        warm = np.mean([r["segments"]["warm_21_plus"] for r in development.values()])
        cold = np.mean([r["segments"]["all_cold_sparse"] for r in development.values()])
        inserted = sum(r["inserted"] for r in development.values())
        removed = sum(r["removed"] for r in development.values())
        gates = {
            "mean_positive": bool(delta.mean() > 0),
            "three_nondegrade": bool((delta >= -1e-5).sum() >= 3),
            "worst": bool(delta.min() >= -5e-5),
            "warm": bool(warm >= -2e-5),
            "cold_sparse": bool(cold >= 0),
            "inserted_positive": bool(inserted >= 1),
            "efficiency": bool(inserted >= removed),
        }
        passed = all(gates.values())
    output = {
        "status": "completed", "stage": contract["stage"],
        "historical": historical, "historical_aggregate": aggregate,
        "selected": selected, "historical_gate_pass": selected is not None,
        "development": development, "development_gates": gates,
        "fullscale_allowed": passed, "fullscale_status": "not_run",
        "fallback": "challenger" if passed else "WV3-741",
        "final_week": "not_run", "seconds": time.perf_counter() - start,
    }
    dump(repo / REPORT / "TEMPORAL_ITEM_E16.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"TEMPORAL_ITEM_E16_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
