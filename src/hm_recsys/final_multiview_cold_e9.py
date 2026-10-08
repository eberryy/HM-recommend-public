"""FINAL-E9: full real-Cold supervision with both content-teacher views."""
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
import torch
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import segment_metrics, stable_percentile
from .final_binary_admission_e5 import one_per_user
from .final_full_cold_e8 import (
    DEV_ACTION, FULL_DATES, PARAMS, PREPARED, P37, COARSE, action_scores,
    earlier_full, relation_path, rrf,
)
from .final_candidate_e2 import VALID, WINDOWS, embedding_path, histories
from .final_oracle_audit import dump
from .final_relation_pilot import identify
from .p42f_data import save_frame
from .p43a_policy import exact_ap
from .p37b_features import compute_relation_feature_batch, compute_user_state_batch


RUN = Path("artifacts/final/multiview-cold-e9-v3")
FINAL_RELATION = Path("artifacts/final/multiview-relation-e9-v1")
REPORT = Path("reports/final")
M4 = [f"m4_{i}" for i in range(24)]
P33_REL = [f"p33_relation_{i}" for i in range(16)]
P33_GLOBAL = ["p33_fixed_decay_score", "p33_minus_m4_fixed_decay_score"]
USER = [f"user_state_{i}" for i in range(10)]
CANDIDATE = [f"candidate_state_{i}" for i in range(5)]
FEATURES = M4 + P33_REL + P33_GLOBAL + USER + CANDIDATE
ALPHAS = (.05, .10, .25, .50, 1.0)
QUOTAS = (.001, .005, .01, .02)
MODES = ("b0_best", "classifier_best", "rrf025_best")


def feature_dir(repo: Path, cutoff: str) -> Path:
    if cutoff in FULL_DATES:
        return repo / P37 / "features-v1" / "training" / cutoff
    window = next(w for w, date in WINDOWS.items() if date == cutoff)
    return repo / P37 / "features-v1" / "outer_validation" / window


def candidate_path(repo: Path, cutoff: str) -> Path:
    if cutoff in FULL_DATES:
        return repo / COARSE / "training" / cutoff / "candidates_top200.npz"
    window = next(w for w, date in WINDOWS.items() if date == cutoff)
    return repo / COARSE / "outer" / window / "candidates_top200.npz"


def feature_arrays(folder: Path):
    return (
        np.load(folder / "m4_relation.float32.npy", mmap_mode="r").reshape(-1, 24),
        np.load(folder / "p33_relation.float32.npy", mmap_mode="r").reshape(-1, 16),
        np.load(folder / "p33_global.float32.npy", mmap_mode="r"),
        np.load(folder / "user_state.float32.npy", mmap_mode="r"),
        np.load(folder / "candidate_state.float32.npy", mmap_mode="r"),
    )


def sampled_full(repo: Path, cutoff: str):
    with np.load(candidate_path(repo, cutoff), mmap_mode="r") as candidates:
        user = candidates["user_index"]
        item = candidates["catalog_row"]
        target = candidates["target"].astype(np.int8)
        seed = np.uint64(int(cutoff.replace("-", "")))
        mix = ((user.astype(np.uint64) * np.uint64(11400714819323198485)) ^
               (item.astype(np.uint64) * np.uint64(14029467366897019727)) ^ seed)
        keep = (target > 0) | ((mix % np.uint64(100)) == 0)
        users = user[keep]
        y = target[keep]
    m4, p33_rel, p33_global, user_state, candidate_state = feature_arrays(feature_dir(repo, cutoff))
    x = np.concatenate([
        m4[keep], p33_rel[keep], p33_global[keep], user_state[users], candidate_state[keep]
    ], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (len(y), len(FEATURES))
    return x, y, {
        "cutoff": cutoff,
        "full_rows": int(len(target)),
        "retained_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "negative_sampling": "按用户与商品稳定散列保留 1% 负例；正例全部保留",
    }


def fit(repo: Path, cutoff: str, folder: Path):
    xs, ys, audit = [], [], []
    for training_cutoff in earlier_full(cutoff):
        x, y, source = sampled_full(repo, training_cutoff)
        xs.append(x)
        ys.append(y)
        audit.append(source)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = lgb.LGBMClassifier(**PARAMS)
    with threadpool_limits(limits=4):
        model.fit(x, y, feature_name=FEATURES)
    model.booster_.save_model(str(folder / f"{cutoff}-F57.txt"))
    info = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "training": earlier_full(cutoff),
        "sources": audit,
    }
    del x, y, xs, ys
    gc.collect()
    return model, info


def p33_embedding_path(repo: Path, cutoff: str) -> Path:
    base = repo / "artifacts/phase3/phase3-p3.3-v1-multiview-graph-student/students-v1"
    if cutoff in WINDOWS.values():
        window = next(w for w, date in WINDOWS.items() if date == cutoff)
        return base / "outer" / window / "catalog_embeddings.float16.npy"
    return base / "training" / cutoff / "catalog_embeddings.float16.npy"


def build_final_relation(repo: Path, cutoff: str, data: dict):
    """Compute held-out cohort P3.3 features with the exact P3.7B formulas."""
    folder = repo / FINAL_RELATION / cutoff
    if all((folder / name).exists() for name in ("p33-relation.npy", "p33-global.npy", "user-state.npy")):
        return
    folder.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(
        repo / "artifacts/m4/m4-v1-supervised-cold-representation/student-v1/static_catalog/catalog_items.csv",
        dtype={"article_id": str},
    )
    articles = pd.read_csv(
        repo / "data/raw/articles.csv", dtype={"article_id": str},
        usecols=["article_id", "product_type_no", "garment_group_no"],
    ).set_index("article_id")
    aligned = articles.reindex(catalog.article_id)
    if aligned.isna().any().any():
        raise RuntimeError("catalog attributes do not align")
    product_type = aligned.product_type_no.to_numpy(np.int32)
    garment = aligned.garment_group_no.to_numpy(np.int32)
    article_to_row = {article: row for row, article in enumerate(catalog.article_id)}
    history_rows, history_days, history_mask, history_audit = histories(
        repo, data["users"], cutoff, article_to_row,
    )
    m4_embedding = np.load(embedding_path(repo, cutoff), mmap_mode="r")
    p33_embedding = np.load(p33_embedding_path(repo, cutoff), mmap_mode="r")
    user_state = compute_user_state_batch(
        history_rows=history_rows, history_days=history_days,
        history_mask=history_mask, m4_embeddings=m4_embedding,
        catalog_product_type=product_type,
    )
    cold = data["cold"]
    p33_relation = np.lib.format.open_memmap(
        folder / "p33-relation.npy", mode="w+", dtype=np.float32, shape=(len(cold), 16),
    )
    p33_global = np.lib.format.open_memmap(
        folder / "p33-global.npy", mode="w+", dtype=np.float32, shape=(len(cold), 2),
    )
    groups = cold.groupby("user_index", sort=False).indices
    active = np.array(list(groups), dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for start in range(0, len(active), 128):
        users = active[start:start + 128]
        indices = [np.asarray(groups[user], dtype=np.int64) for user in users]
        width = max(map(len, indices))
        items = np.zeros((len(users), width), dtype=np.int32)
        scores = np.zeros((len(users), width), dtype=np.float32)
        for local, rows in enumerate(indices):
            items[local, :len(rows)] = [article_to_row[a] for a in cold.iloc[rows].article_id]
            scores[local, :len(rows)] = cold.iloc[rows].m4_coarse_score.to_numpy(np.float32)
        result = compute_relation_feature_batch(
            candidate_rows=items, m4_coarse_scores=scores,
            history_rows=history_rows[users], history_days=history_days[users],
            history_mask=history_mask[users], m4_embeddings=m4_embedding,
            p33_embeddings=p33_embedding, catalog_product_type=product_type,
            catalog_garment_group=garment, device=device,
        )
        for local, rows in enumerate(indices):
            p33_relation[rows] = result["p33_relation"][local, :len(rows)].reshape(len(rows), 16)
            p33_global[rows] = result["p33_global"][local, :len(rows)]
    p33_relation.flush()
    p33_global.flush()
    np.save(folder / "user-state.npy", user_state)
    del p33_relation, p33_global, user_state, m4_embedding, p33_embedding
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dump(folder / "AUDIT.json", {
        "cutoff": cutoff, "candidate_rows": int(len(cold)),
        "candidate_users": int(len(active)), "history": history_audit,
        "formula": "P3.7B compute_relation_feature_batch exact reuse",
        "device": device, "labels_used": False, "final_week": "not_run",
    })


def final_matrix(repo: Path, cutoff: str, data: dict):
    cold = data["cold"]
    build_final_relation(repo, cutoff, data)
    folder = repo / FINAL_RELATION / cutoff
    m4 = np.load(relation_path(repo, cutoff))
    p33_rel = np.load(folder / "p33-relation.npy")
    p33_global = np.load(folder / "p33-global.npy")
    user_state = np.load(folder / "user-state.npy")
    candidate_state = cold[[
        "m4_coarse_score", "m4_coarse_rank", "interaction_count_before_cutoff",
        "strict_cold_flag", "sparse1_5_flag",
    ]].to_numpy(np.float32)
    x = np.concatenate([
        m4, p33_rel, p33_global,
        user_state[cold.user_index.to_numpy(np.int64)], candidate_state,
    ], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (len(cold), len(FEATURES))
    return x, {
        "candidate_rows": int(len(cold)), "feature_width": int(x.shape[1]),
        "heldout_features_recomputed_without_labels": True,
    }


def select_candidate(data: dict, score: np.ndarray, mode: str):
    if mode == "b0_best":
        order = -data["cold"].b0_rank.to_numpy(float)
    elif mode == "classifier_best":
        order = score
    elif mode == "rrf025_best":
        order = rrf(data, score, .25)
    else:
        raise KeyError(mode)
    return one_per_user(data, order)


def evaluate(data: dict, score: np.ndarray, actions: np.ndarray, mode: str, quota: float):
    chosen = select_candidate(data, score, mode)
    admission = stable_percentile(score[chosen])
    count = max(1, int(math.ceil(quota * len(chosen))))
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    selected = np.lexsort((customer, -admission))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in selected:
        ci = int(chosen[local])
        row = data["cold"].iloc[ci]
        ui = int(row.user_index)
        slot = int(np.argmax(actions[ci]))
        victim = lists[ui, slot]
        lists[ui, slot] = row.article_id
        delta = exact_ap(lists[ui], data["truthsets"][data["users"][ui]]) - data["baseline_ap"][ui]
        events.append({
            "customer_id": data["users"][ui], "candidate": row.article_id,
            "victim": victim, "slot": slot + 1, "delta": float(delta),
            "inserted": int(row.target),
            "removed": int(victim in data["truthsets"][data["users"][ui]]),
            "score": float(score[ci]),
        })
    events = pd.DataFrame(events)
    return {
        "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)),
        "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()),
        "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()),
        "removed": int(events.removed.sum()),
    }, lists, events


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E9_v3_fit",
        "stage": "FINAL-E9-v3 多视角真实 Cold 全量监督",
        "recovery_note": "E9-v1/v2 证明 P3.7B 内层训练用户与最终历史验证用户是互斥子队列，不能按行映射。V3 对最终 Cold50 只读重算同公式 P3.3 关系和用户状态；模型及门槛不变。",
        "observed_bottleneck": "E8 的单教师 35 特征模型在两个历史窗改善 Top5，但 Top1 与高置信准入不稳定。",
        "hypothesis": "P3.3 多视角图教师关系可能补足同品类内具体商品选择；用全量真实 Cold 标签判断，避免伪冷域偏移。",
        "features": {
            "F57": "每个用户—Cold 候选对的 57 维特征：24 维 M4 关系、16 维多视角教师关系、2 维多视角全局时间衰减分、10 维用户状态、5 维候选状态。",
            "denominator": "P3.7B M4 Top200 全量候选行；训练保留全部正例和稳定散列抽取的 1% 负例。",
        },
        "parameters": PARAMS,
        "historical_validation": VALID,
        "development_windows": WINDOWS,
        "candidate_rules": MODES,
        "candidate_diagnostics": "F57 与 B0 的 Cold50 正例 Recall@1/5，并审计 RRF 融合。",
        "admission": "候选用户按所选候选的 F57 分数排序，检验头部 0.1%、0.5%、1%、2%；原 action 模型决定替换位置。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "沿用既有扩量门槛，不修改。",
        "if_fail": "保留 WV3-741，停止扩大 Cold 通道。",
        "cost_limit": "本机 GPU 只做已冻结 embedding 的关系计算，随后 CPU 训练；预计 15 分钟内，不重新训练 embedding。",
        "final_week": "not_run",
        "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "MULTIVIEW_COLD_E9_V3_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E9 v3")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        model, fit_info = fit(repo, cutoff, repo / RUN)
        x, mapping = final_matrix(repo, cutoff, data)
        score = model.booster_.predict(x, num_threads=4)
        np.save(repo / RUN / f"{cutoff}-F57-scores.npy", score)
        actions = action_scores(repo, cutoff, data, True)
        historical[cutoff] = {
            "fit": fit_info, "mapping": mapping,
            "B0": identify(data, -data["cold"].b0_rank.to_numpy(float)),
            "F57": identify(data, score), "rules": {},
        }
        for alpha in ALPHAS:
            historical[cutoff][f"F57-rrf-{alpha:g}"] = identify(data, rrf(data, score, alpha))
        for mode in MODES:
            for quota in QUOTAS:
                key = f"{mode}|q={quota:g}"
                historical[cutoff]["rules"][key], _, _ = evaluate(data, score, actions, mode, quota)
        print("E9", cutoff, historical[cutoff]["F57"], flush=True)
        del data, model, x, score, actions
        gc.collect()
    for mode in MODES:
        for quota in QUOTAS:
            key = f"{mode}|q={quota:g}"
            rows = [historical[t]["rules"][key] for t in VALID]
            aggregate[key] = {
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "beneficial": int(sum(r["beneficial"] for r in rows)),
                "harmful": int(sum(r["harmful"] for r in rows)),
                "mode": mode, "quota": quota,
            }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], MODES.index(pair[1]["mode"])))
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
            actions = action_scores(repo, cutoff, data, False)
            result, lists, events = evaluate(data, score, actions, spec["mode"], spec["quota"])
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = fit_info
            result["mapping"] = mapping
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, model, x, score, actions, lists, events
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
    dump(repo / REPORT / "MULTIVIEW_COLD_E9_V3.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"MULTIVIEW_COLD_E9_V3_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
