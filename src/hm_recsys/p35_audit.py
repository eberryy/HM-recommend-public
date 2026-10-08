from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd

from .m4_contract import atomic_json, file_identity


WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("winter_20200122", "winter", "2020-01-22"),
    ("spring_20200318", "spring", "2020-03-18"),
    ("early_summer_20200624", "early-summer", "2020-06-24"),
    ("late_summer_20200819", "late-summer", "2020-08-19"),
)
FINAL_CUTOFF = "2020-09-16"
P31_RUN = "phase3-v1-candidate-aware-history"
P33_RUN = "phase3-p3.3-v1-multiview-graph-student"
K_VALUES = (5, 10, 20, 50)
AGE_BUCKETS = ("0_7", "8_28", "29_84", "over_84")
PROVENANCE_NAMES = {
    1: "i2v_only",
    2: "direct_only",
    4: "deepwalk_only",
    3: "i2v_direct",
    5: "i2v_deepwalk",
    6: "direct_deepwalk",
    7: "all_three",
}
P33_SIGNS = {
    "winter_20200122": "positive",
    "spring_20200318": "negative",
    "early_summer_20200624": "negative",
    "late_summer_20200819": "positive",
}


@dataclass(frozen=True)
class Config:
    repo_root: Path
    source_root: Path
    output_dir: Path
    memory_limit: str = "8GB"
    threads: int = 8


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_cutoff(cutoff: str) -> None:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError(f"P3.5 refuses final or later cutoff: {cutoff}")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _summary(values: Iterable[float]) -> dict[str, Any]:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p25": float(np.quantile(a, 0.25)),
        "p75": float(np.quantile(a, 0.75)),
    }


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def _jsd(left: Counter[int], right: Counter[int]) -> float | None:
    keys = sorted(set(left) | set(right))
    if not keys or not sum(left.values()) or not sum(right.values()):
        return None
    p = np.asarray([left[k] for k in keys], dtype=np.float64)
    q = np.asarray([right[k] for k in keys], dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    left_term = np.zeros_like(p)
    right_term = np.zeros_like(q)
    left_valid = p > 0
    right_valid = q > 0
    left_term[left_valid] = p[left_valid] * np.log2(p[left_valid] / m[left_valid])
    right_term[right_valid] = q[right_valid] * np.log2(q[right_valid] / m[right_valid])
    return float(0.5 * (left_term.sum() + right_term.sum()))


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[cursor]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _rank_by_score(users: np.ndarray, items: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(users), dtype=np.int32)
    cursor = 0
    while cursor < len(users):
        end = cursor + 1
        while end < len(users) and users[end] == users[cursor]:
            end += 1
        order = np.lexsort((items[cursor:end], -scores[cursor:end]))
        ranks[cursor + order] = np.arange(1, end - cursor + 1, dtype=np.int32)
        cursor = end
    return ranks


def _quartile_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    labels = np.full(len(values), "ineligible", dtype=object)
    valid = np.isfinite(values)
    if not valid.any():
        return labels, []
    cuts = np.quantile(values[valid], [0.25, 0.5, 0.75]).tolist()
    labels[valid] = np.asarray(
        [f"Q{min(int(np.searchsorted(cuts, value, side='right')) + 1, 4)}" for value in values[valid]],
        dtype=object,
    )
    return labels, [float(value) for value in cuts]


def _age_bucket(days: float | None) -> str:
    if days is None or not math.isfinite(float(days)) or float(days) > 84:
        return "over_84_or_none"
    if days <= 7:
        return "0_7"
    if days <= 28:
        return "8_28"
    return "29_84"


def _pairwise_dispersion(rows: list[int], embeddings: np.ndarray) -> tuple[float | None, float | None]:
    unique = np.unique(np.asarray(rows, dtype=np.int32))
    if len(unique) < 2:
        return None, None
    matrix = np.asarray(embeddings[unique], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    total = matrix.sum(axis=0)
    similarity_sum = float(total @ total - len(matrix))
    mean_similarity = similarity_sum / (len(matrix) * (len(matrix) - 1))
    centroid_norm = float(np.linalg.norm(total / len(matrix)))
    return 1.0 - mean_similarity, 1.0 - centroid_norm


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _window_paths(
    config: Config, window: str, cutoff: str, p31: dict[str, Any], p32: dict[str, Any], p33: dict[str, Any]
) -> dict[str, Any]:
    coarse = p31["windows"][window]
    training_cutoffs = p33["students"]["assets"]["outer_validation"][window]["training_cutoffs"]
    return {
        "candidates": Path(coarse["artifacts"]["candidates"]["path"]),
        "histories": Path(coarse["artifacts"]["histories"]["path"]),
        "users": Path(coarse["artifacts"]["users"]["path"]),
        "m4_embedding": Path(coarse["student_embedding"]["path"]),
        "p32_scores": Path(p32["windows"][window]["scoring"]["scores"]["path"]),
        "p33_embedding": config.repo_root / "artifacts" / "phase3" / P33_RUN / "students-v1" / "outer" / window / "catalog_embeddings.float16.npy",
        "teacher_dirs": [
            config.repo_root / "artifacts" / "phase3" / P33_RUN / "teachers-v1" / value
            for value in training_cutoffs
        ],
        "teacher_cutoffs": list(training_cutoffs),
    }


def _load_history_frame(
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    users: pd.DataFrame,
    cutoff: str,
) -> pd.DataFrame:
    _safe_cutoff(cutoff)
    connection.register("p35_eval_users", users[["customer_id"]])
    try:
        frame = connection.execute(
            f"""
            SELECT t.customer_id,CAST(t.article_id AS VARCHAR) AS article_id,
                   date_diff('day',max(t.t_dat),DATE '{cutoff}')::INTEGER AS days_since,
                   count(*)::INTEGER AS events
            FROM read_parquet('{_sql_path(transactions)}') t
            SEMI JOIN p35_eval_users u USING(customer_id)
            WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 730 DAY AND t.t_dat<DATE '{cutoff}'
            GROUP BY t.customer_id,t.article_id
            ORDER BY t.customer_id,days_since,article_id
            """
        ).fetch_df()
        latest = connection.execute(
            f"SELECT max(t_dat)::VARCHAR FROM read_parquet('{_sql_path(transactions)}') WHERE t_dat<DATE '{cutoff}'"
        ).fetchone()[0]
    finally:
        connection.unregister("p35_eval_users")
    if latest is None or latest >= cutoff:
        raise RuntimeError(f"cutoff boundary failed for {cutoff}: {latest}")
    return frame


def _load_truth_and_counts(
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    users: pd.DataFrame,
    cutoff: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection.register("p35_eval_users", users[["customer_id"]])
    try:
        truth = connection.execute(
            f"""
            SELECT DISTINCT t.customer_id,CAST(t.article_id AS VARCHAR) AS article_id
            FROM read_parquet('{_sql_path(transactions)}') t
            SEMI JOIN p35_eval_users u USING(customer_id)
            WHERE t.t_dat>=DATE '{cutoff}' AND t.t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
            """
        ).fetch_df()
        counts = connection.execute(
            f"""
            SELECT CAST(article_id AS VARCHAR) AS article_id,count(*)::INTEGER AS events
            FROM read_parquet('{_sql_path(transactions)}')
            WHERE t_dat<DATE '{cutoff}' GROUP BY article_id
            """
        ).fetch_df()
    finally:
        connection.unregister("p35_eval_users")
    return truth, counts


def _profiles_and_catalog_diversity(
    *,
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    articles: pd.DataFrame,
    users: pd.DataFrame,
    history: pd.DataFrame,
    truth: pd.DataFrame,
    counts: pd.DataFrame,
    embeddings: np.ndarray,
    cutoff: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame]:
    article_fields = articles[["article_id", "catalog_row", "product_type_no", "garment_group_no"]]
    h = history.merge(article_fields, on="article_id", how="left", validate="many_to_one")
    user_to_index = {value: index for index, value in enumerate(users["customer_id"].astype(str))}
    user_count = len(users)
    recent_type_counts: list[Counter[int]] = [Counter() for _ in range(user_count)]
    past_type_counts: list[Counter[int]] = [Counter() for _ in range(user_count)]
    recent_rows: list[list[int]] = [[] for _ in range(user_count)]
    past_rows: list[list[int]] = [[] for _ in range(user_count)]
    recent_days: list[list[float]] = [[] for _ in range(user_count)]
    past_days: list[list[float]] = [[] for _ in range(user_count)]
    type_support: dict[tuple[int, int], int] = defaultdict(int)
    garment_support: dict[tuple[int, int], int] = defaultdict(int)
    type_last: dict[tuple[int, int], float] = {}
    garment_last: dict[tuple[int, int], float] = {}
    within_type_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in h.itertuples(index=False):
        user_index = user_to_index.get(str(row.customer_id))
        if user_index is None or pd.isna(row.catalog_row):
            continue
        catalog_row, days = int(row.catalog_row), float(row.days_since)
        ptype = int(row.product_type_no) if not pd.isna(row.product_type_no) else -1
        garment = int(row.garment_group_no) if not pd.isna(row.garment_group_no) else -1
        if days <= 84:
            type_support[(user_index, ptype)] += 1
            garment_support[(user_index, garment)] += 1
            type_last[(user_index, ptype)] = min(type_last.get((user_index, ptype), math.inf), days)
            garment_last[(user_index, garment)] = min(garment_last.get((user_index, garment), math.inf), days)
            within_type_rows[(user_index, ptype)].append(catalog_row)
        if days <= 28:
            recent_type_counts[user_index][ptype] += int(row.events)
            recent_rows[user_index].append(catalog_row)
            recent_days[user_index].append(days)
        elif days <= 84:
            past_type_counts[user_index][ptype] += int(row.events)
            past_rows[user_index].append(catalog_row)
            past_days[user_index].append(days)

    global_counts_frame = connection.execute(
        f"""
        SELECT a.product_type_no,count(*)::BIGINT AS events
        FROM read_parquet('{_sql_path(transactions)}') t
        JOIN read_csv_auto('{_sql_path(articles.attrs['source_path'])}',all_varchar=true) a USING(article_id)
        WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t.t_dat<DATE '{cutoff}'
        GROUP BY a.product_type_no
        """
    ).fetch_df()
    global_type = Counter({int(row.product_type_no): int(row.events) for row in global_counts_frame.itertuples(index=False)})

    drift = np.full(user_count, np.nan, dtype=np.float64)
    user_global = np.full(user_count, np.nan, dtype=np.float64)
    stability = np.full(user_count, np.nan, dtype=np.float64)
    weighted_stability = np.full(user_count, np.nan, dtype=np.float64)
    q_recent: list[np.ndarray | None] = [None] * user_count
    q_past: list[np.ndarray | None] = [None] * user_count
    for index in range(user_count):
        if sum(recent_type_counts[index].values()) >= 2 and sum(past_type_counts[index].values()) >= 2:
            drift[index] = float(_jsd(recent_type_counts[index], past_type_counts[index]))
            user_global[index] = float(_jsd(recent_type_counts[index], global_type))
        if recent_rows[index] and past_rows[index]:
            qr = _normalize(np.asarray(embeddings[recent_rows[index]], dtype=np.float32).mean(axis=0))
            qp = _normalize(np.asarray(embeddings[past_rows[index]], dtype=np.float32).mean(axis=0))
            q_recent[index], q_past[index] = qr, qp
            if qr is not None and qp is not None:
                stability[index] = float(qr @ qp)
            wr = np.power(0.5, np.asarray(recent_days[index]) / 28.0)
            wp = np.power(0.5, np.asarray(past_days[index]) / 28.0)
            qrw = _normalize((np.asarray(embeddings[recent_rows[index]], dtype=np.float32) * wr[:, None]).sum(axis=0))
            qpw = _normalize((np.asarray(embeddings[past_rows[index]], dtype=np.float32) * wp[:, None]).sum(axis=0))
            if qrw is not None and qpw is not None:
                weighted_stability[index] = float(qrw @ qpw)

    dispersion = np.full(user_count, np.nan, dtype=np.float64)
    centroid_dispersion = np.full(user_count, np.nan, dtype=np.float64)
    qualifying_groups = np.zeros(user_count, dtype=np.int32)
    unique_items = np.zeros(user_count, dtype=np.int32)
    accum_distance: dict[int, list[float]] = defaultdict(list)
    accum_centroid: dict[int, list[float]] = defaultdict(list)
    type_dispersion: dict[tuple[int, int], float] = {}
    for (user_index, _ptype), rows in within_type_rows.items():
        pairwise, centroid = _pairwise_dispersion(rows, embeddings)
        if pairwise is not None:
            type_dispersion[(user_index, _ptype)] = float(pairwise)
            accum_distance[user_index].append(pairwise)
            accum_centroid[user_index].append(float(centroid))
            qualifying_groups[user_index] += 1
            unique_items[user_index] += len(set(rows))
    for user_index, values in accum_distance.items():
        dispersion[user_index] = float(np.mean(values))
        centroid_dispersion[user_index] = float(np.mean(accum_centroid[user_index]))

    truth_rows = truth.merge(counts, on="article_id", how="left").fillna({"events": 0})
    truth_rows = truth_rows.merge(article_fields, on="article_id", how="left")
    truth_rows["user_index"] = truth_rows["customer_id"].astype(str).map(user_to_index)
    truth_rows = truth_rows[(truth_rows["events"] <= 5) & truth_rows["user_index"].notna()].copy()
    truth_rows["user_index"] = truth_rows["user_index"].astype(int)
    truth_rows["segment"] = np.where(truth_rows["events"] == 0, "strict_cold", "sparse_1_5")
    truth_rows["same_type_support"] = [type_support.get((u, int(t)), 0) for u, t in zip(truth_rows.user_index, truth_rows.product_type_no)]
    truth_rows["same_garment_support"] = [garment_support.get((u, int(g)), 0) for u, g in zip(truth_rows.user_index, truth_rows.garment_group_no)]
    truth_rows["days_since_same_type"] = [type_last.get((u, int(t)), np.nan) for u, t in zip(truth_rows.user_index, truth_rows.product_type_no)]
    truth_rows["days_since_same_garment"] = [garment_last.get((u, int(g)), np.nan) for u, g in zip(truth_rows.user_index, truth_rows.garment_group_no)]

    recent_catalog = connection.execute(
        f"""
        SELECT CAST(t.article_id AS VARCHAR) AS article_id,a.product_type_no,count(*)::BIGINT AS events
        FROM read_parquet('{_sql_path(transactions)}') t
        JOIN read_csv_auto('{_sql_path(articles.attrs['source_path'])}',all_varchar=true) a USING(article_id)
        WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t.t_dat<DATE '{cutoff}'
        GROUP BY t.article_id,a.product_type_no
        """
    ).fetch_df()
    type_stats: dict[int, dict[str, Any]] = {}
    for ptype, group in recent_catalog.groupby("product_type_no"):
        values = group["events"].to_numpy(dtype=np.float64)
        probabilities = values / values.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
        type_stats[int(ptype)] = {
            "active_sku_count": int(len(values)),
            "article_entropy": entropy,
            "normalized_entropy": entropy / math.log(len(values)) if len(values) > 1 else 0.0,
            "hhi": float(np.square(probabilities).sum()),
        }
    truth_type_counts = Counter(int(value) for value in truth_rows["product_type_no"].dropna())
    denominator = sum(truth_type_counts.values())
    weighted = {
        field: (
            sum(type_stats.get(ptype, {}).get(field, 0.0) * count for ptype, count in truth_type_counts.items()) / denominator
            if denominator else None
        )
        for field in ("active_sku_count", "article_entropy", "normalized_entropy", "hhi")
    }
    truth_users = set(truth_rows["user_index"].astype(int))
    profile = {
        "eligible_evaluation_users": user_count,
        "recent_vs_past_product_type_jsd": _summary(drift),
        "recent_vs_global_product_type_jsd": _summary(user_global),
        "embedding_centroid_stability_cosine": _summary(stability),
        "recency_weighted_centroid_stability_cosine": _summary(weighted_stability),
        "within_type_embedding_dispersion": {
            "all_evaluation_users": _summary(dispersion),
            "cold_sparse_truth_users": _summary(dispersion[list(truth_users)] if truth_users else []),
            "centroid_dispersion_all": _summary(centroid_dispersion),
            "qualifying_type_groups": int(qualifying_groups.sum()),
            "unique_items_in_qualifying_groups": int(unique_items.sum()),
        },
        "cold_sparse_truth_history_support": {
            "truth_pairs": int(len(truth_rows)),
            "same_product_type_count": _summary(truth_rows["same_type_support"]),
            "same_garment_count": _summary(truth_rows["same_garment_support"]),
            "days_since_last_same_product_type": _summary(truth_rows["days_since_same_type"]),
            "days_since_last_same_garment": _summary(truth_rows["days_since_same_garment"]),
        },
        "recent_28d_within_product_type_catalog_diversity": {
            "product_types": int(len(type_stats)),
            "truth_product_type_weighted": weighted,
            "truth_pair_weight_denominator": denominator,
        },
    }
    arrays = {
        "recent_past_jsd": drift,
        "user_global_jsd": user_global,
        "stability": stability,
        "weighted_stability": weighted_stability,
        "dispersion": dispersion,
        "type_support_map": type_support,
        "type_last_map": type_last,
        "type_dispersion_map": type_dispersion,
    }
    return profile, arrays, truth_rows


def _ordering_metrics(
    candidates: dict[str, np.ndarray],
    ranks: np.ndarray,
    truth_rows: pd.DataFrame,
    counts_by_row: np.ndarray,
    user_count: int,
) -> dict[str, Any]:
    users = candidates["user_index"].astype(np.int32)
    items = candidates["catalog_row"].astype(np.int32)
    target = candidates["target"].astype(bool)
    output: dict[str, Any] = {"at_k": {}}
    truth_sets: dict[str, dict[int, set[int]]] = {}
    for segment in ("strict_cold", "sparse_1_5"):
        sub = truth_rows[truth_rows["segment"] == segment]
        truth_sets[segment] = {
            int(user): set(group["catalog_row"].dropna().astype(int))
            for user, group in sub.groupby("user_index")
        }
    truth_sets["cold_universe"] = {
        int(user): set(group["catalog_row"].dropna().astype(int))
        for user, group in truth_rows.groupby("user_index")
    }
    for k in K_VALUES:
        selected = ranks <= k
        selected_by_user: dict[int, set[int]] = defaultdict(set)
        for user, item in zip(users[selected], items[selected]):
            selected_by_user[int(user)].add(int(item))
        segments: dict[str, Any] = {}
        for name, mapping in truth_sets.items():
            recalls, hits, covered, pairs = [], [], 0, 0
            for user, relevant in mapping.items():
                matched = len(relevant & selected_by_user.get(user, set()))
                recalls.append(matched / len(relevant))
                hits.append(float(matched > 0))
                covered += matched
                pairs += len(relevant)
            segments[name] = {
                "recall": float(np.mean(recalls)) if recalls else 0.0,
                "hit_rate": float(np.mean(hits)) if hits else 0.0,
                "truth_users": len(recalls),
                "truth_pairs": pairs,
                "covered_truth_pairs": covered,
            }
        output["at_k"][str(k)] = {
            "candidate_rows": int(selected.sum()),
            "positive_rows": int(np.count_nonzero(selected & target)),
            "positive_density": float(np.count_nonzero(selected & target) / max(int(selected.sum()), 1)),
            "segments": segments,
        }
    reciprocal = []
    first_ranks = []
    for user in truth_sets["cold_universe"]:
        mask = (users == user) & target
        if mask.any():
            first = int(ranks[mask].min())
            reciprocal.append(1.0 / first)
            first_ranks.append(first)
        else:
            reciprocal.append(0.0)
    positive_ranks = ranks[target]
    output["ranking"] = {
        "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
        "mrr_truth_user_denominator": len(reciprocal),
        "top200_positive_pairs": int(target.sum()),
        "top200_to_top20": float(np.count_nonzero(positive_ranks <= 20) / max(len(positive_ranks), 1)),
        "first_positive_rank_median_hit_users": float(np.median(first_ranks)) if first_ranks else None,
    }
    return output


def _same_user_percentile(users: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    percentiles: list[float] = []
    cursor = 0
    while cursor < len(users):
        end = cursor + 1
        while end < len(users) and users[end] == users[cursor]:
            end += 1
        local_labels = labels[cursor:end].astype(bool)
        negatives = scores[cursor:end][~local_labels]
        for value in scores[cursor:end][local_labels]:
            if len(negatives):
                percentiles.append(float((np.count_nonzero(negatives < value) + 0.5 * np.count_nonzero(negatives == value)) / len(negatives)))
        cursor = end
    result = _summary(percentiles)
    result["unit"] = "Top200 positive user-item pair"
    return result


def _fixed_query_and_half_life(
    *,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    m4_embeddings: np.ndarray,
    p32_scores: np.ndarray,
    truth_rows: pd.DataFrame,
    counts_by_row: np.ndarray,
    user_states: dict[str, np.ndarray],
    catalog_types: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    users = candidates["user_index"].astype(np.int32)
    items = candidates["catalog_row"].astype(np.int32)
    labels = candidates["target"].astype(np.uint8)
    coarse_scores = candidates["coarse_score"].astype(np.float32)
    history_rows = histories["catalog_row"].astype(np.int32)
    history_days = histories["days_since_purchase"].astype(np.float32)
    history_mask = histories["mask"].astype(bool)
    mean_scores = np.full(len(users), -np.inf, dtype=np.float32)
    recency_scores = np.full(len(users), -np.inf, dtype=np.float32)
    bucket_features: dict[str, dict[str, np.ndarray]] = {
        bucket: {
            "best_cosine": np.full(len(users), np.nan, dtype=np.float32),
            "mean_top3_cosine": np.full(len(users), np.nan, dtype=np.float32),
            "same_product_type_support": np.zeros(len(users), dtype=np.float32),
            "fixed_centroid_similarity": np.full(len(users), np.nan, dtype=np.float32),
        }
        for bucket in AGE_BUCKETS
    }
    cursor = 0
    while cursor < len(users):
        end = cursor + 1
        user_index = int(users[cursor])
        while end < len(users) and users[end] == user_index:
            end += 1
        valid_rows = history_rows[user_index][history_mask[user_index]]
        valid_days = history_days[user_index][history_mask[user_index]]
        candidate_matrix = np.asarray(m4_embeddings[items[cursor:end]], dtype=np.float32)
        if len(valid_rows):
            history_matrix = np.asarray(m4_embeddings[valid_rows], dtype=np.float32)
            query_mean = _normalize(history_matrix.mean(axis=0))
            weights = np.power(0.5, valid_days / 28.0).astype(np.float32)
            query_recency = _normalize((history_matrix * weights[:, None]).sum(axis=0))
            if query_mean is not None:
                mean_scores[cursor:end] = candidate_matrix @ query_mean
            if query_recency is not None:
                recency_scores[cursor:end] = candidate_matrix @ query_recency
            similarities = candidate_matrix @ history_matrix.T
            for bucket in AGE_BUCKETS:
                if bucket == "0_7":
                    selected = valid_days <= 7
                elif bucket == "8_28":
                    selected = (valid_days > 7) & (valid_days <= 28)
                elif bucket == "29_84":
                    selected = (valid_days > 28) & (valid_days <= 84)
                else:
                    selected = valid_days > 84
                if not selected.any():
                    continue
                local = similarities[:, selected]
                top_count = min(3, local.shape[1])
                top_values = np.partition(local, local.shape[1] - top_count, axis=1)[:, -top_count:]
                centroid = _normalize(history_matrix[selected].mean(axis=0))
                bucket_features[bucket]["best_cosine"][cursor:end] = local.max(axis=1)
                bucket_features[bucket]["mean_top3_cosine"][cursor:end] = top_values.mean(axis=1)
                if centroid is not None:
                    bucket_features[bucket]["fixed_centroid_similarity"][cursor:end] = candidate_matrix @ centroid
                candidate_types = catalog_types[items[cursor:end]]
                selected_types = catalog_types[valid_rows[selected]]
                bucket_features[bucket]["same_product_type_support"][cursor:end] = np.asarray(
                    [np.count_nonzero(selected_types == value) for value in candidate_types], dtype=np.float32
                )
        cursor = end

    score_map = {
        "p31_seed_based": coarse_scores,
        "fixed_mean_centroid": mean_scores,
        "fixed_recency_centroid": recency_scores,
        "p32_candidate_aware": np.asarray(p32_scores, dtype=np.float32),
    }
    rank_map = {name: _rank_by_score(users, items, scores) for name, scores in score_map.items()}
    fixed: dict[str, Any] = {
        "candidate_identity": {
            "rows": int(len(users)),
            "exact_p31_top200_reuse": True,
            "identity_sha256": hashlib.sha256(np.column_stack([users, items]).astype(np.int32).tobytes()).hexdigest(),
        },
        "orderings": {},
        "stability_strata": {},
    }
    for name, scores in score_map.items():
        metrics = _ordering_metrics(candidates, rank_map[name], truth_rows, counts_by_row, len(history_rows))
        metrics["score_auc"] = _auc(labels, scores)
        metrics["same_user_truth_percentile"] = _same_user_percentile(users, labels, scores)
        fixed["orderings"][name] = metrics
    stability_labels, stability_cuts = _quartile_labels(user_states["stability"])
    fixed["stability_quartile_cuts"] = stability_cuts
    for bin_name in ("Q1", "Q2", "Q3", "Q4", "ineligible"):
        row_mask = stability_labels[users] == bin_name
        truth_users = set(
            truth_rows.loc[stability_labels[truth_rows["user_index"].to_numpy(dtype=np.int32)] == bin_name, "user_index"].astype(int)
        )
        fixed["stability_strata"][bin_name] = {
            "users": int(len(truth_users)),
            "positive_pairs_in_top200": int(labels[row_mask].sum()),
            "orderings": {},
        }
        for name, scores in score_map.items():
            local: dict[str, Any] = {}
            for k in K_VALUES:
                selected = row_mask & (rank_map[name] <= k)
                local[f"density@{k}"] = float(labels[selected].sum() / max(int(selected.sum()), 1))
            positive = row_mask & (labels == 1)
            local["top200_to_top20"] = float(np.count_nonzero(positive & (rank_map[name] <= 20)) / max(int(np.count_nonzero(positive)), 1))
            reciprocal = []
            for user in truth_users:
                user_positive = row_mask & (users == user) & (labels == 1)
                reciprocal.append(1.0 / int(rank_map[name][user_positive].min()) if user_positive.any() else 0.0)
            local["mrr"] = float(np.mean(reciprocal)) if reciprocal else 0.0
            local["score_auc"] = _auc(labels[row_mask], scores[row_mask])
            local["same_user_truth_percentile"] = _same_user_percentile(users[row_mask], labels[row_mask], scores[row_mask])
            fixed["stability_strata"][bin_name]["orderings"][name] = local

    half_life: dict[str, Any] = {"age_buckets": {}, "most_informative_truth_history_age": {}}
    positive = labels == 1
    for bucket in AGE_BUCKETS:
        half_life["age_buckets"][bucket] = {}
        for feature, values in bucket_features[bucket].items():
            truth_values = values[positive]
            other_values = values[~positive]
            half_life["age_buckets"][bucket][feature] = {
                "auc": _auc(labels, values),
                "truth": _summary(truth_values),
                "unobserved": _summary(other_values),
                "median_gap": (
                    float(np.nanmedian(truth_values) - np.nanmedian(other_values))
                    if np.isfinite(truth_values).any() and np.isfinite(other_values).any() else None
                ),
            }
    truth_indices = np.flatnonzero(positive)
    most = Counter()
    for index in truth_indices:
        values = [bucket_features[bucket]["best_cosine"][index] for bucket in AGE_BUCKETS]
        if np.isfinite(values).any():
            most[AGE_BUCKETS[int(np.nanargmax(values))]] += 1
        else:
            most["no_history"] += 1
    half_life["most_informative_truth_history_age"] = {
        "counts": dict(most),
        "denominator_top200_truth_pairs": int(len(truth_indices)),
    }
    return fixed, half_life, rank_map


def _stratum_metrics(
    *,
    candidates: dict[str, np.ndarray],
    coarse_rank: np.ndarray,
    candidate_rank: np.ndarray,
    candidate_bins: np.ndarray,
    truth_rows: pd.DataFrame,
    truth_bins: np.ndarray,
    bin_name: str,
) -> dict[str, Any]:
    users = candidates["user_index"].astype(np.int32)
    labels = candidates["target"].astype(bool)
    row_mask = candidate_bins == bin_name
    truth_mask = truth_bins == bin_name
    truth_sub = truth_rows.loc[truth_mask]
    truth_users = set(truth_sub["user_index"].astype(int))

    def metrics(ranks: np.ndarray) -> dict[str, float]:
        selected = row_mask & (ranks <= 20)
        density = float(np.count_nonzero(selected & labels) / max(int(np.count_nonzero(selected)), 1))
        reciprocal: list[float] = []
        for user in truth_users:
            local = row_mask & labels & (users == user)
            reciprocal.append(1.0 / int(ranks[local].min()) if local.any() else 0.0)
        positives = row_mask & labels
        conversion = float(np.count_nonzero(positives & (ranks <= 20)) / max(int(np.count_nonzero(positives)), 1))
        return {"density@20": density, "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0, "top200_to_top20": conversion}

    coarse = metrics(coarse_rank)
    candidate = metrics(candidate_rank)
    return {
        "users": len(truth_users),
        "truth_pairs": int(len(truth_sub)),
        "top200_positive_pairs": int(np.count_nonzero(row_mask & labels)),
        "candidate_rows": int(np.count_nonzero(row_mask)),
        "coarse": coarse,
        "candidate_aware": candidate,
        "delta": {key: candidate[key] - coarse[key] for key in coarse},
    }


def _p32_stratification(
    *,
    candidates: dict[str, np.ndarray],
    ranks: dict[str, np.ndarray],
    truth_rows: pd.DataFrame,
    user_states: dict[str, Any],
    catalog_types: np.ndarray,
) -> dict[str, Any]:
    users = candidates["user_index"].astype(np.int32)
    items = candidates["catalog_row"].astype(np.int32)
    result: dict[str, Any] = {"variables": {}}
    specifications: list[tuple[str, np.ndarray, np.ndarray, list[str], list[float] | None]] = []
    for name, key in (
        ("user_recent_vs_past_product_type_jsd", "recent_past_jsd"),
        ("user_vs_global_recent_product_type_jsd", "user_global_jsd"),
        ("embedding_centroid_stability_cosine", "stability"),
        ("within_type_embedding_dispersion", "dispersion"),
    ):
        user_labels, cuts = _quartile_labels(np.asarray(user_states[key], dtype=np.float64))
        specifications.append(
            (name, user_labels[users], user_labels[truth_rows["user_index"].to_numpy(dtype=np.int32)], ["Q1", "Q2", "Q3", "Q4", "ineligible"], cuts)
        )

    support_map = user_states["type_support_map"]
    last_map = user_states["type_last_map"]
    dispersion_map = user_states["type_dispersion_map"]
    candidate_types = catalog_types[items]
    candidate_support = np.asarray([support_map.get((int(u), int(t)), 0) for u, t in zip(users, candidate_types)], dtype=np.int32)
    truth_support = truth_rows["same_type_support"].to_numpy(dtype=np.int32)
    candidate_support_bins = np.where(candidate_support == 0, "0", np.where(candidate_support == 1, "1", "2_plus"))
    truth_support_bins = np.where(truth_support == 0, "0", np.where(truth_support == 1, "1", "2_plus"))
    specifications.append(("same_product_type_history_support", candidate_support_bins, truth_support_bins, ["0", "1", "2_plus"], None))

    candidate_last = np.asarray([last_map.get((int(u), int(t)), np.nan) for u, t in zip(users, candidate_types)], dtype=np.float64)
    truth_last = truth_rows["days_since_same_type"].to_numpy(dtype=np.float64)
    specifications.append(
        (
            "days_since_last_same_product_type",
            np.asarray([_age_bucket(value) for value in candidate_last], dtype=object),
            np.asarray([_age_bucket(value) for value in truth_last], dtype=object),
            ["0_7", "8_28", "29_84", "over_84_or_none"],
            None,
        )
    )

    candidate_type_dispersion = np.asarray([dispersion_map.get((int(u), int(t)), np.nan) for u, t in zip(users, candidate_types)], dtype=np.float64)
    truth_type_dispersion = np.asarray(
        [dispersion_map.get((int(u), int(t)), np.nan) for u, t in zip(truth_rows.user_index, truth_rows.product_type_no)], dtype=np.float64
    )
    combined = np.concatenate([candidate_type_dispersion, truth_type_dispersion])
    combined_labels, cuts = _quartile_labels(combined)
    specifications.append(
        (
            "candidate_relative_within_type_embedding_dispersion",
            combined_labels[: len(candidate_type_dispersion)],
            combined_labels[len(candidate_type_dispersion) :],
            ["Q1", "Q2", "Q3", "Q4", "ineligible"],
            cuts,
        )
    )

    for name, candidate_bins, truth_bins, bin_names, cuts in specifications:
        result["variables"][name] = {
            "quartile_cuts": cuts,
            "bins": {
                bin_name: _stratum_metrics(
                    candidates=candidates,
                    coarse_rank=ranks["p31_seed_based"],
                    candidate_rank=ranks["p32_candidate_aware"],
                    candidate_bins=np.asarray(candidate_bins),
                    truth_rows=truth_rows,
                    truth_bins=np.asarray(truth_bins),
                    bin_name=bin_name,
                )
                for bin_name in bin_names
            },
        }
    return result


def _neighbor_pairs(path: Path, depth: int, catalog_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = np.load(path)
    anchors = loaded["anchors"].astype(np.int32)
    neighbors = loaded["neighbors"][:, :depth].astype(np.int32)
    ranks = np.broadcast_to(np.arange(1, depth + 1, dtype=np.int16), neighbors.shape)
    repeated = np.broadcast_to(anchors[:, None], neighbors.shape)
    valid = neighbors >= 0
    pair_ids = repeated[valid].astype(np.int64) * catalog_size + neighbors[valid]
    return pair_ids, repeated[valid].astype(np.int32), ranks[valid].astype(np.int16)


def _teacher_union(
    teacher_dir: Path,
    catalog_size: int,
    catalog_types: np.ndarray,
    catalog_garments: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    paths = {
        "item2vec": teacher_dir / "neighbors_item2vec.npz",
        "direct_covisit": teacher_dir / "neighbors_direct_covisit.npz",
        "deepwalk": teacher_dir / "neighbors_deepwalk.npz",
    }
    view_bits = {"item2vec": 1, "direct_covisit": 2, "deepwalk": 4}
    top20, top100 = {}, {}
    ranks20 = {}
    anchors20 = {}
    for name, path in paths.items():
        ids20, anchors, ranks = _neighbor_pairs(path, 20, catalog_size)
        ids100, _, _ = _neighbor_pairs(path, 100, catalog_size)
        top20[name] = np.unique(ids20)
        top100[name] = np.unique(ids100)
        ranks20[name] = dict(zip(ids20.tolist(), ranks.tolist()))
        anchors20[name] = anchors
    joined = np.concatenate([top20[name] for name in view_bits])
    bits = np.concatenate([np.full(len(top20[name]), bit, dtype=np.uint8) for name, bit in view_bits.items()])
    relation_ids, inverse = np.unique(joined, return_inverse=True)
    masks = np.zeros(len(relation_ids), dtype=np.uint8)
    np.bitwise_or.at(masks, inverse, bits)
    anchors = (relation_ids // catalog_size).astype(np.int32)
    neighbors = (relation_ids % catalog_size).astype(np.int32)
    direct_npz = np.load(paths["direct_covisit"])
    degree_map = dict(zip(direct_npz["anchors"].astype(int), np.sum(direct_npz["neighbors"] >= 0, axis=1).astype(int)))
    conflict = np.zeros(len(relation_ids), dtype=bool)
    for name, bit in view_bits.items():
        is_top20 = (masks & bit) > 0
        for other, other_bit in view_bits.items():
            if other == name:
                continue
            conflict |= is_top20 & ~np.isin(relation_ids, top100[other], assume_unique=True)
    frame = pd.DataFrame(
        {
            "relation_id": relation_ids,
            "anchor_catalog_row": anchors,
            "neighbor_catalog_row": neighbors,
            "provenance_mask": masks,
            "provenance": [PROVENANCE_NAMES[int(mask)] for mask in masks],
            "product_type_match": catalog_types[anchors] == catalog_types[neighbors],
            "garment_match": catalog_garments[anchors] == catalog_garments[neighbors],
            "anchor_graph_degree": [degree_map.get(int(anchor), 0) for anchor in anchors],
            "teacher_conflict": conflict,
        }
    )
    overlap: dict[str, Any] = {}
    names = list(view_bits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap[f"{left}__{right}"] = {}
            for depth, sets in ((20, top20), (100, top100)):
                intersection = int(len(np.intersect1d(sets[left], sets[right], assume_unique=True)))
                union = int(len(np.union1d(sets[left], sets[right])))
                overlap[f"{left}__{right}"][f"top{depth}"] = {
                    "intersection_relations": intersection,
                    "union_relations": union,
                    "jaccard": intersection / max(union, 1),
                }
    audit = {
        "top20_union_relations": int(len(frame)),
        "top20_input_relations": {name: int(len(value)) for name, value in top20.items()},
        "provenance_union_conserved": int(sum(Counter(frame["provenance"]).values())) == len(frame),
        "pairwise_teacher_overlap": overlap,
        "teacher_conflict_relations": int(frame["teacher_conflict"].sum()),
        "teacher_conflict_rate": float(frame["teacher_conflict"].mean()),
    }
    lookups = {"relation_ids": relation_ids, "provenance_masks": masks, **{f"top100_{name}": values for name, values in top100.items()}}
    return frame, audit, lookups


def _relation_support(
    connection: duckdb.DuckDBPyConnection,
    *,
    transactions: Path,
    catalog_items: np.ndarray,
    relations: pd.DataFrame,
    cutoff: str,
) -> pd.DataFrame:
    canonical_left = np.minimum(relations["anchor_catalog_row"].to_numpy(), relations["neighbor_catalog_row"].to_numpy())
    canonical_right = np.maximum(relations["anchor_catalog_row"].to_numpy(), relations["neighbor_catalog_row"].to_numpy())
    pair_frame = pd.DataFrame(
        {
            "left_article_id": catalog_items[canonical_left],
            "right_article_id": catalog_items[canonical_right],
        }
    ).drop_duplicates()
    connection.register("p35_relation_pairs", pair_frame)
    try:
        support = connection.execute(
            f"""
            WITH valid_days AS (
              SELECT customer_id,t_dat,count(DISTINCT article_id) AS n
              FROM read_parquet('{_sql_path(transactions)}')
              WHERE t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t_dat<DATE '{cutoff}'
              GROUP BY customer_id,t_dat HAVING n BETWEEN 2 AND 20
            ), day_items AS (
              SELECT DISTINCT t.customer_id,t.t_dat,CAST(t.article_id AS VARCHAR) AS article_id
              FROM read_parquet('{_sql_path(transactions)}') t SEMI JOIN valid_days d USING(customer_id,t_dat)
              WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t.t_dat<DATE '{cutoff}'
            ), observed AS (
              SELECT l.article_id AS left_article_id,r.article_id AS right_article_id,l.t_dat
              FROM day_items l JOIN day_items r
                ON l.customer_id=r.customer_id AND l.t_dat=r.t_dat AND l.article_id<r.article_id
              SEMI JOIN p35_relation_pairs p
                ON l.article_id=p.left_article_id AND r.article_id=p.right_article_id
            )
            SELECT left_article_id,right_article_id,
                   count(DISTINCT customer_id||':'||CAST(t_dat AS VARCHAR)) FILTER(
                     WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::INTEGER AS recent_28d_support,
                   count(DISTINCT customer_id||':'||CAST(t_dat AS VARCHAR)) FILTER(
                     WHERE t_dat<DATE '{cutoff}'-INTERVAL 28 DAY)::INTEGER AS older_29_84d_support,
                   date_diff('day',max(t_dat),DATE '{cutoff}')::INTEGER AS last_observed_age_days
            FROM (
              SELECT d.customer_id,d.t_dat,d.article_id AS left_article_id,r.article_id AS right_article_id
              FROM day_items d JOIN day_items r
                ON d.customer_id=r.customer_id AND d.t_dat=r.t_dat AND d.article_id<r.article_id
              SEMI JOIN p35_relation_pairs p
                ON d.article_id=p.left_article_id AND r.article_id=p.right_article_id
            ) q
            GROUP BY left_article_id,right_article_id
            """
        ).fetch_df()
    finally:
        connection.unregister("p35_relation_pairs")
    pair_to_support = {
        (str(row.left_article_id), str(row.right_article_id)): (int(row.recent_28d_support), int(row.older_29_84d_support), int(row.last_observed_age_days))
        for row in support.itertuples(index=False)
    }
    recent, older, last = [], [], []
    for left, right in zip(canonical_left, canonical_right):
        values = pair_to_support.get((str(catalog_items[left]), str(catalog_items[right])))
        if values is None:
            recent.append(0)
            older.append(0)
            last.append(np.nan)
        else:
            recent.append(values[0])
            older.append(values[1])
            last.append(values[2])
    output = relations.copy()
    output["recent_28d_support"] = np.asarray(recent, dtype=np.int32)
    output["older_29_84d_support"] = np.asarray(older, dtype=np.int32)
    total = output["recent_28d_support"] + output["older_29_84d_support"]
    output["recent_share"] = np.where(total > 0, output["recent_28d_support"] / total, np.nan)
    output["last_observed_age_days"] = np.asarray(last, dtype=np.float64)
    return output


def _half_direct_graph(
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    cutoff: str,
    recent: bool,
    article_to_row: dict[str, int],
    catalog_size: int,
) -> pd.DataFrame:
    lower, upper = ((0, 42) if recent else (42, 84))
    frame = connection.execute(
        f"""
        WITH valid_days AS (
          SELECT customer_id,t_dat,count(DISTINCT article_id) AS n
          FROM read_parquet('{_sql_path(transactions)}')
          WHERE t_dat>=DATE '{cutoff}'-INTERVAL {upper} DAY
            AND t_dat<DATE '{cutoff}'-INTERVAL {lower} DAY
          GROUP BY customer_id,t_dat HAVING n BETWEEN 2 AND 20
        ), items AS (
          SELECT DISTINCT t.customer_id,t.t_dat,CAST(t.article_id AS VARCHAR) AS article_id
          FROM read_parquet('{_sql_path(transactions)}') t SEMI JOIN valid_days d USING(customer_id,t_dat)
        ), support AS (
          SELECT article_id,count(*) AS n FROM items GROUP BY article_id
        ), undirected AS (
          SELECT l.article_id AS left_id,r.article_id AS right_id,count(*) AS pair_n
          FROM items l JOIN items r ON l.customer_id=r.customer_id AND l.t_dat=r.t_dat AND l.article_id<r.article_id
          GROUP BY l.article_id,r.article_id HAVING pair_n>=2
        ), directed AS (
          SELECT left_id AS anchor,right_id AS neighbor,pair_n FROM undirected
          UNION ALL SELECT right_id,left_id,pair_n FROM undirected
        ), ranked AS (
          SELECT d.anchor,d.neighbor,
                 row_number() OVER(PARTITION BY d.anchor ORDER BY d.pair_n/sqrt(a.n*b.n) DESC,d.pair_n DESC,d.neighbor) AS rank
          FROM directed d JOIN support a ON d.anchor=a.article_id JOIN support b ON d.neighbor=b.article_id
        )
        SELECT anchor,neighbor,rank::INTEGER AS rank FROM ranked WHERE rank<=100
        """
    ).fetch_df()
    frame["anchor_row"] = frame["anchor"].astype(str).map(article_to_row)
    frame["neighbor_row"] = frame["neighbor"].astype(str).map(article_to_row)
    frame = frame.dropna(subset=["anchor_row", "neighbor_row"])
    frame["relation_id"] = frame["anchor_row"].astype(np.int64) * catalog_size + frame["neighbor_row"].astype(np.int64)
    return frame[["relation_id", "rank"]]


def _rank_correlation(left: pd.DataFrame, right: pd.DataFrame, k: int) -> dict[str, Any]:
    l = left[left["rank"] <= k].set_index("relation_id")["rank"]
    r = right[right["rank"] <= k].set_index("relation_id")["rank"]
    union = l.index.union(r.index)
    lv = l.reindex(union, fill_value=k + 1).to_numpy(dtype=np.float64)
    rv = r.reindex(union, fill_value=k + 1).to_numpy(dtype=np.float64)
    correlation = float(np.corrcoef(lv, rv)[0, 1]) if len(union) > 1 and np.std(lv) > 0 and np.std(rv) > 0 else None
    intersection = int(len(l.index.intersection(r.index)))
    return {
        "older_relations": int(len(l)),
        "recent_relations": int(len(r)),
        "shared_relations": intersection,
        "overlap_over_smaller_set": intersection / max(min(len(l), len(r)), 1),
        "union_missing_rank_correlation": correlation,
    }


def _teacher_reliability(
    *,
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    teacher_dir: Path,
    cutoff: str,
    catalog_items: np.ndarray,
    article_to_row: dict[str, int],
    catalog_types: np.ndarray,
    catalog_garments: np.ndarray,
    downstream_sign: str,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    relations, union_audit, lookups = _teacher_union(
        teacher_dir, len(catalog_items), catalog_types, catalog_garments
    )
    relations = _relation_support(
        connection,
        transactions=transactions,
        catalog_items=catalog_items,
        relations=relations,
        cutoff=cutoff,
    )
    direct_npz = np.load(teacher_dir / "neighbors_direct_covisit.npz")
    adjacency = {
        int(anchor): set(int(value) for value in row[row >= 0])
        for anchor, row in zip(direct_npz["anchors"], direct_npz["neighbors"])
    }
    deep_only = relations[relations["provenance"] == "deepwalk_only"]
    if len(deep_only) > 20000:
        sample = deep_only.iloc[np.linspace(0, len(deep_only) - 1, 20000, dtype=int)]
    else:
        sample = deep_only
    shared_neighbors = [
        len(adjacency.get(int(row.anchor_catalog_row), set()) & adjacency.get(int(row.neighbor_catalog_row), set()))
        for row in sample.itertuples(index=False)
    ]
    provenance: dict[str, Any] = {}
    for name in PROVENANCE_NAMES.values():
        group = relations[relations["provenance"] == name]
        provenance[name] = {
            "relation_count": int(len(group)),
            "anchor_count": int(group["anchor_catalog_row"].nunique()),
            "product_type_match_rate": float(group["product_type_match"].mean()) if len(group) else None,
            "garment_match_rate": float(group["garment_match"].mean()) if len(group) else None,
            "anchor_graph_degree": _summary(group["anchor_graph_degree"]),
            "recent_28d_support": _summary(group["recent_28d_support"]),
            "older_29_84d_support": _summary(group["older_29_84d_support"]),
            "recent_share": _summary(group["recent_share"]),
            "last_observed_age_days": _summary(group["last_observed_age_days"]),
            "teacher_conflict_rate": float(group["teacher_conflict"].mean()) if len(group) else None,
        }
    older = _half_direct_graph(connection, transactions, cutoff, False, article_to_row, len(catalog_items))
    recent = _half_direct_graph(connection, transactions, cutoff, True, article_to_row, len(catalog_items))
    temporal = {
        "definition": "direct co-visitation graph rebuilt independently on older 42d and recent 42d; no Item2Vec or DeepWalk retraining",
        "top20": _rank_correlation(older, recent, 20),
        "top100": _rank_correlation(older, recent, 100),
    }
    result = {
        "cutoff": cutoff,
        "teacher_cutoff": cutoff,
        "downstream_sign": downstream_sign,
        "provenance": provenance,
        "union_audit": union_audit,
        "deepwalk_only_shared_neighbor_proxy": {
            "definition": "number of shared direct-co-vis Top100 neighbors; deterministic evenly-spaced sample when more than 20,000 relations",
            "sample_relations": int(len(sample)),
            "population_relations": int(len(deep_only)),
            "shared_neighbor_count": _summary(shared_neighbors),
            "positive_shared_neighbor_rate": float(np.mean(np.asarray(shared_neighbors) > 0)) if shared_neighbors else None,
        },
        "older42_vs_recent42_direct_graph": temporal,
        "cutoff_safe": True,
    }
    return result, relations, lookups


def _seed_scores(
    candidates: dict[str, np.ndarray], histories: dict[str, np.ndarray], embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    users = candidates["user_index"].astype(np.int32)
    items = candidates["catalog_row"].astype(np.int32)
    history_rows = histories["catalog_row"].astype(np.int32)
    history_days = histories["days_since_purchase"].astype(np.float32)
    history_mask = histories["mask"].astype(bool)
    scores = np.full(len(users), -np.inf, dtype=np.float32)
    best_anchor = np.full(len(users), -1, dtype=np.int32)
    cursor = 0
    while cursor < len(users):
        end = cursor + 1
        user_index = int(users[cursor])
        while end < len(users) and users[end] == user_index:
            end += 1
        valid_rows = history_rows[user_index][history_mask[user_index]]
        valid_days = history_days[user_index][history_mask[user_index]]
        if len(valid_rows):
            candidate_matrix = np.asarray(embeddings[items[cursor:end]], dtype=np.float32)
            history_matrix = np.asarray(embeddings[valid_rows], dtype=np.float32)
            weighted = candidate_matrix @ history_matrix.T
            weighted *= np.power(0.5, valid_days / 28.0)[None, :]
            local_best = weighted.argmax(axis=1)
            scores[cursor:end] = weighted[np.arange(end - cursor), local_best]
            best_anchor[cursor:end] = valid_rows[local_best]
        cursor = end
    return scores, best_anchor


def _movement_group(
    mask: np.ndarray,
    labels: np.ndarray,
    base_rank: np.ndarray,
    new_rank: np.ndarray,
) -> dict[str, Any]:
    selected = mask & labels.astype(bool)
    deltas = base_rank[selected] - new_rank[selected]
    return {
        "truth_pairs_in_fixed_top200": int(len(deltas)),
        "truth_moved_up": int(np.count_nonzero(deltas > 0)),
        "truth_unchanged": int(np.count_nonzero(deltas == 0)),
        "truth_moved_down": int(np.count_nonzero(deltas < 0)),
        "median_rank_delta_positive_means_up": float(np.median(deltas)) if len(deltas) else None,
        "density20_positive_pair_contribution": int(np.count_nonzero(selected & (new_rank <= 20)) - np.count_nonzero(selected & (base_rank <= 20))),
    }


def _teacher_rank_movement(
    *,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    p33_embeddings: np.ndarray,
    counts_by_row: np.ndarray,
    catalog_types: np.ndarray,
    relation_frame: pd.DataFrame,
    catalog_size: int,
) -> dict[str, Any]:
    users = candidates["user_index"].astype(np.int32)
    items = candidates["catalog_row"].astype(np.int32)
    labels = candidates["target"].astype(np.uint8)
    base_rank = candidates["rank"].astype(np.int32)
    scores, best_anchor = _seed_scores(candidates, histories, p33_embeddings)
    p33_rank = _rank_by_score(users, items, scores)
    relation_ids = best_anchor.astype(np.int64) * catalog_size + items
    relation_lookup = relation_frame.set_index("relation_id")
    target_indices = np.flatnonzero(labels == 1)
    proxy = pd.DataFrame(
        {
            "row_index": target_indices,
            "relation_id": relation_ids[target_indices],
        }
    ).join(relation_lookup, on="relation_id")
    provenance_by_row = np.full(len(labels), "no_top20_teacher_relation", dtype=object)
    freshness_by_row = np.full(len(labels), "no_direct_observation", dtype=object)
    agreement_by_row = np.full(len(labels), "no_top20_teacher_relation", dtype=object)
    conflict_by_row = np.full(len(labels), "no_top20_teacher_relation", dtype=object)
    degree_by_row = np.full(len(labels), "no_anchor_degree", dtype=object)
    for row in proxy.itertuples(index=False):
        index = int(row.row_index)
        if not pd.isna(row.provenance):
            provenance_by_row[index] = str(row.provenance)
            mask = int(row.provenance_mask)
            agreement_by_row[index] = "multi_teacher" if mask in (3, 5, 6, 7) else "single_teacher"
            conflict_by_row[index] = "conflict" if bool(row.teacher_conflict) else "no_conflict"
            degree = int(row.anchor_graph_degree)
            degree_by_row[index] = "degree_0_20" if degree <= 20 else ("degree_21_80" if degree <= 80 else "degree_81_plus")
            if not pd.isna(row.last_observed_age_days):
                age = float(row.last_observed_age_days)
                freshness_by_row[index] = "age_0_7" if age <= 7 else ("age_8_28" if age <= 28 else "age_29_84")
    result: dict[str, Any] = {
        "fixed_candidate_universe": "exact P3.1 Top200 user-item identities",
        "all_truth": _movement_group(np.ones(len(labels), dtype=bool), labels, base_rank, p33_rank),
        "unobserved_rank_delta": _summary((base_rank - p33_rank)[labels == 0]),
        "groups": {},
    }
    item_counts = counts_by_row[items]
    group_specs: dict[str, tuple[np.ndarray, list[str]]] = {
        "cold_segment": (np.where(item_counts == 0, "strict_cold", "sparse_1_5"), ["strict_cold", "sparse_1_5"]),
        "warm_proxy_teacher_provenance": (provenance_by_row, [*PROVENANCE_NAMES.values(), "no_top20_teacher_relation"]),
        "teacher_agreement": (agreement_by_row, ["single_teacher", "multi_teacher", "no_top20_teacher_relation"]),
        "teacher_conflict": (conflict_by_row, ["conflict", "no_conflict", "no_top20_teacher_relation"]),
        "relation_freshness": (freshness_by_row, ["age_0_7", "age_8_28", "age_29_84", "no_direct_observation"]),
        "anchor_graph_degree": (degree_by_row, ["degree_0_20", "degree_21_80", "degree_81_plus", "no_anchor_degree"]),
    }
    for name, (values, bins) in group_specs.items():
        result["groups"][name] = {
            bin_name: _movement_group(values == bin_name, labels, base_rank, p33_rank) for bin_name in bins
        }
    type_values = catalog_types[items]
    result["groups"]["truth_product_type"] = {
        str(int(value)): _movement_group(type_values == value, labels, base_rank, p33_rank)
        for value in np.unique(type_values[labels == 1])
    }
    return result


def _lane_rows(connection: duckdb.DuckDBPyConnection, kind: str) -> pd.DataFrame:
    if kind == "warm":
        return connection.execute(
            """
            WITH marginal AS (
              SELECT s.customer_id,s.article_id,
                     s.personalized_present::INTEGER AS personalized,
                     s.current_global_present::INTEGER AS global_current,
                     s.prior_direct_present::INTEGER AS prior_direct
              FROM source s
              JOIN eval_truth t USING(customer_id,article_id)
              ANTI JOIN base_300 b USING(customer_id,article_id)
            )
            SELECT * FROM marginal
            """
        ).fetch_df()
    return connection.execute(
        """
        SELECT s.customer_id,s.article_id,
               CASE WHEN s.personalized_rank IS NOT NULL THEN 1 ELSE 0 END AS personalized,
               CASE WHEN s.global_rank IS NOT NULL THEN 1 ELSE 0 END AS global_current,
               0 AS prior_direct
        FROM source_soft_deep_split_100 s
        JOIN eval_truth t USING(customer_id,article_id)
        ANTI JOIN base_300 b USING(customer_id,article_id)
        WHERE t.item_temperature='cold'
        """
    ).fetch_df()


def _exclusive_lane_summary(frame: pd.DataFrame) -> dict[str, Any]:
    combos = {
        "personalized_only": (1, 0, 0),
        "global_only": (0, 1, 0),
        "prior_only": (0, 0, 1),
        "personalized_and_global": (1, 1, 0),
        "personalized_and_prior": (1, 0, 1),
        "global_and_prior": (0, 1, 1),
        "all_three": (1, 1, 1),
    }
    output: dict[str, Any] = {}
    for name, combo in combos.items():
        selected = frame[
            (frame["personalized"] == combo[0])
            & (frame["global_current"] == combo[1])
            & (frame["prior_direct"] == combo[2])
        ]
        output[name] = {
            "marginal_truth_pairs": int(len(selected)),
            "truth_users": int(selected["customer_id"].nunique()),
            "unique_truth_items": int(selected["article_id"].nunique()),
            "share_of_marginal_union": float(len(selected) / max(len(frame), 1)),
        }
    memberships = int(frame["personalized"].sum() + frame["global_current"].sum() + frame["prior_direct"].sum())
    return {
        "unit": "distinct (cutoff, customer_id, truth_article_id)",
        "marginal_truth_pairs": int(len(frame)),
        "truth_users": int(frame["customer_id"].nunique()),
        "unique_truth_items": int(frame["article_id"].nunique()),
        "exclusive_combinations": output,
        "route_membership": {
            "personalized": int(frame["personalized"].sum()),
            "global": int(frame["global_current"].sum()),
            "prior": int(frame["prior_direct"].sum()),
            "membership_denominator": memberships,
            "personalized_share": float(frame["personalized"].sum() / max(memberships, 1)),
            "global_share": float(frame["global_current"].sum() / max(memberships, 1)),
        },
        "conservation_passed": sum(value["marginal_truth_pairs"] for value in output.values()) == len(frame),
    }


def _source_crosscheck(config: Config) -> dict[str, Any]:
    m34 = _json(config.repo_root / "reports" / "m3_4" / "m3-4-v3-season-aware-retrieval" / "metrics.json")
    m37 = _json(config.repo_root / "reports" / "m3_7" / "m3-7-v3-deep-visual-soft-season-cpu-prefix" / "metrics.json")
    windows: dict[str, Any] = {}
    differing = 0
    for window, _short, _cutoff in WINDOWS:
        warm_db = Path(m34["development"][window]["artifacts"]["evaluation_db"]["path"])
        cold_db = Path(m37["development"][window]["artifacts"]["evaluation_db"]["path"])
        with duckdb.connect(str(warm_db), read_only=True) as con:
            warm = _exclusive_lane_summary(_lane_rows(con, "warm"))
        with duckdb.connect(str(cold_db), read_only=True) as con:
            cold = _exclusive_lane_summary(_lane_rows(con, "cold"))
        warm_dominant = "personalized" if warm["route_membership"]["personalized_share"] > warm["route_membership"]["global_share"] else "global"
        cold_dominant = "personalized" if cold["route_membership"]["personalized_share"] > cold["route_membership"]["global_share"] else "global"
        differing += int(warm_dominant != cold_dominant)
        windows[window] = {
            "warm_seasonal_retrieval": warm,
            "cold_visual_retrieval": cold,
            "warm_dominant_route": warm_dominant,
            "cold_dominant_route": cold_dominant,
            "same_balance_direction": warm_dominant == cold_dominant,
        }
    return {
        "windows": windows,
        "windows_with_different_warm_cold_route_direction": differing,
        "season_router_evidence": "unsupported" if differing >= 2 else "inconclusive",
    }


def _expanded_sign_matrix() -> list[dict[str, Any]]:
    return [
        {
            "task": "warm seasonal retrieval",
            "source": "M3.4 expanded-350",
            "winter": {"global": "+", "personalized": "+", "content": "n/a", "graph_collaborative": "n/a", "recent_history": "+"},
            "spring": {"global": "+", "personalized": "+", "content": "n/a", "graph_collaborative": "n/a", "recent_history": "+"},
            "early_summer": {"global": "+", "personalized": "+", "content": "n/a", "graph_collaborative": "n/a", "recent_history": "+"},
            "late_summer": {"global": "+", "personalized": "+", "content": "n/a", "graph_collaborative": "n/a", "recent_history": "+"},
        },
        {
            "task": "cold visual retrieval",
            "source": "M3.7 soft Top500 minus soft Top100 under fixed 50+50",
            "winter": {"global": "mixed", "personalized": "mixed", "content": "-", "graph_collaborative": "n/a", "recent_history": "mixed"},
            "spring": {"global": "+", "personalized": "mixed", "content": "+", "graph_collaborative": "n/a", "recent_history": "mixed"},
            "early_summer": {"global": "mixed", "personalized": "+", "content": "+", "graph_collaborative": "n/a", "recent_history": "+"},
            "late_summer": {"global": "mixed", "personalized": "+", "content": "-", "graph_collaborative": "n/a", "recent_history": "+"},
        },
        {
            "task": "single-teacher cold retrieval",
            "source": "M4.3 multimodal Student versus raw FashionCLIP",
            "winter": {"global": "n/a", "personalized": "+", "content": "+", "graph_collaborative": "+", "recent_history": "+"},
            "spring": {"global": "n/a", "personalized": "+", "content": "+", "graph_collaborative": "+", "recent_history": "+"},
            "early_summer": {"global": "n/a", "personalized": "+", "content": "+", "graph_collaborative": "+", "recent_history": "+"},
            "late_summer": {"global": "n/a", "personalized": "+", "content": "+", "graph_collaborative": "+", "recent_history": "+"},
        },
        {
            "task": "candidate-aware cold reranking",
            "source": "P3.2 candidate-aware minus P3.1 seed ordering",
            "winter": {"global": "n/a", "personalized": "-", "content": "same", "graph_collaborative": "same", "recent_history": "-"},
            "spring": {"global": "n/a", "personalized": "-", "content": "same", "graph_collaborative": "same", "recent_history": "-"},
            "early_summer": {"global": "n/a", "personalized": "+", "content": "same", "graph_collaborative": "same", "recent_history": "+"},
            "late_summer": {"global": "n/a", "personalized": "-", "content": "same", "graph_collaborative": "same", "recent_history": "-"},
        },
        {
            "task": "multi-view cold representation",
            "source": "P3.3 multi-view Student minus M4 single-teacher Student",
            "winter": {"global": "n/a", "personalized": "+", "content": "+", "graph_collaborative": "+", "recent_history": "+"},
            "spring": {"global": "n/a", "personalized": "-", "content": "-", "graph_collaborative": "-", "recent_history": "-"},
            "early_summer": {"global": "n/a", "personalized": "-", "content": "-", "graph_collaborative": "-", "recent_history": "-"},
            "late_summer": {"global": "n/a", "personalized": "+", "content": "+", "graph_collaborative": "+", "recent_history": "+"},
        },
    ]


def _derive_verdicts(
    user_stability: dict[str, Any],
    stratification: dict[str, Any],
    fixed_query: dict[str, Any],
    half_life: dict[str, Any],
    teacher_movement: dict[str, Any],
    source_crosscheck: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    evidence: dict[str, Any] = {}
    early = stratification["windows"]["early_summer_20200624"]["variables"]
    drift = early["user_recent_vs_past_product_type_jsd"]["bins"]
    stability = early["embedding_centroid_stability_cosine"]["bins"]
    support = early["same_product_type_history_support"]["bins"]
    lower_drift_better = drift["Q1"]["delta"]["mrr"] > drift["Q4"]["delta"]["mrr"]
    higher_stability_better = stability["Q4"]["delta"]["mrr"] > stability["Q1"]["delta"]["mrr"]
    support_positive_better = min(support["1"]["delta"]["mrr"], support["2_plus"]["delta"]["mrr"]) > support["0"]["delta"]["mrr"]
    early_diversity = user_stability["windows"]["early_summer_20200624"]["recent_28d_within_product_type_catalog_diversity"]["truth_product_type_weighted"]["active_sku_count"]
    other_diversity = [
        value["recent_28d_within_product_type_catalog_diversity"]["truth_product_type_weighted"]["active_sku_count"]
        for name, value in user_stability["windows"].items() if name != "early_summer_20200624"
    ]
    diversity_higher = early_diversity is not None and early_diversity > float(np.median(other_diversity))
    repeated = 0
    for window, data in stratification["windows"].items():
        if window == "early_summer_20200624":
            continue
        variables = data["variables"]
        repeated += int(variables["user_recent_vs_past_product_type_jsd"]["bins"]["Q1"]["delta"]["mrr"] > variables["user_recent_vs_past_product_type_jsd"]["bins"]["Q4"]["delta"]["mrr"])
        repeated += int(variables["embedding_centroid_stability_cosine"]["bins"]["Q4"]["delta"]["mrr"] > variables["embedding_centroid_stability_cosine"]["bins"]["Q1"]["delta"]["mrr"])
    h1_checks = {
        "lower_drift_better_in_early_summer": lower_drift_better,
        "higher_stability_better_in_early_summer": higher_stability_better,
        "same_type_support_positive_better": support_positive_better,
        "early_summer_truth_type_active_sku_above_other_window_median": diversity_higher,
        "matching_relationship_occurrences_in_other_windows": repeated,
    }
    if all((lower_drift_better, higher_stability_better, support_positive_better, diversity_higher, repeated >= 2)):
        h1 = "supported"
    elif not lower_drift_better and not higher_stability_better and not support_positive_better:
        h1 = "rejected"
    else:
        h1 = "inconclusive"
    evidence["H1"] = h1_checks

    stable_nondegrade, overall_joint_improve, monotonic = 0, 0, 0
    fixed_window_checks: dict[str, Any] = {}
    for window, data in fixed_query["windows"].items():
        stable = data["stability_strata"]["Q4"]["orderings"]
        seed_stable, proxy_stable = stable["p31_seed_based"], stable["fixed_recency_centroid"]
        nondegrade = proxy_stable["density@20"] >= seed_stable["density@20"] and proxy_stable["mrr"] >= seed_stable["mrr"]
        stable_nondegrade += int(nondegrade)
        seed = data["orderings"]["p31_seed_based"]
        proxy = data["orderings"]["fixed_recency_centroid"]
        joint = proxy["at_k"]["20"]["positive_density"] > seed["at_k"]["20"]["positive_density"] and proxy["ranking"]["mrr"] > seed["ranking"]["mrr"]
        overall_joint_improve += int(joint)
        deltas = [
            data["stability_strata"][f"Q{q}"]["orderings"]["fixed_recency_centroid"]["mrr"]
            - data["stability_strata"][f"Q{q}"]["orderings"]["p31_seed_based"]["mrr"]
            for q in range(1, 5)
        ]
        near_monotonic = sum(deltas[index + 1] >= deltas[index] for index in range(3)) >= 2
        monotonic += int(near_monotonic)
        fixed_window_checks[window] = {"stable_Q4_nondegrade": nondegrade, "overall_density20_and_mrr_improve": joint, "stability_gain_near_monotonic": near_monotonic, "mrr_delta_Q1_to_Q4": deltas}
    fixed_verdict = "supported" if stable_nondegrade >= 3 and overall_joint_improve >= 2 and monotonic >= 3 else ("rejected" if stable_nondegrade <= 1 and overall_joint_improve == 0 else "inconclusive")
    evidence["H2"] = {"stable_nondegrading_windows": stable_nondegrade, "joint_improving_windows": overall_joint_improve, "near_monotonic_windows": monotonic, "windows": fixed_window_checks}

    best_buckets = []
    for window, data in half_life["windows"].items():
        aucs = {bucket: values["best_cosine"]["auc"] for bucket, values in data["age_buckets"].items()}
        valid = {key: value for key, value in aucs.items() if value is not None}
        best_buckets.append(max(valid, key=valid.get) if valid else "none")
    counts = Counter(best_buckets)
    repeat_count = max(counts.values()) if counts else 0
    time_verdict = "supported" if repeat_count >= 3 else "inconclusive"
    evidence["H4"] = {"best_age_bucket_by_window": dict(zip(half_life["windows"], best_buckets)), "maximum_repeat_count": repeat_count}

    candidate_properties = ["warm_proxy_teacher_provenance", "teacher_agreement", "teacher_conflict", "relation_freshness", "anchor_graph_degree"]
    explanatory_bins = []
    for prop in candidate_properties:
        bins = set.intersection(*(set(value["groups"][prop]) for value in teacher_movement["windows"].values()))
        for bin_name in bins:
            positive_rows = [teacher_movement["windows"][window]["groups"][prop][bin_name] for window in ("winter_20200122", "late_summer_20200819")]
            negative_rows = [teacher_movement["windows"][window]["groups"][prop][bin_name] for window in ("spring_20200318", "early_summer_20200624")]
            positive_pairs = sum(row["truth_pairs_in_fixed_top200"] for row in positive_rows)
            negative_pairs = sum(row["truth_pairs_in_fixed_top200"] for row in negative_rows)
            positive_contribution = sum(row["density20_positive_pair_contribution"] for row in positive_rows)
            negative_contribution = sum(row["density20_positive_pair_contribution"] for row in negative_rows)
            if positive_pairs >= 5 and negative_pairs >= 5 and positive_contribution > 0 and negative_contribution <= 0:
                explanatory_bins.append({"property": prop, "bin": bin_name, "positive_contribution": positive_contribution, "negative_contribution": negative_contribution})
    teacher_verdict = "supported" if explanatory_bins else "inconclusive"
    evidence["H3"] = {"explanatory_bins": explanatory_bins, "rule": "positive windows contribute >0 and negative windows <=0 with at least five truth pairs on each side"}
    verdicts = {
        "early_summer_stable_taste_hypothesis": h1,
        "fixed_user_query_premise": fixed_verdict,
        "teacher_selection_or_time_weighting": teacher_verdict,
        "time_aware_behavior_modeling": time_verdict,
        "season_router": source_crosscheck["season_router_evidence"],
    }
    return verdicts, evidence


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return lines


def render_report(result: dict[str, Any]) -> str:
    verdicts = result["verdicts"]
    lines = [
        "# P3.5：细粒度时间、用户状态与教师可靠性审计", "", "## 结论", "",
        f"- early-summer 稳定偏好/类内多样性假设：`{verdicts['early_summer_stable_taste_hypothesis']}`。",
        f"- 固定用户查询前提：`{verdicts['fixed_user_query_premise']}`。",
        f"- 教师选择或时间加权前提：`{verdicts['teacher_selection_or_time_weighting']}`。",
        f"- 时间感知行为建模前提：`{verdicts['time_aware_behavior_modeling']}`。",
        f"- 日历季节路由：`{verdicts['season_router']}`。", "",
        "本阶段只做截止日安全的离线诊断；没有训练或重训模型，没有生成新候选，没有读取 2020-09-16 最终验证周。局部相关关系不是因果结论，也不授权按月份切换模型。", "",
        "## 术语与统计口径", "",
        "- `fixed query`（固定用户查询，行业常见概念）：每名用户只有一个与候选无关的向量；本阶段用历史商品 Student 向量的均值或28天半衰期加权均值做代理。它只重排原 P3.1 Top200。",
        "- `profile stability`（用户表示稳定度，本项目诊断量）：最近0–28天与更早29–84天两个归一化历史中心向量的余弦；越高表示两段内容偏好方向越接近。分母只含两段都有可映射历史的评测用户。",
        "- `JSD`（Jensen–Shannon divergence，行业通用分布距离）：以2为底、范围0–1；越高表示两段商品类型购买分布越不同。",
        "- `SKU`（Stock Keeping Unit，行业常用商品粒度）：本项目以 article_id 为可推荐商品；活跃 SKU 数是最近28天某商品类型内有交易的不同 article_id 数。",
        "- `density@20`（前20正例密度，本项目核心指标）：分桶内前20候选中的未来购买用户—商品对数除以该桶前20候选行数。",
        "- `Top200→Top20`（正例集中率，本项目自定义指标）：固定 Top200 内正例被排进前20的数量除以 Top200 内正例数。",
        "- `AUC`（行业通用二分类排序统计）：随机抽一条正例和一条未观察候选时，前者分数更高的概率；这里只作可分性诊断。",
        "- `teacher provenance`（教师来源类别，本项目自定义）：一条 Top20 商品关系来自 Item2Vec、直接共购、DeepWalk 中哪几个；七类互斥且并集等于三路 Top20 关系并集。",
        "- `teacher conflict`（教师冲突，本项目自定义）：一条关系进入至少一个教师 Top20，但没有进入另一个教师 Top100。",
        "- `recent support / older support`（近期/较旧直接支持，本项目自定义）：同一用户同一自然日共购该商品对的不同用户日数，分别在截止日前0–28天与29–84天统计。无直接共购的 DeepWalk 间接关系记0，最后出现时间缺失。",
        "- `rank delta`（名次变化，本项目自定义）：P3.1 单教师名次减去同候选集合上的 P3.3 多教师重算名次；正数表示多教师使候选上升。",
        "- `unobserved`（未观察候选，本项目标签）：验证周没有购买记录的 Top200 候选，不代表曝光后拒绝。", "",
        "## 按任务拆分的跨窗信号方向", "",
        "下表的 `+/-/mixed/same/n/a` 分别表示相对对照的正向、负向、混合、输入未改变、不适用。每一格只属于该行任务，不能跨任务推导整窗路由。", "",
    ]
    rows = []
    for task in result["expanded_task_specific_sign_matrix"]:
        for key, display in (("winter", "winter"), ("spring", "spring"), ("early_summer", "early-summer"), ("late_summer", "late-summer")):
            state = task[key]
            rows.append([task["task"], display, state["global"], state["personalized"], state["content"], state["graph_collaborative"], state["recent_history"]])
    lines.extend(_table(["task", "window", "global", "personalized", "content", "graph/collaborative", "recent history"], rows))

    lines.extend(["", "## 用户稳定度、历史支持与类内商品多样性", ""])
    rows = []
    for window, data in result["user_stability"]["windows"].items():
        diversity = data["recent_28d_within_product_type_catalog_diversity"]["truth_product_type_weighted"]
        rows.append([window, data["embedding_centroid_stability_cosine"]["count"], data["embedding_centroid_stability_cosine"]["median"], data["recent_vs_past_product_type_jsd"]["median"], data["recent_vs_global_product_type_jsd"]["median"], diversity["active_sku_count"], diversity["normalized_entropy"]])
    lines.extend(_table(["window", "稳定度用户数", "稳定度中位数", "自身JSD中位数", "用户-全局JSD中位数", "truth类型加权活跃SKU", "truth类型加权归一化熵"], rows))
    lines.extend(["", "support 表以完整 cold/sparse 验证正例用户—商品对为分母；同类型/同服装组计数读取截止日前84天不同历史商品。", ""])
    rows = []
    for window, data in result["user_stability"]["windows"].items():
        support, dispersion = data["cold_sparse_truth_history_support"], data["within_type_embedding_dispersion"]
        rows.append([window, support["truth_pairs"], support["same_product_type_count"]["mean"], support["same_garment_count"]["mean"], support["days_since_last_same_product_type"]["median"], support["days_since_last_same_garment"]["median"], dispersion["all_evaluation_users"]["median"], dispersion["cold_sparse_truth_users"]["median"]])
    lines.extend(_table(["window", "truth pairs", "同类型历史均值", "同服装组历史均值", "最近同类型天数中位数", "最近同服装组天数中位数", "全用户类内距离中位数", "truth用户类内距离中位数"], rows))

    lines.extend(["", "## P3.2 用户状态分层（所有预注册桶）", ""])
    for variable in next(iter(result["p32_gain_stratification"]["windows"].values()))["variables"]:
        lines.extend([f"### {variable}", ""])
        rows = []
        for window, data in result["p32_gain_stratification"]["windows"].items():
            for bin_name, values in data["variables"][variable]["bins"].items():
                rows.append([window, bin_name, values["users"], values["truth_pairs"], values["coarse"]["density@20"], values["candidate_aware"]["density@20"], values["delta"]["density@20"], values["coarse"]["mrr"], values["candidate_aware"]["mrr"], values["delta"]["mrr"], values["coarse"]["top200_to_top20"], values["candidate_aware"]["top200_to_top20"], values["delta"]["top200_to_top20"]])
        lines.extend(_table(["window", "bin", "users", "truth pairs", "coarse d20", "aware d20", "Δd20", "coarse MRR", "aware MRR", "ΔMRR", "coarse 200→20", "aware 200→20", "Δconversion"], rows))
        lines.append("")

    lines.extend(["## 固定用户查询只读探针", ""])
    rows = []
    for window, data in result["fixed_query_probe"]["windows"].items():
        for name, metrics in data["orderings"].items():
            rows.append([window, name, metrics["at_k"]["5"]["positive_density"], metrics["at_k"]["10"]["positive_density"], metrics["at_k"]["20"]["positive_density"], metrics["at_k"]["50"]["positive_density"], metrics["ranking"]["mrr"], metrics["ranking"]["top200_to_top20"], metrics["at_k"]["20"]["segments"]["strict_cold"]["recall"], metrics["at_k"]["20"]["segments"]["sparse_1_5"]["recall"], metrics["score_auc"], metrics["same_user_truth_percentile"]["median"]])
    lines.extend(_table(["window", "ordering", "d@5", "d@10", "d@20", "d@50", "MRR", "200→20", "strict R@20", "sparse R@20", "AUC", "truth百分位中位数"], rows))
    lines.extend(["", "Q1 为用户表示稳定度最低、Q4 为最高，`ineligible` 表示最近与较旧两段不能同时形成中心向量。", ""])
    rows = []
    for window, data in result["fixed_query_probe"]["windows"].items():
        for bin_name, stratum in data["stability_strata"].items():
            for ordering, metrics in stratum["orderings"].items():
                rows.append([window, bin_name, ordering, stratum["users"], stratum["positive_pairs_in_top200"], metrics["density@20"], metrics["mrr"], metrics["top200_to_top20"], metrics["score_auc"]])
    lines.extend(_table(["window", "稳定度桶", "ordering", "truth users", "Top200正例", "d@20", "MRR", "200→20", "AUC"], rows))

    lines.extend(["", "## 历史信号年龄", ""])
    rows = []
    for window, data in result["history_half_life"]["windows"].items():
        for bucket, metrics in data["age_buckets"].items():
            best = metrics["best_cosine"]
            rows.append([window, bucket, best["auc"], best["median_gap"], metrics["mean_top3_cosine"]["auc"], metrics["same_product_type_support"]["auc"], metrics["fixed_centroid_similarity"]["auc"]])
    lines.extend(_table(["window", "history age", "best cosine AUC", "median gap", "top3 mean AUC", "same-type AUC", "centroid AUC"], rows))
    lines.extend(["", "每个 Top200 正例在四个年龄桶中取最佳余弦最大者；分母是该窗 Top200 内正例，不是全部未来正例。", ""])
    rows = []
    for window, data in result["history_half_life"]["windows"].items():
        distribution = data["most_informative_truth_history_age"]
        for bucket in (*AGE_BUCKETS, "no_history"):
            count = distribution["counts"].get(bucket, 0)
            rows.append([window, bucket, count, count / max(distribution["denominator_top200_truth_pairs"], 1), distribution["denominator_top200_truth_pairs"]])
    lines.extend(_table(["window", "最有信息年龄桶", "truth pairs", "share", "Top200 truth分母"], rows))

    lines.extend(["", "## 教师关系可靠性", ""])
    rows = []
    for window, data in result["teacher_reliability"]["windows"].items():
        for provenance, values in data["provenance"].items():
            rows.append([window, data["representative_training_cutoff"], data["downstream_sign"], provenance, values["relation_count"], values["anchor_count"], values["product_type_match_rate"], values["recent_28d_support"]["mean"], values["older_29_84d_support"]["mean"], values["recent_share"]["median"], values["teacher_conflict_rate"]])
    lines.extend(_table(["outer window", "代表训练截点", "P3.3方向", "互斥来源", "relations", "anchors", "同类型率", "近期支持均值", "较旧支持均值", "近期占比中位数", "冲突率"], rows))
    lines.extend(["", "主表使用训练该 outer Student 的两个冻结教师截点中较新的一个；两个训练截点的完整七类统计都在机器 JSON。42/42 天只重建直接共购图，不重训 Item2Vec 或 DeepWalk。", ""])
    rows = []
    for window, data in result["teacher_reliability"]["windows"].items():
        for teacher_cutoff, snapshot in data["per_training_cutoff"].items():
            for depth in ("top20", "top100"):
                values = snapshot["older42_vs_recent42_direct_graph"][depth]
                rows.append([window, teacher_cutoff, depth, values["older_relations"], values["recent_relations"], values["shared_relations"], values["overlap_over_smaller_set"], values["union_missing_rank_correlation"]])
    lines.extend(_table(["outer window", "teacher cutoff", "depth", "older relations", "recent relations", "shared", "overlap", "rank correlation"], rows))

    lines.extend(["", "## 单教师到多教师的固定候选名次变化", ""])
    rows = []
    for window, data in result["teacher_rank_movement"]["windows"].items():
        overall = data["all_truth"]
        rows.append([window, P33_SIGNS[window], overall["truth_pairs_in_fixed_top200"], overall["truth_moved_up"], overall["truth_unchanged"], overall["truth_moved_down"], overall["median_rank_delta_positive_means_up"], overall["density20_positive_pair_contribution"]])
    lines.extend(_table(["window", "正式方向", "truth pairs", "up", "same", "down", "median Δrank", "Top20净正例"], rows))
    lines.extend(["", "以下完整列出教师来源、同意/冲突、新鲜度与图度数分组；Top20净正例只属于固定 P3.1 Top200 的表示探针，不是 P3.3 正式端到端指标。", ""])
    rows = []
    for window, data in result["teacher_rank_movement"]["windows"].items():
        for group_name, bins in data["groups"].items():
            if group_name == "truth_product_type":
                continue
            for bin_name, values in bins.items():
                rows.append([window, group_name, bin_name, values["truth_pairs_in_fixed_top200"], values["truth_moved_up"], values["truth_unchanged"], values["truth_moved_down"], values["median_rank_delta_positive_means_up"], values["density20_positive_pair_contribution"]])
    lines.extend(_table(["window", "group", "bin", "truth pairs", "up", "same", "down", "median Δrank", "Top20净正例"], rows))

    lines.extend(["", "## Warm 与 cold 召回来源交叉检查", ""])
    rows = []
    for window, data in result["source_crosscheck"]["windows"].items():
        warm, cold = data["warm_seasonal_retrieval"], data["cold_visual_retrieval"]
        rows.append([window, warm["marginal_truth_pairs"], warm["route_membership"]["personalized_share"], warm["route_membership"]["global_share"], cold["marginal_truth_pairs"], cold["route_membership"]["personalized_share"], cold["route_membership"]["global_share"], data["same_balance_direction"]])
    lines.extend(_table(["window", "warm marginal", "warm personal share", "warm global share", "cold marginal", "cold personal share", "cold global share", "方向一致"], rows))
    lines.extend(["", "M3.4 warm 有个性化当前、当前全局、去年同期三路；M3.7 cold 只有个性化视觉与全局视觉两路。下表组合互斥，份额分母为该窗相对 base 新增命中的 truth pairs。", ""])
    rows = []
    for window, data in result["source_crosscheck"]["windows"].items():
        for task_name, task in (("warm", data["warm_seasonal_retrieval"]), ("cold", data["cold_visual_retrieval"])):
            for combo, values in task["exclusive_combinations"].items():
                rows.append([window, task_name, combo, values["marginal_truth_pairs"], values["truth_users"], values["unique_truth_items"], values["share_of_marginal_union"]])
    lines.extend(_table(["window", "task", "互斥组合", "marginal pairs", "truth users", "unique items", "overlap/share"], rows))
    lines.extend(["", "同一日历窗口中，warm 与 cold 的来源占比不是同一个任务。方向不一致时，不能用 `if month == June` 一类规则决定全系统走个性化还是全局路径。", "",
        "## 审计与解释边界", "",
        f"- 全部要求的历史窗口安全、候选身份、互斥归因和分桶输出检查：`{result['audit_checks']['all_passed']}`。",
        f"- 运行耗时 `{result['runtime']['wall_seconds']:.2f}` 秒；DuckDB 内存上限 `{result['runtime']['duckdb_memory_limit']}`；GPU 未使用。",
        "- 教师选择/时间加权只有在可提前计算的关系属性跨正负窗口重复时才支持；固定查询也必须通过稳定用户层与总体联合门槛。",
        "- H&M 数据没有曝光、库存、上架时间与订单边界；未观察候选不能解释为负反馈，日期级共购也不是真实订单篮。", "",
        "## 下一步边界", "",
        "下一步必须另行预注册；本阶段没有启动模型训练。固定查询被拒绝时不能把均值中心探针包装成 Two-Tower 成功；时间信号被支持也只允许研究连续衰减或状态输入，不允许日历季节专家。", "",
        "机器可读证据：`P3_5_FINE_GRAINED_AUDIT.json`、七个分项 JSON 与 `P3_5_OUTPUT_MANIFEST.json`。最终周保持 `not_run`。",
    ])
    return "\n".join(lines) + "\n"


def run_audit(config: Config) -> dict[str, Any]:
    started = time.perf_counter()
    contract_path = config.output_dir / "P3_5_EXPERIMENT_CONTRACT.json"
    if not contract_path.is_file():
        raise FileNotFoundError("P3.5 experiment contract must exist before formal computation")
    contract = _json(contract_path)
    if contract.get("status") != "preregistered_before_formal_computation":
        raise RuntimeError("P3.5 contract is not frozen")
    for _name, _short, cutoff in WINDOWS:
        _safe_cutoff(cutoff)

    required_reports = [
        config.repo_root / "reports" / "m3_3" / "M3_3_FINAL.md",
        config.repo_root / "reports" / "m3_4" / "M3_4_FINAL.md",
        config.repo_root / "reports" / "m3_6" / "M3_6_FINAL.md",
        config.repo_root / "reports" / "m3_7" / "M3_7_FINAL.md",
        config.repo_root / "reports" / "m4" / "M4_2_FINAL.md",
        config.repo_root / "reports" / "m4" / "M4_3_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_1_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_2_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_3_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_4_REGIME_AUDIT.md",
    ]
    historical_before = {str(path.relative_to(config.repo_root)): _file_sha(path) for path in required_reports}
    p31 = _json(config.output_dir / "P3_1_metrics.json")
    p32 = _json(config.output_dir / "P3_2_metrics.json")
    p33 = _json(config.output_dir / "P3_3_metrics.json")
    if any(value.get("final_week") != "not_run" for value in (p31, p32, p33)):
        raise RuntimeError("a frozen Phase 3 input violated final-week boundary")

    static_dir = config.repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation" / "student-v1" / "static_catalog"
    catalog = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str})
    catalog_items = catalog.sort_values("catalog_row")["article_id"].astype(str).to_numpy()
    article_to_row = {value: index for index, value in enumerate(catalog_items)}
    articles_path = config.source_root / "data" / "raw" / "articles.csv"
    transactions = config.source_root / "data" / "interim" / "audit" / "transactions.parquet"
    attributes = pd.read_csv(
        articles_path,
        dtype={"article_id": str},
        usecols=["article_id", "product_type_no", "garment_group_no"],
    )
    articles = catalog.merge(attributes, on="article_id", how="left", validate="one_to_one")
    articles.attrs["source_path"] = articles_path
    articles["product_type_no"] = articles["product_type_no"].fillna(-1).astype(int)
    articles["garment_group_no"] = articles["garment_group_no"].fillna(-1).astype(int)
    catalog_types = articles.sort_values("catalog_row")["product_type_no"].to_numpy(dtype=np.int32)
    catalog_garments = articles.sort_values("catalog_row")["garment_group_no"].to_numpy(dtype=np.int32)

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{config.memory_limit}'")
    connection.execute(f"SET threads={config.threads}")
    user_stability_windows: dict[str, Any] = {}
    stratification_windows: dict[str, Any] = {}
    fixed_windows: dict[str, Any] = {}
    half_windows: dict[str, Any] = {}
    teacher_windows: dict[str, Any] = {}
    movement_windows: dict[str, Any] = {}
    window_audits: dict[str, Any] = {}
    try:
        for window, short_name, cutoff in WINDOWS:
            window_started = time.perf_counter()
            print(f"P3.5 {window}: load frozen assets", flush=True)
            paths = _window_paths(config, window, cutoff, p31, p32, p33)
            required_paths = [
                paths["candidates"], paths["histories"], paths["users"], paths["m4_embedding"],
                paths["p32_scores"], paths["p33_embedding"], *paths["teacher_dirs"],
            ]
            missing = [str(path) for path in required_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"missing frozen P3.5 inputs for {window}: {missing}")
            users_frame = pd.read_csv(paths["users"], dtype={"customer_id": str})
            candidates_npz = np.load(paths["candidates"])
            candidates = {name: np.asarray(candidates_npz[name]) for name in candidates_npz.files}
            histories_npz = np.load(paths["histories"])
            histories = {name: np.asarray(histories_npz[name]) for name in histories_npz.files}
            p32_scores = np.load(paths["p32_scores"], mmap_mode="r")
            m4_embeddings = np.load(paths["m4_embedding"], mmap_mode="r")
            p33_embeddings = np.load(paths["p33_embedding"], mmap_mode="r")
            if len(p32_scores) != len(candidates["user_index"]):
                raise RuntimeError(f"P3.2 score identity length mismatch at {window}")
            if not np.array_equal(candidates["rank"], _rank_by_score(candidates["user_index"], candidates["catalog_row"], candidates["coarse_score"])):
                raise RuntimeError(f"P3.1 coarse rank reconstruction mismatch at {window}")

            history = _load_history_frame(connection, transactions, users_frame, cutoff)
            truth, count_frame = _load_truth_and_counts(connection, transactions, users_frame, cutoff)
            count_frame["catalog_row"] = count_frame["article_id"].astype(str).map(article_to_row)
            counts_by_row = np.zeros(len(catalog_items), dtype=np.int32)
            valid_counts = count_frame.dropna(subset=["catalog_row"])
            counts_by_row[valid_counts["catalog_row"].astype(int)] = valid_counts["events"].to_numpy(dtype=np.int32)
            print(f"P3.5 {window}: user-state and fixed-query diagnostics", flush=True)
            profile, user_states, truth_rows = _profiles_and_catalog_diversity(
                connection=connection,
                transactions=transactions,
                articles=articles,
                users=users_frame,
                history=history,
                truth=truth,
                counts=count_frame[["article_id", "events"]],
                embeddings=m4_embeddings,
                cutoff=cutoff,
            )
            truth_rows["catalog_row"] = truth_rows["article_id"].astype(str).map(article_to_row)
            truth_rows = truth_rows.dropna(subset=["catalog_row", "product_type_no", "garment_group_no"]).copy()
            truth_rows["catalog_row"] = truth_rows["catalog_row"].astype(int)
            fixed, half_life, ranks = _fixed_query_and_half_life(
                candidates=candidates,
                histories=histories,
                m4_embeddings=m4_embeddings,
                p32_scores=p32_scores,
                truth_rows=truth_rows,
                counts_by_row=counts_by_row,
                user_states=user_states,
                catalog_types=catalog_types,
            )
            stratification = _p32_stratification(
                candidates=candidates,
                ranks=ranks,
                truth_rows=truth_rows,
                user_states=user_states,
                catalog_types=catalog_types,
            )
            print(f"P3.5 {window}: teacher provenance, freshness, and fixed-universe movement", flush=True)
            teacher_snapshots: dict[str, Any] = {}
            relation_frame = None
            for teacher_cutoff, teacher_dir in zip(paths["teacher_cutoffs"], paths["teacher_dirs"]):
                snapshot, snapshot_relations, _lookups = _teacher_reliability(
                    connection=connection,
                    transactions=transactions,
                    teacher_dir=teacher_dir,
                    cutoff=teacher_cutoff,
                    catalog_items=catalog_items,
                    article_to_row=article_to_row,
                    catalog_types=catalog_types,
                    catalog_garments=catalog_garments,
                    downstream_sign=P33_SIGNS[window],
                )
                teacher_snapshots[teacher_cutoff] = snapshot
                relation_frame = snapshot_relations
            if relation_frame is None:
                raise RuntimeError(f"no P3.3 teacher snapshot for {window}")
            latest_teacher_cutoff = paths["teacher_cutoffs"][-1]
            teacher = {
                **teacher_snapshots[latest_teacher_cutoff],
                "outer_window": window,
                "outer_cutoff": cutoff,
                "representative_training_cutoff": latest_teacher_cutoff,
                "representation_rule": "the latest of the two frozen teacher cutoffs used to train this outer-window Student",
                "per_training_cutoff": teacher_snapshots,
            }
            movement = _teacher_rank_movement(
                candidates=candidates,
                histories=histories,
                p33_embeddings=p33_embeddings,
                counts_by_row=counts_by_row,
                catalog_types=catalog_types,
                relation_frame=relation_frame,
                catalog_size=len(catalog_items),
            )
            cold_sparse_truth_pairs = int(len(truth_rows))
            strict_pairs = int(np.count_nonzero(truth_rows["segment"] == "strict_cold"))
            sparse_pairs = int(np.count_nonzero(truth_rows["segment"] == "sparse_1_5"))
            window_audits[window] = {
                "cutoff": cutoff,
                "latest_history_days_min": int(history["days_since"].min()) if len(history) else None,
                "recent_past_disjoint": True,
                "centroids_cutoff_safe": bool((history["days_since"] > 0).all()),
                "fixed_query_candidate_identity_exact": fixed["candidate_identity"]["exact_p31_top200_reuse"],
                "strict_sparse_conservation": strict_pairs + sparse_pairs == cold_sparse_truth_pairs,
                "teacher_provenance_conservation": teacher["union_audit"]["provenance_union_conserved"],
                "teacher_support_cutoff_safe": teacher["cutoff_safe"],
                "runtime_seconds": time.perf_counter() - window_started,
            }
            user_stability_windows[window] = profile
            stratification_windows[window] = stratification
            fixed_windows[window] = fixed
            half_windows[window] = half_life
            teacher_windows[window] = teacher
            movement_windows[window] = movement
            del relation_frame, m4_embeddings, p33_embeddings
    finally:
        connection.close()

    source_crosscheck = _source_crosscheck(config)
    user_stability = {"stage": "P3.5", "windows": user_stability_windows, "final_week": "not_run"}
    stratification = {"stage": "P3.5", "windows": stratification_windows, "final_week": "not_run"}
    fixed_query = {"stage": "P3.5", "windows": fixed_windows, "final_week": "not_run"}
    half_life = {"stage": "P3.5", "windows": half_windows, "final_week": "not_run"}
    teacher_reliability = {"stage": "P3.5", "windows": teacher_windows, "final_week": "not_run"}
    teacher_movement = {"stage": "P3.5", "windows": movement_windows, "final_week": "not_run"}
    source_crosscheck = {"stage": "P3.5", **source_crosscheck, "final_week": "not_run"}
    verdicts, verdict_evidence = _derive_verdicts(
        user_stability, stratification, fixed_query, half_life, teacher_movement, source_crosscheck
    )
    historical_after = {str(path.relative_to(config.repo_root)): _file_sha(path) for path in required_reports}
    expected_bins = {
        "user_recent_vs_past_product_type_jsd": {"Q1", "Q2", "Q3", "Q4", "ineligible"},
        "user_vs_global_recent_product_type_jsd": {"Q1", "Q2", "Q3", "Q4", "ineligible"},
        "embedding_centroid_stability_cosine": {"Q1", "Q2", "Q3", "Q4", "ineligible"},
        "same_product_type_history_support": {"0", "1", "2_plus"},
        "days_since_last_same_product_type": {"0_7", "8_28", "29_84", "over_84_or_none"},
        "within_type_embedding_dispersion": {"Q1", "Q2", "Q3", "Q4", "ineligible"},
        "candidate_relative_within_type_embedding_dispersion": {"Q1", "Q2", "Q3", "Q4", "ineligible"},
    }
    all_bins = all(
        all(set(data["variables"][name]["bins"]) == bins for name, bins in expected_bins.items())
        for data in stratification_windows.values()
    )
    audit_checks = {
        "recent_past_windows_disjoint_and_before_cutoff": all(value["recent_past_disjoint"] and value["centroids_cutoff_safe"] for value in window_audits.values()),
        "fixed_query_candidate_identity_exact_p31": all(value["fixed_query_candidate_identity_exact"] for value in window_audits.values()),
        "lane_mutually_exclusive_attribution_conserved": all(value["warm_seasonal_retrieval"]["conservation_passed"] and value["cold_visual_retrieval"]["conservation_passed"] for value in source_crosscheck["windows"].values()),
        "teacher_provenance_seven_classes_union_conserved": all(value["teacher_provenance_conservation"] for value in window_audits.values()),
        "teacher_support_cutoff_safe": all(value["teacher_support_cutoff_safe"] for value in window_audits.values()),
        "strict_sparse_conserved": all(value["strict_sparse_conservation"] for value in window_audits.values()),
        "final_cutoff_rejected": True,
        "all_preregistered_bins_output": all_bins,
        "historical_reports_not_overwritten": historical_before == historical_after,
    }
    audit_checks["all_passed"] = all(audit_checks.values())
    if not audit_checks["all_passed"]:
        raise RuntimeError(f"P3.5 audit check failed: {audit_checks}")
    result = {
        "schema_version": "phase3-p3.5-fine-grained-audit-v1",
        "stage": "P3.5",
        "status": "measured",
        "run_id": "p3-5-v1-fine-grained-temporal-user-teacher-audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "read-only no-training diagnostic",
        "contract": contract,
        "expanded_task_specific_sign_matrix": _expanded_sign_matrix(),
        "user_stability": user_stability,
        "p32_gain_stratification": stratification,
        "fixed_query_probe": fixed_query,
        "history_half_life": half_life,
        "teacher_reliability": teacher_reliability,
        "teacher_rank_movement": teacher_movement,
        "source_crosscheck": source_crosscheck,
        "verdicts": verdicts,
        "verdict_evidence": verdict_evidence,
        "window_audits": window_audits,
        "audit_checks": audit_checks,
        "input_evidence": {
            "contract": file_identity(contract_path),
            "transactions": file_identity(transactions),
            "articles": file_identity(articles_path),
            "historical_report_sha256_before_after_equal": historical_before,
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "duckdb_memory_limit": config.memory_limit,
            "threads": config.threads,
            "gpu_used": False,
        },
        "final_week": "not_run",
    }
    return _json_ready(result)


def _write_outputs(config: Config, result: dict[str, Any]) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    components = {
        "p3_5_user_stability.json": result["user_stability"],
        "p3_5_p32_gain_stratification.json": result["p32_gain_stratification"],
        "p3_5_fixed_query_probe.json": result["fixed_query_probe"],
        "p3_5_history_half_life.json": result["history_half_life"],
        "p3_5_teacher_reliability.json": result["teacher_reliability"],
        "p3_5_teacher_rank_movement.json": result["teacher_rank_movement"],
        "p3_5_source_crosscheck.json": result["source_crosscheck"],
    }
    for name, value in components.items():
        atomic_json(config.output_dir / name, value)
    master_path = config.output_dir / "P3_5_FINE_GRAINED_AUDIT.json"
    report_path = config.output_dir / "P3_5_FINE_GRAINED_AUDIT.md"
    atomic_json(master_path, result)
    report_path.write_text(render_report(result), encoding="utf-8")
    evidence = {
        "schema_version": "phase3-p3.5-output-manifest-v1",
        "stage": "P3.5",
        "status": "completed",
        "artifacts": {
            name: file_identity(config.output_dir / name)
            for name in [*components, master_path.name, report_path.name, "P3_5_EXPERIMENT_CONTRACT.json"]
        },
        "final_week": "not_run",
    }
    atomic_json(config.output_dir / "P3_5_OUTPUT_MANIFEST.json", evidence)
    return evidence


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only P3.5 fine-grained audit")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Config(
        repo_root=args.repo_root.resolve(),
        source_root=args.source_root.resolve(),
        output_dir=args.output_dir.resolve(),
        memory_limit=args.memory_limit,
        threads=args.threads,
    )
    result = run_audit(config)
    manifest = _write_outputs(config, result)
    print(json.dumps({"status": result["status"], "verdicts": result["verdicts"], "runtime": result["runtime"], "manifest": manifest["artifacts"]}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
