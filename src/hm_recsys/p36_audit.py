from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from .m4_contract import atomic_json, file_identity
from .p35_audit import _rank_by_score, _seed_scores


WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("winter_20200122", "winter", "2020-01-22"),
    ("spring_20200318", "spring", "2020-03-18"),
    ("early_summer_20200624", "early-summer", "2020-06-24"),
    ("late_summer_20200819", "late-summer", "2020-08-19"),
)
FINAL_CUTOFF = "2020-09-16"
P31_RUN = "phase3-v1-candidate-aware-history"
P33_RUN = "phase3-p3.3-v1-multiview-graph-student"
K_PROXY = 5
HISTORY_N = 20
LABELS = (
    "similarity_like",
    "direct_complement_like",
    "multihop_complement_like",
    "behavioral_mixed",
    "content_only",
    "unclassified",
)
LABEL_TO_CODE = {name: index for index, name in enumerate(LABELS)}
AGE_BUCKETS = ("0_7", "8_28", "29_84", "over_84")
P32_SIGNS = {
    "winter_20200122": "negative",
    "spring_20200318": "negative",
    "early_summer_20200624": "positive",
    "late_summer_20200819": "negative",
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
    artifact_dir: Path
    memory_limit: str = "8GB"
    threads: int = 8
    device: str = "cuda"
    user_batch: int = 48


@dataclass
class TeacherAssets:
    cutoff: str
    i2v_vectors: np.ndarray
    i2v_valid: np.ndarray
    deep_vectors: np.ndarray
    deep_valid: np.ndarray
    direct_score: sparse.csr_matrix
    direct_recent: sparse.csr_matrix
    direct_older: sparse.csr_matrix
    direct_last_age: sparse.csr_matrix
    pair_support: pd.DataFrame
    category_pair: dict[tuple[int, int], tuple[float, float]]
    identities: dict[str, Any]
    cutoff_audit: dict[str, Any]


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
        raise RuntimeError(f"P3.6 refuses final or later cutoff: {cutoff}")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _summary(values: Iterable[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _age_codes(days: np.ndarray) -> np.ndarray:
    values = np.asarray(days, dtype=np.float32)
    output = np.full(values.shape, 3, dtype=np.uint8)
    output[values <= 84] = 2
    output[values <= 28] = 1
    output[values <= 7] = 0
    return output


def _quartile_codes(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    output = np.full(array.shape, 255, dtype=np.uint8)
    if not finite.any():
        return output, []
    cuts = np.quantile(array[finite].astype(np.float64), [0.25, 0.5, 0.75]).tolist()
    output[finite] = np.searchsorted(cuts, array[finite], side="right").astype(np.uint8)
    return output, [float(value) for value in cuts]


def empirical_mid_percentile(values: np.ndarray) -> np.ndarray:
    """Exact empirical mid-rank percentile; missing values remain NaN."""
    array = np.asarray(values, dtype=np.float32)
    output = np.full(array.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return output
    sorted_values = np.sort(array[finite].copy())
    flat = array[finite]
    left = np.searchsorted(sorted_values, flat, side="left")
    right = np.searchsorted(sorted_values, flat, side="right")
    output[finite] = ((left + right) * 0.5 / len(sorted_values)).astype(np.float32)
    return output


def _topk_rows(
    query_rows: np.ndarray,
    pool_rows: np.ndarray,
    embeddings: torch.Tensor,
    *,
    k: int,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if len(query_rows) == 0 or len(pool_rows) == 0 or k <= 0:
        return (
            np.empty((len(query_rows), 0), dtype=np.int32),
            np.empty((len(query_rows), 0), dtype=np.float32),
        )
    take = min(k, len(pool_rows))
    buffer = min(len(pool_rows), max(take + 16, take))
    pool_index = torch.as_tensor(pool_rows, dtype=torch.long, device=device)
    pool = embeddings[pool_index]
    all_rows: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(query_rows), batch_size):
            rows = query_rows[start : start + batch_size]
            query = embeddings[torch.as_tensor(rows, dtype=torch.long, device=device)]
            score = query @ pool.T
            values, indices = torch.topk(score, k=buffer, dim=1, sorted=False)
            values_np = values.float().cpu().numpy()
            candidate_np = pool_rows[indices.cpu().numpy()]
            selected_rows = np.empty((len(rows), take), dtype=np.int32)
            selected_scores = np.empty((len(rows), take), dtype=np.float32)
            for local in range(len(rows)):
                order = np.lexsort((candidate_np[local], -values_np[local]))[:take]
                selected_rows[local] = candidate_np[local, order]
                selected_scores[local] = values_np[local, order]
            all_rows.append(selected_rows)
            all_scores.append(selected_scores)
    return np.concatenate(all_rows), np.concatenate(all_scores)


def select_warm_proxies(
    candidate_rows: np.ndarray,
    warm_rows: np.ndarray,
    fashion_embeddings: np.ndarray,
    product_type: np.ndarray,
    garment_group: np.ndarray,
    *,
    k: int = K_PROXY,
    device_name: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select content-only proxies with a fixed same-type/garment/global fallback."""
    candidates = np.asarray(candidate_rows, dtype=np.int32)
    warm = np.asarray(warm_rows, dtype=np.int32)
    vectors = np.asarray(fashion_embeddings, dtype=np.float32)
    valid = np.isfinite(vectors).all(axis=1)
    candidates = candidates[valid[candidates]]
    warm = warm[valid[warm]]
    proxy_rows = np.full((len(candidates), k), -1, dtype=np.int32)
    proxy_scores = np.full((len(candidates), k), np.nan, dtype=np.float32)
    fallback = np.full((len(candidates), k), 255, dtype=np.uint8)
    if len(warm) < k:
        return proxy_rows, proxy_scores, fallback
    device = torch.device(device_name)
    tensor = torch.from_numpy(vectors).to(device)
    type_pools = {int(value): warm[product_type[warm] == value] for value in np.unique(product_type[warm])}
    garment_pools = {int(value): warm[garment_group[warm] == value] for value in np.unique(garment_group[warm])}
    candidate_position = {int(row): index for index, row in enumerate(candidates.tolist())}
    for value in np.unique(product_type[candidates]):
        local = candidates[product_type[candidates] == value]
        pool = type_pools.get(int(value), np.empty(0, dtype=np.int32))
        if len(pool) >= k:
            rows, scores = _topk_rows(local, pool, tensor, k=k, device=device)
            positions = np.asarray([candidate_position[int(row)] for row in local], dtype=np.int32)
            proxy_rows[positions] = rows
            proxy_scores[positions] = scores
            fallback[positions] = 0
    incomplete = np.flatnonzero(proxy_rows[:, -1] < 0)
    for position in incomplete.tolist():
        candidate = int(candidates[position])
        selected: list[int] = []
        scores: list[float] = []
        levels: list[int] = []
        for level, pool in (
            (0, type_pools.get(int(product_type[candidate]), np.empty(0, dtype=np.int32))),
            (1, garment_pools.get(int(garment_group[candidate]), np.empty(0, dtype=np.int32))),
            (2, warm),
        ):
            pool = np.asarray([row for row in pool.tolist() if int(row) not in set(selected)], dtype=np.int32)
            needed = k - len(selected)
            if needed <= 0 or len(pool) == 0:
                continue
            rows, values = _topk_rows(
                np.asarray([candidate], dtype=np.int32), pool, tensor, k=needed, device=device
            )
            selected.extend(rows[0].tolist())
            scores.extend(values[0].tolist())
            levels.extend([level] * rows.shape[1])
        length = min(k, len(selected))
        proxy_rows[position, :length] = np.asarray(selected[:length], dtype=np.int32)
        proxy_scores[position, :length] = np.asarray(scores[:length], dtype=np.float32)
        fallback[position, :length] = np.asarray(levels[:length], dtype=np.uint8)
    del tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return proxy_rows, proxy_scores, fallback


def relation_labels(
    raw_content: np.ndarray,
    same_product_type: np.ndarray,
    direct_percentile: np.ndarray,
    deepwalk_percentile: np.ndarray,
    behavior_any: np.ndarray,
    direct_support: np.ndarray,
    *,
    content_q50: float,
    content_q75: float,
) -> np.ndarray:
    """Return one mutually exclusive secondary proxy label per observation."""
    raw = np.asarray(raw_content, dtype=np.float32)
    same = np.asarray(same_product_type, dtype=bool)
    direct = np.asarray(direct_percentile, dtype=np.float32)
    deep = np.asarray(deepwalk_percentile, dtype=np.float32)
    any_score = np.asarray(behavior_any, dtype=np.float32)
    support = np.asarray(direct_support)
    output = np.full(raw.shape, LABEL_TO_CODE["unclassified"], dtype=np.uint8)
    content_only = np.isfinite(raw) & (raw >= content_q75)
    output[content_only] = LABEL_TO_CODE["content_only"]
    mixed = np.isfinite(any_score) & (any_score >= 0.75)
    output[mixed] = LABEL_TO_CODE["behavioral_mixed"]
    multihop = (~same) & np.isfinite(raw) & (raw <= content_q50) & np.isfinite(deep) & (deep >= 0.75) & (support == 0)
    output[multihop] = LABEL_TO_CODE["multihop_complement_like"]
    direct_like = (~same) & np.isfinite(raw) & (raw <= content_q50) & np.isfinite(direct) & (direct >= 0.75)
    output[direct_like] = LABEL_TO_CODE["direct_complement_like"]
    similarity = same & np.isfinite(raw) & (raw >= content_q75)
    output[similarity] = LABEL_TO_CODE["similarity_like"]
    return output


def _sparse_values(matrix: sparse.csr_matrix, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[left.reshape(-1), right.reshape(-1)]).reshape(left.shape)


def _validate_candidate_identity(candidates: dict[str, np.ndarray]) -> dict[str, Any]:
    users = np.asarray(candidates["user_index"], dtype=np.int32)
    items = np.asarray(candidates["catalog_row"], dtype=np.int32)
    rank = np.asarray(candidates["rank"], dtype=np.int32)
    if len(users) % 200:
        raise RuntimeError("P3.1 outer rows are not divisible by 200")
    user_matrix = users.reshape(-1, 200)
    rank_matrix = rank.reshape(-1, 200)
    exact = bool(
        np.all(user_matrix == user_matrix[:, :1])
        and np.all(rank_matrix == np.arange(1, 201, dtype=np.int32)[None, :])
        and len(np.unique(np.column_stack([users, items]), axis=0)) == len(users)
    )
    if not exact:
        raise RuntimeError("P3.1 Top200 candidate identity invariant failed")
    return {
        "rows": int(len(users)),
        "active_users": int(len(user_matrix)),
        "exact_top200_groups": exact,
        "identity_sha256": hashlib.sha256(np.column_stack([users, items]).astype(np.int32).tobytes()).hexdigest(),
    }


def _window_paths(
    config: Config,
    window: str,
    p31: dict[str, Any],
    p32: dict[str, Any],
    p33: dict[str, Any],
) -> dict[str, Any]:
    coarse = p31["windows"][window]
    training_cutoffs = list(p33["students"]["assets"]["outer_validation"][window]["training_cutoffs"])
    teacher_cutoff = training_cutoffs[-1]
    p32_window = p32["windows"][window]
    p33_root = config.repo_root / "artifacts" / "phase3" / P33_RUN
    return {
        "candidates": Path(coarse["artifacts"]["candidates"]["path"]),
        "histories": Path(coarse["artifacts"]["histories"]["path"]),
        "users": Path(coarse["artifacts"]["users"]["path"]),
        "m4_embedding": Path(coarse["student_embedding"]["path"]),
        "p32_scores": Path(p32_window["scoring"]["scores"]["path"]),
        "p32_attention": Path(p32_window["scoring"]["attention"]["path"]),
        "p33_embedding": p33_root / "students-v1" / "outer" / window / "catalog_embeddings.float16.npy",
        "p33_student_manifest": p33_root / "students-v1" / "outer" / window / "manifest.json",
        "p33_teacher_dir": p33_root / "teachers-v1" / teacher_cutoff,
        "p33_teacher_cutoff": teacher_cutoff,
        "p33_training_cutoffs": training_cutoffs,
        "p33_official_coarse": p33_root / "pipeline-v1" / "coarse-v1" / "outer" / window / "candidates_top200.npz",
        "p33_official_scores": p33_root / "pipeline-v1" / "scores-v1" / window / "candidate_aware_scores.float32.npy",
        "p33_official_attention": p33_root / "pipeline-v1" / "scores-v1" / window / "attention.float16.npy",
    }


def _verify_declared(path: Path, identity: dict[str, Any]) -> None:
    observed = file_identity(path)
    if observed["bytes"] != int(identity["bytes"]) or observed["sha256"] != identity["sha256"]:
        raise RuntimeError(f"artifact identity drift: {path}")


def _load_catalog_and_fashion(config: Config) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    static_dir = (
        config.repo_root
        / "artifacts"
        / "m4"
        / "m4-v1-supervised-cold-representation"
        / "student-v1"
        / "static_catalog"
    )
    catalog = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str}).sort_values("catalog_row")
    articles_path = config.source_root / "data" / "raw" / "articles.csv"
    attributes = pd.read_csv(
        articles_path,
        dtype={"article_id": str},
        usecols=["article_id", "product_type_no", "garment_group_no", "department_no"],
    )
    catalog = catalog.merge(attributes, on="article_id", how="left", validate="one_to_one").sort_values("catalog_row")
    for column in ("product_type_no", "garment_group_no", "department_no"):
        catalog[column] = catalog[column].fillna(-1).astype(np.int32)
    image_rows = np.load(static_dir / "catalog_image_rows.int32.npy", mmap_mode="r")
    metrics = _json(config.repo_root / "reports" / "m2_4" / "m2-4-v4-fashionclip-full" / "metrics.json")
    embedding_path = Path(metrics["embeddings"]["artifact"])
    items_path = Path(metrics["embeddings"]["items_artifact"])
    if _sha(embedding_path) != metrics["embeddings"]["artifact_sha256"]:
        raise RuntimeError("FashionCLIP embedding SHA drift")
    if _sha(items_path) != metrics["embeddings"]["items_sha256"]:
        raise RuntimeError("FashionCLIP items SHA drift")
    raw = np.load(embedding_path, mmap_mode="r")
    image_items = pd.read_csv(items_path, dtype={"article_id": str}).sort_values("row_index")
    aligned = np.full((len(catalog), raw.shape[1]), np.nan, dtype=np.float32)
    valid = np.asarray(image_rows) >= 0
    aligned[valid] = np.asarray(raw[np.asarray(image_rows[valid], dtype=np.int64)], dtype=np.float32)
    mapped_items = image_items["article_id"].astype(str).to_numpy()
    if not np.array_equal(catalog.loc[valid, "article_id"].to_numpy(), mapped_items[np.asarray(image_rows[valid], dtype=np.int64)]):
        raise RuntimeError("FashionCLIP catalog row alignment failed")
    return catalog, aligned, np.asarray(image_rows), {
        "embeddings": file_identity(embedding_path),
        "items": file_identity(items_path),
        "catalog_image_rows": file_identity(static_dir / "catalog_image_rows.int32.npy"),
        "catalog_items": file_identity(static_dir / "catalog_items.csv"),
        "valid_catalog_rows": int(valid.sum()),
    }


def _direct_score_matrix(path: Path, catalog_size: int) -> sparse.csr_matrix:
    loaded = np.load(path)
    anchors = np.asarray(loaded["anchors"], dtype=np.int32)
    neighbors = np.asarray(loaded["neighbors"], dtype=np.int32)
    scores = np.asarray(loaded["scores"], dtype=np.float32)
    left = np.repeat(anchors, neighbors.shape[1])
    right = neighbors.reshape(-1)
    values = scores.reshape(-1)
    valid = right >= 0
    matrix = sparse.csr_matrix((values[valid], (left[valid], right[valid])), shape=(catalog_size, catalog_size))
    matrix = matrix.maximum(matrix.T).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _support_matrices(
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    catalog: pd.DataFrame,
    cutoff: str,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix, pd.DataFrame, dict[str, Any]]:
    _safe_cutoff(cutoff)
    catalog_map = catalog[["article_id", "catalog_row", "product_type_no"]].copy()
    connection.register("p36_catalog", catalog_map)
    try:
        frame = connection.execute(
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
            ), pairs AS (
              SELECT l.article_id AS left_article_id,r.article_id AS right_article_id,
                     count(*)::INTEGER AS total_support,
                     count(*) FILTER(WHERE l.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::INTEGER AS recent_support,
                     count(*) FILTER(WHERE l.t_dat<DATE '{cutoff}'-INTERVAL 28 DAY)::INTEGER AS older_support,
                     date_diff('day',max(l.t_dat),DATE '{cutoff}')::INTEGER AS last_age
              FROM day_items l JOIN day_items r
                ON l.customer_id=r.customer_id AND l.t_dat=r.t_dat AND l.article_id<r.article_id
              GROUP BY l.article_id,r.article_id
            )
            SELECT a.catalog_row::INTEGER AS left_row,b.catalog_row::INTEGER AS right_row,
                   a.product_type_no::INTEGER AS left_type,b.product_type_no::INTEGER AS right_type,
                   p.total_support,p.recent_support,p.older_support,p.last_age
            FROM pairs p JOIN p36_catalog a ON p.left_article_id=a.article_id
                         JOIN p36_catalog b ON p.right_article_id=b.article_id
            """
        ).fetch_df()
        latest = connection.execute(
            f"SELECT max(t_dat)::VARCHAR FROM read_parquet('{_sql_path(transactions)}') WHERE t_dat<DATE '{cutoff}'"
        ).fetchone()[0]
    finally:
        connection.unregister("p36_catalog")
    size = len(catalog)
    left = frame["left_row"].to_numpy(dtype=np.int32)
    right = frame["right_row"].to_numpy(dtype=np.int32)

    def symmetric(column: str, dtype: Any) -> sparse.csr_matrix:
        values = frame[column].to_numpy(dtype=dtype)
        rows = np.concatenate([left, right])
        cols = np.concatenate([right, left])
        data = np.concatenate([values, values])
        return sparse.csr_matrix((data, (rows, cols)), shape=(size, size))

    recent = symmetric("recent_support", np.int32)
    older = symmetric("older_support", np.int32)
    last = symmetric("last_age", np.float32)
    audit = {
        "cutoff": cutoff,
        "latest_transaction_before_cutoff": latest,
        "cutoff_safe": bool(latest < cutoff),
        "observed_item_pairs": int(len(frame)),
        "recent_plus_older_equals_total": bool(
            np.array_equal(
                frame["recent_support"].to_numpy() + frame["older_support"].to_numpy(),
                frame["total_support"].to_numpy(),
            )
        ),
    }
    return recent, older, last, frame, audit


def _category_pair_association(frame: pd.DataFrame) -> dict[tuple[int, int], tuple[float, float]]:
    cross = frame[frame["left_type"] != frame["right_type"]].copy()
    if cross.empty:
        return {}
    cross["type_left"] = np.minimum(cross["left_type"], cross["right_type"])
    cross["type_right"] = np.maximum(cross["left_type"], cross["right_type"])
    grouped = cross.groupby(["type_left", "type_right"], as_index=False)["total_support"].sum()
    endpoint = Counter()
    for row in grouped.itertuples(index=False):
        endpoint[int(row.type_left)] += int(row.total_support)
        endpoint[int(row.type_right)] += int(row.total_support)
    total = float(grouped["total_support"].sum())
    endpoint_total = 2.0 * total
    lifts = []
    for row in grouped.itertuples(index=False):
        expected = 2.0 * (endpoint[int(row.type_left)] / endpoint_total) * (endpoint[int(row.type_right)] / endpoint_total)
        observed = float(row.total_support) / total
        lifts.append(observed / expected if expected > 0 else np.nan)
    grouped["lift"] = np.asarray(lifts, dtype=np.float64)
    percentiles = empirical_mid_percentile(grouped["lift"].to_numpy(dtype=np.float32))
    return {
        (int(row.type_left), int(row.type_right)): (float(row.lift), float(percentiles[index]))
        for index, row in enumerate(grouped.itertuples(index=False))
    }


def _load_teacher_assets(
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    catalog: pd.DataFrame,
    teacher_dir: Path,
    cutoff: str,
) -> TeacherAssets:
    manifest_path = teacher_dir / "manifest.json"
    manifest = _json(manifest_path)
    if manifest["cutoff"] != cutoff or not manifest["audits"]["all_cutoff_safe"]:
        raise RuntimeError(f"teacher cutoff identity failed at {cutoff}")
    for section in ("neighbors", "relations"):
        for identity in manifest[section].values():
            _verify_declared(Path(identity["path"]), identity)
    _verify_declared(Path(manifest["deepwalk_vectors"]["path"]), manifest["deepwalk_vectors"])
    m4_manifest = _json(Path(manifest["inputs"]["m4_item2vec_relation"]["path"]).parent / "manifest.json")
    i2v_declared = m4_manifest["inputs"]["teacher_artifacts"]
    for identity in i2v_declared.values():
        _verify_declared(Path(identity["path"]), identity)
    catalog_size = len(catalog)
    article_to_row = dict(zip(catalog["article_id"].astype(str), catalog["catalog_row"].astype(int)))
    i2v_items = pd.read_csv(i2v_declared["items"]["path"], dtype={"article_id": str}).sort_values("row_index")
    i2v_raw = np.load(i2v_declared["normalized_vectors"]["path"], mmap_mode="r")
    i2v_vectors = np.zeros((catalog_size, i2v_raw.shape[1]), dtype=np.float32)
    i2v_valid = np.zeros(catalog_size, dtype=bool)
    i2v_rows = np.asarray([article_to_row[value] for value in i2v_items["article_id"]], dtype=np.int32)
    i2v_vectors[i2v_rows] = np.asarray(i2v_raw, dtype=np.float32)
    i2v_valid[i2v_rows] = True
    direct_path = Path(manifest["neighbors"]["direct_covisit"]["path"])
    direct_loaded = np.load(direct_path)
    deep_anchors = np.asarray(direct_loaded["anchors"], dtype=np.int32)
    deep_raw = np.load(manifest["deepwalk_vectors"]["path"], mmap_mode="r")
    if len(deep_anchors) != len(deep_raw):
        raise RuntimeError(f"DeepWalk anchor/vector alignment failed at {cutoff}")
    deep_vectors = np.zeros((catalog_size, deep_raw.shape[1]), dtype=np.float32)
    deep_valid = np.zeros(catalog_size, dtype=bool)
    deep_vectors[deep_anchors] = np.asarray(deep_raw, dtype=np.float32)
    deep_valid[deep_anchors] = True
    recent, older, last, pair_support, support_audit = _support_matrices(
        connection, transactions, catalog, cutoff
    )
    category_pair = _category_pair_association(pair_support)
    return TeacherAssets(
        cutoff=cutoff,
        i2v_vectors=i2v_vectors,
        i2v_valid=i2v_valid,
        deep_vectors=deep_vectors,
        deep_valid=deep_valid,
        direct_score=_direct_score_matrix(direct_path, catalog_size),
        direct_recent=recent,
        direct_older=older,
        direct_last_age=last,
        pair_support=pair_support,
        category_pair=category_pair,
        identities={
            "manifest": file_identity(manifest_path),
            "item2vec_items": file_identity(Path(i2v_declared["items"]["path"])),
            "item2vec_vectors": file_identity(Path(i2v_declared["normalized_vectors"]["path"])),
            "direct_neighbors": file_identity(direct_path),
            "deepwalk_vectors": file_identity(Path(manifest["deepwalk_vectors"]["path"])),
        },
        cutoff_audit=support_audit,
    )


def _teacher_pair_scores(
    history_rows: np.ndarray,
    proxy_rows: np.ndarray,
    history_mask: np.ndarray,
    teacher_vectors: torch.Tensor,
    teacher_valid: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    safe_history = np.maximum(history_rows, 0)
    safe_proxy = np.maximum(proxy_rows, 0)
    h = teacher_vectors[torch.as_tensor(safe_history, dtype=torch.long, device=device)]
    p = teacher_vectors[torch.as_tensor(safe_proxy, dtype=torch.long, device=device)]
    score = torch.einsum("uhd,ukpd->ukhp", h, p)
    valid_h = history_mask & teacher_valid[safe_history]
    valid_p = (proxy_rows >= 0) & teacher_valid[safe_proxy]
    valid = torch.as_tensor(valid_h[:, None, :, None], device=device) & torch.as_tensor(
        valid_p[:, :, None, :], device=device
    )
    score = score.masked_fill(~valid, -torch.inf)
    maximum = score.max(dim=3).values
    count = valid.sum(dim=3).clamp(min=1, max=3)
    top = torch.topk(score, k=min(3, score.shape[3]), dim=3).values
    top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
    top3 = top.sum(dim=3) / count
    no_valid = ~valid.any(dim=3)
    maximum = maximum.masked_fill(no_valid, torch.nan)
    top3 = top3.masked_fill(no_valid, torch.nan)
    return maximum.float().cpu().numpy(), top3.float().cpu().numpy()


def _compute_pair_features(
    *,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    proxies_by_catalog: np.ndarray,
    fashion: np.ndarray,
    student: np.ndarray,
    product_type: np.ndarray,
    garment_group: np.ndarray,
    department: np.ndarray,
    teacher: TeacherAssets,
    device_name: str,
    user_batch: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    identity = _validate_candidate_identity(candidates)
    users = np.asarray(candidates["user_index"], dtype=np.int32)
    items = np.asarray(candidates["catalog_row"], dtype=np.int32)
    active_users = users.reshape(-1, 200)[:, 0]
    item_matrix = items.reshape(-1, 200)
    history_rows_all = np.asarray(histories["catalog_row"], dtype=np.int32)
    history_days_all = np.asarray(histories["days_since_purchase"], dtype=np.float32)
    history_mask_all = np.asarray(histories["mask"], dtype=bool)
    if history_rows_all.shape[1] != HISTORY_N:
        raise RuntimeError("P3.2 history width drift")
    shape = (len(items), HISTORY_N)
    float_names = (
        "raw_content",
        "student_content",
        "i2v_max",
        "i2v_top3",
        "deepwalk_max",
        "deepwalk_top3",
        "direct_max",
        "direct_last_age",
    )
    pair = {name: np.full(shape, np.nan, dtype=np.float32) for name in float_names}
    pair["direct_recent"] = np.zeros(shape, dtype=np.uint16)
    pair["direct_older"] = np.zeros(shape, dtype=np.uint16)
    pair["same_product_type"] = np.zeros(shape, dtype=bool)
    pair["same_garment_group"] = np.zeros(shape, dtype=bool)
    pair["same_department"] = np.zeros(shape, dtype=bool)
    pair["history_age"] = np.zeros(shape, dtype=np.float32)
    pair["valid_history"] = np.zeros(shape, dtype=bool)
    device = torch.device(device_name)
    fashion_tensor = torch.from_numpy(np.nan_to_num(fashion, nan=0.0).astype(np.float32)).to(device)
    student_tensor = torch.from_numpy(np.asarray(student, dtype=np.float32)).to(device)
    i2v_tensor = torch.from_numpy(teacher.i2v_vectors).to(device)
    deep_tensor = torch.from_numpy(teacher.deep_vectors).to(device)
    fashion_valid = np.isfinite(fashion).all(axis=1)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, len(active_users), user_batch):
            batch_end = min(batch_start + user_batch, len(active_users))
            local_users = active_users[batch_start:batch_end]
            local_items = item_matrix[batch_start:batch_end]
            local_history = history_rows_all[local_users]
            local_mask = history_mask_all[local_users]
            local_days = history_days_all[local_users]
            safe_history = np.maximum(local_history, 0)
            local_proxies = proxies_by_catalog[local_items]
            if np.any((local_proxies < 0) & (local_proxies != -1)):
                raise RuntimeError("invalid proxy row sentinel")
            row_start = batch_start * 200
            row_end = batch_end * 200
            out_shape = ((batch_end - batch_start) * 200, HISTORY_N)

            c_fashion = fashion_tensor[torch.as_tensor(local_items, dtype=torch.long, device=device)]
            h_fashion = fashion_tensor[torch.as_tensor(safe_history, dtype=torch.long, device=device)]
            raw = torch.einsum("ukd,uhd->ukh", c_fashion, h_fashion)
            raw_valid = (
                local_mask[:, None, :]
                & fashion_valid[local_items][:, :, None]
                & fashion_valid[safe_history][:, None, :]
            )
            raw_np = raw.float().cpu().numpy()
            raw_np[~raw_valid] = np.nan
            pair["raw_content"][row_start:row_end] = raw_np.reshape(out_shape)

            c_student = student_tensor[torch.as_tensor(local_items, dtype=torch.long, device=device)]
            h_student = student_tensor[torch.as_tensor(safe_history, dtype=torch.long, device=device)]
            student_score = torch.einsum("ukd,uhd->ukh", c_student, h_student).float().cpu().numpy()
            student_valid = np.broadcast_to(local_mask[:, None, :], student_score.shape)
            student_score[~student_valid] = np.nan
            pair["student_content"][row_start:row_end] = student_score.reshape(out_shape)

            i2v_max, i2v_top3 = _teacher_pair_scores(
                local_history,
                local_proxies,
                local_mask,
                i2v_tensor,
                teacher.i2v_valid,
                device=device,
            )
            deep_max, deep_top3 = _teacher_pair_scores(
                local_history,
                local_proxies,
                local_mask,
                deep_tensor,
                teacher.deep_valid,
                device=device,
            )
            pair["i2v_max"][row_start:row_end] = i2v_max.reshape(out_shape)
            pair["i2v_top3"][row_start:row_end] = i2v_top3.reshape(out_shape)
            pair["deepwalk_max"][row_start:row_end] = deep_max.reshape(out_shape)
            pair["deepwalk_top3"][row_start:row_end] = deep_top3.reshape(out_shape)

            left = np.broadcast_to(safe_history[:, None, :, None], (*local_items.shape, HISTORY_N, K_PROXY))
            right = np.broadcast_to(local_proxies[:, :, None, :], (*local_items.shape, HISTORY_N, K_PROXY))
            valid_direct = local_mask[:, None, :, None] & (right >= 0)
            safe_right = np.maximum(right, 0)
            direct_values = _sparse_values(teacher.direct_score, left, safe_right).astype(np.float32)
            direct_values[~valid_direct] = 0.0
            direct_max = direct_values.max(axis=3)
            direct_max[direct_max <= 0] = np.nan
            recent = _sparse_values(teacher.direct_recent, left, safe_right).astype(np.int32)
            older = _sparse_values(teacher.direct_older, left, safe_right).astype(np.int32)
            recent[~valid_direct] = 0
            older[~valid_direct] = 0
            last = _sparse_values(teacher.direct_last_age, left, safe_right).astype(np.float32)
            last[(~valid_direct) | ((recent + older) == 0)] = np.nan
            last_min = np.where(np.isfinite(last), last, np.inf).min(axis=3)
            last_min[~np.isfinite(last_min)] = np.nan
            pair["direct_max"][row_start:row_end] = direct_max.reshape(out_shape)
            pair["direct_recent"][row_start:row_end] = np.clip(recent.max(axis=3), 0, 65535).astype(np.uint16).reshape(out_shape)
            pair["direct_older"][row_start:row_end] = np.clip(older.max(axis=3), 0, 65535).astype(np.uint16).reshape(out_shape)
            pair["direct_last_age"][row_start:row_end] = last_min.reshape(out_shape)
            same = product_type[local_items][:, :, None] == product_type[safe_history][:, None, :]
            pair["same_product_type"][row_start:row_end] = (same & local_mask[:, None, :]).reshape(out_shape)
            same_garment = garment_group[local_items][:, :, None] == garment_group[safe_history][:, None, :]
            pair["same_garment_group"][row_start:row_end] = (
                same_garment & local_mask[:, None, :]
            ).reshape(out_shape)
            same_department = department[local_items][:, :, None] == department[safe_history][:, None, :]
            pair["same_department"][row_start:row_end] = (
                same_department & local_mask[:, None, :]
            ).reshape(out_shape)
            pair["history_age"][row_start:row_end] = np.broadcast_to(local_days[:, None, :], (*local_items.shape, HISTORY_N)).reshape(out_shape)
            pair["valid_history"][row_start:row_end] = np.broadcast_to(local_mask[:, None, :], (*local_items.shape, HISTORY_N)).reshape(out_shape)
            if batch_start == 0 or batch_end == len(active_users) or batch_end % 1000 < user_batch:
                print(f"P3.6 pair features users {batch_end}/{len(active_users)}", flush=True)
    del fashion_tensor, student_tensor, i2v_tensor, deep_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pair["direct_support"] = pair["direct_recent"].astype(np.int32) + pair["direct_older"].astype(np.int32)
    return pair, {**identity, "elapsed_seconds": time.perf_counter() - started}


def _safe_row_nanmax(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    output = np.full(len(values), np.nan, dtype=np.float32)
    if valid.any():
        filled = np.where(valid, values, -np.inf)
        maximum = filled.max(axis=1)
        output[np.isfinite(maximum)] = maximum[np.isfinite(maximum)]
    return output


def _safe_row_nanmean(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    numerator = np.where(valid, values, 0.0).sum(axis=1)
    denominator = valid.sum(axis=1)
    output = np.full(len(values), np.nan, dtype=np.float32)
    usable = denominator > 0
    output[usable] = (numerator[usable] / denominator[usable]).astype(np.float32)
    return output


def _category_pair_values(
    candidate_types: np.ndarray,
    history_types: np.ndarray,
    mapping: dict[tuple[int, int], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    left = np.minimum(candidate_types, history_types)
    right = np.maximum(candidate_types, history_types)
    lift = np.full(left.shape, np.nan, dtype=np.float32)
    percentile = np.full(left.shape, np.nan, dtype=np.float32)
    keys = left.astype(np.int64) * 10000 + right.astype(np.int64)
    if mapping:
        ordered = sorted((int(a) * 10000 + int(b), value, pct) for (a, b), (value, pct) in mapping.items())
        map_keys = np.asarray([row[0] for row in ordered], dtype=np.int64)
        map_lift = np.asarray([row[1] for row in ordered], dtype=np.float32)
        map_pct = np.asarray([row[2] for row in ordered], dtype=np.float32)
        flat = keys.reshape(-1)
        position = np.searchsorted(map_keys, flat)
        valid = position < len(map_keys)
        valid[valid] &= map_keys[position[valid]] == flat[valid]
        lift.reshape(-1)[valid] = map_lift[position[valid]]
        percentile.reshape(-1)[valid] = map_pct[position[valid]]
    return lift, percentile


def relation_matrix(
    *,
    content_codes: np.ndarray,
    behavior_codes: np.ndarray,
    same_type: np.ndarray,
    valid_history: np.ndarray,
    target: np.ndarray,
    history_age: np.ndarray,
    raw_content: np.ndarray,
    behavior_any: np.ndarray,
) -> dict[str, Any]:
    target_matrix = np.broadcast_to(np.asarray(target, dtype=bool)[:, None], valid_history.shape)
    eligible = valid_history & (content_codes < 4) & (behavior_codes < 4)
    overall_rate = float(np.asarray(target, dtype=bool).mean()) if len(target) else 0.0
    cells: dict[str, Any] = {}
    observation_sum = 0
    for content in range(4):
        for behavior in range(4):
            for same in (0, 1):
                selected = eligible & (content_codes == content) & (behavior_codes == behavior) & (same_type == bool(same))
                candidate_selected = selected.any(axis=1)
                truth_candidates = candidate_selected & np.asarray(target, dtype=bool)
                candidate_rows = int(candidate_selected.sum())
                observations = int(selected.sum())
                observation_sum += observations
                truth_rate = float(truth_candidates.sum() / candidate_rows) if candidate_rows else None
                name = f"content_Q{content + 1}__behavior_Q{behavior + 1}__{'same_type' if same else 'cross_type'}"
                cells[name] = {
                    "candidate_history_observations": observations,
                    "candidate_rows": candidate_rows,
                    "truth_candidate_rows": int(truth_candidates.sum()),
                    "unobserved_candidate_rows": int(candidate_rows - truth_candidates.sum()),
                    "truth_rate": truth_rate,
                    "truth_rate_lift_vs_all_valid_history_candidates": (
                        truth_rate / overall_rate if truth_rate is not None and overall_rate > 0 else None
                    ),
                    "history_age_days": _summary(history_age[selected]),
                    "raw_content_similarity": _summary(raw_content[selected]),
                    "behavior_affinity_any": _summary(behavior_any[selected]),
                }
    ineligible = valid_history & ~eligible
    return {
        "cells": cells,
        "eligible_observations": int(eligible.sum()),
        "ineligible_observations": int(ineligible.sum()),
        "valid_history_observations": int(valid_history.sum()),
        "observation_conservation_passed": observation_sum + int(ineligible.sum()) == int(valid_history.sum()),
        "overall_candidate_truth_rate": float(np.asarray(target, dtype=bool).mean()),
        "note": "candidate rows can occur in multiple cells because one candidate has up to 20 history observations; only observation counts are additive",
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    numerator = np.where(valid, values * weights, 0.0).sum(axis=1)
    denominator = np.where(valid, weights, 0.0).sum(axis=1)
    output = np.full(len(values), np.nan, dtype=np.float32)
    usable = denominator > 0
    output[usable] = (numerator[usable] / denominator[usable]).astype(np.float32)
    return output


def _topk_mean(values: np.ndarray, k: int = 3) -> np.ndarray:
    """Return the mean of up to k largest finite values per candidate row."""
    array = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(array)
    filled = np.where(valid, array, -np.inf)
    take = min(k, array.shape[1])
    selected = np.partition(filled, array.shape[1] - take, axis=1)[:, -take:]
    selected_valid = np.isfinite(selected)
    count = selected_valid.sum(axis=1)
    result = np.full(len(array), np.nan, dtype=np.float32)
    usable = count > 0
    result[usable] = np.where(selected_valid, selected, 0.0).sum(axis=1)[usable] / count[usable]
    return result


def _aggregate_candidate_features(
    *,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    pair: dict[str, np.ndarray],
    attention: np.ndarray,
    product_type: np.ndarray,
    category_pair: dict[tuple[int, int], tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    users = np.asarray(candidates["user_index"], dtype=np.int32)
    items = np.asarray(candidates["catalog_row"], dtype=np.int32)
    target = np.asarray(candidates["target"], dtype=np.uint8)
    valid = pair["valid_history"]
    if attention.shape != valid.shape:
        raise RuntimeError("P3.2 attention shape does not match P3.1 candidate-history observations")
    weights = np.asarray(attention, dtype=np.float32)
    weights[~valid] = 0.0
    weight_sum = weights.sum(axis=1)
    if not np.allclose(weight_sum[weight_sum > 0], 1.0, atol=5e-3):
        raise RuntimeError("P3.2 attention normalization drift")
    label = pair["relation_label"]
    counts = np.column_stack([np.count_nonzero((label == code) & valid, axis=1) for code in range(len(LABELS))]).astype(np.uint8)
    strengths = np.zeros((len(items), len(LABELS)), dtype=np.float32)
    strength_sources = (
        pair["raw_content"],
        pair["direct_pct"],
        pair["deepwalk_pct"],
        pair["behavior_any"],
        pair["raw_content"],
        np.zeros_like(pair["raw_content"]),
    )
    for code, source in enumerate(strength_sources):
        strengths[:, code] = _safe_row_nanmax(np.where((label == code) & valid, source, np.nan))
    best = np.zeros(len(items), dtype=np.uint8)
    best_count = counts[:, 0].copy()
    best_strength = np.nan_to_num(strengths[:, 0], nan=-np.inf)
    for code in range(1, len(LABELS)):
        local_strength = np.nan_to_num(strengths[:, code], nan=-np.inf)
        replace = (counts[:, code] > best_count) | ((counts[:, code] == best_count) & (local_strength > best_strength))
        best[replace] = code
        best_count[replace] = counts[replace, code]
        best_strength[replace] = local_strength[replace]
    attention_mass = np.column_stack(
        [np.where((label == code) & valid, weights, 0.0).sum(axis=1) for code in range(len(LABELS))]
    ).astype(np.float32)
    attention_dominant = np.argmax(attention_mass, axis=1).astype(np.uint8)

    same = pair["same_product_type"]
    primary_code = np.full(valid.shape, 255, dtype=np.uint8)
    eligible_primary = valid & (pair["content_q"] < 4) & (pair["behavior_q"] < 4)
    primary_code[eligible_primary] = (
        pair["content_q"][eligible_primary] * 8
        + pair["behavior_q"][eligible_primary] * 2
        + same[eligible_primary].astype(np.uint8)
    )
    primary_mass = np.zeros((len(items), 32), dtype=np.float32)
    for code in range(32):
        primary_mass[:, code] = np.where(primary_code == code, weights, 0.0).sum(axis=1)
    primary_dominant = np.argmax(primary_mass, axis=1).astype(np.uint8)
    primary_dominant[primary_mass.max(axis=1) <= 0] = 255

    behavior_for_arg = np.nan_to_num(pair["behavior_any"], nan=-1.0)
    content_for_arg = np.nan_to_num(pair["raw_content"], nan=-1.0)
    history_position = np.arange(HISTORY_N, dtype=np.float32)[None, :]
    best_pair_position = np.argmax(behavior_for_arg * 2.0 + content_for_arg * 1e-3 - history_position * 1e-7, axis=1)
    best_pair_same = same[np.arange(len(items)), best_pair_position]
    no_behavior = ~np.isfinite(pair["behavior_any"]).any(axis=1)
    best_pair_same[no_behavior] = False

    history_rows = np.asarray(histories["catalog_row"], dtype=np.int32)[users]
    safe_history = np.maximum(history_rows, 0)
    candidate_types = product_type[items][:, None]
    history_types = product_type[safe_history]
    category_lift, category_pct = _category_pair_values(candidate_types, history_types, category_pair)
    category_lift[~valid] = np.nan
    category_pct[~valid] = np.nan
    complement_mask = np.isin(
        label,
        [LABEL_TO_CODE["direct_complement_like"], LABEL_TO_CODE["multihop_complement_like"]],
    ) & valid

    multihop_eligible = valid & (pair["direct_support"] == 0)
    multihop_score = _safe_row_nanmax(np.where(multihop_eligible, pair["deepwalk_pct"], np.nan))
    direct_score = _safe_row_nanmax(pair["direct_pct"])
    frame = pd.DataFrame(
        {
            "user_index": users,
            "catalog_row": items,
            "target": target,
            "coarse_rank": np.asarray(candidates["rank"], dtype=np.uint16),
            "coarse_score": np.asarray(candidates["coarse_score"], dtype=np.float32),
            "max_raw_content_similarity": _safe_row_nanmax(pair["raw_content"]),
            "top3_raw_content_similarity": _topk_mean(pair["raw_content"]),
            "max_student_similarity": _safe_row_nanmax(pair["student_content"]),
            "top3_student_similarity": _topk_mean(pair["student_content"]),
            "max_i2v_affinity_pct": _safe_row_nanmax(pair["i2v_pct"]),
            "max_direct_affinity_pct": direct_score,
            "max_deepwalk_affinity_pct": _safe_row_nanmax(pair["deepwalk_pct"]),
            "max_behavior_affinity_any": _safe_row_nanmax(pair["behavior_any"]),
            "multihop_behavior_score_direct_absent": multihop_score,
            "best_relation_type_code": best,
            "attention_dominant_relation_type_code": attention_dominant,
            "attention_dominant_primary_cell_code": primary_dominant,
            "best_behavior_pair_same_product_type": best_pair_same.astype(np.uint8),
            "attention_weighted_raw_content_similarity": _weighted_mean(pair["raw_content"], weights),
            "attention_weighted_behavior_affinity": _weighted_mean(pair["behavior_any"], weights),
            "max_complement_category_pair_lift": _safe_row_nanmax(np.where(complement_mask, category_lift, np.nan)),
            "max_complement_category_pair_percentile": _safe_row_nanmax(np.where(complement_mask, category_pct, np.nan)),
        }
    )
    for code, name in enumerate(LABELS):
        frame[f"{name}_count"] = counts[:, code]
        frame[f"attention_mass_{name}"] = attention_mass[:, code]
    frame["max_raw_content_quartile"], content_cuts = _quartile_codes(frame["max_raw_content_similarity"].to_numpy())
    frame["max_behavior_quartile"], behavior_cuts = _quartile_codes(frame["max_behavior_affinity_any"].to_numpy())
    frame["multihop_score_quartile"], multihop_cuts = _quartile_codes(frame["multihop_behavior_score_direct_absent"].to_numpy())
    frame["direct_score_quartile"], direct_cuts = _quartile_codes(frame["max_direct_affinity_pct"].to_numpy())
    audit = {
        "rows": int(len(frame)),
        "target_rows": int(target.sum()),
        "history_observations": int(valid.sum()),
        "secondary_label_observation_conservation": int(sum(np.count_nonzero((label == code) & valid) for code in range(len(LABELS)))) == int(valid.sum()),
        "attention_normalized_candidate_rows": int(np.count_nonzero(weight_sum > 0)),
        "candidate_quartile_cuts": {
            "max_raw_content_similarity": content_cuts,
            "max_behavior_affinity_any": behavior_cuts,
            "multihop_behavior_score_direct_absent": multihop_cuts,
            "max_direct_affinity_pct": direct_cuts,
        },
        "label_codebook": {str(code): name for code, name in enumerate(LABELS)},
        "primary_cell_code": "content_quartile_zero_based*8 + behavior_quartile_zero_based*2 + same_product_type",
    }
    return frame, audit


def _rank_attribution(
    *,
    frame: pd.DataFrame,
    baseline_rank: np.ndarray,
    changed_rank: np.ndarray,
    group_values: np.ndarray,
    group_names: dict[int, str],
) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=bool)
    delta = np.asarray(baseline_rank, dtype=np.int32) - np.asarray(changed_rank, dtype=np.int32)
    result: dict[str, Any] = {}
    density_denominator = int(len(np.unique(frame["user_index"])) * 20)
    for code, name in group_names.items():
        selected = np.asarray(group_values) == code
        truth = selected & target
        local = delta[truth]
        before = np.asarray(baseline_rank)[truth]
        after = np.asarray(changed_rank)[truth]
        result[name] = {
            "candidate_rows": int(selected.sum()),
            "truth_candidate_rows": int(truth.sum()),
            "unobserved_candidate_rows": int(selected.sum() - truth.sum()),
            "truth_moved_up": int(np.count_nonzero(local > 0)),
            "truth_same_rank": int(np.count_nonzero(local == 0)),
            "truth_moved_down": int(np.count_nonzero(local < 0)),
            "median_rank_delta": float(np.median(local)) if len(local) else None,
            "mean_rank_delta": float(np.mean(local)) if len(local) else None,
            "top200_to_top50_before": float(np.mean(before <= 50)) if len(local) else None,
            "top200_to_top50_after": float(np.mean(after <= 50)) if len(local) else None,
            "top200_to_top20_before": float(np.mean(before <= 20)) if len(local) else None,
            "top200_to_top20_after": float(np.mean(after <= 20)) if len(local) else None,
            "top200_to_top10_before": float(np.mean(before <= 10)) if len(local) else None,
            "top200_to_top10_after": float(np.mean(after <= 10)) if len(local) else None,
            "top20_net_truth": int(np.count_nonzero(after <= 20) - np.count_nonzero(before <= 20)),
            "density20_additive_contribution": (
                float((np.count_nonzero(after <= 20) - np.count_nonzero(before <= 20)) / density_denominator)
                if density_denominator else None
            ),
            "reciprocal_rank_pair_proxy_delta_mean": (
                float(np.mean(1.0 / after - 1.0 / before)) if len(local) else None
            ),
        }
    return result


def _relation_presence_attribution(
    *,
    frame: pd.DataFrame,
    baseline_rank: np.ndarray,
    changed_rank: np.ndarray,
) -> dict[str, Any]:
    """Overlapping candidate groups defined by at least one matching history relation."""
    masks = {
        name: frame[f"{name}_count"].to_numpy() > 0 for name in LABELS
    }
    masks["complement_like_any"] = (
        masks["direct_complement_like"] | masks["multihop_complement_like"]
    )
    return {
        name: _rank_attribution(
            frame=frame,
            baseline_rank=baseline_rank,
            changed_rank=changed_rank,
            group_values=mask.astype(np.uint8),
            group_names={1: "selected"},
        )["selected"]
        for name, mask in masks.items()
    }


def _primary_code_names() -> dict[int, str]:
    result = {}
    for content in range(4):
        for behavior in range(4):
            for same in (0, 1):
                code = content * 8 + behavior * 2 + same
                result[code] = f"content_Q{content + 1}__behavior_Q{behavior + 1}__{'same_type' if same else 'cross_type'}"
    result[255] = "ineligible"
    return result


def _quartile_names() -> dict[int, str]:
    return {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4", 255: "ineligible"}


def _p32_attribution(frame: pd.DataFrame, p32_scores: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    users = frame["user_index"].to_numpy(dtype=np.int32)
    items = frame["catalog_row"].to_numpy(dtype=np.int32)
    coarse_rank = frame["coarse_rank"].to_numpy(dtype=np.int32)
    aware_rank = _rank_by_score(users, items, np.asarray(p32_scores, dtype=np.float32))
    label_names = {code: name for code, name in enumerate(LABELS)}
    family_code = np.full(len(frame), 255, dtype=np.uint8)
    best_code = frame["best_relation_type_code"].to_numpy(dtype=np.uint8)
    family_code[best_code == LABEL_TO_CODE["similarity_like"]] = 0
    family_code[np.isin(best_code, [LABEL_TO_CODE["direct_complement_like"], LABEL_TO_CODE["multihop_complement_like"]])] = 1
    return {
        "primary_attention_dominant_matrix_cell": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=frame["attention_dominant_primary_cell_code"].to_numpy(),
            group_names=_primary_code_names(),
        ),
        "best_relation_type": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=frame["best_relation_type_code"].to_numpy(),
            group_names=label_names,
        ),
        "relation_presence": _relation_presence_attribution(
            frame=frame, baseline_rank=coarse_rank, changed_rank=aware_rank
        ),
        "attention_dominant_relation_type": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=frame["attention_dominant_relation_type_code"].to_numpy(),
            group_names=label_names,
        ),
        "best_relation_family": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=family_code,
            group_names={0: "similarity_like", 1: "complement_like", 255: "other_or_ineligible"},
        ),
        "candidate_max_content_quartile": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=frame["max_raw_content_quartile"].to_numpy(),
            group_names=_quartile_names(),
        ),
        "candidate_max_behavior_quartile": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=frame["max_behavior_quartile"].to_numpy(),
            group_names=_quartile_names(),
        ),
        "overall": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=aware_rank,
            group_values=np.zeros(len(frame), dtype=np.uint8),
            group_names={0: "all_candidates"},
        )["all_candidates"],
    }, aware_rank


def _p33_attribution(
    frame: pd.DataFrame,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    p33_embeddings: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    scores, _ = _seed_scores(candidates, histories, p33_embeddings)
    users = frame["user_index"].to_numpy(dtype=np.int32)
    items = frame["catalog_row"].to_numpy(dtype=np.int32)
    single_rank = frame["coarse_rank"].to_numpy(dtype=np.int32)
    multiview_rank = _rank_by_score(users, items, scores)
    label_names = {code: name for code, name in enumerate(LABELS)}
    same_names = {0: "cross_product_type", 1: "same_product_type"}
    family_code = np.full(len(frame), 255, dtype=np.uint8)
    best_code = frame["best_relation_type_code"].to_numpy(dtype=np.uint8)
    family_code[best_code == LABEL_TO_CODE["similarity_like"]] = 0
    family_code[np.isin(best_code, [LABEL_TO_CODE["direct_complement_like"], LABEL_TO_CODE["multihop_complement_like"]])] = 1
    return {
        "candidate_max_content_quartile": _rank_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank,
            group_values=frame["max_raw_content_quartile"].to_numpy(), group_names=_quartile_names()
        ),
        "candidate_max_behavior_quartile": _rank_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank,
            group_values=frame["max_behavior_quartile"].to_numpy(), group_names=_quartile_names()
        ),
        "best_behavior_pair_type": _rank_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank,
            group_values=frame["best_behavior_pair_same_product_type"].to_numpy(), group_names=same_names
        ),
        "best_relation_type": _rank_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank,
            group_values=frame["best_relation_type_code"].to_numpy(), group_names=label_names
        ),
        "relation_presence": _relation_presence_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank
        ),
        "best_relation_family": _rank_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank,
            group_values=family_code,
            group_names={0: "similarity_like", 1: "complement_like", 255: "other_or_ineligible"}
        ),
        "overall": _rank_attribution(
            frame=frame, baseline_rank=single_rank, changed_rank=multiview_rank,
            group_values=np.zeros(len(frame), dtype=np.uint8), group_names={0: "all_candidates"}
        )["all_candidates"],
    }, multiview_rank


def _proxy_quality(
    *,
    candidates: dict[str, np.ndarray],
    counts_by_row: np.ndarray,
    proxies_by_catalog: np.ndarray,
    proxy_scores_by_catalog: np.ndarray,
    fallback_by_catalog: np.ndarray,
) -> dict[str, Any]:
    items = np.asarray(candidates["catalog_row"], dtype=np.int32)
    target = np.asarray(candidates["target"], dtype=bool)
    unique_items = np.unique(items)
    unique_proxy = proxies_by_catalog[unique_items]
    row_proxy = proxies_by_catalog[items]
    row_scores = proxy_scores_by_catalog[items]
    unique_full = np.all(unique_proxy >= 0, axis=1)
    row_full = np.all(row_proxy >= 0, axis=1)
    assignment = fallback_by_catalog[unique_items]
    valid_assignment = assignment < 3
    per_row_mean = _safe_row_nanmean(row_scores)
    per_row_max = _safe_row_nanmax(row_scores)
    proxy_rows = unique_proxy[unique_proxy >= 0]
    return {
        "candidate_rows": int(len(items)),
        "strict_cold_candidate_rows": int(np.count_nonzero(counts_by_row[items] == 0)),
        "sparse_1_5_candidate_rows": int(np.count_nonzero((counts_by_row[items] >= 1) & (counts_by_row[items] <= 5))),
        "unique_candidate_items": int(len(unique_items)),
        "candidate_rows_with_at_least_one_proxy": int(np.count_nonzero(np.any(row_proxy >= 0, axis=1))),
        "candidate_rows_with_full_5_proxies": int(row_full.sum()),
        "candidate_row_full_5_share": float(row_full.mean()),
        "unique_items_with_at_least_one_proxy": int(np.count_nonzero(np.any(unique_proxy >= 0, axis=1))),
        "unique_items_with_full_5_proxies": int(unique_full.sum()),
        "unique_item_full_5_share": float(unique_full.mean()),
        "proxy_assignments": int(valid_assignment.sum()),
        "same_product_type_proxy_share": float(np.count_nonzero(assignment == 0) / max(int(valid_assignment.sum()), 1)),
        "garment_fallback_share": float(np.count_nonzero(assignment == 1) / max(int(valid_assignment.sum()), 1)),
        "global_fallback_share": float(np.count_nonzero(assignment == 2) / max(int(valid_assignment.sum()), 1)),
        "proxy_cosine_all_assignments": _summary(proxy_scores_by_catalog[unique_items][valid_assignment]),
        "truth_candidate_proxy_mean_cosine": _summary(per_row_mean[target]),
        "unobserved_candidate_proxy_mean_cosine": _summary(per_row_mean[~target]),
        "truth_candidate_proxy_max_cosine": _summary(per_row_max[target]),
        "unobserved_candidate_proxy_max_cosine": _summary(per_row_max[~target]),
        "all_selected_proxies_are_warm_before_outer_cutoff": bool(
            len(proxy_rows) == 0 or np.all(counts_by_row[proxy_rows] > 5)
        ),
        "proxy_selection_identity_sha256": hashlib.sha256(
            np.column_stack(
                [
                    np.repeat(unique_items, K_PROXY),
                    unique_proxy.reshape(-1),
                    assignment.reshape(-1).astype(np.int32),
                ]
            ).astype(np.int32).tobytes()
        ).hexdigest(),
    }


def _attention_relation_audit(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=bool)
    output: dict[str, Any] = {
        "attention_weighted_content": {},
        "attention_weighted_behavior": {},
        "relation_type_mass": {},
    }
    for name, selected in (("truth", target), ("unobserved", ~target)):
        output["attention_weighted_content"][name] = _summary(
            frame.loc[selected, "attention_weighted_raw_content_similarity"].to_numpy()
        )
        output["attention_weighted_behavior"][name] = _summary(
            frame.loc[selected, "attention_weighted_behavior_affinity"].to_numpy()
        )
        output["relation_type_mass"][name] = {
            label: _summary(frame.loc[selected, f"attention_mass_{label}"].to_numpy()) for label in LABELS
        }
    return output


def _mask_rank_metrics(
    frame: pd.DataFrame,
    mask: np.ndarray,
    baseline_rank: np.ndarray,
    changed_rank: np.ndarray,
) -> dict[str, Any]:
    groups = _rank_attribution(
        frame=frame,
        baseline_rank=baseline_rank,
        changed_rank=changed_rank,
        group_values=np.asarray(mask, dtype=np.uint8),
        group_names={1: "selected"},
    )
    return groups["selected"]


def _history_age_audit(
    *,
    frame: pd.DataFrame,
    pair: dict[str, np.ndarray],
    p32_rank: np.ndarray,
    p33_rank: np.ndarray,
) -> dict[str, Any]:
    coarse_rank = frame["coarse_rank"].to_numpy(dtype=np.int32)
    age_code = _age_codes(pair["history_age"])
    target = frame["target"].to_numpy(dtype=bool)
    result: dict[str, Any] = {
        "history_age_by_relation": {},
        "history_age_by_relation_family": {},
        "direct_relation_freshness": {},
    }
    for age_index, age_name in enumerate(AGE_BUCKETS):
        result["history_age_by_relation"][age_name] = {}
        for label_code, label_name in enumerate(LABELS):
            observation = pair["valid_history"] & (age_code == age_index) & (pair["relation_label"] == label_code)
            candidate_mask = observation.any(axis=1)
            result["history_age_by_relation"][age_name][label_name] = {
                "candidate_history_observations": int(observation.sum()),
                "candidate_rows": int(candidate_mask.sum()),
                "truth_candidate_rows": int(np.count_nonzero(candidate_mask & target)),
                "raw_content_similarity": _summary(pair["raw_content"][observation]),
                "behavior_affinity_any": _summary(pair["behavior_any"][observation]),
                "p32_rank_movement": _mask_rank_metrics(frame, candidate_mask, coarse_rank, p32_rank),
                "p33_rank_movement": _mask_rank_metrics(frame, candidate_mask, coarse_rank, p33_rank),
            }
        result["history_age_by_relation_family"][age_name] = {}
        for family_name, family_codes in (
            ("similarity_like", [LABEL_TO_CODE["similarity_like"]]),
            (
                "complement_like",
                [LABEL_TO_CODE["direct_complement_like"], LABEL_TO_CODE["multihop_complement_like"]],
            ),
        ):
            observation = pair["valid_history"] & (age_code == age_index) & np.isin(
                pair["relation_label"], family_codes
            )
            candidate_mask = observation.any(axis=1)
            result["history_age_by_relation_family"][age_name][family_name] = {
                "candidate_history_observations": int(observation.sum()),
                "candidate_rows": int(candidate_mask.sum()),
                "truth_candidate_rows": int(np.count_nonzero(candidate_mask & target)),
                "raw_content_similarity": _summary(pair["raw_content"][observation]),
                "behavior_affinity_any": _summary(pair["behavior_any"][observation]),
                "p32_rank_movement": _mask_rank_metrics(frame, candidate_mask, coarse_rank, p32_rank),
                "p33_rank_movement": _mask_rank_metrics(frame, candidate_mask, coarse_rank, p33_rank),
            }
    direct_label = pair["relation_label"] == LABEL_TO_CODE["direct_complement_like"]
    freshness_code = _age_codes(pair["direct_last_age"])
    for age_index, age_name in enumerate(AGE_BUCKETS):
        observation = pair["valid_history"] & direct_label & np.isfinite(pair["direct_last_age"]) & (freshness_code == age_index)
        candidate_mask = observation.any(axis=1)
        result["direct_relation_freshness"][age_name] = {
            "candidate_history_observations": int(observation.sum()),
            "candidate_rows": int(candidate_mask.sum()),
            "truth_candidate_rows": int(np.count_nonzero(candidate_mask & target)),
            "p32_rank_movement": _mask_rank_metrics(frame, candidate_mask, coarse_rank, p32_rank),
            "p33_rank_movement": _mask_rank_metrics(frame, candidate_mask, coarse_rank, p33_rank),
        }
    result["note"] = "candidate rows may occur in multiple relation-age cells; observations are conserved within each age bucket"
    return result


def _higher_order_audit(
    frame: pd.DataFrame,
    p33_rank: np.ndarray,
) -> dict[str, Any]:
    coarse_rank = frame["coarse_rank"].to_numpy(dtype=np.int32)
    return {
        "deepwalk_high_order_proxy_quartiles": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=p33_rank,
            group_values=frame["multihop_score_quartile"].to_numpy(),
            group_names=_quartile_names(),
        ),
        "direct_covisit_affinity_control_quartiles": _rank_attribution(
            frame=frame,
            baseline_rank=coarse_rank,
            changed_rank=p33_rank,
            group_values=frame["direct_score_quartile"].to_numpy(),
            group_names=_quartile_names(),
        ),
        "definition": "DeepWalk affinity across five content-selected warm proxies, maximized only over history observations with zero raw direct user-day support",
    }


def _category_pair_audit(frame: pd.DataFrame, pair: dict[str, np.ndarray]) -> dict[str, Any]:
    target = frame["target"].to_numpy(dtype=bool)
    target_pair = np.broadcast_to(target[:, None], pair["valid_history"].shape)
    result = {}
    for name in ("direct_complement_like", "multihop_complement_like"):
        selected = pair["valid_history"] & (pair["relation_label"] == LABEL_TO_CODE[name])
        values = pair["category_pair_pct"][selected]
        truth_values = pair["category_pair_pct"][selected & target_pair]
        result[name] = {
            "candidate_history_observations": int(selected.sum()),
            "observed_category_pair_association": _summary(values),
            "high_Q4_category_pair_association_share": (
                float(np.mean(values >= 0.75)) if np.isfinite(values).any() else None
            ),
            "truth_observations": int(np.count_nonzero(selected & target_pair)),
            "truth_high_Q4_category_pair_association_share": (
                float(np.mean(truth_values >= 0.75)) if np.isfinite(truth_values).any() else None
            ),
        }
    return result


def _teacher_relation_composition(
    *,
    teacher_dir: Path,
    teacher: TeacherAssets,
    fashion: np.ndarray,
    student: np.ndarray,
    product_type: np.ndarray,
    garment_group: np.ndarray,
) -> dict[str, Any]:
    rows_by_view: dict[str, dict[str, np.ndarray]] = {}
    all_content = []
    for view in ("item2vec", "direct_covisit", "deepwalk"):
        loaded = np.load(teacher_dir / f"neighbors_{view}.npz")
        anchors = np.repeat(np.asarray(loaded["anchors"], dtype=np.int32), 20)
        neighbors = np.asarray(loaded["neighbors"], dtype=np.int32)[:, :20].reshape(-1)
        scores = np.asarray(loaded["scores"], dtype=np.float32)[:, :20].reshape(-1)
        valid = neighbors >= 0
        anchors, neighbors, scores = anchors[valid], neighbors[valid], scores[valid]
        content_valid = np.isfinite(fashion[anchors]).all(axis=1) & np.isfinite(fashion[neighbors]).all(axis=1)
        raw_content = np.full(len(anchors), np.nan, dtype=np.float32)
        raw_content[content_valid] = np.einsum(
            "ij,ij->i", fashion[anchors[content_valid]], fashion[neighbors[content_valid]]
        )
        student_content = np.einsum(
            "ij,ij->i", np.asarray(student[anchors], dtype=np.float32), np.asarray(student[neighbors], dtype=np.float32)
        )
        recent = _sparse_values(teacher.direct_recent, anchors, neighbors).astype(np.int32)
        older = _sparse_values(teacher.direct_older, anchors, neighbors).astype(np.int32)
        rows_by_view[view] = {
            "anchor": anchors,
            "neighbor": neighbors,
            "teacher_score": scores,
            "raw_content": raw_content,
            "student_content": student_content,
            "recent": recent,
            "older": older,
        }
        all_content.append(raw_content[np.isfinite(raw_content)])
    content_q50 = float(np.quantile(np.concatenate(all_content), 0.5))
    output: dict[str, Any] = {
        "combined_raw_content_q50": content_q50,
        "views": {},
    }
    for view, data in rows_by_view.items():
        anchors = data["anchor"]
        neighbors = data["neighbor"]
        score_q75 = float(np.quantile(data["teacher_score"], 0.75))
        total = data["recent"] + data["older"]
        recent_share = np.full(len(total), np.nan, dtype=np.float32)
        observed = total > 0
        recent_share[observed] = data["recent"][observed] / total[observed]
        low_high = (data["raw_content"] <= content_q50) & (data["teacher_score"] >= score_q75)
        output["views"][view] = {
            "relations": int(len(anchors)),
            "same_product_type_rate": float(np.mean(product_type[anchors] == product_type[neighbors])),
            "cross_product_type_rate": float(np.mean(product_type[anchors] != product_type[neighbors])),
            "same_garment_group_rate": float(np.mean(garment_group[anchors] == garment_group[neighbors])),
            "raw_fashionclip_cosine": _summary(data["raw_content"]),
            "m4_student_cosine": _summary(data["student_content"]),
            "teacher_score": _summary(data["teacher_score"]),
            "direct_support": _summary(total),
            "recent_direct_support_share": float(np.mean(data["recent"] > 0)),
            "recent_fraction_when_observed": _summary(recent_share),
            "low_content_high_teacher_affinity_relations": int(np.count_nonzero(low_high)),
            "low_content_high_teacher_affinity_share": float(np.mean(low_high)),
            "high_teacher_affinity_threshold_q75": score_q75,
        }
    deep = output["views"]["deepwalk"]
    direct = output["views"]["direct_covisit"]
    i2v = output["views"]["item2vec"]
    output["deepwalk_has_more_cross_type_and_low_content_than_both_controls"] = bool(
        deep["cross_product_type_rate"] > max(direct["cross_product_type_rate"], i2v["cross_product_type_rate"])
        and deep["low_content_high_teacher_affinity_share"]
        > max(direct["low_content_high_teacher_affinity_share"], i2v["low_content_high_teacher_affinity_share"])
    )
    return output


def _event_counts_before_cutoff(
    connection: duckdb.DuckDBPyConnection,
    transactions: Path,
    catalog: pd.DataFrame,
    cutoff: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Count only events strictly before the outer cutoff."""
    _safe_cutoff(cutoff)
    connection.register("p36_count_catalog", catalog[["article_id", "catalog_row"]])
    try:
        frame = connection.execute(
            f"""
            SELECT c.catalog_row::INTEGER AS catalog_row,count(*)::INTEGER AS events
            FROM read_parquet('{_sql_path(transactions)}') t
            JOIN p36_count_catalog c ON CAST(t.article_id AS VARCHAR)=c.article_id
            WHERE t.t_dat<DATE '{cutoff}'
            GROUP BY c.catalog_row
            """
        ).fetch_df()
        latest = connection.execute(
            f"SELECT max(t_dat)::VARCHAR FROM read_parquet('{_sql_path(transactions)}') "
            f"WHERE t_dat<DATE '{cutoff}'"
        ).fetchone()[0]
    finally:
        connection.unregister("p36_count_catalog")
    counts = np.zeros(len(catalog), dtype=np.int32)
    counts[frame["catalog_row"].to_numpy(dtype=np.int32)] = frame["events"].to_numpy(dtype=np.int32)
    return counts, {
        "cutoff": cutoff,
        "latest_transaction_before_cutoff": latest,
        "cutoff_safe": bool(latest < cutoff),
        "catalog_rows_with_events": int(np.count_nonzero(counts)),
        "warm_rows_events_gt_5": int(np.count_nonzero(counts > 5)),
    }


def _prepare_relation_fields(
    *,
    pair: dict[str, np.ndarray],
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    product_type: np.ndarray,
    category_pair: dict[tuple[int, int], tuple[float, float]],
) -> dict[str, Any]:
    """Freeze label-free percentile thresholds, then attach relation proxies."""
    valid = pair["valid_history"]
    teacher_fields = {
        "i2v_pct": "i2v_max",
        "direct_pct": "direct_max",
        "deepwalk_pct": "deepwalk_max",
    }
    teacher_summaries: dict[str, Any] = {}
    for output_name, raw_name in teacher_fields.items():
        percentile = empirical_mid_percentile(np.where(valid, pair[raw_name], np.nan))
        pair[output_name] = percentile
        finite = percentile[np.isfinite(percentile)]
        if len(finite) and (float(finite.min()) < 0.0 or float(finite.max()) > 1.0):
            raise RuntimeError(f"teacher percentile outside [0,1]: {output_name}")
        teacher_summaries[output_name] = {
            "raw": _summary(pair[raw_name][valid]),
            "percentile": _summary(finite),
        }
    behavior = np.fmax(np.fmax(pair["i2v_pct"], pair["direct_pct"]), pair["deepwalk_pct"])
    behavior[~valid] = np.nan
    pair["behavior_any"] = behavior.astype(np.float32)

    raw_values = pair["raw_content"][valid & np.isfinite(pair["raw_content"])]
    behavior_values = behavior[valid & np.isfinite(behavior)]
    if not len(raw_values) or not len(behavior_values):
        raise RuntimeError("relation quartiles require finite content and behavioral observations")
    content_cuts = np.quantile(raw_values.astype(np.float64), [0.25, 0.5, 0.75])
    behavior_cuts = np.quantile(behavior_values.astype(np.float64), [0.25, 0.5, 0.75])
    content_q = np.full(valid.shape, 255, dtype=np.uint8)
    behavior_q = np.full(valid.shape, 255, dtype=np.uint8)
    content_eligible = valid & np.isfinite(pair["raw_content"])
    behavior_eligible = valid & np.isfinite(behavior)
    content_q[content_eligible] = np.searchsorted(
        content_cuts, pair["raw_content"][content_eligible], side="right"
    ).astype(np.uint8)
    behavior_q[behavior_eligible] = np.searchsorted(
        behavior_cuts, behavior[behavior_eligible], side="right"
    ).astype(np.uint8)
    pair["content_q"] = content_q
    pair["behavior_q"] = behavior_q
    pair["relation_label"] = relation_labels(
        pair["raw_content"],
        pair["same_product_type"],
        pair["direct_pct"],
        pair["deepwalk_pct"],
        behavior,
        pair["direct_support"],
        content_q50=float(content_cuts[1]),
        content_q75=float(content_cuts[2]),
    )

    users = np.asarray(candidates["user_index"], dtype=np.int32)
    items = np.asarray(candidates["catalog_row"], dtype=np.int32)
    history_rows = np.asarray(histories["catalog_row"], dtype=np.int32)[users]
    safe_history = np.maximum(history_rows, 0)
    lift, percentile = _category_pair_values(
        product_type[items][:, None], product_type[safe_history], category_pair
    )
    lift[~valid | pair["same_product_type"]] = np.nan
    percentile[~valid | pair["same_product_type"]] = np.nan
    pair["category_pair_lift"] = lift
    pair["category_pair_pct"] = percentile
    label_counts = {
        name: int(np.count_nonzero(valid & (pair["relation_label"] == code)))
        for code, name in enumerate(LABELS)
    }
    return {
        "normalization_scope": "all finite candidate-history observations in this outer window; truth labels unused",
        "teacher_scales": teacher_summaries,
        "raw_content_quartile_cuts": [float(value) for value in content_cuts],
        "behavior_affinity_quartile_cuts": [float(value) for value in behavior_cuts],
        "valid_history_observations": int(valid.sum()),
        "secondary_label_counts": label_counts,
        "secondary_label_conservation_passed": sum(label_counts.values()) == int(valid.sum()),
    }


def _write_parquet(connection: duckdb.DuckDBPyConnection, frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection.register("p36_output_frame", frame)
    try:
        connection.execute(
            f"COPY p36_output_frame TO '{_sql_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.unregister("p36_output_frame")
    return file_identity(path)


def _write_window_evidence(
    *,
    connection: duckdb.DuckDBPyConnection,
    config: Config,
    window: str,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    pair: dict[str, np.ndarray],
    candidate_frame: pd.DataFrame,
    proxies_by_catalog: np.ndarray,
    proxy_scores_by_catalog: np.ndarray,
    fallback_by_catalog: np.ndarray,
    catalog: pd.DataFrame,
) -> dict[str, Any]:
    """Persist ignored, audit-only evidence; no recommendation candidates are generated."""
    root = config.artifact_dir / window
    article_ids = catalog.sort_values("catalog_row")["article_id"].astype(str).to_numpy()
    unique_items = np.unique(np.asarray(candidates["catalog_row"], dtype=np.int32))
    repeated_items = np.repeat(unique_items, K_PROXY)
    proxy_rows = proxies_by_catalog[unique_items].reshape(-1)
    valid_proxy = proxy_rows >= 0
    proxy_map = pd.DataFrame(
        {
            "candidate_catalog_row": repeated_items.astype(np.int32),
            "candidate_article_id": article_ids[repeated_items],
            "proxy_rank": np.tile(np.arange(1, K_PROXY + 1, dtype=np.uint8), len(unique_items)),
            "proxy_catalog_row": proxy_rows.astype(np.int32),
            "proxy_article_id": np.where(valid_proxy, article_ids[np.maximum(proxy_rows, 0)], ""),
            "raw_fashionclip_cosine": proxy_scores_by_catalog[unique_items].reshape(-1).astype(np.float32),
            "same_product_type": (fallback_by_catalog[unique_items].reshape(-1) == 0).astype(np.uint8),
            "same_garment_group_fallback": (fallback_by_catalog[unique_items].reshape(-1) == 1).astype(np.uint8),
            "fallback_level": fallback_by_catalog[unique_items].reshape(-1).astype(np.uint8),
        }
    )
    proxy_identity = _write_parquet(connection, proxy_map, root / "warm_proxy_mapping.parquet")
    candidate_identity = _write_parquet(
        connection, candidate_frame, root / "candidate_relation_features.parquet"
    )

    users = np.asarray(candidates["user_index"], dtype=np.int32)
    items = np.asarray(candidates["catalog_row"], dtype=np.int32)
    target = np.asarray(candidates["target"], dtype=np.uint8)
    history_all = np.asarray(histories["catalog_row"], dtype=np.int32)
    part_identities: list[dict[str, Any]] = []
    observation_rows = 0
    chunk = 75_000
    for part_index, start in enumerate(range(0, len(items), chunk)):
        end = min(start + chunk, len(items))
        local_valid = pair["valid_history"][start:end]
        row_position, history_position = np.nonzero(local_valid)
        global_position = row_position + start
        local_users = users[global_position]
        local_history_rows = history_all[local_users, history_position]
        label_code = pair["relation_label"][global_position, history_position]
        detail = pd.DataFrame(
            {
                "user_index": local_users.astype(np.int32),
                "candidate_catalog_row": items[global_position].astype(np.int32),
                "history_catalog_row": local_history_rows.astype(np.int32),
                "history_position": history_position.astype(np.uint8),
                "target": target[global_position],
                "raw_fashionclip_cosine": pair["raw_content"][global_position, history_position],
                "m4_student_cosine": pair["student_content"][global_position, history_position],
                "same_product_type": pair["same_product_type"][global_position, history_position].astype(np.uint8),
                "same_garment_group": pair["same_garment_group"][global_position, history_position].astype(np.uint8),
                "same_department": pair["same_department"][global_position, history_position].astype(np.uint8),
                "history_age_days": pair["history_age"][global_position, history_position],
                "item2vec_max_raw": pair["i2v_max"][global_position, history_position],
                "item2vec_top3_mean_raw": pair["i2v_top3"][global_position, history_position],
                "deepwalk_max_raw": pair["deepwalk_max"][global_position, history_position],
                "deepwalk_top3_mean_raw": pair["deepwalk_top3"][global_position, history_position],
                "direct_covisit_max_raw": pair["direct_max"][global_position, history_position],
                "direct_recent_28d_support": pair["direct_recent"][global_position, history_position],
                "direct_older_29_84d_support": pair["direct_older"][global_position, history_position],
                "last_direct_observed_age_days": pair["direct_last_age"][global_position, history_position],
                "item2vec_affinity_percentile": pair["i2v_pct"][global_position, history_position],
                "deepwalk_affinity_percentile": pair["deepwalk_pct"][global_position, history_position],
                "direct_affinity_percentile": pair["direct_pct"][global_position, history_position],
                "behavior_affinity_any": pair["behavior_any"][global_position, history_position],
                "content_quartile_code": pair["content_q"][global_position, history_position],
                "behavior_quartile_code": pair["behavior_q"][global_position, history_position],
                "secondary_relation_label_code": label_code.astype(np.uint8),
                "category_pair_lift": pair["category_pair_lift"][global_position, history_position],
                "category_pair_percentile": pair["category_pair_pct"][global_position, history_position],
            }
        )
        path = root / "pair_relation_features" / f"part-{part_index:04d}.parquet"
        identity = _write_parquet(connection, detail, path)
        identity["rows"] = int(len(detail))
        part_identities.append(identity)
        observation_rows += len(detail)
    return {
        "window": window,
        "warm_proxy_mapping": {**proxy_identity, "rows": int(len(proxy_map))},
        "candidate_relation_features": {**candidate_identity, "rows": int(len(candidate_frame))},
        "pair_relation_features": {
            "format": "partitioned parquet",
            "rows": int(observation_rows),
            "parts": part_identities,
        },
        "label_codebook": {str(code): name for code, name in enumerate(LABELS)},
        "fallback_codebook": {
            "0": "same product_type_no content-nearest pool",
            "1": "same garment_group_no content-nearest fallback",
            "2": "global warm content-nearest fallback",
            "255": "no proxy",
        },
    }


def _verify_window_inputs(
    *,
    paths: dict[str, Any],
    p31_window: dict[str, Any],
    p32_window: dict[str, Any],
    p33: dict[str, Any],
    window: str,
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
) -> dict[str, Any]:
    for name in ("candidates", "histories", "users"):
        _verify_declared(paths[name], p31_window["artifacts"][name])
    _verify_declared(paths["m4_embedding"], p31_window["student_embedding"])
    _verify_declared(paths["p32_scores"], p32_window["scoring"]["scores"])
    _verify_declared(paths["p32_attention"], p32_window["scoring"]["attention"])
    p33_encoding = p33["students"]["assets"]["outer_validation"][window]["encoding"]["artifact"]
    _verify_declared(paths["p33_embedding"], p33_encoding)
    p31_identity = _validate_candidate_identity(candidates)
    p32_scores = np.load(paths["p32_scores"], mmap_mode="r")
    p32_attention = np.load(paths["p32_attention"], mmap_mode="r")
    if len(p32_scores) != p31_identity["rows"] or p32_attention.shape != (p31_identity["rows"], HISTORY_N):
        raise RuntimeError(f"P3.2 prediction identity mismatch at {window}")
    if np.asarray(histories["catalog_row"]).shape[1] != HISTORY_N:
        raise RuntimeError(f"history width mismatch at {window}")

    official_loaded = np.load(paths["p33_official_coarse"])
    official_candidates = {name: np.asarray(official_loaded[name]) for name in official_loaded.files}
    official_identity = _validate_candidate_identity(official_candidates)
    official_scores = np.load(paths["p33_official_scores"], mmap_mode="r")
    official_attention = np.load(paths["p33_official_attention"], mmap_mode="r")
    if len(official_scores) != official_identity["rows"] or official_attention.shape != (
        official_identity["rows"], HISTORY_N
    ):
        raise RuntimeError(f"P3.3 official prediction identity mismatch at {window}")
    return {
        "p31_candidate_identity": p31_identity,
        "p31_candidates": file_identity(paths["candidates"]),
        "p31_histories": file_identity(paths["histories"]),
        "p31_users": file_identity(paths["users"]),
        "m4_single_teacher_embedding": file_identity(paths["m4_embedding"]),
        "p32_scores": {**file_identity(paths["p32_scores"]), "rows": int(len(p32_scores))},
        "p32_attention": {**file_identity(paths["p32_attention"]), "shape": list(p32_attention.shape)},
        "p33_multiview_embedding": file_identity(paths["p33_embedding"]),
        "p33_official_candidate_identity": official_identity,
        "p33_official_candidates": file_identity(paths["p33_official_coarse"]),
        "p33_official_scores": {**file_identity(paths["p33_official_scores"]), "rows": int(len(official_scores))},
        "p33_official_attention": {
            **file_identity(paths["p33_official_attention"]),
            "shape": list(official_attention.shape),
        },
        "p33_fixed_universe_attribution_rule": (
            "P3.3 multiview Student seed scores are reconstructed on exact P3.1 Top200; "
            "official P3.3 predictions are identity-verified but not substituted because their candidate universe differs"
        ),
    }


def _positive_group(metric: dict[str, Any], minimum: int = 5, require_top20_positive: bool = False) -> bool:
    top20 = metric.get("top20_net_truth")
    median = metric.get("median_rank_delta")
    return bool(
        metric.get("truth_candidate_rows", 0) >= minimum
        and median is not None
        and median > 0
        and top20 is not None
        and (top20 > 0 if require_top20_positive else top20 >= 0)
    )


def _nonpositive_group(metric: dict[str, Any]) -> bool:
    median = metric.get("median_rank_delta")
    top20 = metric.get("top20_net_truth")
    return bool(median is not None and (median <= 0 or (top20 is not None and top20 <= 0)))


def _derive_verdicts(
    *,
    proxy_quality: dict[str, Any],
    p32: dict[str, Any],
    p33: dict[str, Any],
    higher_order: dict[str, Any],
    history_age: dict[str, Any],
    teacher_composition: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    minimum = 5
    coverage_pass = all(
        data["candidate_row_full_5_share"] >= 0.95 for data in proxy_quality["windows"].values()
    )
    p32_family = {
        window: {
            "similarity_like": data["relation_presence"]["similarity_like"],
            "complement_like": data["relation_presence"]["complement_like_any"],
        }
        for window, data in p32["windows"].items()
    }
    similarity_positive = {
        window: _positive_group(data["similarity_like"], minimum) for window, data in p32_family.items()
    }
    complement_positive = {
        window: _positive_group(data["complement_like"], minimum) for window, data in p32_family.items()
    }
    early = p32_family["early_summer_20200624"]
    early_s = early["similarity_like"]
    early_c = early["complement_like"]
    early_mode = "inconclusive"
    enough_early = min(early_s["truth_candidate_rows"], early_c["truth_candidate_rows"]) >= minimum
    if enough_early and similarity_positive["early_summer_20200624"] and complement_positive["early_summer_20200624"]:
        early_mode = "mixed"
    elif enough_early and early_s["median_rank_delta"] is not None and early_c["median_rank_delta"] is not None:
        if (
            early_s["median_rank_delta"] - early_c["median_rank_delta"] >= 5
            and early_s["top20_net_truth"] >= 0
            and any(value for key, value in similarity_positive.items() if key != "early_summer_20200624")
        ):
            early_mode = "similarity_dominant"
        elif (
            early_c["median_rank_delta"] - early_s["median_rank_delta"] >= 5
            and early_c["top20_net_truth"] >= 0
            and any(value for key, value in complement_positive.items() if key != "early_summer_20200624")
        ):
            early_mode = "complement_like_dominant"

    similarity_repeats = sum(similarity_positive.values())
    complement_repeats = sum(complement_positive.values())
    if similarity_repeats >= 2 and complement_repeats >= 2:
        cross_mode = "mixed"
    elif similarity_repeats >= 3 and complement_repeats < 3:
        cross_mode = "similarity_dominant"
    elif complement_repeats >= 3 and similarity_repeats < 3:
        cross_mode = "complement_like_dominant"
    else:
        cross_mode = "inconclusive"

    p33_family = {
        window: {
            "similarity_like": data["relation_presence"]["similarity_like"],
            "complement_like": data["relation_presence"]["complement_like_any"],
        }
        for window, data in p33["windows"].items()
    }
    positive_windows = ("winter_20200122", "late_summer_20200819")
    negative_windows = ("spring_20200318", "early_summer_20200624")
    comp_pos = {
        window: _positive_group(p33_family[window]["complement_like"], minimum)
        for window in positive_windows
    }
    comp_neg = {
        window: _nonpositive_group(p33_family[window]["complement_like"])
        for window in negative_windows
    }
    similar_pos = {
        window: _positive_group(p33_family[window]["similarity_like"], minimum)
        for window in positive_windows
    }
    complement_positive_adequate = all(
        p33_family[window]["complement_like"]["truth_candidate_rows"] >= minimum
        for window in positive_windows
    )
    if coverage_pass and complement_positive_adequate and all(comp_pos.values()) and all(comp_neg.values()):
        complement_value = "supported"
    elif complement_positive_adequate and not any(comp_pos.values()) and all(similar_pos.values()):
        complement_value = "rejected"
    else:
        complement_value = "inconclusive"

    deep_q4 = {
        window: data["deepwalk_high_order_proxy_quartiles"]["Q4"]
        for window, data in higher_order["windows"].items()
    }
    deep_pos = {
        window: _positive_group(deep_q4[window], minimum, require_top20_positive=True)
        for window in positive_windows
    }
    deep_neg = {window: _nonpositive_group(deep_q4[window]) for window in negative_windows}
    if coverage_pass and all(deep_pos.values()) and all(deep_neg.values()):
        higher_value = "supported"
    elif not any(deep_pos.values()):
        higher_value = "rejected"
    else:
        higher_value = "inconclusive"

    state_signs: dict[str, dict[str, list[str]]] = {
        family: {age: [] for age in AGE_BUCKETS} for family in ("similarity_like", "complement_like")
    }
    all_state_cells_adequate = True
    for window_data in history_age["windows"].values():
        for age in AGE_BUCKETS:
            for family in state_signs:
                metric = window_data["history_age_by_relation_family"][age][family]["p32_rank_movement"]
                adequate = metric["truth_candidate_rows"] >= minimum and metric["median_rank_delta"] is not None
                all_state_cells_adequate &= adequate
                if adequate:
                    state_signs[family][age].append("positive" if metric["median_rank_delta"] > 0 else "nonpositive")
    state_support = False
    for family, ages in state_signs.items():
        for age, signs in ages.items():
            for direction in ("positive", "nonpositive"):
                if signs.count(direction) < 3:
                    continue
                opposite = "nonpositive" if direction == "positive" else "positive"
                if any(other_signs.count(opposite) >= 2 for other_age, other_signs in ages.items() if other_age != age):
                    state_support = True
    if state_support:
        state_value = "supported"
    elif all_state_cells_adequate:
        state_value = "rejected"
    else:
        state_value = "inconclusive"

    verdicts = {
        "early_summer_candidate_aware_relation_mode": early_mode,
        "cross_window_candidate_aware_relation_mode": cross_mode,
        "multiview_complementarity_value": complement_value,
        "higher_order_deepwalk_value": higher_value,
        "relation_type_state_dependence": state_value,
    }
    evidence = {
        "proxy_full_5_coverage_gate_passed_all_windows": coverage_pass,
        "p32_similarity_positive_windows": similarity_positive,
        "p32_complement_like_positive_windows": complement_positive,
        "p32_positive_window_counts": {
            "similarity_like": similarity_repeats,
            "complement_like": complement_repeats,
        },
        "p33_complement_like_positive_window_checks": comp_pos,
        "p33_complement_like_positive_windows_have_minimum_evidence": complement_positive_adequate,
        "p33_complement_like_negative_window_checks": comp_neg,
        "p33_similarity_like_positive_window_checks": similar_pos,
        "deepwalk_Q4_positive_window_checks": deep_pos,
        "deepwalk_Q4_negative_window_checks": deep_neg,
        "state_relation_age_signs": state_signs,
        "teacher_deepwalk_structure_by_window": {
            window: data["deepwalk_has_more_cross_type_and_low_content_than_both_controls"]
            for window, data in teacher_composition["windows"].items()
        },
        "rule_note": (
            "verdict comparisons use the preregistered non-exclusive presence of at least one similarity-like "
            "or complement-like history relation, require at least five truth candidate rows in each compared group, "
            "and do not search a winner cell after labels"
        ),
    }
    return verdicts, evidence


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        return f"{float(value):.{digits}f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return lines


def _verdict_answer(key: str, value: str) -> str:
    explanations = {
        "early_summer_candidate_aware_relation_mode": {
            "similarity_dominant": "证据更接近相似商品重复兴趣，而且该方向在至少另一个窗口重复。",
            "complement_like_dominant": "证据更接近跨类别行为关联代理，而且该方向在至少另一个窗口重复。",
            "mixed": "相似关系代理和跨类别行为关联代理都出现合格的正向证据，不能归为单一路径。",
            "inconclusive": "现有正例数量、名次差或跨窗重复性不足，无法在相似、跨类别行为关联或混合三者中定性。",
        },
        "multiview_complementarity_value": {
            "supported": "多教师 Student 的正窗收益集中于跨类别行为关联代理，且负窗不呈相同方向。",
            "rejected": "两个正窗都没有跨类别行为关联代理增益，而内容相似关系可以解释两个正窗。",
            "inconclusive": "多教师的正负窗差异不能由跨类别行为关联代理稳定解释。",
        },
        "higher_order_deepwalk_value": {
            "supported": "直接共购为零而 DeepWalk 高亲和的 Q4 关系在两个正窗都改善，并在两个负窗减弱或反向。",
            "rejected": "两个正式正窗都没有观察到合格的 DeepWalk-only 高阶关系增益。",
            "inconclusive": "DeepWalk-only 高阶关系没有达到正负窗共同要求。",
        },
        "relation_type_state_dependence": {
            "supported": "同一关系类型×历史年龄方向在至少三个窗口重复，且另一年龄桶在至少两个窗口呈相反方向。",
            "rejected": "所有关系年龄单元证据量充分，但没有可重复方向。",
            "inconclusive": "部分关系年龄单元正例不足，或没有达到预注册的重复/对照要求。",
        },
    }
    return explanations.get(key, {}).get(value, value)


def render_report(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# P3.6 相似关系代理与互补关系代理机制审计",
        "",
        f"- 状态：`{result['status']}`。本阶段只读取冻结资产并做离线归因，没有训练模型、生成新推荐候选或调参。",
        f"- 运行标识：`{result['run_id']}`；耗时 `{result['runtime']['wall_seconds'] / 60.0:.2f}` 分钟。",
        "- 最终周 `2020-09-16`：`not_run`。四个外层窗口只使用各自截止日前的历史与教师资产。",
        "- 解释边界：下文的 similarity-like / complement-like 都是代理关系，不是已验证的替代品、搭配品或因果兼容关系。",
        "",
        "## 结论先行",
        "",
    ]
    verdict_rows = [
        ["P3.2 early-summer 关系模式", result["verdicts"]["early_summer_candidate_aware_relation_mode"]],
        ["P3.2 跨窗口关系模式", result["verdicts"]["cross_window_candidate_aware_relation_mode"]],
        ["P3.3 跨类别行为关系价值", result["verdicts"]["multiview_complementarity_value"]],
        ["DeepWalk-only 高阶关系价值", result["verdicts"]["higher_order_deepwalk_value"]],
        ["关系类型×状态依赖", result["verdicts"]["relation_type_state_dependence"]],
    ]
    lines.extend(_table(["问题", "机器可读结论"], verdict_rows))
    lines.extend([
        "",
        "这些结论严格使用预注册门槛：每个关系组至少5个正例候选；P3.2 主导性还要求中位名次优势至少5名、Top20不变差并在另一窗口重复。局部正数不等于通过。",
        "",
        "## 术语与统计口径",
        "",
        "| 名称 | 中文定义、单位与分母 |",
        "|---|---|",
        "| candidate row（候选行） | 一条 `(用户, 候选商品)` 记录；每名有历史的用户固定200行，沿用 P3.1 Top200，不是本阶段新召回。 |",
        "| truth candidate（正例候选） | 候选商品出现在截止日后7天真实购买集合中的候选行；只在代理、分位点和规则冻结后用于评价。 |",
        "| unobserved candidate（未观察候选） | 7天标签窗内没有购买记录的候选行；没有曝光日志，不能解释为用户明确拒绝。 |",
        "| warm proxy（暖商品代理） | 截止日前购买事件数大于5、具有 FashionCLIP 向量，并由静态内容最近邻选出的商品；每个冷/稀疏候选最多5个。 |",
        "| Item2Vec | 行业常用的商品行为序列嵌入：把用户购买序列中的商品当作词学习向量；本阶段用历史商品与暖代理的向量余弦。 |",
        "| direct co-vis（直接共购） | 本项目以同一用户、同一自然日的去重商品集合近似购物篮得到的直接共现关系；不是H&M真实订单边界。 |",
        "| DeepWalk | 行业常用的图随机游走嵌入：在直接共购图上采样多跳路径学习商品向量；本阶段只把它视为高阶行为关系代理。 |",
        "| M4 Student（M4学生表示） | 用 Item2Vec 行为教师蒸馏、推理时只读图片和静态属性的128维商品向量；它不是纯内容基线。 |",
        "| P3.2 candidate-aware attention（候选感知注意力） | P3.2排序器针对每条 `(用户,候选)` 对20件历史分别分配的权重；同一用户面对不同候选可关注不同历史。 |",
        "| product type / garment group | H&M目录中的商品细分类别编号 / 服装大类编号；本阶段用于同类判断和暖代理回退，不是模型预测标签。 |",
        "| raw FashionCLIP cosine（原始内容余弦） | 原始图片/文本内容向量余弦，作为主要纯内容参照；与已接受协同蒸馏的 M4 Student 余弦分开。 |",
        "| behavior affinity（行为亲和度） | Item2Vec、直接共购、DeepWalk 三路原始分数分别在当前窗口转成经验百分位后取最大值；仅用于审计，不把不同原始尺度直接相加。 |",
        "| similarity-like（相似关系代理） | 同 product type 且原始内容余弦不低于本窗Q75；本项目自定义代理名。 |",
        "| direct complement-like（直接跨类别行为关系代理） | 跨 product type、原始内容余弦不高于Q50且直接共购亲和度百分位不低于0.75；不代表真实互补。 |",
        "| multihop complement-like（多跳跨类别行为关系代理） | 跨 product type、低内容相似、DeepWalk百分位不低于0.75且五个暖代理与该历史商品的直接共购支持为0；不代表因果多跳关系。 |",
        "| rank delta（名次变化） | 原排序名次减新排序名次，单位为名次；正数表示新方法把该正例上移。 |",
        "| Top20 net truth（Top20净正例） | 同一关系组在新排序Top20中的正例数减原排序Top20正例数，单位为用户—商品正例对。 |",
        "| density@20 contribution（Top20密度贡献） | Top20净正例除以 `活动用户数×20`；只表示固定候选池内的加性贡献。 |",
        "| attention mass（注意力质量） | P3.2 对候选的20件历史注意力中，落在某关系代理上的权重和；每条具有有效历史的候选行总质量约为1。 |",
        "| empirical percentile（经验百分位） | 当前窗口所有有限候选—历史观察上按中秩计算的0到1位置；边界不读取正例标签。 |",
        "| category-pair lift（品类对提升度） | 跨 product type 用户日共现份额除以按端点边际流行度独立假设得到的期望份额；1为独立期望。 |",
        "",
        "## 冻结设计与无环归因",
        "",
        "冷/稀疏候选本身通常没有可靠图节点，所以本阶段不再要求它直接进入教师 Top20。先仅用原始 FashionCLIP 与静态品类，把候选映射到5个截止日安全的暖商品代理；选择顺序固定为同 product type、同 garment group、全局暖目录。只有映射完成后，才查询历史商品与这些代理之间的 Item2Vec、直接共购和 DeepWalk 关系。代理选择没有读取教师分数、P3.2分数/注意力或未来正例，因此避免了循环归因。",
        "",
        "P3.3归因使用完全相同的 P3.1 Top200：分别以 M4 单教师 Student 与 P3.3 多教师 Student 重建冻结的‘历史种子相似度×时间衰减’分数。P3.3正式端到端预测另做SHA与行数核验，但不拿它替换固定候选池，因为正式P3.3候选身份与P3.1并不相同。",
        "",
        "## 暖商品代理质量",
        "",
    ])
    rows = []
    for window, data in result["proxy_quality"]["windows"].items():
        rows.append([
            window, data["candidate_rows"], data["strict_cold_candidate_rows"],
            data["sparse_1_5_candidate_rows"], data["candidate_rows_with_at_least_one_proxy"],
            data["candidate_rows_with_full_5_proxies"], data["candidate_row_full_5_share"],
            data["same_product_type_proxy_share"], data["garment_fallback_share"],
            data["global_fallback_share"], data["proxy_cosine_all_assignments"]["median"],
        ])
    lines.extend(_table(
        ["window", "候选行", "strict-cold行", "sparse1-5行", ">=1代理", "完整5代理", "完整率", "同类型份额", "garment回退", "全局回退", "代理余弦中位数"], rows
    ))
    lines.extend(["", "正例与未观察候选的代理质量分布：", ""])
    rows = []
    for window, data in result["proxy_quality"]["windows"].items():
        for population, field in (
            ("truth", "truth_candidate_proxy_mean_cosine"),
            ("unobserved", "unobserved_candidate_proxy_mean_cosine"),
        ):
            summary = data[field]
            rows.append([window, population, summary["count"], summary["mean"], summary["median"], summary["p25"], summary["p75"]])
    lines.extend(_table(["window", "候选集合", "行数", "mean", "median", "p25", "p75"], rows))

    lines.extend(["", "## 4×4×2 原始内容—行为亲和矩阵", ""])
    lines.append("每个单元格的 observation 是一条候选—历史商品关系；同一候选最多进入20个单元格，所以 candidate rows 不能跨单元格相加。矩阵的观察数加显式不合格观察必须等于全部有效历史观察。")
    lines.append("")
    rows = []
    for window, matrix in result["relation_matrix"]["windows"].items():
        for cell, data in matrix["cells"].items():
            rows.append([window, cell, data["candidate_history_observations"], data["candidate_rows"], data["truth_candidate_rows"], data["unobserved_candidate_rows"], data["truth_rate"], data["truth_rate_lift_vs_all_valid_history_candidates"], data["history_age_days"]["median"]])
    lines.extend(_table(["window", "矩阵单元", "关系观察", "候选行", "正例候选", "未观察候选", "正例率", "相对总体提升", "历史年龄中位天数"], rows))
    lines.extend(["", "矩阵守恒与关系代理计数：", ""])
    rows = []
    for window, matrix in result["relation_matrix"]["windows"].items():
        labels = result["thresholds_and_labels"]["windows"][window]["secondary_label_counts"]
        rows.append([window, matrix["valid_history_observations"], matrix["eligible_observations"], matrix["ineligible_observations"], matrix["observation_conservation_passed"], *[labels[name] for name in LABELS]])
    lines.extend(_table(["window", "有效观察", "矩阵合格", "矩阵不合格", "守恒", *LABELS], rows))

    lines.extend(["", "## P3.2 候选感知排序归因", ""])
    lines.append("下表以每个候选的预注册 best relation family 分组。该名称表示20条历史关系中计数最多的关系代理族；计数并列时比较最大关系强度，再按冻结优先级，不读取正例标签。")
    lines.append("")
    rows = []
    for window, data in result["p32_relation_attribution"]["windows"].items():
        for family, metric in data["best_relation_family"].items():
            rows.append([window, P32_SIGNS[window], family, metric["truth_candidate_rows"], metric["truth_moved_up"], metric["truth_same_rank"], metric["truth_moved_down"], metric["median_rank_delta"], metric["mean_rank_delta"], metric["top200_to_top50_after"], metric["top200_to_top20_after"], metric["top200_to_top10_after"], metric["top20_net_truth"], metric["density20_additive_contribution"], metric["reciprocal_rank_pair_proxy_delta_mean"]])
    lines.extend(_table(["window", "正式方向", "关系族", "truth", "up", "same", "down", "median Δrank", "mean Δrank", "Top50后", "Top20后", "Top10后", "Top20净正例", "density@20贡献", "倒数名次代理增量"], rows))
    lines.extend(["", "机器结论使用的非互斥关系存在性归因：只要候选与20件历史中至少有一条对应关系就进入该组；同一候选可以同时进入相似组和跨类别行为关系组，因此两组不能相加。", ""])
    rows = []
    for window, data in result["p32_relation_attribution"]["windows"].items():
        for name in ("similarity_like", "complement_like_any"):
            metric = data["relation_presence"][name]
            rows.append([window, name, metric["truth_candidate_rows"], metric["truth_moved_up"], metric["truth_moved_down"], metric["median_rank_delta"], metric["top20_net_truth"]])
    lines.extend(_table(["window", "至少存在一条关系", "truth", "up", "down", "median Δrank", "Top20净正例"], rows))
    lines.extend(["", "完整32个注意力主导矩阵单元归因：", ""])
    rows = []
    for window, data in result["p32_relation_attribution"]["windows"].items():
        for cell, metric in data["primary_attention_dominant_matrix_cell"].items():
            rows.append([window, cell, metric["truth_candidate_rows"], metric["truth_moved_up"], metric["truth_moved_down"], metric["median_rank_delta"], metric["top20_net_truth"]])
    lines.extend(_table(["window", "注意力主导矩阵单元", "truth", "up", "down", "median Δrank", "Top20净正例"], rows))

    lines.extend(["", "P3.2实际注意力在正例与未观察候选上的关系质量：", ""])
    rows = []
    for window, data in result["attention_relation_audit"]["windows"].items():
        for population in ("truth", "unobserved"):
            rows.append([window, population, data["attention_weighted_content"][population]["mean"], data["attention_weighted_behavior"][population]["mean"], *[data["relation_type_mass"][population][label]["mean"] for label in LABELS]])
    lines.extend(_table(["window", "集合", "加权原始内容", "加权行为亲和", *[f"质量:{label}" for label in LABELS]], rows))

    lines.extend(["", "## P3.3 多教师 Student 固定候选池归因", ""])
    lines.append("这里的正式方向来自 P3.3 端到端结果；分组名次变化只比较同一 P3.1 Top200 上的单教师与多教师表示，因此是表示层归因探针，不是重放正式P3.3端到端指标。")
    lines.append("")
    rows = []
    for window, data in result["p33_relation_attribution"]["windows"].items():
        for family, metric in data["best_relation_family"].items():
            rows.append([window, P33_SIGNS[window], family, metric["truth_candidate_rows"], metric["truth_moved_up"], metric["truth_same_rank"], metric["truth_moved_down"], metric["median_rank_delta"], metric["mean_rank_delta"], metric["top20_net_truth"], metric["density20_additive_contribution"]])
    lines.extend(_table(["window", "P3.3正式方向", "关系族", "truth", "up", "same", "down", "median Δrank", "mean Δrank", "Top20净正例", "density@20贡献"], rows))
    lines.extend(["", "P3.3非互斥关系存在性归因（也是跨类别行为价值结论的证据口径）：", ""])
    rows = []
    for window, data in result["p33_relation_attribution"]["windows"].items():
        for name in ("similarity_like", "direct_complement_like", "multihop_complement_like", "complement_like_any"):
            metric = data["relation_presence"][name]
            rows.append([window, name, metric["truth_candidate_rows"], metric["truth_moved_up"], metric["truth_moved_down"], metric["median_rank_delta"], metric["top20_net_truth"]])
    lines.extend(_table(["window", "至少存在一条关系", "truth", "up", "down", "median Δrank", "Top20净正例"], rows))
    lines.extend(["", "内容与行为亲和四分位归因：", ""])
    rows = []
    for window, data in result["p33_relation_attribution"]["windows"].items():
        for dimension in ("candidate_max_content_quartile", "candidate_max_behavior_quartile"):
            for quartile, metric in data[dimension].items():
                rows.append([window, dimension, quartile, metric["truth_candidate_rows"], metric["median_rank_delta"], metric["top20_net_truth"]])
    lines.extend(_table(["window", "分组维度", "桶", "truth", "median Δrank", "Top20净正例"], rows))

    lines.extend(["", "## DeepWalk-only 高阶关系代理", ""])
    rows = []
    for window, data in result["deepwalk_higher_order_audit"]["windows"].items():
        for control, title in (("deepwalk_high_order_proxy_quartiles", "DeepWalk且直接支持为0"), ("direct_covisit_affinity_control_quartiles", "直接共购对照")):
            for quartile, metric in data[control].items():
                rows.append([window, P33_SIGNS[window], title, quartile, metric["truth_candidate_rows"], metric["truth_moved_up"], metric["truth_moved_down"], metric["median_rank_delta"], metric["top20_net_truth"], metric["density20_additive_contribution"]])
    lines.extend(_table(["window", "正式方向", "关系", "桶", "truth", "up", "down", "median Δrank", "Top20净正例", "density贡献"], rows))

    lines.extend(["", "## 多教师训练关系构成", ""])
    lines.append("每窗使用实际训练该外层 Student 的两个教师快照中较晚的一个；每路取教师Top20关系。low-content + high-teacher-affinity 表示原始内容余弦不高于三路合并Q50且该路教师分数不低于本路Q75。")
    lines.append("")
    rows = []
    for window, data in result["teacher_relation_composition"]["windows"].items():
        for view, metric in data["views"].items():
            rows.append([window, view, metric["relations"], metric["same_product_type_rate"], metric["cross_product_type_rate"], metric["same_garment_group_rate"], metric["raw_fashionclip_cosine"]["median"], metric["m4_student_cosine"]["median"], metric["direct_support"]["mean"], metric["recent_direct_support_share"], metric["low_content_high_teacher_affinity_share"]])
    lines.extend(_table(["window", "teacher view", "关系数", "同类型率", "跨类型率", "同garment率", "原始内容中位", "Student中位", "直接支持均值", "有近28日支持率", "低内容高教师份额"], rows))

    lines.extend(["", "## 跨品类边际流行度校正", ""])
    rows = []
    for window, data in result["category_pair_audit"]["windows"].items():
        for label, metric in data.items():
            rows.append([window, label, metric["candidate_history_observations"], metric["observed_category_pair_association"]["median"], metric["high_Q4_category_pair_association_share"], metric["truth_observations"], metric["truth_high_Q4_category_pair_association_share"]])
    lines.extend(_table(["window", "关系代理", "观察数", "品类对lift中位", "落在品类对Q4份额", "正例观察", "正例落Q4份额"], rows))

    lines.extend(["", "## 历史年龄与关系类型交互", ""])
    lines.append("同一候选只要在某年龄桶中至少有一条对应关系就计入该桶，因此候选行可跨年龄桶重复；每个单元的关系观察数独立给出。四桶全部预注册并完整输出，没有事后挑最佳桶。")
    lines.append("")
    rows = []
    for window, data in result["history_age_relation_audit"]["windows"].items():
        for age, families in data["history_age_by_relation_family"].items():
            for family, metric in families.items():
                p32m, p33m = metric["p32_rank_movement"], metric["p33_rank_movement"]
                rows.append([window, age, family, metric["candidate_history_observations"], metric["truth_candidate_rows"], metric["raw_content_similarity"]["median"], metric["behavior_affinity_any"]["median"], p32m["median_rank_delta"], p32m["top20_net_truth"], p33m["median_rank_delta"], p33m["top20_net_truth"]])
    lines.extend(_table(["window", "history age", "关系族", "关系观察", "truth候选", "内容中位", "行为中位", "P3.2 median Δrank", "P3.2 Top20净", "P3.3 median Δrank", "P3.3 Top20净"], rows))

    lines.extend(["", "## 最终问题回答", ""])
    early_value = result["verdicts"]["early_summer_candidate_aware_relation_mode"]
    p33_value = result["verdicts"]["multiview_complementarity_value"]
    deep_value = result["verdicts"]["higher_order_deepwalk_value"]
    state_value = result["verdicts"]["relation_type_state_dependence"]
    lines.extend([
        f"1. **P3.2 early-summer 更像什么？** `{early_value}`。{_verdict_answer('early_summer_candidate_aware_relation_mode', early_value)}",
        f"2. **P3.3正收益集中在哪里？** 跨类别行为关系代理结论为 `{p33_value}`。{_verdict_answer('multiview_complementarity_value', p33_value)} 分组表同时保留内容四分位、直接关系和DeepWalk-only关系，不能只凭均值挑解释。",
        f"3. **DeepWalk是否提供低内容相似、高行为亲和的额外结构？** 教师结构检查见上表；对未来正例的机器结论为 `{deep_value}`。{_verdict_answer('higher_order_deepwalk_value', deep_value)} 结构占比提高与推荐收益是两个不同问题。",
        f"4. **关系价值是否与历史年龄/新鲜度有可重复交互？** `{state_value}`。{_verdict_answer('relation_type_state_dependence', state_value)} 该机器结论预注册地使用 P3.2 名次变化；P3.3 名次变化完整列出但不参与此项门槛，避免把两种不同机制混成一个符号。",
        "",
        "## 下一阶段边界与回退",
        "",
    ])
    if early_value in ("similarity_dominant", "mixed") or result["verdicts"]["cross_window_candidate_aware_relation_mode"] in ("similarity_dominant", "mixed"):
        lines.append("- 相似兴趣头：有资格进入后续方案讨论，但本阶段没有训练。进入前仍需预注册成本、跨窗门槛和失败回退。")
    else:
        lines.append("- 相似兴趣头：当前证据不足，不因单窗局部收益直接启动。")
    if p33_value == "supported" or deep_value == "supported":
        lines.append("- 跨类别/图条件兴趣头或选择性多教师：有资格进入后续方案讨论；不得沿用固定等权并事后调同一四窗。")
    else:
        lines.append("- 跨类别/图条件兴趣头与选择性多教师：当前未获稳定支持，暂不启动高成本模型。")
    if state_value == "supported":
        lines.append("- 下一步若使用关系混合，应优先连续时间/用户状态条件，而不是按日历季节硬路由。")
    else:
        lines.append("- 时间/状态关系混合：P3.5一般时间信号仍存在，但本次关系类型交互没有单独达到门槛，需保留为假设而非结论。")
    lines.extend([
        "- 失败回退始终是冻结的 M4 单教师 Student、P3.1候选和既有 P3.2/P3.3 失败证据；本阶段没有产生新 baseline。",
        "",
        "## 工程守恒与证据索引",
        "",
        f"- 所有守恒检查：`{result['audit_checks']['all_passed']}`。P3.1候选、P3.2预测/注意力、P3.3预测/表示和 FashionCLIP 均做身份核验。",
        f"- 暖代理完整覆盖门槛（每窗完整5个代理比例至少95%）：`{result['verdict_evidence']['proxy_full_5_coverage_gate_passed_all_windows']}`。",
        "- 大型暖代理映射、候选级特征和候选—历史关系明细位于 Git ignored 的 `artifacts/phase3/p3-6-v1-similarity-complementarity-audit/`，逐文件SHA与行数写入 `p3_6_candidate_relation_features.json`。",
        "- 机器可读主证据与11个分项JSON、输出清单均位于 `reports/phase3/`；最终周保持 `not_run`。",
    ])
    return "\n".join(lines) + "\n"


def run_audit(config: Config) -> dict[str, Any]:
    started = time.perf_counter()
    contract_path = config.output_dir / "P3_6_EXPERIMENT_CONTRACT.json"
    if not contract_path.is_file():
        raise FileNotFoundError("P3.6 experiment contract must exist before formal computation")
    contract = _json(contract_path)
    if contract.get("status") != "preregistered_before_formal_computation":
        raise RuntimeError("P3.6 contract is not preregistered")
    for _window, _short, cutoff in WINDOWS:
        _safe_cutoff(cutoff)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("P3.6 requested CUDA but CUDA is unavailable")

    required_reports = [
        config.repo_root / "docs" / "ROADMAP_PHASE3.zh-CN.md",
        config.repo_root / "reports" / "phase3" / "P3_1_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_2_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_3_FINAL.md",
        config.repo_root / "reports" / "phase3" / "P3_4_REGIME_AUDIT.md",
        config.repo_root / "reports" / "phase3" / "P3_5_FINE_GRAINED_AUDIT.md",
        config.repo_root / "reports" / "phase3" / "P3_5_FINE_GRAINED_AUDIT.json",
        config.repo_root / "reports" / "phase3" / "P3_5_EXPERIMENT_CONTRACT.json",
        config.repo_root / "reports" / "m4" / "M4_3_FINAL.md",
        config.repo_root / "reports" / "m3_7" / "M3_7_FINAL.md",
    ]
    missing_reports = [str(path) for path in required_reports if not path.is_file()]
    if missing_reports:
        raise FileNotFoundError(f"missing required P3.6 historical evidence: {missing_reports}")
    historical_before = {str(path.relative_to(config.repo_root)): _sha(path) for path in required_reports}

    p31 = _json(config.output_dir / "P3_1_metrics.json")
    p32 = _json(config.output_dir / "P3_2_metrics.json")
    p33 = _json(config.output_dir / "P3_3_metrics.json")
    p35 = _json(config.output_dir / "P3_5_FINE_GRAINED_AUDIT.json")
    if any(value.get("final_week") != "not_run" for value in (p31, p32, p33, p35)):
        raise RuntimeError("a frozen Phase 3 input violated final-week boundary")

    catalog, fashion, image_rows, fashion_identity = _load_catalog_and_fashion(config)
    product_type = catalog["product_type_no"].to_numpy(dtype=np.int32)
    garment_group = catalog["garment_group_no"].to_numpy(dtype=np.int32)
    department = catalog["department_no"].to_numpy(dtype=np.int32)
    transactions = config.source_root / "data" / "interim" / "audit" / "transactions.parquet"
    if not transactions.is_file():
        raise FileNotFoundError(transactions)

    proxy_windows: dict[str, Any] = {}
    relation_windows: dict[str, Any] = {}
    threshold_windows: dict[str, Any] = {}
    candidate_manifest_windows: dict[str, Any] = {}
    p32_windows: dict[str, Any] = {}
    p33_windows: dict[str, Any] = {}
    attention_windows: dict[str, Any] = {}
    teacher_windows: dict[str, Any] = {}
    higher_windows: dict[str, Any] = {}
    age_windows: dict[str, Any] = {}
    category_windows: dict[str, Any] = {}
    identity_windows: dict[str, Any] = {}
    window_audits: dict[str, Any] = {}

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{config.memory_limit}'")
    connection.execute(f"SET threads={config.threads}")
    try:
        for window, _short, cutoff in WINDOWS:
            window_started = time.perf_counter()
            print(f"P3.6 {window}: verify and load frozen assets", flush=True)
            paths = _window_paths(config, window, p31, p32, p33)
            required_paths = [
                paths["candidates"], paths["histories"], paths["users"], paths["m4_embedding"],
                paths["p32_scores"], paths["p32_attention"], paths["p33_embedding"],
                paths["p33_student_manifest"], paths["p33_teacher_dir"] / "manifest.json",
                paths["p33_official_coarse"], paths["p33_official_scores"], paths["p33_official_attention"],
            ]
            missing = [str(path) for path in required_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(f"missing frozen P3.6 inputs for {window}: {missing}")
            candidates_loaded = np.load(paths["candidates"])
            histories_loaded = np.load(paths["histories"])
            candidates = {name: np.asarray(candidates_loaded[name]) for name in candidates_loaded.files}
            histories = {name: np.asarray(histories_loaded[name]) for name in histories_loaded.files}
            identity_windows[window] = _verify_window_inputs(
                paths=paths,
                p31_window=p31["windows"][window],
                p32_window=p32["windows"][window],
                p33=p33,
                window=window,
                candidates=candidates,
                histories=histories,
            )
            p32_scores = np.load(paths["p32_scores"], mmap_mode="r")
            attention = np.load(paths["p32_attention"], mmap_mode="r")
            m4_student = np.load(paths["m4_embedding"], mmap_mode="r")
            p33_student = np.load(paths["p33_embedding"], mmap_mode="r")
            counts, count_audit = _event_counts_before_cutoff(
                connection, transactions, catalog, cutoff
            )
            unique_items = np.unique(np.asarray(candidates["catalog_row"], dtype=np.int32))
            fashion_valid = np.isfinite(fashion).all(axis=1)
            candidate_valid = unique_items[fashion_valid[unique_items]]
            warm_rows = np.flatnonzero((counts > 5) & fashion_valid).astype(np.int32)
            if np.intersect1d(unique_items, warm_rows).size:
                raise RuntimeError(f"cold/sparse candidate entered warm proxy universe at {window}")
            print(
                f"P3.6 {window}: content-only warm proxies for {len(candidate_valid)} candidate items",
                flush=True,
            )
            local_proxy, local_proxy_score, local_fallback = select_warm_proxies(
                candidate_valid,
                warm_rows,
                fashion,
                product_type,
                garment_group,
                k=K_PROXY,
                device_name=config.device,
            )
            proxies_by_catalog = np.full((len(catalog), K_PROXY), -1, dtype=np.int32)
            proxy_scores_by_catalog = np.full((len(catalog), K_PROXY), np.nan, dtype=np.float32)
            fallback_by_catalog = np.full((len(catalog), K_PROXY), 255, dtype=np.uint8)
            proxies_by_catalog[candidate_valid] = local_proxy
            proxy_scores_by_catalog[candidate_valid] = local_proxy_score
            fallback_by_catalog[candidate_valid] = local_fallback
            if np.any((proxies_by_catalog[unique_items] >= 0) & (counts[proxies_by_catalog[unique_items].clip(min=0)] <= 5)):
                raise RuntimeError(f"non-warm proxy selected at {window}")

            teacher = _load_teacher_assets(
                connection,
                transactions,
                catalog,
                paths["p33_teacher_dir"],
                paths["p33_teacher_cutoff"],
            )
            print(f"P3.6 {window}: candidate-history-proxy relation features", flush=True)
            pair, pair_audit = _compute_pair_features(
                candidates=candidates,
                histories=histories,
                proxies_by_catalog=proxies_by_catalog,
                fashion=fashion,
                student=m4_student,
                product_type=product_type,
                garment_group=garment_group,
                department=department,
                teacher=teacher,
                device_name=config.device,
                user_batch=config.user_batch,
            )
            thresholds = _prepare_relation_fields(
                pair=pair,
                candidates=candidates,
                histories=histories,
                product_type=product_type,
                category_pair=teacher.category_pair,
            )
            frame, aggregate_audit = _aggregate_candidate_features(
                candidates=candidates,
                histories=histories,
                pair=pair,
                attention=attention,
                product_type=product_type,
                category_pair=teacher.category_pair,
            )
            matrix = relation_matrix(
                content_codes=pair["content_q"],
                behavior_codes=pair["behavior_q"],
                same_type=pair["same_product_type"],
                valid_history=pair["valid_history"],
                target=np.asarray(candidates["target"], dtype=bool),
                history_age=pair["history_age"],
                raw_content=pair["raw_content"],
                behavior_any=pair["behavior_any"],
            )
            p32_attribution, p32_rank = _p32_attribution(frame, p32_scores)
            p33_attribution, p33_rank = _p33_attribution(
                frame, candidates, histories, p33_student
            )
            proxy_quality = _proxy_quality(
                candidates=candidates,
                counts_by_row=counts,
                proxies_by_catalog=proxies_by_catalog,
                proxy_scores_by_catalog=proxy_scores_by_catalog,
                fallback_by_catalog=fallback_by_catalog,
            )
            attention_audit = _attention_relation_audit(frame)
            higher = _higher_order_audit(frame, p33_rank)
            age = _history_age_audit(
                frame=frame, pair=pair, p32_rank=p32_rank, p33_rank=p33_rank
            )
            category = _category_pair_audit(frame, pair)
            teacher_composition = _teacher_relation_composition(
                teacher_dir=paths["p33_teacher_dir"],
                teacher=teacher,
                fashion=fashion,
                student=m4_student,
                product_type=product_type,
                garment_group=garment_group,
            )
            print(f"P3.6 {window}: persist ignored audit evidence", flush=True)
            artifact_manifest = _write_window_evidence(
                connection=connection,
                config=config,
                window=window,
                candidates=candidates,
                histories=histories,
                pair=pair,
                candidate_frame=frame,
                proxies_by_catalog=proxies_by_catalog,
                proxy_scores_by_catalog=proxy_scores_by_catalog,
                fallback_by_catalog=fallback_by_catalog,
                catalog=catalog,
            )
            proxy_windows[window] = proxy_quality
            relation_windows[window] = matrix
            threshold_windows[window] = thresholds
            candidate_manifest_windows[window] = artifact_manifest
            p32_windows[window] = p32_attribution
            p33_windows[window] = p33_attribution
            attention_windows[window] = attention_audit
            teacher_windows[window] = teacher_composition
            higher_windows[window] = higher
            age_windows[window] = age
            category_windows[window] = category
            history_mask = np.asarray(histories["mask"], dtype=bool)
            history_days = np.asarray(histories["days_since_purchase"], dtype=np.float32)
            window_audits[window] = {
                "cutoff": cutoff,
                "event_count_audit": count_audit,
                "teacher_cutoff": teacher.cutoff,
                "teacher_cutoff_audit": teacher.cutoff_audit,
                "teacher_assets": teacher.identities,
                "history_rows": int(history_mask.sum()),
                "history_days_strictly_positive": bool(np.all(history_days[history_mask] > 0)),
                "proxy_selection_inputs": ["raw FashionCLIP", "product_type_no", "garment_group_no", "pre-cutoff event count warmness"],
                "proxy_selection_excludes_truth_and_teachers": True,
                "warm_proxy_count_rule_cutoff_safe": count_audit["cutoff_safe"],
                "all_selected_proxies_warm": proxy_quality["all_selected_proxies_are_warm_before_outer_cutoff"],
                "pair_feature_audit": pair_audit,
                "candidate_aggregation_audit": aggregate_audit,
                "matrix_conservation": matrix["observation_conservation_passed"],
                "label_conservation": thresholds["secondary_label_conservation_passed"],
                "runtime_seconds": time.perf_counter() - window_started,
            }
            del (
                pair, frame, p32_rank, p33_rank, teacher, m4_student, p33_student,
                attention, p32_scores, proxies_by_catalog, proxy_scores_by_catalog,
                fallback_by_catalog, local_proxy, local_proxy_score, local_fallback,
            )
            print(
                f"P3.6 {window}: completed in {window_audits[window]['runtime_seconds'] / 60.0:.2f} min",
                flush=True,
            )
    finally:
        connection.close()

    proxy_quality = {"stage": "P3.6", "windows": proxy_windows, "final_week": "not_run"}
    relation_output = {"stage": "P3.6", "windows": relation_windows, "final_week": "not_run"}
    thresholds_output = {"stage": "P3.6", "windows": threshold_windows, "final_week": "not_run"}
    candidate_manifest = {
        "stage": "P3.6",
        "status": "measured",
        "description": "manifest for ignored proxy, candidate-level, and candidate-history pair-level parquet evidence",
        "windows": candidate_manifest_windows,
        "final_week": "not_run",
    }
    p32_output = {"stage": "P3.6", "windows": p32_windows, "final_week": "not_run"}
    p33_output = {"stage": "P3.6", "windows": p33_windows, "final_week": "not_run"}
    attention_output = {"stage": "P3.6", "windows": attention_windows, "final_week": "not_run"}
    teacher_output = {"stage": "P3.6", "windows": teacher_windows, "final_week": "not_run"}
    higher_output = {"stage": "P3.6", "windows": higher_windows, "final_week": "not_run"}
    age_output = {"stage": "P3.6", "windows": age_windows, "final_week": "not_run"}
    category_output = {"stage": "P3.6", "windows": category_windows, "final_week": "not_run"}
    verdicts, verdict_evidence = _derive_verdicts(
        proxy_quality=proxy_quality,
        p32=p32_output,
        p33=p33_output,
        higher_order=higher_output,
        history_age=age_output,
        teacher_composition=teacher_output,
    )
    historical_after = {str(path.relative_to(config.repo_root)): _sha(path) for path in required_reports}
    audit_checks = {
        "p31_top200_identity_exact": all(data["p31_candidate_identity"]["exact_top200_groups"] for data in identity_windows.values()),
        "p32_prediction_and_attention_identity_verified": all(data["p32_scores"]["rows"] == data["p31_candidate_identity"]["rows"] for data in identity_windows.values()),
        "p33_prediction_and_embedding_identity_verified": all(data["p33_official_candidate_identity"]["exact_top200_groups"] for data in identity_windows.values()),
        "fashionclip_sha_identity_verified": True,
        "history_strictly_before_cutoff": all(data["history_days_strictly_positive"] for data in window_audits.values()),
        "warmness_uses_only_pre_cutoff_counts": all(data["warm_proxy_count_rule_cutoff_safe"] for data in window_audits.values()),
        "proxy_selection_excludes_truth_and_behavioral_teachers": all(data["proxy_selection_excludes_truth_and_teachers"] for data in window_audits.values()),
        "all_selected_proxies_are_warm": all(data["all_selected_proxies_warm"] for data in window_audits.values()),
        "teacher_assets_cutoff_safe": all(data["teacher_cutoff_audit"]["cutoff_safe"] for data in window_audits.values()),
        "quartiles_label_free_by_contract": True,
        "primary_matrix_observations_conserved": all(data["matrix_conservation"] for data in window_audits.values()),
        "secondary_labels_mutually_exclusive_and_conserved": all(data["label_conservation"] for data in window_audits.values()),
        "historical_evidence_not_overwritten": historical_before == historical_after,
        "final_cutoff_rejected": True,
    }
    audit_checks["all_passed"] = all(audit_checks.values())
    if not audit_checks["all_passed"]:
        raise RuntimeError(f"P3.6 audit checks failed: {audit_checks}")
    return _json_ready(
        {
            "schema_version": "phase3-p3.6-similarity-complementarity-audit-v1",
            "stage": "P3.6",
            "status": "measured",
            "run_id": "p3-6-v1-similarity-complementarity-audit",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_type": "read-only no-training no-new-candidate-generation diagnostic",
            "contract": contract,
            "proxy_quality": proxy_quality,
            "relation_matrix": relation_output,
            "thresholds_and_labels": thresholds_output,
            "candidate_relation_features": candidate_manifest,
            "p32_relation_attribution": p32_output,
            "p33_relation_attribution": p33_output,
            "attention_relation_audit": attention_output,
            "teacher_relation_composition": teacher_output,
            "deepwalk_higher_order_audit": higher_output,
            "history_age_relation_audit": age_output,
            "category_pair_audit": category_output,
            "verdicts": verdicts,
            "verdict_evidence": verdict_evidence,
            "input_identities": {
                "fashionclip": fashion_identity,
                "catalog_rows_with_valid_fashionclip": int(np.count_nonzero(np.asarray(image_rows) >= 0)),
                "transactions": file_identity(transactions),
                "windows": identity_windows,
                "historical_sha256_before_after_equal": historical_before,
            },
            "window_audits": window_audits,
            "audit_checks": audit_checks,
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "duckdb_memory_limit": config.memory_limit,
                "threads": config.threads,
                "device": torch.cuda.get_device_name() if config.device == "cuda" else "cpu",
                "model_training": False,
                "new_recommendation_candidate_generation": False,
            },
            "final_week": "not_run",
        }
    )


def _write_outputs(config: Config, result: dict[str, Any]) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    components = {
        "p3_6_proxy_quality.json": result["proxy_quality"],
        "p3_6_relation_matrix.json": result["relation_matrix"],
        "p3_6_candidate_relation_features.json": result["candidate_relation_features"],
        "p3_6_p32_relation_attribution.json": result["p32_relation_attribution"],
        "p3_6_p33_relation_attribution.json": result["p33_relation_attribution"],
        "p3_6_teacher_relation_composition.json": result["teacher_relation_composition"],
        "p3_6_deepwalk_higher_order_audit.json": result["deepwalk_higher_order_audit"],
        "p3_6_history_age_relation_audit.json": result["history_age_relation_audit"],
        "p3_6_attention_relation_audit.json": result["attention_relation_audit"],
        "p3_6_category_pair_audit.json": result["category_pair_audit"],
        "p3_6_thresholds_and_labels.json": result["thresholds_and_labels"],
    }
    for name, value in components.items():
        atomic_json(config.output_dir / name, value)
    master_path = config.output_dir / "P3_6_SIMILARITY_COMPLEMENTARITY_AUDIT.json"
    report_path = config.output_dir / "P3_6_SIMILARITY_COMPLEMENTARITY_AUDIT.md"
    atomic_json(master_path, result)
    report_path.write_text(render_report(result), encoding="utf-8")
    artifact_names = [
        *components,
        master_path.name,
        report_path.name,
        "P3_6_EXPERIMENT_CONTRACT.json",
    ]
    manifest = {
        "schema_version": "phase3-p3.6-output-manifest-v1",
        "stage": "P3.6",
        "status": "completed",
        "artifacts": {
            name: file_identity(config.output_dir / name) for name in artifact_names
        },
        "large_ignored_artifacts": result["candidate_relation_features"],
        "audit_checks": result["audit_checks"],
        "verdicts": result["verdicts"],
        "final_week": "not_run",
    }
    atomic_json(config.output_dir / "P3_6_OUTPUT_MANIFEST.json", manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only P3.6 relation audit")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/phase3/p3-6-v1-similarity-complementarity-audit"),
    )
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--user-batch", type=int, default=48)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = Config(
        repo_root=args.repo_root.resolve(),
        source_root=args.source_root.resolve(),
        output_dir=(args.repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve(),
        artifact_dir=(args.repo_root / args.artifact_dir).resolve() if not args.artifact_dir.is_absolute() else args.artifact_dir.resolve(),
        memory_limit=args.memory_limit,
        threads=args.threads,
        device=args.device,
        user_batch=args.user_batch,
    )
    result = run_audit(config)
    manifest = _write_outputs(config, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdicts": result["verdicts"],
                "runtime": result["runtime"],
                "manifest": manifest["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
