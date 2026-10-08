from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m33 import ROLLING_PROTOCOL
from .m4_contract import M4_RUN_ID, STATIC_FIELDS, atomic_json, file_identity, stable_u64


TEACHER_CONTRACT = {
    "history_weeks": 12,
    "sequence": "distinct user-day-item with fixed hash ordering",
    "vector_size": 64,
    "context_window": 10,
    "objective": "skip-gram negative sampling",
    "negative": 10,
    "epochs": 5,
    "min_count": 2,
    "workers": 1,
    "positive_top_k": 20,
    "hard_negative_exclusion_top_k": 100,
}


def required_teacher_cutoffs() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                cutoff
                for protocol in ROLLING_PROTOCOL.values()
                for cutoff in [
                    *protocol["inner_train"],
                    protocol["inner_validation"],
                    *protocol["outer_train"],
                ]
            }
        )
    )


def _resolve_source(source_root: Path, cutoff: str) -> dict[str, Any]:
    manifest_path = source_root / "artifacts" / "m2_9" / "item2vec-source-v1" / cutoff / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("cutoff") != cutoff:
        raise ValueError(f"invalid Item2Vec source manifest: {manifest_path}")
    contract = manifest.get("contract", {})
    expected = {
        "history_weeks": 12,
        "vector_size": 64,
        "context_window": 10,
        "min_count": 2,
        "negative": 10,
        "epochs": 5,
        "workers": 1,
    }
    drift = [key for key, value in expected.items() if contract.get(key) != value]
    if drift:
        raise ValueError(f"Item2Vec teacher contract drift at {cutoff}: {drift}")
    artifacts = manifest.get("artifacts", {})
    if not artifacts:
        base = manifest_path.parent
        paths = {
            "items": base / "items.csv",
            "neighbor_indices": base / "neighbor_indices.int32.npy",
            "neighbor_scores": base / "neighbor_scores.float32.npy",
            "normalized_vectors": base / "normalized_vectors.float32.npy",
        }
    else:
        paths = {name: Path(artifacts[name]["path"]) for name in (
            "items", "neighbor_indices", "neighbor_scores", "normalized_vectors"
        )}
    identities = {name: file_identity(path) for name, path in paths.items()}
    if artifacts:
        for name, identity in identities.items():
            declared = artifacts[name]
            if identity["bytes"] != int(declared["bytes"]) or identity["sha256"] != declared["sha256"]:
                raise ValueError(f"Item2Vec artifact identity drift: {cutoff}/{name}")
    return {
        "manifest": manifest,
        "manifest_identity": file_identity(manifest_path),
        "paths": paths,
        "identities": identities,
    }


def _category_pool(values: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    pools: dict[str, list[int]] = {}
    for row in eligible.tolist():
        pools.setdefault(str(values[row]), []).append(int(row))
    return {key: np.asarray(rows, dtype=np.int32) for key, rows in pools.items()}


def _choose_hard_negatives(
    *, anchor_catalog: int, teacher_anchor: int, excluded_teacher: np.ndarray,
    positive_rank: int, teacher_to_catalog: np.ndarray, product_values: np.ndarray,
    garment_values: np.ndarray, product_pools: dict[str, np.ndarray], garment_pools: dict[str, np.ndarray],
) -> tuple[int, str]:
    excluded_catalog = set(teacher_to_catalog[excluded_teacher].tolist())
    excluded_catalog.add(anchor_catalog)
    choices = (
        ("same_product_type", product_pools.get(str(product_values[anchor_catalog]), np.empty(0, dtype=np.int32))),
        ("same_garment_group", garment_pools.get(str(garment_values[anchor_catalog]), np.empty(0, dtype=np.int32))),
    )
    salt = stable_u64(f"{teacher_anchor}:{positive_rank}:m4-hard-negative")
    for source, pool in choices:
        if len(pool) == 0:
            continue
        start = salt % len(pool)
        for offset in range(len(pool)):
            candidate = int(pool[(start + offset) % len(pool)])
            if candidate not in excluded_catalog:
                return candidate, source
    vocab = teacher_to_catalog
    start = salt % len(vocab)
    for offset in range(len(vocab)):
        candidate = int(vocab[(start + offset) % len(vocab)])
        if candidate not in excluded_catalog:
            return candidate, "fallback_teacher_vocab"
    raise RuntimeError("unable to construct a teacher hard negative")


def build_teacher_relations(
    *, source_root: Path, artifact_dir: Path, output_dir: Path
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    articles_path = source_root / "data" / "raw" / "articles.csv"
    transactions_path = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    articles = pd.read_csv(articles_path, dtype=str)
    if articles["article_id"].duplicated().any() or len(articles) != 105_542:
        raise ValueError("unexpected article catalog identity")
    article_to_row = {item: row for row, item in enumerate(articles["article_id"].tolist())}
    product_values = articles["product_type_no"].fillna("__MISSING__").to_numpy()
    garment_values = articles["garment_group_no"].fillna("__MISSING__").to_numpy()
    con = duckdb.connect()
    tx_sql = str(transactions_path.resolve()).replace("'", "''")
    results: dict[str, Any] = {}
    all_pair_keys: list[np.ndarray] = []
    try:
        for cutoff in required_teacher_cutoffs():
            started = time.perf_counter()
            source = _resolve_source(source_root, cutoff)
            items = pd.read_csv(source["paths"]["items"], dtype={"article_id": str}).sort_values("row_index")
            if items["article_id"].duplicated().any():
                raise ValueError(f"duplicate teacher vocabulary: {cutoff}")
            try:
                teacher_to_catalog = np.asarray(
                    [article_to_row[item] for item in items["article_id"]], dtype=np.int32
                )
            except KeyError as error:
                raise ValueError(f"teacher item missing from article catalog: {error}") from error
            neighbors = np.load(source["paths"]["neighbor_indices"], mmap_mode="r")
            scores = np.load(source["paths"]["neighbor_scores"], mmap_mode="r")
            if neighbors.shape != (len(items), 100) or scores.shape != neighbors.shape:
                raise ValueError(f"teacher neighbor shape drift: {cutoff} {neighbors.shape}")
            product_pools = _category_pool(product_values, teacher_to_catalog)
            garment_pools = _category_pool(garment_values, teacher_to_catalog)
            pair_count = len(items) * TEACHER_CONTRACT["positive_top_k"]
            anchor = np.empty(pair_count, dtype=np.int32)
            positive = np.empty(pair_count, dtype=np.int32)
            negative = np.empty(pair_count, dtype=np.int32)
            cosine = np.empty(pair_count, dtype=np.float32)
            rank = np.empty(pair_count, dtype=np.uint8)
            negative_source = np.empty(pair_count, dtype=np.uint8)
            cursor = 0
            source_counts = {"same_product_type": 0, "same_garment_group": 0, "fallback_teacher_vocab": 0}
            for teacher_row in range(len(items)):
                anchor_catalog = int(teacher_to_catalog[teacher_row])
                excluded = np.asarray(neighbors[teacher_row, :100], dtype=np.int32)
                for positive_rank in range(1, 21):
                    positive_teacher = int(neighbors[teacher_row, positive_rank - 1])
                    hard_negative, hard_source = _choose_hard_negatives(
                        anchor_catalog=anchor_catalog,
                        teacher_anchor=teacher_row,
                        excluded_teacher=excluded,
                        positive_rank=positive_rank,
                        teacher_to_catalog=teacher_to_catalog,
                        product_values=product_values,
                        garment_values=garment_values,
                        product_pools=product_pools,
                        garment_pools=garment_pools,
                    )
                    anchor[cursor] = anchor_catalog
                    positive[cursor] = int(teacher_to_catalog[positive_teacher])
                    negative[cursor] = hard_negative
                    cosine[cursor] = float(scores[teacher_row, positive_rank - 1])
                    rank[cursor] = positive_rank
                    source_code = {"same_product_type": 1, "same_garment_group": 2, "fallback_teacher_vocab": 3}[hard_source]
                    negative_source[cursor] = source_code
                    source_counts[hard_source] += 1
                    cursor += 1
            if cursor != pair_count:
                raise RuntimeError("teacher pair materialization row mismatch")
            if np.any(anchor == positive) or np.any(anchor == negative) or np.any(positive == negative):
                raise RuntimeError("teacher relation contains invalid self/equal pair")
            relation_path = artifact_dir / cutoff / "teacher_relations.npz"
            relation_path.parent.mkdir(parents=True, exist_ok=True)
            catalog_token_count = np.zeros(len(articles), dtype=np.int32)
            catalog_token_count[teacher_to_catalog] = items["token_count"].to_numpy(dtype=np.int32)
            np.savez_compressed(
                relation_path,
                anchor=anchor,
                positive=positive,
                negative=negative,
                teacher_cosine=cosine,
                teacher_rank=rank,
                hard_negative_source=negative_source,
                catalog_token_count=catalog_token_count,
            )
            latest = con.execute(
                f"SELECT max(t_dat) FROM read_parquet('{tx_sql}') WHERE t_dat < DATE '{cutoff}'"
            ).fetchone()[0]
            if latest is None or str(latest) >= cutoff:
                raise RuntimeError(f"teacher cutoff audit failed: {cutoff}")
            pair_keys = (anchor.astype(np.int64) << 17) | positive.astype(np.int64)
            all_pair_keys.append(pair_keys)
            token_counts = items["token_count"].to_numpy(dtype=np.int64)
            manifest = {
                "schema_version": "m4.1-teacher-relations-v1",
                "stage": "M4.1",
                "status": "completed",
                "run_id": M4_RUN_ID,
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": TEACHER_CONTRACT,
                "audit": {
                    "teacher_vocab_items": len(items),
                    "anchors": len(items),
                    "positive_pairs": pair_count,
                    "hard_negatives": pair_count,
                    "hard_negative_source_counts": source_counts,
                    "product_type_hard_negative_coverage": source_counts["same_product_type"] / pair_count,
                    "teacher_cosine": {
                        "min": float(cosine.min()),
                        "mean": float(cosine.mean()),
                        "p50": float(np.quantile(cosine, 0.5)),
                        "p95": float(np.quantile(cosine, 0.95)),
                        "max": float(cosine.max()),
                    },
                    "teacher_token_count": {
                        "min": int(token_counts.min()),
                        "median": float(np.median(token_counts)),
                        "max": int(token_counts.max()),
                    },
                    "latest_transaction_before_cutoff": str(latest),
                    "cutoff_safe": str(latest) < cutoff,
                },
                "inputs": {
                    "source_manifest": source["manifest_identity"],
                    "teacher_artifacts": source["identities"],
                    "transactions": file_identity(transactions_path),
                    "articles": file_identity(articles_path),
                    "code": file_identity(Path(__file__)),
                },
                "artifact": file_identity(relation_path),
                "elapsed_seconds": time.perf_counter() - started,
                "final_week": "not_run",
            }
            manifest_path = relation_path.parent / "manifest.json"
            atomic_json(manifest_path, manifest)
            results[cutoff] = {**manifest, "manifest": file_identity(manifest_path)}
            print(f"M4.1 {cutoff}: {pair_count:,} relations", flush=True)
    finally:
        con.close()
    combined = np.concatenate(all_pair_keys)
    unique_pairs = int(len(np.unique(combined)))
    total_pairs = int(len(combined))
    summary = {
        "schema_version": "m4.1-teacher-summary-v1",
        "stage": "M4.1",
        "status": "measured",
        "run_id": M4_RUN_ID,
        "contract": TEACHER_CONTRACT,
        "cutoffs": results,
        "cross_cutoff": {
            "temporal_pair_observations": total_pairs,
            "unique_anchor_positive_pairs": unique_pairs,
            "repeated_observations": total_pairs - unique_pairs,
            "repeated_observation_ratio": (total_pairs - unique_pairs) / total_pairs,
            "unit": "one cutoff-anchor-positive relation; repeats across cutoffs remain independent temporal observations",
        },
        "student_static_fields": list(STATIC_FIELDS),
        "final_week": "not_run",
    }
    atomic_json(output_dir / "M4_1_metrics.json", summary)
    return summary
