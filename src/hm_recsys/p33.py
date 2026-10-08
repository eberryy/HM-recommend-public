from __future__ import annotations

import copy
import gc
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import duckdb
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from torch.nn import functional as F

from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, STATIC_FIELDS, atomic_json, file_identity, stable_u64
from .m4_student import (
    CatalogAccessor,
    StaticItemEncoder,
    TRAINING,
    _evaluate_pairs,
    _positive_weight,
    _split_mask,
    encode_catalog,
    representation_diagnostics,
)
from .m4_teacher import required_teacher_cutoffs
from .p3 import run_p31, run_p32


RUN_ID = "phase3-p3.3-v1-multiview-graph-student"
VIEWS = ("item2vec", "direct_covisit", "deepwalk")
GRAPH_CONTRACT = {
    "history_weeks": 12,
    "basket": "distinct items for one customer on one calendar day",
    "maximum_basket_items": 20,
    "minimum_pair_user_days": 2,
    "direct_neighbor_k": 100,
    "direct_score": "pair_user_days / sqrt(anchor_user_days * neighbor_user_days)",
    "deepwalk_graph": "symmetrized union of directed direct-co-vis Top100 edges",
    "deepwalk_walks_per_node": 2,
    "deepwalk_walk_length": 20,
    "deepwalk_vector_size": 64,
    "deepwalk_window": 10,
    "deepwalk_negative": 10,
    "deepwalk_epochs": 3,
    "deepwalk_workers": 1,
    "positive_top_k": 20,
    "hard_negative_exclusion_top_k": 100,
    "seed": 20260905,
}
STUDENT_CONTRACT = {
    "variant": "multimodal",
    "architecture": "identical StaticItemEncoder used by M4 multimodal Student",
    "teacher_view_weights": {view: 1.0 / 3.0 for view in VIEWS},
    "pairwise_lambda": 0.25,
    "batch_size_per_view": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "max_epochs": 3,
    "early_stopping_patience": 1,
    "seed": 20260905,
    "inference_inputs": ["FashionCLIP image embedding", *STATIC_FIELDS],
    "teacher_embeddings_at_inference": False,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for P3.3 but unavailable")
    return torch.device(name)


@dataclass
class ViewNeighbors:
    anchors: np.ndarray
    neighbors: np.ndarray
    scores: np.ndarray

    def validate(self, name: str) -> None:
        if self.neighbors.shape != self.scores.shape:
            raise RuntimeError(f"{name} neighbor/score shape mismatch")
        if self.neighbors.ndim != 2 or self.neighbors.shape[1] != 100:
            raise RuntimeError(f"{name} must materialize Top100")
        if len(self.anchors) != len(self.neighbors) or len(np.unique(self.anchors)) != len(self.anchors):
            raise RuntimeError(f"{name} anchor identity failure")
        valid = self.neighbors >= 0
        if np.any(self.neighbors[valid] == np.repeat(self.anchors, 100)[valid.ravel()]):
            raise RuntimeError(f"{name} contains self neighbor")
        if not np.isfinite(self.scores[valid]).all():
            raise RuntimeError(f"{name} contains non-finite score")


def _item2vec_neighbors(source_root: Path, cutoff: str, article_to_row: dict[str, int]) -> ViewNeighbors:
    root = source_root / "artifacts" / "m2_9" / "item2vec-source-v1" / cutoff
    manifest = _json(root / "source-manifest.json")
    if manifest.get("status") != "completed" or manifest.get("cutoff") != cutoff:
        raise RuntimeError(f"invalid Item2Vec source at {cutoff}")
    expected = {
        "history_weeks": 12,
        "vector_size": 64,
        "context_window": 10,
        "min_count": 2,
        "negative": 10,
        "epochs": 5,
        "workers": 1,
    }
    drift = [key for key, value in expected.items() if manifest.get("contract", {}).get(key) != value]
    if drift:
        raise RuntimeError(f"Item2Vec contract drift at {cutoff}: {drift}")
    declared = manifest.get("artifacts", {})
    paths = {
        name: Path(declared[name]["path"]) if declared else root / filename
        for name, filename in {
            "items": "items.csv",
            "neighbor_indices": "neighbor_indices.int32.npy",
            "neighbor_scores": "neighbor_scores.float32.npy",
        }.items()
    }
    if declared:
        for name, path in paths.items():
            observed = file_identity(path)
            if observed["sha256"] != declared[name]["sha256"] or observed["bytes"] != int(declared[name]["bytes"]):
                raise RuntimeError(f"Item2Vec artifact identity drift at {cutoff}/{name}")
    items = pd.read_csv(paths["items"], dtype={"article_id": str}).sort_values("row_index")
    teacher_to_catalog = np.asarray([article_to_row[item] for item in items["article_id"]], dtype=np.int32)
    local_neighbors = np.load(paths["neighbor_indices"], mmap_mode="r")
    scores = np.load(paths["neighbor_scores"], mmap_mode="r")
    if local_neighbors.shape != (len(items), 100) or scores.shape != local_neighbors.shape:
        raise RuntimeError(f"Item2Vec Top100 shape drift at {cutoff}")
    result = ViewNeighbors(
        anchors=teacher_to_catalog,
        neighbors=teacher_to_catalog[np.asarray(local_neighbors, dtype=np.int64)],
        scores=np.asarray(scores, dtype=np.float32),
    )
    result.validate("item2vec")
    return result


def _direct_covisit_neighbors(
    *, transactions_path: Path, cutoff: str, article_to_row: dict[str, int]
) -> tuple[ViewNeighbors, dict[str, Any]]:
    started = time.perf_counter()
    tx = transactions_path.resolve().as_posix().replace("'", "''")
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            CREATE TEMP TABLE p33_valid_days AS
            SELECT customer_id,t_dat,count(DISTINCT article_id) AS distinct_items
            FROM read_parquet('{tx}')
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL {GRAPH_CONTRACT['history_weeks']} WEEK
              AND t_dat<DATE '{cutoff}'
            GROUP BY customer_id,t_dat
            HAVING distinct_items BETWEEN 2 AND {GRAPH_CONTRACT['maximum_basket_items']}
            """
        )
        con.execute(
            f"""
            CREATE TEMP TABLE p33_day_items AS
            SELECT DISTINCT t.customer_id,t.t_dat,t.article_id
            FROM read_parquet('{tx}') t
            SEMI JOIN p33_valid_days d USING(customer_id,t_dat)
            WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL {GRAPH_CONTRACT['history_weeks']} WEEK
              AND t.t_dat<DATE '{cutoff}'
            """
        )
        con.execute(
            """
            CREATE TEMP TABLE p33_support AS
            SELECT article_id,count(*) AS item_user_days
            FROM p33_day_items GROUP BY article_id
            """
        )
        con.execute(
            f"""
            CREATE TEMP TABLE p33_pairs AS
            SELECT l.article_id AS left_article_id,r.article_id AS right_article_id,
                   count(*) AS pair_user_days
            FROM p33_day_items l JOIN p33_day_items r
              ON l.customer_id=r.customer_id AND l.t_dat=r.t_dat AND l.article_id<r.article_id
            GROUP BY l.article_id,r.article_id
            HAVING pair_user_days>={GRAPH_CONTRACT['minimum_pair_user_days']}
            """
        )
        frame = con.execute(
            f"""
            WITH directed AS (
              SELECT left_article_id AS anchor_article_id,right_article_id AS neighbor_article_id,pair_user_days
              FROM p33_pairs
              UNION ALL
              SELECT right_article_id,left_article_id,pair_user_days FROM p33_pairs
            ), scored AS (
              SELECT d.anchor_article_id,d.neighbor_article_id,d.pair_user_days,
                     d.pair_user_days/sqrt(a.item_user_days*b.item_user_days) AS score
              FROM directed d JOIN p33_support a ON d.anchor_article_id=a.article_id
              JOIN p33_support b ON d.neighbor_article_id=b.article_id
            )
            SELECT anchor_article_id,neighbor_article_id,pair_user_days,score,
                   row_number() OVER(PARTITION BY anchor_article_id
                     ORDER BY score DESC,pair_user_days DESC,neighbor_article_id) AS neighbor_rank
            FROM scored QUALIFY neighbor_rank<={GRAPH_CONTRACT['direct_neighbor_k']}
            ORDER BY anchor_article_id,neighbor_rank
            """
        ).fetch_df()
        counts = con.execute(
            "SELECT (SELECT count(*) FROM p33_valid_days),"
            "(SELECT count(*) FROM p33_day_items),(SELECT count(*) FROM p33_pairs)"
        ).fetchone()
        latest = con.execute(
            f"SELECT max(t_dat) FROM read_parquet('{tx}') WHERE t_dat<DATE '{cutoff}'"
        ).fetchone()[0]
    finally:
        con.close()
    if frame.empty:
        raise RuntimeError(f"empty direct co-vis graph at {cutoff}")
    frame["anchor_catalog"] = frame["anchor_article_id"].astype(str).map(article_to_row)
    frame["neighbor_catalog"] = frame["neighbor_article_id"].astype(str).map(article_to_row)
    if frame[["anchor_catalog", "neighbor_catalog"]].isna().any().any():
        raise RuntimeError(f"co-vis item missing from catalog at {cutoff}")
    anchors = frame["anchor_catalog"].drop_duplicates().to_numpy(dtype=np.int32)
    anchor_position = {int(value): index for index, value in enumerate(anchors)}
    neighbors = np.full((len(anchors), 100), -1, dtype=np.int32)
    scores = np.zeros((len(anchors), 100), dtype=np.float32)
    for row in frame.itertuples(index=False):
        position = anchor_position[int(row.anchor_catalog)]
        rank = int(row.neighbor_rank) - 1
        neighbors[position, rank] = int(row.neighbor_catalog)
        scores[position, rank] = float(row.score)
    result = ViewNeighbors(anchors=anchors, neighbors=neighbors, scores=scores)
    result.validate("direct_covisit")
    audit = {
        "valid_user_days": int(counts[0]),
        "distinct_user_day_items": int(counts[1]),
        "undirected_pairs": int(counts[2]),
        "anchors": int(len(anchors)),
        "directed_top100_edges": int(len(frame)),
        "anchors_with_top20": int(np.count_nonzero(np.sum(neighbors >= 0, axis=1) >= 20)),
        "latest_transaction_before_cutoff": str(latest),
        "cutoff_safe": str(latest) < cutoff,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, audit


class _WalkCorpus:
    def __init__(self, adjacency: dict[int, np.ndarray], *, cutoff: str) -> None:
        self.adjacency = adjacency
        self.nodes = sorted(adjacency)
        self.seed = int(stable_u64(f"{cutoff}:p33-deepwalk") & 0xFFFFFFFF)

    def __iter__(self) -> Iterator[list[str]]:
        rng = np.random.default_rng(self.seed)
        for walk_index in range(GRAPH_CONTRACT["deepwalk_walks_per_node"]):
            order = np.asarray(self.nodes, dtype=np.int32)
            if walk_index:
                rng.shuffle(order)
            for start in order.tolist():
                walk = [int(start)]
                current = int(start)
                for _ in range(GRAPH_CONTRACT["deepwalk_walk_length"] - 1):
                    choices = self.adjacency.get(current)
                    if choices is None or len(choices) == 0:
                        break
                    current = int(choices[int(rng.integers(0, len(choices)))])
                    walk.append(current)
                yield [str(value) for value in walk]


def _deepwalk_neighbors(
    *, direct: ViewNeighbors, cutoff: str, device: torch.device
) -> tuple[ViewNeighbors, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    adjacency_lists: dict[int, list[int]] = {}
    for anchor, row in zip(direct.anchors.tolist(), direct.neighbors.tolist()):
        for neighbor in row:
            if neighbor < 0:
                continue
            adjacency_lists.setdefault(int(anchor), []).append(int(neighbor))
            adjacency_lists.setdefault(int(neighbor), []).append(int(anchor))
    adjacency = {
        node: np.unique(np.asarray(values, dtype=np.int32))
        for node, values in adjacency_lists.items()
    }
    corpus = _WalkCorpus(adjacency, cutoff=cutoff)
    model = Word2Vec(
        sentences=corpus,
        vector_size=GRAPH_CONTRACT["deepwalk_vector_size"],
        window=GRAPH_CONTRACT["deepwalk_window"],
        min_count=1,
        sg=1,
        hs=0,
        negative=GRAPH_CONTRACT["deepwalk_negative"],
        workers=GRAPH_CONTRACT["deepwalk_workers"],
        epochs=GRAPH_CONTRACT["deepwalk_epochs"],
        seed=GRAPH_CONTRACT["seed"],
        sorted_vocab=1,
        hashfxn=lambda value: int(stable_u64(value) & 0xFFFFFFFF),
    )
    anchors = np.asarray(sorted(adjacency), dtype=np.int32)
    vectors = np.asarray([model.wv[str(int(item))] for item in anchors], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    tensor = torch.from_numpy(vectors).to(device)
    neighbors = np.full((len(anchors), 100), -1, dtype=np.int32)
    scores = np.zeros((len(anchors), 100), dtype=np.float32)
    buffer_k = min(len(anchors), 117)
    with torch.inference_mode():
        for start in range(0, len(anchors), 512):
            end = min(start + 512, len(anchors))
            similarity = tensor[start:end] @ tensor.T
            local = torch.arange(end - start, device=device)
            similarity[local, torch.arange(start, end, device=device)] = -torch.inf
            values, indices = torch.topk(similarity, k=buffer_k, dim=1, sorted=False)
            values_np = values.cpu().numpy()
            indices_np = indices.cpu().numpy()
            for local_row in range(end - start):
                candidate_catalog = anchors[indices_np[local_row]]
                order = np.lexsort((candidate_catalog, -values_np[local_row]))[:100]
                length = len(order)
                neighbors[start + local_row, :length] = candidate_catalog[order]
                scores[start + local_row, :length] = values_np[local_row, order]
    del tensor, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result = ViewNeighbors(anchors=anchors, neighbors=neighbors, scores=scores)
    result.validate("deepwalk")
    audit = {
        "nodes": int(len(anchors)),
        "symmetrized_edges": int(sum(len(value) for value in adjacency.values()) // 2),
        "walks": int(len(anchors) * GRAPH_CONTRACT["deepwalk_walks_per_node"]),
        "walk_tokens_upper_bound": int(
            len(anchors)
            * GRAPH_CONTRACT["deepwalk_walks_per_node"]
            * GRAPH_CONTRACT["deepwalk_walk_length"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result, vectors, audit


def _save_neighbors(path: Path, view: ViewNeighbors) -> dict[str, Any]:
    np.savez_compressed(path, anchors=view.anchors, neighbors=view.neighbors, scores=view.scores)
    return file_identity(path)


def _load_neighbors(path: Path) -> ViewNeighbors:
    value = np.load(path)
    result = ViewNeighbors(
        anchors=np.asarray(value["anchors"]),
        neighbors=np.asarray(value["neighbors"]),
        scores=np.asarray(value["scores"]),
    )
    result.validate(path.stem)
    return result


def _category_pools(values: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    output: dict[str, list[int]] = {}
    for row in eligible.tolist():
        output.setdefault(str(values[row]), []).append(int(row))
    return {key: np.asarray(rows, dtype=np.int32) for key, rows in output.items()}


def _row_lookup(view: ViewNeighbors) -> dict[int, int]:
    return {int(anchor): row for row, anchor in enumerate(view.anchors.tolist())}


def _choose_negative(
    *, cutoff: str, view_name: str, anchor: int, rank: int, excluded: set[int],
    eligible: np.ndarray, product: np.ndarray, garment: np.ndarray,
    product_pools: dict[str, np.ndarray], garment_pools: dict[str, np.ndarray],
) -> tuple[int, str]:
    salt = stable_u64(f"{cutoff}:{view_name}:{anchor}:{rank}:negative")
    for name, pool in (
        ("same_product_type", product_pools.get(str(product[anchor]), np.empty(0, dtype=np.int32))),
        ("same_garment_group", garment_pools.get(str(garment[anchor]), np.empty(0, dtype=np.int32))),
        ("fallback_teacher_vocab", eligible),
    ):
        if len(pool) == 0:
            continue
        start = int(salt % len(pool))
        for offset in range(len(pool)):
            candidate = int(pool[(start + offset) % len(pool)])
            if candidate not in excluded:
                return candidate, name
    raise RuntimeError(f"unable to construct P3.3 hard negative for {cutoff}/{view_name}/{anchor}")


def _materialize_relations(
    *, cutoff: str, views: dict[str, ViewNeighbors], articles: pd.DataFrame,
    token_count: np.ndarray, output_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    all_items = []
    for view in views.values():
        all_items.extend([view.anchors, view.neighbors[view.neighbors >= 0]])
    eligible = np.unique(np.concatenate(all_items)).astype(np.int32)
    product = articles["product_type_no"].fillna("__MISSING__").astype(str).to_numpy()
    garment = articles["garment_group_no"].fillna("__MISSING__").astype(str).to_numpy()
    product_pools = _category_pools(product, eligible)
    garment_pools = _category_pools(garment, eligible)
    lookups = {name: _row_lookup(view) for name, view in views.items()}
    identities: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    pair_sets: dict[str, set[int]] = {}
    source_codes = {"same_product_type": 1, "same_garment_group": 2, "fallback_teacher_vocab": 3}
    for view_name, view in views.items():
        rows: list[tuple[int, int, int, float, int, int]] = []
        negative_counts = {name: 0 for name in source_codes}
        for anchor_position, anchor in enumerate(view.anchors.tolist()):
            positives = view.neighbors[anchor_position, : GRAPH_CONTRACT["positive_top_k"]]
            positive_scores = view.scores[anchor_position, : GRAPH_CONTRACT["positive_top_k"]]
            excluded = {int(anchor)}
            for other_name, other_view in views.items():
                other_row = lookups[other_name].get(int(anchor))
                if other_row is not None:
                    excluded.update(int(value) for value in other_view.neighbors[other_row] if value >= 0)
            for rank, (positive, score) in enumerate(zip(positives.tolist(), positive_scores.tolist()), 1):
                if positive < 0:
                    continue
                negative, source = _choose_negative(
                    cutoff=cutoff, view_name=view_name, anchor=int(anchor), rank=rank,
                    excluded=excluded, eligible=eligible, product=product, garment=garment,
                    product_pools=product_pools, garment_pools=garment_pools,
                )
                rows.append((int(anchor), int(positive), negative, float(score), rank, source_codes[source]))
                negative_counts[source] += 1
        if not rows:
            raise RuntimeError(f"empty P3.3 relation view {cutoff}/{view_name}")
        matrix = np.asarray(rows, dtype=np.float64)
        relation_path = output_dir / f"relations_{view_name}.npz"
        np.savez_compressed(
            relation_path,
            anchor=matrix[:, 0].astype(np.int32),
            positive=matrix[:, 1].astype(np.int32),
            negative=matrix[:, 2].astype(np.int32),
            teacher_cosine=matrix[:, 3].astype(np.float32),
            teacher_rank=matrix[:, 4].astype(np.uint8),
            hard_negative_source=matrix[:, 5].astype(np.uint8),
            catalog_token_count=token_count.astype(np.int32),
        )
        identities[view_name] = file_identity(relation_path)
        pair_keys = (
            matrix[:, 0].astype(np.int64) * len(articles) + matrix[:, 1].astype(np.int64)
        )
        pair_sets[view_name] = set(pair_keys.tolist())
        audits[view_name] = {
            "anchors": int(len(np.unique(matrix[:, 0]))),
            "positive_pairs": int(len(matrix)),
            "hard_negative_source_counts": negative_counts,
            "all_negatives_outside_union_top100": True,
            "score": {
                "min": float(matrix[:, 3].min()),
                "mean": float(matrix[:, 3].mean()),
                "max": float(matrix[:, 3].max()),
            },
        }
    overlap = {}
    for left_index, left in enumerate(VIEWS):
        for right in VIEWS[left_index + 1 :]:
            intersection = len(pair_sets[left].intersection(pair_sets[right]))
            overlap[f"{left}__{right}"] = {
                "shared_pairs": intersection,
                "left_fraction": intersection / max(len(pair_sets[left]), 1),
                "right_fraction": intersection / max(len(pair_sets[right]), 1),
            }
    return identities, {"views": audits, "cross_view_positive_overlap": overlap}


def build_multiview_teachers(
    *, source_root: Path, repo_root: Path, artifact_dir: Path, device_name: str,
    reuse_completed: bool,
) -> dict[str, Any]:
    device = _device(device_name)
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    articles_path = source_root / "data" / "raw" / "articles.csv"
    articles = pd.read_csv(articles_path, dtype=str)
    if len(articles) != 105_542 or articles["article_id"].duplicated().any():
        raise RuntimeError("P3.3 article catalog identity failed")
    article_to_row = {item: row for row, item in enumerate(articles["article_id"].tolist())}
    m4_teacher = (
        repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation" / "teacher-v1"
    ).resolve()
    windows: dict[str, Any] = {}
    for cutoff in required_teacher_cutoffs():
        if cutoff >= FINAL_CUTOFF:
            raise RuntimeError("P3.3 final-week boundary failed")
        cutoff_dir = artifact_dir / cutoff
        manifest_path = cutoff_dir / "manifest.json"
        if reuse_completed and manifest_path.is_file():
            manifest = _json(manifest_path)
            relation_ok = all(
                file_identity(cutoff_dir / f"relations_{view}.npz")["sha256"]
                == manifest["relations"][view]["sha256"] for view in VIEWS
            )
            if manifest.get("status") == "completed" and relation_ok:
                print(f"P3.3 reuse teachers {cutoff}", flush=True)
                windows[cutoff] = manifest
                continue
        if cutoff_dir.exists():
            raise FileExistsError(f"refusing to overwrite incomplete P3.3 teacher cutoff: {cutoff_dir}")
        cutoff_dir.mkdir(parents=True)
        started = time.perf_counter()
        print(f"P3.3 direct co-vis {cutoff}", flush=True)
        item2vec = _item2vec_neighbors(source_root, cutoff, article_to_row)
        direct, direct_audit = _direct_covisit_neighbors(
            transactions_path=transactions, cutoff=cutoff, article_to_row=article_to_row
        )
        print(f"P3.3 DeepWalk {cutoff} (nodes={len(direct.anchors):,})", flush=True)
        deepwalk, vectors, deepwalk_audit = _deepwalk_neighbors(
            direct=direct, cutoff=cutoff, device=device
        )
        neighbor_identities = {
            "item2vec": _save_neighbors(cutoff_dir / "neighbors_item2vec.npz", item2vec),
            "direct_covisit": _save_neighbors(cutoff_dir / "neighbors_direct_covisit.npz", direct),
            "deepwalk": _save_neighbors(cutoff_dir / "neighbors_deepwalk.npz", deepwalk),
        }
        vector_path = cutoff_dir / "deepwalk_vectors.float32.npy"
        np.save(vector_path, vectors, allow_pickle=False)
        m4_relation = np.load(m4_teacher / cutoff / "teacher_relations.npz")
        token_count = np.asarray(m4_relation["catalog_token_count"], dtype=np.int32)
        relations, relation_audit = _materialize_relations(
            cutoff=cutoff,
            views={"item2vec": item2vec, "direct_covisit": direct, "deepwalk": deepwalk},
            articles=articles,
            token_count=token_count,
            output_dir=cutoff_dir,
        )
        manifest = {
            "schema_version": "phase3-p3.3-multiview-teacher-v1",
            "stage": "P3.3 teacher",
            "status": "completed",
            "run_id": RUN_ID,
            "cutoff": cutoff,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": GRAPH_CONTRACT,
            "audits": {
                "direct_covisit": direct_audit,
                "deepwalk": deepwalk_audit,
                "relations": relation_audit,
                "all_cutoff_safe": bool(direct_audit["cutoff_safe"]),
            },
            "neighbors": neighbor_identities,
            "relations": relations,
            "deepwalk_vectors": file_identity(vector_path),
            "inputs": {
                "transactions": file_identity(transactions),
                "articles": file_identity(articles_path),
                "m4_item2vec_relation": file_identity(m4_teacher / cutoff / "teacher_relations.npz"),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "final_week": "not_run",
        }
        atomic_json(manifest_path, manifest)
        windows[cutoff] = manifest
        del item2vec, direct, deepwalk, vectors, m4_relation
        gc.collect()
    return {
        "schema_version": "phase3-p3.3-multiview-teacher-summary-v1",
        "status": "measured",
        "run_id": RUN_ID,
        "contract": GRAPH_CONTRACT,
        "cutoffs": windows,
        "resources": {"peak_working_set_bytes": _peak_working_set_bytes()},
        "final_week": "not_run",
    }


def _load_relation_paths(teacher_dir: Path, cutoffs: Iterable[str]) -> dict[str, list[Path]]:
    return {
        view: [teacher_dir / cutoff / f"relations_{view}.npz" for cutoff in cutoffs]
        for view in VIEWS
    }


def _concatenate_relations(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    fields = ("anchor", "positive", "negative", "teacher_cosine", "teacher_rank")
    chunks = {field: [] for field in fields}
    token_count = None
    for path in paths:
        value = np.load(path)
        for field in fields:
            chunks[field].append(np.asarray(value[field]))
        if token_count is None:
            token_count = np.asarray(value["catalog_token_count"])
    if token_count is None:
        raise RuntimeError("no P3.3 relation paths")
    return {field: np.concatenate(values) for field, values in chunks.items()} | {
        "catalog_token_count": token_count
    }


def _balanced_train_encoder(
    *, relations: dict[str, dict[str, np.ndarray]], catalog_items: list[str],
    accessor: CatalogAccessor, cardinalities: list[int], output_dir: Path,
) -> tuple[StaticItemEncoder, dict[str, Any]]:
    torch.manual_seed(STUDENT_CONTRACT["seed"])
    np.random.seed(STUDENT_CONTRACT["seed"])
    if accessor.device.type == "cuda":
        torch.cuda.manual_seed_all(STUDENT_CONTRACT["seed"])
        torch.cuda.reset_peak_memory_stats()
    model = StaticItemEncoder(variant="multimodal", cardinalities=cardinalities).to(accessor.device)
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view, relation in relations.items():
        validation = _split_mask(relation["anchor"], catalog_items)
        indices = np.arange(len(relation["anchor"]), dtype=np.int64)
        splits[view] = (indices[~validation], indices[validation])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STUDENT_CONTRACT["learning_rate"],
        weight_decay=STUDENT_CONTRACT["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=accessor.device.type == "cuda")
    rng = np.random.default_rng(STUDENT_CONTRACT["seed"])
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_size = STUDENT_CONTRACT["batch_size_per_view"]

    def balanced_step(selected_by_view: dict[str, np.ndarray]) -> torch.Tensor:
        anchors = np.concatenate([
            relations[view]["anchor"][selected_by_view[view]] for view in VIEWS
        ])
        positives = np.concatenate([
            relations[view]["positive"][selected_by_view[view]] for view in VIEWS
        ])
        negatives = np.concatenate([
            relations[view]["negative"][selected_by_view[view]] for view in VIEWS
        ])
        ai, ac, am = accessor.batch(anchors)
        pi, pc, pm = accessor.batch(positives)
        ni, nc, nm = accessor.batch(negatives)
        with torch.amp.autocast(
            device_type=accessor.device.type,
            dtype=torch.float16,
            enabled=accessor.device.type == "cuda",
        ):
            anchor_embedding = model(ai, ac, am)
            positive_embedding = model(pi, pc, pm)
            negative_embedding = model(ni, nc, nm)
            losses = []
            cursor = 0
            for view in VIEWS:
                selected = selected_by_view[view]
                end = cursor + len(selected)
                ae = anchor_embedding[cursor:end]
                pe = positive_embedding[cursor:end]
                ne = negative_embedding[cursor:end]
                cosine = torch.as_tensor(
                    relations[view]["teacher_cosine"][selected], device=accessor.device
                )
                rank = torch.as_tensor(
                    relations[view]["teacher_rank"][selected], device=accessor.device
                )
                weight = _positive_weight(cosine, rank)
                logits = (ae @ pe.T) / TRAINING["temperature"]
                contrastive = (
                    F.cross_entropy(
                        logits,
                        torch.arange(len(selected), device=accessor.device),
                        reduction="none",
                    )
                    * weight
                ).mean()
                positive_similarity = (ae * pe).sum(dim=1)
                negative_similarity = (ae * ne).sum(dim=1)
                pairwise = (
                    F.softplus(
                        TRAINING["pairwise_margin"] - positive_similarity + negative_similarity
                    )
                    * weight
                ).mean()
                losses.append(contrastive + STUDENT_CONTRACT["pairwise_lambda"] * pairwise)
                cursor = end
        return torch.stack(losses).mean()

    for epoch in range(1, STUDENT_CONTRACT["max_epochs"] + 1):
        shuffled: dict[str, np.ndarray] = {}
        for view in VIEWS:
            shuffled[view] = splits[view][0].copy()
            rng.shuffle(shuffled[view])
        steps = max(math.ceil(len(shuffled[view]) / batch_size) for view in VIEWS)
        model.train()
        train_loss = 0.0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            selected_by_view: dict[str, np.ndarray] = {}
            for view in VIEWS:
                values = shuffled[view]
                start = (step * batch_size) % len(values)
                if start + batch_size <= len(values):
                    selected = values[start : start + batch_size]
                else:
                    selected = np.concatenate([values[start:], values[: batch_size - (len(values) - start)]])
                selected_by_view[view] = selected
            loss = balanced_step(selected_by_view)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach())
        validation: dict[str, Any] = {}
        validation_losses = []
        for view in VIEWS:
            selected = splits[view][1]
            if len(selected) > 100_000:
                selected = selected[np.linspace(0, len(selected) - 1, 100_000, dtype=np.int64)]
            result = _evaluate_pairs(
                model=model,
                accessor=accessor,
                relation=relations[view],
                indices=selected,
                lambda_pairwise=STUDENT_CONTRACT["pairwise_lambda"],
            )
            validation[view] = result
            validation_losses.append(result["loss"])
        mean_validation = float(np.mean(validation_losses))
        history.append({
            "epoch": epoch,
            "train_equal_view_mean_loss": train_loss / max(steps, 1),
            "validation_equal_view_mean_loss": mean_validation,
            "validation_by_view": validation,
        })
        print(
            f"P3.3 Student epoch {epoch}: train={history[-1]['train_equal_view_mean_loss']:.5f} "
            f"valid={mean_validation:.5f}",
            flush=True,
        )
        if mean_validation < best_loss - 1e-5:
            best_loss = mean_validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale > STUDENT_CONTRACT["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("P3.3 Student did not produce a checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "student_model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "variant": "multimodal", "cardinalities": cardinalities},
        model_path,
    )
    return model, {
        "best_epoch": best_epoch,
        "best_equal_view_validation_loss": best_loss,
        "train_rows_by_view": {view: int(len(splits[view][0])) for view in VIEWS},
        "validation_rows_by_view": {view: int(len(splits[view][1])) for view in VIEWS},
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if accessor.device.type == "cuda" else 0,
        "model": file_identity(model_path),
    }


def _train_one_student(
    *, label: str, cutoffs: list[str], teacher_dir: Path, static_dir: Path,
    fashionclip_root: Path, output_dir: Path, device: torch.device,
) -> dict[str, Any]:
    relation_paths = _load_relation_paths(teacher_dir, cutoffs)
    relations = {
        view: _concatenate_relations(relation_paths[view]) for view in VIEWS
    }
    catalog_items = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str})[
        "article_id"
    ].tolist()
    categories = np.load(static_dir / "catalog_categories.int32.npy", mmap_mode="r")
    cardinalities = [int(value) for value in (np.max(categories, axis=0) + 1)]
    accessor = CatalogAccessor(fashionclip_root=fashionclip_root, catalog_dir=static_dir, device=device)
    model, training = _balanced_train_encoder(
        relations=relations,
        catalog_items=catalog_items,
        accessor=accessor,
        cardinalities=cardinalities,
        output_dir=output_dir,
    )
    embedding_path = output_dir / "catalog_embeddings.float16.npy"
    encoding = encode_catalog(
        model=model, accessor=accessor, rows=len(catalog_items), output_path=embedding_path
    )
    diagnostics = {
        view: representation_diagnostics(
            embeddings_path=embedding_path,
            relation_paths=relation_paths[view],
            catalog_items=catalog_items,
            device=device,
        )
        for view in VIEWS
    }
    manifest = {
        "schema_version": "phase3-p3.3-content-only-student-v1",
        "stage": "P3.3 Student",
        "status": "completed",
        "run_id": RUN_ID,
        "label": label,
        "training_cutoffs": cutoffs,
        "contract": STUDENT_CONTRACT,
        "training": training,
        "encoding": encoding,
        "diagnostics": diagnostics,
        "teacher_relations": {
            view: [file_identity(path) for path in relation_paths[view]] for view in VIEWS
        },
        "student_inference_audit": {
            "content_only": True,
            "inputs": STUDENT_CONTRACT["inference_inputs"],
            "forbidden_teacher_embeddings": ["Item2Vec", "direct co-vis", "DeepWalk"],
            "teacher_embedding_used_at_inference": False,
        },
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_week": "not_run",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    del model, relations, accessor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return manifest


def train_multiview_students(
    *, source_root: Path, repo_root: Path, teacher_dir: Path, artifact_dir: Path,
    device_name: str, reuse_completed: bool,
) -> dict[str, Any]:
    device = _device(device_name)
    static_dir = repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation" / "student-v1" / "static_catalog"
    fashionclip = source_root / "artifacts" / "m2_4" / "fashionclip-full-v1"
    outputs: dict[str, Any] = {"training": {}, "outer_validation": {}}
    for cutoff in sorted({value for p in ROLLING_PROTOCOL.values() for value in p["outer_train"]}):
        output_dir = artifact_dir / "training" / cutoff
        manifest_path = output_dir / "manifest.json"
        if reuse_completed and manifest_path.is_file():
            manifest = _json(manifest_path)
            embedding = output_dir / "catalog_embeddings.float16.npy"
            if manifest.get("status") == "completed" and file_identity(embedding)["sha256"] == manifest["encoding"]["artifact"]["sha256"]:
                print(f"P3.3 reuse PIT Student {cutoff}", flush=True)
                outputs["training"][cutoff] = manifest
                continue
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite incomplete P3.3 Student: {output_dir}")
        print(f"P3.3 train PIT Student {cutoff}", flush=True)
        outputs["training"][cutoff] = _train_one_student(
            label=f"training:{cutoff}", cutoffs=[cutoff], teacher_dir=teacher_dir,
            static_dir=static_dir, fashionclip_root=fashionclip, output_dir=output_dir,
            device=device,
        )
    for window, protocol in ROLLING_PROTOCOL.items():
        cutoff = protocol["outer_validation"]
        output_dir = artifact_dir / "outer" / window
        manifest_path = output_dir / "manifest.json"
        if reuse_completed and manifest_path.is_file():
            manifest = _json(manifest_path)
            embedding = output_dir / "catalog_embeddings.float16.npy"
            if manifest.get("status") == "completed" and file_identity(embedding)["sha256"] == manifest["encoding"]["artifact"]["sha256"]:
                print(f"P3.3 reuse outer Student {window}", flush=True)
                outputs["outer_validation"][window] = manifest
                continue
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite incomplete P3.3 Student: {output_dir}")
        print(f"P3.3 train outer Student {window}", flush=True)
        outputs["outer_validation"][window] = _train_one_student(
            label=f"outer_validation:{window}", cutoffs=list(protocol["outer_train"]),
            teacher_dir=teacher_dir, static_dir=static_dir, fashionclip_root=fashionclip,
            output_dir=output_dir, device=device,
        )
    return {
        "schema_version": "phase3-p3.3-content-only-student-summary-v1",
        "status": "measured",
        "run_id": RUN_ID,
        "contract": STUDENT_CONTRACT,
        "assets": outputs,
        "resources": {"peak_working_set_bytes": _peak_working_set_bytes()},
        "final_week": "not_run",
    }


def _embedding_map(student_dir: Path) -> dict[tuple[str, str], Path]:
    result = {
        ("training", cutoff): student_dir / "training" / cutoff / "catalog_embeddings.float16.npy"
        for cutoff in sorted({value for p in ROLLING_PROTOCOL.values() for value in p["outer_train"]})
    }
    for window, protocol in ROLLING_PROTOCOL.items():
        result[("outer_validation", protocol["outer_validation"])] = (
            student_dir / "outer" / window / "catalog_embeddings.float16.npy"
        )
    return result


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _compare(
    *, baseline_p31: dict[str, Any], baseline_p32: dict[str, Any],
    multiview_p31: dict[str, Any], multiview_p32: dict[str, Any],
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    retrieval_strict_non_degrade = []
    retrieval_sparse_non_degrade = []
    density_non_degrade = []
    mrr_non_degrade = []
    conversion_non_degrade = []
    for window in ROLLING_PROTOCOL:
        base_coarse = baseline_p31["windows"][window]["coarse_metrics"]
        new_coarse = multiview_p31["windows"][window]["coarse_metrics"]
        base_rank = baseline_p32["windows"][window]["metrics"]["candidate_aware_order"]
        new_rank = multiview_p32["windows"][window]["metrics"]["candidate_aware_order"]
        values = {
            "strict_recall200": (
                float(base_coarse["at_k"]["200"]["segments"]["strict_cold"]["recall"]),
                float(new_coarse["at_k"]["200"]["segments"]["strict_cold"]["recall"]),
            ),
            "sparse_recall200": (
                float(base_coarse["at_k"]["200"]["segments"]["sparse_1_5"]["recall"]),
                float(new_coarse["at_k"]["200"]["segments"]["sparse_1_5"]["recall"]),
            ),
            "density20": (
                float(base_rank["at_k"]["20"]["positive_density"]),
                float(new_rank["at_k"]["20"]["positive_density"]),
            ),
            "mrr": (
                float(base_rank["ranking"]["mrr"]),
                float(new_rank["ranking"]["mrr"]),
            ),
            "top200_to_top20": (
                float(base_rank["ranking"]["conversion"]["top200_to_top20"]),
                float(new_rank["ranking"]["conversion"]["top200_to_top20"]),
            ),
        }
        windows[window] = {
            name: {"single_teacher": old, "multiview_teacher": new, "delta": new - old}
            for name, (old, new) in values.items()
        }
        retrieval_strict_non_degrade.append(values["strict_recall200"][1] >= values["strict_recall200"][0])
        retrieval_sparse_non_degrade.append(values["sparse_recall200"][1] >= values["sparse_recall200"][0])
        density_non_degrade.append(values["density20"][1] >= values["density20"][0])
        mrr_non_degrade.append(values["mrr"][1] >= values["mrr"][0])
        conversion_non_degrade.append(values["top200_to_top20"][1] >= values["top200_to_top20"][0])
    means = {}
    for metric in ("strict_recall200", "sparse_recall200", "density20", "mrr", "top200_to_top20"):
        old = _mean([windows[window][metric]["single_teacher"] for window in windows])
        new = _mean([windows[window][metric]["multiview_teacher"] for window in windows])
        means[metric] = {
            "single_teacher": old,
            "multiview_teacher": new,
            "delta": new - old,
            "relative_delta": (new - old) / old if old else None,
        }
    representation_gate = {
        "strict_recall200_mean_improves": means["strict_recall200"]["delta"] > 0,
        "sparse_recall200_mean_improves": means["sparse_recall200"]["delta"] > 0,
        "strict_recall200_nondegrade_3_of_4": sum(retrieval_strict_non_degrade) >= 3,
        "sparse_recall200_nondegrade_3_of_4": sum(retrieval_sparse_non_degrade) >= 3,
    }
    end_to_end_mean = {
        metric: means[metric]["delta"] > 0 for metric in ("density20", "mrr", "top200_to_top20")
    }
    end_to_end_windows = {
        "density20_nondegrade_3_of_4": sum(density_non_degrade) >= 3,
        "mrr_nondegrade_3_of_4": sum(mrr_non_degrade) >= 3,
        "top200_to_top20_nondegrade_3_of_4": sum(conversion_non_degrade) >= 3,
    }
    end_to_end_gate = {
        "all_three_means_improve": all(end_to_end_mean.values()),
        "at_least_two_metrics_nondegrade_3_of_4": sum(end_to_end_windows.values()) >= 2,
    }
    return {
        "windows": windows,
        "means": means,
        "representation_gate": {
            "checks": representation_gate,
            "passed": all(representation_gate.values()),
        },
        "end_to_end_gate": {
            "mean_checks": end_to_end_mean,
            "window_checks": end_to_end_windows,
            "checks": end_to_end_gate,
            "passed": all(end_to_end_gate.values()),
        },
        "promotion_passed": all(representation_gate.values()) and all(end_to_end_gate.values()),
    }


def run_p33(
    *, source_root: Path, repo_root: Path, artifact_dir: Path, report_dir: Path,
    device_name: str = "cuda", reuse_completed: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    report_path = report_dir / "P3_3_metrics.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite formal P3.3 evidence: {report_path}")
    baseline_p31 = _json(report_dir / "P3_1_metrics.json")
    baseline_p32 = _json(report_dir / "P3_2_metrics.json")
    if baseline_p31.get("final_week") != "not_run" or baseline_p32.get("final_week") != "not_run":
        raise RuntimeError("P3.3 baseline final-week boundary failed")
    contract = {
        "schema_version": "phase3-p3.3-contract-v1",
        "stage": "P3.3",
        "status": "frozen_before_run",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "multi-view graph teacher to the same content-only Student",
        "teacher": GRAPH_CONTRACT,
        "student": STUDENT_CONTRACT,
        "reuse": "exact P3.1 Top200 and P3.2 candidate-aware pipeline, architecture and training policy",
        "prohibited": ["P3.2 tuning", "Two-Tower", "LightGCN", "PCA", "teacher embedding at inference", "final week"],
        "success_gate": {
            "representation": "strict and sparse Recall@200 means improve and each is non-degraded in at least 3/4 windows",
            "end_to_end": "density@20, MRR and Top200-to-Top20 means all improve; at least two are non-degraded in at least 3/4 windows",
        },
        "fallback": "if representation fails, stop at retrieval diagnosis; if it passes but ranking fails, retain retrieval-only evidence and do not tune the reranker",
        "final_week": "not_run",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_dir / "P3_3_experiment_contract.json", contract)
    teacher_summary = build_multiview_teachers(
        source_root=source_root,
        repo_root=repo_root,
        artifact_dir=artifact_dir / "teachers-v1",
        device_name=device_name,
        reuse_completed=reuse_completed,
    )
    student_summary = train_multiview_students(
        source_root=source_root,
        repo_root=repo_root,
        teacher_dir=artifact_dir / "teachers-v1",
        artifact_dir=artifact_dir / "students-v1",
        device_name=device_name,
        reuse_completed=reuse_completed,
    )
    pipeline_report = artifact_dir / "pipeline-reports-v1"
    pipeline_report.mkdir(parents=True, exist_ok=True)
    atomic_json(pipeline_report / "P3_0_metrics.json", {
        "schema_version": "phase3-p3.3-reuse-gate-v1",
        "stage": "P3.3 reuse gate",
        "status": "measured",
        "run_id": RUN_ID,
        "checks": {"teacher_complete": True, "student_content_only": True, "final_week_not_run": True},
        "final_week": "not_run",
    })
    m4_artifact = repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation"
    m5_artifact = repo_root / "artifacts" / "m5" / "m5-v1-cold-expert-admission"
    static_dir = m4_artifact / "student-v1" / "static_catalog"
    embeddings = _embedding_map(artifact_dir / "students-v1")
    multiview_p31 = run_p31(
        source_root=source_root,
        repo_root=repo_root,
        m4_artifact_dir=m4_artifact,
        m5_artifact_dir=m5_artifact,
        artifact_dir=artifact_dir / "pipeline-v1",
        report_dir=pipeline_report,
        device_name=device_name,
        reuse_completed=reuse_completed,
        student_embeddings=embeddings,
        static_catalog_dir=static_dir,
        run_id=RUN_ID,
    )
    multiview_p32 = run_p32(
        source_root=source_root,
        repo_root=repo_root,
        m4_artifact_dir=m4_artifact,
        m5_artifact_dir=m5_artifact,
        artifact_dir=artifact_dir / "pipeline-v1",
        report_dir=pipeline_report,
        device_name=device_name,
        student_embeddings=embeddings,
        static_catalog_dir=static_dir,
        run_id=RUN_ID,
    )
    comparison = _compare(
        baseline_p31=baseline_p31,
        baseline_p32=baseline_p32,
        multiview_p31=multiview_p31,
        multiview_p32=multiview_p32,
    )
    result = {
        "schema_version": "phase3-p3.3-multiview-graph-student-v1",
        "stage": "P3.3",
        "status": "measured",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "teachers": teacher_summary,
        "students": student_summary,
        "pipeline": {"p3_1": multiview_p31, "p3_2": multiview_p32},
        "comparison": comparison,
        "decision": (
            "promote_multiview_student_as_phase3_baseline"
            if comparison["promotion_passed"]
            else (
                "retain_multiview_as_retrieval_only_evidence"
                if comparison["representation_gate"]["passed"]
                else "stop_multiview_teacher_after_representation_failure"
            )
        ),
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "device": torch.cuda.get_device_name(0) if device_name == "cuda" else "CPU",
        },
        "artifacts": {
            "teacher_summary": file_identity(artifact_dir / "teachers-v1" / next(iter(teacher_summary["cutoffs"])) / "manifest.json"),
            "p3_1_metrics": file_identity(pipeline_report / "P3_1_metrics.json"),
            "p3_2_metrics": file_identity(pipeline_report / "P3_2_metrics.json"),
        },
        "final_week": "not_run",
    }
    atomic_json(report_path, result)
    for source_name, target_name in (
        ("candidate_density_audit.json", "P3_3_candidate_density_audit.json"),
        ("rank_funnel_audit.json", "P3_3_rank_funnel_audit.json"),
        ("attention_audit.json", "P3_3_attention_audit.json"),
        ("score_separation_audit.json", "P3_3_score_separation_audit.json"),
    ):
        shutil.copyfile(pipeline_report / source_name, report_dir / target_name)
    return result
