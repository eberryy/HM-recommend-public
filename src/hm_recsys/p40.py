from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .m2 import M2Config, _create_static_dimensions, build_category_maps, build_point_in_time_dataset
from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, WARM_MAP, atomic_json, file_identity
from .m54 import (
    SOURCE_FEATURES as M54_SOURCE_FEATURES,
    WARM_EVIDENCE_FEATURES,
    _add_no_decay_features,
    _prepare_pit_connection,
)
from .m55 import (
    _baseline_top12,
    _evaluate,
    _load_validation,
    _prepare,
    _rank_top12,
    _truth_and_counts,
    load_training_sample,
    train_ranker,
)
from .p37b import MaterializedFeatureAsset, _reload_outer_model, _score_model
from .p37b_model import FeaturePreprocessor
from .p40_contract import (
    EARLY_STOPPING_ROUNDS,
    FUSION_FEATURES,
    K_COLD,
    K_FINAL,
    K_WARM,
    MAX_BOOST_ROUNDS,
    NEGATIVES_PER_POSITIVE,
    RUN_ID,
    SEED,
    SOURCE_FEATURES,
)


VARIANTS = ("W0", "W1", "F0", "F1")
BRANCHES = ("warm_only", "cold_only", "warm_and_cold")
P37B_ROOT_NAME = "phase3-p3.7b-v1-time-aware-hybrid"
M54_ROOT_NAME = "m5-4-v1-temporal-oof-dual-channel"

# score cutoff -> (model window, model kind, training cutoffs)
B0_LINEAGE: dict[str, tuple[str, str, list[str]] | None] = {
    "2019-11-27": None,
    "2019-12-25": ("winter_20200122", "inner", ["2019-11-27"]),
    "2020-01-22": ("winter_20200122", "outer", ["2019-11-27", "2019-12-25"]),
    "2020-02-19": ("spring_20200318", "inner", ["2020-01-22"]),
    "2020-03-18": ("spring_20200318", "outer", ["2020-01-22", "2020-02-19"]),
    "2020-04-29": ("spring_20200318", "outer", ["2020-01-22", "2020-02-19"]),
    "2020-05-27": ("early_summer_20200624", "inner", ["2020-04-29"]),
    "2020-06-24": ("early_summer_20200624", "outer", ["2020-04-29", "2020-05-27"]),
    "2020-07-22": ("late_summer_20200819", "inner", ["2020-06-24"]),
    "2020-08-19": ("late_summer_20200819", "outer", ["2020-06-24", "2020-07-22"]),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _outer_window_for_cutoff(cutoff: str) -> str | None:
    for window, protocol in ROLLING_PROTOCOL.items():
        if str(protocol["outer_validation"]) == cutoff:
            return window
    return None


def _p31_asset(p31: dict[str, Any], cutoff: str) -> tuple[dict[str, Any], str, str]:
    window = _outer_window_for_cutoff(cutoff)
    role = "outer_validation" if window else "training"
    key = f"outer_validation:{window}" if window else f"training:{cutoff}"
    return p31["all_assets"][key], role, str(window or cutoff)


def _load_candidates(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _b0_score(
    *, cutoff: str, declared: dict[str, Any], role: str, label: str,
    repo_root: Path, device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    lineage = B0_LINEAGE[cutoff]
    candidates = _load_candidates(Path(declared["artifacts"]["candidates"]["path"]))
    if lineage is None:
        return None, None, {
            "score_cutoff": cutoff, "available": False, "lineage_safe": True,
            "fallback": "no-safe-upstream fallback: M4 coarse Top50",
            "model_training_cutoffs": [], "model_label_end": None,
        }
    window, kind, training_cutoffs = lineage
    latest_label_end = max(
        (datetime.strptime(value, "%Y-%m-%d") + timedelta(days=7)).date()
        for value in training_cutoffs
    ).isoformat()
    if latest_label_end > cutoff or any(value >= cutoff for value in training_cutoffs):
        raise RuntimeError(f"unsafe B0 lineage for {cutoff}")
    model_root = repo_root / "artifacts" / "phase3" / P37B_ROOT_NAME / "models-v1" / window
    model_path = model_root / ("B0_inner_selected.pt" if kind == "inner" else "B0_outer.pt")
    preprocessing_path = model_root / (
        "B0_inner_preprocessing.json" if kind == "inner" else "B0_outer_preprocessing.json"
    )
    feature_root = repo_root / "artifacts" / "phase3" / P37B_ROOT_NAME / "features-v1" / role / label
    manifest = _json(feature_root / "manifest.json")
    expected_rows = len(candidates["rank"])
    for name in ("m4_relation", "p33_relation", "p33_global", "user_state", "candidate_state"):
        if not Path(manifest["artifacts"][name]["path"]).is_file():
            raise FileNotFoundError(manifest["artifacts"][name]["path"])
    frozen = SimpleNamespace(candidates=candidates, cutoff=cutoff)
    feature = MaterializedFeatureAsset(
        frozen=frozen, root=feature_root,
        m4_relation_path=feature_root / "m4_relation.float32.npy",
        p33_relation_path=feature_root / "p33_relation.float32.npy",
        p33_global_path=feature_root / "p33_global.float32.npy",
        user_state_path=feature_root / "user_state.float32.npy",
        candidate_state_path=feature_root / "candidate_state.float32.npy",
        manifest_path=feature_root / "manifest.json",
    )
    preprocessing = FeaturePreprocessor.from_manifest(_json(preprocessing_path))
    model = _reload_outer_model(variant="B0", model_path=model_path, device=device)
    scores, ranks, _alpha, resources = _score_model(
        model=model, feature=feature, preprocessing=preprocessing, device=device,
        keep_alpha=False,
    )
    if len(scores) != expected_rows or len(ranks) != expected_rows:
        raise RuntimeError(f"B0 scoring row drift at {cutoff}")
    audit = {
        "score_cutoff": cutoff, "available": True, "lineage_safe": True,
        "model_training_cutoffs": training_cutoffs, "model_label_end": latest_label_end,
        "model_kind": kind, "model_window": window,
        "model": file_identity(model_path), "preprocessing": file_identity(preprocessing_path),
        "feature_manifest": file_identity(feature_root / "manifest.json"),
        "candidate_source": file_identity(Path(declared["artifacts"]["candidates"]["path"])),
        "scoring": resources,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(scores).copy(), np.asarray(ranks).copy(), audit


def _write_cold_shortlist(
    *, cutoff: str, declared: dict[str, Any], scores: np.ndarray | None,
    ranks: np.ndarray | None, output_path: Path, catalog_items: np.ndarray,
) -> dict[str, Any]:
    candidates = _load_candidates(Path(declared["artifacts"]["candidates"]["path"]))
    users = pd.read_csv(
        declared["artifacts"]["users"]["path"], dtype={"customer_id": str}
    )["customer_id"].to_numpy()
    if scores is None:
        selection_rank = candidates["rank"].astype(np.int32)
        b0_score = np.full(len(selection_rank), np.nan, dtype=np.float32)
        b0_rank = np.full(len(selection_rank), np.nan, dtype=np.float32)
        b0_available = np.zeros(len(selection_rank), dtype=np.uint8)
    else:
        assert ranks is not None
        selection_rank = ranks.astype(np.int32)
        b0_score = scores.astype(np.float32)
        b0_rank = ranks.astype(np.float32)
        b0_available = np.ones(len(selection_rank), dtype=np.uint8)
    keep = selection_rank <= K_COLD
    rows = np.flatnonzero(keep)
    frame = pd.DataFrame({
        "customer_id": users[candidates["user_index"][rows]],
        "article_id": catalog_items[candidates["catalog_row"][rows]],
        "cold_rank": selection_rank[rows].astype(np.int16),
        "cold_rank_pct": selection_rank[rows].astype(np.float32) / K_COLD,
        "b0_score": b0_score[rows],
        "b0_score_available": b0_available[rows],
        "b0_rank": b0_rank[rows],
        "b0_rank_pct": b0_rank[rows] / K_COLD,
        "m4_coarse_score": candidates["coarse_score"][rows].astype(np.float32),
        "m4_coarse_rank": candidates["rank"][rows].astype(np.int16),
        "m4_coarse_rank_pct": candidates["rank"][rows].astype(np.float32) / 200.0,
    })
    frame["b0_delta_available"] = frame["b0_score_available"]
    frame["b0_delta_vs_m4"] = np.where(
        frame["b0_delta_available"] == 1,
        frame["b0_score"] - frame["m4_coarse_score"], np.nan,
    )
    frame.sort_values(["customer_id", "cold_rank", "article_id"], inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.connect().execute(
        f"COPY frame TO {_literal(output_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
    )
    exact_source = bool(np.all(np.isin(rows, np.arange(len(candidates["rank"])))))
    return {
        "rows": len(frame), "users": int(frame["customer_id"].nunique()),
        "all_from_m4_top200": exact_source,
        "selection": "B0 Top50" if scores is not None else "M4 coarse Top50 fallback",
        "artifact": file_identity(output_path),
    }


def _build_union(warm_path: Path, cold_path: Path, output_path: Path) -> dict[str, Any]:
    con = duckdb.connect()
    try:
        warm_cols = ",".join(f"w.{name}" for name in WARM_EVIDENCE_FEATURES)
        con.execute(f"""
        COPY (
          WITH merged AS (
            SELECT coalesce(w.customer_id,c.customer_id) customer_id,
                   coalesce(w.article_id,c.article_id) article_id,
                   (w.article_id IS NOT NULL)::UTINYINT warm_present,
                   (c.article_id IS NOT NULL)::UTINYINT cold_present,
                   (w.article_id IS NOT NULL AND c.article_id IS NOT NULL)::UTINYINT both_present,
                   CASE WHEN w.article_id IS NOT NULL AND c.article_id IS NOT NULL THEN 'warm_and_cold'
                        WHEN w.article_id IS NOT NULL THEN 'warm_only' ELSE 'cold_only' END source_branch,
                   w.warm_rank,w.warm_rank_pct,w.warm_candidate_rank,w.warm_model_score,
                   w.warm_model_score_available,{warm_cols},c.* EXCLUDE(customer_id,article_id)
            FROM read_parquet({_literal(warm_path)}) w FULL OUTER JOIN read_parquet({_literal(cold_path)}) c
            USING(customer_id,article_id)
          )
          SELECT customer_id,article_id,row_number() OVER(PARTITION BY customer_id ORDER BY
                 CASE source_branch WHEN 'warm_and_cold' THEN 0 WHEN 'warm_only' THEN 1 ELSE 2 END,
                 warm_rank NULLS LAST,cold_rank NULLS LAST,article_id)::INTEGER candidate_rank,
                 * EXCLUDE(customer_id,article_id)
          FROM merged ORDER BY customer_id,candidate_rank,article_id
        ) TO {_literal(output_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
        """)
        row = con.execute(f"""
          WITH g AS (SELECT customer_id,count(*) n,count(DISTINCT article_id) u,
             min(candidate_rank) lo,max(candidate_rank) hi,count(DISTINCT candidate_rank) r
             FROM read_parquet({_literal(output_path)}) GROUP BY customer_id)
          SELECT count(*),sum(n),min(n),max(n),sum((n<>u OR lo<>1 OR hi<>n OR r<>n)::INTEGER)
          FROM g
        """).fetchone()
        branches = dict(con.execute(
            f"SELECT source_branch,count(*) FROM read_parquet({_literal(output_path)}) GROUP BY source_branch"
        ).fetchall())
    finally:
        con.close()
    if int(row[4]) != 0:
        raise RuntimeError("P4.0 union identity failure")
    return {
        "users": int(row[0]), "rows": int(row[1]), "min_rows": int(row[2]),
        "max_rows": int(row[3]), "invalid_groups": int(row[4]),
        "branch_rows": {name: int(branches.get(name, 0)) for name in BRANCHES},
        "artifact": file_identity(output_path),
    }


def _materialize_features(
    *, cutoff: str, union_path: Path, old_feature_path: Path, output_root: Path,
    transactions: Path, articles: Path, customers: Path,
) -> dict[str, Any]:
    novel_path = output_root / "novel_candidates.parquet"
    con = duckdb.connect()
    try:
        con.execute(f"""
          COPY (SELECT customer_id,article_id,
             row_number() OVER(PARTITION BY customer_id ORDER BY candidate_rank,article_id)::INTEGER candidate_rank,
             cold_present
          FROM read_parquet({_literal(union_path)}) u
          ANTI JOIN read_parquet({_literal(old_feature_path)}) o USING(customer_id,article_id)
          ORDER BY customer_id,candidate_rank,article_id)
          TO {_literal(novel_path)} (FORMAT PARQUET,COMPRESSION ZSTD)
        """)
        novel_rows = int(con.execute(f"SELECT count(*) FROM read_parquet({_literal(novel_path)})").fetchone()[0])
    finally:
        con.close()
    novel_features = output_root / "novel_features.parquet"
    if novel_rows:
        base_path = output_root / "novel_base_features.parquet"
        pit = _prepare_pit_connection(
            database_path=output_root / "pit-feature-build.duckdb",
            temp_dir=output_root / "pit-temp", transactions_path=transactions,
            articles_path=articles, customers_path=customers,
        )
        try:
            _create_static_dimensions(pit)
            build_point_in_time_dataset(
                pit, novel_path, {"cutoff": cutoff, "declared_rows": novel_rows}, base_path,
                M2Config(candidate_k=200, evaluation_role="development"),
                retrieval_features=["candidate_rank", "cold_present"], candidate_group_range=(1, 200),
            )
        finally:
            pit.close()
        _add_no_decay_features(
            base_path=base_path, output_path=novel_features, transactions_path=transactions,
            articles_path=articles, cutoff=cutoff,
        )
    con = duckdb.connect()
    try:
        old_columns = [row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_literal(old_feature_path)})"
        ).fetchall()]
        excluded = set(M54_SOURCE_FEATURES) | {"customer_id", "article_id", "target_cutoff"}
        shared = [name for name in old_columns if name not in excluded]
        if novel_rows:
            novel_columns = {row[0] for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({_literal(novel_features)})"
            ).fetchall()}
            missing = sorted(set(shared) - novel_columns)
            if missing:
                raise RuntimeError(f"novel PIT schema missing shared fields: {missing}")
            joined = (
                f"LEFT JOIN read_parquet({_literal(novel_features)}) n USING(customer_id,article_id)"
            )
            shared_select = ",".join(f"coalesce(o.{name},n.{name}) AS {name}" for name in shared)
        else:
            joined = ""
            shared_select = ",".join(f"o.{name}" for name in shared)
        feature_path = output_root / "features.parquet"
        con.execute(f"""
          COPY (
            WITH combined AS (
              SELECT DATE '{cutoff}' target_cutoff,u.*,{shared_select}
              FROM read_parquet({_literal(union_path)}) u
              LEFT JOIN read_parquet({_literal(old_feature_path)}) o USING(customer_id,article_id)
              {joined}
            )
            SELECT *,item_events_before_cutoff::BIGINT interaction_count_before_cutoff,
                   (item_events_before_cutoff=0)::UTINYINT strict_cold_flag,
                   (item_events_before_cutoff BETWEEN 1 AND 5)::UTINYINT sparse1_5_flag
            FROM combined ORDER BY customer_id,candidate_rank,article_id
          ) TO {_literal(feature_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
        """)
        audit = con.execute(f"""
          SELECT count(*),count(DISTINCT (customer_id,article_id)),sum(target),
                 count(*) FILTER(WHERE cold_present=1 AND item_events_before_cutoff>5),
                 count(*) FILTER(WHERE target_cutoff>=DATE '{FINAL_CUTOFF}')
          FROM read_parquet({_literal(feature_path)})
        """).fetchone()
    finally:
        con.close()
    if int(audit[0]) != int(audit[1]) or int(audit[3]) or int(audit[4]):
        raise RuntimeError(f"P4.0 feature invariant failed at {cutoff}: {audit}")
    return {
        "rows": int(audit[0]), "unique_rows": int(audit[1]), "positive_pairs": int(audit[2]),
        "novel_rows_recomputed": novel_rows,
        "reused_rows": int(audit[0]) - novel_rows,
        "cold_pool_t_le_5": int(audit[3]) == 0, "final_week_rejected": int(audit[4]) == 0,
        "artifact": file_identity(feature_path),
        "novel_candidates": file_identity(novel_path),
        "novel_features": file_identity(novel_features) if novel_rows else None,
    }


def materialize(
    *, repo_root: Path, source_root: Path, artifact_dir: Path, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    p31 = _json(repo_root / "reports" / "phase3" / "P3_1_metrics.json")
    p37b = _json(repo_root / "reports" / "phase3" / "P3_7B_metrics.json")
    m54 = _json(repo_root / "reports" / "m5_4" / "M5_4_metrics.json")
    if p37b["decision"]["selected_variant"] != "B0" or not p37b["decision"]["warm_cold_fusion_allowed"]:
        raise RuntimeError("P3.7B does not authorize B0 Fusion")
    catalog_path = repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation" / "student-v1" / "static_catalog" / "catalog_items.csv"
    catalog_items = pd.read_csv(catalog_path, dtype={"article_id": str}).sort_values("catalog_row")["article_id"].to_numpy()
    m3 = _json(source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json")
    transactions = Path(m3["inputs"]["transactions"]["path"])
    articles = source_root / "data" / "raw" / "articles.csv"
    customers = source_root / "data" / "raw" / "customers.csv"
    cutoffs: dict[str, Any] = {}
    lineage_audit: dict[str, Any] = {}
    for cutoff in sorted(B0_LINEAGE):
        print(f"P4.0 materialize {cutoff}", flush=True)
        root = artifact_dir / "cutoffs" / cutoff
        root.mkdir(parents=True, exist_ok=True)
        declared, role, label = _p31_asset(p31, cutoff)
        scores, ranks, lineage = _b0_score(
            cutoff=cutoff, declared=declared, role=role, label=label,
            repo_root=repo_root, device=device,
        )
        score_path = root / "b0_scores.float32.npy"
        rank_path = root / "b0_ranks.float32.npy"
        if scores is not None:
            np.save(score_path, scores.astype(np.float32), allow_pickle=False)
            np.save(rank_path, ranks.astype(np.float32), allow_pickle=False)
            lineage["score_artifact"] = file_identity(score_path)
            lineage["rank_artifact"] = file_identity(rank_path)
        cold_path = root / "cold_shortlist.parquet"
        cold = _write_cold_shortlist(
            cutoff=cutoff, declared=declared, scores=scores, ranks=ranks,
            output_path=cold_path, catalog_items=catalog_items,
        )
        old_root = repo_root / "artifacts" / "m5_4" / M54_ROOT_NAME / cutoff
        warm_path = old_root / "warm_shortlist.parquet"
        expected_warm = m54["cutoffs"][cutoff]["warm"]["artifact"]
        observed_warm = file_identity(warm_path)
        warm_exact = observed_warm["sha256"] == expected_warm["sha256"] and observed_warm["bytes"] == expected_warm["bytes"]
        if not warm_exact:
            raise RuntimeError(f"Warm Top150 identity drift at {cutoff}")
        union_path = root / "union_candidates.parquet"
        union = _build_union(warm_path, cold_path, union_path)
        features = _materialize_features(
            cutoff=cutoff, union_path=union_path, old_feature_path=old_root / "features.parquet",
            output_root=root, transactions=transactions, articles=articles, customers=customers,
        )
        if union["rows"] != features["rows"]:
            raise RuntimeError(f"candidate/feature conservation failed at {cutoff}")
        lineage_audit[cutoff] = lineage
        cutoffs[cutoff] = {
            "cutoff": cutoff, "role": role, "p31_label": label,
            "warm": {"exact_frozen_identity": warm_exact, "artifact": observed_warm},
            "cold": cold, "union": union, "features": features,
            "p31_candidate": file_identity(Path(declared["artifacts"]["candidates"]["path"])),
            "final_week": "not_run",
        }
        gc.collect()
    return cutoffs, lineage_audit


def _stable_order(frame: pd.DataFrame) -> np.ndarray:
    hashes = pd.util.hash_pandas_object(
        frame[["target_cutoff", "customer_id", "article_id"]], index=False,
        hash_key="p40fixedhash0000",
    ).to_numpy(dtype=np.uint64)
    return np.argsort(hashes ^ np.uint64(SEED), kind="stable")


def _boundary_sample_one(group: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    positives = group.index[group["target"].to_numpy() == 1].to_numpy()
    if not len(positives):
        return np.empty(0, dtype=np.int64), {"positive_rows": 0, "negative_rows": 0}
    neg = group.loc[group["target"] == 0].copy()
    cp = int(((group.loc[positives, "cold_present"]) == 1).sum())
    wp = len(positives) - cp
    selected: list[int] = []
    selected_set: set[int] = set()
    fills: dict[str, int] = {}

    def take(name: str, mask: pd.Series, quota: int) -> None:
        if quota <= 0:
            fills[name] = 0
            return
        pool = neg.loc[mask & ~neg.index.isin(selected_set)]
        ordered = pool.iloc[_stable_order(pool)]
        chosen = ordered.index.to_numpy(dtype=np.int64)[:quota]
        selected.extend(chosen.tolist())
        selected_set.update(chosen.tolist())
        fills[name] = len(chosen)

    warm_rank = pd.to_numeric(neg["warm_rank"], errors="coerce")
    cold_rank = pd.to_numeric(neg["cold_rank"], errors="coerce")
    take("cold_positive_warm_hard", neg["warm_present"].eq(1) & warm_rank.between(1, 75), 12 * cp)
    take("cold_positive_warm_boundary", neg["warm_present"].eq(1) & warm_rank.between(76, 150), 12 * cp)
    take("cold_positive_cold", neg["cold_present"].eq(1) & cold_rank.between(1, 50), 6 * cp)
    take("warm_positive_cold_hard", neg["cold_present"].eq(1) & cold_rank.between(1, 20), 12 * wp)
    take("warm_positive_cold_tail", neg["cold_present"].eq(1) & cold_rank.between(21, 50), 6 * wp)
    take("warm_positive_warm", neg["warm_present"].eq(1) & warm_rank.between(1, 150), 12 * wp)
    cap = min(NEGATIVES_PER_POSITIVE * len(positives), len(neg))
    spill_masks = [
        neg["warm_present"].eq(1),
        neg["cold_present"].eq(1) & cold_rank.le(20),
        pd.Series(True, index=neg.index),
    ]
    spill = 0
    for mask in spill_masks:
        if len(selected) >= cap:
            break
        before = len(selected)
        take(f"spill_{spill}", mask, cap - len(selected))
        spill += 1
        if len(selected) == before:
            continue
    chosen = np.concatenate([positives.astype(np.int64), np.asarray(selected, dtype=np.int64)])
    return chosen, {
        "positive_rows": len(positives), "cold_capable_positive_rows": cp,
        "warm_only_positive_rows": wp, "negative_rows": len(selected),
        "negative_cap": cap, "fills": fills,
    }


def load_boundary_training_sample(paths: list[Path]) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    selected_columns = list(dict.fromkeys([
        "target_cutoff", "customer_id", "article_id", "candidate_rank", "target",
        "warm_rank", "cold_rank", "warm_present", "cold_present", *FUSION_FEATURES,
    ]))
    frames: list[pd.DataFrame] = []
    audits: dict[str, Any] = {}
    for path in paths:
        con = duckdb.connect()
        try:
            frame = con.execute(f"""
              WITH pg AS (SELECT target_cutoff,customer_id FROM read_parquet({_literal(path)})
                GROUP BY target_cutoff,customer_id HAVING sum(target)>0)
              SELECT {','.join('s.' + name for name in selected_columns)}
              FROM read_parquet({_literal(path)}) s JOIN pg USING(target_cutoff,customer_id)
              ORDER BY s.customer_id,s.candidate_rank,s.article_id
            """).fetchdf()
        finally:
            con.close()
        selected_groups = []
        cutoff_audits = []
        for (_cutoff, _user), group in frame.groupby(["target_cutoff", "customer_id"], sort=False):
            chosen, audit = _boundary_sample_one(group)
            selected_groups.append(frame.loc[chosen])
            cutoff_audits.append(audit)
        sampled = pd.concat(selected_groups, ignore_index=True)
        sampled.sort_values(["target_cutoff", "customer_id", "candidate_rank", "article_id"], inplace=True)
        cutoff = str(sampled["target_cutoff"].iloc[0])[:10]
        audits[cutoff] = {
            "groups": len(cutoff_audits),
            "positive_rows": sum(row["positive_rows"] for row in cutoff_audits),
            "cold_capable_positive_rows": sum(row["cold_capable_positive_rows"] for row in cutoff_audits),
            "warm_only_positive_rows": sum(row["warm_only_positive_rows"] for row in cutoff_audits),
            "negative_rows": sum(row["negative_rows"] for row in cutoff_audits),
            "max_negative_per_positive": max(row["negative_rows"] / row["positive_rows"] for row in cutoff_audits),
            "quota_fills": {
                name: sum(row["fills"].get(name, 0) for row in cutoff_audits)
                for name in sorted({key for row in cutoff_audits for key in row["fills"]})
            },
        }
        frames.append(sampled)
    result = pd.concat(frames, ignore_index=True)
    sizes = result.groupby(["target_cutoff", "customer_id"], sort=False, observed=True).size().astype(int).tolist()
    if not len(result) or sum(sizes) != len(result):
        raise RuntimeError("P4.0 boundary sample conservation failure")
    if any(value["max_negative_per_positive"] > NEGATIVES_PER_POSITIVE for value in audits.values()):
        raise RuntimeError("P4.0 boundary sampler exceeded negative cap")
    return result, sizes, {
        "sample_rule": "cross-source boundary-aware deterministic sampling",
        "cutoffs": audits, "rows": len(result), "groups": len(sizes),
        "positive_rows": int(result["target"].sum()),
        "negative_rows": int((result["target"] == 0).sum()),
    }


def _candidate_metrics(frame: pd.DataFrame, truth: dict[str, set[str]]) -> dict[str, Any]:
    pools = {str(u): set(g["article_id"].astype(str)) for u, g in frame.groupby("customer_id", sort=False)}
    recalls, oracle = [], []
    for user, items in truth.items():
        hits = len(items & pools.get(user, set()))
        recalls.append(hits / len(items))
        oracle.append(min(hits, 12) / min(len(items), 12))
    return {"candidate_recall": float(np.mean(recalls)), "oracle_map@12": float(np.mean(oracle))}


def _cold_funnel(frame: pd.DataFrame, ranks: np.ndarray) -> dict[str, Any]:
    work = frame[["target", "source_branch", "item_events_before_cutoff"]].copy()
    work["rank"] = ranks
    result: dict[str, Any] = {}
    for segment, count_mask in {
        "all_cold_sparse": work["item_events_before_cutoff"].le(5),
        "strict_cold": work["item_events_before_cutoff"].eq(0),
        "sparse1_5": work["item_events_before_cutoff"].between(1, 5),
    }.items():
        selected = work.loc[(work["target"] == 1) & work["source_branch"].eq("cold_only") & count_mask, "rank"]
        result[segment] = {
            "union": int(len(selected)), "Top100": int((selected <= 100).sum()),
            "Top50": int((selected <= 50).sum()), "Top20": int((selected <= 20).sum()),
            "Top12": int((selected <= 12).sum()),
            "best": int(selected.min()) if len(selected) else None,
            "p25": float(selected.quantile(0.25)) if len(selected) else None,
            "median": float(selected.median()) if len(selected) else None,
            "p75": float(selected.quantile(0.75)) if len(selected) else None,
        }
    return result


def _composition(top12: pd.DataFrame) -> dict[str, Any]:
    total = len(top12)
    return {
        "all_slots": {
            name: {"count": int((top12["source_branch"] == name).sum()),
                   "share": float((top12["source_branch"] == name).mean()) if total else 0.0}
            for name in BRANCHES
        },
        "positive_slots": {
            "cold_only": int(((top12["source_branch"] == "cold_only") & (top12["target"] == 1)).sum()),
            "warm_and_cold": int(((top12["source_branch"] == "warm_and_cold") & (top12["target"] == 1)).sum()),
        },
        "slot_denominator": total,
    }


def _replacement(
    top12: pd.DataFrame, baseline_top12: pd.DataFrame, truth: dict[str, set[str]], counts: dict[str, int]
) -> dict[str, Any]:
    pred = {str(u): set(g["article_id"].astype(str)) for u, g in top12.groupby("customer_id", sort=False)}
    base = {str(u): set(g["article_id"].astype(str)) for u, g in baseline_top12.groupby("customer_id", sort=False)}
    inserted_positive = removed_positive = inserted_cs_positive = removed_warm_positive = 0
    inserted_cs_slots = 0
    for user, items in truth.items():
        inserted = pred.get(user, set()) - base.get(user, set())
        removed = base.get(user, set()) - pred.get(user, set())
        inserted_truth = inserted & items
        removed_truth = removed & items
        inserted_positive += len(inserted_truth)
        removed_positive += len(removed_truth)
        inserted_cs_positive += sum(counts.get(item, 0) <= 5 for item in inserted_truth)
        removed_warm_positive += sum(counts.get(item, 0) >= 21 for item in removed_truth)
        inserted_cs_slots += sum(counts.get(item, 0) <= 5 for item in inserted)
    return {
        "inserted_all_positive_pairs": inserted_positive,
        "removed_all_positive_pairs": removed_positive,
        "inserted_cold_sparse_positive_pairs": inserted_cs_positive,
        "removed_warm_positive_pairs": removed_warm_positive,
        "net_positive_pairs": inserted_positive - removed_positive,
        "inserted_cold_sparse_recommendation_pairs": inserted_cs_slots,
        "cold_sparse_insertion_efficiency": (
            inserted_cs_positive / inserted_cs_slots if inserted_cs_slots else None
        ),
    }


def _calibration(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    work = frame[["customer_id", "target", "cold_present", "warm_present", "warm_rank",
                  "b0_rank", "m4_coarse_rank"]].copy()
    work["cold_rank"] = pd.to_numeric(work["b0_rank"], errors="coerce").fillna(
        pd.to_numeric(work["m4_coarse_rank"], errors="coerce")
    )
    work["score"] = score
    outputs: dict[str, list[np.ndarray]] = {"cold_positive_vs_warm_boundary": [], "warm_positive_vs_cold_hard": []}
    for _user, group in work.groupby("customer_id", sort=False):
        cold_pos = group.loc[(group["target"] == 1) & (group["cold_present"] == 1), "score"].to_numpy()
        warm_boundary = group.loc[(group["target"] == 0) & (group["warm_present"] == 1) & pd.to_numeric(group["warm_rank"], errors="coerce").between(76, 150), "score"].to_numpy()
        warm_pos = group.loc[(group["target"] == 1) & (group["cold_present"] == 0), "score"].to_numpy()
        cold_hard = group.loc[(group["target"] == 0) & (group["cold_present"] == 1) & pd.to_numeric(group["cold_rank"], errors="coerce").between(1, 20), "score"].to_numpy()
        if len(cold_pos) and len(warm_boundary):
            outputs["cold_positive_vs_warm_boundary"].append((cold_pos[:, None] - warm_boundary[None, :]).reshape(-1))
        if len(warm_pos) and len(cold_hard):
            outputs["warm_positive_vs_cold_hard"].append((warm_pos[:, None] - cold_hard[None, :]).reshape(-1))
    result: dict[str, Any] = {}
    for name, chunks in outputs.items():
        gaps = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        result[name] = {
            "pair_rows": len(gaps), "pair_accuracy": float(np.mean(gaps > 0)) if len(gaps) else None,
            "score_gap_mean": float(np.mean(gaps)) if len(gaps) else None,
            "score_gap_median": float(np.median(gaps)) if len(gaps) else None,
        }
    return result


def _train_variant(
    *, name: str, inner_train_paths: list[Path], inner_valid_path: Path,
    outer_train_paths: list[Path], root: Path, inner_truth: dict[str, set[str]],
) -> tuple[lgb.Booster, dict[str, dict[int, int]], dict[str, Any], pd.DataFrame]:
    warm_only = name == "W1"
    inner_model_path = root / f"inner-{name}.txt"
    outer_model_path = root / f"outer-{name}.txt"
    map_path = root / f"outer-{name}-category-maps.json"
    if inner_model_path.is_file() and outer_model_path.is_file() and map_path.is_file():
        if name == "F1":
            _inner_train, _inner_sizes, inner_sample = load_boundary_training_sample(inner_train_paths)
            outer_train, _outer_sizes, outer_sample = load_boundary_training_sample(outer_train_paths)
        else:
            _inner_train, _inner_sizes, inner_sample = load_training_sample(
                paths=inner_train_paths, features=FUSION_FEATURES, warm_only=warm_only,
            )
            outer_train, _outer_sizes, outer_sample = load_training_sample(
                paths=outer_train_paths, features=FUSION_FEATURES, warm_only=warm_only,
            )
        model = lgb.Booster(model_file=str(outer_model_path))
        maps = _json(map_path)
        audit = {
            "inner_sampling": inner_sample,
            "inner_model": {"model": file_identity(inner_model_path),
                            "best_iteration": lgb.Booster(model_file=str(inner_model_path)).num_trees(),
                            "reused_after_fail_closed_summary_error": True},
            "outer_sampling": outer_sample,
            "outer_model": {"model": file_identity(outer_model_path),
                            "best_iteration": model.num_trees(),
                            "reused_after_fail_closed_summary_error": True},
            "category_maps": file_identity(map_path),
        }
        del _inner_train
        gc.collect()
        return model, maps, audit, outer_train
    if name == "F1":
        inner_train, inner_sizes, inner_sample = load_boundary_training_sample(inner_train_paths)
    else:
        inner_train, inner_sizes, inner_sample = load_training_sample(
            paths=inner_train_paths, features=FUSION_FEATURES, warm_only=warm_only,
        )
    inner_valid, inner_valid_sizes = _load_validation(
        path=inner_valid_path, features=FUSION_FEATURES, warm_only=warm_only,
    )
    inner_maps = build_category_maps(inner_train)
    inner_model, inner_evidence = train_ranker(
        train=inner_train, train_sizes=inner_sizes, validation=inner_valid,
        validation_sizes=inner_valid_sizes, validation_truth=inner_truth,
        features=FUSION_FEATURES, category_maps=inner_maps,
        output_path=inner_model_path,
    )
    rounds = inner_evidence["best_iteration"]
    del inner_model, inner_train, inner_valid
    gc.collect()
    if name == "F1":
        outer_train, outer_sizes, outer_sample = load_boundary_training_sample(outer_train_paths)
    else:
        outer_train, outer_sizes, outer_sample = load_training_sample(
            paths=outer_train_paths, features=FUSION_FEATURES, warm_only=warm_only,
        )
    outer_maps = build_category_maps(outer_train)
    atomic_json(map_path, outer_maps)
    model, outer_evidence = train_ranker(
        train=outer_train, train_sizes=outer_sizes, validation=None,
        validation_sizes=None, validation_truth=None, features=FUSION_FEATURES,
        category_maps=outer_maps, output_path=outer_model_path, rounds=rounds,
    )
    return model, outer_maps, {
        "inner_sampling": inner_sample, "inner_model": inner_evidence,
        "outer_sampling": outer_sample, "outer_model": outer_evidence,
        "category_maps": file_identity(map_path),
    }, outer_train


def _audit_cached_materialization(
    repo_root: Path, artifact_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-audit immutable cutoff assets after a fail-closed downstream diagnostic."""
    m54 = _json(repo_root / "reports" / "m5_4" / "M5_4_metrics.json")
    p31 = _json(repo_root / "reports" / "phase3" / "P3_1_metrics.json")
    cutoffs: dict[str, Any] = {}
    lineages: dict[str, Any] = {}
    for cutoff, specification in B0_LINEAGE.items():
        root = artifact_dir / "cutoffs" / cutoff
        warm_path = repo_root / "artifacts" / "m5_4" / M54_ROOT_NAME / cutoff / "warm_shortlist.parquet"
        cold_path = root / "cold_shortlist.parquet"
        union_path = root / "union_candidates.parquet"
        feature_path = root / "features.parquet"
        novel_path = root / "novel_candidates.parquet"
        for path in (warm_path, cold_path, union_path, feature_path, novel_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        warm_identity = file_identity(warm_path)
        expected = m54["cutoffs"][cutoff]["warm"]["artifact"]
        warm_exact = warm_identity["bytes"] == expected["bytes"] and warm_identity["sha256"] == expected["sha256"]
        if not warm_exact:
            raise RuntimeError(f"cached Warm identity drift at {cutoff}")
        con = duckdb.connect()
        try:
            cold_stats = con.execute(
                f"SELECT count(*),count(DISTINCT customer_id),min(cold_rank),max(cold_rank) FROM read_parquet({_literal(cold_path)})"
            ).fetchone()
            union_stats = con.execute(f"""
                WITH g AS (SELECT customer_id,count(*) n,count(DISTINCT article_id) u,min(candidate_rank) lo,max(candidate_rank) hi,count(DISTINCT candidate_rank) r
                  FROM read_parquet({_literal(union_path)}) GROUP BY customer_id)
                SELECT count(*),sum(n),min(n),max(n),sum((n<>u OR lo<>1 OR hi<>n OR r<>n)::INTEGER) FROM g
            """).fetchone()
            branch_rows = dict(con.execute(
                f"SELECT source_branch,count(*) FROM read_parquet({_literal(union_path)}) GROUP BY source_branch"
            ).fetchall())
            feature_stats = con.execute(f"""
                SELECT count(*),count(DISTINCT (customer_id,article_id)),sum(target),
                  count(*) FILTER(WHERE cold_present=1 AND item_events_before_cutoff>5),
                  count(*) FILTER(WHERE target_cutoff>=DATE '{FINAL_CUTOFF}')
                FROM read_parquet({_literal(feature_path)})
            """).fetchone()
            novel_rows = int(con.execute(f"SELECT count(*) FROM read_parquet({_literal(novel_path)})").fetchone()[0])
        finally:
            con.close()
        declared, role, label = _p31_asset(p31, cutoff)
        if int(union_stats[1]) != int(feature_stats[0]) or int(feature_stats[0]) != int(feature_stats[1]):
            raise RuntimeError(f"cached materialization conservation drift at {cutoff}")
        if specification is None:
            lineage = {"score_cutoff": cutoff, "available": False, "lineage_safe": True,
                       "fallback": "no-safe-upstream fallback: M4 coarse Top50",
                       "model_training_cutoffs": [], "model_label_end": None}
        else:
            window, kind, training_cutoffs = specification
            model_root = repo_root / "artifacts" / "phase3" / P37B_ROOT_NAME / "models-v1" / window
            model_path = model_root / ("B0_inner_selected.pt" if kind == "inner" else "B0_outer.pt")
            preprocessing_path = model_root / ("B0_inner_preprocessing.json" if kind == "inner" else "B0_outer_preprocessing.json")
            label_end = max(
                (datetime.strptime(value, "%Y-%m-%d") + timedelta(days=7)).date()
                for value in training_cutoffs
            ).isoformat()
            lineage = {"score_cutoff": cutoff, "available": True, "lineage_safe": label_end <= cutoff,
                       "model_training_cutoffs": training_cutoffs, "model_label_end": label_end,
                       "model_kind": kind, "model_window": window,
                       "model": file_identity(model_path), "preprocessing": file_identity(preprocessing_path),
                       "score_artifact": file_identity(root / "b0_scores.float32.npy"),
                       "rank_artifact": file_identity(root / "b0_ranks.float32.npy")}
        lineages[cutoff] = lineage
        cutoffs[cutoff] = {
            "cutoff": cutoff, "role": role, "p31_label": label,
            "warm": {"exact_frozen_identity": True, "artifact": warm_identity},
            "cold": {"rows": int(cold_stats[0]), "users": int(cold_stats[1]),
                     "all_from_m4_top200": int(cold_stats[2]) >= 1 and int(cold_stats[3]) <= K_COLD,
                     "selection": "B0 Top50" if specification else "M4 coarse Top50 fallback",
                     "artifact": file_identity(cold_path)},
            "union": {"users": int(union_stats[0]), "rows": int(union_stats[1]),
                      "min_rows": int(union_stats[2]), "max_rows": int(union_stats[3]),
                      "invalid_groups": int(union_stats[4]),
                      "branch_rows": {name: int(branch_rows.get(name, 0)) for name in BRANCHES},
                      "artifact": file_identity(union_path)},
            "features": {"rows": int(feature_stats[0]), "unique_rows": int(feature_stats[1]),
                         "positive_pairs": int(feature_stats[2]), "novel_rows_recomputed": novel_rows,
                         "reused_rows": int(feature_stats[0]) - novel_rows,
                         "cold_pool_t_le_5": int(feature_stats[3]) == 0,
                         "final_week_rejected": int(feature_stats[4]) == 0,
                         "artifact": file_identity(feature_path), "novel_candidates": file_identity(novel_path),
                         "novel_features": file_identity(root / "novel_features.parquet") if novel_rows else None},
            "p31_candidate": file_identity(Path(declared["artifacts"]["candidates"]["path"])),
            "final_week": "not_run",
        }
    return cutoffs, lineages


def _gate_for_variant(windows: dict[str, Any], variant: str) -> dict[str, Any]:
    overall = [windows[w]["evaluations"][variant]["segments"]["overall"]["map@12"] for w in windows]
    base = [windows[w]["evaluations"]["W0"]["segments"]["overall"]["map@12"] for w in windows]
    sparse = [windows[w]["evaluations"][variant]["segments"]["sparse_1_5"]["map@12"] for w in windows]
    sparse_base = [windows[w]["evaluations"]["W0"]["segments"]["sparse_1_5"]["map@12"] for w in windows]
    warm = [windows[w]["evaluations"][variant]["segments"]["warm_21_plus"]["map@12"] for w in windows]
    warm_base = [windows[w]["evaluations"]["W0"]["segments"]["warm_21_plus"]["map@12"] for w in windows]
    deltas = [value - ref for value, ref in zip(overall, base)]
    sparse_deltas = [value - ref for value, ref in zip(sparse, sparse_base)]
    top12 = [windows[w]["cold_truth_funnel"][variant]["all_cold_sparse"]["Top12"] for w in windows]
    return {
        "overall_gate": float(np.mean(overall)) > float(np.mean(base)) and sum(x >= 0 for x in deltas) >= 3 and min(deltas) >= -0.000300,
        "sparse_gate": float(np.mean(sparse)) > float(np.mean(sparse_base)) and sum(x >= 0 for x in sparse_deltas) >= 3,
        "cold_top12_gate": sum(x > 0 for x in top12) >= 2 and sum(top12) > 0,
        "warm_protection_gate": float(np.mean(warm)) >= float(np.mean(warm_base)),
        "mean_map@12": float(np.mean(overall)), "mean_delta_vs_W0": float(np.mean(deltas)),
        "non_degrade_count": sum(x >= 0 for x in deltas), "worst_window_delta": min(deltas),
        "mean_sparse_map@12": float(np.mean(sparse)), "sparse_non_degrade_count": sum(x >= 0 for x in sparse_deltas),
        "mean_warm_21_plus_map@12": float(np.mean(warm)),
        "cold_top12_windows": sum(x > 0 for x in top12), "pooled_cold_top12": sum(top12),
    }


def run(
    *, repo_root: Path, source_root: Path, artifact_dir: Path, report_dir: Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    contract = _json(report_dir / "P4_0_EXPERIMENT_CONTRACT.json")
    if contract.get("status") != "preregistered_before_formal_computation" or contract.get("run_id") != RUN_ID:
        raise RuntimeError("P4.0 preregistered contract missing or drifted")
    if contract.get("final_week") != "not_run":
        raise RuntimeError("P4.0 final-week boundary drift")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    completed_features = all(
        (artifact_dir / "cutoffs" / cutoff / "features.parquet").is_file()
        for cutoff in B0_LINEAGE
    )
    if completed_features:
        print("P4.0 reuse completed cutoff materialization after fail-closed post-training audit", flush=True)
        cutoffs, lineage = _audit_cached_materialization(repo_root, artifact_dir)
    else:
        cutoffs, lineage = materialize(
            repo_root=repo_root, source_root=source_root, artifact_dir=artifact_dir, device=device,
        )
    m3 = _json(source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json")
    transactions = Path(m3["inputs"]["transactions"]["path"])
    windows: dict[str, Any] = {}
    feature_usage: dict[str, Any] = {}
    calibrations: dict[str, Any] = {}
    funnels: dict[str, Any] = {}
    replacements: dict[str, Any] = {}
    sampling: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"P4.0 formal rolling window {window}", flush=True)
        root = artifact_dir / "models" / window
        root.mkdir(parents=True, exist_ok=True)
        feature = lambda cutoff: artifact_dir / "cutoffs" / str(cutoff) / "features.parquet"
        inner_train_paths = [feature(value) for value in protocol["inner_train"]]
        inner_valid_path = feature(protocol["inner_validation"])
        outer_train_paths = [feature(value) for value in protocol["outer_train"]]
        outer_valid_path = feature(protocol["outer_validation"])
        inner_users = duckdb.connect().execute(
            f"SELECT DISTINCT customer_id FROM read_parquet({_literal(inner_valid_path)}) ORDER BY customer_id"
        ).fetchnumpy()["customer_id"].astype(str).tolist()
        inner_truth, _ = _truth_and_counts(
            transactions_path=transactions, cutoff=str(protocol["inner_validation"]), users=inner_users,
        )
        models: dict[str, tuple[lgb.Booster, dict[str, dict[int, int]], pd.DataFrame]] = {}
        training: dict[str, Any] = {}
        for variant in ("W1", "F0", "F1"):
            model, maps, audit, outer_train = _train_variant(
                name=variant, inner_train_paths=inner_train_paths,
                inner_valid_path=inner_valid_path, outer_train_paths=outer_train_paths,
                root=root, inner_truth=inner_truth,
            )
            models[variant] = (model, maps, outer_train)
            training[variant] = audit
        union, _ = _load_validation(path=outer_valid_path, features=FUSION_FEATURES, warm_only=False)
        users = union.groupby("customer_id", sort=False).size().index.astype(str).tolist()
        truth, counts = _truth_and_counts(
            transactions_path=transactions, cutoff=str(protocol["outer_validation"]), users=users,
        )
        baseline_db = Path(m3["development"][window]["scoring"]["evaluation_db"])
        baseline_top12 = _baseline_top12(baseline_db)
        if set(baseline_top12["customer_id"].astype(str)) != set(users):
            raise RuntimeError(f"W0 full-user denominator drift at {window}")
        evaluations: dict[str, Any] = {}
        window_funnel: dict[str, Any] = {}
        window_replacement: dict[str, Any] = {}
        prediction = union[["target_cutoff", "customer_id", "article_id", "candidate_rank", "target", "source_branch"]].copy()
        baseline_join = baseline_top12.merge(
            union[["customer_id", "article_id", "target", "source_branch", "item_events_before_cutoff"]],
            on=["customer_id", "article_id"], how="left", validate="one_to_one",
        ).rename(columns={"warm_rank": "final_rank"})
        if baseline_join[["target", "source_branch"]].isna().any().any():
            raise RuntimeError(f"W0 Top12 absent from Warm150 union at {window}")
        w0_ranks = np.full(len(baseline_join), 999, dtype=np.int32)
        w0_eval = _evaluate(
            frame=baseline_join.assign(candidate_rank=baseline_join["final_rank"], user_history_events_12w=1, warm_rank=baseline_join["final_rank"]),
            top12=baseline_join, ranks=w0_ranks, baseline_top12=baseline_top12,
            truth=truth, counts=counts,
        )
        w0_eval.update(_candidate_metrics(baseline_join, truth))
        w0_eval["top12_source_composition"] = _composition(baseline_join)
        w0_eval["replacement_accounting"] = _replacement(baseline_join, baseline_top12, truth, counts)
        evaluations["W0"] = w0_eval
        window_funnel["W0"] = {
            segment: {"union": 0, "Top100": 0, "Top50": 0, "Top20": 0, "Top12": 0,
                      "best": None, "p25": None, "median": None, "p75": None}
            for segment in ("all_cold_sparse", "strict_cold", "sparse1_5")
        }
        window_replacement["W0"] = w0_eval["replacement_accounting"]
        window_calibration: dict[str, Any] = {}
        for variant, (model, maps, outer_train) in models.items():
            frame = union.loc[union["warm_present"] == 1].copy() if variant == "W1" else union
            score = model.predict(_prepare(frame, FUSION_FEATURES, maps))
            top12, ranks = _rank_top12(frame, score)
            evaluation = _evaluate(
                frame=frame, top12=top12, ranks=ranks, baseline_top12=baseline_top12,
                truth=truth, counts=counts,
            )
            evaluation.update(_candidate_metrics(frame, truth))
            evaluation["top12_source_composition"] = _composition(top12)
            evaluation["replacement_accounting"] = _replacement(top12, baseline_top12, truth, counts)
            evaluations[variant] = evaluation
            window_funnel[variant] = _cold_funnel(frame, ranks)
            window_replacement[variant] = evaluation["replacement_accounting"]
            prediction[f"score_{variant}"] = np.nan
            prediction.loc[frame.index, f"score_{variant}"] = score
            if variant in {"F0", "F1"}:
                outer_train_scores = model.predict(_prepare(outer_train, FUSION_FEATURES, maps))
                window_calibration[variant] = {
                    "outer_training_sample": _calibration(outer_train, outer_train_scores),
                    "outer_validation": _calibration(frame, score),
                }
            importance = {
                name: {"gain": float(gain), "split": int(split)}
                for name, gain, split in zip(
                    FUSION_FEATURES, model.feature_importance("gain"), model.feature_importance("split")
                )
            }
            feature_usage.setdefault(variant, {})[window] = importance
        if abs(evaluations["W0"]["segments"]["overall"]["map@12"] - WARM_MAP[window]) > 1e-12:
            raise RuntimeError(f"W0 exact MAP parity failed at {window}")
        prediction_path = root / "outer_predictions.parquet"
        con = duckdb.connect()
        con.execute(f"COPY prediction TO {_literal(prediction_path)} (FORMAT PARQUET,COMPRESSION ZSTD)")
        con.close()
        windows[window] = {
            "protocol": protocol, "evaluations": evaluations, "training": training,
            "cold_truth_funnel": window_funnel,
            "replacement_accounting": window_replacement,
            "cross_source_calibration": window_calibration,
            "prediction": file_identity(prediction_path), "final_week": "not_run",
        }
        calibrations[window] = window_calibration
        funnels[window] = window_funnel
        replacements[window] = window_replacement
        sampling[window] = {variant: training[variant]["outer_sampling"] for variant in training}
        del models, union, prediction
        gc.collect()
    f0_gate = _gate_for_variant(windows, "F0")
    f1_gate = _gate_for_variant(windows, "F1")
    f1_vs_f0 = [
        windows[w]["evaluations"]["F1"]["segments"]["overall"]["map@12"]
        - windows[w]["evaluations"]["F0"]["segments"]["overall"]["map@12"] for w in windows
    ]
    boundary_gate = float(np.mean(f1_vs_f0)) > 0 and sum(value >= 0 for value in f1_vs_f0) >= 3
    f1_gate["boundary_sampling_gate"] = boundary_gate
    if boundary_gate and all(f1_gate[key] for key in ("overall_gate", "sparse_gate", "cold_top12_gate", "warm_protection_gate")):
        decision = "promote_fusion_v2_boundary_sampling"
        selected = "F1"
    elif all(f0_gate[key] for key in ("overall_gate", "sparse_gate", "cold_top12_gate", "warm_protection_gate")):
        decision = "promote_fusion_v2_historical_sampling"
        selected = "F0"
    elif f1_gate["mean_delta_vs_W0"] > 0 and not f1_gate["cold_top12_gate"]:
        decision, selected = "fusion_improves_map_but_fails_cold_top12_gate", "W0"
    elif f1_gate["cold_top12_gate"] and not f1_gate["warm_protection_gate"]:
        decision, selected = "fusion_improves_cold_but_hurts_warm", "W0"
    else:
        decision, selected = "stop_phase4_p4_0_after_gate_failure", "W0"
    feature_summary = {
        variant: {
            name: {
                "gain": sum(w[name]["gain"] for w in by_window.values()),
                "split": sum(w[name]["split"] for w in by_window.values()),
                "windows_with_nonzero_gain": sum(w[name]["gain"] > 0 for w in by_window.values()),
            } for name in FUSION_FEATURES
        } for variant, by_window in feature_usage.items()
    }
    m55 = _json(repo_root / "reports" / "m5_5" / "M5_5_metrics.json")
    historical_comparison = {
        window: {
            "old_M5_5_dual_fusion_map@12": float(m55["windows"][window]["evaluations"]["dual_fusion"]["segments"]["overall"]["map@12"]),
            "P4_F0_B0_fusion_map@12": float(windows[window]["evaluations"]["F0"]["segments"]["overall"]["map@12"]),
            "delta_P4_F0_minus_old_M5_5": float(windows[window]["evaluations"]["F0"]["segments"]["overall"]["map@12"] - m55["windows"][window]["evaluations"]["dual_fusion"]["segments"]["overall"]["map@12"]),
        } for window in windows
    }
    result = {
        "schema_version": "phase4-p4.0-metrics-v1", "stage": "P4.0", "status": "measured",
        "run_id": RUN_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": file_identity(report_dir / "P4_0_EXPERIMENT_CONTRACT.json"),
        "cutoffs": cutoffs, "windows": windows,
        "historical_M5_cold_branch_comparison": historical_comparison,
        "gates": {"F0": f0_gate, "F1": f1_gate, "boundary_sampling": {
            "passed": boundary_gate, "window_delta_F1_minus_F0": dict(zip(windows, f1_vs_f0)),
            "mean_delta": float(np.mean(f1_vs_f0)), "non_degrade_count": sum(x >= 0 for x in f1_vs_f0),
        }},
        "decision": {"machine_decision": decision, "selected_variant": selected,
                     "P4_0_promoted": selected in {"F0", "F1"},
                     "P4_1": "deferred_waiting_for_manual_decision"},
        "audits": {"upstream_lineage": lineage, "sampling": sampling,
                   "feature_usage": feature_summary, "cross_source_calibration": calibrations,
                   "cold_truth_funnel": funnels, "replacement_accounting": replacements},
        "resources": {"elapsed_seconds": time.perf_counter() - started,
                      "peak_working_set_bytes": _peak_working_set_bytes(),
                      "device_for_b0_scoring": str(device)},
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P4_0_metrics.json", result)
    atomic_json(report_dir / "p4_0_candidate_union_audit.json", {"cutoffs": cutoffs, "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_upstream_lineage_audit.json", {"cutoffs": lineage, "all_safe": all(x["lineage_safe"] for x in lineage.values()), "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_sampling_audit.json", {"windows": sampling, "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_feature_contract.json", {"features": FUSION_FEATURES, "new_source_features": SOURCE_FEATURES, "forbidden_b0_internal_present": False, "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_feature_usage.json", {"variants": feature_summary, "gain_is_not_causal": True, "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_cross_source_calibration.json", {"windows": calibrations, "promotion_metric": False, "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_cold_truth_funnel.json", {"windows": funnels, "final_week": "not_run"})
    atomic_json(report_dir / "p4_0_replacement_accounting.json", {"windows": replacements, "final_week": "not_run"})
    return result
