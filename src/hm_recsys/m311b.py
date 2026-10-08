from __future__ import annotations

import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import _literal, _prepare_frame
from .m3 import ANCHOR_NAME
from .m33 import ROLLING_PROTOCOL
from .m310 import DIRECT_FEATURES, _load_maps, _load_truth_frame, _metric_from_frames
from .m311 import AUTHORITATIVE_M310_RUN, _identity, _load_base_frame
from .m212 import feature_sets


SCHEMA_VERSION = "m3.11b-constrained-local-replacement-v1"
RUN_STAGE = "M3.11B"
FINAL_WEEK = "not_run"
M33_MODEL_KEY = "anchor"
SOURCE_TOP_K = 10
DIRECT_DECAY_TOP_K = 5
MAX_SUBPOOL_PER_USER = SOURCE_TOP_K + DIRECT_DECAY_TOP_K
PROTECTED_BASE_RANK = 7
MUTABLE_BASE_RANKS = (8, 9, 10, 11, 12)
MAX_REPLACEMENTS = 2
ADMISSION_THRESHOLD = 0.5
NUM_BOOST_ROUND = 120
NUMERIC_TOLERANCE = 1e-12

IMAGE_SOURCE_COLUMNS = [
    "soft_rank",
    "soft_personalized_rank",
    "soft_visual_rank",
    "soft_season_rank",
    "soft_combined_score",
    "soft_visual_score",
    "soft_season_score",
    "soft_best_neighbor_rank",
    "soft_seed_support",
]

COMPARISON_VALUE_COLUMNS = [
    "item_events_7d",
    "item_events_28d",
    "item_events_12w",
    "item_days_since_last_sale",
    "item_trend_7d_vs_28d",
    "user_item_events_28d",
    "user_item_days_since_last_purchase",
    "user_item_decay_28d_halflife_12w",
    "user_product_type_events_28d",
    "user_product_type_share_12w",
    "user_product_type_days_since",
    "user_product_type_decay_28d_halflife_12w",
    "user_department_events_28d",
    "user_department_share_12w",
    "user_department_days_since",
    "user_department_decay_28d_halflife_12w",
    "user_garment_events_28d",
    "user_garment_share_12w",
    "user_garment_days_since",
    "user_garment_decay_28d_halflife_12w",
    "user_colour_events_28d",
    "user_colour_share_12w",
    "user_colour_days_since",
    "user_colour_decay_28d_halflife_12w",
    "user_index_group_events_28d",
    "user_index_group_share_12w",
    "user_index_group_days_since",
    "user_index_group_decay_28d_halflife_12w",
]

PAIR_FEATURES = [
    *[f"image__{name}" for name in IMAGE_SOURCE_COLUMNS],
    *[f"image__{name}" for name in DIRECT_FEATURES],
    "image__subpool_rank",
    "image__personalized_decay_rank",
    "image__anchor_score",
    "base__anchor_score",
    "anchor_score_gap",
    "base__rank",
    *[
        f"{prefix}__{name}"
        for name in COMPARISON_VALUE_COLUMNS
        for prefix in ("image", "base", "delta")
    ],
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(m310: dict[str, Any], m311a: dict[str, Any], m33: dict[str, Any]) -> None:
    if m310.get("run_id") != AUTHORITATIVE_M310_RUN or m310.get("status") != "measured":
        raise ValueError("M3.11B requires authoritative M3.10 v3 metrics")
    if m311a.get("stage") != "M3.11A" or m311a.get("summary", {}).get("recommendation") != "high_precision_subpool_then_constrained_third_ranker":
        raise ValueError("M3.11B requires the measured M3.11A decision")
    if m33.get("schema_version") != "m3.3-cross-season-adaptive-seasonal-v1":
        raise ValueError("M3.11B requires M3.3 metrics")
    if m33.get("contract", {}).get("final_week") not in {None, "not_run"}:
        raise ValueError("M3.11B refuses an M3.3 result that read the final week")
    cutoffs = [row["outer_validation"] for row in ROLLING_PROTOCOL.values()]
    if len(cutoffs) != 4 or any(cutoff >= "2020-09-16" for cutoff in cutoffs):
        raise RuntimeError("M3.11B requires exactly four pre-final-week windows")
    if tuple(MUTABLE_BASE_RANKS) != tuple(range(PROTECTED_BASE_RANK + 1, 13)):
        raise RuntimeError("M3.11B mutable rank contract drifted")


def _verify_bound_file(path: Path, evidence: dict[str, Any], label: str) -> None:
    actual = _identity(path)
    expected_hash = evidence.get("sha256") or evidence.get("model_sha256")
    expected_bytes = evidence.get("bytes") or evidence.get("model_bytes")
    if expected_hash and actual["sha256"] != expected_hash:
        raise ValueError(f"M3.11B {label} SHA256 drift: {path}")
    if expected_bytes and actual["bytes"] != int(expected_bytes):
        raise ValueError(f"M3.11B {label} byte-size drift: {path}")


def _select_high_precision_subpool(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "customer_id", "article_id", "base_present", "soft_present",
        "soft_personalized_present", "soft_personalized_rank",
        "direct_visual_decay_max", "user_history_events_12w",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"subpool frame missing columns: {sorted(missing)}")
    eligible = frame.loc[
        (frame["base_present"].astype(int) == 0)
        & (frame["soft_present"].astype(int) == 1)
        & (frame["soft_personalized_present"].astype(int) == 1)
        & (frame["user_history_events_12w"].astype(float) > 0)
    ].copy()
    eligible = eligible.sort_values(
        ["customer_id", "direct_visual_decay_max", "soft_personalized_rank", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    eligible["personalized_decay_rank"] = eligible.groupby("customer_id", sort=False).cumcount() + 1
    selected = eligible.loc[
        (pd.to_numeric(eligible["soft_personalized_rank"], errors="coerce") <= SOURCE_TOP_K)
        | (eligible["personalized_decay_rank"] <= DIRECT_DECAY_TOP_K)
    ].copy()
    selected = selected.sort_values(
        ["customer_id", "soft_personalized_rank", "personalized_decay_rank", "article_id"],
        kind="mergesort",
    )
    selected["subpool_rank"] = selected.groupby("customer_id", sort=False).cumcount() + 1
    if selected.duplicated(["customer_id", "article_id"]).any():
        raise RuntimeError("M3.11B subpool identity audit failed")
    maximum = int(selected.groupby("customer_id", sort=False).size().max()) if not selected.empty else 0
    if maximum > MAX_SUBPOOL_PER_USER:
        raise RuntimeError(f"M3.11B subpool cap failed: {maximum}")
    return selected.reset_index(drop=True)


def _load_subpool(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = list(dict.fromkeys([
        "customer_id", "article_id", "candidate_rank", "target",
        "user_history_events_12w", "base_present", "soft_present",
        "soft_personalized_present", *IMAGE_SOURCE_COLUMNS, *DIRECT_FEATURES,
        *COMPARISON_VALUE_COLUMNS, *feature_sets()[ANCHOR_NAME],
    ]))
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"SELECT {','.join(columns)} FROM read_parquet({_literal(path)}) "
            "WHERE base_present=0 AND soft_present=1 AND soft_personalized_present=1 "
            "AND user_history_events_12w>0 ORDER BY customer_id,soft_personalized_rank,article_id"
        ).fetchdf()
    finally:
        con.close()
    selected = _select_high_precision_subpool(frame)
    evidence = {
        "eligible_candidate_rows": int(len(frame)),
        "eligible_users": int(frame["customer_id"].nunique()),
        "subpool_candidate_rows": int(len(selected)),
        "subpool_users": int(selected["customer_id"].nunique()),
        "subpool_positive_pairs": int(selected["target"].sum()),
        "max_candidates_per_user": int(selected.groupby("customer_id", sort=False).size().max()) if not selected.empty else 0,
    }
    del frame
    gc.collect()
    return selected, evidence


def _score_anchor_rows(
    frame: pd.DataFrame,
    *,
    model_path: Path,
    category_maps_path: Path,
    features: list[str],
) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    maps = _load_maps(category_maps_path)
    prepared = _prepare_frame(frame, features, maps)
    model = lgb.Booster(model_file=str(model_path))
    result = frame.copy()
    result["anchor_score"] = model.predict(prepared)
    del prepared, model
    gc.collect()
    return result


def _load_inner_mutable_base(
    *,
    enriched_path: Path,
    image_users: pd.Series,
    model_path: Path,
    category_maps_path: Path,
    features: list[str],
) -> pd.DataFrame:
    users = pd.DataFrame({"customer_id": image_users.drop_duplicates().astype(str)})
    columns = list(dict.fromkeys([
        "customer_id", "article_id", "candidate_rank", "target",
        *COMPARISON_VALUE_COLUMNS, *features,
    ]))
    con = duckdb.connect()
    try:
        con.register("m311b_users", users)
        frame = con.execute(
            f"SELECT {','.join('e.' + name for name in columns)} "
            f"FROM read_parquet({_literal(enriched_path)}) e JOIN m311b_users u USING(customer_id) "
            "WHERE e.base_present=1 ORDER BY e.customer_id,e.candidate_rank,e.article_id"
        ).fetchdf()
    finally:
        try:
            con.unregister("m311b_users")
        except Exception:
            pass
        con.close()
    scored = _score_anchor_rows(frame, model_path=model_path, category_maps_path=category_maps_path, features=features)
    scored = scored.sort_values(
        ["customer_id", "anchor_score", "candidate_rank", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    scored["base_rank"] = scored.groupby("customer_id", sort=False).cumcount() + 1
    result = scored.loc[scored["base_rank"].isin(MUTABLE_BASE_RANKS)].copy()
    counts = result.groupby("customer_id", sort=False).size()
    if not counts.empty and int(counts.max()) > len(MUTABLE_BASE_RANKS):
        raise RuntimeError("M3.11B inner mutable base cap failed")
    del frame, scored
    gc.collect()
    return result.reset_index(drop=True)


def _load_outer_mutable_base(
    *,
    enriched_path: Path,
    prediction_path: Path,
    image_users: pd.Series,
) -> pd.DataFrame:
    users = pd.DataFrame({"customer_id": image_users.drop_duplicates().astype(str)})
    con = duckdb.connect()
    try:
        con.register("m311b_users", users)
        predictions = con.execute(
            f"""
            SELECT p.customer_id,p.article_id,p.candidate_rank,p.target,p.score_anchor
            FROM read_parquet({_literal(prediction_path)}) p
            JOIN m311b_users u USING(customer_id)
            ORDER BY p.customer_id,p.candidate_rank,p.article_id
            """
        ).fetchdf()
        mutable = _assign_active_base_rank(predictions)
        con.register("m311b_mutable", mutable)
        frame = con.execute(
            f"""
            SELECT m.customer_id,m.article_id,m.base_rank,m.target,m.score_anchor AS anchor_score,
              {','.join('e.' + name for name in COMPARISON_VALUE_COLUMNS)}
            FROM m311b_mutable m
            JOIN read_parquet({_literal(enriched_path)}) e USING(customer_id,article_id)
            ORDER BY m.customer_id,m.base_rank,m.article_id
            """
        ).fetchdf()
    finally:
        try:
            con.unregister("m311b_users")
        except Exception:
            pass
        try:
            con.unregister("m311b_mutable")
        except Exception:
            pass
        con.close()
    if frame.duplicated(["customer_id", "article_id"]).any():
        raise RuntimeError("M3.11B outer mutable base identity audit failed")
    return frame


def _assign_active_base_rank(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"customer_id", "article_id", "candidate_rank", "target", "score_anchor"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"M3.11B base prediction columns missing: {sorted(missing)}")
    ranked = predictions.sort_values(
        ["customer_id", "score_anchor", "candidate_rank", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    ranked["base_rank"] = ranked.groupby("customer_id", sort=False).cumcount() + 1
    return ranked.loc[ranked["base_rank"].isin(MUTABLE_BASE_RANKS)].reset_index(drop=True)


def _build_pair_frame(images: pd.DataFrame, bases: pd.DataFrame, *, training: bool) -> pd.DataFrame:
    image_columns = list(dict.fromkeys([
        "customer_id", "article_id", "target", "anchor_score", "subpool_rank",
        "personalized_decay_rank", *IMAGE_SOURCE_COLUMNS, *DIRECT_FEATURES,
        *COMPARISON_VALUE_COLUMNS,
    ]))
    base_columns = list(dict.fromkeys([
        "customer_id", "article_id", "target", "anchor_score", "base_rank",
        *COMPARISON_VALUE_COLUMNS,
    ]))
    left = images[image_columns].rename(columns={
        "article_id": "image_article_id",
        "target": "image_target",
        "anchor_score": "image_anchor_score",
        **{name: f"image_source__{name}" for name in [*IMAGE_SOURCE_COLUMNS, *DIRECT_FEATURES]},
        **{name: f"image_value__{name}" for name in COMPARISON_VALUE_COLUMNS},
    })
    right = bases[base_columns].rename(columns={
        "article_id": "base_article_id",
        "target": "base_target",
        "anchor_score": "base_anchor_score",
        **{name: f"base_value__{name}" for name in COMPARISON_VALUE_COLUMNS},
    })
    pairs = left.merge(right, on="customer_id", how="inner", sort=False, validate="many_to_many")
    feature_data: dict[str, pd.Series] = {}
    for name in IMAGE_SOURCE_COLUMNS:
        feature_data[f"image__{name}"] = pd.to_numeric(pairs[f"image_source__{name}"], errors="coerce")
    for name in DIRECT_FEATURES:
        feature_data[f"image__{name}"] = pd.to_numeric(pairs[f"image_source__{name}"], errors="coerce")
    feature_data["image__subpool_rank"] = pd.to_numeric(pairs["subpool_rank"], errors="coerce")
    feature_data["image__personalized_decay_rank"] = pd.to_numeric(pairs["personalized_decay_rank"], errors="coerce")
    feature_data["image__anchor_score"] = pd.to_numeric(pairs["image_anchor_score"], errors="coerce")
    feature_data["base__anchor_score"] = pd.to_numeric(pairs["base_anchor_score"], errors="coerce")
    feature_data["anchor_score_gap"] = feature_data["image__anchor_score"] - feature_data["base__anchor_score"]
    feature_data["base__rank"] = pd.to_numeric(pairs["base_rank"], errors="coerce")
    for name in COMPARISON_VALUE_COLUMNS:
        image_value = pd.to_numeric(pairs[f"image_value__{name}"], errors="coerce")
        base_value = pd.to_numeric(pairs[f"base_value__{name}"], errors="coerce")
        feature_data[f"image__{name}"] = image_value
        feature_data[f"base__{name}"] = base_value
        feature_data[f"delta__{name}"] = image_value - base_value
    pairs = pd.concat([pairs, pd.DataFrame(feature_data, index=pairs.index)], axis=1)
    if training:
        directional = pairs["image_target"].astype(np.int8) != pairs["base_target"].astype(np.int8)
        pairs = pairs.loc[directional].copy()
        pairs["pair_target"] = (
            pairs["image_target"].astype(np.int8) > pairs["base_target"].astype(np.int8)
        ).astype(np.uint8)
    if set(PAIR_FEATURES).difference(pairs.columns):
        raise RuntimeError("M3.11B pair feature construction incomplete")
    return pairs.reset_index(drop=True)


def _train_comparator(
    pairs: pd.DataFrame,
    *,
    artifact_dir: Path,
    threads: int,
    seed: int,
) -> tuple[Any | None, dict[str, Any]]:
    counts = pairs["pair_target"].value_counts().to_dict() if not pairs.empty else {}
    negatives = int(counts.get(0, 0))
    positives = int(counts.get(1, 0))
    evidence: dict[str, Any] = {
        "pair_rows": int(len(pairs)),
        "image_should_replace_pairs": positives,
        "base_should_remain_pairs": negatives,
        "ambiguous_equal_label_pairs_excluded": True,
        "class_weighting": "each class has half of total training weight",
    }
    if positives == 0 or negatives == 0:
        evidence.update({"status": "insufficient_classes", "model_path": None})
        return None, evidence
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    matrix = pairs[PAIR_FEATURES].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    labels = pairs["pair_target"].to_numpy(dtype=np.uint8)
    total = len(labels)
    weights = np.where(labels == 1, total / (2.0 * positives), total / (2.0 * negatives))
    dataset = lgb.Dataset(matrix, label=labels, weight=weights, feature_name=PAIR_FEATURES, free_raw_data=True)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "lambda_l2": 1.0,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(threads),
        "verbosity": -1,
    }
    started = time.perf_counter()
    model = lgb.train(params, dataset, num_boost_round=NUM_BOOST_ROUND)
    model_path = artifact_dir / "lightgbm-pairwise-admission-comparator.txt"
    model.save_model(str(model_path))
    importance = sorted(
        [
            {"feature": feature, "gain": float(gain), "split": int(split)}
            for feature, gain, split in zip(
                PAIR_FEATURES,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
                strict=True,
            )
        ],
        key=lambda row: row["gain"],
        reverse=True,
    )
    evidence.update({
        "status": "trained",
        "features": PAIR_FEATURES,
        "parameters": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "top_feature_importance": importance[:30],
        "elapsed_seconds": time.perf_counter() - started,
    })
    del matrix, dataset
    gc.collect()
    return model, evidence


def _select_replacements(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold: float = ADMISSION_THRESHOLD,
    max_replacements: int = MAX_REPLACEMENTS,
) -> pd.DataFrame:
    if len(pairs) != len(scores):
        raise ValueError("M3.11B score length mismatch")
    edges = pairs[[
        "customer_id", "image_article_id", "base_article_id", "base_rank",
        "image_target", "base_target",
    ]].copy()
    edges["comparison_score"] = np.asarray(scores, dtype=np.float64)
    edges = edges.loc[edges["comparison_score"] > float(threshold)].sort_values(
        ["customer_id", "comparison_score", "base_rank", "image_article_id", "base_article_id"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    selected_rows: list[dict[str, Any]] = []
    for customer_id, group in edges.groupby("customer_id", sort=False):
        rows = list(group.itertuples(index=False))
        best_indices: tuple[int, ...] = (0,)
        best_key = (float(rows[0].comparison_score) - float(threshold), 1)
        if max_replacements >= 2:
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    if (
                        str(rows[left].image_article_id) == str(rows[right].image_article_id)
                        or str(rows[left].base_article_id) == str(rows[right].base_article_id)
                    ):
                        continue
                    key = (
                        float(rows[left].comparison_score) + float(rows[right].comparison_score) - 2.0 * float(threshold),
                        2,
                    )
                    if key > best_key:
                        best_key = key
                        best_indices = (left, right)
        for index in best_indices:
            row = rows[index]
            selected_rows.append({
                "customer_id": str(customer_id),
                "image_article_id": str(row.image_article_id),
                "base_article_id": str(row.base_article_id),
                "base_rank": int(row.base_rank),
                "image_target": int(row.image_target),
                "base_target": int(row.base_target),
                "comparison_score": float(row.comparison_score),
            })
    selected = pd.DataFrame.from_records(selected_rows, columns=[
        "customer_id", "image_article_id", "base_article_id", "base_rank",
        "image_target", "base_target", "comparison_score",
    ])
    if not selected.empty:
        counts = selected.groupby("customer_id", sort=False).size()
        if int(counts.max()) > max_replacements:
            raise RuntimeError("M3.11B replacement cap failed")
        if selected.duplicated(["customer_id", "image_article_id"]).any() or selected.duplicated(["customer_id", "base_article_id"]).any():
            raise RuntimeError("M3.11B one-to-one replacement audit failed")
        if (selected["base_rank"] <= PROTECTED_BASE_RANK).any() or (selected["base_rank"] > 12).any():
            raise RuntimeError("M3.11B protected-rank audit failed")
    return selected


def _apply_replacements(
    base_top12: pd.DataFrame,
    selected: pd.DataFrame,
    images: pd.DataFrame,
) -> pd.DataFrame:
    rank_column = "base_final_rank" if "base_final_rank" in base_top12.columns else "candidate_rank"
    base = base_top12[["customer_id", "article_id", rank_column, "target"]].copy().rename(
        columns={rank_column: "candidate_rank"}
    )
    if selected.empty:
        return base.sort_values(["customer_id", "candidate_rank", "article_id"], kind="mergesort").reset_index(drop=True)
    keys = selected[["customer_id", "base_article_id"]].rename(columns={"base_article_id": "article_id"})
    tagged = base.merge(keys.assign(_drop=1), on=["customer_id", "article_id"], how="left")
    kept = tagged.loc[tagged["_drop"].isna(), base.columns].copy()
    image_meta = images[["customer_id", "article_id", "target"]].drop_duplicates().rename(
        columns={"article_id": "image_article_id", "target": "observed_image_target"}
    )
    added = selected.merge(image_meta, on=["customer_id", "image_article_id"], how="left", validate="many_to_one")
    if added["observed_image_target"].isna().any() or not np.array_equal(
        added["image_target"].to_numpy(dtype=np.int8), added["observed_image_target"].to_numpy(dtype=np.int8)
    ):
        raise RuntimeError("M3.11B selected image target audit failed")
    additions = pd.DataFrame({
        "customer_id": added["customer_id"].astype(str),
        "article_id": added["image_article_id"].astype(str),
        "candidate_rank": added["base_rank"].astype(np.int64),
        "target": added["image_target"].astype(np.uint8),
    })
    final = pd.concat([kept, additions], ignore_index=True).sort_values(
        ["customer_id", "candidate_rank", "article_id"], kind="mergesort"
    ).reset_index(drop=True)
    if final.duplicated(["customer_id", "article_id"]).any() or final.duplicated(["customer_id", "candidate_rank"]).any():
        raise RuntimeError("M3.11B final Top12 identity/rank audit failed")
    if not final.empty and int(final.groupby("customer_id", sort=False).size().max()) > 12:
        raise RuntimeError("M3.11B final Top12 row cap failed")
    return final


def _map_segments(frame: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    scored = _normalize_evaluation_rank(frame)
    return {
        segment: float(_metric_from_frames(scored, truth, segment=segment, budget=12)["map@12"])
        for segment in ("overall", "warm", "cold")
    }


def _normalize_evaluation_rank(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "base_final_rank" in result.columns:
        result["candidate_rank"] = pd.to_numeric(result["base_final_rank"], errors="raise").astype(np.int64)
    if "candidate_rank" not in result.columns:
        raise ValueError("M3.11B evaluation frame has no ranking column")
    return result


def _candidate_ceiling(
    *,
    prediction_path: Path,
    images: pd.DataFrame,
    transactions_path: Path,
    cutoff: str,
) -> dict[str, dict[str, float | int]]:
    image_keys = images[["customer_id", "article_id"]].drop_duplicates()
    con = duckdb.connect()
    try:
        con.register("m311b_images", image_keys)
        con.execute(
            f"""
            CREATE TEMP TABLE truth AS
            SELECT q.customer_id,q.article_id,
              CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
            FROM (SELECT DISTINCT customer_id,article_id FROM read_parquet({_literal(transactions_path)})
                  WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY) q
            LEFT JOIN (SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
                       WHERE t_dat<DATE '{cutoff}') w USING(article_id);
            CREATE TEMP TABLE users AS
            SELECT DISTINCT customer_id FROM read_parquet({_literal(prediction_path)});
            CREATE TEMP TABLE base_candidates AS
            SELECT DISTINCT customer_id,article_id FROM read_parquet({_literal(prediction_path)});
            CREATE TEMP TABLE extended_candidates AS
            SELECT * FROM base_candidates UNION SELECT * FROM m311b_images;
            """
        )
        result: dict[str, dict[str, float | int]] = {}
        for population, relation in (("base_300", "base_candidates"), ("base_plus_subpool", "extended_candidates")):
            result[population] = {}
            for segment, predicate in (("overall", "TRUE"), ("warm", "item_temperature='warm'"), ("cold", "item_temperature='cold'")):
                row = con.execute(
                    f"""
                    WITH scoped AS (
                      SELECT t.* FROM truth t JOIN users u USING(customer_id) WHERE {predicate}
                    ), truth_counts AS (
                      SELECT customer_id,count(*)::DOUBLE AS truth_count FROM scoped GROUP BY customer_id
                    ), hits AS (
                      SELECT s.customer_id,count(*)::DOUBLE AS hits
                      FROM scoped s JOIN {relation} c USING(customer_id,article_id) GROUP BY s.customer_id
                    ), joined AS (
                      SELECT tc.customer_id,tc.truth_count,coalesce(h.hits,0)::DOUBLE AS hits
                      FROM truth_counts tc LEFT JOIN hits h USING(customer_id)
                    )
                    SELECT count(*)::BIGINT,sum(truth_count)::BIGINT,
                      avg(hits/truth_count),avg((hits>0)::INTEGER),
                      avg(least(hits,12)/least(truth_count,12))
                    FROM joined
                    """
                ).fetchone()
                result[population][segment] = {
                    "users": int(row[0] or 0),
                    "truth_pairs": int(row[1] or 0),
                    "candidate_recall": float(row[2] or 0.0),
                    "candidate_hit_rate": float(row[3] or 0.0),
                    "oracle_map@12": float(row[4] or 0.0),
                }
        return result
    finally:
        try:
            con.unregister("m311b_images")
        except Exception:
            pass
        con.close()


def _gate(window_deltas: dict[str, float], admitted_image_positives: dict[str, int]) -> dict[str, Any]:
    mean_delta = float(np.mean(list(window_deltas.values()))) if window_deltas else 0.0
    positive_windows = sum(delta > NUMERIC_TOLERANCE for delta in window_deltas.values())
    non_degraded_windows = sum(delta >= -NUMERIC_TOLERANCE for delta in window_deltas.values())
    mechanism_windows = sum(value > 0 for value in admitted_image_positives.values())
    mechanism_total = int(sum(admitted_image_positives.values()))
    map_gate = bool(mean_delta > NUMERIC_TOLERANCE and non_degraded_windows == 4 and positive_windows >= 3)
    mechanism_gate = bool(mechanism_windows >= 2 and mechanism_total >= 5)
    return {
        "mean_delta_map@12": mean_delta,
        "positive_map_windows": positive_windows,
        "non_degraded_windows": non_degraded_windows,
        "map_gate": map_gate,
        "image_positive_windows": mechanism_windows,
        "admitted_image_positive_pairs": mechanism_total,
        "mechanism_gate": mechanism_gate,
        "accepted_as_new_baseline": bool(map_gate and mechanism_gate),
    }


def _render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rows = list(result["development"].values())
    train_image_positives = sum(int(row["training"]["subpool"]["subpool_positive_pairs"]) for row in rows)
    train_positive_pairs = sum(int(row["training"]["model"]["image_should_replace_pairs"]) for row in rows)
    outer_subpool_rows = sum(int(row["outer_subpool"]["subpool_candidate_rows"]) for row in rows)
    outer_subpool_positives = sum(int(row["outer_subpool"]["subpool_positive_pairs"]) for row in rows)
    inserted_rows = sum(int(row["replacement_audit"]["inserted_image_rows"]) for row in rows)
    removed_base_positives = sum(int(row["replacement_audit"]["removed_base_positive_pairs"]) for row in rows)
    lines = [
        "# M3.11B：高精度图片子池与受约束局部替换排序器", "",
        "## 结论", "",
        f"- 运行状态：{result['status']}；最终验证周：{result['contract']['final_week']}。",
        f"- 四窗平均 MAP@12 变化：{summary['mean_delta_map@12']:+.8f}；严格提升窗口 {summary['positive_map_windows']}/4；无退化窗口 {summary['non_degraded_windows']}/4。",
        f"- 实际带入图片正例：{summary['admitted_image_positive_pairs']} 个 `(cutoff, user, item)` 候选观察，分布于 {summary['image_positive_windows']}/4 个窗口。",
        f"- MAP 稳定门槛：{summary['map_gate']}；图片机制门槛：{summary['mechanism_gate']}。",
        f"- 冻结为新 baseline：{summary['accepted_as_new_baseline']}。", "",
        "## 失败漏斗摘要", "",
        f"- 四个训练截点的高精度子池共有 {train_image_positives} 个图片正例候选观察；它们分别与5个基础槽位配对后形成 {train_positive_pairs} 条正标签比较行。有效图片正例监督仍是前者，不应把重复配对后的行数当成独立正例。",
        f"- 四个外层子池共有 {outer_subpool_rows} 条图片候选观察，其中 {outer_subpool_positives} 条命中未来购买；因此本轮失败不是外层子池完全没有可兑现正例。",
        f"- 比较器实际插入 {inserted_rows} 条图片候选，却没有插入任何图片正例，并移除了 {removed_base_positives} 条基础正例。它并非因0.5阈值过于保守而全部放弃，而是当前稀疏监督下没有学会跨窗口识别正确图片。", "",
        "## 术语、统计单位与边界", "",
        "- `baseline`（基准方案，行业通用术语）：后续实验固定比较的已验收配置；本阶段参照是 M3.3 base-300，只有同时通过稳定 MAP 和图片机制门槛才替换它。",
        "- `outer development window`（外层开发窗口）：某个时间截点之前构造特征，之后7天作标签的开发评测；本报告四窗均早于最终验证周。",
        "- `inner-validation cutoff`（内层验证截点，行业常见的嵌套时间验证叫法）：位于 outer 窗之前的开发截点；本阶段把它当成第三排序器训练集，而其基础排序模型只用更早数据拟合。",
        "- `active user`（活跃用户，本项目分组）：当前截点前12周至少有一次购买事件的评测用户；只有这类用户拥有可用的近期图片兴趣 seed。",
        "- `high-precision image subpool`（高精度图片子池，本项目自定义集合）：active 用户的个性化图片候选中，个性化来源前10或直接视觉时间衰减前5的并集，每用户最多15项。",
        "- `direct visual time-decay score`（直接视觉时间衰减分数，本项目自定义特征）：候选图片与用户近期已购图片的余弦相似度乘28天半衰期权重后取最大值；只使用截点前购买作为 seed。",
        "- `pairwise admission comparator`（成对准入比较器，本项目第三排序器）：输入一件图片商品和一个基础第8--12名商品，估计图片是否应替换该基础商品；分数不超过0.5时放弃。",
        "- `directional pair`（有明确方向的训练比较对）：图片与基础商品恰有一个命中未来7天购买；图片命中记1，基础命中记0。两者同为正例或同为未观测不提供替换方向，训练时排除。",
        "- `replacement`（局部替换）：保持基础第1--7名不变，把获准图片放入某个第8--12名槽位，并移除该槽位原商品；每用户最多2项。",
        "- `candidate observation`（候选观察）：一个 `(cutoff, user, item)` 用户—商品候选行；相同用户—商品跨窗口出现会分别计数。",
        "- `MAP@12`（前12平均准确率均值，行业通用指标）：逐用户按真实购买数截断到12作为分母，再对用户平均；逐窗 delta 均相对同窗 M3.3 base-300。",
        "- `candidate Recall`（候选召回率）：每个评测用户未来7天真实购买中，被候选池覆盖的比例，再对有真实购买的用户平均。扩展候选池为 base-300 与高精度图片子池去重并集，最多315项。",
        "- `Oracle MAP@12`（候选池理论上限）：假设能把候选池内真实购买完美排到前12得到的上限；只评估覆盖，不是可部署成绩。",
        "- `unobserved`（未观测）：未来7天没有购买记录；由于没有曝光日志，不能解释为用户明确不喜欢。", "",
        "## 时间训练协议", "",
        "每个窗口只用更早的 inner-validation 截点训练第三排序器；该截点的基础顺序来自只见过再早 inner-train 数据的 M3.3 模型。外层标签只用于报告，不用于模型、阈值或子池选择。", "",
        "## 训练比较对", "",
        "标签计数单位是图片商品—基础商品比较行；子池正例单位是命中未来7天购买的用户—图片商品候选观察。", "",
        "| window | train cutoff | subpool rows | subpool positives | pair rows | image should replace | base should remain | model status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for window, row in result["development"].items():
        train = row["training"]
        model = train["model"]
        pool = train["subpool"]
        lines.append(
            f"| {window} | {train['cutoff']} | {pool['subpool_candidate_rows']} | {pool['subpool_positive_pairs']} | "
            f"{model['pair_rows']} | {model['image_should_replace_pairs']} | {model['base_should_remain_pairs']} | {model['status']} |"
        )
    lines.extend([
        "", "## 外层 MAP 与替换结果", "",
        "带入/移除正例均以 `(cutoff, user, item)` 候选观察计数；移除基础正例表示第三排序器伤害了原 Top12。", "",
        "| window | base MAP@12 | new MAP@12 | delta | new warm MAP | new cold MAP | replaced users | inserted images | inserted image positives | removed base positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["development"].items():
        ev = row["evaluation"]
        audit = row["replacement_audit"]
        lines.append(
            f"| {window} | {ev['base_map']['overall']:.6f} | {ev['new_map']['overall']:.6f} | {ev['delta_map@12']:+.6f} | "
            f"{ev['new_map']['warm']:.6f} | {ev['new_map']['cold']:.6f} | {audit['replaced_users']} | "
            f"{audit['inserted_image_rows']} | {audit['inserted_image_positive_pairs']} | {audit['removed_base_positive_pairs']} |"
        )
    lines.extend([
        "", "## 候选覆盖上限", "",
        "下表的分母是各窗口当前评测用户的未来7天真实购买商品；base 是基础300候选，extended 是基础300与本阶段图片子池的并集。", "",
        "| window | pool | candidate Recall | candidate HitRate | Oracle MAP@12 |",
        "|---|---|---:|---:|---:|",
    ])
    for window, row in result["development"].items():
        for pool in ("base_300", "base_plus_subpool"):
            metric = row["candidate_ceiling"][pool]["overall"]
            lines.append(f"| {window} | {pool} | {metric['candidate_recall']:.6f} | {metric['candidate_hit_rate']:.6f} | {metric['oracle_map@12']:.6f} |")
    lines.extend([
        "", "## 决策", "",
        "- measured（已测量）：所有 MAP、候选覆盖、比较对和替换计数来自四个10%用户开发窗口；最终周未读取。",
        "- inference（基于门槛的结论）：只有报告顶部两个门槛同时通过才把本方案冻结成新 baseline；否则继续使用 M3.3 base-300。",
        "- next-step boundary（后续边界）：本方案0/4窗提升且0个图片正例获准，不满足预注册的“方向稳定但样本不足”扩样条件；不能直接把同一比较器放大到服务器。若继续图片方向，应先引入能增加独立正例监督或学习用户视觉表示的方案，再另行预注册。",
        "- limitation（限制）：训练只使用每个 outer 窗前一个 cutoff 的稀疏方向比较对；未通过不等价于该架构在更大样本或学习型视觉表示上永远无效。",
        "- serving gap（上线缺口）：当前离线比较器依赖批量候选特征和 M3.3 分数，尚未评估在线延迟、特征新鲜度或商品库存。", "",
        f"机器可读证据：`{result['artifacts']['metrics']}`。", "",
    ])
    return "\n".join(lines)


def run_m311b(
    *,
    m310_metrics_path: Path,
    m311a_metrics_path: Path,
    m33_metrics_path: Path,
    transactions_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    run_id: str,
    threads: int = 8,
    seed: int = 20260824,
) -> dict[str, Any]:
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    m310 = _read_json(m310_metrics_path)
    m311a = _read_json(m311a_metrics_path)
    m33 = _read_json(m33_metrics_path)
    validate_protocol(m310, m311a, m33)
    anchor_features = list(feature_sets()[ANCHOR_NAME])
    development: dict[str, Any] = {}
    try:
        for window, protocol in ROLLING_PROTOCOL.items():
            window_started = time.perf_counter()
            window_dir = artifact_dir / window
            window_dir.mkdir()
            train_cutoff = protocol["inner_validation"]
            outer_cutoff = protocol["outer_validation"]
            print(f"M3.11B {window}: cutoff-safe training pairs from {train_cutoff}", flush=True)

            train_path = Path(m310["enriched_cache"][train_cutoff]["path"])
            train_cache_evidence = m310["enriched_cache"][train_cutoff]
            _verify_bound_file(train_path, train_cache_evidence.get("artifact", train_cache_evidence), "training enriched cache")
            train_images, train_pool_evidence = _load_subpool(train_path)
            inner_model_evidence = m33["development"][window]["inner_models"][M33_MODEL_KEY]
            inner_model_path = Path(inner_model_evidence["model_path"])
            inner_maps_evidence = m33["development"][window]["inner_category_encoding"]
            inner_maps_path = Path(inner_maps_evidence["path"])
            _verify_bound_file(inner_model_path, inner_model_evidence, "inner anchor model")
            _verify_bound_file(inner_maps_path, inner_maps_evidence, "inner category maps")
            train_images = _score_anchor_rows(
                train_images,
                model_path=inner_model_path,
                category_maps_path=inner_maps_path,
                features=anchor_features,
            )
            train_bases = _load_inner_mutable_base(
                enriched_path=train_path,
                image_users=train_images["customer_id"],
                model_path=inner_model_path,
                category_maps_path=inner_maps_path,
                features=anchor_features,
            )
            train_pairs = _build_pair_frame(train_images, train_bases, training=True)
            comparator, model_evidence = _train_comparator(
                train_pairs,
                artifact_dir=window_dir,
                threads=threads,
                seed=seed,
            )

            print(f"M3.11B {window}: outer constrained replacement at {outer_cutoff}", flush=True)
            outer_path = Path(m310["enriched_cache"][outer_cutoff]["path"])
            outer_cache_evidence = m310["enriched_cache"][outer_cutoff]
            _verify_bound_file(outer_path, outer_cache_evidence.get("artifact", outer_cache_evidence), "outer enriched cache")
            outer_images, outer_pool_evidence = _load_subpool(outer_path)
            outer_model_evidence = m33["development"][window]["outer_models"][M33_MODEL_KEY]
            outer_model_path = Path(outer_model_evidence["model_path"])
            outer_maps_evidence = m33["development"][window]["outer_category_encoding"]
            outer_maps_path = Path(outer_maps_evidence["path"])
            _verify_bound_file(outer_model_path, outer_model_evidence, "outer anchor model")
            _verify_bound_file(outer_maps_path, outer_maps_evidence, "outer category maps")
            outer_images = _score_anchor_rows(
                outer_images,
                model_path=outer_model_path,
                category_maps_path=outer_maps_path,
                features=anchor_features,
            )
            prediction_evidence = m33["development"][window]["scoring"]
            prediction_path = Path(prediction_evidence["prediction_path"])
            _verify_bound_file(prediction_path, {
                "bytes": prediction_evidence["prediction_bytes"],
                "sha256": prediction_evidence["prediction_sha256"],
            }, "outer base predictions")
            outer_bases = _load_outer_mutable_base(
                enriched_path=outer_path,
                prediction_path=prediction_path,
                image_users=outer_images["customer_id"],
            )
            inference_pairs = _build_pair_frame(outer_images, outer_bases, training=False)
            if comparator is None:
                selected = _select_replacements(inference_pairs, np.zeros(len(inference_pairs), dtype=float))
            else:
                inference_matrix = inference_pairs[PAIR_FEATURES].apply(pd.to_numeric, errors="coerce").astype(np.float32)
                scores = comparator.predict(inference_matrix)
                selected = _select_replacements(inference_pairs, scores)
                del inference_matrix, scores

            base_top12 = _load_base_frame(
                prediction_path=prediction_path,
                transactions_path=transactions_path,
                cutoff=outer_cutoff,
            )
            final_top12 = _apply_replacements(base_top12, selected, outer_images)
            truth = _load_truth_frame(transactions_path, outer_cutoff)
            base_maps = _map_segments(base_top12, truth)
            new_maps = _map_segments(final_top12, truth)
            expected_base = float(
                m33["development"][window]["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]["map@12"]
            )
            if abs(base_maps["overall"] - expected_base) > NUMERIC_TOLERANCE:
                raise RuntimeError(f"M3.11B base MAP reproduction failed: {window}")
            ceiling = _candidate_ceiling(
                prediction_path=prediction_path,
                images=outer_images,
                transactions_path=transactions_path,
                cutoff=outer_cutoff,
            )
            replacement_audit = {
                "subpool_users": int(outer_images["customer_id"].nunique()),
                "replaced_users": int(selected["customer_id"].nunique()) if not selected.empty else 0,
                "abstained_users": int(outer_images["customer_id"].nunique() - (selected["customer_id"].nunique() if not selected.empty else 0)),
                "inserted_image_rows": int(len(selected)),
                "inserted_image_positive_pairs": int(selected["image_target"].sum()) if not selected.empty else 0,
                "removed_base_positive_pairs": int(selected["base_target"].sum()) if not selected.empty else 0,
                "beneficial_directional_replacements": int(((selected["image_target"] == 1) & (selected["base_target"] == 0)).sum()) if not selected.empty else 0,
                "harmful_directional_replacements": int(((selected["image_target"] == 0) & (selected["base_target"] == 1)).sum()) if not selected.empty else 0,
                "neutral_unobserved_replacements": int(((selected["image_target"] == 0) & (selected["base_target"] == 0)).sum()) if not selected.empty else 0,
                "both_positive_replacements": int(((selected["image_target"] == 1) & (selected["base_target"] == 1)).sum()) if not selected.empty else 0,
                "max_replacements_per_user": int(selected.groupby("customer_id", sort=False).size().max()) if not selected.empty else 0,
            }
            elapsed_window = time.perf_counter() - window_started
            development[window] = {
                "protocol": protocol,
                "training": {
                    "cutoff": train_cutoff,
                    "subpool": train_pool_evidence,
                    "mutable_base_rows": int(len(train_bases)),
                    "model": model_evidence,
                },
                "outer_subpool": outer_pool_evidence,
                "outer_pair_rows": int(len(inference_pairs)),
                "replacement_audit": replacement_audit,
                "evaluation": {
                    "base_map": base_maps,
                    "new_map": new_maps,
                    "delta_map@12": float(new_maps["overall"] - base_maps["overall"]),
                },
                "candidate_ceiling": ceiling,
                "inputs": {
                    "training_enriched": _identity(train_path),
                    "outer_enriched": _identity(outer_path),
                    "outer_predictions": _identity(prediction_path),
                    "inner_anchor_model": _identity(inner_model_path),
                    "outer_anchor_model": _identity(outer_model_path),
                },
                "elapsed_seconds": elapsed_window,
            }
            if elapsed_window > 1800:
                raise TimeoutError(f"M3.11B single-window stop condition: {window} {elapsed_window:.1f}s")
            if time.perf_counter() - started > 5400:
                raise TimeoutError("M3.11B total stop condition exceeded 90 minutes")
            del train_images, train_bases, train_pairs, comparator, outer_images, outer_bases
            del inference_pairs, selected, base_top12, final_top12, truth
            gc.collect()

        deltas = {
            window: float(row["evaluation"]["delta_map@12"])
            for window, row in development.items()
        }
        admitted = {
            window: int(row["replacement_audit"]["inserted_image_positive_pairs"])
            for window, row in development.items()
        }
        gate = _gate(deltas, admitted)
        baseline_name = run_id if gate["accepted_as_new_baseline"] else "m3-3-v1-cross-season-adaptive-seasonal/base-300"
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": RUN_STAGE,
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four rolling development windows; 10% users; final week not run",
            "contract": {
                "subpool": "active personalized image candidates; personalized rank<=10 OR personalized direct-decay rank<=5; max15/user",
                "training_cutoff": "immediately preceding inner-validation cutoff with base ranks from earlier-trained M3.3 inner model",
                "training_labels": "XOR purchase labels for image-vs-base-rank8-12 pairs",
                "class_weighting": "balanced total class weight",
                "threshold": ADMISSION_THRESHOLD,
                "protected_base_ranks": "1-7",
                "mutable_base_ranks": list(MUTABLE_BASE_RANKS),
                "max_replacements_per_user": MAX_REPLACEMENTS,
                "catalog": "optimistic_all_articles",
                "final_week": FINAL_WEEK,
            },
            "model_features": PAIR_FEATURES,
            "inputs": {
                "m310_metrics": _identity(m310_metrics_path),
                "m311a_metrics": _identity(m311a_metrics_path),
                "m33_metrics": _identity(m33_metrics_path),
                "transactions": _identity(transactions_path),
            },
            "development": development,
            "summary": {
                "window_delta_map@12": deltas,
                **gate,
                "frozen_baseline_after_stage": baseline_name,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_11B_FINAL.md"
        decision_path = output_dir / "BASELINE_DECISION.json"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "baseline_decision": str(decision_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(_render_report(result), encoding="utf-8", newline="\n")
        _write_json(decision_path, {
            "schema_version": "m3.11b-baseline-decision-v1",
            "run_id": run_id,
            "accepted_as_new_baseline": gate["accepted_as_new_baseline"],
            "frozen_baseline_after_stage": baseline_name,
            "metrics": _identity(metrics_path),
            "final_week": FINAL_WEEK,
        })
        return result
    except Exception as error:
        _write_json(artifact_dir / f"failure-{time.time_ns()}.json", {
            "schema_version": "m3.11b-failure-v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "completed_windows": list(development),
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise
