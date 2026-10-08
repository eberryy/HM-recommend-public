from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import CATEGORICAL_FEATURES, M2Config, _literal, _prepare_frame, _write_json, build_category_maps
from .m3 import ANCHOR_NAME
from .m25_retrieval import _candidate_metrics, _load_neighbors
from .m33 import ALL_CUTOFFS, ROLLING_PROTOCOL
from .m38 import IMAGE_SOURCE_FEATURES, _variant_path
from .m211 import train_lambdarank
from .m212 import train_inner_ranker, feature_sets


SCHEMA_VERSION = "m3.10-source-specific-image-admission-v1"
VARIANT = "base_historical_soft_500"
DIRECT_FEATURES = [
    "direct_visual_raw_max",
    "direct_visual_decay_max",
    "direct_visual_best_seed_age_days",
    "direct_visual_best_seed_rank",
    "direct_visual_seed_support",
    "direct_visual_top1_top2_margin",
]
ADMISSION_K = (0, 1, 3, 5)
HALF_LIFE_DAYS = 28.0
FINAL_WEEK_CUTOFF = "2020-09-16"
NUMERIC_TOLERANCE = 1e-12


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_identity(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    _write_json(path, value)
    return _identity(path)


def _direct_path(cache_dir: Path, cutoff: str) -> Path:
    return cache_dir / cutoff / "direct-visual.parquet"


def _direct_manifest(cache_dir: Path, cutoff: str) -> Path:
    return cache_dir / cutoff / "manifest.json"


def _enriched_path(cache_dir: Path, cutoff: str) -> Path:
    return cache_dir / cutoff / f"{VARIANT}.parquet"


def _date(value: str) -> date:
    return date.fromisoformat(value)


def validate_protocol() -> None:
    if VARIANT != "base_historical_soft_500":
        raise RuntimeError("M3.10 variant drifted")
    if set(ADMISSION_K) != {0, 1, 3, 5}:
        raise RuntimeError("M3.10 admission policy drifted")
    if any(cutoff >= FINAL_WEEK_CUTOFF for cutoff in ALL_CUTOFFS):
        raise RuntimeError("M3.10 must not read final week")
    if len(ROLLING_PROTOCOL) != 4:
        raise RuntimeError("M3.10 requires four rolling windows")


def _load_candidate_sets(variant_path: Path, context_db: Path) -> dict[str, set[int]]:
    con = duckdb.connect(str(context_db), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT v.customer_id,v.article_id,i.row_index
            FROM read_parquet(?) v JOIN image_items i USING(article_id)
            WHERE v.base_present=0 AND v.soft_present=1
            """,
            [str(variant_path)],
        ).fetchall()
    finally:
        con.close()
    result: dict[str, set[int]] = {}
    for customer_id, _article_id, row_index in rows:
        result.setdefault(str(customer_id), set()).add(int(row_index))
    return result


def _aggregate_direct_features(
    *,
    cutoff: str,
    variant_path: Path,
    context_db: Path,
    neighbor_metrics_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reaggregate the already-computed static neighbors against cutoff-safe seeds.

    The six columns are direct user-candidate image matching statistics. They are built
    only for soft-shallow image rows; base and historical-only rows receive zeros when
    joined later. The 100-neighbor prefix is exactly the frozen M3.7 soft-shallow source.
    """
    started = time.perf_counter()
    con = duckdb.connect(str(context_db), read_only=True)
    try:
        seeds = con.execute(
            "SELECT customer_id,row_index,seed_rank,latest_t_dat FROM seeds ORDER BY customer_id,seed_rank"
        ).fetchdf()
        image_items = con.execute("SELECT row_index,article_id FROM image_items").fetchdf()
    finally:
        con.close()
    candidate_sets = _load_candidate_sets(variant_path, context_db)
    _query_rows, indices, scores, _items = _load_neighbors(neighbor_metrics_path)
    if indices.shape != scores.shape or indices.shape[1] < 100:
        raise RuntimeError("M3.10 neighbor matrix does not contain frozen Top100 prefix")
    row_to_article = {
        int(row.row_index): str(row.article_id) for row in image_items.itertuples(index=False)
    }
    cutoff_day = _date(cutoff)
    records: list[dict[str, Any]] = []
    groups = seeds.groupby("customer_id", sort=True)
    for user_index, (customer_id, group) in enumerate(groups, start=1):
        wanted = candidate_sets.get(str(customer_id), set())
        if not wanted:
            continue
        stats: dict[int, list[float | int]] = {}
        for seed in group.itertuples(index=False):
            seed_row = int(seed.row_index)
            seed_rank = int(seed.seed_rank)
            latest_day = seed.latest_t_dat.date() if hasattr(seed.latest_t_dat, "date") else seed.latest_t_dat
            age_days = max(0, (cutoff_day - latest_day).days)
            decay = math.exp(-math.log(2.0) * age_days / HALF_LIFE_DAYS)
            for neighbor_row, cosine in zip(indices[seed_row, :100], scores[seed_row, :100]):
                row = int(neighbor_row)
                if row not in wanted:
                    continue
                raw = float(cosine)
                decayed = raw * decay
                entry = stats.get(row)
                if entry is None:
                    # raw max, decayed max, age at decayed max, seed rank at raw max,
                    # support count, second-best raw (used for a margin feature).
                    stats[row] = [raw, decayed, age_days, seed_rank, 1, -np.inf]
                    continue
                entry[4] = int(entry[4]) + 1
                if raw > float(entry[0]):
                    entry[5] = float(entry[0])
                    entry[0] = raw
                    entry[3] = seed_rank
                elif raw > float(entry[5]):
                    entry[5] = raw
                if decayed > float(entry[1]):
                    entry[1] = decayed
                    entry[2] = age_days
        for row, values in stats.items():
            second = float(values[5]) if np.isfinite(float(values[5])) else 0.0
            records.append(
                {
                    "customer_id": str(customer_id),
                    "article_id": row_to_article[row],
                    DIRECT_FEATURES[0]: float(values[0]),
                    DIRECT_FEATURES[1]: float(values[1]),
                    DIRECT_FEATURES[2]: int(values[2]),
                    DIRECT_FEATURES[3]: int(values[3]),
                    DIRECT_FEATURES[4]: int(values[4]),
                    DIRECT_FEATURES[5]: max(0.0, float(values[0]) - second),
                }
            )
        if user_index % 1000 == 0 or user_index == len(groups):
            print(f"M3.10 direct visual features {user_index}/{len(groups)} users", flush=True)
    frame = pd.DataFrame.from_records(records, columns=["customer_id", "article_id", *DIRECT_FEATURES])
    if frame.empty:
        raise RuntimeError(f"M3.10 produced no direct features for {cutoff}")
    if frame.duplicated(["customer_id", "article_id"]).any():
        raise RuntimeError(f"M3.10 direct feature identity audit failed: {cutoff}")
    return frame, {
        "cutoff": cutoff,
        "rows": int(len(frame)),
        "users": int(frame.customer_id.nunique()),
        "neighbor_prefix": 100,
        "half_life_days": HALF_LIFE_DAYS,
        "candidate_source": "base_present=0 AND soft_present=1",
        "elapsed_seconds": time.perf_counter() - started,
    }


def build_direct_cache(
    *,
    variant_cache_dir: Path,
    soft_cache_dir: Path,
    neighbor_metrics_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    neighbor_identity = _identity(neighbor_metrics_path)
    for cutoff in ALL_CUTOFFS:
        output = _direct_path(cache_dir, cutoff)
        manifest_path = _direct_manifest(cache_dir, cutoff)
        variant_path = _variant_path(variant_cache_dir, cutoff, VARIANT)
        context_db = soft_cache_dir / cutoff / "context.duckdb"
        if output.is_file() and manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != "m3.10-direct-visual-cache-v1":
                raise ValueError(f"M3.10 direct cache schema drift: {cutoff}")
            if _identity(output) != manifest["artifact"]:
                raise ValueError(f"M3.10 direct cache artifact drift: {cutoff}")
            results[cutoff] = manifest
            continue
        if output.parent.exists():
            raise FileExistsError(f"incomplete M3.10 direct cache: {output.parent}")
        output.parent.mkdir(parents=True)
        frame, evidence = _aggregate_direct_features(
            cutoff=cutoff,
            variant_path=variant_path,
            context_db=context_db,
            neighbor_metrics_path=neighbor_metrics_path,
        )
        con = duckdb.connect()
        try:
            con.register("direct_frame", frame)
            con.execute(
                f"COPY (SELECT * FROM direct_frame ORDER BY customer_id,article_id) TO {_literal(output)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            audit = con.execute(
                f"SELECT count(*),count(DISTINCT (customer_id,article_id)) FROM read_parquet({_literal(output)})"
            ).fetchone()
        finally:
            try:
                con.unregister("direct_frame")
            except Exception:
                pass
            con.close()
        if int(audit[0]) != int(audit[1]):
            raise RuntimeError(f"M3.10 direct cache duplicate audit failed: {cutoff}")
        manifest = {
            "schema_version": "m3.10-direct-visual-cache-v1",
            "status": "completed",
            "cutoff": cutoff,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "variant": _identity(variant_path),
                "context": _identity(context_db),
                "neighbor_metrics": neighbor_identity,
            },
            "evidence": evidence,
            "artifact": _identity(output),
        }
        _write_json(manifest_path, manifest)
        results[cutoff] = manifest
    return results


def build_enriched_cache(
    *, variant_cache_dir: Path, direct_cache_dir: Path, cache_dir: Path
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cutoff in ALL_CUTOFFS:
        source = _variant_path(variant_cache_dir, cutoff, VARIANT)
        direct = _direct_path(direct_cache_dir, cutoff)
        output = _enriched_path(cache_dir, cutoff)
        if output.is_file():
            results[cutoff] = _identity(output)
            continue
        if output.parent.exists():
            raise FileExistsError(f"incomplete M3.10 enriched cache: {output.parent}")
        output.parent.mkdir(parents=True)
        con = duckdb.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT v.*,
                    coalesce(d.{DIRECT_FEATURES[0]},0.0)::DOUBLE AS {DIRECT_FEATURES[0]},
                    coalesce(d.{DIRECT_FEATURES[1]},0.0)::DOUBLE AS {DIRECT_FEATURES[1]},
                    coalesce(d.{DIRECT_FEATURES[2]},0)::BIGINT AS {DIRECT_FEATURES[2]},
                    coalesce(d.{DIRECT_FEATURES[3]},0)::BIGINT AS {DIRECT_FEATURES[3]},
                    coalesce(d.{DIRECT_FEATURES[4]},0)::BIGINT AS {DIRECT_FEATURES[4]},
                    coalesce(d.{DIRECT_FEATURES[5]},0.0)::DOUBLE AS {DIRECT_FEATURES[5]}
                  FROM read_parquet({_literal(source)}) v
                  LEFT JOIN read_parquet({_literal(direct)}) d USING(customer_id,article_id)
                  ORDER BY v.customer_id,v.candidate_rank,v.article_id
                ) TO {_literal(output)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
                """
            )
            audit = con.execute(
                f"SELECT count(*),count(DISTINCT (customer_id,article_id)),sum(target) FROM read_parquet({_literal(output)})"
            ).fetchone()
        finally:
            con.close()
        if int(audit[0]) != int(audit[1]):
            raise RuntimeError(f"M3.10 enriched cache identity audit failed: {cutoff}")
        results[cutoff] = {"path": str(output.resolve()), "rows": int(audit[0]), "positives": int(audit[2]), "artifact": _identity(output)}
    return results


def _image_relation(path: Path) -> str:
    return f"read_parquet({_literal(path)})"


def _sample_image_relation(path: Path, seed: int) -> str:
    source = _image_relation(path)
    return f"""
    WITH source AS (
      SELECT * FROM {source}
      WHERE base_present=0 AND soft_present=1 AND user_history_events_12w>0
    ), positive_groups AS (
      SELECT customer_id,sum(target)::BIGINT AS positives FROM source
      GROUP BY customer_id HAVING sum(target)>0
    ), positives AS (
      SELECT s.* FROM source s JOIN positive_groups g USING(customer_id) WHERE s.target=1
    ), negatives AS (
      SELECT s.*,g.positives,
        row_number() OVER(PARTITION BY s.customer_id ORDER BY hash(s.customer_id,s.article_id,{seed}),s.candidate_rank,s.article_id) AS neg_rank
      FROM source s JOIN positive_groups g USING(customer_id) WHERE s.target=0
    ), selected_negatives AS (
      SELECT * FROM negatives WHERE neg_rank<=30*positives
    )
    SELECT * FROM positives UNION ALL SELECT * EXCLUDE(positives,neg_rank) FROM selected_negatives
    """


def load_image_sample(paths: list[Path], features: list[str], *, seed: int) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    source_rows = 0
    for path in paths:
        relation = _sample_image_relation(path, seed)
        con = duckdb.connect()
        try:
            frame = con.execute(
                f"SELECT customer_id,article_id,{','.join(features)},target FROM ({relation}) ORDER BY customer_id,candidate_rank,article_id"
            ).fetchdf()
            groups = con.execute(f"SELECT count(*),sum(target) FROM ({relation}) GROUP BY customer_id ORDER BY customer_id").fetchall()
            source_rows += int(con.execute(f"SELECT count(*) FROM {_image_relation(path)} WHERE base_present=0 AND soft_present=1 AND user_history_events_12w>0").fetchone()[0])
        finally:
            con.close()
        if not groups or any(int(row[1]) <= 0 for row in groups):
            continue
        frames.append(frame)
        sizes.extend(int(row[0]) for row in groups)
    if not frames:
        raise RuntimeError("M3.10 image sample has no positive groups")
    result = pd.concat(frames, ignore_index=True)
    positives = int(result.target.sum())
    negatives = len(result) - positives
    if negatives > 30 * positives:
        raise RuntimeError("M3.10 image sampling exceeded 30 negatives per positive")
    return result, sizes, {
        "mode": "soft-image-only-hash-30x",
        "seed": seed,
        "source_rows": source_rows,
        "sampled_rows": int(len(result)),
        "groups": len(sizes),
        "positives": positives,
        "negatives": negatives,
        "unobserved_per_positive": negatives / max(positives, 1),
    }


def load_image_validation(*, path: Path, transactions_path: Path, cutoff: str, features: list[str]) -> tuple[pd.DataFrame, list[int], np.ndarray, dict[str, Any]]:
    source = _image_relation(path)
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"""
            WITH source AS (
              SELECT * FROM {source} WHERE base_present=0 AND soft_present=1
            ), eligible AS (
              SELECT customer_id FROM source GROUP BY customer_id
              HAVING sum(target)>0 AND max(user_history_events_12w)>0
            ), truth AS (
              SELECT customer_id,count(DISTINCT article_id)::BIGINT AS truth_count
              FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
              GROUP BY customer_id
            )
            SELECT s.customer_id,s.article_id,{','.join('s.'+feature for feature in features)},s.target,s.user_history_events_12w,t.truth_count
            FROM source s JOIN eligible e USING(customer_id) JOIN truth t USING(customer_id)
            ORDER BY s.customer_id,s.candidate_rank,s.article_id
            """
        ).fetchdf()
    finally:
        con.close()
    if frame.empty:
        raise RuntimeError(f"M3.10 image validation empty: {cutoff}")
    groups = frame.groupby("customer_id", sort=False).agg(rows=("target", "size"), positives=("target", "sum"), truth=("truth_count", "first"))
    sizes = groups.rows.astype(int).tolist()
    if (groups.positives<=0).any() or (groups.truth<=0).any():
        raise RuntimeError(f"M3.10 image validation group audit failed: {cutoff}")
    return frame, sizes, groups.truth.to_numpy(dtype=np.int64), {
        "cutoff": cutoff,
        "rows": int(len(frame)),
        "groups": int(len(sizes)),
        "min_group_rows": int(min(sizes)),
        "max_group_rows": int(max(sizes)),
        "role": "active soft-image candidate-covered inner validation",
    }


def _feature_list() -> list[str]:
    anchor = list(feature_sets()[ANCHOR_NAME])
    source = [name for name in IMAGE_SOURCE_FEATURES if name not in {"base_present", "soft_present"}]
    return list(dict.fromkeys([*anchor, *source, *DIRECT_FEATURES]))


def _score_image_rows(model: Any, path: Path, features: list[str], category_maps: dict[str, dict[int, int]]) -> pd.DataFrame:
    columns = ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w", "base_present", "historical_present", "soft_present", *features]
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"SELECT {','.join(dict.fromkeys(columns))} FROM read_parquet({_literal(path)}) "
            "WHERE base_present=0 AND soft_present=1 ORDER BY customer_id,candidate_rank,article_id"
        ).fetchdf()
    finally:
        con.close()
    if frame.empty:
        raise RuntimeError(f"M3.10 scoring has no soft image rows: {path}")
    # Build the matrix in one shot; the shared helper intentionally favors clarity
    # and emits pandas fragmentation warnings for this wider image-only frame.
    prepared_columns: dict[str, pd.Series] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if feature in CATEGORICAL_FEATURES:
            mapping = category_maps.get(feature, {})
            prepared_columns[feature] = values.map(mapping).fillna(0).astype(np.int32)
        else:
            prepared_columns[feature] = values.astype(np.float32)
    prepared = pd.DataFrame(prepared_columns, index=frame.index)
    frame["image_expert_score"] = model.predict(prepared)
    return frame[["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w", "base_present", "historical_present", "soft_present", "image_expert_score"]]


def _load_maps(path: Path) -> dict[str, dict[int, int]]:
    raw = _read_json(path)
    return {feature: {int(value): int(encoded) for value, encoded in mapping.items()} for feature, mapping in raw.items()}


def _score_anchor_base(*, model_path: Path, category_maps_path: Path, path: Path, features: list[str]) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"SELECT customer_id,article_id,candidate_rank,{','.join(features)} FROM read_parquet({_literal(path)}) WHERE base_present=1 ORDER BY customer_id,candidate_rank,article_id"
        ).fetchdf()
    finally:
        con.close()
    maps = _load_maps(category_maps_path)
    prepared_columns: dict[str, pd.Series] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if feature in CATEGORICAL_FEATURES:
            prepared_columns[feature] = values.map(maps.get(feature, {})).fillna(0).astype(np.int32)
        else:
            prepared_columns[feature] = values.astype(np.float32)
    prepared = pd.DataFrame(prepared_columns, index=frame.index)
    model = lgb.Booster(model_file=str(model_path))
    frame["anchor_score"] = model.predict(prepared)
    return frame[["customer_id", "article_id", "candidate_rank", "anchor_score"]]


def _load_truth_frame(transactions_path: Path, cutoff: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            WITH q AS (
              SELECT DISTINCT customer_id,article_id
              FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
            ), w AS (
              SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat<DATE '{cutoff}'
            )
            SELECT q.customer_id,q.article_id,
              CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
            FROM q LEFT JOIN w USING(article_id)
            """
        ).fetchdf()
    finally:
        con.close()


def _metric_from_frames(candidates: pd.DataFrame, truth: pd.DataFrame, *, segment: str, budget: int) -> dict[str, float | int]:
    # Scope the denominator to customers present in this candidate set.  The raw
    # transaction truth contains customers outside the evaluation population.
    candidate_users = candidates["customer_id"].drop_duplicates()
    scoped_truth = truth.loc[truth["customer_id"].isin(candidate_users)]
    segment_truth = scoped_truth if segment == "overall" else scoped_truth.loc[scoped_truth["item_temperature"] == segment]
    truth_counts = segment_truth.groupby("customer_id", sort=False).size().rename("truth_count")
    if truth_counts.empty:
        return {"users": 0, "truth_pairs": 0, f"candidate_recall@{budget}": 0.0, "candidate_hit_rate@500": 0.0, "oracle_map@12": 0.0, "map@12": 0.0}
    hit = candidates.loc[candidates["target"].astype(bool), ["customer_id", "article_id", "candidate_rank"]].merge(
        scoped_truth[["customer_id", "article_id", "item_temperature"]], on=["customer_id", "article_id"], how="inner"
    )
    if segment != "overall":
        hit = hit.loc[hit["item_temperature"] == segment]
    hit_counts = hit.groupby("customer_id", sort=False).size().rename("hits")
    joined = truth_counts.to_frame().join(hit_counts, how="left").fillna({"hits": 0})
    joined["hits"] = joined["hits"].astype(np.int64)
    denom = joined["truth_count"].clip(upper=12)
    top = hit.loc[hit["candidate_rank"] <= 12].sort_values(["customer_id", "candidate_rank", "article_id"])
    if top.empty:
        map_values = pd.Series(0.0, index=joined.index)
    else:
        top = top.copy()
        top["cum_hits"] = top.groupby("customer_id", sort=False).cumcount() + 1
        top["precision"] = top["cum_hits"] / top["candidate_rank"].astype(float)
        sums = top.groupby("customer_id", sort=False)["precision"].sum()
        map_values = sums.reindex(joined.index, fill_value=0.0)
    return {
        "users": int(len(joined)),
        "truth_pairs": int(joined["truth_count"].sum()),
        f"candidate_recall@{budget}": float((joined["hits"] / joined["truth_count"]).mean()),
        "candidate_hit_rate@500": float((joined["hits"] > 0).mean()),
        "oracle_map@12": float((joined["hits"].clip(upper=12) / denom).mean()),
        "map@12": float((map_values / denom).mean()),
    }


def _load_policy_candidates(candidates_path: Path, image_scores: pd.DataFrame, base_scores: pd.DataFrame | None = None) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        candidates = con.execute(
            f"SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w,base_present,historical_present,soft_present "
            f"FROM read_parquet({_literal(candidates_path)}) ORDER BY customer_id,candidate_rank,article_id"
        ).fetchdf()
    finally:
        con.close()
    score = image_scores[["customer_id", "article_id", "image_expert_score"]].copy()
    score = score.sort_values(["customer_id", "image_expert_score", "article_id"], ascending=[True, False, True])
    score["image_rank"] = score.groupby("customer_id", sort=False).cumcount() + 1
    candidates = candidates.merge(score[["customer_id", "article_id", "image_expert_score", "image_rank"]], on=["customer_id", "article_id"], how="left", sort=False)
    if base_scores is not None:
        base = base_scores[["customer_id", "article_id", "candidate_rank", "anchor_score"]].copy()
        base = base.sort_values(["customer_id", "anchor_score", "candidate_rank", "article_id"], ascending=[True, False, True, True], kind="mergesort")
        base["anchor_rank"] = base.groupby("customer_id", sort=False).cumcount() + 1
        candidates = candidates.merge(base[["customer_id", "article_id", "anchor_rank"]], on=["customer_id", "article_id"], how="left", sort=False)
    else:
        candidates["anchor_rank"] = np.nan
    candidates["base_order"] = candidates["anchor_rank"].where(candidates["base_present"] == 1, candidates["candidate_rank"])
    candidates = candidates.sort_values(["customer_id", "base_order", "article_id"], kind="mergesort").reset_index(drop=True)
    if candidates.duplicated(["customer_id", "article_id"]).any():
        raise RuntimeError("M3.10 policy candidate identity audit failed")
    return candidates


def _safe_image_ranks(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return integer image ranks without casting missing values to int64 minimum."""
    numeric = pd.to_numeric(values, errors="coerce")
    available = numeric.notna().to_numpy(dtype=bool)
    safe = np.zeros(len(numeric), dtype=np.int64)
    if available.any():
        observed = numeric.loc[available].to_numpy(dtype=np.float64)
        if not np.isfinite(observed).all() or (observed < 1).any() or not np.equal(observed, np.floor(observed)).all():
            raise RuntimeError("M3.10 image ranks must be positive finite integers")
        safe[available] = observed.astype(np.int64)
    return safe, available


def _validated_admission_mask(
    *,
    customer_ids: pd.Series,
    active: np.ndarray,
    soft: np.ndarray,
    image_rank: np.ndarray,
    image_rank_available: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict[str, int]]:
    admitted = active & soft & image_rank_available & (image_rank <= int(k))
    admitted_counts = pd.Series(admitted.astype(np.int8)).groupby(customer_ids, sort=False).sum()
    max_per_user = int(admitted_counts.max()) if not admitted_counts.empty else 0
    if k == 0 and admitted.any():
        raise RuntimeError("M3.10 K=0 invariant failed: image rows were admitted")
    if max_per_user > int(k):
        raise RuntimeError(f"M3.10 admission cap failed: max_per_user={max_per_user}, K={k}")
    return admitted, {
        "admitted_rows": int(admitted.sum()),
        "admitted_users": int((admitted_counts > 0).sum()),
        "max_admitted_per_user": max_per_user,
    }


def _evaluate_policies_fast(
    *, candidates_path: Path, image_scores: pd.DataFrame, transactions_path: Path, cutoff: str, ks: tuple[int, ...] = ADMISSION_K, budget: int = 500, base_scores: pd.DataFrame | None = None
) -> dict[int, dict[str, Any]]:
    candidates = _load_policy_candidates(candidates_path, image_scores, base_scores)
    truth = _load_truth_frame(transactions_path, cutoff)
    results: dict[int, dict[str, Any]] = {}
    active = candidates["user_history_events_12w"].to_numpy(dtype=np.int64) > 0
    soft = candidates["soft_present"].to_numpy(dtype=np.int8) == 1
    image_rank, image_rank_available = _safe_image_ranks(candidates["image_rank"])
    original_rank = candidates["candidate_rank"].to_numpy(dtype=np.int64)
    base_order = candidates["base_order"].to_numpy(dtype=np.int64)
    for k in ks:
        if k not in ADMISSION_K:
            raise ValueError(f"unsupported M3.10 admission K: {k}")
        admitted, admission_audit = _validated_admission_mask(
            customer_ids=candidates["customer_id"],
            active=active,
            soft=soft,
            image_rank=image_rank,
            image_rank_available=image_rank_available,
            k=k,
        )
        rest_order = (~admitted).astype(np.int64)
        rest_cum = pd.Series(rest_order, index=candidates.index).groupby(candidates["customer_id"], sort=False).cumsum().to_numpy(dtype=np.int64)
        final_rank = np.where(admitted, image_rank, np.where(active, int(k) + rest_cum, original_rank))
        frame = candidates[["customer_id", "article_id", "target", "user_history_events_12w", "base_present", "soft_present"]].copy()
        frame["candidate_rank"] = final_rank
        segments = {
            name: _metric_from_frames(frame, truth, segment=name, budget=budget)
            for name in ("overall", "warm", "cold")
        }
        image_truth = frame.loc[(frame["target"].astype(bool)) & (frame["base_present"] == 0) & (frame["soft_present"] == 1)].copy()
        funnel = {
            "image_truth_observations": int(len(image_truth)),
            "active_image_truth_observations": int((image_truth["user_history_events_12w"] > 0).sum()),
            "top12": int((image_truth["candidate_rank"] <= 12).sum()),
            "top20": int((image_truth["candidate_rank"] <= 20).sum()),
            "top100": int((image_truth["candidate_rank"] <= 100).sum()),
            "active_top12": int(((image_truth["user_history_events_12w"] > 0) & (image_truth["candidate_rank"] <= 12)).sum()),
            "active_top100": int(((image_truth["user_history_events_12w"] > 0) & (image_truth["candidate_rank"] <= 100)).sum()),
            "median_model_rank": float(image_truth["candidate_rank"].median()) if not image_truth.empty else None,
            "median_initial_rank": float(candidates.loc[image_truth.index, "candidate_rank"].median()) if not image_truth.empty else None,
        }
        results[int(k)] = {
            "k": int(k),
            "segments": segments,
            "image_truth_funnel": funnel,
            "active_inserted_rows": int(admission_audit["admitted_rows"]),
            "admission_audit": admission_audit,
        }
    del candidates, truth
    gc.collect()
    return results


def _truth_table_sql(transactions_path: Path, cutoff: str) -> str:
    return f"""
      SELECT q.customer_id,q.article_id,
        CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
      FROM (SELECT DISTINCT customer_id,article_id FROM read_parquet({_literal(transactions_path)})
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY) q
      LEFT JOIN (SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
                 WHERE t_dat<DATE '{cutoff}') w USING(article_id)
    """


def _evaluate_policy(
    *,
    candidates_path: Path,
    image_scores: pd.DataFrame,
    transactions_path: Path,
    cutoff: str,
    k: int,
    budget: int = 500,
) -> dict[str, Any]:
    if k not in ADMISSION_K:
        raise ValueError(f"unsupported M3.10 admission K: {k}")
    con = duckdb.connect()
    try:
        con.register("m310_scores", image_scores[["customer_id", "article_id", "image_expert_score"]])
        con.execute(
            f"""
            CREATE TABLE candidates AS SELECT * FROM read_parquet({_literal(candidates_path)});
            CREATE TABLE eval_truth AS {_truth_table_sql(transactions_path, cutoff)};
            CREATE TABLE ranked AS
            WITH image_order AS (
              SELECT customer_id,article_id,image_expert_score,
                row_number() OVER(PARTITION BY customer_id ORDER BY image_expert_score DESC NULLS LAST,candidate_rank,article_id) AS image_rank
              FROM candidates c JOIN m310_scores s USING(customer_id,article_id)
              WHERE c.base_present=0 AND c.soft_present=1
            ), tagged AS (
              SELECT c.*,
                CASE
                  WHEN c.user_history_events_12w=0 THEN 0
                  WHEN io.image_rank<={int(k)} THEN 1
                  WHEN c.base_present=1 THEN 2
                  ELSE 3
                END AS bucket,
                CASE
                  WHEN c.user_history_events_12w=0 THEN c.candidate_rank
                  WHEN io.image_rank<={int(k)} THEN io.image_rank
                  ELSE c.candidate_rank
                END AS tie_rank
              FROM candidates c LEFT JOIN image_order io USING(customer_id,article_id)
            )
            SELECT customer_id,article_id,
              row_number() OVER(PARTITION BY customer_id ORDER BY bucket,tie_rank,article_id) AS candidate_rank
            FROM tagged;
            """
        )
        segments = {
            name: _candidate_metrics(con, "ranked", budget, predicate)
            for name, predicate in (("overall", "TRUE"), ("warm", "item_temperature='warm'"), ("cold", "item_temperature='cold'"))
        }
        con.execute(
            """
            CREATE TABLE image_truth AS
            SELECT c.customer_id,c.article_id,r.candidate_rank AS model_rank,c.candidate_rank AS initial_rank,
              c.user_history_events_12w
            FROM candidates c JOIN ranked r USING(customer_id,article_id)
            JOIN eval_truth t USING(customer_id,article_id)
            WHERE c.base_present=0 AND c.soft_present=1;
            """
        )
        funnel = con.execute(
            """
            SELECT count(*) AS image_truth,
              count(*) FILTER(WHERE user_history_events_12w>0) AS active_image_truth,
              count(*) FILTER(WHERE model_rank<=12) AS top12,
              count(*) FILTER(WHERE model_rank<=20) AS top20,
              count(*) FILTER(WHERE model_rank<=100) AS top100,
              count(*) FILTER(WHERE user_history_events_12w>0 AND model_rank<=12) AS active_top12,
              count(*) FILTER(WHERE user_history_events_12w>0 AND model_rank<=100) AS active_top100,
              median(model_rank) AS median_rank,
              median(initial_rank) AS median_initial_rank
            FROM image_truth
            """
        ).fetchone()
        inserted = con.execute(
            f"SELECT count(*) FROM ranked r JOIN candidates c USING(customer_id,article_id) WHERE c.user_history_events_12w>0 AND c.base_present=0 AND r.candidate_rank<={int(k)}"
        ).fetchone()[0]
        return {
            "k": int(k),
            "segments": segments,
            "image_truth_funnel": {
                "image_truth_observations": int(funnel[0]),
                "active_image_truth_observations": int(funnel[1]),
                "top12": int(funnel[2]), "top20": int(funnel[3]), "top100": int(funnel[4]),
                "active_top12": int(funnel[5]), "active_top100": int(funnel[6]),
                "median_model_rank": float(funnel[7]) if funnel[7] is not None else None,
                "median_initial_rank": float(funnel[8]) if funnel[8] is not None else None,
            },
            "active_inserted_rows": int(inserted),
        }
    finally:
        try:
            con.unregister("m310_scores")
        except Exception:
            pass
        con.close()


def _select_k(inner_results: dict[int, dict[str, Any]]) -> tuple[int, dict[int, float]]:
    maps = {int(k): float(v["segments"]["overall"]["map@12"]) for k, v in inner_results.items()}
    selected = max(maps, key=lambda k: (maps[k], -int(k)))
    return int(selected), maps


def _passes_map_gate(deltas: dict[str, float]) -> bool:
    if not deltas:
        return False
    mean_delta = float(np.mean(list(deltas.values())))
    return bool(
        mean_delta > NUMERIC_TOLERANCE
        and all(float(delta) >= -NUMERIC_TOLERANCE for delta in deltas.values())
    )


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# M3.10：来源专属图片准入与直接用户—候选视觉匹配", "",
        "## 结论", "",
        f"- 运行状态：{result['status']}；最终验证周：{result['contract']['final_week']}。",
        f"- 10%样本机械门槛：active 图片 truth Top100={result['summary']['active_image_truth_top100_total']}，Top12={result['summary']['active_image_truth_top12_total']}。",
        f"- MAP 门槛（均值提升超过 {NUMERIC_TOLERANCE:.0e} 且四窗不退化）：{result['summary'].get('map_gate', result['summary']['accepted'])}。",
        f"- 图片机制门槛（至少有图片正例进入 Top100 和 Top12）：{result['summary']['mechanism_gate']}。",
        f"- 新 baseline 晋级：{result['summary']['accepted']}；若为 false，继续冻结 M3.3 base-300。", "",
        "## 术语与协议", "",
        "- `direct visual features`（直接视觉特征）：由 cutoff 前最近5个有图购买 seed 与静态 FashionCLIP Top100 近邻重新聚合的六个用户—候选匹配统计，不是把512维 embedding直接喂给 LightGBM。",
        "- `image expert`（图片专家）：只在 soft-shallow 图片候选内部训练的 LambdaRank，不与 base 候选共享来源竞争分数。",
        "- `admission K`（准入数量）：active 用户最多把 K 个图片候选插入冻结 base 顺序之前；K 只在预注册 `{0,1,3,5}` 中由 inner MAP 选择。",
        "- `image truth observation`（图片正例观察）：某个 cutoff 的 soft 图片候选行同时命中未来7天购买标签；跨 cutoff 可重复计数。", "",
        "- `soft-shallow`（浅层季节软控制图片召回，本项目自定义名）：由 FashionCLIP 视觉 Top100 与季节属性软打分合并产生的图片候选；本阶段只对其中相对 base 新增的候选训练图片专家。",
        "- `active user`（活跃用户）：在当前 cutoff 前12周至少有一次购买事件的用户；只有这类用户允许图片准入。",
        "- `inner/outer`（内层选择/外层评测）：inner 只用更早时间窗选择训练轮数和 K，outer 再重训并在后续开发窗口评测，防止用同一窗口既选方案又报结果。",
        "- `candidate Recall@500`（候选召回率）：对每个评测用户计算进入最多500个候选的未来7天真实购买商品比例，再对用户取平均。", "",
        "## Inner K 选择", "",
        "| window | selected K | " + " | ".join(f"K={k}" for k in ADMISSION_K) + " |",
        "|---|---:|" + "---:|" * len(ADMISSION_K),
    ]
    for window, row in result["development"].items():
        inner_maps = row["inner"]["k_maps"]
        lines.append(f"| {window} | {row['inner']['selected_k']} | " + " | ".join(f"{inner_maps[str(k) if str(k) in inner_maps else k]:.6f}" for k in ADMISSION_K) + " |")
    lines.extend([
        "", "## Inner 准入审计", "",
        "下表的准入行数单位是用户—商品候选行；`max/user` 是任一用户被插入的最大图片候选数。",
        "K=0 必须为0，且每个 K 的 `max/user` 不得超过 K。", "",
        "| window | K=0 rows | K=1 rows (max/user) | K=3 rows (max/user) | K=5 rows (max/user) |",
        "|---|---:|---:|---:|---:|",
    ])
    for window, row in result["development"].items():
        policy = row["inner"]["policy"]
        values = []
        for k in ADMISSION_K:
            audit = policy[str(k) if str(k) in policy else k]["admission_audit"]
            values.append(f"{audit['admitted_rows']} ({audit['max_admitted_per_user']})" if k else str(audit["admitted_rows"]))
        lines.append(f"| {window} | " + " | ".join(values) + " |")
    lines.extend([
        "", "## Inner 图片正例兑现", "",
        "下表统计 active 用户的图片正例候选进入最终 Top12 的候选观察数；同一用户—商品跨窗口可重复计数。", "",
        "| window | K=0 | K=1 | K=3 | K=5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for window, row in result["development"].items():
        policy = row["inner"]["policy"]
        values = [str(policy[str(k) if str(k) in policy else k]["image_truth_funnel"]["active_top12"]) for k in ADMISSION_K]
        lines.append(f"| {window} | " + " | ".join(values) + " |")
    lines.extend(["", "## Outer 结果", "", "| window | selected K | overall MAP@12 | warm MAP@12 | cold MAP@12 | candidate Recall@500 | image Top12 | image Top100 | active image Top12 | active image Top100 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for window, row in result["development"].items():
        ev = row["outer"]["evaluation"]
        f = ev["image_truth_funnel"]
        lines.append(f"| {window} | {row['inner']['selected_k']} | {ev['segments']['overall']['map@12']:.6f} | {ev['segments']['warm']['map@12']:.6f} | {ev['segments']['cold']['map@12']:.6f} | {ev['segments']['overall']['candidate_recall@500']:.6f} | {f['top12']} | {f['top100']} | {f['active_top12']} | {f['active_top100']} |")
    lines.extend(["", "## 解释与边界", "", "- 图片专家只影响 soft-shallow 图片候选；historical-only 与未准入图片保持 M3.8 原始追加顺序。inactive 用户保持候选原顺序。", "- 直接特征由已验收的近邻矩阵和 cutoff 前 seeds 聚合，未重跑 FashionCLIP；没有使用验证周 truth 选择特征或 K。", "- 10%训练规模仍然是机制检查，不能据此宣称图片排序已达到上限；若门槛失败，按预注册停止并转特征可分性审计，不扩大训练规模。", "", f"机器可读证据：`{result['artifacts']['metrics']}`。", ""])
    return "\n".join(lines)


def run_m310(
    *,
    variant_cache_dir: Path,
    soft_cache_dir: Path,
    neighbor_metrics_path: Path,
    transactions_path: Path,
    m33_metrics_path: Path,
    direct_cache_dir: Path,
    enriched_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    validate_protocol()
    config.validate()
    if config.evaluation_role != "development":
        raise ValueError("M3.10 requires development evaluation_role")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    features = _feature_list()
    anchor_features = list(feature_sets()[ANCHOR_NAME])
    try:
        m33 = _read_json(m33_metrics_path)
        if m33.get("schema_version") != "m3.3-cross-season-adaptive-seasonal-v1":
            raise ValueError("M3.10 requires the measured M3.3 metric contract")
        if m33.get("contract", {}).get("final_week") not in {"not_run", None}:
            raise ValueError("M3.10 refuses metrics that include final week")
        print("M3.10: building direct user-candidate visual feature cache", flush=True)
        direct = build_direct_cache(
            variant_cache_dir=variant_cache_dir,
            soft_cache_dir=soft_cache_dir,
            neighbor_metrics_path=neighbor_metrics_path,
            cache_dir=direct_cache_dir,
        )
        print("M3.10: enriching the frozen M3.8 candidate variant", flush=True)
        enriched = build_enriched_cache(
            variant_cache_dir=variant_cache_dir,
            direct_cache_dir=direct_cache_dir,
            cache_dir=enriched_cache_dir,
        )
        development: dict[str, Any] = {}
        for window, protocol in ROLLING_PROTOCOL.items():
            print(f"M3.10 {window}: image expert inner training", flush=True)
            window_dir = artifact_dir / window
            window_dir.mkdir()
            inner_dir = window_dir / "inner-image-expert"
            inner_dir.mkdir()
            inner_train, inner_sizes, inner_evidence = load_image_sample(
                [_enriched_path(enriched_cache_dir, cutoff) for cutoff in protocol["inner_train"]],
                features,
                seed=config.seed,
            )
            inner_maps_obj = build_category_maps(inner_train)
            validation, validation_sizes, truth_counts, validation_evidence = load_image_validation(
                path=_enriched_path(enriched_cache_dir, protocol["inner_validation"]),
                transactions_path=transactions_path,
                cutoff=protocol["inner_validation"],
                features=features,
            )
            inner_model, inner_model_evidence = train_inner_ranker(
                train_frame=inner_train,
                train_group_sizes=inner_sizes,
                train_evidence=inner_evidence,
                validation_frame=validation,
                validation_group_sizes=validation_sizes,
                validation_truth_counts=truth_counts,
                validation_evidence=validation_evidence,
                features=features,
                name="m310_image_expert",
                artifact_dir=inner_dir,
                config=config,
                category_maps=inner_maps_obj,
            )
            inner_scores = _score_image_rows(
                inner_model,
                _enriched_path(enriched_cache_dir, protocol["inner_validation"]),
                features,
                inner_maps_obj,
            )
            inner_anchor_scores = _score_anchor_base(
                model_path=Path(m33["development"][window]["inner_models"]["anchor"]["model_path"]),
                category_maps_path=Path(m33["development"][window]["inner_category_encoding"]["path"]),
                path=_enriched_path(enriched_cache_dir, protocol["inner_validation"]),
                features=anchor_features,
            )
            inner_policy = _evaluate_policies_fast(
                candidates_path=_enriched_path(enriched_cache_dir, protocol["inner_validation"]),
                image_scores=inner_scores,
                transactions_path=transactions_path,
                cutoff=protocol["inner_validation"],
                base_scores=inner_anchor_scores,
            )
            selected_k, inner_maps = _select_k(inner_policy)
            del inner_train, validation, inner_scores, inner_anchor_scores, inner_model
            gc.collect()

            print(f"M3.10 {window}: outer refit with K={selected_k}", flush=True)
            outer_dir = window_dir / "outer-image-expert"
            outer_dir.mkdir()
            outer_train, outer_sizes, outer_evidence = load_image_sample(
                [_enriched_path(enriched_cache_dir, cutoff) for cutoff in protocol["outer_train"]],
                features,
                seed=config.seed,
            )
            outer_maps_obj = build_category_maps(outer_train)
            rounds = int(inner_model_evidence["best_iteration"])
            outer_model, outer_model_evidence = train_lambdarank(
                frame=outer_train,
                group_sizes=outer_sizes,
                group_evidence=outer_evidence,
                features=features,
                name="m310_image_expert",
                use_ipw=False,
                artifact_dir=outer_dir,
                config=replace(config, num_boost_round=rounds),
                category_maps=outer_maps_obj,
            )
            outer_model_evidence["selected_rounds_from_inner"] = rounds
            outer_scores = _score_image_rows(
                outer_model,
                _enriched_path(enriched_cache_dir, protocol["outer_validation"]),
                features,
                outer_maps_obj,
            )
            outer_anchor_scores = _score_anchor_base(
                model_path=Path(m33["development"][window]["outer_models"]["anchor"]["model_path"]),
                category_maps_path=Path(m33["development"][window]["outer_category_encoding"]["path"]),
                path=_enriched_path(enriched_cache_dir, protocol["outer_validation"]),
                features=anchor_features,
            )
            outer_eval = _evaluate_policies_fast(
                candidates_path=_enriched_path(enriched_cache_dir, protocol["outer_validation"]),
                image_scores=outer_scores,
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                ks=(selected_k,),
                base_scores=outer_anchor_scores,
            )[selected_k]
            development[window] = {
                "protocol": protocol,
                "inner": {
                    "selected_k": selected_k,
                    "k_maps": inner_maps,
                    "policy": inner_policy,
                    "model": inner_model_evidence,
                },
                "outer": {
                    "selected_k": selected_k,
                    "evaluation": outer_eval,
                    "model": outer_model_evidence,
                    "sampling": outer_evidence,
                },
            }
            del outer_train, outer_scores, outer_anchor_scores, outer_model
            gc.collect()
        anchor_values = {
            window: float(m33["development"][window]["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]["map@12"])
            for window in ROLLING_PROTOCOL
        }
        values = {
            window: float(row["outer"]["evaluation"]["segments"]["overall"]["map@12"])
            for window, row in development.items()
        }
        deltas = {window: values[window] - anchor_values[window] for window in values}
        active_top12 = sum(int(row["outer"]["evaluation"]["image_truth_funnel"]["active_top12"]) for row in development.values())
        active_top100 = sum(int(row["outer"]["evaluation"]["image_truth_funnel"]["active_top100"]) for row in development.values())
        summary = {
            "anchor_window_map@12": anchor_values,
            "anchor_mean_map@12": float(np.mean(list(anchor_values.values()))),
            "window_map@12": values,
            "window_delta_vs_base": deltas,
            "mean_map@12": float(np.mean(list(values.values()))),
            "mean_delta_vs_base": float(np.mean(list(deltas.values()))),
            "active_image_truth_top12_total": int(active_top12),
            "active_image_truth_top100_total": int(active_top100),
            "mechanism_gate": bool(active_top100 > 0 and active_top12 > 0),
        }
        # A MAP gain without any admitted image truth is not an image-ranking
        # success: K=0 simply reproduces the frozen base order.  The preregistered
        # promotion gate therefore requires both non-regression MAP and actual
        # image-truth movement into Top100/Top12.
        map_gate = _passes_map_gate(deltas)
        summary["map_gate"] = map_gate
        summary["accepted"] = bool(map_gate and summary["mechanism_gate"])
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.10",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four rolling development windows; 10% users; final week not run",
            "config": asdict(config),
            "contract": {
                "variant": VARIANT,
                "direct_features": DIRECT_FEATURES,
                "half_life_days": HALF_LIFE_DAYS,
                "image_training_filter": "base_present=0 AND soft_present=1 AND user_history_events_12w>0",
                "negative_sampling": "all image positives plus at most 30 same-user unobserved negatives",
                "admission_k": list(ADMISSION_K),
                "inactive_policy": "preserve original candidate order",
                "catalog": "optimistic_all_articles",
                "final_week": "not_run",
            },
            "inputs": {
                "variant_cache": str(variant_cache_dir.resolve()),
                "soft_cache": str(soft_cache_dir.resolve()),
                "neighbor_metrics": _identity(neighbor_metrics_path),
                "transactions": _identity(transactions_path),
                "m33_metrics": _identity(m33_metrics_path),
            },
            "direct_cache": direct,
            "enriched_cache": enriched,
            "features": features,
            "development": development,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_10_FINAL.md"
        result["artifacts"] = {"metrics": str(metrics_path.resolve()), "report": str(report_path.resolve()), "artifact_dir": str(artifact_dir.resolve())}
        _write_json(metrics_path, result)
        report_path.write_text(_render_report(result), encoding="utf-8", newline="\n")
        return result
    except Exception as error:
        _write_json(artifact_dir / f"failure-{time.time_ns()}.json", {
            "schema_version": "m3.10-failure-v1", "status": "failed",
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise
