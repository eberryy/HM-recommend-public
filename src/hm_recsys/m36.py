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

from .m2 import _literal, _write_json
from .m25 import _load_embedding_contract, _stable_topk
from .m25_retrieval import _build_image_candidates, _load_neighbors
from .m28 import _file_identity
from .m34 import OUTER_WINDOWS
from .m35 import CONFIG as M35_CONFIG
from .m35 import _date, _global_visual_candidates


SCHEMA_VERSION = "m3.6-cold-truth-funnel-v1"
CONFIG = {
    "focus_window": "spring_20200318",
    "history_weeks": 12,
    "personalized_seed_k": 5,
    "existing_neighbor_depth": 100,
    "diagnostic_neighbor_depth": 500,
    "diagnostic_rank_buckets": [100, 200, 500],
    "tie_buffer": 32,
    "query_batch_size": 128,
}


def validate_protocol() -> None:
    if any(cutoff >= "2020-09-16" for cutoff in OUTER_WINDOWS.values()):
        raise RuntimeError("M3.6 must not read final-week interactions")
    if CONFIG["diagnostic_neighbor_depth"] != 500:
        raise RuntimeError("M3.6 diagnostic depth drifted")
    if CONFIG["personalized_seed_k"] != M35_CONFIG["image_seed_k"]:
        raise RuntimeError("M3.6 seed contract differs from M3.5")
    if CONFIG["existing_neighbor_depth"] != M35_CONFIG["image_neighbors_per_seed"]:
        raise RuntimeError("M3.6 existing neighbor contract differs from M3.5")
    if CONFIG["focus_window"] not in OUTER_WINDOWS:
        raise RuntimeError("M3.6 focus window is not an outer development window")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_m35_metrics(path: Path) -> dict[str, Any]:
    metrics = _read_json(path)
    if metrics.get("schema_version") != "m3.5-content-season-image-cold-retrieval-v1":
        raise ValueError("M3.6 requires M3.5 cold-retrieval metrics")
    if metrics.get("status") != "measured" or metrics.get("summary", {}).get("final_week") != "not_run":
        raise ValueError("M3.5 evidence is not a measured development-only run")
    if set(metrics.get("development", {})) != set(OUTER_WINDOWS):
        raise ValueError("M3.5 development windows differ from M3.6")
    for window, evidence in metrics["development"].items():
        identity = evidence["artifacts"]["evaluation_db"]
        path_value = Path(identity["path"])
        if _file_identity(path_value) != identity:
            raise ValueError(f"M3.5 evaluation DB identity drift: {window}")
    return metrics


def classify_loss_reason(row: pd.Series) -> str:
    if bool(row["image_selected_hit"]):
        return "selected_top50"
    if bool(row["image_source_hit"]):
        return "top50_fusion_pruning"
    if not bool(row["image_covered"]):
        return "truth_without_image"
    personal_rank = row["personalized_visual_min_rank"]
    global_rank = row["global_visual_min_rank"]
    personal_top100 = pd.notna(personal_rank) and int(personal_rank) <= 100
    global_top100 = pd.notna(global_rank) and int(global_rank) <= 100
    any_top100 = personal_top100 or global_top100
    if any_top100 and not bool(row["cold_scored_eligible"]):
        return "seasonal_hard_gate"
    if any_top100:
        return "post_gate_or_source_pruning"
    personal_top500 = pd.notna(personal_rank) and int(personal_rank) <= 500
    global_top500 = pd.notna(global_rank) and int(global_rank) <= 500
    if personal_top500 or global_top500:
        return "neighbor_depth_101_500"
    if not bool(row["user_has_seed"]):
        return "no_personalized_seed_and_no_global_reach"
    return "visual_not_reached_top500"


def _rank_bucket(personal: float | int | None, global_rank: float | int | None) -> str:
    values = [int(value) for value in (personal, global_rank) if pd.notna(value)]
    if not values:
        return ">500_or_unreached"
    rank = min(values)
    if rank <= 100:
        return "le_100"
    if rank <= 200:
        return "101_200"
    if rank <= 500:
        return "201_500"
    return ">500_or_unreached"


def _exact_subset_neighbors(
    *,
    query_rows: np.ndarray,
    embedding_metrics_path: Path,
    existing_indices: np.ndarray,
    existing_scores: np.ndarray | None = None,
    output_dir: Path,
    device: str,
    stage_label: str = "M3.6",
    preserve_existing_top100_on_backend_drift: bool = False,
) -> dict[str, Any]:
    import torch

    source = _load_embedding_contract(embedding_metrics_path)
    matrix = np.load(Path(source["embedding_path"]), mmap_mode="r")
    query_rows = np.asarray(sorted(set(map(int, query_rows))), dtype=np.int64)
    if len(query_rows) == 0:
        raise RuntimeError("M3.6 has no image seed query rows")
    if np.any(query_rows < 0) or np.any(query_rows >= len(matrix)):
        raise ValueError("M3.6 query row out of embedding range")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("M3.6 requested CUDA but torch.cuda is unavailable")
    output_dir.mkdir(parents=True)
    depth = int(CONFIG["diagnostic_neighbor_depth"])
    tie_buffer = int(CONFIG["tie_buffer"])
    out_indices = np.empty((len(query_rows), depth), dtype=np.int32)
    out_scores = np.empty((len(query_rows), depth), dtype=np.float16)
    corpus = torch.from_numpy(np.asarray(matrix, dtype=np.float32)).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    ambiguous_rows: list[int] = []
    batch_size = int(CONFIG["query_batch_size"])
    for start in range(0, len(query_rows), batch_size):
        end = min(start + batch_size, len(query_rows))
        row_tensor = torch.from_numpy(query_rows[start:end]).to(device=device, dtype=torch.long)
        similarities = corpus[row_tensor] @ corpus.T
        local = torch.arange(end - start, device=device)
        similarities[local, row_tensor] = -torch.inf
        values, neighbors = torch.topk(
            similarities, k=depth + tie_buffer, dim=1, largest=True, sorted=False
        )
        candidate_scores = values.cpu().numpy()
        candidate_indices = neighbors.cpu().numpy()
        for offset in range(end - start):
            stable_indices, stable_scores, ambiguous = _stable_topk(
                candidate_indices[offset], candidate_scores[offset], depth
            )
            if ambiguous:
                full_scores = similarities[offset].cpu().numpy()
                all_rows = np.arange(len(full_scores), dtype=np.int64)
                order = np.lexsort((all_rows, -full_scores))[:depth]
                stable_indices = order
                stable_scores = full_scores[order]
                ambiguous_rows.append(int(query_rows[start + offset]))
            out_indices[start + offset] = stable_indices.astype(np.int32)
            out_scores[start + offset] = stable_scores.astype(np.float16)
        del similarities, values, neighbors, row_tensor
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"{stage_label} exact Top500 {end}/{len(query_rows)} query seeds", flush=True)
    elapsed = time.perf_counter() - started
    raw_top100_mismatch = int(
        np.count_nonzero(out_indices[:, :100] != existing_indices[query_rows, :100])
    )
    prefix_policy = "raw_backend_exact_parity"
    if raw_top100_mismatch:
        if not preserve_existing_top100_on_backend_drift:
            raise RuntimeError(
                f"M3.6 exact Top500 does not reproduce frozen Top100: {raw_top100_mismatch} cells"
            )
        if existing_scores is None:
            raise ValueError("frozen Top100 prefix requires existing scores")
        for offset, query_row in enumerate(query_rows):
            prefix = np.asarray(existing_indices[int(query_row), :100], dtype=np.int32)
            prefix_set = set(map(int, prefix))
            tail_positions = [
                position
                for position, candidate in enumerate(out_indices[offset])
                if int(candidate) not in prefix_set
            ][:400]
            if len(tail_positions) != 400:
                raise RuntimeError("M3.7 CPU extension cannot fill frozen-prefix Top500")
            tail_positions_array = np.asarray(tail_positions, dtype=np.int64)
            out_indices[offset] = np.concatenate(
                [prefix, out_indices[offset, tail_positions_array]]
            )
            out_scores[offset] = np.concatenate(
                [
                    np.asarray(existing_scores[int(query_row), :100], dtype=np.float16),
                    out_scores[offset, tail_positions_array],
                ]
            )
        if not np.array_equal(out_indices[:, :100], existing_indices[query_rows, :100]):
            raise RuntimeError("M3.7 frozen Top100 prefix reconstruction failed")
        prefix_policy = "frozen_cuda_top100_then_exhaustive_cpu_unique_101_500"
    query_path = output_dir / "query_rows.int64.npy"
    indices_path = output_dir / "neighbor_indices.int32.npy"
    scores_path = output_dir / "neighbor_scores.float16.npy"
    np.save(query_path, query_rows)
    np.save(indices_path, out_indices)
    np.save(scores_path, out_scores)
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    del corpus
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {
        "query_rows": query_rows,
        "indices": out_indices,
        "scores": out_scores,
        "evidence": {
            "query_count": int(len(query_rows)),
            "depth": depth,
            "top100_exact_parity": True,
            "raw_backend_top100_mismatch_cells_before_prefix_reconstruction": raw_top100_mismatch,
            "top100_prefix_policy": prefix_policy,
            "ambiguous_boundaries_resolved_full_sort": len(ambiguous_rows),
            "elapsed_seconds": elapsed,
            "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
            "cuda_peak_allocated_bytes": peak,
            "artifacts": {
                "query_rows": _file_identity(query_path),
                "indices": _file_identity(indices_path),
                "scores": _file_identity(scores_path),
            },
        },
    }


def _prepare_window(
    *,
    window: str,
    cutoff: str,
    m35_db_path: Path,
    transactions_path: Path,
    articles_path: Path,
    image_items: pd.DataFrame,
    existing_indices: np.ndarray,
    existing_scores: np.ndarray,
    article_ids: np.ndarray,
    db_path: Path,
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"ATTACH {_literal(m35_db_path)} AS m35 (READ_ONLY)")
        con.register("image_items_frame", image_items[["row_index", "article_id"]])
        con.execute(
            f"""
            CREATE TABLE eval_truth AS SELECT * FROM m35.eval_truth;
            CREATE TABLE cold_truth AS
              SELECT customer_id,article_id FROM eval_truth WHERE item_temperature='cold';
            CREATE TABLE base AS SELECT * FROM m35.base;
            CREATE TABLE attribute_source AS SELECT * FROM m35.attribute_source;
            CREATE TABLE attr_selected AS SELECT * FROM m35.attr_selected;
            CREATE TABLE image_source AS SELECT * FROM m35.image_source;
            CREATE TABLE image_selected AS SELECT * FROM m35.image_selected;
            CREATE TABLE combined_new AS SELECT * FROM m35.combined_new;
            DETACH m35;
            CREATE TABLE image_items AS SELECT * FROM image_items_frame;
            CREATE TABLE article_dim AS
              SELECT article_id,try_cast(product_type_no AS INTEGER) AS product_type_no,
                     try_cast(graphical_appearance_no AS INTEGER) AS graphical_appearance_no,
                     try_cast(colour_group_code AS INTEGER) AS colour_group_code,
                     try_cast(perceived_colour_value_id AS INTEGER) AS perceived_colour_value_id
              FROM read_csv_auto({_literal(articles_path)},header=true,all_varchar=true);
            CREATE TABLE warm_catalog AS SELECT DISTINCT article_id
              FROM read_parquet({_literal(transactions_path)}) WHERE t_dat<{_date(cutoff)};
            CREATE TABLE cold_catalog AS SELECT a.* FROM article_dim a
              LEFT JOIN warm_catalog w USING(article_id) WHERE w.article_id IS NULL;
            CREATE TABLE current_events AS
              SELECT t.t_dat,t.article_id,a.product_type_no,a.graphical_appearance_no,
                     a.colour_group_code,a.perceived_colour_value_id
              FROM read_parquet({_literal(transactions_path)}) t JOIN article_dim a USING(article_id)
              WHERE t.t_dat>={_date(cutoff)}-INTERVAL 84 DAY AND t.t_dat<{_date(cutoff)};
            CREATE TABLE prior_events AS
              SELECT t.t_dat,t.article_id,a.product_type_no
              FROM read_parquet({_literal(transactions_path)}) t JOIN article_dim a USING(article_id)
              WHERE t.t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 84 DAY
                AND t.t_dat<{_date(cutoff)}-INTERVAL 1 YEAR;
            """
        )
        con.unregister("image_items_frame")
        current84, current28 = con.execute(
            f"SELECT count(*),count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 28 DAY) FROM current_events"
        ).fetchone()
        prior84, prior28 = con.execute(
            f"SELECT count(*),count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 28 DAY) FROM prior_events"
        ).fetchone()
        con.execute(
            f"""
            CREATE TABLE prior_type AS
            WITH s AS (
              SELECT product_type_no,
                count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 28 DAY) AS events28,
                count(*) AS events84,
                count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::DOUBLE/{int(prior28)}.0 AS share28,
                count(*)::DOUBLE/{int(prior84)}.0 AS share84
              FROM prior_events GROUP BY product_type_no)
            SELECT *,share28/nullif(share84,0) AS lift FROM s;
            """
        )
        for key in (
            "product_type_no",
            "graphical_appearance_no",
            "colour_group_code",
            "perceived_colour_value_id",
        ):
            con.execute(
                f"""
                CREATE TABLE current_{key} AS
                WITH s AS (
                  SELECT {key},
                    count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 28 DAY) AS events28,
                    count(*) AS events84,
                    count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 28 DAY)::DOUBLE/{int(current28)}.0 AS share28,
                    count(*)::DOUBLE/{int(current84)}.0 AS share84
                  FROM current_events GROUP BY {key})
                SELECT *,share28/nullif(share84,0) AS lift FROM s;
                """
            )
        score = " + ".join(
            [
                "greatest(ln(greatest(p.lift,1.0)),0.0)*p.events28/(p.events28+50.0)",
                "greatest(ln(greatest(t.lift,1.0)),0.0)*t.events28/(t.events28+50.0)",
                "0.5*greatest(ln(greatest(g.lift,1.0)),0.0)*g.events28/(g.events28+30.0)",
                "0.5*greatest(ln(greatest(c.lift,1.0)),0.0)*c.events28/(c.events28+30.0)",
                "0.25*greatest(ln(greatest(v.lift,1.0)),0.0)*v.events28/(v.events28+30.0)",
            ]
        )
        con.execute(
            f"""
            CREATE TABLE cold_scored AS
              SELECT a.*,({score}) AS attribute_season_score,p.lift AS prior_type_lift,
                     p.events28 AS prior_type_events28
              FROM cold_catalog a JOIN prior_type p USING(product_type_no)
              LEFT JOIN current_product_type_no t USING(product_type_no)
              LEFT JOIN current_graphical_appearance_no g USING(graphical_appearance_no)
              LEFT JOIN current_colour_group_code c USING(colour_group_code)
              LEFT JOIN current_perceived_colour_value_id v USING(perceived_colour_value_id)
              WHERE p.lift>={M35_CONFIG['seasonal_product_type_lift']}
                AND p.events28>={M35_CONFIG['minimum_prior_type_events_28d']};
            CREATE TABLE cold_type_top AS SELECT *,row_number() OVER(
              PARTITION BY product_type_no ORDER BY attribute_season_score DESC,article_id) AS type_rank
              FROM cold_scored QUALIFY type_rank<={M35_CONFIG['attribute_cold_items_per_type']};
            CREATE TABLE user_type AS
            WITH counts AS (
              SELECT x.customer_id,a.product_type_no,count(*) AS type_events
              FROM read_parquet({_literal(transactions_path)}) x
              JOIN (SELECT DISTINCT customer_id FROM cold_truth) u USING(customer_id)
              JOIN article_dim a USING(article_id)
              WHERE x.t_dat>={_date(cutoff)}-INTERVAL {CONFIG['history_weeks']} WEEK
                AND x.t_dat<{_date(cutoff)} GROUP BY x.customer_id,a.product_type_no)
            SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY type_events DESC,product_type_no) AS type_rank
              FROM counts QUALIFY type_rank<={M35_CONFIG['user_top_product_types']};
            CREATE TABLE seeds AS
            WITH latest AS (
              SELECT x.customer_id,x.article_id,max(x.t_dat) AS latest_t_dat
              FROM read_parquet({_literal(transactions_path)}) x
              JOIN (SELECT DISTINCT customer_id FROM cold_truth) u USING(customer_id)
              JOIN image_items i USING(article_id)
              WHERE x.t_dat>={_date(cutoff)}-INTERVAL {CONFIG['history_weeks']} WEEK
                AND x.t_dat<{_date(cutoff)} GROUP BY x.customer_id,x.article_id), ranked AS (
              SELECT l.*,i.row_index,row_number() OVER(PARTITION BY customer_id
                ORDER BY latest_t_dat DESC,article_id) AS seed_rank FROM latest l JOIN image_items i USING(article_id))
            SELECT * FROM ranked WHERE seed_rank<={CONFIG['personalized_seed_k']};
            """
        )
        seed_frame = con.execute(
            "SELECT customer_id,row_index,seed_rank FROM seeds ORDER BY customer_id,seed_rank"
        ).fetchdf()
        _build_image_candidates(
            con,
            seed_frame,
            existing_indices,
            existing_scores,
            article_ids,
            M35_CONFIG["raw_image_candidate_k"],
            user_chunk=1000,
        )
        global_seeds = con.execute(
            f"""
            WITH s AS (
              SELECT e.article_id,count(*) AS item_events FROM current_events e
              JOIN (SELECT DISTINCT product_type_no FROM cold_scored) c USING(product_type_no)
              JOIN image_items i ON e.article_id=i.article_id
              WHERE e.t_dat>={_date(cutoff)}-INTERVAL 28 DAY GROUP BY e.article_id), ranked AS (
              SELECT s.*,i.row_index,row_number() OVER(ORDER BY item_events DESC,s.article_id) AS seed_rank
              FROM s JOIN image_items i USING(article_id))
            SELECT row_index,item_events,seed_rank FROM ranked
              WHERE seed_rank<={M35_CONFIG['global_visual_seed_k']} ORDER BY seed_rank
            """
        ).fetchdf()
        global_visual = _global_visual_candidates(global_seeds, existing_indices, existing_scores, article_ids)
        con.register("global_visual_frame", global_visual)
        con.execute("CREATE TABLE global_visual AS SELECT * FROM global_visual_frame")
        con.unregister("global_visual_frame")
        latest = con.execute("SELECT max(latest_t_dat) FROM seeds").fetchone()[0]
        if latest is not None and str(latest) >= cutoff:
            raise RuntimeError(f"M3.6 temporal leakage in seeds: {window}")
        return {
            "connection": con,
            "seed_frame": seed_frame,
            "global_seeds": global_seeds,
            "latest_seed_date": str(latest) if latest is not None else None,
            "truth_pairs": int(con.execute("SELECT count(*) FROM cold_truth").fetchone()[0]),
        }
    except Exception:
        con.close()
        raise


def _add_depth_and_finalize(
    *,
    context: dict[str, Any],
    deep_query_rows: np.ndarray,
    deep_indices: np.ndarray,
    article_row_by_id: dict[str, int],
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    con: duckdb.DuckDBPyConnection = context["connection"]
    frame = con.execute(
        """
        SELECT t.customer_id,t.article_id,a.product_type_no,
          (a.article_id IS NOT NULL) AS article_covered,
          (i.row_index IS NOT NULL) AS image_covered,
          (b.article_id IS NOT NULL) AS base_hit,
          (cs.article_id IS NOT NULL) AS cold_scored_eligible,
          (ct.article_id IS NOT NULL) AS cold_type_top100_eligible,
          (ut.customer_id IS NOT NULL) AS user_top5_type,
          (s.customer_id IS NOT NULL) AS user_has_seed,
          ar.personalized_rank AS attribute_personalized_rank,
          ar.global_rank AS attribute_global_rank,
          ar.attribute_rank AS attribute_source_rank,
          ats.source_rank AS attribute_selected_rank,
          ic.image_rank AS raw_image_candidate_rank,
          gv.global_visual_raw_rank AS global_visual_raw_rank,
          im.personalized_image_rank,im.global_visual_rank,im.image_cold_rank AS image_source_rank,
          ims.selected_rank AS image_selected_rank,
          cn.new_rank AS combined_new_rank
        FROM cold_truth t
        LEFT JOIN article_dim a USING(article_id)
        LEFT JOIN image_items i USING(article_id)
        LEFT JOIN (SELECT DISTINCT customer_id,article_id FROM base) b USING(customer_id,article_id)
        LEFT JOIN cold_scored cs USING(article_id)
        LEFT JOIN cold_type_top ct USING(article_id)
        LEFT JOIN user_type ut ON ut.customer_id=t.customer_id AND ut.product_type_no=a.product_type_no
        LEFT JOIN (SELECT DISTINCT customer_id FROM seeds) s USING(customer_id)
        LEFT JOIN attribute_source ar USING(customer_id,article_id)
        LEFT JOIN attr_selected ats USING(customer_id,article_id)
        LEFT JOIN image_candidates ic USING(customer_id,article_id)
        LEFT JOIN global_visual gv USING(article_id)
        LEFT JOIN image_source im USING(customer_id,article_id)
        LEFT JOIN image_selected ims USING(customer_id,article_id)
        LEFT JOIN combined_new cn USING(customer_id,article_id)
        ORDER BY t.customer_id,t.article_id
        """
    ).fetchdf()
    query_position = {int(row): index for index, row in enumerate(deep_query_rows)}
    user_seeds = {
        customer: group["row_index"].astype(int).tolist()
        for customer, group in context["seed_frame"].groupby("customer_id", sort=False)
    }
    global_seed_rows = context["global_seeds"]["row_index"].astype(int).tolist()

    def minimum_rank(seed_rows: list[int], truth_row: int | None) -> float:
        if truth_row is None:
            return np.nan
        best: int | None = None
        for seed_row in seed_rows:
            neighbors = deep_indices[query_position[int(seed_row)]]
            matches = np.flatnonzero(neighbors == truth_row)
            if len(matches):
                rank = int(matches[0]) + 1
                best = rank if best is None else min(best, rank)
        return float(best) if best is not None else np.nan

    personal_ranks: list[float] = []
    global_ranks: list[float] = []
    for row in frame.itertuples(index=False):
        truth_row = article_row_by_id.get(str(row.article_id))
        personal_ranks.append(minimum_rank(user_seeds.get(str(row.customer_id), []), truth_row))
        global_ranks.append(minimum_rank(global_seed_rows, truth_row))
    frame["personalized_visual_min_rank"] = personal_ranks
    frame["global_visual_min_rank"] = global_ranks
    for source, target in (
        ("attribute_personalized_rank", "attribute_personalized_hit"),
        ("attribute_global_rank", "attribute_global_hit"),
        ("attribute_source_rank", "attribute_source_hit"),
        ("attribute_selected_rank", "attribute_selected_hit"),
        ("image_source_rank", "image_source_hit"),
        ("image_selected_rank", "image_selected_hit"),
        ("personalized_image_rank", "image_personalized_hit"),
        ("global_visual_rank", "image_global_hit"),
        ("raw_image_candidate_rank", "raw_image_candidate_hit"),
        ("global_visual_raw_rank", "global_raw_candidate_hit"),
    ):
        frame[target] = frame[source].notna()
    frame["combined_hit"] = frame["combined_new_rank"].notna() | frame["base_hit"]
    frame["visual_rank_bucket"] = [
        _rank_bucket(personal, global_rank)
        for personal, global_rank in zip(
            frame["personalized_visual_min_rank"], frame["global_visual_min_rank"]
        )
    ]
    frame["loss_reason"] = frame.apply(classify_loss_reason, axis=1)
    con.register("truth_funnel_frame", frame)
    con.execute(
        f"COPY truth_funnel_frame TO {_literal(output_path)} (FORMAT PARQUET,COMPRESSION ZSTD)"
    )
    con.unregister("truth_funnel_frame")
    con.execute("CHECKPOINT")
    con.close()
    context["connection"] = None
    audit = {
        "rows": int(len(frame)),
        "unique_pairs": int(frame[["customer_id", "article_id"]].drop_duplicates().shape[0]),
        "users": int(frame["customer_id"].nunique()),
        "items": int(frame["article_id"].nunique()),
        "latest_seed_date": context["latest_seed_date"],
        "artifact": _file_identity(output_path),
    }
    if audit["rows"] != audit["unique_pairs"]:
        raise RuntimeError("M3.6 truth funnel contains duplicate user-item rows")
    return frame, audit


def _flag_summary(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    flag = frame[column].fillna(False).astype(bool)
    per_user = frame.assign(_flag=flag).groupby("customer_id").agg(hits=("_flag", "sum"), n=("_flag", "size"))
    return {
        "pairs": int(flag.sum()),
        "pair_rate": float(flag.mean()) if len(flag) else 0.0,
        "users_with_hit": int(frame.loc[flag, "customer_id"].nunique()),
        "unique_items_hit": int(frame.loc[flag, "article_id"].nunique()),
        "user_mean_recall": float((per_user["hits"] / per_user["n"]).mean()) if len(per_user) else 0.0,
    }


def _summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    missed = frame[~frame["base_hit"].astype(bool)].copy()
    flags = [
        "article_covered",
        "image_covered",
        "cold_scored_eligible",
        "cold_type_top100_eligible",
        "user_top5_type",
        "user_has_seed",
        "attribute_personalized_hit",
        "attribute_global_hit",
        "attribute_source_hit",
        "attribute_selected_hit",
        "raw_image_candidate_hit",
        "global_raw_candidate_hit",
        "image_personalized_hit",
        "image_global_hit",
        "image_source_hit",
        "image_selected_hit",
        "combined_hit",
    ]
    personal = pd.to_numeric(missed["personalized_visual_min_rank"], errors="coerce")
    global_rank = pd.to_numeric(missed["global_visual_min_rank"], errors="coerce")
    best = pd.concat([personal, global_rank], axis=1).min(axis=1, skipna=True)
    eligible = missed["cold_scored_eligible"].astype(bool)
    visual_gate_cross = {}
    for label, mask in (
        ("le_100", best <= 100),
        ("101_200", (best > 100) & (best <= 200)),
        ("201_500", (best > 200) & (best <= 500)),
        ("gt_500_or_unreached", best.isna() | (best > 500)),
    ):
        visual_gate_cross[label] = {
            "eligible": int((mask & eligible).sum()),
            "ineligible": int((mask & ~eligible).sum()),
        }
    route_reachability = {
        "personal_le_100": int((personal <= 100).sum()),
        "personal_101_500": int(((personal > 100) & (personal <= 500)).sum()),
        "global_le_100": int((global_rank <= 100).sum()),
        "global_101_500": int(((global_rank > 100) & (global_rank <= 500)).sum()),
        "both_le_100": int(((personal <= 100) & (global_rank <= 100)).sum()),
        "eligible_personal_le_100": int(((personal <= 100) & eligible).sum()),
        "eligible_global_le_100": int(((global_rank <= 100) & eligible).sum()),
        "attribute_structural_reachable": int(
            (missed["cold_type_top100_eligible"].astype(bool) & missed["user_top5_type"].astype(bool)).sum()
        ),
    }
    return {
        "denominators": {
            "cold_truth_pairs": int(len(frame)),
            "cold_truth_users": int(frame["customer_id"].nunique()),
            "cold_truth_items": int(frame["article_id"].nunique()),
            "base_missed_pairs": int(len(missed)),
            "base_missed_users": int(missed["customer_id"].nunique()),
        },
        "all_cold_truth": {column: _flag_summary(frame, column) for column in flags},
        "base_missed_truth": {column: _flag_summary(missed, column) for column in flags},
        "visual_rank_buckets_base_missed": {
            str(key): int(value)
            for key, value in missed["visual_rank_bucket"].value_counts().sort_index().items()
        },
        "visual_gate_cross_base_missed": visual_gate_cross,
        "route_reachability_base_missed": route_reachability,
        "exclusive_loss_reasons_base_missed": {
            str(key): int(value)
            for key, value in missed["loss_reason"].value_counts().sort_index().items()
        },
    }


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# M3.6：cold-truth retrieval funnel",
        "",
        "- status：measured diagnostic；无 retrieval/ranking 晋级 gate。",
        "- focus：spring_20200318；其余三窗为对照。",
        "- final week：not run。",
        "- Top500 仅为 exact reachability 诊断，不是新候选预算。",
        "",
        "## Terminology",
        "",
        "- `cold truth pair`：验证周的去重用户—商品正例，且商品在 cutoff 前零交易。",
        "- `base-missed pair`：该 cold truth pair 未被该用户的冻结 base 候选池命中；不是负样本。",
        "- 本报告所有 funnel 数字默认单位为 user-item pair；用户数或商品数会另列。",
        "- `cold_scored eligible`：商品通过去年同期 product-type lift/support 硬门槛，不代表可售。",
        "- `visual reachable`：truth 商品位于至少一个 seed 的 exact TopK；不保证经过聚合后保留。",
        "",
        "## Base-missed cold truth funnel",
        "",
        "| window | base-missed pairs | pairs with image | cold_scored-eligible pairs | pairs whose user has seed | pairs in personalized raw Top300 | pairs in image source Top100 | pairs in strict image Top50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window in OUTER_WINDOWS:
        summary = result["windows"][window]["summary"]
        missed = summary["denominators"]["base_missed_pairs"]
        flags = summary["base_missed_truth"]
        lines.append(
            f"| {window} | {missed} | {flags['image_covered']['pairs']} | "
            f"{flags['cold_scored_eligible']['pairs']} | {flags['user_has_seed']['pairs']} | "
            f"{flags['raw_image_candidate_hit']['pairs']} | {flags['image_source_hit']['pairs']} | "
            f"{flags['image_selected_hit']['pairs']} |"
        )
    lines.extend(["", "## Visual reachability buckets among base misses", "", "| window | <=100 | 101-200 | 201-500 | >500/unreached |", "|---|---:|---:|---:|---:|"])
    for window in OUTER_WINDOWS:
        buckets = result["windows"][window]["summary"]["visual_rank_buckets_base_missed"]
        lines.append(
            f"| {window} | {buckets.get('le_100', 0)} | {buckets.get('101_200', 0)} | "
            f"{buckets.get('201_500', 0)} | {buckets.get('>500_or_unreached', 0)} |"
        )
    lines.extend(["", "## Mutually exclusive loss reasons among base misses", ""])
    reasons = sorted({reason for window in result["windows"].values() for reason in window["summary"]["exclusive_loss_reasons_base_missed"]})
    lines.append("| reason | " + " | ".join(OUTER_WINDOWS) + " |")
    lines.append("|---|" + "---:|" * len(OUTER_WINDOWS))
    for reason in reasons:
        values = [result["windows"][window]["summary"]["exclusive_loss_reasons_base_missed"].get(reason, 0) for window in OUTER_WINDOWS]
        lines.append("| " + reason + " | " + " | ".join(map(str, values)) + " |")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- `articles.csv` 没有上架、库存或曝光时间；真实时点 eligibility 不可识别。",
            "- `cold_scored` 是 M3.5 的属性季节硬门槛，不等同于商品可售性。",
            "- exact Top500 复现冻结 Top100 后才用于分桶；结果不自动授权扩大 candidate budget。",
            "- 机器可读证据：`metrics.json` 与各窗 `truth-funnel.parquet`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_m36(
    *,
    m35_metrics_path: Path,
    neighbor_metrics_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    run_id: str,
    device: str = "cuda",
) -> dict[str, Any]:
    validate_protocol()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"M3.6 output exists: {output_dir} or {artifact_dir}")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    m35_metrics_path = m35_metrics_path.resolve()
    neighbor_metrics_path = neighbor_metrics_path.resolve()
    m35 = _validate_m35_metrics(m35_metrics_path)
    if _file_identity(neighbor_metrics_path) != m35["inputs"]["neighbor_metrics"]:
        raise ValueError("M3.6 neighbor artifact differs from frozen M3.5 input")
    _, existing_indices, existing_scores, image_items = _load_neighbors(neighbor_metrics_path)
    neighbor_metrics = _read_json(neighbor_metrics_path)
    embedding_metrics_path = Path(neighbor_metrics["contract"]["source"]["metrics_path"]).resolve()
    embedding_source = _load_embedding_contract(embedding_metrics_path)
    article_ids = image_items.sort_values("row_index")["article_id"].astype(str).to_numpy()
    article_row_by_id = {article_id: index for index, article_id in enumerate(article_ids)}
    transactions_path = Path(m35["inputs"]["transactions"]["path"]).resolve()
    articles_path = Path(m35["inputs"]["articles"]["path"]).resolve()
    contexts: dict[str, dict[str, Any]] = {}
    all_query_rows: set[int] = set()
    try:
        for window, cutoff in OUTER_WINDOWS.items():
            print(f"M3.6 preparing {window} ({cutoff})", flush=True)
            db_identity = m35["development"][window]["artifacts"]["evaluation_db"]
            context = _prepare_window(
                window=window,
                cutoff=cutoff,
                m35_db_path=Path(db_identity["path"]),
                transactions_path=transactions_path,
                articles_path=articles_path,
                image_items=image_items,
                existing_indices=existing_indices,
                existing_scores=existing_scores,
                article_ids=article_ids,
                db_path=artifact_dir / window / "diagnostic.duckdb",
            )
            contexts[window] = context
            all_query_rows.update(context["seed_frame"]["row_index"].astype(int).tolist())
            all_query_rows.update(context["global_seeds"]["row_index"].astype(int).tolist())
        deep = _exact_subset_neighbors(
            query_rows=np.asarray(sorted(all_query_rows), dtype=np.int64),
            embedding_metrics_path=embedding_metrics_path,
            existing_indices=existing_indices,
            output_dir=artifact_dir / "subset-exact-top500",
            device=device,
        )
        windows: dict[str, Any] = {}
        for window in OUTER_WINDOWS:
            print(f"M3.6 finalizing {window}", flush=True)
            path = artifact_dir / window / "truth-funnel.parquet"
            frame, audit = _add_depth_and_finalize(
                context=contexts[window],
                deep_query_rows=deep["query_rows"],
                deep_indices=deep["indices"],
                article_row_by_id=article_row_by_id,
                output_path=path,
            )
            windows[window] = {"cutoff": OUTER_WINDOWS[window], "audit": audit, "summary": _summarize_frame(frame)}
        elapsed = time.perf_counter() - started
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.6",
            "status": "measured_diagnostic",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four outer development windows; spring focus; final week not run",
            "contract": CONFIG,
            "inputs": {
                "m35_metrics": _file_identity(m35_metrics_path),
                "neighbor_metrics": _file_identity(neighbor_metrics_path),
                "embedding_metrics": _file_identity(embedding_metrics_path),
                "transactions": _file_identity(transactions_path),
                "articles": _file_identity(articles_path),
                "m35_run_id": m35["run_id"],
                "embedding_source": embedding_source,
            },
            "deep_neighbors": deep["evidence"],
            "windows": windows,
            "final_week": "not_run",
            "elapsed_seconds": elapsed,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_6_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path),
            "report": str(report_path),
            "artifact_dir": str(artifact_dir),
        }
        _write_json(metrics_path, result)
        report_path.write_text(_render_report(result), encoding="utf-8")
        return result
    finally:
        for context in contexts.values():
            connection = context.get("connection")
            if connection is not None:
                connection.close()
