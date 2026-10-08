from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import validate_candidate_artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _date(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    for attempt in range(1, 6):
        try:
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * attempt)


def _load_neighbors(metrics_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    metrics_path = metrics_path.resolve()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != "m2.5-exact-image-neighbors-v1" or metrics.get("status") != "completed":
        raise ValueError("requires completed m2.5 exact neighbors")
    if int(metrics["query_rows"]) != int(metrics["corpus_rows"]):
        raise ValueError("retrieval requires full-corpus neighbor rows")
    audit = metrics["audit"]
    if any(int(audit[key]) for key in ("self_neighbors", "duplicate_neighbors", "ambiguous_boundaries")):
        raise ValueError("neighbor audit did not pass")
    artifacts = metrics["artifacts"]
    indices_path = Path(artifacts["indices"]).resolve()
    scores_path = Path(artifacts["scores"]).resolve()
    for path, size, sha in (
        (indices_path, int(artifacts["indices_bytes"]), artifacts["indices_sha256"]),
        (scores_path, int(artifacts["scores_bytes"]), artifacts["scores_sha256"]),
    ):
        if not path.is_file() or path.stat().st_size != size or _sha256(path) != sha:
            raise ValueError(f"neighbor artifact identity mismatch: {path}")
    indices = np.load(indices_path, mmap_mode="r")
    scores = np.load(scores_path, mmap_mode="r")
    source = metrics["contract"]["source"]
    items_path = Path(source["items_path"]).resolve()
    if _sha256(items_path) != source["items_sha256"]:
        raise ValueError("neighbor source items SHA mismatch")
    items = pd.read_csv(items_path, dtype={"article_id": str})[["row_index", "article_id"]]
    if indices.shape != scores.shape or indices.shape[0] != len(items):
        raise ValueError("neighbor arrays/items shape mismatch")
    return metrics, indices, scores, items


def _build_image_candidates(
    connection: duckdb.DuckDBPyConnection,
    seeds: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    article_ids: np.ndarray,
    max_candidates: int,
    user_chunk: int = 500,
) -> dict[str, Any]:
    connection.execute(
        """
        CREATE TABLE image_candidates (
            customer_id VARCHAR, article_id VARCHAR, image_rank INTEGER,
            image_score DOUBLE, image_cosine DOUBLE, best_seed_rank INTEGER,
            best_neighbor_rank INTEGER, seed_support INTEGER
        )
        """
    )
    users = seeds["customer_id"].drop_duplicates().tolist()
    rows_written = 0
    for chunk_start in range(0, len(users), user_chunk):
        chunk_users = set(users[chunk_start : chunk_start + user_chunk])
        chunk = seeds[seeds["customer_id"].isin(chunk_users)]
        records: list[dict[str, Any]] = []
        for customer_id, group in chunk.groupby("customer_id", sort=True):
            candidates: dict[int, list[float | int]] = {}
            for seed in group.itertuples(index=False):
                seed_row = int(seed.row_index)
                seed_rank = int(seed.seed_rank)
                discount = 1.0 / (1.0 + 0.1 * (seed_rank - 1))
                for neighbor_rank, (neighbor_row, cosine) in enumerate(
                    zip(indices[seed_row], scores[seed_row]), start=1
                ):
                    row = int(neighbor_row)
                    raw = float(cosine)
                    weighted = raw * discount
                    prior = candidates.get(row)
                    if prior is None:
                        candidates[row] = [weighted, raw, seed_rank, neighbor_rank, 1]
                    else:
                        prior[4] = int(prior[4]) + 1
                        if (weighted, raw, -seed_rank, -neighbor_rank) > (
                            float(prior[0]), float(prior[1]), -int(prior[2]), -int(prior[3])
                        ):
                            prior[0:4] = [weighted, raw, seed_rank, neighbor_rank]
            ordered = sorted(
                candidates.items(),
                key=lambda value: (
                    -float(value[1][0]), -float(value[1][1]),
                    int(value[1][2]), int(value[1][3]), value[0],
                ),
            )[:max_candidates]
            for rank, (row, evidence) in enumerate(ordered, start=1):
                records.append(
                    {
                        "customer_id": customer_id,
                        "article_id": str(article_ids[row]),
                        "image_rank": rank,
                        "image_score": float(evidence[0]),
                        "image_cosine": float(evidence[1]),
                        "best_seed_rank": int(evidence[2]),
                        "best_neighbor_rank": int(evidence[3]),
                        "seed_support": int(evidence[4]),
                    }
                )
        frame = pd.DataFrame.from_records(records)
        connection.register("image_chunk", frame)
        connection.execute("INSERT INTO image_candidates SELECT * FROM image_chunk")
        connection.unregister("image_chunk")
        rows_written += len(frame)
        print(
            f"image candidates users {min(chunk_start + user_chunk, len(users))}/{len(users)}, rows={rows_written}",
            flush=True,
        )
    return {"users_with_candidates": len(users), "rows": rows_written}


def _candidate_metrics(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    budget: int,
    segment_predicate: str = "TRUE",
    truth_table: str = "eval_truth",
) -> dict[str, float | int]:
    row = connection.execute(
        f"""
        WITH segment_truth AS (
            SELECT customer_id, article_id FROM {truth_table} WHERE {segment_predicate}
        ), truth_counts AS (
            SELECT customer_id, count(*) AS truth_count FROM segment_truth GROUP BY customer_id
        ), candidate_hits AS (
            SELECT c.customer_id, count(t.article_id) AS hits
            FROM {table} c JOIN truth_counts u USING (customer_id)
            LEFT JOIN segment_truth t USING (customer_id, article_id)
            WHERE c.candidate_rank <= {budget}
            GROUP BY c.customer_id
        ), top_rows AS (
            SELECT c.customer_id, c.candidate_rank,
                   (t.article_id IS NOT NULL)::INTEGER AS is_hit
            FROM {table} c JOIN truth_counts u USING (customer_id)
            LEFT JOIN segment_truth t USING (customer_id, article_id)
            WHERE c.candidate_rank <= 12
        ), precision_rows AS (
            SELECT *, sum(is_hit) OVER (
                PARTITION BY customer_id ORDER BY candidate_rank
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_hits
            FROM top_rows
        ), ranking AS (
            SELECT customer_id,
                   sum(CASE WHEN is_hit=1 THEN cumulative_hits::DOUBLE/candidate_rank ELSE 0 END) AS precision_sum
            FROM precision_rows GROUP BY customer_id
        )
        SELECT count(*), sum(truth_count),
               avg(coalesce(candidate_hits.hits,0)::DOUBLE/truth_count),
               avg((coalesce(candidate_hits.hits,0)>0)::INTEGER),
               avg(least(coalesce(candidate_hits.hits,0),12)::DOUBLE/least(truth_count,12)),
               avg(coalesce(ranking.precision_sum,0)/least(truth_count,12))
        FROM truth_counts
        LEFT JOIN candidate_hits USING (customer_id)
        LEFT JOIN ranking USING (customer_id)
        """
    ).fetchone()
    if row is None or int(row[0]) == 0:
        return {"users": 0, "truth_pairs": 0, f"candidate_recall@{budget}": 0.0,
                f"candidate_hit_rate@{budget}": 0.0, "oracle_map@12": 0.0, "map@12": 0.0}
    return {
        "users": int(row[0]), "truth_pairs": int(row[1]),
        f"candidate_recall@{budget}": float(row[2]),
        f"candidate_hit_rate@{budget}": float(row[3]),
        "oracle_map@12": float(row[4]), "map@12": float(row[5]),
    }


def _evaluate_table(connection: duckdb.DuckDBPyConnection, table: str, budget: int) -> dict[str, Any]:
    segments = {
        "overall": _candidate_metrics(connection, table, budget),
        "warm": _candidate_metrics(connection, table, budget, "item_temperature='warm'"),
        "cold": _candidate_metrics(connection, table, budget, "item_temperature='cold'"),
    }
    popularity = {
        label: _candidate_metrics(
            connection,
            table,
            budget,
            f"popularity_segment='{label}'",
        )
        for label in (
            "head_top20pct_items",
            "middle_next40pct_items",
            "tail_bottom40pct_items",
            "cold_or_unseen",
        )
    }
    activity: dict[str, Any] = {}
    for label in ("inactive_12w", "low_1_5", "medium_6_20", "high_21_plus"):
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW activity_truth AS SELECT t.* FROM eval_truth t JOIN user_activity a USING (customer_id) WHERE a.activity_segment='{label}'"
        )
        activity[label] = _candidate_metrics(
            connection, table, budget, truth_table="activity_truth"
        )
        connection.execute("DROP VIEW activity_truth")
    return {
        "budget": budget,
        "segments": segments,
        "activity_segments": activity,
        "popularity_segments": popularity,
    }


def run_m25_retrieval(
    *,
    neighbor_metrics_path: Path,
    transactions_path: Path,
    reference_metrics_path: Path,
    windows: list[tuple[str, Path, Path]],
    output_dir: Path,
    report_dir: Path,
    history_weeks: int = 12,
    seed_k: int = 5,
    image_candidate_k: int = 300,
    rrf_constant: int = 60,
    image_weight: float = 1.0,
) -> dict[str, Any]:
    if len(windows) != 2:
        raise ValueError("M2.5 primary run requires exactly two development windows")
    output_dir = output_dir.resolve()
    report_dir = report_dir.resolve()
    if output_dir.exists() or report_dir.exists():
        raise FileExistsError("refusing to overwrite M2.5 retrieval run")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        neighbor_metrics, indices, scores, items = _load_neighbors(neighbor_metrics_path)
        reference = json.loads(reference_metrics_path.read_text(encoding="utf-8"))
        article_ids = items.sort_values("row_index")["article_id"].to_numpy()
        transactions_path = transactions_path.resolve()
        results: dict[str, Any] = {}
        for cutoff, candidate_path, manifest_path in windows:
            identity = validate_candidate_artifact(candidate_path, manifest_path, 100)
            if identity["cutoff"] != cutoff or identity["sample_rate"] != 0.1:
                raise ValueError("window identity differs from frozen M2.5 contract")
            window_dir = output_dir / cutoff
            window_dir.mkdir()
            database_path = window_dir / "evaluation.duckdb"
            con = duckdb.connect(str(database_path))
            con.execute("SET threads=8")
            con.register("image_items_frame", items)
            candidate_sql = _literal(candidate_path)
            transaction_sql = _literal(transactions_path)
            cutoff_sql = _date(cutoff)
            con.execute(f"CREATE TABLE baseline AS SELECT customer_id,article_id,candidate_rank,fused_score FROM read_parquet({candidate_sql})")
            con.execute("CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM baseline")
            con.execute("CREATE TABLE image_items AS SELECT * FROM image_items_frame")
            con.unregister("image_items_frame")
            con.execute(
                f"""
                CREATE TABLE user_activity AS
                WITH counts AS (
                    SELECT u.customer_id, count(t.article_id) AS events
                    FROM eval_users u LEFT JOIN read_parquet({transaction_sql}) t
                    ON u.customer_id=t.customer_id AND t.t_dat >= {cutoff_sql}-INTERVAL {history_weeks} WEEK AND t.t_dat < {cutoff_sql}
                    GROUP BY u.customer_id
                )
                SELECT *, CASE WHEN events=0 THEN 'inactive_12w' WHEN events<=5 THEN 'low_1_5'
                    WHEN events<=20 THEN 'medium_6_20' ELSE 'high_21_plus' END AS activity_segment FROM counts;
                CREATE TABLE seeds AS
                WITH latest AS (
                    SELECT t.customer_id,t.article_id,max(t.t_dat) AS latest_t_dat
                    FROM read_parquet({transaction_sql}) t JOIN eval_users u USING (customer_id)
                    JOIN image_items i USING (article_id)
                    WHERE t.t_dat >= {cutoff_sql}-INTERVAL {history_weeks} WEEK AND t.t_dat < {cutoff_sql}
                    GROUP BY t.customer_id,t.article_id
                ), ranked AS (
                    SELECT latest.*,i.row_index,row_number() OVER (PARTITION BY customer_id ORDER BY latest_t_dat DESC,article_id) AS seed_rank
                    FROM latest JOIN image_items i USING (article_id)
                ) SELECT * FROM ranked WHERE seed_rank<={seed_k};
                CREATE TABLE warm_catalog AS SELECT DISTINCT article_id FROM read_parquet({transaction_sql}) WHERE t_dat < {cutoff_sql};
                CREATE TABLE item_popularity AS
                WITH counts AS (
                    SELECT article_id,count(*) AS events
                    FROM read_parquet({transaction_sql})
                    WHERE t_dat >= {cutoff_sql}-INTERVAL {history_weeks} WEEK AND t_dat < {cutoff_sql}
                    GROUP BY article_id
                ), bucketed AS (
                    SELECT *,ntile(5) OVER(ORDER BY events DESC,article_id) AS bucket FROM counts
                ) SELECT article_id,CASE WHEN bucket=1 THEN 'head_top20pct_items'
                    WHEN bucket<=3 THEN 'middle_next40pct_items'
                    ELSE 'tail_bottom40pct_items' END AS popularity_segment FROM bucketed;
                CREATE TABLE eval_truth AS
                SELECT q.customer_id,q.article_id,CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature,
                       coalesce(p.popularity_segment,'cold_or_unseen') AS popularity_segment
                FROM (SELECT DISTINCT t.customer_id,t.article_id FROM read_parquet({transaction_sql}) t JOIN eval_users u USING(customer_id)
                      WHERE t.t_dat >= {cutoff_sql} AND t.t_dat < {cutoff_sql}+INTERVAL 7 DAY) q
                LEFT JOIN warm_catalog w USING(article_id)
                LEFT JOIN item_popularity p USING(article_id);
                """
            )
            seeds = con.execute("SELECT customer_id,row_index,seed_rank FROM seeds ORDER BY customer_id,seed_rank").fetchdf()
            source_stats = _build_image_candidates(con, seeds, indices, scores, article_ids, image_candidate_k)
            image_parquet = window_dir / "image_candidates.parquet"
            con.execute(f"COPY image_candidates TO {_literal(image_parquet)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
            con.execute(
                f"""
                CREATE TABLE image_100 AS SELECT customer_id,article_id,image_rank AS candidate_rank FROM image_candidates WHERE image_rank<=100;
                CREATE TABLE image_300 AS SELECT customer_id,article_id,image_rank AS candidate_rank FROM image_candidates WHERE image_rank<=300;
                CREATE TABLE fixed_100 AS WITH unioned AS (
                    SELECT coalesce(b.customer_id,i.customer_id) customer_id,coalesce(b.article_id,i.article_id) article_id,
                           coalesce(b.fused_score,0)+coalesce({image_weight}/({rrf_constant}+i.image_rank),0) score,
                           b.candidate_rank base_rank,i.image_rank
                    FROM baseline b FULL OUTER JOIN image_candidates i USING(customer_id,article_id)
                ) SELECT customer_id,article_id,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,base_rank NULLS LAST,image_rank NULLS LAST,article_id) candidate_rank
                  FROM unioned QUALIFY candidate_rank<=100;
                CREATE TABLE expanded_200 AS SELECT customer_id,article_id,candidate_rank FROM baseline UNION ALL
                    SELECT customer_id,article_id,100+row_number() OVER(PARTITION BY customer_id ORDER BY image_rank,article_id)
                    FROM image_candidates i WHERE NOT EXISTS(SELECT 1 FROM baseline b WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
                    QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY image_rank,article_id)<=100;
                CREATE TABLE expanded_300 AS SELECT customer_id,article_id,candidate_rank FROM baseline UNION ALL
                    SELECT customer_id,article_id,100+row_number() OVER(PARTITION BY customer_id ORDER BY image_rank,article_id)
                    FROM image_candidates i WHERE NOT EXISTS(SELECT 1 FROM baseline b WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
                    QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY image_rank,article_id)<=200;
                """
            )
            evaluations = {
                "six_source_100": _evaluate_table(con, "baseline", 100),
                "image_standalone_100": _evaluate_table(con, "image_100", 100),
                "image_standalone_300": _evaluate_table(con, "image_300", 300),
                "fixed_100": _evaluate_table(con, "fixed_100", 100),
                "expanded_200": _evaluate_table(con, "expanded_200", 200),
                "expanded_300": _evaluate_table(con, "expanded_300", 300),
            }
            overlap = con.execute("SELECT count(*) image_rows,count(*) FILTER(WHERE b.article_id IS NOT NULL) overlap_rows FROM image_candidates i LEFT JOIN baseline b USING(customer_id,article_id)").fetchone()
            marginal_truth = con.execute("SELECT count(*) FROM eval_truth t JOIN image_candidates i USING(customer_id,article_id) LEFT JOIN baseline b USING(customer_id,article_id) WHERE b.article_id IS NULL").fetchone()[0]
            seed_summary = con.execute("SELECT count(*) users_with_seed,avg(seed_count),min(seed_count),max(seed_count) FROM (SELECT customer_id,count(*) seed_count FROM seeds GROUP BY customer_id)").fetchone()
            ref_key = "dev_a" if cutoff == "2020-07-22" else "dev_b"
            expected = reference["development"][ref_key]["evaluation"]["orderings"]["rrf"]["segments"]
            parity: dict[str, float] = {}
            for segment in ("overall", "warm", "cold"):
                actual = evaluations["six_source_100"]["segments"][segment]
                for metric in ("candidate_recall@100", "candidate_hit_rate@100", "oracle_map@12", "map@12"):
                    difference = abs(float(actual[metric])-float(expected[segment][metric]))
                    parity[f"{segment}.{metric}"] = difference
                    if difference > 1e-12:
                        raise RuntimeError(f"baseline parity failed: {cutoff} {segment} {metric} {difference}")
            results[cutoff] = {
                "candidate_identity": identity,
                "seed_protocol": {"history_weeks": history_weeks,"distinct_latest_image_seeds": seed_k,"recency_discount": "1/(1+0.1*(seed_rank-1))","neighbors_per_seed": int(indices.shape[1]),"image_candidate_k": image_candidate_k},
                "seed_summary": {"eval_users": int(con.execute('select count(*) from eval_users').fetchone()[0]),"users_with_seed": int(seed_summary[0]),"users_without_seed": int(con.execute('select count(*) from eval_users u where not exists(select 1 from seeds s where s.customer_id=u.customer_id)').fetchone()[0]),"avg_seed_count_among_seeded": float(seed_summary[1]),"min_seed_count": int(seed_summary[2]),"max_seed_count": int(seed_summary[3])},
                "source": {**source_stats,"overlap_rows": int(overlap[1]),"new_rows": int(overlap[0]-overlap[1]),"overlap_rate": float(overlap[1]/overlap[0]),"marginal_truth_pairs": int(marginal_truth),"artifact": str(image_parquet),"artifact_bytes": image_parquet.stat().st_size,"artifact_sha256": _sha256(image_parquet)},
                "baseline_parity_absolute_differences": parity,
                "evaluations": evaluations,
            }
            _atomic_json(window_dir / "metrics.json", results[cutoff])
            con.close()

        gate_windows = {}
        for cutoff, result in results.items():
            base = result["evaluations"]["six_source_100"]["segments"]["overall"]
            expanded = result["evaluations"]["expanded_300"]["segments"]["overall"]
            gate_windows[cutoff] = {"marginal_truth_positive": result["source"]["marginal_truth_pairs"] > 0,"expanded_oracle_non_decreasing": expanded["oracle_map@12"] >= base["oracle_map@12"],"oracle_delta": expanded["oracle_map@12"]-base["oracle_map@12"]}
        metrics = {"schema_version":"m2.5-image-retrieval-dev-v1","status":"completed","boundary":"10% hash-sampled development users; full catalog and cutoff-safe full history statistics","neighbor_metrics_path":str(neighbor_metrics_path.resolve()),"neighbor_metrics_sha256":_sha256(neighbor_metrics_path.resolve()),"reference_metrics_sha256":_sha256(reference_metrics_path.resolve()),"config":{"history_weeks":history_weeks,"seed_k":seed_k,"image_candidate_k":image_candidate_k,"rrf_constant":rrf_constant,"image_weight":image_weight,"catalog_protocol":"optimistic_all_articles"},"windows":results,"development_gate":{"windows":gate_windows,"passed":all(x["marginal_truth_positive"] and x["expanded_oracle_non_decreasing"] for x in gate_windows.values())},"elapsed_seconds":time.perf_counter()-started}
        _atomic_json(output_dir / "metrics.json", metrics)
        report_dir.mkdir(parents=True)
        _atomic_json(report_dir / "metrics.json", metrics)
        return metrics
    except Exception as error:
        _atomic_json(output_dir / f"failure-{time.time_ns()}.json", {"schema_version":"m2.5-image-retrieval-failure-v1","status":"failed","error_type":type(error).__name__,"error":str(error),"elapsed_seconds":time.perf_counter()-started})
        raise
