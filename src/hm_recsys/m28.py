from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd

from .m2 import validate_candidate_artifact
from .m25 import _stable_topk
from .m25_retrieval import _atomic_json, _evaluate_table, _sha256


def _literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _date(value: str) -> str:
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise ValueError(f"invalid cutoff date: {value}")
    return f"DATE '{value}'"


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


class ParquetSequenceCorpus:
    """Replayable corpus backed by one row/list per customer."""

    def __init__(self, path: Path, batch_size: int = 4096) -> None:
        self.path = path.resolve()
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[str]]:
        connection = duckdb.connect()
        try:
            cursor = connection.execute(
                f"SELECT items FROM read_parquet({_literal(self.path)}) ORDER BY customer_id"
            )
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break
                for (items,) in rows:
                    if items:
                        yield [str(item) for item in items]
        finally:
            connection.close()


def _materialize_sequences(
    *, transactions_path: Path, cutoff: str, history_weeks: int,
    output_path: Path, ordering_seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    cutoff_sql = _date(cutoff)
    transaction_sql = _literal(transactions_path)
    connection.execute(
        f"""
        COPY (
            WITH deduplicated AS (
                SELECT DISTINCT customer_id, t_dat, article_id
                FROM read_parquet({transaction_sql})
                WHERE t_dat >= {cutoff_sql} - INTERVAL {history_weeks} WEEK
                  AND t_dat < {cutoff_sql}
            )
            SELECT customer_id,
                   list(article_id ORDER BY t_dat,
                        hash(article_id || ':{ordering_seed}'), article_id) AS items
            FROM deduplicated GROUP BY customer_id ORDER BY customer_id
        ) TO {_literal(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    row = connection.execute(
        f"""
        WITH raw AS (
            SELECT customer_id,t_dat,article_id FROM read_parquet({transaction_sql})
            WHERE t_dat >= {cutoff_sql}-INTERVAL {history_weeks} WEEK AND t_dat < {cutoff_sql}
        ), deduplicated AS (
            SELECT DISTINCT customer_id,t_dat,article_id FROM raw
        ), lengths AS (
            SELECT customer_id,count(*) AS sequence_length FROM deduplicated GROUP BY customer_id
        )
        SELECT (SELECT count(*) FROM raw),(SELECT count(*) FROM deduplicated),
               (SELECT count(*) FROM lengths),(SELECT count(DISTINCT article_id) FROM deduplicated),
               (SELECT avg(sequence_length) FROM lengths),
               (SELECT quantile_cont(sequence_length,0.95) FROM lengths),
               (SELECT max(sequence_length) FROM lengths)
        """
    ).fetchone()
    connection.close()
    return {
        "protocol": "12w_customer_pseudo_sequence_distinct_user_day_item",
        "within_day_order": f"duckdb_hash(article_id || ':{ordering_seed}') then article_id",
        "raw_rows": int(row[0]), "unique_user_day_item_rows": int(row[1]),
        "customer_sequences": int(row[2]), "distinct_items": int(row[3]),
        "mean_sequence_length": float(row[4]), "p95_sequence_length": float(row[5]),
        "max_sequence_length": int(row[6]), "artifact": _file_identity(output_path),
        "elapsed_seconds": time.perf_counter()-started,
    }


def _train_item2vec(
    *, sequence_path: Path, output_dir: Path, vector_size: int, window: int,
    min_count: int, negative: int, epochs: int, seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    from gensim import __version__ as gensim_version
    from gensim.models import Word2Vec

    started = time.perf_counter()
    corpus = ParquetSequenceCorpus(sequence_path)
    model = Word2Vec(
        vector_size=vector_size, window=window, min_count=min_count, workers=1,
        sg=1, negative=negative, hs=0, sample=1e-3, seed=seed, sorted_vocab=1,
        hashfxn=_stable_hash,
    )
    model.build_vocab(corpus)
    if len(model.wv) <= 100:
        raise RuntimeError("Item2Vec vocabulary unexpectedly small")
    model.train(corpus, total_examples=model.corpus_count, epochs=epochs, compute_loss=True)
    if model.corpus_count <= 0 or model.corpus_total_words <= 0:
        raise RuntimeError("Item2Vec corpus evidence is empty")
    keyed_vectors_path = output_dir/"item2vec.kv"
    model.wv.save(str(keyed_vectors_path), separately=[])
    items = pd.DataFrame({
        "row_index": np.arange(len(model.wv), dtype=np.int32),
        "article_id": [str(key) for key in model.wv.index_to_key],
        "token_count": [int(model.wv.get_vecattr(key,"count")) for key in model.wv.index_to_key],
    })
    if items["article_id"].duplicated().any():
        raise RuntimeError("Item2Vec vocabulary contains duplicate items")
    items_path=output_dir/"items.csv"
    items.to_csv(items_path,index=False,lineterminator="\n")
    vectors=np.asarray(model.wv.get_normed_vectors(),dtype=np.float32)
    if vectors.shape!=(len(items),vector_size) or not np.isfinite(vectors).all():
        raise RuntimeError("Item2Vec normalized vectors failed shape/finite gate")
    vectors_path=output_dir/"normalized_vectors.float32.npy"
    np.save(vectors_path,vectors,allow_pickle=False)
    norms=np.linalg.norm(vectors,axis=1)
    metrics={
        "library":"gensim.Word2Vec","gensim_version":gensim_version,
        "objective":"skip_gram_negative_sampling",
        "parameters":{"vector_size":vector_size,"window":window,"min_count":min_count,
            "workers":1,"sg":1,"negative":negative,"hs":0,"sample":1e-3,
            "epochs":epochs,"seed":seed,"sorted_vocab":1,
            "hashfxn":"sha256_first_32_bits_little_endian"},
        "corpus_count":int(model.corpus_count),"corpus_total_words":int(model.corpus_total_words),
        "vocabulary_items":len(items),"retained_token_occurrences":int(items["token_count"].sum()),
        "latest_training_loss":float(model.get_latest_training_loss()),
        "vector_norm_min":float(norms.min()),"vector_norm_mean":float(norms.mean()),
        "vector_norm_max":float(norms.max()),"keyed_vectors":_file_identity(keyed_vectors_path),
        "items":_file_identity(items_path),"normalized_vectors":_file_identity(vectors_path),
        "elapsed_seconds":time.perf_counter()-started,
    }
    return metrics,items,vectors


def _build_exact_neighbors(
    *, vectors: np.ndarray, output_dir: Path, top_k: int, tie_buffer: int,
    query_batch_size: int, device: str,
) -> tuple[dict[str, Any],np.ndarray,np.ndarray]:
    import torch
    rows=len(vectors)
    if min(top_k,tie_buffer,query_batch_size)<1 or top_k+tie_buffer>=rows:
        raise ValueError("invalid exact-neighbor parameters")
    if device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started=time.perf_counter()
    corpus=torch.from_numpy(vectors).to(device)
    if device=="cuda": torch.cuda.reset_peak_memory_stats()
    output_indices=np.empty((rows,top_k),dtype=np.int32)
    output_scores=np.empty((rows,top_k),dtype=np.float32)
    ambiguous_rows=[]; search_seconds=0.0; retrieve_k=top_k+tie_buffer
    for start in range(0,rows,query_batch_size):
        stop=min(start+query_batch_size,rows)
        if device=="cuda": torch.cuda.synchronize()
        batch_started=time.perf_counter()
        similarities=corpus[start:stop]@corpus.T
        similarities[torch.arange(stop-start,device=device),torch.arange(start,stop,device=device)]=-torch.inf
        values,indices=torch.topk(similarities,k=retrieve_k,dim=1,largest=True,sorted=False)
        if device=="cuda": torch.cuda.synchronize()
        search_seconds+=time.perf_counter()-batch_started
        values_np=values.detach().cpu().numpy().astype(np.float32,copy=False)
        indices_np=indices.detach().cpu().numpy().astype(np.int32,copy=False)
        for offset in range(stop-start):
            stable_indices,stable_scores,ambiguous=_stable_topk(indices_np[offset],values_np[offset],top_k)
            row_index=start+offset
            if ambiguous: ambiguous_rows.append(row_index)
            output_indices[row_index]=stable_indices; output_scores[row_index]=stable_scores
    del corpus
    indices_path=output_dir/"neighbor_indices.int32.npy"
    scores_path=output_dir/"neighbor_scores.float32.npy"
    np.save(indices_path,output_indices,allow_pickle=False); np.save(scores_path,output_scores,allow_pickle=False)
    self_neighbors=int(np.count_nonzero(output_indices==np.arange(rows,dtype=np.int32)[:,None]))
    duplicate_rows=int(sum(len(set(row.tolist()))!=top_k for row in output_indices))
    if self_neighbors or duplicate_rows or ambiguous_rows:
        raise RuntimeError(f"exact-neighbor audit failed: self={self_neighbors}, duplicate_rows={duplicate_rows}, ambiguous={len(ambiguous_rows)}")
    metrics={"protocol":"full_vocabulary_exact_cosine","query_rows":rows,"corpus_rows":rows,
        "top_k":top_k,"tie_buffer":tie_buffer,"query_batch_size":query_batch_size,
        "tie_order":"cosine_desc_then_neighbor_row_index_asc","device":device,
        "device_name":torch.cuda.get_device_name(0) if device=="cuda" else "cpu",
        "torch_version":torch.__version__,"self_neighbors":self_neighbors,
        "duplicate_neighbor_rows":duplicate_rows,"ambiguous_boundaries":len(ambiguous_rows),
        "score_min":float(output_scores.min()),"score_mean":float(output_scores.mean()),
        "score_max":float(output_scores.max()),"indices":_file_identity(indices_path),
        "scores":_file_identity(scores_path),"search_seconds":search_seconds,
        "elapsed_seconds":time.perf_counter()-started,
        "peak_cuda_allocated_bytes":int(torch.cuda.max_memory_allocated()) if device=="cuda" else 0}
    return metrics,output_indices,output_scores



def _aggregate_item2vec_candidates(
    *, seeds: pd.DataFrame, neighbor_indices: np.ndarray, neighbor_scores: np.ndarray,
    article_ids: np.ndarray, max_candidates: int,
) -> pd.DataFrame:
    rows:list[tuple[Any,...]]=[]
    for customer_id,group in seeds.groupby("customer_id",sort=True):
        candidates:dict[int,list[float|int]]={}
        for seed in group.sort_values(["seed_rank","row_index"]).itertuples(index=False):
            seed_row=int(seed.row_index); seed_rank=int(seed.seed_rank)
            discount=1.0/(1.0+0.1*(seed_rank-1))
            for neighbor_rank,(candidate_row,cosine) in enumerate(
                zip(neighbor_indices[seed_row],neighbor_scores[seed_row]),start=1
            ):
                candidate_row=int(candidate_row); raw=float(cosine); weighted=raw*discount
                prior=candidates.get(candidate_row)
                if prior is None:
                    candidates[candidate_row]=[weighted,raw,seed_rank,neighbor_rank,1]
                else:
                    prior[4]=int(prior[4])+1
                    current_key=(weighted,raw,-seed_rank,-neighbor_rank)
                    prior_key=(float(prior[0]),float(prior[1]),-int(prior[2]),-int(prior[3]))
                    if current_key>prior_key: prior[0:4]=[weighted,raw,seed_rank,neighbor_rank]
        ranked=sorted(candidates.items(),key=lambda pair:(-float(pair[1][0]),-float(pair[1][1]),
            int(pair[1][2]),int(pair[1][3]),str(article_ids[pair[0]])))[:max_candidates]
        for rank,(candidate_row,evidence) in enumerate(ranked,start=1):
            rows.append((str(customer_id),str(article_ids[candidate_row]),rank,float(evidence[0]),
                float(evidence[1]),int(evidence[2]),int(evidence[3]),int(evidence[4])))
    return pd.DataFrame(rows,columns=["customer_id","article_id","item2vec_rank",
        "item2vec_score","item2vec_cosine","best_seed_rank","best_neighbor_rank","seed_support"])


def _insert_item2vec_candidates(
    connection: duckdb.DuckDBPyConnection, *, seeds: pd.DataFrame,
    neighbor_indices: np.ndarray, neighbor_scores: np.ndarray, article_ids: np.ndarray,
    max_candidates: int, user_chunk: int=1000,
) -> dict[str,Any]:
    connection.execute("""CREATE TABLE item2vec_candidates (
        customer_id VARCHAR,article_id VARCHAR,item2vec_rank INTEGER,item2vec_score DOUBLE,
        item2vec_cosine DOUBLE,best_seed_rank INTEGER,best_neighbor_rank INTEGER,seed_support INTEGER)""")
    users=sorted(seeds["customer_id"].unique().tolist()) if len(seeds) else []
    rows_written=0
    for start in range(0,len(users),user_chunk):
        selected=set(users[start:start+user_chunk])
        frame=_aggregate_item2vec_candidates(seeds=seeds[seeds["customer_id"].isin(selected)],
            neighbor_indices=neighbor_indices,neighbor_scores=neighbor_scores,
            article_ids=article_ids,max_candidates=max_candidates)
        if len(frame):
            connection.register("item2vec_chunk",frame)
            connection.execute("INSERT INTO item2vec_candidates SELECT * FROM item2vec_chunk")
            connection.unregister("item2vec_chunk"); rows_written+=len(frame)
        print(f"item2vec candidates users {min(start+user_chunk,len(users))}/{len(users)}, rows={rows_written}",flush=True)
    return {"users_with_candidates":len(users),"rows":rows_written}


def _load_image_reference(metrics_path: Path) -> tuple[dict[str,Any],dict[str,dict[str,Any]]]:
    metrics_path=metrics_path.resolve(); metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status")!="completed" or not metrics.get("development_gate",{}).get("passed"):
        raise ValueError("M2.8 requires passed M2.5 image retrieval evidence")
    windows={}
    for cutoff,window in metrics["windows"].items():
        source=window["source"]; artifact=Path(source["artifact"]).resolve(); identity=_file_identity(artifact)
        if identity["bytes"]!=int(source["artifact_bytes"]) or identity["sha256"]!=str(source["artifact_sha256"]):
            raise ValueError(f"M2.5 image artifact identity mismatch: {cutoff}")
        windows[cutoff]=identity
    return _file_identity(metrics_path),windows


def _prepare_evaluation_tables(
    connection: duckdb.DuckDBPyConnection, *, transactions_path: Path,
    cutoff: str, history_weeks: int,
) -> None:
    transaction_sql=_literal(transactions_path); cutoff_sql=_date(cutoff)
    connection.execute(f"""
        CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM baseline;
        CREATE TABLE user_activity AS WITH counts AS (
            SELECT u.customer_id,count(t.article_id) AS events
            FROM eval_users u LEFT JOIN read_parquet({transaction_sql}) t
              ON u.customer_id=t.customer_id AND t.t_dat>={cutoff_sql}-INTERVAL {history_weeks} WEEK
             AND t.t_dat<{cutoff_sql} GROUP BY u.customer_id
        ) SELECT *,CASE WHEN events=0 THEN 'inactive_12w' WHEN events<=5 THEN 'low_1_5'
            WHEN events<=20 THEN 'medium_6_20' ELSE 'high_21_plus' END AS activity_segment FROM counts;
        CREATE TABLE warm_catalog AS SELECT DISTINCT article_id FROM read_parquet({transaction_sql}) WHERE t_dat<{cutoff_sql};
        CREATE TABLE item_popularity AS WITH counts AS (
            SELECT article_id,count(*) AS events FROM read_parquet({transaction_sql})
            WHERE t_dat>={cutoff_sql}-INTERVAL {history_weeks} WEEK AND t_dat<{cutoff_sql} GROUP BY article_id
        ),bucketed AS (SELECT *,ntile(5) OVER(ORDER BY events DESC,article_id) AS bucket FROM counts)
        SELECT article_id,CASE WHEN bucket=1 THEN 'head_top20pct_items' WHEN bucket<=3 THEN 'middle_next40pct_items'
            ELSE 'tail_bottom40pct_items' END AS popularity_segment FROM bucketed;
        CREATE TABLE eval_truth AS SELECT q.customer_id,q.article_id,
            CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature,
            coalesce(p.popularity_segment,'cold_or_unseen') AS popularity_segment
        FROM (SELECT DISTINCT t.customer_id,t.article_id FROM read_parquet({transaction_sql}) t
            JOIN eval_users u USING(customer_id) WHERE t.t_dat>={cutoff_sql}
              AND t.t_dat<{cutoff_sql}+INTERVAL 7 DAY) q
        LEFT JOIN warm_catalog w USING(article_id) LEFT JOIN item_popularity p USING(article_id);
    """)


def _create_seeds(
    connection: duckdb.DuckDBPyConnection, *, transactions_path: Path,
    cutoff: str, history_weeks: int, seed_k: int,
) -> pd.DataFrame:
    transaction_sql=_literal(transactions_path); cutoff_sql=_date(cutoff)
    connection.execute(f"""CREATE TABLE seeds AS WITH latest AS (
        SELECT t.customer_id,t.article_id,max(t.t_dat) AS latest_t_dat
        FROM read_parquet({transaction_sql}) t JOIN eval_users u USING(customer_id)
        JOIN item2vec_vocab v USING(article_id)
        WHERE t.t_dat>={cutoff_sql}-INTERVAL {history_weeks} WEEK AND t.t_dat<{cutoff_sql}
        GROUP BY t.customer_id,t.article_id
    ),ranked AS (SELECT latest.*,v.row_index,row_number() OVER(
        PARTITION BY customer_id ORDER BY latest_t_dat DESC,article_id) AS seed_rank
        FROM latest JOIN item2vec_vocab v USING(article_id))
    SELECT * FROM ranked WHERE seed_rank<={seed_k}""")
    return connection.execute("SELECT customer_id,row_index,seed_rank FROM seeds ORDER BY customer_id,seed_rank").fetchdf()


def _create_candidate_pools(
    connection: duckdb.DuckDBPyConnection, *, rrf_constant: int, source_weight: float,
) -> None:
    connection.execute(f"""
        CREATE TABLE item2vec_100 AS SELECT customer_id,article_id,item2vec_rank AS candidate_rank
          FROM item2vec_candidates WHERE item2vec_rank<=100;
        CREATE TABLE item2vec_300 AS SELECT customer_id,article_id,item2vec_rank AS candidate_rank
          FROM item2vec_candidates WHERE item2vec_rank<=300;
        CREATE TABLE fixed_100 AS WITH unioned AS (
            SELECT coalesce(b.customer_id,i.customer_id) customer_id,
                   coalesce(b.article_id,i.article_id) article_id,
                   coalesce(b.fused_score,0)+coalesce({source_weight}/({rrf_constant}+i.item2vec_rank),0) score,
                   b.candidate_rank base_rank,i.item2vec_rank
            FROM baseline b FULL OUTER JOIN item2vec_candidates i USING(customer_id,article_id)
        ) SELECT customer_id,article_id,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,
            base_rank NULLS LAST,item2vec_rank NULLS LAST,article_id) candidate_rank
          FROM unioned QUALIFY candidate_rank<=100;
        CREATE TABLE expanded_200 AS SELECT customer_id,article_id,candidate_rank FROM baseline UNION ALL
          SELECT customer_id,article_id,100+row_number() OVER(PARTITION BY customer_id ORDER BY item2vec_rank,article_id)
          FROM item2vec_candidates i WHERE NOT EXISTS(SELECT 1 FROM baseline b
            WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
          QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY item2vec_rank,article_id)<=100;
        CREATE TABLE expanded_300 AS SELECT customer_id,article_id,candidate_rank FROM baseline UNION ALL
          SELECT customer_id,article_id,100+row_number() OVER(PARTITION BY customer_id ORDER BY item2vec_rank,article_id)
          FROM item2vec_candidates i WHERE NOT EXISTS(SELECT 1 FROM baseline b
            WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
          QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY item2vec_rank,article_id)<=200;
    """)


def _create_combined_image_pool(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("""
        CREATE TABLE image_append AS SELECT customer_id,article_id,
          row_number() OVER(PARTITION BY customer_id ORDER BY image_rank,article_id) AS append_rank
        FROM image_candidates i WHERE NOT EXISTS(SELECT 1 FROM baseline b
          WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id) QUALIFY append_rank<=200;
        CREATE TABLE item2vec_after_image AS SELECT customer_id,article_id,
          row_number() OVER(PARTITION BY customer_id ORDER BY item2vec_rank,article_id) AS append_rank
        FROM item2vec_candidates i WHERE NOT EXISTS(SELECT 1 FROM baseline b
          WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
          AND NOT EXISTS(SELECT 1 FROM image_append x
          WHERE x.customer_id=i.customer_id AND x.article_id=i.article_id) QUALIFY append_rank<=200;
        CREATE TABLE combined_image_item2vec_500 AS
          SELECT customer_id,article_id,candidate_rank FROM baseline UNION ALL
          SELECT customer_id,article_id,100+append_rank FROM image_append UNION ALL
          SELECT customer_id,article_id,300+append_rank FROM item2vec_after_image;
    """)



def _render_report(metrics: dict[str,Any]) -> str:
    lines=["# M2.8 cutoff-safe Item2Vec retrieval","",
        f"- status: {metrics['status']}",
        f"- development gate: {'passed' if metrics['development_gate']['passed'] else 'failed'}",
        "- final week: not run",
        "- boundary: 10% development users; full cutoff-safe 12-week representation history","",
        "## Overall candidate metrics","",
        "| cutoff | pool | Recall@K | HitRate@K | Oracle MAP@12 | MAP@12 |",
        "|---|---|---:|---:|---:|---:|"]
    pools=(("six_source_100",100),("item2vec_100",100),("item2vec_300",300),
        ("fixed_100",100),("expanded_200",200),("expanded_300",300),
        ("combined_image_item2vec_500",500))
    for cutoff,result in metrics["windows"].items():
        for pool,budget in pools:
            overall=result["evaluations"][pool]["segments"]["overall"]
            lines.append(f"| {cutoff} | {pool} | {overall[f'candidate_recall@{budget}']:.6f} | "
                f"{overall[f'candidate_hit_rate@{budget}']:.6f} | {overall['oracle_map@12']:.6f} | "
                f"{overall['map@12']:.6f} |")
    lines.extend(["","## Source evidence","",
        "| cutoff | vocab | seeded users | source rows | baseline overlap | image overlap | marginal truth vs baseline | marginal truth vs baseline+image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for cutoff,result in metrics["windows"].items():
        source=result["source"]
        lines.append(f"| {cutoff} | {result['model']['vocabulary_items']:,} | "
            f"{result['seed_summary']['users_with_seed']:,} | {source['rows']:,} | "
            f"{source['baseline_overlap_rows']:,} | {source['image_overlap_rows']:,} | "
            f"{source['marginal_truth_pairs']:,} | {source['marginal_truth_pairs_beyond_image']:,} |")
    lines.extend(["","## Evidence boundary","",
        "- `t_dat` is a date, not an order/session sequence. Same-day order is deterministic hash order.",
        "- Distinct user-day-item is only a context-construction rule; raw transactions remain unchanged.",
        "- Missing purchases are unobserved, not proven negatives; optimistic all-articles catalog remains.",
        "- fixed-100 RRF is an untuned diagnostic and is not part of the retrieval promotion gate.",
        "- `combined_image_item2vec_500` is an Oracle/coverage diagnostic, not a deployable ordering.",
        "- No final-week interactions, full-user evaluation, ranking claim, or Kaggle leaderboard claim.",""])
    return "\n".join(lines)


def run_m28(
    *, transactions_path: Path, reference_metrics_path: Path, image_metrics_path: Path,
    windows: list[tuple[str,Path,Path]], output_dir: Path, report_dir: Path,
    history_weeks: int=12, vector_size: int=64, context_window: int=10,
    min_count: int=2, negative: int=10, epochs: int=5, seed: int=20260828,
    seed_k: int=5, neighbor_k: int=100, candidate_k: int=300,
    tie_buffer: int=32, query_batch_size: int=512, device: str="cuda",
    rrf_constant: int=60, source_weight: float=1.0,
) -> dict[str,Any]:
    if len(windows)!=2: raise ValueError("M2.8 primary run requires exactly two development windows")
    if candidate_k<200 or neighbor_k<100 or seed_k<1:
        raise ValueError("M2.8 candidate/neighbor/seed budgets violate frozen protocol")
    transactions_path=transactions_path.expanduser().resolve()
    reference_metrics_path=reference_metrics_path.expanduser().resolve()
    image_metrics_path=image_metrics_path.expanduser().resolve()
    output_dir=output_dir.expanduser().resolve(); report_dir=report_dir.expanduser().resolve()
    if output_dir.exists() or report_dir.exists(): raise FileExistsError("refusing to overwrite M2.8 run")
    output_dir.mkdir(parents=True); started=time.perf_counter()
    try:
        transaction_identity=_file_identity(transactions_path)
        reference_identity=_file_identity(reference_metrics_path)
        reference=json.loads(reference_metrics_path.read_text(encoding="utf-8"))
        image_identity,image_windows=_load_image_reference(image_metrics_path)
        results={}
        for cutoff,candidate_path,manifest_path in windows:
            window_started=time.perf_counter()
            identity=validate_candidate_artifact(candidate_path,manifest_path,100)
            if identity["cutoff"]!=cutoff or identity["sample_rate"]!=0.1:
                raise ValueError("window identity differs from frozen M2.8 contract")
            if cutoff not in image_windows: raise ValueError(f"missing M2.5 image window: {cutoff}")
            window_dir=output_dir/cutoff; window_dir.mkdir()
            sequences_path=window_dir/"customer_sequences.parquet"
            sequence_metrics=_materialize_sequences(transactions_path=transactions_path,cutoff=cutoff,
                history_weeks=history_weeks,output_path=sequences_path,ordering_seed=seed)
            model_metrics,items,vectors=_train_item2vec(sequence_path=sequences_path,
                output_dir=window_dir,vector_size=vector_size,window=context_window,
                min_count=min_count,negative=negative,epochs=epochs,seed=seed)
            model_metrics["retained_unique_context_rate"]=model_metrics["retained_token_occurrences"]/sequence_metrics["unique_user_day_item_rows"]
            neighbor_metrics,neighbor_indices,neighbor_scores=_build_exact_neighbors(
                vectors=vectors,output_dir=window_dir,top_k=neighbor_k,tie_buffer=tie_buffer,
                query_batch_size=query_batch_size,device=device)
            database_path=window_dir/"evaluation.duckdb"; connection=duckdb.connect(str(database_path))
            connection.execute("SET threads=8")
            connection.execute(f"""CREATE TABLE baseline AS SELECT customer_id,article_id,
                candidate_rank,fused_score,user_day_covisit_present
                FROM read_parquet({_literal(candidate_path)})""")
            connection.register("item2vec_items_frame",items)
            connection.execute("CREATE TABLE item2vec_vocab AS SELECT row_index,article_id,token_count FROM item2vec_items_frame")
            connection.unregister("item2vec_items_frame")
            _prepare_evaluation_tables(connection,transactions_path=transactions_path,
                cutoff=cutoff,history_weeks=history_weeks)
            seeds=_create_seeds(connection,transactions_path=transactions_path,cutoff=cutoff,
                history_weeks=history_weeks,seed_k=seed_k)
            article_ids=items.sort_values("row_index")["article_id"].to_numpy()
            source_stats=_insert_item2vec_candidates(connection,seeds=seeds,
                neighbor_indices=neighbor_indices,neighbor_scores=neighbor_scores,
                article_ids=article_ids,max_candidates=candidate_k)
            candidates_path=window_dir/"item2vec_candidates.parquet"
            connection.execute(f"COPY item2vec_candidates TO {_literal(candidates_path)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
            _create_candidate_pools(connection,rrf_constant=rrf_constant,source_weight=source_weight)
            image_path=Path(image_windows[cutoff]["path"])
            connection.execute(f"CREATE TABLE image_candidates AS SELECT * FROM read_parquet({_literal(image_path)})")
            _create_combined_image_pool(connection)
            evaluations={
                "six_source_100":_evaluate_table(connection,"baseline",100),
                "item2vec_100":_evaluate_table(connection,"item2vec_100",100),
                "item2vec_300":_evaluate_table(connection,"item2vec_300",300),
                "fixed_100":_evaluate_table(connection,"fixed_100",100),
                "expanded_200":_evaluate_table(connection,"expanded_200",200),
                "expanded_300":_evaluate_table(connection,"expanded_300",300),
                "combined_image_item2vec_500":_evaluate_table(connection,"combined_image_item2vec_500",500),
            }
            overlap=connection.execute("""SELECT count(*) AS source_rows,
                count(*) FILTER(WHERE b.article_id IS NOT NULL) AS baseline_overlap,
                count(*) FILTER(WHERE b.user_day_covisit_present=1) AS retained_covisit_overlap,
                count(*) FILTER(WHERE x.article_id IS NOT NULL) AS image_overlap
                FROM item2vec_candidates i LEFT JOIN baseline b USING(customer_id,article_id)
                LEFT JOIN image_candidates x USING(customer_id,article_id)""").fetchone()
            marginal_truth=connection.execute("""SELECT count(*) FROM eval_truth t
                JOIN item2vec_candidates i USING(customer_id,article_id)
                LEFT JOIN baseline b USING(customer_id,article_id) WHERE b.article_id IS NULL""").fetchone()[0]
            marginal_beyond_image=connection.execute("""SELECT count(*) FROM eval_truth t
                JOIN item2vec_candidates i USING(customer_id,article_id)
                LEFT JOIN baseline b USING(customer_id,article_id)
                LEFT JOIN image_candidates x USING(customer_id,article_id)
                WHERE b.article_id IS NULL AND x.article_id IS NULL""").fetchone()[0]


            seed_row=connection.execute("""SELECT count(*) users_with_seed,avg(seed_count),min(seed_count),max(seed_count)
                FROM (SELECT customer_id,count(*) seed_count FROM seeds GROUP BY customer_id)""").fetchone()
            eval_users=int(connection.execute("SELECT count(*) FROM eval_users").fetchone()[0])
            users_with_history=int(connection.execute("SELECT count(*) FROM user_activity WHERE events>0").fetchone()[0])
            truth_vocab=connection.execute("""SELECT count(*) truth_pairs,
                count(*) FILTER(WHERE v.article_id IS NOT NULL) in_vocab_pairs
                FROM eval_truth t LEFT JOIN item2vec_vocab v USING(article_id)""").fetchone()
            ref_key="dev_a" if cutoff=="2020-07-22" else "dev_b"
            expected=reference["development"][ref_key]["evaluation"]["orderings"]["rrf"]["segments"]
            parity={}
            for segment in ("overall","warm","cold"):
                actual=evaluations["six_source_100"]["segments"][segment]
                for metric in ("candidate_recall@100","candidate_hit_rate@100","oracle_map@12","map@12"):
                    difference=abs(float(actual[metric])-float(expected[segment][metric]))
                    parity[f"{segment}.{metric}"]=difference
                    if difference>1e-12:
                        raise RuntimeError(f"baseline parity failed: {cutoff} {segment} {metric} {difference}")
            results[cutoff]={
                "candidate_identity":identity,"sequence":sequence_metrics,"model":model_metrics,
                "neighbors":neighbor_metrics,"seed_protocol":{"history_weeks":history_weeks,
                    "latest_distinct_in_vocab_seeds":seed_k,
                    "recency_discount":"1/(1+0.1*(seed_rank-1))",
                    "neighbors_per_seed":neighbor_k,"candidate_k":candidate_k},
                "seed_summary":{"eval_users":eval_users,"users_with_history":users_with_history,
                    "users_with_seed":int(seed_row[0] or 0),
                    "users_without_seed":eval_users-int(seed_row[0] or 0),
                    "users_with_history_but_without_vocab_seed":users_with_history-int(seed_row[0] or 0),
                    "avg_seed_count_among_seeded":float(seed_row[1] or 0.0),
                    "min_seed_count":int(seed_row[2] or 0),"max_seed_count":int(seed_row[3] or 0),
                    "truth_pairs":int(truth_vocab[0]),"truth_pairs_in_vocabulary":int(truth_vocab[1]),
                    "truth_vocabulary_coverage":float(truth_vocab[1]/truth_vocab[0]) if truth_vocab[0] else 0.0},
                "source":{**source_stats,"baseline_overlap_rows":int(overlap[1]),
                    "retained_covisit_overlap_rows":int(overlap[2]),"image_overlap_rows":int(overlap[3]),
                    "new_rows_vs_baseline":int(overlap[0]-overlap[1]),
                    "baseline_overlap_rate":float(overlap[1]/overlap[0]),
                    "image_overlap_rate":float(overlap[3]/overlap[0]),
                    "marginal_truth_pairs":int(marginal_truth),
                    "marginal_truth_pairs_beyond_image":int(marginal_beyond_image),
                    "artifact":_file_identity(candidates_path)},
                "image_reference_artifact":image_windows[cutoff],
                "baseline_parity_absolute_differences":parity,"evaluations":evaluations,
                "elapsed_seconds":time.perf_counter()-window_started,
            }
            _atomic_json(window_dir/"metrics.json",results[cutoff]); connection.close()
        gate_windows={}
        for cutoff,result in results.items():
            base=result["evaluations"]["six_source_100"]["segments"]["overall"]
            expanded=result["evaluations"]["expanded_300"]["segments"]["overall"]
            gate_windows[cutoff]={"marginal_truth_positive":result["source"]["marginal_truth_pairs"]>0,
                "expanded_oracle_non_decreasing":expanded["oracle_map@12"]>=base["oracle_map@12"],
                "recall_delta":expanded["candidate_recall@300"]-base["candidate_recall@100"],
                "oracle_delta":expanded["oracle_map@12"]-base["oracle_map@12"],
                "marginal_truth_pairs_beyond_image":result["source"]["marginal_truth_pairs_beyond_image"]}
        metrics={"schema_version":"m2.8-item2vec-retrieval-dev-v1","stage":"M2.8",
            "status":"measured","run_id":output_dir.name,
            "created_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "boundary":"10% hash-sampled development users; full catalog and cutoff-safe full 12-week history statistics",
            "inputs":{"transactions":transaction_identity,"reference_metrics":reference_identity,
                "image_metrics":image_identity},
            "config":{"history_weeks":history_weeks,"vector_size":vector_size,
                "context_window":context_window,"min_count":min_count,"negative":negative,
                "epochs":epochs,"workers":1,"seed":seed,"seed_k":seed_k,
                "neighbor_k":neighbor_k,"candidate_k":candidate_k,"tie_buffer":tie_buffer,
                "query_batch_size":query_batch_size,"device":device,"rrf_constant":rrf_constant,
                "source_weight":source_weight,"catalog_protocol":"optimistic_all_articles",
                "fixed_100_role":"untuned_diagnostic_not_gate"},
            "windows":results,
            "development_gate":{"rule":"both windows have positive marginal truth coverage and non-decreasing expanded-300 Oracle MAP@12 versus frozen six-source",
                "windows":gate_windows,"passed":all(e["marginal_truth_positive"] and
                e["expanded_oracle_non_decreasing"] for e in gate_windows.values())},
            "final_week":"not_run","elapsed_seconds":time.perf_counter()-started}
        _atomic_json(output_dir/"metrics.json",metrics); report_dir.mkdir(parents=True)
        _atomic_json(report_dir/"metrics.json",metrics)
        (report_dir/"M2_8_REPORT.md").write_text(_render_report(metrics),encoding="utf-8",newline="\n")
        return metrics
    except Exception as error:
        _atomic_json(output_dir/f"failure-{time.time_ns()}.json",{
            "schema_version":"m2.8-item2vec-retrieval-failure-v1","status":"failed",
            "error_type":type(error).__name__,"error":str(error),
            "elapsed_seconds":time.perf_counter()-started})
        raise
