from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .audit import prepare_tabular_connection
from .m2 import (
    FULL_FEATURES, M2Config, RETRIEVAL_FEATURES, _create_static_dimensions,
    _literal, _prepare_frame, _sha256, _write_json, build_category_maps,
    build_point_in_time_dataset, validate_candidate_artifact,
)
from .m26 import (
    PROTOCOL, REQUIRED_CUTOFFS, _base_retrieval_projection,
    _empty_retrieval_projection, _evaluate_models, _frozen_reference,
    _load_training_data, _save_category_maps, _train_lambdarank,
    realized_oracle_fraction,
)
from .m21 import inactive_fallback_order_expression
from .m25_retrieval import _candidate_metrics
from .m28 import (
    _build_exact_neighbors, _create_seeds, _date, _file_identity,
    _insert_item2vec_candidates, _materialize_sequences, _train_item2vec,
)


SCHEMA_VERSION="m2.9-item2vec-ranker-development-v1"
SOURCE_SCHEMA_VERSION="m2.9-item2vec-source-v1"
CANDIDATE_SCHEMA_VERSION="m2.9-item2vec-expanded-candidates-v1"
FEATURE_SCHEMA_VERSION="m2.9-item2vec-features-v1"
ITEM2VEC_FEATURES=[
    "item2vec_present","item2vec_is_new","item2vec_rank","item2vec_score",
    "item2vec_cosine","item2vec_best_seed_rank","item2vec_best_neighbor_rank",
    "item2vec_seed_support","item2vec_vocab_count",
]
ALL_FEATURES=list(dict.fromkeys(FULL_FEATURES+ITEM2VEC_FEATURES))
FEATURE_SETS={
    "expanded_full_no_item2vec":FULL_FEATURES,
    "expanded_full_plus_item2vec":ALL_FEATURES,
}
ITEM2VEC_CONFIG={
    "history_weeks":12,"vector_size":64,"context_window":10,"min_count":2,
    "negative":10,"epochs":5,"seed":20260828,"seed_k":5,"neighbor_k":100,
    "candidate_k":300,"append_k":200,"tie_buffer":32,"query_batch_size":512,
    "workers":1,"device":"cuda",
}


@dataclass(frozen=True)
class M29Window:
    cutoff:str
    candidate_path:Path
    manifest_path:Path


def _read_json(path:Path)->dict[str,Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_file_identity(identity:dict[str,Any],label:str)->dict[str,Any]:
    path=Path(str(identity["path"])).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    actual={"path":str(path),"bytes":path.stat().st_size,"sha256":_sha256(path)}
    if actual["bytes"]!=int(identity["bytes"]) or actual["sha256"]!=str(identity["sha256"]):
        raise ValueError(f"artifact identity mismatch: {label}")
    return actual


def validate_group_sizes(sizes:list[int],minimum:int=100,maximum:int=300)->dict[str,int]:
    if not sizes: raise ValueError("ranking groups are empty")
    invalid=sum(size<minimum or size>maximum for size in sizes)
    if invalid: raise ValueError(f"invalid variable ranking groups: {invalid}")
    return {"groups":len(sizes),"rows":int(sum(sizes)),
        "min_group_rows":int(min(sizes)),"max_group_rows":int(max(sizes))}


def _source_manifest_valid(
    manifest_path:Path, baseline_identity:dict[str,Any], transaction_identity:dict[str,Any]
)->dict[str,Any]|None:
    if not manifest_path.is_file(): return None
    manifest=_read_json(manifest_path)
    if manifest.get("schema_version")!=SOURCE_SCHEMA_VERSION or manifest.get("status")!="completed":
        raise ValueError(f"invalid M2.9 source manifest: {manifest_path}")
    if manifest["inputs"]["baseline_candidate_sha256"]!=baseline_identity["candidate_sha256"]:
        raise ValueError(f"M2.9 source baseline drift: {manifest_path}")
    if manifest["inputs"]["transactions_sha256"]!=transaction_identity["sha256"]:
        raise ValueError(f"M2.9 source transaction drift: {manifest_path}")
    if manifest["contract"]!=ITEM2VEC_CONFIG:
        raise ValueError(f"M2.9 source config drift: {manifest_path}")
    for key,identity in manifest["artifacts"].items():
        _validate_file_identity(identity,f"{manifest_path}:{key}")
    return manifest


def _load_m28_reuse(
    metrics_path:Path, transaction_identity:dict[str,Any]
)->tuple[dict[str,Any],dict[str,dict[str,Any]]]:
    metrics_path=metrics_path.resolve(); metrics=_read_json(metrics_path)
    if metrics.get("schema_version")!="m2.8-item2vec-retrieval-dev-v1" or metrics.get("status")!="measured":
        raise ValueError("M2.9 requires measured M2.8 Item2Vec metrics")
    if not metrics.get("development_gate",{}).get("passed"):
        raise ValueError("M2.8 retrieval gate did not pass")
    comparable=("history_weeks","vector_size","context_window","min_count","negative",
        "epochs","workers","seed","seed_k","neighbor_k","candidate_k","tie_buffer",
        "query_batch_size","device")
    drift={key:(metrics["config"].get(key),ITEM2VEC_CONFIG[key]) for key in comparable
        if metrics["config"].get(key)!=ITEM2VEC_CONFIG[key]}
    if drift:
        raise ValueError(f"M2.8 Item2Vec config differs from M2.9: {drift}")
    if metrics["inputs"]["transactions"]["sha256"]!=transaction_identity["sha256"]:
        raise ValueError("M2.8 transaction identity differs from M2.9")
    reuse={}
    for cutoff,window in metrics["windows"].items():
        artifacts={
            "sequences":window["sequence"]["artifact"],
            "keyed_vectors":window["model"]["keyed_vectors"],
            "items":window["model"]["items"],
            "normalized_vectors":window["model"]["normalized_vectors"],
            "neighbor_indices":window["neighbors"]["indices"],
            "neighbor_scores":window["neighbors"]["scores"],
            "candidates":window["source"]["artifact"],
        }
        reuse[cutoff]={"artifacts":{key:_validate_file_identity(value,f"M2.8:{cutoff}:{key}")
            for key,value in artifacts.items()},"m2_8_window":window}
    return _file_identity(metrics_path),reuse


def _build_new_source(
    *, cutoff:str, window:M29Window, baseline_identity:dict[str,Any],
    transactions_path:Path, transaction_identity:dict[str,Any], source_dir:Path,
)->dict[str,Any]:
    source_dir.mkdir(parents=True)
    started=time.perf_counter()
    try:
        sequence_path=source_dir/"customer_sequences.parquet"
        sequence=_materialize_sequences(transactions_path=transactions_path,cutoff=cutoff,
            history_weeks=ITEM2VEC_CONFIG["history_weeks"],output_path=sequence_path,
            ordering_seed=ITEM2VEC_CONFIG["seed"])
        model,items,vectors=_train_item2vec(sequence_path=sequence_path,output_dir=source_dir,
            vector_size=ITEM2VEC_CONFIG["vector_size"],window=ITEM2VEC_CONFIG["context_window"],
            min_count=ITEM2VEC_CONFIG["min_count"],negative=ITEM2VEC_CONFIG["negative"],
            epochs=ITEM2VEC_CONFIG["epochs"],seed=ITEM2VEC_CONFIG["seed"])
        neighbors,indices,scores=_build_exact_neighbors(vectors=vectors,output_dir=source_dir,
            top_k=ITEM2VEC_CONFIG["neighbor_k"],tie_buffer=ITEM2VEC_CONFIG["tie_buffer"],
            query_batch_size=ITEM2VEC_CONFIG["query_batch_size"],device=ITEM2VEC_CONFIG["device"])
        database=source_dir/"source-build.duckdb"; connection=duckdb.connect(str(database))
        connection.execute("SET threads=8")
        connection.execute(f"CREATE TABLE baseline AS SELECT * FROM read_parquet({_literal(window.candidate_path)})")
        connection.execute("CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM baseline")
        connection.register("item2vec_items_frame",items)
        connection.execute("CREATE TABLE item2vec_vocab AS SELECT row_index,article_id,token_count FROM item2vec_items_frame")
        connection.unregister("item2vec_items_frame")
        seeds=_create_seeds(connection,transactions_path=transactions_path,cutoff=cutoff,
            history_weeks=ITEM2VEC_CONFIG["history_weeks"],seed_k=ITEM2VEC_CONFIG["seed_k"])
        article_ids=items.sort_values("row_index")["article_id"].to_numpy()
        source_stats=_insert_item2vec_candidates(connection,seeds=seeds,
            neighbor_indices=indices,neighbor_scores=scores,article_ids=article_ids,
            max_candidates=ITEM2VEC_CONFIG["candidate_k"])
        candidate_path=source_dir/"item2vec_candidates.parquet"
        connection.execute(f"COPY item2vec_candidates TO {_literal(candidate_path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
        audit=connection.execute("""WITH g AS (SELECT customer_id,count(*) n,
            count(DISTINCT article_id) u,min(item2vec_rank) lo,max(item2vec_rank) hi,
            count(DISTINCT item2vec_rank) r FROM item2vec_candidates GROUP BY customer_id)
            SELECT count(*),min(n),max(n),sum(n),sum(u),count(*) FILTER(
                WHERE n<>u OR lo<>1 OR hi<>n OR r<>n OR n>300) FROM g""").fetchone()
        seed_summary=connection.execute("SELECT count(DISTINCT customer_id),count(*) FROM seeds").fetchone()
        connection.close()
        if int(audit[5]): raise RuntimeError(f"invalid source groups for {cutoff}")
        artifacts={"sequences":sequence["artifact"],"keyed_vectors":model["keyed_vectors"],
            "items":model["items"],"normalized_vectors":model["normalized_vectors"],
            "neighbor_indices":neighbors["indices"],"neighbor_scores":neighbors["scores"],
            "candidates":_file_identity(candidate_path)}
        manifest={"schema_version":SOURCE_SCHEMA_VERSION,"status":"completed","cutoff":cutoff,
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"mode":"generated_cutoff_safe",
            "contract":ITEM2VEC_CONFIG,"inputs":{"baseline":baseline_identity,
                "baseline_candidate_sha256":baseline_identity["candidate_sha256"],
                "transactions":transaction_identity,"transactions_sha256":transaction_identity["sha256"]},
            "sequence":sequence,"model":model,"neighbors":neighbors,"source":source_stats,
            "seed_summary":{"users":int(seed_summary[0]),"rows":int(seed_summary[1])},
            "audit":{"users":int(audit[0]),"min_rows":int(audit[1]),"max_rows":int(audit[2]),
                "rows":int(audit[3]),"unique_rows":int(audit[4]),"invalid_groups":int(audit[5])},
            "artifacts":artifacts,"elapsed_seconds":time.perf_counter()-started}
        _write_json(source_dir/"source-manifest.json",manifest)
        return manifest
    except Exception as error:
        _write_json(source_dir/f"failure-{time.time_ns()}.json",{"schema_version":"m2.9-source-failure-v1",
            "status":"failed","error_type":type(error).__name__,"error":str(error),
            "elapsed_seconds":time.perf_counter()-started})
        raise



def build_source_cache(
    *, windows:list[M29Window], transactions_path:Path, m28_metrics_path:Path,
    source_cache_dir:Path, cutoffs:tuple[str,...]=REQUIRED_CUTOFFS,
)->dict[str,Any]:
    by_cutoff={window.cutoff:window for window in windows}
    if tuple(sorted(by_cutoff))!=tuple(sorted(cutoffs)):
        raise ValueError(f"candidate windows must match cutoffs {cutoffs}")
    source_cache_dir.mkdir(parents=True,exist_ok=True)
    transaction_identity=_file_identity(transactions_path.resolve())
    m28_identity,reuse=_load_m28_reuse(m28_metrics_path,transaction_identity)
    results={}
    for cutoff in cutoffs:
        window=by_cutoff[cutoff]
        baseline=validate_candidate_artifact(window.candidate_path,window.manifest_path,100)
        if baseline["cutoff"]!=cutoff or baseline["sample_rate"]!=0.1:
            raise ValueError(f"baseline identity mismatch: {cutoff}")
        source_dir=source_cache_dir/cutoff; manifest_path=source_dir/"source-manifest.json"
        existing=_source_manifest_valid(manifest_path,baseline,transaction_identity)
        if existing is not None:
            results[cutoff]=existing; continue
        if source_dir.exists():
            raise FileExistsError(f"incomplete source cache must be preserved: {source_dir}")
        if cutoff in reuse:
            source_dir.mkdir(parents=True)
            window_reuse=reuse[cutoff]
            m28_window=window_reuse["m2_8_window"]
            if m28_window["candidate_identity"]["candidate_sha256"]!=baseline["candidate_sha256"]:
                raise ValueError(f"M2.8 baseline differs for {cutoff}")
            manifest={"schema_version":SOURCE_SCHEMA_VERSION,"status":"completed","cutoff":cutoff,
                "created_at_utc":datetime.now(timezone.utc).isoformat(),"mode":"reused_m2_8_exact",
                "contract":ITEM2VEC_CONFIG,"inputs":{"baseline":baseline,
                    "baseline_candidate_sha256":baseline["candidate_sha256"],
                    "transactions":transaction_identity,"transactions_sha256":transaction_identity["sha256"],
                    "m2_8_metrics":m28_identity},"artifacts":window_reuse["artifacts"],
                "sequence":m28_window["sequence"],"model":m28_window["model"],
                "neighbors":m28_window["neighbors"],"source":m28_window["source"],
                "audit":{"reuse_validation":"path_bytes_sha256_all_passed"},"elapsed_seconds":0.0}
            _write_json(manifest_path,manifest); results[cutoff]=manifest
        else:
            results[cutoff]=_build_new_source(cutoff=cutoff,window=window,
                baseline_identity=baseline,transactions_path=transactions_path,
                transaction_identity=transaction_identity,source_dir=source_dir)
    return results


def _candidate_paths(cache_dir:Path,cutoff:str)->tuple[Path,Path,Path,Path]:
    root=cache_dir/cutoff
    return root/"expanded-candidates.parquet",root/"candidate-manifest.json",root/"features.parquet",root/"feature-manifest.json"


def _candidate_manifest_valid(
    manifest_path:Path,candidate_path:Path,baseline:dict[str,Any],source:dict[str,Any]
)->dict[str,Any]|None:
    if not manifest_path.is_file() or not candidate_path.is_file(): return None
    manifest=_read_json(manifest_path)
    if manifest.get("schema_version")!=CANDIDATE_SCHEMA_VERSION:
        raise ValueError(f"invalid M2.9 candidate manifest: {manifest_path}")
    if manifest["inputs"]["baseline_sha256"]!=baseline["candidate_sha256"]:
        raise ValueError(f"M2.9 baseline candidate cache drift: {manifest_path}")
    if manifest["inputs"]["source_candidate_sha256"]!=source["artifacts"]["candidates"]["sha256"]:
        raise ValueError(f"M2.9 source candidate cache drift: {manifest_path}")
    if manifest["inputs"]["source_items_sha256"]!=source["artifacts"]["items"]["sha256"]:
        raise ValueError(f"M2.9 source item vocabulary cache drift: {manifest_path}")
    actual=_file_identity(candidate_path)
    if actual["sha256"]!=manifest["artifact"]["sha256"] or actual["bytes"]!=manifest["artifact"]["bytes"]:
        raise ValueError(f"M2.9 expanded candidate artifact drift: {candidate_path}")
    return manifest


def build_expanded_candidate_cache(
    *, windows:list[M29Window], sources:dict[str,Any], cache_dir:Path,
    cutoffs:tuple[str,...]=REQUIRED_CUTOFFS,
)->dict[str,Any]:
    by_cutoff={window.cutoff:window for window in windows}; cache_dir.mkdir(parents=True,exist_ok=True)
    if tuple(sorted(by_cutoff))!=tuple(sorted(cutoffs)):
        raise ValueError(f"candidate windows must match cutoffs {cutoffs}")
    results={}
    for cutoff in cutoffs:
        started=time.perf_counter(); window=by_cutoff[cutoff]
        baseline=validate_candidate_artifact(window.candidate_path,window.manifest_path,100)
        source=sources[cutoff]
        candidate_path,manifest_path,_,_=_candidate_paths(cache_dir,cutoff)
        existing=_candidate_manifest_valid(manifest_path,candidate_path,baseline,source)
        if existing is not None: results[cutoff]=existing; continue
        root=candidate_path.parent
        if root.exists(): raise FileExistsError(f"incomplete M2.9 candidate cache: {root}")
        root.mkdir(parents=True)
        connection=duckdb.connect(str(root/"candidate-build.duckdb"))
        connection.execute("SET threads=8"); connection.execute("SET memory_limit='11GB'")
        temp=root/"duckdb-temp"; temp.mkdir(); connection.execute(f"SET temp_directory={_literal(temp)}")
        try:
            source_candidate=Path(source["artifacts"]["candidates"]["path"])
            items_path=Path(source["artifacts"]["items"]["path"])
            connection.execute(f"""CREATE TABLE baseline AS SELECT * FROM read_parquet({_literal(window.candidate_path)});
                CREATE TABLE item2vec_source AS SELECT * FROM read_parquet({_literal(source_candidate)});
                CREATE TABLE item2vec_vocab AS SELECT article_id,token_count FROM read_csv_auto({_literal(items_path)},header=true,all_varchar=false);
                CREATE TABLE item2vec_new AS SELECT i.*,row_number() OVER(
                    PARTITION BY customer_id ORDER BY item2vec_rank,article_id) AS new_rank
                FROM item2vec_source i WHERE NOT EXISTS(SELECT 1 FROM baseline b
                    WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
                QUALIFY new_rank<={ITEM2VEC_CONFIG['append_k']};""")
            base_projection=_base_retrieval_projection("b"); empty_projection=_empty_retrieval_projection()
            connection.execute(f"""CREATE TABLE expanded AS
                SELECT b.customer_id,b.article_id,{base_projection},
                    (i.article_id IS NOT NULL)::INTEGER AS item2vec_present,
                    0::INTEGER AS item2vec_is_new,i.item2vec_rank,i.item2vec_score,i.item2vec_cosine,
                    i.best_seed_rank AS item2vec_best_seed_rank,
                    i.best_neighbor_rank AS item2vec_best_neighbor_rank,
                    i.seed_support AS item2vec_seed_support,
                    coalesce(v.token_count,0)::BIGINT AS item2vec_vocab_count
                FROM baseline b LEFT JOIN item2vec_source i USING(customer_id,article_id)
                LEFT JOIN item2vec_vocab v USING(article_id)
                UNION ALL
                SELECT n.customer_id,n.article_id,(100+n.new_rank)::INTEGER AS candidate_rank,
                    {empty_projection},1::INTEGER AS item2vec_present,1::INTEGER AS item2vec_is_new,
                    n.item2vec_rank,n.item2vec_score,n.item2vec_cosine,
                    n.best_seed_rank AS item2vec_best_seed_rank,
                    n.best_neighbor_rank AS item2vec_best_neighbor_rank,
                    n.seed_support AS item2vec_seed_support,
                    coalesce(v.token_count,0)::BIGINT AS item2vec_vocab_count
                FROM item2vec_new n LEFT JOIN item2vec_vocab v USING(article_id);""")
            audit=connection.execute("""WITH g AS (SELECT customer_id,count(*) n,
                count(DISTINCT article_id) u,min(candidate_rank) lo,max(candidate_rank) hi,
                count(DISTINCT candidate_rank) r FROM expanded GROUP BY customer_id)
                SELECT (SELECT count(*) FROM expanded),(SELECT count(DISTINCT (customer_id,article_id)) FROM expanded),
                    count(*),min(n),max(n),count(*) FILTER(WHERE n<>u OR lo<>1 OR hi<>n OR r<>n OR n<100 OR n>300),
                    (SELECT count(*) FROM expanded WHERE item2vec_is_new=1),
                    count(*) FILTER(WHERE n=100),
                    (SELECT count(*) FROM expanded WHERE item2vec_present=1 AND item2vec_vocab_count<2)
                FROM g""").fetchone()
            values=list(map(int,audit))
            if values[0]!=values[1] or values[5] or values[8]:
                raise RuntimeError(f"M2.9 expanded candidate audit failed: {cutoff} {values}")
            connection.execute(f"COPY (SELECT * FROM expanded ORDER BY customer_id,candidate_rank,article_id) "
                f"TO {_literal(candidate_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
            manifest={"schema_version":CANDIDATE_SCHEMA_VERSION,"status":"completed","cutoff":cutoff,
                "created_at_utc":datetime.now(timezone.utc).isoformat(),"contract":{"baseline_k":100,
                    "append_item2vec_only_k":ITEM2VEC_CONFIG["append_k"],"group_rows":"variable_100_to_300_no_padding",
                    "source_features":ITEM2VEC_FEATURES},"inputs":{"baseline":baseline,
                    "baseline_sha256":baseline["candidate_sha256"],"source_mode":source["mode"],
                    "source_candidate_sha256":source["artifacts"]["candidates"]["sha256"],
                    "source_items_sha256":source["artifacts"]["items"]["sha256"]},
                "audit":{"rows":values[0],"unique_rows":values[1],"users":values[2],
                    "min_group_rows":values[3],"max_group_rows":values[4],"invalid_groups":values[5],
                    "item2vec_only_rows":values[6],"base_only_users":values[7],
                    "invalid_vocab_feature_rows":values[8]},"artifact":_file_identity(candidate_path),
                "elapsed_seconds":time.perf_counter()-started}
            _write_json(manifest_path,manifest); results[cutoff]=manifest
        finally:
            connection.close()
    return results


def build_feature_cache(
    *, raw_dir:Path,work_dir:Path,cache_dir:Path,candidates:dict[str,Any],config:M2Config,
    cutoffs:tuple[str,...]=REQUIRED_CUTOFFS,
)->dict[str,Any]:
    connection=prepare_tabular_connection(raw_dir,work_dir); results={}
    try:
        _create_static_dimensions(connection)
        for cutoff in cutoffs:
            candidate_path,_,feature_path,evidence_path=_candidate_paths(cache_dir,cutoff)
            candidate_sha=candidates[cutoff]["artifact"]["sha256"]
            if feature_path.is_file() and evidence_path.is_file():
                evidence=_read_json(evidence_path)
                if evidence.get("schema_version")!=FEATURE_SCHEMA_VERSION or evidence.get("candidate_sha256")!=candidate_sha or evidence.get("dataset_sha256")!=_sha256(feature_path):
                    raise ValueError(f"stale M2.9 feature cache: {feature_path}")
                results[cutoff]=evidence; continue
            if feature_path.exists() or evidence_path.exists(): raise FileExistsError(f"incomplete M2.9 feature cache: {feature_path.parent}")
            evidence=build_point_in_time_dataset(connection,candidate_path,
                {"cutoff":cutoff,"declared_rows":candidates[cutoff]["audit"]["rows"]},
                feature_path,config,retrieval_features=RETRIEVAL_FEATURES+ITEM2VEC_FEATURES,
                candidate_group_range=(100,300))
            evidence.update({"schema_version":FEATURE_SCHEMA_VERSION,"candidate_sha256":candidate_sha,
                "dataset_sha256":evidence["dataset_sha256"],"item2vec_features":ITEM2VEC_FEATURES})
            _write_json(evidence_path,evidence); results[cutoff]=evidence
    finally:
        connection.close()
    return results



def _score_models(
    *,dataset_path:Path,models:dict[str,tuple[Any,list[str],dict[str,dict[int,int]]]],
    evaluation_db:Path,prediction_path:Path,config:M2Config,
)->dict[str,Any]:
    if evaluation_db.exists() or prediction_path.exists():
        raise FileExistsError(evaluation_db if evaluation_db.exists() else prediction_path)
    started=time.perf_counter(); source=duckdb.connect(); target=duckdb.connect(str(evaluation_db))
    temp=evaluation_db.parent/"duckdb-temp"; temp.mkdir(exist_ok=True)
    target.execute(f"SET threads={config.threads}"); target.execute("SET memory_limit='11GB'")
    target.execute(f"SET temp_directory={_literal(temp)}")
    identity=["customer_id","article_id","candidate_rank","target","user_history_events_12w"]
    cursor=source.execute(f"SELECT {','.join(identity+ALL_FEATURES)} FROM read_parquet({_literal(dataset_path)})")
    rows=0;batches=0;initialized=False
    try:
        while True:
            batch=cursor.fetch_df_chunk(config.prediction_chunk_vectors)
            if batch.empty: break
            output=batch[identity].copy()
            for name,(model,features,maps) in models.items():
                output[f"score_{name}"]=model.predict(_prepare_frame(batch,features,maps))
            target.register("m29_prediction_batch",output)
            if not initialized:
                target.execute("CREATE TABLE predictions AS SELECT * FROM m29_prediction_batch WHERE FALSE")
                initialized=True
            target.execute("INSERT INTO predictions SELECT * FROM m29_prediction_batch")
            target.unregister("m29_prediction_batch"); rows+=len(output); batches+=1
        if not initialized: raise RuntimeError("validation dataset produced no rows")
        duplicate=int(target.execute("SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM predictions").fetchone()[0])
        if duplicate: raise RuntimeError("prediction rows are not unique by customer-item")
        target.execute(f"COPY predictions TO {_literal(prediction_path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
    finally:
        source.close();target.close()
    return {"rows":rows,"batches":batches,"prediction_path":str(prediction_path.resolve()),
        "prediction_bytes":prediction_path.stat().st_size,"prediction_sha256":_sha256(prediction_path),
        "evaluation_db":str(evaluation_db.resolve()),"elapsed_seconds":time.perf_counter()-started}


def _attach_popularity_segments(
    *,evaluation:dict[str,Any],evaluation_db:Path,transactions_path:Path,
    cutoff:str,variant_names:list[str],budget:int,
)->dict[str,Any]:
    """Add cutoff-safe head/middle/tail diagnostics to the shared M2 evaluation."""
    connection=duckdb.connect(str(evaluation_db))
    try:
        connection.execute("SET threads=8")
        transaction_sql=_literal(transactions_path.resolve());cutoff_sql=_date(cutoff)
        connection.execute(f"""
            CREATE OR REPLACE TABLE m29_eval_users AS
                SELECT DISTINCT customer_id FROM predictions;
            CREATE OR REPLACE TABLE m29_warm_catalog AS
                SELECT DISTINCT article_id FROM read_parquet({transaction_sql})
                WHERE t_dat < {cutoff_sql};
            CREATE OR REPLACE TABLE m29_item_popularity AS
            WITH counts AS (
                SELECT article_id,count(*) AS events
                FROM read_parquet({transaction_sql})
                WHERE t_dat >= {cutoff_sql}-INTERVAL 12 WEEK AND t_dat < {cutoff_sql}
                GROUP BY article_id
            ), bucketed AS (
                SELECT *,ntile(5) OVER(ORDER BY events DESC,article_id) AS bucket
                FROM counts
            )
            SELECT article_id,CASE WHEN bucket=1 THEN 'head_top20pct_items'
                WHEN bucket<=3 THEN 'middle_next40pct_items'
                ELSE 'tail_bottom40pct_items' END AS popularity_segment
            FROM bucketed;
            CREATE OR REPLACE TABLE m29_eval_truth AS
            SELECT q.customer_id,q.article_id,
                CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature,
                coalesce(p.popularity_segment,'cold_or_unseen') AS popularity_segment
            FROM (
                SELECT DISTINCT t.customer_id,t.article_id
                FROM read_parquet({transaction_sql}) t
                JOIN m29_eval_users u USING(customer_id)
                WHERE t.t_dat >= {cutoff_sql} AND t.t_dat < {cutoff_sql}+INTERVAL 7 DAY
            ) q
            LEFT JOIN m29_warm_catalog w USING(article_id)
            LEFT JOIN m29_item_popularity p USING(article_id);
        """)
        ordering_sql={"expanded_append_order":"candidate_rank ASC"}
        for name in variant_names:
            score=f"score_{name}"
            ordering_sql[name]=f"{score} DESC"
            ordering_sql[f"{name}__inactive_rrf"]=inactive_fallback_order_expression(score)
        labels=("head_top20pct_items","middle_next40pct_items",
            "tail_bottom40pct_items","cold_or_unseen")
        for index,(name,order_expression) in enumerate(ordering_sql.items()):
            table=f"m29_popularity_ranked_{index}"
            connection.execute(f"""CREATE OR REPLACE TEMP TABLE {table} AS
                SELECT customer_id,article_id,row_number() OVER(
                    PARTITION BY customer_id
                    ORDER BY {order_expression},candidate_rank,article_id
                ) AS candidate_rank
                FROM predictions""")
            segments={}
            for label in labels:
                value=_candidate_metrics(connection,table,budget,
                    f"popularity_segment='{label}'",truth_table="m29_eval_truth")
                value["candidate_recall@expanded_pool"]=value[f"candidate_recall@{budget}"]
                value["candidate_hit_rate@expanded_pool"]=value[f"candidate_hit_rate@{budget}"]
                segments[label]=value
            evaluation["orderings"][name]["popularity_segments"]=segments
        return evaluation
    finally:
        connection.close()


def _activity_by_name(rows:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    mapped={str(row["activity_segment"]):row for row in rows}
    if len(mapped)!=len(rows): raise ValueError("duplicate activity segment rows")
    return mapped

def _summary(development:dict[str,Any],frozen:dict[str,Any],metric_k:int)->dict[str,Any]:
    metric=f"map@{metric_k}"; ordering_names=["expanded_append_order"]+[
        name for variant in FEATURE_SETS for name in (variant,f"{variant}__inactive_rrf")]
    orderings={}
    for name in ordering_names:
        values={dev:float(result["evaluation"]["orderings"][name]["segments"]["overall"][metric])
            for dev,result in development.items()}
        orderings[name]={"window_map@12":values,"mean_map@12":float(np.mean(list(values.values()))),
            "min_map@12":float(np.min(list(values.values())))}
    frozen_values={dev:float(ordering["segments"]["overall"][metric]) for dev,ordering in frozen.items()}
    frozen_mean=float(np.mean(list(frozen_values.values())))
    eligible=[f"{name}__inactive_rrf" for name in FEATURE_SETS]
    selected=max(eligible,key=lambda name:(orderings[name]["mean_map@12"],orderings[name]["min_map@12"],name))
    selected_values=orderings[selected]["window_map@12"]
    deltas={dev:selected_values[dev]-frozen_values[dev] for dev in PROTOCOL}
    gate={"accepted":all(delta>=0 for delta in deltas.values()) and orderings[selected]["mean_map@12"]>frozen_mean,
        "rule":"mean MAP@12 improves over frozen M2.2 and neither dev window regresses",
        "selected_variant":selected,"window_delta_vs_frozen_map@12":deltas,
        "mean_delta_vs_frozen_map@12":orderings[selected]["mean_map@12"]-frozen_mean}
    realization={};activity={};popularity={}
    for dev,result in development.items():
        selected_eval=result["evaluation"]["orderings"][selected]
        expanded=selected_eval["segments"]["overall"];frozen_eval=frozen[dev]
        frozen_segment=frozen_eval["segments"]["overall"]
        realization[dev]={"expanded_recall":float(expanded["candidate_recall@expanded_pool"]),
            "expanded_oracle_map@12":float(expanded[f"oracle_map@{metric_k}"]),
            "frozen_oracle_map@12":float(frozen_segment[f"oracle_map@{metric_k}"]),
            "realized_fraction":realized_oracle_fraction(selected_values[dev],frozen_values[dev],
                float(expanded[f"oracle_map@{metric_k}"]),float(frozen_segment[f"oracle_map@{metric_k}"]))}
        selected_activity=_activity_by_name(selected_eval["activity_segments"])
        frozen_activity=_activity_by_name(frozen_eval["activity_segments"])
        activity[dev]={segment:{"map@12":float(value[metric]),
            "delta_vs_frozen_map@12":float(value[metric]-frozen_activity[segment][metric])}
            for segment,value in selected_activity.items()}
        popularity[dev]={segment:{"map@12":float(value[metric]),
            "candidate_recall@expanded_pool":float(value["candidate_recall@expanded_pool"]),
            "oracle_map@12":float(value[f"oracle_map@{metric_k}"])}
            for segment,value in selected_eval["popularity_segments"].items()}
    return {"orderings":orderings,"frozen_m2_2":{"window_map@12":frozen_values,
        "mean_map@12":frozen_mean},"selection_gate":gate,"oracle_realization":realization,
        "selected_activity":activity,"selected_popularity":popularity,"final_week":"not_run"}


def _render_report(result:dict[str,Any])->str:
    summary=result["development_summary"];gate=summary["selection_gate"];frozen=summary["frozen_m2_2"]
    lines=["# M2.9 Item2Vec expanded-pool ranking","","## 结论","",
        f"- development gate：{'通过' if gate['accepted'] else '未通过'}；最佳候选 `{gate['selected_variant']}`。",
        f"- 相对冻结 M2.2 mean MAP@12：{gate['mean_delta_vs_frozen_map@12']:+.6f}。",
        "- strict train/validation 不补正例；200 rounds、树参数和 inactive fallback 冻结。",
        "- final week 未运行；本报告不是 Kaggle leaderboard 分数。","","## MAP@12","",
        "| ordering | dev-A | dev-B | mean | worst |","|---|---:|---:|---:|---:|",
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | {frozen['window_map@12']['dev_b']:.6f} | {frozen['mean_map@12']:.6f} | {min(frozen['window_map@12'].values()):.6f} |"]
    for name,row in summary["orderings"].items():
        lines.append(f"| {name} | {row['window_map@12']['dev_a']:.6f} | {row['window_map@12']['dev_b']:.6f} | {row['mean_map@12']:.6f} | {row['min_map@12']:.6f} |")
    lines.extend(["","## Oracle 兑现",""])
    for dev,value in summary["oracle_realization"].items():
        fraction=value["realized_fraction"];label="n/a" if fraction is None else f"{fraction:.2%}"
        lines.append(f"- {dev}：expanded Recall={value['expanded_recall']:.6f}，expanded Oracle={value['expanded_oracle_map@12']:.6f}，frozen Oracle={value['frozen_oracle_map@12']:.6f}，兑现率={label}。")
    lines.extend(["","## 审计边界","",
        "- 05-27/06-24 新建 cutoff-safe Item2Vec；07-22/08-19 只在 M2.8 path/bytes/SHA 全通过后复用。",
        "- 候选池固定为 six-source Top100 + up to 200 Item2Vec-only；不包含 image candidates。",
        "- no-item2vec/full-item2vec 两模型使用完全相同候选池，仅 source feature set 不同。",
        "- `candidate_recall@expanded_pool` 是完整 100–300 变长候选池；历史字段 `candidate_recall@100` 仅为共享评测函数遗留别名。",
        "- optimistic all-articles、unobserved-negative 与在既有 dev windows 上选择 retrieval 的非 nested 边界仍存在。","",
        "## 产物","",f"- metrics：{result['artifacts']['metrics']}",
        f"- private artifacts：{result['artifacts']['artifact_dir']}（Git ignored）",""])
    return "\n".join(lines)



def run_m29(
    *,raw_dir:Path,work_dir:Path,transactions_path:Path,m21_metrics_path:Path,
    m28_metrics_path:Path,windows:list[M29Window],source_cache_dir:Path,cache_dir:Path,
    output_dir:Path,artifact_dir:Path,config:M2Config,run_id:str,
)->dict[str,Any]:
    if config.evaluation_role!="development": raise ValueError("M2.9 is development-only")
    if config.candidate_k!=300: raise ValueError("M2.9 candidate_k must be 300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True);artifact_dir.mkdir(parents=True);started=time.perf_counter()
    try:
        sources=build_source_cache(windows=windows,transactions_path=transactions_path,
            m28_metrics_path=m28_metrics_path,source_cache_dir=source_cache_dir)
        candidates=build_expanded_candidate_cache(windows=windows,sources=sources,cache_dir=cache_dir)
        features=build_feature_cache(raw_dir=raw_dir,work_dir=work_dir,cache_dir=cache_dir,
            candidates=candidates,config=config)
        m21=_read_json(m21_metrics_path.resolve());frozen_name,frozen=_frozen_reference(m21)
        development={}
        for dev,protocol in PROTOCOL.items():
            dev_dir=artifact_dir/dev;dev_dir.mkdir()
            train_paths=[_candidate_paths(cache_dir,cutoff)[2] for cutoff in protocol["train"]]
            frame,group_sizes,group_evidence=_load_training_data(train_paths,ALL_FEATURES)
            group_evidence=validate_group_sizes(group_sizes)|{
                "positive_groups":group_evidence["positive_groups"],
                "zero_positive_groups":group_evidence["zero_positive_groups"],
                "positive_group_rate":group_evidence["positive_group_rate"],
            }
            maps=build_category_maps(frame);map_evidence=_save_category_maps(dev_dir/"category-maps.json",maps)
            models={};model_evidence={}
            for name,feature_set in FEATURE_SETS.items():
                model,evidence=_train_lambdarank(frame=frame,group_sizes=group_sizes,
                    group_evidence=group_evidence,features=feature_set,name=name,
                    artifact_dir=dev_dir,config=config,category_maps=maps)
                gains=model.feature_importance(importance_type="gain")
                splits=model.feature_importance(importance_type="split")
                evidence["item2vec_feature_importance"]=[{"feature":feature,
                    "gain":float(gains[feature_set.index(feature)]),
                    "split":int(splits[feature_set.index(feature)])}
                    for feature in ITEM2VEC_FEATURES if feature in feature_set]
                models[name]=(model,feature_set,maps);model_evidence[name]=evidence
            del frame;gc.collect()
            valid_cutoff=protocol["validation"];valid_path=_candidate_paths(cache_dir,valid_cutoff)[2]
            scoring=_score_models(dataset_path=valid_path,models=models,
                evaluation_db=dev_dir/"evaluation.duckdb",
                prediction_path=dev_dir/"validation-predictions.parquet",config=config)
            del models;gc.collect()
            evaluation=_evaluate_models(evaluation_db=dev_dir/"evaluation.duckdb",
                transactions_path=transactions_path,cutoff=valid_cutoff,
                metric_k=config.metric_k,variant_names=list(FEATURE_SETS))
            evaluation=_attach_popularity_segments(evaluation=evaluation,
                evaluation_db=dev_dir/"evaluation.duckdb",
                transactions_path=transactions_path,cutoff=valid_cutoff,
                variant_names=list(FEATURE_SETS),budget=config.candidate_k)
            development[dev]={"train_cutoffs":protocol["train"],
                "validation_cutoff":valid_cutoff,"models":model_evidence,
                "category_encoding":map_evidence,"scoring":scoring,"evaluation":evaluation}
        summary=_summary(development,frozen,config.metric_k)
        result={"schema_version":SCHEMA_VERSION,"stage":"M2.9","status":"measured",
            "run_id":run_id,"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"development_only_no_final_week","config":asdict(config),
            "contract":{"candidate_pool":"baseline_top100_plus_up_to_200_item2vec_only",
                "variable_group_rows":[100,300],"negative_sampling":"none_strict_keep_all_candidates",
                "objective":"pooled_lambdarank_full_only","frozen_reference":frozen_name,
                "feature_sets":FEATURE_SETS,"item2vec_source_config":ITEM2VEC_CONFIG,
                "image_candidates":"excluded"},"source_cache":sources,
            "candidate_cache":candidates,"feature_cache":features,"development":development,
            "development_summary":summary,"elapsed_seconds":time.perf_counter()-started}
        metrics_path=output_dir/"metrics.json";report_path=output_dir/"M2_9_REPORT.md"
        result["artifacts"]={"metrics":str(metrics_path.resolve()),"report":str(report_path.resolve()),
            "artifact_dir":str(artifact_dir.resolve()),"source_cache_dir":str(source_cache_dir.resolve()),
            "cache_dir":str(cache_dir.resolve())}
        _write_json(metrics_path,result);report_path.write_text(_render_report(result),encoding="utf-8",newline="\n")
        return result
    except Exception as error:
        _write_json(output_dir/f"failure-{time.time_ns()}.json",{"schema_version":"m2.9-failure-v1",
            "status":"failed","error_type":type(error).__name__,"error":str(error),
            "elapsed_seconds":time.perf_counter()-started})
        raise
