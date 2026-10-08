"""Frozen Warm-v1 mechanics with isolated, additive feature experiments."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .m2 import CATEGORICAL_FEATURES, build_category_maps
from .m211 import _distribution_relation_sql, SAMPLING_SEED
from .m212 import make_exact_map_evaluator
from .m3 import _peak_working_set_bytes
from .metrics import apk
from .warm_v2_contract import ARTIFACT, M33, REPORT, evidence_id, guard, read, write


def literal(path):
    return "'"+str(path).replace("'", "''")+"'"


def connection():
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='3GB'")
    (ARTIFACT/"spill").mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory={literal((ARTIFACT/'spill').resolve())}")
    return con


def save_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.register("outframe", frame)
        con.execute(f"COPY outframe TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)")


def load_parquet(path):
    with connection() as con:
        return con.execute(f"SELECT * FROM read_parquet({literal(path)})").fetchdf()


def prepare(frame, features, maps):
    # Same conversions/column order as frozen _prepare_frame, allocated in one pass.
    values = {}
    for name in features:
        v = pd.to_numeric(frame[name], errors="coerce")
        values[name] = v.map(maps[name]).fillna(0).astype(np.int32) if name in CATEGORICAL_FEATURES else v.astype(np.float32)
    return pd.DataFrame(values, index=frame.index)


class Engine:
    def __init__(self, contract, parameter_overrides=None):
        self.contract = contract
        self.history = read(M33)
        self.features = contract["frozen_feature_columns"]
        self.transactions = self.history["inputs"]["transactions"]["path"]
        self.parameter_overrides=dict(parameter_overrides or {})
        if self.parameter_overrides:
            assert set(self.parameter_overrides)=={'num_leaves'}
            assert self.parameter_overrides['num_leaves'] in (15,63)

    def base_path(self, cutoff):
        guard(cutoff)
        return Path(self.history["feature_cache"][cutoff]["artifact"]["path"])

    def verify_historical_inputs(self):
        destination = REPORT/"WARM_V1_INPUT_IDENTITY.json"
        if destination.exists():
            return read(destination)
        # Old, mutable shared D-drive caches, compared to trusted committed M3.3.
        expected = [v["artifact"] for v in self.history["feature_cache"].values()]
        expected += [self.history["inputs"]["transactions"], self.history["prerequisite_cache"]["target_features"]["2020-01-22"]["inputs"]["articles"]]
        for window, row in self.history["development"].items():
            expected.append(row["outer_category_encoding"])
            model = row["outer_models"]["anchor"]
            expected.append({"path": model["model_path"], "bytes": model["model_bytes"], "sha256": model["model_sha256"]})
        results = []
        for entry in expected:
            observed = evidence_id(entry["path"], reason="compare_frozen_m33")
            assert all(observed[k] == entry[k] for k in ("bytes", "sha256")), entry["path"]
            results.append({**observed, "trusted_reference": str(M33), "match": True})
        result = {"purpose": "prior shared caches/models versus trusted committed M3.3 checksums; once per continuous run",
            "files": results, "final_week": "not_run"}
        write(destination, result)
        return result

    def cached_data(self, cutoff, role):
        guard(cutoff)
        destination = ARTIFACT/"frozen_data"/cutoff/f"{role}.parquet"
        meta = destination.with_suffix(".json")
        if destination.exists() and meta.exists():
            return destination, read(meta)
        columns = list(dict.fromkeys(["customer_id", "article_id", *self.features, "target", "user_history_events_12w"]))
        source = f"read_parquet({literal(self.base_path(cutoff))})"
        with connection() as con:
            if role == "sample":
                relation = _distribution_relation_sql(self.base_path(cutoff), seed=SAMPLING_SEED)
                frame = con.execute(f"SELECT {','.join(columns)} FROM ({relation}) ORDER BY customer_id,candidate_rank,article_id").fetchdf()
                group = frame.groupby("customer_id", sort=False).target.agg(["size", "sum"])
                assert group['sum'].gt(0).all() and int((frame.target==0).sum()) <= 30*int(frame.target.sum())
            elif role == "inner":
                frame = con.execute(f"""WITH eligible AS (
                    SELECT customer_id FROM {source} GROUP BY customer_id
                    HAVING sum(target)>0 AND max(user_history_events_12w)>0
                ), truth AS (
                    SELECT customer_id,count(DISTINCT article_id)::BIGINT AS truth_count
                    FROM read_parquet({literal(self.transactions)})
                    WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY GROUP BY customer_id
                ) SELECT {','.join('s.'+v for v in columns)},t.truth_count
                FROM {source} s JOIN eligible USING(customer_id) JOIN truth t USING(customer_id)
                ORDER BY s.customer_id,s.candidate_rank,s.article_id""").fetchdf()
                group = frame.groupby("customer_id", sort=False).target.agg(["size", "sum"])
                assert group['sum'].gt(0).all() and frame.user_history_events_12w.gt(0).all()
            else:
                raise ValueError(role)
            original = con.execute(f"SELECT count(*),count(DISTINCT customer_id) FROM {source}").fetchone()
        assert not frame.duplicated(["customer_id", "article_id"]).any()
        assert group['size'].between(2, 300).all()
        save_parquet(frame, destination)
        stats = {"cutoff": cutoff, "role": role, "rows": len(frame), "groups": len(group),
            "positive_rows": int(frame.target.sum()), "source_rows": int(original[0]), "source_users": int(original[1]),
            "min_group_rows": int(group['size'].min()), "max_group_rows": int(group['size'].max()),
            "source_path": str(self.base_path(cutoff)), "source_sha256": self.history["feature_cache"][cutoff]["artifact"]["sha256"],
            "candidate_feature_contract": "same ordered cached rows, unchanged frozen84 numeric values; selected with original SQL",
            "artifact": evidence_id(destination, reason="explicit_registry_evidence")}
        write(meta, stats)
        return destination, stats

    def dataset(self, cutoffs, role, families):
        frames, sizes, stats = [], [], []
        for cutoff in cutoffs:
            path, meta = self.cached_data(cutoff, role)
            frame = load_parquet(path)
            if families:
                from .warm_v2_features import attach_features
                frame = attach_features(frame, cutoff, families, self)
            size = frame.groupby("customer_id", sort=False).size().tolist()
            frames.append(frame)
            sizes.extend(size)
            stats.append(meta)
        frame = pd.concat(frames, ignore_index=True)
        assert sum(sizes) == len(frame)
        return frame, sizes, stats

    def train_inner(self, window, trial, families, extra_columns):
        start = time.perf_counter()
        protocol = self.contract["rolling_protocol"][window]
        train, groups, ts = self.dataset(protocol["inner_train"], "sample", families)
        valid, vgroups, vs = self.dataset([protocol["inner_validation"]], "inner", families)
        maps = build_category_maps(train)
        features = self.features+extra_columns
        x, xv = prepare(train, features, maps), prepare(valid, features, maps)
        labels = valid.target.to_numpy(np.uint8)
        truth = valid.groupby("customer_id", sort=False).truth_count.first().to_numpy(np.int64)
        cats = [v for v in CATEGORICAL_FEATURES if v in features]
        dt = lgb.Dataset(x, label=train.target.astype(np.uint8), group=groups, feature_name=features, categorical_feature=cats, free_raw_data=True)
        dv = lgb.Dataset(xv, label=labels, group=vgroups, feature_name=features, categorical_feature=cats, reference=dt, free_raw_data=True)
        params = {**self.history["development"][window]["inner_models"]["anchor"]["parameters"],**self.parameter_overrides}
        curve = {}
        model = lgb.train(params, dt, num_boost_round=self.contract["config"]["num_boost_round"],
            valid_sets=[dv], valid_names=["inner_validation"], feval=make_exact_map_evaluator(labels,vgroups,truth,k=12),
            callbacks=[lgb.early_stopping(20,first_metric_only=True,verbose=False),lgb.record_evaluation(curve)])
        best = int(model.best_iteration)
        score = float(curve["inner_validation"]["exact_map_at_12"][best-1])
        root = ARTIFACT/trial/window
        root.mkdir(parents=True, exist_ok=True)
        path = root/"inner_model.txt"
        model.save_model(str(path), num_iteration=best)
        write(root/"inner_category_maps.json", maps)
        result = {"window": window, "role": "historical_inner_only", "feature_count": len(features), "features": features,
            "families": families, "cutoff": protocol["inner_validation"], "train_cutoffs": protocol["inner_train"],
            "parameters": params, "best_iteration": best, "inner_covered_active_map": score,
            "metric_curve": curve["inner_validation"]["exact_map_at_12"],
            "inner_population_weight": vs[0]["groups"]/vs[0]["source_users"],
            "sampling": ts, "validation": vs, "runtime_seconds": time.perf_counter()-start,
            "process_peak_working_set_bytes": _peak_working_set_bytes(),
            "peak_memory_scope": "process lifetime through this fit; may include earlier fits in the same batch",
            "model": evidence_id(path, reason="explicit_registry_evidence")}
        if trial == "WV2-000":
            old = self.history["development"][window]["inner_models"]["anchor"]
            assert best == old["best_iteration"] and abs(score-old["best_score"]) < 1e-12, (window,best,score,old['best_iteration'],old['best_score'])
            assert sum(s["rows"] for s in ts) == old["train_sampling"]["sampled_rows"]
            assert sum(s["groups"] for s in ts) == old["train_sampling"]["groups"]
            assert vs[0]["rows"] == old["validation"]["selected_rows"]
            result["historical_parity"] = True
        write(root/"inner_metrics.json", result)
        del train, valid, x, xv, dt, dv, model
        gc.collect()
        return result

    def train_outer(self, window, trial, families, extra_columns, inner):
        start = time.perf_counter()
        protocol = self.contract["rolling_protocol"][window]
        frame, groups, stats = self.dataset(protocol["outer_train"], "sample", families)
        maps = build_category_maps(frame)
        features = self.features+extra_columns
        x = prepare(frame, features, maps)
        ds = lgb.Dataset(x,label=frame.target.astype(np.uint8),group=groups,feature_name=features,
            categorical_feature=[v for v in CATEGORICAL_FEATURES if v in features],free_raw_data=True)
        params = {**self.history["development"][window]["outer_models"]["anchor"]["parameters"],**self.parameter_overrides}
        model = lgb.train(params,ds,num_boost_round=inner["best_iteration"],valid_sets=[ds],valid_names=["train"])
        root = ARTIFACT/trial/window
        path = root/"outer_model.txt"
        model.save_model(str(path))
        write(root/"outer_category_maps.json", maps)
        evidence = {"train_cutoffs": protocol["outer_train"], "cutoff": protocol["outer_validation"],
            "parameters": params, "rounds": inner["best_iteration"], "sampling": stats,
            "features": features, "families": families, "feature_count": len(features),
            "model": evidence_id(path, reason="explicit_registry_evidence"),
            "feature_importance": [{"feature": v,"gain": float(g),"splits": int(n)} for v,g,n in zip(features,model.feature_importance('gain'),model.feature_importance('split'))]}
        if trial == "WV2-000":
            old = self.history["development"][window]
            assert sum(s["rows"] for s in stats) == old["outer_sampling"]["sampled_rows"]
            assert sum(s["groups"] for s in stats) == old["outer_sampling"]["groups"]
            assert maps == {k:{int(a):int(b) for a,b in v.items()} for k,v in read(old["outer_category_encoding"]["path"]).items()}
        del frame,x,ds
        gc.collect()
        evidence["evaluation"] = self.score(window,trial,model,maps,features,families)
        evidence["runtime_seconds"] = time.perf_counter()-start
        evidence["process_peak_working_set_bytes"] = _peak_working_set_bytes()
        evidence["peak_memory_scope"] = "process lifetime through this fit; not isolated per-window peak"
        write(root/"outer_metrics.json",evidence)
        del model
        gc.collect()
        return evidence

    def score(self, window, trial, model, maps, features, families):
        cutoff = guard(self.contract["rolling_protocol"][window]["outer_validation"])
        root = ARTIFACT/trial/window
        source = connection()
        dest = duckdb.connect(str(root/"evaluation.duckdb"))
        try:
            cols = list(dict.fromkeys(["customer_id","article_id",*self.features,"target","user_history_events_12w"]))
            cursor = source.execute(f"SELECT {','.join(cols)} FROM read_parquet({literal(self.base_path(cutoff))})")
            batches, rows = 0,0
            while True:
                frame = cursor.fetch_df_chunk(vectors_per_chunk=64)
                if frame.empty:
                    break
                if families:
                    from .warm_v2_features import attach_features
                    frame = attach_features(frame,cutoff,families,self)
                result = frame[["customer_id","article_id","candidate_rank","target","user_history_events_12w"]].copy()
                result['score'] = model.predict(prepare(frame,features,maps),num_threads=8)
                dest.register("batch",result)
                if batches == 0:
                    dest.execute("CREATE TABLE predictions AS SELECT * FROM batch")
                else:
                    dest.execute("INSERT INTO predictions SELECT * FROM batch")
                rows += len(frame)
                batches += 1
            expected_rows = self.history["feature_cache"][cutoff]["audit"]["source_rows"]
            assert rows == expected_rows
            # Explicitly retain all users including those with no covered truth.
            dest.execute(f"""CREATE TEMP TABLE truth AS SELECT DISTINCT t.customer_id,t.article_id
                FROM read_parquet({literal(self.transactions)}) t
                WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
                AND customer_id IN (SELECT DISTINCT customer_id FROM predictions)""")
            dest.execute("""CREATE TABLE top12 AS SELECT * FROM (
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY
                CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
                CASE WHEN user_history_events_12w>0 THEN score END DESC NULLS LAST,
                candidate_rank,article_id)::INTEGER AS final_rank FROM predictions
            ) WHERE final_rank<=12""")
            top = dest.execute("SELECT * FROM top12 ORDER BY customer_id,final_rank").fetchdf()
            truth = dest.execute("SELECT * FROM truth ORDER BY customer_id,article_id").fetchdf()
            truthsets = {u:set(g.article_id) for u,g in truth.groupby("customer_id",sort=False)}
            pred = {u:g.article_id.tolist() for u,g in top.groupby("customer_id",sort=False)}
            assert set(truthsets) == set(pred) and all(len(set(items))==12 for items in pred.values())
            ap = np.array([apk(list(items),pred[u]) for u,items in truthsets.items()])
            hit_counts = dict(dest.execute("SELECT customer_id,sum(target) FROM predictions GROUP BY customer_id").fetchall())
            result = {"map@12": float(ap.mean()), "users": len(pred), "truth_pairs": len(truth), "candidate_rows": rows,
                "candidate_recall": float(np.mean([hit_counts[u]/len(t) for u,t in truthsets.items()])),
                "candidate_oracle_map@12": float(np.mean([min(hit_counts[u],12)/min(len(t),12) for u,t in truthsets.items()])),
                "final_week": "not_run"}
            if trial == "WV2-000":
                old = self.history["development"][window]
                dest.execute(f"ATTACH {literal(old['scoring']['evaluation_db'])} AS frozen (READ_ONLY)")
                diff = dest.execute("SELECT max(abs(n.score-f.score_anchor)) FROM predictions n JOIN frozen.predictions f USING(customer_id,article_id)").fetchone()[0]
                assert diff <= 1e-12, (window,diff)
                old_top = dest.execute("""SELECT customer_id,article_id,final_rank FROM (
                    SELECT customer_id,article_id,row_number() OVER(PARTITION BY customer_id ORDER BY
                    CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
                    CASE WHEN user_history_events_12w>0 THEN score_anchor END DESC NULLS LAST,
                    candidate_rank,article_id)::INTEGER AS final_rank FROM frozen.predictions
                ) WHERE final_rank<=12 ORDER BY customer_id,final_rank""").fetchdf()
                pd.testing.assert_frame_equal(top[["customer_id","article_id","final_rank"]],old_top,check_dtype=False)
                reference = old["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]
                assert abs(result["map@12"]-reference["map@12"]) < 1e-12
                assert abs(result["candidate_recall"]-reference["candidate_recall@expanded_pool"]) < 1e-12
                result["parity"] = {"max_abs_prediction_error": float(diff), "top12_exact": True,"MAP_abs_error": abs(result["map@12"]-reference["map@12"]),"denominator_exact": result['users']==reference['users'] and result['truth_pairs']==reference['truth_pairs']}
            destination = root/"top12.parquet"
            dest.execute(f"COPY (SELECT * FROM top12 ORDER BY customer_id,final_rank) TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD)")
            result["top12"] = evidence_id(destination,reason="explicit_registry_evidence")
            return result
        finally:
            source.close()
            dest.close()
