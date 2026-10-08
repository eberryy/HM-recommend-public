"""Read frozen P4.0 assets; verify identity without importing model code."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .metrics import apk
from .p41a_contract import check_identity, guard_cutoff, identity
from .p41a_stats import LCM, ap_units, normalize_cold50


def identities(tree):
    if isinstance(tree, dict):
        if {"path", "bytes", "sha256"} <= tree.keys():
            yield tree
        else:
            for value in tree.values():
                yield from identities(value)
    elif isinstance(tree, list):
        for value in tree:
            yield from identities(value)


def verify_inputs(contract, include_authorities=True):
    tree = {k: v for k, v in contract.items() if include_authorities or k != "authoritative_inputs"}
    inputs = {entry["path"]: entry for entry in identities(tree)}
    for entry in inputs.values():
        check_identity(entry)
    return {"files": len(inputs), "bytes": sum(e["bytes"] for e in inputs.values()), "all_sha_pass": True}


def parquet(path, connection=None):
    con = connection or duckdb.connect()
    try:
        return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    finally:
        if connection is None:
            con.close()


def save_frame(frame, path, cutoff, lineage):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.register("output_frame", frame)
        escaped = str(path).replace("'", "''")
        con.execute(f"COPY output_frame TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        con.close()
    return {**identity(path), "row_count": len(frame), "cutoff": cutoff, "source_lineage": lineage}


def load_window(contract, window):
    a = contract["frozen_inputs"][window]
    cutoff = guard_cutoff(a["cutoff"])
    b0 = a["b0_lineage"]
    if b0["available"]:
        assert b0["lineage_safe"] and b0["model_label_end"] <= cutoff
        assert all(x < cutoff for x in b0["model_training_cutoffs"])
        assert "B0" in b0["model"]["path"]
    warmline = a["warm_lineage"]
    if warmline["available"]:
        assert warmline["safe"] and warmline["latest_training_label_end"] <= cutoff
    with duckdb.connect(a["w0_database"]["path"], read_only=True) as db:
        warm = db.execute("""
          SELECT customer_id,article_id,warm_rank FROM (
            SELECT customer_id,article_id,row_number() OVER(
              PARTITION BY customer_id ORDER BY
              CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
              CASE WHEN user_history_events_12w>0 THEN score_anchor END DESC NULLS LAST,
              candidate_rank,article_id)::INTEGER AS warm_rank FROM predictions
          ) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank
        """).fetchdf()
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        frozen_warm = con.execute("""SELECT customer_id,article_id,warm_rank,warm_rank_pct,
            warm_model_score,warm_model_score_available FROM read_parquet(?)
            WHERE warm_rank<=12 ORDER BY customer_id,warm_rank""", [a["warm150"]["path"]]).fetchdf()
        pd.testing.assert_frame_equal(warm, frozen_warm[list(warm.columns)], check_dtype=False)
        warm = frozen_warm
        users = warm.customer_id.drop_duplicates().to_numpy()
        assert len(users) == a["w0_users"] and len(warm) == 12*len(users)
        assert np.array_equal(warm.warm_rank.to_numpy().reshape(-1, 12), np.tile(np.arange(1, 13), (len(users), 1)))
        assert not warm.duplicated(["customer_id", "article_id"]).any()
        user_frame = pd.DataFrame({"customer_id": users})
        con.register("eval_users", user_frame)
        end = (date.fromisoformat(cutoff)+timedelta(days=7)).isoformat()
        truth = con.execute("""SELECT DISTINCT customer_id,article_id FROM read_parquet(?)
          JOIN eval_users USING(customer_id) WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)
          ORDER BY customer_id,article_id""", [contract["transactions"]["path"], cutoff, end]).fetchdf()
        counts = con.execute("""SELECT article_id,count(*) AS n FROM read_parquet(?)
          WHERE t_dat<CAST(? AS DATE) GROUP BY article_id""", [contract["transactions"]["path"], cutoff]).fetchdf()
        cold = parquet(a["cold50"]["path"], con).sort_values(["customer_id", "cold_rank", "article_id"]).reset_index(drop=True)
        con.register("cold_ids", cold[["customer_id", "article_id"]])
        features = con.execute("""SELECT f.customer_id,f.article_id,f.target,
          f.interaction_count_before_cutoff,f.strict_cold_flag,f.sparse1_5_flag
          FROM read_parquet(?) f JOIN cold_ids USING(customer_id,article_id)""", [a["features"]["path"]]).fetchdf()
    truthsets = {u: set(g.article_id) for u, g in truth.groupby("customer_id", sort=False)}
    assert set(truthsets) == set(users)
    countmap = dict(zip(counts.article_id, counts.n.astype(int)))
    truth["interaction_count_before_cutoff"] = truth.article_id.map(countmap).fillna(0).astype(np.int64)
    warm["target"] = [int(i in truthsets[u]) for u, i in zip(warm.customer_id, warm.article_id)]
    warm["interaction_count_before_cutoff"] = warm.article_id.map(countmap).fillna(0).astype(np.int64)
    r = warm.target.to_numpy().reshape(-1, 12)
    ntruth = np.array([len(truthsets[u]) for u in users])
    denom = np.minimum(ntruth, 12)*LCM
    lists = warm.article_id.to_numpy().reshape(-1, 12)
    reference_ap = np.array([apk(list(truthsets[u]), list(items)) for u, items in zip(users, lists)])
    original_map = float(reference_ap.mean())
    # The reference evaluator and ordering are reproduced, not just rounded MAP.
    assert original_map == a["w0_map"], (window, original_map, a["w0_map"], original_map-a["w0_map"])
    assert np.allclose(reference_ap, ap_units(r)/denom, rtol=0, atol=2e-16)
    assert len(features) == len(cold) and not features.duplicated(["customer_id", "article_id"]).any()
    cold = cold.merge(features, on=["customer_id", "article_id"], validate="one_to_one", sort=False)
    computed_target = np.array([int(i in truthsets[u]) for u, i in zip(cold.customer_id, cold.article_id)])
    computed_count = cold.article_id.map(countmap).fillna(0).astype(np.int64).to_numpy()
    assert np.array_equal(computed_target, cold.target)
    assert np.array_equal(computed_count, cold.interaction_count_before_cutoff)
    assert np.array_equal(computed_count == 0, cold.strict_cold_flag.astype(bool))
    assert np.array_equal((computed_count >= 1) & (computed_count <= 5), cold.sparse1_5_flag.astype(bool))
    assert (computed_count <= 5).all()
    source_users = pd.read_csv(a["p31_assets"]["users"]["path"], dtype={"customer_id": str}).customer_id.to_numpy()
    catalog = pd.read_csv(contract["catalog"]["path"], dtype={"article_id": str}).sort_values("catalog_row").article_id.to_numpy()
    with np.load(a["p31_assets"]["candidates"]["path"]) as candidates:
        ranks = np.load(b0["rank_artifact"]["path"]) if b0["available"] else candidates["rank"]
        scores = np.load(b0["score_artifact"]["path"]) if b0["available"] else np.full(len(ranks), np.nan)
        keep = ranks <= 50
        expected = pd.DataFrame({"customer_id": source_users[candidates["user_index"][keep]],
            "article_id": catalog[candidates["catalog_row"][keep]], "cold_rank": ranks[keep],
            "b0_score": scores[keep], "m4_coarse_rank": candidates["rank"][keep],
            "m4_coarse_score": candidates["coarse_score"][keep], "target": candidates["target"][keep]})
    expected.sort_values(["customer_id", "cold_rank", "article_id"], inplace=True, ignore_index=True)
    pd.testing.assert_frame_equal(cold[list(expected.columns)], expected, check_dtype=False, check_exact=True)
    assert not cold.duplicated(["customer_id", "article_id"]).any()
    assert (cold.groupby("customer_id").size() == 50).all()
    if b0["available"]:
        assert np.array_equal(cold.b0_rank, cold.cold_rank)
        assert np.array_equal(cold.b0_rank_pct, cold.b0_rank.to_numpy(dtype=np.float32)/50)
        assert np.array_equal(cold.b0_delta_vs_m4, cold.b0_score-cold.m4_coarse_score)
        assert cold.b0_score_available.eq(1).all() and cold.b0_delta_available.eq(1).all()
    else:
        assert cold.b0_score_available.eq(0).all() and cold.b0_delta_available.eq(0).all()
    state = np.load(a["user_state"]["path"])
    assert len(state) == len(source_users) and len(set(source_users)) == len(source_users)
    stateframe = pd.DataFrame(state, columns=a["user_state_columns"], index=source_users).loc[users]
    userframe = pd.DataFrame({"customer_id": users,
        "recent_active": (stateframe.history_count_0_7.to_numpy()+stateframe.history_count_8_28.to_numpy()) > 0,
        "profile_available": stateframe.recent_vs_older_profile_available.to_numpy() == 1})
    cold = normalize_cold50(cold)
    user_index = {u: i for i, u in enumerate(users)}
    cold["user_index"] = cold.customer_id.map(user_index).astype(np.int32)
    warmsets = {u: set(items) for u, items in zip(users, lists)}
    cold["already_w0"] = [i in warmsets[u] for u, i in zip(cold.customer_id, cold.article_id)]
    return {"window": window, "cutoff": cutoff, "users": users, "userframe": userframe,
        "cold": cold, "warm": warm, "warm_lists": lists, "truth": truth, "truthsets": truthsets,
        "relevance": r, "denominator": denom, "baseline_ap": reference_ap, "baseline_map": original_map,
        "parity": {"w0_identity_order_exact": True, "w0_map_bit_exact": True,
            "w0_map": original_map, "w0_map_hex": original_map.hex(), "b0_cold50_exact": True,
            "m4_top200_source_exact": True, "b0_lineage_safe": True, "warm_score_lineage_safe": True,
            "future_labels_and_full_history_counts_match_p40": True, "cold50_rows": len(cold),
            "cold50_users": int(cold.customer_id.nunique()), "all_users": len(users)}}
