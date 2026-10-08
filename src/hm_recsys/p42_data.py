"""P4.2 frozen-asset loading and label-separated confidence normalization.

No recommendation model is imported or trained here. Cold historical users and
Warm historical users are intentionally separate populations. Each confidence
normalization uses its complete source pool before admission overlap removal.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .metrics import apk
from .p41a_contract import guard_cutoff
from .p41a_stats import LCM, ap_units, normalize_cold50


STATE_COLUMNS = (
    "history_count_0_7", "history_count_8_28", "history_count_29_84",
    "history_count_over_84", "days_since_last_purchase", "recent_0_28_purchase_share",
)
WARM_DERIVED = ("warm_user_percentile", "warm_user_zscore")


def _path(identity):
    return str(identity["path"] if isinstance(identity, dict) else identity)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def normalize_cold_confidence(cold):
    """Population: full B0 Cold50 per user, including eventual overlap rows."""
    result = normalize_cold50(cold)
    score = result.b0_score.astype(float)
    sd = score.groupby(result.customer_id, sort=False).transform("std", ddof=0)
    result["b0_full50_score_std"] = sd
    for suffix in ("rank2", "rank5", "user_median"):
        key = "normalized_margin_to_" + suffix
        result[key] = (result["margin_to_" + suffix] / sd).where(sd > 1e-12)
        result[key + "_available"] = np.isfinite(result[key]).astype(np.uint8)
    return result


def normalize_warm_confidence(warm):
    """Population: all exact W0 Top12 rows, not Warm150 or paired-only rows."""
    result = warm.copy()
    score = result.warm_model_score.astype(float).where(result.warm_model_score_available.eq(1))
    score = score.where(np.isfinite(score))
    groups = score.groupby(result.customer_id, sort=False)
    count = groups.transform("count")
    std = groups.transform("std", ddof=0)
    result["warm_user_percentile"] = ((groups.rank(method="average") - 1) / (count - 1)).where(count >= 2)
    result["warm_user_zscore"] = ((score - groups.transform("mean")) / std).where(std > 1e-12)
    for key in WARM_DERIVED:
        result[key + "_available"] = np.isfinite(result[key]).astype(np.uint8)
    return result


def validate_lineages(a, cutoff):
    """Verify frozen upstream models, independently of later propensity fits."""
    b0, warm = a["b0_lineage"], a["warm_lineage"]
    for line, safe_key, cutoffs_key, end_key in (
        (b0, "lineage_safe", "model_training_cutoffs", "model_label_end"),
        (warm, "safe", "training_cutoffs", "latest_training_label_end"),
    ):
        if not line.get(safe_key):
            raise ValueError("unsafe frozen upstream lineage")
        if line["available"]:
            training = line[cutoffs_key]
            end = max((date.fromisoformat(c) + timedelta(days=7)).isoformat() for c in training)
            if end != line[end_key] or end > cutoff or any(c >= cutoff for c in training):
                raise ValueError("future or inconsistent frozen upstream lineage")
    if b0["available"]:
        model = _path(b0["model"])
        if "B0" not in Path(model).name or "B1" in Path(model).name:
            raise ValueError("P4.2 requires frozen B0, not another Cold model")


def validate_warm_identity(warm):
    if warm.empty or warm.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("W0 must contain unique user-article pairs")
    group_sizes = warm.groupby("customer_id", sort=False).size()
    if not group_sizes.eq(12).all():
        raise ValueError("W0 must contain exactly 12 articles per user")
    ranks = warm.warm_rank.to_numpy().reshape(-1, 12)
    if not np.array_equal(ranks, np.tile(np.arange(1, 13), (len(group_sizes), 1))):
        raise ValueError("W0 rank order changed")


def _warm_source_order(a):
    source = a.get("w0_database") or a["warm_lineage"].get("score_database")
    if not source:
        if a["warm_lineage"]["available"]:
            raise ValueError("available Warm model requires original score database")
        return None
    with duckdb.connect(_path(source), read_only=True) as db:
        return db.execute("""
          SELECT customer_id,article_id,warm_rank FROM (
            SELECT customer_id,article_id,row_number() OVER(
              PARTITION BY customer_id ORDER BY
              CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
              CASE WHEN user_history_events_12w>0 THEN score_anchor END DESC NULLS LAST,
              candidate_rank,article_id)::INTEGER AS warm_rank FROM predictions
          ) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank
        """).fetchdf()


def _validate_cold_source(cold, a, catalog_identity):
    source_users = pd.read_csv(_path(a["p31_assets"]["users"]), dtype={"customer_id": str}).customer_id.to_numpy()
    catalog = pd.read_csv(_path(catalog_identity), dtype={"article_id": str}).sort_values("catalog_row").article_id.to_numpy()
    line = a["b0_lineage"]
    with np.load(_path(a["p31_assets"]["candidates"]), allow_pickle=False) as candidates:
        ranks = np.load(_path(line["rank_artifact"]), allow_pickle=False) if line["available"] else candidates["rank"]
        scores = np.load(_path(line["score_artifact"]), allow_pickle=False) if line["available"] else np.full(len(ranks), np.nan)
        if len(ranks) != len(candidates["rank"]) or len(scores) != len(ranks):
            raise ValueError("cached B0 arrays no longer align with exact M4 candidates")
        keep = ranks <= 50
        expected = pd.DataFrame({
            "customer_id": source_users[candidates["user_index"][keep]],
            "article_id": catalog[candidates["catalog_row"][keep]],
            "cold_rank": ranks[keep], "b0_score": scores[keep],
            "m4_coarse_rank": candidates["rank"][keep],
            "m4_coarse_score": candidates["coarse_score"][keep],
            "target": candidates["target"][keep],
        }).sort_values(["customer_id", "cold_rank", "article_id"], ignore_index=True)
    pd.testing.assert_frame_equal(cold[list(expected.columns)], expected, check_dtype=False, check_exact=True)
    if cold.duplicated(["customer_id", "article_id"]).any() or not cold.groupby("customer_id").size().eq(50).all():
        raise ValueError("Cold50 identity or budget changed")
    expected_m4_pct = cold.m4_coarse_rank.to_numpy(dtype=np.float32) / 200
    np.testing.assert_array_equal(cold.m4_coarse_rank_pct, expected_m4_pct)
    if line["available"]:
        np.testing.assert_array_equal(cold.b0_rank, cold.cold_rank)
        np.testing.assert_array_equal(cold.b0_rank_pct, cold.b0_rank.to_numpy(dtype=np.float32) / 50)
        np.testing.assert_array_equal(cold.b0_delta_vs_m4, cold.b0_score - cold.m4_coarse_score)
        if not cold.b0_score_available.eq(1).all() or not cold.b0_delta_available.eq(1).all():
            raise ValueError("B0 availability changed")
    elif not cold.b0_score_available.eq(0).all() or not cold.b0_delta_available.eq(0).all():
        raise ValueError("missing-score fallback was replaced with learned evidence")
    return source_users


def _attach_state(cold, a, source_users):
    state = np.load(_path(a["user_state"]), mmap_mode="r", allow_pickle=False)
    columns = a["user_state_columns"]
    if state.shape != (len(source_users), len(columns)) or len(set(source_users)) != len(source_users):
        raise ValueError("P3.7B state and P3.1 users no longer align")
    chosen = [columns.index(c) for c in STATE_COLUMNS]
    values = pd.DataFrame(np.asarray(state[:, chosen]), columns=STATE_COLUMNS)
    values.insert(0, "customer_id", source_users)
    result = cold.merge(values, on="customer_id", how="left", validate="many_to_one", sort=False)
    if not np.isfinite(result[list(STATE_COLUMNS)].to_numpy()).all():
        raise ValueError("Cold50 row lacks frozen cutoff-safe P3.7B state")
    return result


def load_cutoff(contract, cutoff):
    """Read exact frozen candidates; rederive labels only for auditing/evaluation.

    Caller must preregister before invoking on real assets. No imputation,
    standardization, estimator fit, sampling, or admission selection occurs here.
    """
    cutoff = guard_cutoff(cutoff)
    a = contract["inputs"][cutoff]
    if a["cutoff"] != cutoff:
        raise ValueError("input cutoff mismatch")
    validate_lineages(a, cutoff)
    source_order = _warm_source_order(a)
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='4GB'")
        warm = con.execute("SELECT * FROM read_parquet(?) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank", [_path(a["warm150"])]).fetchdf()
        validate_warm_identity(warm)
        if source_order is not None:
            pd.testing.assert_frame_equal(warm[list(source_order.columns)], source_order, check_dtype=False, check_exact=True)
        else:
            np.testing.assert_array_equal(warm.warm_rank, warm.warm_candidate_rank)
        users = warm.customer_id.drop_duplicates().to_numpy()
        cold = con.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id,cold_rank,article_id", [_path(a["cold50"])]).fetchdf()
        # Select only the requested Warm numeric backbone and Cold identity audit
        # columns; never materialize the much wider full union into pandas.
        needed_warm = contract["feature_spec"]["qW"]["numeric"] + contract["feature_spec"]["qW"]["binary"]
        derived = set(WARM_DERIVED) | {x + "_available" for x in WARM_DERIVED}
        extras = [x for x in needed_warm if x not in warm and x not in derived]
        selected = list(dict.fromkeys(["customer_id", "article_id", "target", "interaction_count_before_cutoff", "strict_cold_flag", "sparse1_5_flag", "source_branch", *extras]))
        ids = pd.concat([cold[["customer_id", "article_id"]], warm[["customer_id", "article_id"]]], ignore_index=True).drop_duplicates()
        con.register("wanted_pairs", ids)
        feature = con.execute("SELECT " + ",".join("f." + _quote(x) for x in selected) + " FROM read_parquet(?) f JOIN wanted_pairs USING(customer_id,article_id)", [_path(a["features"])]).fetchdf()
        if len(feature) != len(ids) or feature.duplicated(["customer_id", "article_id"]).any():
            raise ValueError("P4.0 feature backbone lacks exact frozen candidate pairs")
        shared = ["customer_id", "article_id", "target", "interaction_count_before_cutoff", "strict_cold_flag", "sparse1_5_flag", "source_branch"]
        cold = cold.merge(feature[shared], on=["customer_id", "article_id"], validate="one_to_one", sort=False)
        warm = warm.merge(feature, on=["customer_id", "article_id"], validate="one_to_one", sort=False)
        all_users = ids[["customer_id"]].drop_duplicates()
        con.register("wanted_users", all_users)
        end = (date.fromisoformat(cutoff) + timedelta(days=7)).isoformat()
        all_truth = con.execute("""SELECT DISTINCT customer_id,article_id FROM read_parquet(?)
            JOIN wanted_users USING(customer_id) WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)
            ORDER BY customer_id,article_id""", [_path(contract["transactions"]), cutoff, end]).fetchdf()
        counts = con.execute("SELECT article_id,count(*) AS n FROM read_parquet(?) WHERE t_dat<CAST(? AS DATE) GROUP BY article_id", [_path(contract["transactions"]), cutoff]).fetchdf()
    all_truthsets = {u: set(g.article_id) for u, g in all_truth.groupby("customer_id", sort=False)}
    countmap = dict(zip(counts.article_id, counts.n.astype(int)))
    for frame in (cold, warm):
        actual_target = np.array([int(i in all_truthsets.get(u, set())) for u, i in zip(frame.customer_id, frame.article_id)], dtype=np.uint8)
        actual_count = frame.article_id.map(countmap).fillna(0).to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(frame.target, actual_target)
        np.testing.assert_array_equal(frame.interaction_count_before_cutoff, actual_count)
        np.testing.assert_array_equal(frame.strict_cold_flag, actual_count == 0)
        np.testing.assert_array_equal(frame.sparse1_5_flag, (actual_count >= 1) & (actual_count <= 5))
    if not cold.interaction_count_before_cutoff.le(5).all():
        raise ValueError("Cold50 contains a non-cold/sparse item")
    source_users = _validate_cold_source(cold, a, contract["catalog"])
    cold = normalize_cold_confidence(_attach_state(cold, a, source_users))
    warm = normalize_warm_confidence(warm)
    for kind, frame in (("qC", cold), ("qW", warm)):
        requested = contract["feature_spec"][kind]["numeric"] + contract["feature_spec"][kind]["binary"]
        missing = sorted(set(requested) - set(frame.columns))
        if missing:
            raise ValueError(f"{kind} frozen features are missing: {missing}")
    truth = all_truth.loc[all_truth.customer_id.isin(users)].copy()
    truth["interaction_count_before_cutoff"] = truth.article_id.map(countmap).fillna(0).astype(np.int64)
    truthsets = {u: all_truthsets.get(u, set()) for u in users}
    if any(not t for t in truthsets.values()):
        raise ValueError("W0 evaluation population contains a user without future truth")
    lists = warm.article_id.to_numpy().reshape(-1, 12)
    baseline_ap = np.array([apk(list(truthsets[u]), list(items)) for u, items in zip(users, lists)])
    baseline_map = float(baseline_ap.mean())
    expected_map = a.get("w0_map", a.get("reference_map"))
    expected_users = a.get("w0_users", a.get("reference_users"))
    if expected_map is not None and baseline_map != expected_map:
        raise ValueError(f"exact W0 MAP changed: {baseline_map} != {expected_map}")
    if expected_users is not None and len(users) != expected_users:
        raise ValueError("exact W0 user count changed")
    relevance = warm.target.to_numpy().reshape(-1, 12)
    denominator = np.array([min(len(truthsets[u]), 12) * LCM for u in users])
    np.testing.assert_allclose(baseline_ap, ap_units(relevance) / denominator, rtol=0, atol=2e-16)
    index = {u: i for i, u in enumerate(users)}
    cold["user_index"] = cold.customer_id.map(index).fillna(-1).astype(np.int32)
    warmsets = {u: set(items) for u, items in zip(users, lists)}
    cold["already_w0"] = [i in warmsets.get(u, set()) for u, i in zip(cold.customer_id, cold.article_id)]
    return {"cutoff": cutoff, "users": users, "cold": cold, "warm": warm,
            "truth": truth, "truthsets": truthsets, "countmap": countmap,
            "warm_lists": lists, "relevance": relevance, "denominator": denominator,
            "baseline_ap": baseline_ap, "baseline_map": baseline_map,
            "parity": {"w0_identity_order_exact": True, "w0_map_bit_exact": expected_map is not None,
                "w0_map": baseline_map, "w0_map_hex": baseline_map.hex(),
                "b0_cold50_exact": True, "b0_score_available": a["b0_lineage"]["available"],
                "m4_top200_source_exact": True, "b0_lineage_safe": True,
                "warm_score_lineage_safe": True, "future_labels_and_full_history_counts_match_p40": True,
                "all_users": len(users), "cold50_rows": len(cold),
                "cold50_users": int(cold.customer_id.nunique()),
                "historical_cold_users_not_in_warm": int((cold.user_index < 0).groupby(cold.customer_id).any().sum()),
                "history_population_intersection_filter_applied": False,
                "warm_normalization_population": "exact full W0 Top12",
                "cold_normalization_population": "exact full B0 Cold50 before overlap removal"}}
