"""Exact full-cohort WV3-741 replay for the MIND chronological diagnostic.

The saved model-fusion rank ``rf`` is not the served baseline rank for users
without 84-day history. ``ap_rf`` preserves their original candidate fallback.
Only previously frozen, label-free WV3-741 swaps are applied; this module never
selects actions, fits models, or writes into the Warm research checkout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import mind_warm_ranking as old


def _replay_order(ranks: pd.DataFrame, swaps: pd.DataFrame) -> pd.DataFrame:
    """Apply disjoint saved swaps to complete fallback-aware Top50 lists."""
    required = {"customer_id", "article_id", "rf", "ap_rf", "candidate_rank",
                "user_history_events_12w", "target", "truth_count"}
    if not required.issubset(ranks.columns):
        raise ValueError("baseline rank schema is incomplete")
    keys = ["customer_id", "article_id"]
    if ranks[keys].isna().any().any() or ranks.duplicated(keys).any():
        raise ValueError("baseline identities must be unique and nonnull")
    expected = np.where(ranks.user_history_events_12w == 0,
                        ranks.candidate_rank, ranks.rf)
    if not np.array_equal(expected, ranks.ap_rf.to_numpy()):
        raise ValueError("ap_rf does not match the registered inactive fallback")
    result = ranks.loc[ranks.ap_rf.between(1, 50)].copy()
    sizes = result.groupby("customer_id").size()
    if not len(sizes) or not (sizes == 50).all():
        raise ValueError("each full-cohort user must retain exactly fifty baseline items")
    if result.duplicated(["customer_id", "ap_rf"]).any():
        raise ValueError("baseline ranks must be unique per user")
    result["model_rf"] = result.rf.astype(np.int32)
    # Compatibility with the old exact-AP helpers: rf is the pre-swap list rank.
    result["rf"] = result.ap_rf.astype(np.int32)
    result["champion_rank"] = result.rf
    required_swaps = {"customer_id", "challenger_article_id", "victim_article_id",
                      "challenger_rank", "victim_rank"}
    if not required_swaps.issubset(swaps.columns):
        raise ValueError("saved swap schema is incomplete")
    if not swaps.victim_rank.between(8, 12).all() or not swaps.challenger_rank.between(13, 50).all():
        raise ValueError("saved swaps violate the protected-head policy")
    if not swaps.empty and swaps.groupby("customer_id").size().max() > 2:
        raise ValueError("WV3-741 permits no more than two saved swaps per user")
    changes = []
    for role, other in (("challenger", "victim"), ("victim", "challenger")):
        selected = swaps[["customer_id", f"{role}_article_id", f"{role}_rank", f"{other}_rank"]].rename(
            columns={f"{role}_article_id": "article_id", f"{role}_rank": "expected_rank",
                     f"{other}_rank": "replacement_rank"})
        checked = selected.merge(result[keys + ["rf", "user_history_events_12w"]],
                                 on=keys, how="left", validate="one_to_one")
        if checked.rf.isna().any() or not (checked.rf == checked.expected_rank).all():
            raise ValueError("saved swap item/rank does not match the frozen baseline")
        if (checked.user_history_events_12w == 0).any():
            raise ValueError("saved swap unexpectedly targets an inactive-fallback user")
        changes.append(selected[keys + ["replacement_rank"]])
    changes = pd.concat(changes, ignore_index=True)
    if changes.duplicated(keys).any():
        raise ValueError("saved swaps are not disjoint")
    result = result.merge(changes, on=keys, how="left", validate="one_to_one")
    result["champion_rank"] = result.replacement_rank.fillna(result.rf).astype(np.int32)
    result = result.drop(columns="replacement_rank")
    if result.duplicated(["customer_id", "champion_rank"]).any():
        raise ValueError("replayed champion positions are not unique")
    if (result.loc[result.rf <= 7, "champion_rank"] != result.loc[result.rf <= 7, "rf"]).any():
        raise ValueError("saved swaps changed protected ranks one through seven")
    return result.sort_values(["customer_id", "champion_rank", "article_id"], kind="mergesort").reset_index(drop=True)


def reconstruct_full_baseline(cutoff: str) -> tuple[pd.DataFrame, dict]:
    """Read and exactly replay the original full-cohort WV3-741 Top50."""
    old._guard(cutoff)
    names = [name for name, value in old.OUTER.items() if value == cutoff]
    if len(names) != 1:
        raise ValueError("only the four registered full-cohort outer cutoffs are supported")
    name = names[0]
    ranks_path = old._warm_rank_path("outer", cutoff)
    swaps_path = old._warm_swap_path("outer", name)
    source_meta = old._read_json(ranks_path.with_name("DATA.json"))
    expected_map, total_users = old._baseline_meta("outer", name)
    if source_meta["role"] != "outer" or source_meta["cutoff"] != cutoff:
        raise ValueError("full-cohort outer rank metadata required")
    if source_meta["included_users"] != total_users or source_meta["total_users"] != total_users:
        raise ValueError("source rank population differs from the frozen all-user denominator")
    with old._connection("4GB") as db:
        ranks = db.execute(
            "SELECT customer_id,article_id,rf,ap_rf,candidate_rank,"
            "user_history_events_12w,target,truth_count FROM read_parquet(?) WHERE ap_rf<=50",
            [str(ranks_path)],
        ).fetchdf()
        swaps = db.execute("SELECT * FROM read_parquet(?)", [str(swaps_path)]).fetchdf()
    result = _replay_order(ranks, swaps)
    if result.customer_id.nunique() != total_users:
        raise ValueError("full-cohort replay silently omitted users")
    before = result.copy()
    before["champion_rank"] = before.rf
    before_map = old._mean_ap(before[before.champion_rank <= 12], total_users)
    if abs(before_map - source_meta["baseline_map_population_component"]) > 1e-12:
        raise AssertionError("fallback-aware pre-swap MAP differs from frozen WV2-601")
    observed = old._mean_ap(result[result.champion_rank <= 12], total_users)
    if abs(observed - expected_map) > 1e-12:
        raise AssertionError(f"WV3-741 full-cohort replay drift at {cutoff}: {observed} vs {expected_map}")
    inactive = result.user_history_events_12w == 0
    evidence = {
        "schema": "mind-warm-full-baseline-v1", "cutoff": cutoff,
        "MAP@12": expected_map, "total_users": total_users,
        "replayed_MAP@12": observed, "baseline_replay_error": abs(observed - expected_map),
        "pre_swap_MAP@12": before_map, "rank_source": str(ranks_path),
        "swap_source": str(swaps_path), "source_role": "outer_full_cohort",
        "replayed_users": int(result.customer_id.nunique()), "replayed_top50_rows": len(result),
        "inactive_fallback_users": int(result.loc[inactive, "customer_id"].nunique()),
        "inactive_rank_changes": int((result.loc[inactive, "champion_rank"] != result.loc[inactive, "candidate_rank"]).sum()),
        "saved_swap_rows": len(swaps), "saved_inactive_swap_rows": 0,
        "baseline_rank_definition": "ap_rf: candidate_rank for no-history users, model RRF rank otherwise",
        "final_week": "not_run",
    }
    return result, evidence
