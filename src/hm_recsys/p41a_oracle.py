"""Enumerate a frozen, one-item, same-slot diagnostic oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .p41a_contract import BUDGETS, SLOT_SETS
from .p41a_stats import LCM, ap_units, distribution, replacement_units


def opportunities(data):
    cold = data["cold"]
    proposals = cold.loc[(cold.cold_rank <= 10) & ~cold.already_w0].copy()
    frames = []
    warm = data["warm"]
    for slot in (10, 11, 12):
        frame = proposals.copy()
        ui = frame.user_index.to_numpy()
        relevance = data["relevance"][ui]
        delta = replacement_units(relevance, frame.target.to_numpy(), slot)
        reference = relevance.copy()
        reference[:, slot-1] = frame.target
        # Independent full-list recomputation, for every legal opportunity row.
        assert np.array_equal(delta, ap_units(reference)-ap_units(relevance))
        frame["cutoff"] = data["cutoff"]
        frame["warm_slot_rank"] = slot
        boundary = warm.loc[warm.warm_rank == slot].reset_index(drop=True)
        frame["warm_article_id"] = boundary.article_id.to_numpy()[ui]
        frame["warm_item_score"] = boundary.warm_model_score.to_numpy()[ui]
        frame["warm_score_available"] = boundary.warm_model_score_available.to_numpy()[ui]
        frame.loc[frame.warm_score_available == 0, "warm_item_score"] = np.nan
        frame["warm_rank_pct"] = boundary.warm_rank_pct.to_numpy()[ui]
        frame["warm_target"] = boundary.target.to_numpy()[ui]
        frame["delta_ap_units"] = delta
        frame["ap_denominator_units"] = data["denominator"][ui]
        frame["delta_AP12_if_replaced"] = delta/data["denominator"][ui]
        frame["beneficial"] = delta > 0
        frame["opportunity_label"] = np.where(delta > 0, "beneficial", np.where(delta < 0, "harmful", "neutral"))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["customer_id", "cold_rank", "warm_slot_rank"], ignore_index=True)


def best_choices(positive_rows, budget, slots):
    candidates = positive_rows.loc[(positive_rows.cold_rank <= budget) & positive_rows.warm_slot_rank.isin(slots)]
    # Denominator is constant within user, hence integer units choose exact maximum.
    return candidates.sort_values(["user_index", "delta_ap_units", "cold_rank", "warm_slot_rank", "article_id"],
        ascending=[True, False, True, True, True], kind="stable").drop_duplicates("user_index").sort_values("user_index")


def segment_metrics(data, selected, final_relevance, final_items, delta):
    truth = data["truth"]
    out = {}
    n = len(data["users"])
    for name in ("strict_cold", "sparse1_5", "all_cold_sparse"):
        counts = truth.interaction_count_before_cutoff
        mask = (counts == 0) if name == "strict_cold" else ((counts >= 1) & (counts <= 5)) if name == "sparse1_5" else (counts <= 5)
        truthgroup = {u: set(g.article_id) for u, g in truth.loc[mask].groupby("customer_id", sort=False)}
        nr = np.array([len(truthgroup.get(u, set())) for u in data["users"]])
        valid = nr > 0
        base_r = np.array([[i in truthgroup.get(u, set()) for i in row] for u, row in zip(data["users"], data["warm_lists"])], dtype=np.int64)
        after_r = np.array([[i in truthgroup.get(u, set()) for i in row] for u, row in zip(data["users"], final_items)], dtype=np.int64)
        denom = np.maximum(1, np.minimum(nr, 12))*LCM
        base = float(np.mean((ap_units(base_r)/denom)[valid])) if valid.any() else 0.0
        after = float(np.mean((ap_units(after_r)/denom)[valid])) if valid.any() else 0.0
        selmask = selected.strict_cold_flag.eq(1) if name == "strict_cold" else selected.sparse1_5_flag.eq(1) if name == "sparse1_5" else np.ones(len(selected), bool)
        sel = selected.loc[selmask]
        num = len(sel)
        out[name] = {"map@12": after, "w0_map@12": base, "delta_MAP_vs_W0": after-base,
            "truth_users": int(valid.sum()), "truth_pairs": int(nr.sum()),
            "positive_opportunity_users": num, "positive_opportunity_user_share": num/n,
            "no_admission_users": n-num, "no_admission_share": 1-num/n,
            "inserted_cold_positive_pairs": num, "removed_warm_positive_pairs": 0,
            "net_positive_pairs": num, "overall_delta_contribution": float(delta[sel.user_index.to_numpy()].sum()/n),
            "count_semantics": "same overall-optimal policy; this segment selected versus not selected; all W0 users denominator; subgroup MAP only users with segment truth"}
    return out


def oracle(data):
    n = len(data["users"])
    cold = data["cold"]
    # Nonpositive moves cannot beat mandatory no-admission; enumerate every positive
    # move across all 12 slots. Primary still materializes ALL label classes separately.
    eligible = cold.loc[~cold.already_w0 & cold.target.eq(1)]
    positives = []
    for slot in range(1, 13):
        frame = eligible.copy()
        units = replacement_units(data["relevance"][frame.user_index], frame.target, slot)
        frame["warm_slot_rank"] = slot
        frame["delta_ap_units"] = units
        positives.append(frame.loc[units > 0])
    positive_rows = pd.concat(positives, ignore_index=True)
    cells, lists, selections = {}, {}, []
    for budget in BUDGETS:
        for scope, slots in SLOT_SETS.items():
            cell = f"top{budget}_{scope}"
            selected = best_choices(positive_rows, budget, slots)
            ui = selected.user_index.to_numpy(dtype=int)
            js = selected.warm_slot_rank.to_numpy(dtype=int)-1
            result = data["warm_lists"].copy()
            result[ui, js] = selected.article_id
            new_r = data["relevance"].copy()
            new_r[ui, js] = selected.target
            delta_units = ap_units(new_r)-ap_units(data["relevance"])
            delta = delta_units/data["denominator"]
            assert (delta >= 0).all()
            changes = (result != data["warm_lists"])
            assert np.array_equal(changes.sum(axis=1), delta > 0)
            assert all(len(set(row)) == 12 for row in result)
            assert np.array_equal(result[delta == 0], data["warm_lists"][delta == 0])
            if scope == "slots10_12":
                assert np.array_equal(result[:, :9], data["warm_lists"][:, :9])
            assert np.array_equal(delta_units[ui], selected.delta_ap_units)
            assert not data["relevance"][ui, js].any()
            num = len(selected)
            summary = {"cold_budget": budget, "warm_slots": list(slots), "max_admissions": 1,
                "users": n, "map@12": float(np.mean(data["baseline_ap"]+delta)),
                "w0_map@12": data["baseline_map"], "delta_MAP_vs_W0": float(delta.mean()),
                "positive_opportunity_users": num, "positive_opportunity_user_share": num/n,
                "no_admission_users": n-num, "no_admission_share": 1-num/n,
                "inserted_cold_positive_pairs": num, "removed_warm_positive_pairs": 0,
                "removed_21plus_event_positive_pairs": 0, "net_positive_pairs": num,
                "segments": segment_metrics(data, selected, new_r, result, delta)}
            cells[cell] = summary
            evidence = data["userframe"].copy()
            evidence["cell"] = cell
            evidence["cutoff"] = data["cutoff"]
            evidence["admitted"] = delta > 0
            evidence["ap_W0"] = data["baseline_ap"]
            evidence["ap_after"] = data["baseline_ap"]+delta
            evidence["delta_AP12"] = delta
            evidence["delta_ap_units"] = delta_units
            evidence["selected_cold_rank"] = 0
            evidence["replaced_warm_rank"] = 0
            evidence["selected_article_id"] = ""
            evidence["strict_cold_flag"] = 0
            evidence["sparse1_5_flag"] = 0
            for outcol, incol in (("selected_cold_rank", "cold_rank"), ("replaced_warm_rank", "warm_slot_rank"),
                                  ("selected_article_id", "article_id"), ("strict_cold_flag", "strict_cold_flag"), ("sparse1_5_flag", "sparse1_5_flag")):
                evidence.loc[ui, outcol] = selected[incol].to_numpy()
            selections.append(evidence)
            # All frontier result lists retained as article-id integers (leading zeros
            # restored to 10 digits on read); IDs, not catalog-dependent row indices.
            lists[cell] = result.astype(np.int64)
            if budget == 10 and scope == "slots10_12":
                primary = {**summary, "name": "A_primary", "selected_cold_rank": distribution(selected.cold_rank),
                    "replaced_warm_rank": distribution(selected.warm_slot_rank),
                    "selected_cold_rank_counts": {str(k): int(v) for k, v in selected.cold_rank.value_counts().sort_index().items()},
                    "replaced_warm_rank_counts": {str(k): int(v) for k, v in selected.warm_slot_rank.value_counts().sort_index().items()},
                    "per_user_delta_AP": distribution(delta), "admitted_user_delta_AP": distribution(delta[delta > 0])}
    # Frontier cannot worsen when either allowable action set is expanded.
    for scope in SLOT_SETS:
        assert np.all(np.diff([cells[f"top{b}_{scope}"]["delta_MAP_vs_W0"] for b in BUDGETS]) >= 0)
    for b in BUDGETS:
        assert np.all(np.diff([cells[f"top{b}_{s}"]["delta_MAP_vs_W0"] for s in SLOT_SETS]) >= 0)
    return primary, cells, pd.concat(selections, ignore_index=True), lists
