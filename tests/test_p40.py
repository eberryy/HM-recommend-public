from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

import hm_recsys.m55 as m55
import hm_recsys.p40 as p40
from hm_recsys.m4_contract import FINAL_CUTOFF
from hm_recsys.p40 import B0_LINEAGE, _boundary_sample_one
from hm_recsys.p40_contract import FUSION_FEATURES, K_COLD, K_FINAL, K_WARM, SOURCE_FEATURES


def _frame() -> pd.DataFrame:
    rows = []
    for rank in range(1, 151):
        rows.append({"target_cutoff": "2020-01-01", "customer_id": "u", "article_id": f"w{rank}",
                     "target": int(rank == 1), "warm_present": 1, "cold_present": 0,
                     "warm_rank": rank, "cold_rank": None})
    for rank in range(1, 51):
        rows.append({"target_cutoff": "2020-01-01", "customer_id": "u", "article_id": f"c{rank}",
                     "target": int(rank == 2), "warm_present": 0, "cold_present": 1,
                     "warm_rank": None, "cold_rank": rank})
    return pd.DataFrame(rows)


def test_frozen_budgets() -> None:
    assert (K_WARM, K_COLD, K_FINAL) == (150, 50, 12)


def test_boundary_sampler_is_deterministic_and_bounded() -> None:
    frame = _frame()
    first, audit = _boundary_sample_one(frame)
    second, _ = _boundary_sample_one(frame)
    assert first.tolist() == second.tolist()
    assert audit["positive_rows"] == 2
    assert audit["cold_capable_positive_rows"] == 1
    assert audit["warm_only_positive_rows"] == 1
    assert audit["negative_rows"] <= 60
    assert len(set(first.tolist())) == len(first)
    assert audit["fills"]["cold_positive_warm_hard"] <= 12
    assert audit["fills"]["cold_positive_warm_boundary"] <= 12
    assert audit["fills"]["cold_positive_cold"] <= 6
    assert audit["fills"]["warm_positive_cold_hard"] <= 12
    assert audit["fills"]["warm_positive_cold_tail"] <= 6
    assert audit["fills"]["warm_positive_warm"] <= 12


def test_boundary_sampler_excludes_zero_positive_group() -> None:
    frame = _frame()
    frame["target"] = 0
    rows, audit = _boundary_sample_one(frame)
    assert len(rows) == 0
    assert audit["positive_rows"] == 0


def test_f0_reuses_the_exact_m55_sampler() -> None:
    assert p40.load_training_sample is m55.load_training_sample


def test_b0_lineage_is_forward_only_and_final_week_is_absent() -> None:
    assert FINAL_CUTOFF == "2020-09-16"
    assert FINAL_CUTOFF not in B0_LINEAGE
    assert B0_LINEAGE[min(B0_LINEAGE)] is None
    for score_cutoff, lineage in B0_LINEAGE.items():
        if lineage is None:
            continue
        _window, _kind, training_cutoffs = lineage
        for training_cutoff in training_cutoffs:
            assert training_cutoff < score_cutoff
            label_end = datetime.strptime(training_cutoff, "%Y-%m-%d") + timedelta(days=7)
            assert label_end.date() <= datetime.strptime(score_cutoff, "%Y-%m-%d").date()


def test_fusion_feature_contract_keeps_minimal_b0_summary_only() -> None:
    assert "coldness_bucket" in FUSION_FEATURES
    assert set(SOURCE_FEATURES).issubset(FUSION_FEATURES)
    forbidden = {
        "alpha_0_7", "alpha_8_28", "alpha_29_84", "alpha_over_84", "alpha_null",
        "per_age_max_cosine", "per_age_top3_cosine", "profile_stability",
        "B0_relation_hidden_representation",
    }
    assert forbidden.isdisjoint(FUSION_FEATURES)
