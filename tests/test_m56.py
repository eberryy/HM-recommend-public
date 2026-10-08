from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from hm_recsys.m55 import DUAL_FEATURES, NEGATIVES_PER_POSITIVE, WARM_FEATURES, load_training_sample
from hm_recsys.m56 import (
    FIXED_ROUNDS,
    NO_COLD_POSITIVE_COLD_NEGATIVE_CAP,
    SOURCE_FEATURE_DEFINITIONS,
    SOURCE_FEATURES,
    VARIANTS,
    fixed40_training_contract,
    load_conditional_training_sample,
    validate_development_cutoff,
)


FEATURES = [
    "warm_present", "cold_present", "source_branch", "warm_rank", "warm_rank_pct",
    "cold_rank_pct",
]


def _row(
    customer: str, article: int, target: int, branch: str, rank: int,
    warm_rank: int | None = None,
) -> dict[str, object]:
    warm = branch in {"warm_only", "warm_and_cold"}
    cold = branch in {"cold_only", "warm_and_cold"}
    return {
        "target_cutoff": "2020-01-01",
        "customer_id": customer,
        "article_id": article,
        "candidate_rank": rank,
        "target": target,
        "source_branch": branch,
        "warm_present": int(warm),
        "cold_present": int(cold),
        "warm_rank": warm_rank if warm else None,
        "warm_rank_pct": (rank / 150) if warm else None,
        "cold_rank_pct": (rank / 50) if cold else None,
    }


def _fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # A cold-positive group with enough rows in every reserved source stratum.
    rows.append(_row("cold-positive", 1, 1, "cold_only", 1))
    for i in range(2, 42):
        rows.append(_row("cold-positive", i, 0, "cold_only", i))
    for i in range(42, 57):
        rows.append(_row("cold-positive", i, 0, "warm_only", i, warm_rank=8 + (i % 13)))
    for i in range(57, 72):
        rows.append(_row("cold-positive", i, 0, "warm_only", i, warm_rank=30 + i))
    for i in range(72, 87):
        rows.append(_row("cold-positive", i, 0, "warm_and_cold", i, warm_rank=i))

    # A group with cold candidates but no cold positive.
    rows.append(_row("no-cold-positive", 100, 1, "warm_only", 1, warm_rank=1))
    for i in range(101, 116):
        rows.append(_row("no-cold-positive", i, 0, "cold_only", i - 99))
    for i in range(116, 156):
        rows.append(_row("no-cold-positive", i, 0, "warm_only", i - 99, warm_rank=i - 99))

    # A warm-only group: conditional sampling must equal the frozen M5.5 sampler.
    rows.append(_row("warm-only", 200, 1, "warm_only", 1, warm_rank=1))
    for i in range(201, 246):
        rows.append(_row("warm-only", i, 0, "warm_only", i - 199, warm_rank=i - 199))
    return pd.DataFrame(rows)


class M56Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "features.parquet"
        frame = _fixture()
        connection = duckdb.connect()
        try:
            connection.register("fixture", frame)
            connection.execute(f"COPY fixture TO '{self.path.as_posix()}' (FORMAT PARQUET)")
        finally:
            connection.close()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _conditional(self) -> pd.DataFrame:
        frame, _, _ = load_conditional_training_sample(paths=[self.path], features=FEATURES)
        return frame

    def test_no_cold_positive_group_caps_cold_negatives(self) -> None:
        frame = self._conditional()
        group = frame.loc[frame["customer_id"] == "no-cold-positive"]
        cold_negatives = group.loc[
            (group["target"] == 0) & group["source_branch"].isin(["cold_only", "warm_and_cold"])
        ]
        self.assertLessEqual(len(cold_negatives), NO_COLD_POSITIVE_COLD_NEGATIVE_CAP)

    def test_cold_positive_group_preserves_cross_source_contrast(self) -> None:
        frame = self._conditional()
        negative = frame.loc[
            (frame["customer_id"] == "cold-positive") & (frame["target"] == 0)
        ]
        self.assertTrue({"cold_only", "warm_only", "warm_and_cold"}.issubset(set(negative["source_branch"])))
        boundary = negative.loc[
            (negative["source_branch"] == "warm_only") & negative["warm_rank"].between(8, 20)
        ]
        self.assertGreater(len(boundary), 0)

    def test_total_negative_cap_is_never_exceeded(self) -> None:
        frame = self._conditional()
        groups = frame.groupby("customer_id")["target"].agg(["size", "sum"])
        self.assertTrue(((groups["size"] - groups["sum"]) <= NEGATIVES_PER_POSITIVE * groups["sum"]).all())

    def test_warm_only_group_matches_frozen_sampler(self) -> None:
        conditional = self._conditional()
        current, _, _ = load_training_sample(paths=[self.path], features=FEATURES, warm_only=False)
        conditional_ids = set(conditional.loc[conditional["customer_id"] == "warm-only", "article_id"])
        current_ids = set(current.loc[current["customer_id"] == "warm-only", "article_id"])
        self.assertEqual(conditional_ids, current_ids)

    def test_feature_and_variant_contracts_are_frozen(self) -> None:
        self.assertEqual(len(SOURCE_FEATURES), 23)
        self.assertEqual(set(SOURCE_FEATURES), set(DUAL_FEATURES) - set(WARM_FEATURES))
        self.assertEqual(set(SOURCE_FEATURE_DEFINITIONS), set(SOURCE_FEATURES))
        self.assertEqual(
            list(VARIANTS),
            [
                "A_current_sampling_fixed40",
                "B_conditional_sampling_temporal_early_stopping",
                "C_conditional_sampling_fixed40",
            ],
        )
        self.assertFalse(VARIANTS["A_current_sampling_fixed40"]["deployment_eligible"])
        self.assertFalse(VARIANTS["B_conditional_sampling_temporal_early_stopping"]["deployment_eligible"])
        self.assertTrue(VARIANTS["C_conditional_sampling_fixed40"]["deployment_eligible"])

    def test_fixed_round_contract_disables_early_stopping(self) -> None:
        contract = fixed40_training_contract()
        self.assertEqual(contract["boost_rounds"], FIXED_ROUNDS)
        self.assertFalse(contract["validation_for_early_stopping"])
        self.assertFalse(contract["early_stopping"])

    def test_final_week_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_development_cutoff("2020-09-16")


if __name__ == "__main__":
    unittest.main()
