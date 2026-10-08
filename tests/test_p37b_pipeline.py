from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from hm_recsys.p37a_audit import _rank_by_score, _score_universe_for_frozen_sets
from hm_recsys.p37b import (
    K0,
    _candidate_identity_row,
    _ordering_metrics_fast,
    _rank_hybrid,
)


class P37BPipelineTests(unittest.TestCase):
    def _fixture(self):
        users = np.repeat(np.arange(2, dtype=np.int32), K0)
        items = np.tile(np.arange(K0, dtype=np.int32), 2)
        rank = np.tile(np.arange(1, K0 + 1, dtype=np.uint16), 2)
        target = np.zeros(2 * K0, dtype=np.uint8)
        target[[0, 210]] = 1
        candidates = {
            "user_index": users,
            "catalog_row": items,
            "rank": rank,
            "coarse_score": (1.0 - rank.astype(np.float32) / K0),
            "target": target,
        }
        truth = {0: {0}, 1: {10}}
        counts = np.zeros(K0, dtype=np.int32)
        return candidates, truth, counts

    def test_fixed_candidate_metrics_conserve_top200_under_any_rank_permutation(self) -> None:
        candidates, truth, counts = self._fixture()
        c0 = candidates["rank"]
        reversed_rank = np.tile(np.arange(K0, 0, -1, dtype=np.uint16), 2)
        metrics = {
            "C0": _ordering_metrics_fast(
                candidates=candidates,
                ordering_rank=c0,
                truth=truth,
                counts=counts,
                user_count=2,
            ),
            "C1": _ordering_metrics_fast(
                candidates=candidates,
                ordering_rank=reversed_rank,
                truth=truth,
                counts=counts,
                user_count=2,
            ),
        }
        metrics["B0"] = metrics["C0"]
        metrics["B1"] = metrics["C1"]
        frozen = SimpleNamespace(
            candidates=candidates,
            identity_audit={"candidate_identity_sha256": "frozen"},
        )
        audit = _candidate_identity_row(
            frozen,
            metrics,
            {"C0": c0, "C1": reversed_rank, "B0": c0, "B1": reversed_rank},
        )
        self.assertTrue(audit["all_variants_exact"])
        self.assertEqual(
            {row["top200_truth_pairs"] for row in audit["variants"].values()}, {2}
        )

    def test_hybrid_ties_preserve_frozen_m4_rank(self) -> None:
        candidates, _truth, _counts = self._fixture()
        scores = np.ones(2 * K0, dtype=np.float32)
        observed = _rank_hybrid(candidates, scores)
        np.testing.assert_array_equal(observed, candidates["rank"])

    def test_p37a_full_universe_kernel_exactly_reproduces_static_control(self) -> None:
        rng = np.random.default_rng(37)
        embeddings = rng.normal(size=(205, 128)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        history_rows = np.full((1, 20), -1, dtype=np.int32)
        history_rows[0, 0] = 204
        history_days = np.zeros((1, 20), dtype=np.float32)
        history_days[0, 0] = 1
        history_mask = (history_rows >= 0).astype(np.uint8)
        universe = np.arange(205, dtype=np.int32)
        raw = embeddings @ embeddings[204]
        raw *= np.float32(0.5 ** (1.0 / 28.0))
        order = sorted(range(205), key=lambda item: (-float(raw[item]), item))[:K0]
        items = np.asarray(order, dtype=np.int32)
        coarse = raw[items].astype(np.float32)
        candidates = {
            "user_index": np.zeros(K0, dtype=np.int32),
            "catalog_row": items,
            "rank": np.arange(1, K0 + 1, dtype=np.uint16),
            "coarse_score": coarse,
            "target": np.zeros(K0, dtype=np.uint8),
        }
        outputs, audit = _score_universe_for_frozen_sets(
            {"R_M": candidates, "R_G": candidates},
            {
                "catalog_row": history_rows,
                "days_since_purchase": history_days,
                "mask": history_mask,
            },
            embeddings,
            universe,
            authoritative_set="R_G",
            device=torch.device("cpu"),
        )
        self.assertTrue(audit["candidate_identity_exact_parity"])
        self.assertTrue(audit["passed"])
        rank = _rank_by_score(
            candidates["user_index"], candidates["catalog_row"], outputs["R_M"]
        )
        np.testing.assert_array_equal(rank, candidates["rank"])


if __name__ == "__main__":
    unittest.main()
