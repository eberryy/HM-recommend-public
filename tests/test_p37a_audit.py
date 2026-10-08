from __future__ import annotations

import inspect
import unittest

import numpy as np
import torch

from hm_recsys.p37a_audit import (
    K0,
    TRAINING_ALLOWED,
    _candidate_matrices,
    _candidate_overlap,
    _movement,
    _rank_by_score,
    _safe_cutoff,
    _score_candidates,
    _truth_flow,
    _union_oracle,
    candidate_set_verdict,
    scoring_verdict,
)


def _candidate_asset(items: np.ndarray, targets: np.ndarray | None = None) -> dict[str, np.ndarray]:
    rows = len(items)
    if rows != K0:
        raise ValueError("test fixture must have one Top200 group")
    return {
        "user_index": np.zeros(rows, dtype=np.int32),
        "catalog_row": items.astype(np.int32),
        "rank": np.arange(1, K0 + 1, dtype=np.uint16),
        "coarse_score": np.linspace(1.0, 0.0, K0, dtype=np.float32),
        "target": np.zeros(rows, dtype=np.uint8) if targets is None else targets.astype(np.uint8),
    }


class P37AAuditTests(unittest.TestCase):
    def test_final_week_is_rejected(self) -> None:
        _safe_cutoff("2020-08-19")
        with self.assertRaises(RuntimeError):
            _safe_cutoff("2020-09-16")

    def test_no_training_boundary_is_explicit(self) -> None:
        self.assertFalse(TRAINING_ALLOWED)
        source = inspect.getsource(_score_candidates)
        self.assertIn("torch.inference_mode", source)
        self.assertNotIn("optimizer", inspect.signature(_score_candidates).parameters)

    def test_tie_break_uses_ascending_catalog_row(self) -> None:
        users = np.asarray([0, 0, 0], dtype=np.int32)
        items = np.asarray([9, 2, 5], dtype=np.int32)
        scores = np.ones(3, dtype=np.float32)
        ranks = _rank_by_score(users, items, scores)
        self.assertEqual(ranks.tolist(), [3, 1, 2])

    def test_candidate_top200_identity_and_rank_are_required(self) -> None:
        candidates = _candidate_asset(np.arange(K0))
        active, matrix = _candidate_matrices(candidates, user_count=1)
        self.assertEqual(active.tolist(), [0])
        self.assertEqual(matrix.shape, (1, K0))
        candidates["rank"][10] = 99
        with self.assertRaises(RuntimeError):
            _candidate_matrices(candidates, user_count=1)

    def test_decay_formula_uses_max_history_similarity(self) -> None:
        embeddings = np.zeros((K0 + 2, 2), dtype=np.float16)
        embeddings[:K0, 0] = 1.0
        embeddings[K0] = np.asarray([1.0, 0.0], dtype=np.float16)
        embeddings[K0 + 1] = np.asarray([0.0, 1.0], dtype=np.float16)
        candidates = _candidate_asset(np.arange(K0))
        histories = {
            "catalog_row": np.asarray([[K0, K0 + 1] + [-1] * 18], dtype=np.int32),
            "days_since_purchase": np.asarray([[28.0, 0.0] + [0.0] * 18], dtype=np.float32),
            "mask": np.asarray([[1, 1] + [0] * 18], dtype=np.uint8),
        }
        scores = _score_candidates(candidates, histories, embeddings, device=torch.device("cpu"), batch_users=1)
        self.assertTrue(np.allclose(scores, 0.5))

    def test_candidate_overlap_conserves_union_and_segments(self) -> None:
        left = _candidate_asset(np.arange(0, K0))
        right = _candidate_asset(np.arange(100, 100 + K0))
        counts = np.zeros(100 + K0, dtype=np.int32)
        counts[150:] = 3
        overlap = _candidate_overlap(left, right, counts)
        self.assertEqual(overlap["candidate_rows"]["intersection"], 100)
        self.assertEqual(overlap["candidate_rows"]["m4_only"], 100)
        self.assertEqual(overlap["candidate_rows"]["p33_only"], 100)
        self.assertEqual(overlap["candidate_rows"]["union"], 300)
        self.assertTrue(overlap["conservation_passed"])

    def test_truth_flow_categories_are_mutually_exclusive_and_exhaustive(self) -> None:
        left = _candidate_asset(np.arange(0, K0))
        right = _candidate_asset(np.arange(100, 100 + K0))
        truth = {0: {50, 150, 250, 350}}
        counts = np.zeros(351, dtype=np.int32)
        counts[250] = 2
        overlap = _candidate_overlap(left, right, counts)
        flow = _truth_flow(left, right, truth, counts, 1, overlap)
        self.assertEqual(flow["strict_cold"]["in_both"], 1)
        self.assertEqual(flow["strict_cold"]["only_in_R_M"], 1)
        self.assertEqual(flow["sparse_1_5"]["only_in_R_G"], 1)
        self.assertEqual(flow["strict_cold"]["in_neither"], 1)
        self.assertEqual(flow["cold_universe"]["truth_pairs"], 4)

    def test_union_identity_and_truth_are_conserved(self) -> None:
        left = _candidate_asset(np.arange(0, K0))
        right = _candidate_asset(np.arange(100, 100 + K0))
        truth = {0: {50, 150, 250}}
        counts = np.zeros(300, dtype=np.int32)
        overlap = _candidate_overlap(left, right, counts)
        flow = _truth_flow(left, right, truth, counts, 1, overlap)
        union = _union_oracle(left, right, truth, counts, 1, overlap, flow)
        self.assertEqual(union["candidate_rows"]["union"], 300)
        self.assertEqual(union["segments"]["strict_cold"]["covered_truth_pairs"], 3)
        self.assertTrue(union["identity_conservation_passed"])

    def test_rank_movement_counts_top20_replacements(self) -> None:
        targets = np.zeros(K0, dtype=np.uint8)
        targets[[0, 24]] = 1
        candidates = _candidate_asset(np.arange(K0), targets)
        base = np.arange(1, K0 + 1, dtype=np.int32)
        new = base.copy()
        new[0] = 25
        new[24] = 1
        counts = np.zeros(K0, dtype=np.int32)
        movement = _movement(candidates, base, new, counts)["strict_cold"]
        self.assertEqual(movement["truth_moved_up"], 1)
        self.assertEqual(movement["truth_moved_down"], 1)
        self.assertEqual(movement["top20_net_truth"], 0)

    def test_scoring_verdict_requires_all_means_and_three_windows(self) -> None:
        windows: dict[str, dict] = {}
        names = ("winter_20200122", "spring_20200318", "early_summer_20200624", "late_summer_20200819")
        for index, window in enumerate(names):
            base = 1.0
            new = 2.0 if index < 3 else 0.5
            windows[window] = {
                "A00": {"at_k": {"20": {"positive_density": base}}, "ranking": {"mrr": base, "conversion": {"top200_to_top20": base}}},
                "A01": {"at_k": {"20": {"positive_density": new}}, "ranking": {"mrr": new, "conversion": {"top200_to_top20": new}}},
            }
        result = scoring_verdict(windows, "A00", "A01", True)
        self.assertEqual(result["verdict"], "supported")

    def test_candidate_verdict_marks_opposite_segment_means_mixed(self) -> None:
        windows: dict[str, dict] = {}
        flows: dict[str, dict] = {}
        names = ("winter_20200122", "spring_20200318", "early_summer_20200624", "late_summer_20200819")
        for window in names:
            def metric(strict: float, sparse: float) -> dict:
                return {"at_k": {"200": {"segments": {"strict_cold": {"recall": strict}, "sparse_1_5": {"recall": sparse}}}}}
            windows[window] = {"A00": metric(0.1, 0.1), "A10": metric(0.2, 0.05)}
            flows[window] = {"cold_universe": {"gained_truth_pairs": 2, "lost_truth_pairs": 2, "net_truth_pairs": 0}}
        self.assertEqual(candidate_set_verdict(windows, flows)["verdict"], "mixed")


if __name__ == "__main__":
    unittest.main()
