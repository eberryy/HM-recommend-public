from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from hm_recsys.p3 import _candidate_parity, _safe_cutoff
from hm_recsys.p3_model import CandidateAwareReranker, binary_auc, pairwise_logistic_loss, sample_same_user_pairs


def _model_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    candidate = torch.randn(2, 128)
    history = torch.randn(2, 20, 128)
    days = torch.arange(20, dtype=torch.float32).repeat(2, 1)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[0, 7:] = False
    history[0, 7:] = 0
    return candidate, history, days, mask


class Phase3ModelTests(unittest.TestCase):
    def test_attention_sums_to_one_over_valid_history(self) -> None:
        model = CandidateAwareReranker()
        candidate, history, days, mask = _model_inputs()
        _, attention = model(candidate, history, days, mask)
        self.assertTrue(torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-6))

    def test_padded_history_has_zero_attention(self) -> None:
        model = CandidateAwareReranker()
        candidate, history, days, mask = _model_inputs()
        _, attention = model(candidate, history, days, mask)
        self.assertEqual(int(torch.count_nonzero(attention[~mask])), 0)

    def test_same_user_different_candidate_changes_attention(self) -> None:
        torch.manual_seed(9)
        model = CandidateAwareReranker()
        _, history, days, mask = _model_inputs()
        candidate = torch.stack([torch.ones(128), -torch.ones(128)])
        _, attention = model(candidate, history[:1].repeat(2, 1, 1), days[:1].repeat(2, 1), mask[:1].repeat(2, 1))
        self.assertFalse(torch.allclose(attention[0], attention[1]))

    def test_pairwise_loss_rewards_positive_above_negative(self) -> None:
        better = pairwise_logistic_loss(torch.tensor([2.0]), torch.tensor([-1.0]))
        worse = pairwise_logistic_loss(torch.tensor([-1.0]), torch.tensor([2.0]))
        self.assertLess(better, worse)

    def test_sampler_is_same_user_bounded_and_covers_buckets(self) -> None:
        user_index = np.repeat(np.arange(2, dtype=np.int32), 200)
        rank = np.tile(np.arange(1, 201, dtype=np.uint16), 2)
        target = np.zeros(400, dtype=np.uint8)
        target[[9, 249]] = 1
        pairs, audit = sample_same_user_pairs(user_index=user_index, rank=rank, target=target, cutoff="2020-01-22")
        self.assertTrue(audit["same_user_only"])
        self.assertTrue(audit["all_buckets_covered"])
        self.assertEqual(len(pairs), 100)
        for positive_row in np.unique(pairs[:, 1]):
            self.assertLessEqual(int(np.count_nonzero(pairs[:, 1] == positive_row)), 50)
        self.assertTrue(np.all(user_index[pairs[:, 1]] == user_index[pairs[:, 2]]))

    def test_sampler_is_deterministic(self) -> None:
        user_index = np.zeros(200, dtype=np.int32)
        rank = np.arange(1, 201, dtype=np.uint16)
        target = np.zeros(200, dtype=np.uint8)
        target[0] = 1
        first, _ = sample_same_user_pairs(user_index=user_index, rank=rank, target=target, cutoff="2020-01-22")
        second, _ = sample_same_user_pairs(user_index=user_index, rank=rank, target=target, cutoff="2020-01-22")
        self.assertTrue(np.array_equal(first, second))

    def test_candidate_set_parity_and_rank_continuity(self) -> None:
        asset = SimpleNamespace(candidates={"user_index": np.repeat(np.arange(2, dtype=np.int32), 3), "catalog_row": np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32)}, embeddings=np.empty((10, 128), dtype=np.float16))
        rerank = np.asarray([2, 1, 3, 3, 1, 2], dtype=np.uint16)
        self.assertTrue(_candidate_parity(asset, rerank)["passed"])

    def test_candidate_parity_rejects_noncontinuous_rank(self) -> None:
        asset = SimpleNamespace(candidates={"user_index": np.zeros(3, dtype=np.int32), "catalog_row": np.asarray([1, 2, 3], dtype=np.int32)}, embeddings=np.empty((10, 128), dtype=np.float16))
        self.assertFalse(_candidate_parity(asset, np.asarray([1, 1, 3], dtype=np.uint16))["passed"])

    def test_auc_is_tie_aware(self) -> None:
        self.assertAlmostEqual(binary_auc(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])), 0.5)
        self.assertAlmostEqual(binary_auc(np.asarray([2.0]), np.asarray([1.0])), 1.0)

    def test_final_cutoff_is_rejected(self) -> None:
        _safe_cutoff("2020-08-19")
        with self.assertRaises(RuntimeError):
            _safe_cutoff("2020-09-16")


if __name__ == "__main__":
    unittest.main()
