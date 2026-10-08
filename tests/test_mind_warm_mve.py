from __future__ import annotations

import unittest

import numpy as np
import torch

from hm_recsys.mind_warm_mve import (
    MindEncoder,
    _guard,
    _training_batch,
    age_bucket,
    dynamic_interest_count,
)


class MindWarmMveTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.items = torch.nn.functional.normalize(torch.randn(12, 64), dim=1)

    def test_final_week_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _guard("2020-09-16")

    def test_dynamic_interest_count_has_sparse_fallback(self) -> None:
        lengths = torch.tensor([0, 1, 4, 5, 8, 50])
        self.assertEqual(dynamic_interest_count(lengths, 3).tolist(), [1, 1, 1, 2, 3, 3])

    def test_age_buckets_are_registered_boundaries(self) -> None:
        ages = torch.tensor([1, 7, 8, 28, 29, 84])
        self.assertEqual(age_bucket(ages).tolist(), [0, 0, 1, 1, 2, 2])

    def test_same_day_permutation_does_not_change_interests(self) -> None:
        model = MindEncoder(3, "learned_buckets")
        model.eval()
        first_ids = torch.tensor([[1, 2, 3, 4, -1]])
        first_ages = torch.tensor([[2, 2, 9, 31, 0]])
        second_ids = torch.tensor([[2, 1, 3, 4, -1]])
        second_ages = torch.tensor([[2, 2, 9, 31, 0]])
        lengths = torch.tensor([4])
        with torch.no_grad():
            first, active_first = model(self.items, first_ids, first_ages, lengths)
            second, active_second = model(self.items, second_ids, second_ages, lengths)
        self.assertTrue(torch.equal(active_first, active_second))
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_learned_age_weights_are_positive_and_mean_one(self) -> None:
        model = MindEncoder(3, "learned_buckets")
        weights = np.asarray(model.learned_age_weights())
        self.assertTrue((weights > 0).all())
        np.testing.assert_allclose(weights.mean(axis=1), np.ones(3), atol=1e-7)

    def test_fixed_decay_is_monotone(self) -> None:
        model = MindEncoder(3, "fixed_28d")
        weights = model.age_weights(torch.tensor([[7, 28, 84]]))[0, 0]
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))

    def test_all_same_day_positives_are_excluded_from_negatives(self) -> None:
        arrays = {
            "history": np.array([[1, 2]], np.int32),
            "ages": np.array([[1, 2]], np.int16),
            "lengths": np.array([2], np.int16),
            "positives": np.array([5], np.int32),
            "groups": np.array([0], np.int32),
            "group_offsets": np.array([0, 2], np.int64),
            "group_items": np.array([5, 7], np.int32),
        }
        *_, excluded = _training_batch(arrays, np.array([0]), np.array([3, 5, 7, 9]))
        self.assertEqual(excluded.tolist(), [[False, True, True, False]])


if __name__ == "__main__":
    unittest.main()
