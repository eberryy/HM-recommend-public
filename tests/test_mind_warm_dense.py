from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from hm_recsys.mind_warm_dense import DENSE_FEATURES, _user_features, _validate_keys, build_dense_features


class MindWarmDenseTests(unittest.TestCase):
    def values(self, *, no_history: bool = False, one_interest: bool = False):
        return _user_features(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([True, True, False]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            torch.tensor([True, not one_interest, False]),
            torch.empty((0, 2)) if no_history else torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
            torch.empty(0, dtype=torch.long) if no_history else torch.tensor([1, 7, 29]),
            torch.tensor([10.0, 20.0, 999.0]),
        )

    def test_masked_candidates_and_empty_bin_are_nan(self):
        values = self.values()
        self.assertEqual(set(values), set(DENSE_FEATURES))
        np.testing.assert_array_equal(values["dense_item_missing"], [0, 0, 1])
        self.assertTrue(np.isnan(values["dense_i2v_vocab_count"][2]))
        self.assertTrue(np.isnan(values["dense_mind_max_cosine"][2]))
        self.assertTrue(np.isnan(values["dense_history_max_cosine_0_7"][2]))
        self.assertTrue(np.isnan(values["dense_history_mean_cosine_8_28"]).all())
        np.testing.assert_array_equal(values["dense_history_support_0_7"], [2, 2, 2])
        np.testing.assert_array_equal(values["dense_history_support_8_28"], [0, 0, 0])

    def test_cosine_max_mean_gap_ignore_inactive_interest(self):
        values = self.values()
        np.testing.assert_allclose(values["dense_mind_max_cosine"][:2], [1, 1])
        np.testing.assert_allclose(values["dense_mind_mean_cosine"][:2], [0.5, 0.5])
        np.testing.assert_allclose(values["dense_mind_top2_gap"][:2], [1, 1])
        np.testing.assert_allclose(values["dense_history_mean_cosine_0_7"][:2], [0, 0])
        np.testing.assert_allclose(values["dense_history_max_cosine_0_7"][:2], [1, 0])
        np.testing.assert_allclose(values["dense_history_max_cosine_29_84"][:2], [0, 1])

    def test_no_history_is_missing_and_single_interest_gap_undefined(self):
        values = self.values(no_history=True)
        np.testing.assert_array_equal(values["dense_user_history_missing"], [1, 1, 1])
        self.assertTrue(np.isnan(values["dense_mind_max_cosine"]).all())
        self.assertTrue(np.isnan(values["dense_history_max_cosine_0_7"]).all())
        np.testing.assert_array_equal(values["dense_history_support_29_84"], [0, 0, 0])
        self.assertTrue(np.isnan(self.values(one_interest=True)["dense_mind_top2_gap"]).all())

    def test_key_validation_rejects_duplicates_labels_and_nulls(self):
        keys = pd.DataFrame({"customer_id": ["u1", "u2"], "article_id": ["0001", "0001"]})
        _validate_keys(keys)
        for invalid in (
            pd.concat([keys, keys.iloc[:1]], ignore_index=True),
            keys.assign(target=[0, 1]),
            keys.assign(label=[0, 1]),
            keys.assign(article_id=[None, "0001"]),
            keys.assign(article_id=[1, 2]),
        ):
            with self.assertRaises(ValueError):
                _validate_keys(invalid)

    def test_zero_age_is_rejected_and_final_week_guard_precedes_assets(self):
        with self.assertRaisesRegex(ValueError, "history ages"):
            _user_features(
                torch.ones((1, 2)), torch.tensor([True]), torch.ones((1, 2)),
                torch.tensor([True]), torch.ones((1, 2)), torch.tensor([0]), torch.ones(1),
            )
        with self.assertRaisesRegex(ValueError, "final week"):
            build_dense_features("2020-09-16", pd.DataFrame(columns=["customer_id", "article_id"]))


if __name__ == "__main__":
    unittest.main()
