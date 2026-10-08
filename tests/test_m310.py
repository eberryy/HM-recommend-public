import unittest

import numpy as np
import pandas as pd

from hm_recsys.m310 import (
    ADMISSION_K,
    DIRECT_FEATURES,
    VARIANT,
    _feature_list,
    _passes_map_gate,
    _safe_image_ranks,
    _validated_admission_mask,
    validate_protocol,
)


class M310ProtocolTests(unittest.TestCase):
    def test_protocol_is_frozen(self):
        validate_protocol()
        self.assertEqual(VARIANT, "base_historical_soft_500")
        self.assertEqual(ADMISSION_K, (0, 1, 3, 5))

    def test_direct_feature_contract(self):
        self.assertEqual(len(DIRECT_FEATURES), 6)
        features = _feature_list()
        self.assertEqual(len(features), len(set(features)))
        self.assertTrue(set(DIRECT_FEATURES).issubset(features))
        self.assertNotIn("base_present", features)
        self.assertNotIn("soft_present", features)

    def test_missing_image_rank_is_not_cast_to_negative_integer(self):
        ranks, available = _safe_image_ranks(pd.Series([1.0, np.nan, 3.0]))
        np.testing.assert_array_equal(ranks, np.array([1, 0, 3], dtype=np.int64))
        np.testing.assert_array_equal(available, np.array([True, False, True]))

    def test_k_zero_admits_no_rows_and_k_cap_is_enforced(self):
        customer_ids = pd.Series(["u1", "u1", "u1", "u2", "u2"])
        active = np.array([True, True, True, True, False])
        soft = np.array([True, True, False, True, True])
        ranks, available = _safe_image_ranks(pd.Series([1.0, np.nan, np.nan, 1.0, 1.0]))
        admitted_zero, audit_zero = _validated_admission_mask(
            customer_ids=customer_ids,
            active=active,
            soft=soft,
            image_rank=ranks,
            image_rank_available=available,
            k=0,
        )
        self.assertFalse(admitted_zero.any())
        self.assertEqual(audit_zero["admitted_rows"], 0)
        self.assertEqual(audit_zero["max_admitted_per_user"], 0)

        admitted_one, audit_one = _validated_admission_mask(
            customer_ids=customer_ids,
            active=active,
            soft=soft,
            image_rank=ranks,
            image_rank_available=available,
            k=1,
        )
        np.testing.assert_array_equal(admitted_one, np.array([True, False, False, True, False]))
        self.assertEqual(audit_one["admitted_rows"], 2)
        self.assertEqual(audit_one["max_admitted_per_user"], 1)

    def test_map_gate_ignores_floating_point_noise(self):
        self.assertFalse(_passes_map_gate({"a": 3e-17, "b": 2e-17}))
        self.assertTrue(_passes_map_gate({"a": 2e-6, "b": 1e-6}))
        self.assertFalse(_passes_map_gate({"a": 2e-6, "b": -2e-6}))


if __name__ == "__main__":
    unittest.main()
