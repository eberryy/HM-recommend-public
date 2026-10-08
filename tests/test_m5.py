from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from hm_recsys.m4_contract import stable_u64
from hm_recsys.m5_data import PRIMARY_K, _allocate, _scale_stats
from hm_recsys.m5_model import _average_precision, _stable_row_hash, _write_parquet
from hm_recsys.m5_model import _frame


class M5DataTests(unittest.TestCase):
    def test_allocate_uses_top50_rows(self) -> None:
        arrays = _allocate(2 * PRIMARY_K)
        self.assertEqual(len(arrays["user_index"]), 100)
        self.assertIn("student_raw_weighted_gap", arrays)

    def test_scale_stats_are_nested_and_count_pair_units(self) -> None:
        users = [f"u-{index}" for index in range(200)]
        arrays = {
            "user_index": np.repeat(np.arange(len(users), dtype=np.int32), 2),
            "target": np.tile(np.asarray([1, 0], dtype=np.uint8), len(users)),
            "item_is_strict_cold": np.tile(np.asarray([1, 0], dtype=np.uint8), len(users)),
        }
        result = _scale_stats(arrays, users)
        self.assertLessEqual(result["10pct"]["positive_pairs"], result["30pct"]["positive_pairs"])
        self.assertLessEqual(result["30pct"]["positive_pairs"], result["full"]["positive_pairs"])
        self.assertEqual(result["full"]["positive_pairs"], 200)
        self.assertEqual(result["full"]["positive_groups"], 200)
        self.assertEqual(result["full"]["positive_density"], 0.5)
        expected_10 = sum(stable_u64(user) % 1_000_000 < 100_000 for user in users)
        self.assertEqual(result["10pct"]["positive_pairs"], expected_10)

    def test_average_precision_uses_real_top12_positions(self) -> None:
        ranked = ["x", "a", "y", "b"]
        self.assertAlmostEqual(_average_precision(ranked, {"a", "b"}), (1 / 2 + 2 / 4) / 2)

    def test_category_levels_are_catalog_fixed(self) -> None:
        rows = 3
        arrays = _allocate(rows)
        arrays["catalog_row"][:] = [0, 1, 0]
        arrays["student_rank"][:] = [1, 2, 3]
        arrays["item_events_before_cutoff"] = np.asarray([0, 1, 0], dtype=np.int16)
        arrays["item_is_strict_cold"] = np.asarray([1, 0, 1], dtype=np.uint8)
        arrays["user_history_events_12w"] = np.ones(rows, dtype=np.int32)
        arrays["user_history_distinct_items_12w"] = np.ones(rows, dtype=np.int32)
        arrays["user_days_since_last_event"] = np.ones(rows, dtype=np.int16)
        arrays["user_age"] = np.ones(rows, dtype=np.float32)
        arrays["target"] = np.zeros(rows, dtype=np.uint8)
        arrays["user_index"][:] = 0
        for field in ("product_type_no", "garment_group_no", "department_no", "index_group_no", "perceived_colour_master_id"):
            arrays[f"user_{field}_share_12w"] = np.zeros(rows, dtype=np.float32)
        categories = np.zeros((2, 11), dtype=np.int32)
        categories[1, :] = 2
        frame = _frame(arrays, np.arange(rows), categories)
        self.assertEqual(list(frame["item_product_type_no"].cat.categories), [0, 1, 2])

    def test_duckdb_parquet_writer_has_no_optional_pandas_dependency(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "scores.parquet"
            _write_parquet(pd.DataFrame({"score": [0.1, 0.2]}), path)
            self.assertTrue(path.is_file())

    def test_row_hash_is_identity_based_not_input_order_based(self) -> None:
        users = np.asarray([1, 2, 3], dtype=np.int32)
        items = np.asarray([11, 12, 13], dtype=np.int32)
        ranks = np.asarray([1, 2, 3], dtype=np.uint8)
        direct = _stable_row_hash(user_index=users, catalog_row=items, rank=ranks, salt=7)
        order = np.asarray([2, 0, 1])
        shuffled = _stable_row_hash(
            user_index=users[order], catalog_row=items[order], rank=ranks[order], salt=7
        )
        np.testing.assert_array_equal(direct[order], shuffled)


if __name__ == "__main__":
    unittest.main()
