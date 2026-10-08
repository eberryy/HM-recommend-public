import inspect
import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from hm_recsys.p37b_features import compute_relation_feature_batch
from hm_recsys.p42r_data import (
    _save_frame, history_context, join_labels_last, m4_relation_only,
    population_comparison, prepare_cutoff, shared_users, shortlist,
)


class P42RDataTest(unittest.TestCase):
    def test_history_context_no_future_rows_and_original_tie_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.parquet"
            events = pd.DataFrame({"customer_id": ["a", "a", "a", "a", "b", "c"],
                "article_id": ["x", "x", "y", "z", "z", "x"],
                "t_dat": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01",
                                          "2020-01-02", "2020-01-03", "2019-12-01"])})
            _save_frame(events, path)
            counts, rows, days = history_context(path, "2020-01-02", ["a", "b", "c"], ["x", "y", "z"])
            np.testing.assert_array_equal(counts, [3, 1, 0])  # Duplicate events retained.
            np.testing.assert_array_equal(rows[0, :3], [0, 1, -1])
            self.assertTrue((rows[1] == -1).all())
            np.testing.assert_array_equal(days[0, :2], [1, 1])
            self.assertEqual(rows[2, 0], 0)
            self.assertNotIn("t_dat>=", inspect.getsource(history_context))

    def test_labels_join_after_frozen_identity_not_user_cold_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transactions.parquet"
            events = pd.DataFrame({"customer_id": ["a", "a", "b"], "article_id": ["x", "x", "z"],
                "t_dat": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"])})
            _save_frame(events, path)
            candidates = pd.DataFrame({"customer_id": ["a", "b"], "article_id": ["x", "y"],
                                       "cold_rank": [1, 1]})
            result, truth = join_labels_last(candidates, path, "2020-01-02", ["a", "b"])
            self.assertEqual(len(truth), 2)
            self.assertEqual(result.target.tolist(), [1, 0])
            self.assertEqual(result.customer_id.tolist(), ["a", "b"])

    def test_shared_parity_actual_rows_and_no_ghost_user(self):
        cold = pd.DataFrame({"customer_id": ["a"] * 50, "article_id": list(map(str, range(50)))})
        warm = pd.DataFrame({"customer_id": ["a"] * 12, "article_id": list(map(str, range(12))),
                             "warm_rank": np.arange(1, 13)})
        self.assertTrue(shared_users(["a"], cold, warm)["exact_user_set_parity"])
        with self.assertRaisesRegex(ValueError, "shared_population_contract_failure"):
            shared_users(["a", "no_history"], cold, warm)
        with self.assertRaises(ValueError):
            shared_users(["a"], cold.iloc[:-1], warm)

    def test_m4_only_relation_exact_original_feature_helper(self):
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(300, 128)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        histories = np.full((2, 20), -1, np.int32)
        histories[:, :4] = [[3, 5, 10, 20], [4, 7, 9, 11]]
        days = np.zeros((2, 20), np.float32)
        days[:, :4] = [1, 9, 30, 90]
        items = np.tile(np.arange(200), (2, 1))
        product, garment = np.arange(300) % 9, np.arange(300) % 5
        observed = m4_relation_only(items, histories, days, embeddings, product, garment, torch.device("cpu"))
        expected = compute_relation_feature_batch(candidate_rows=items,
            m4_coarse_scores=np.zeros((2, 200), np.float32), history_rows=histories,
            history_days=days, history_mask=histories >= 0, m4_embeddings=embeddings,
            p33_embeddings=embeddings, catalog_product_type=product, catalog_garment_group=garment)["m4_relation"]
        np.testing.assert_array_equal(observed, expected)

    def test_shortlist_no_labels_and_original_budget_rank_scales(self):
        arrays = {"user_index": np.zeros(200, np.int32), "catalog_row": np.arange(200),
                  "rank": np.arange(1, 201), "coarse_score": np.arange(200, dtype=np.float32)}
        scores = np.arange(200, dtype=np.float32)
        ranks = np.arange(200, 0, -1)
        frame = shortlist(arrays, scores, ranks, ["a"], [f"{i:010d}" for i in range(200)])
        self.assertEqual(len(frame), 50)
        self.assertEqual(frame.article_id.iloc[0], "0000000199")
        self.assertNotIn("target", frame)
        np.testing.assert_array_equal(frame.b0_rank_pct, np.arange(1, 51, dtype=np.float32) / 50)

    def test_preregistration_and_final_guard_before_io(self):
        with self.assertRaises(ValueError):
            prepare_cutoff(Path("."), {}, "2020-09-16", Path("."))
        with self.assertRaisesRegex(ValueError, "preregistration"):
            prepare_cutoff(Path("."), {}, "2020-01-22", Path("."))

    def test_write_preserves_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.parquet"
            _save_frame(pd.DataFrame({"value": [1]}), path)
            with self.assertRaises(FileExistsError):
                _save_frame(pd.DataFrame({"value": [2]}), path)
            with duckdb.connect() as con:
                self.assertEqual(con.execute("SELECT value FROM read_parquet(?)", [str(path)]).fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
