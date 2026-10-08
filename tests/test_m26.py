import unittest

import duckdb
import numpy as np
import pandas as pd

from hm_recsys.m26 import (
    _build_extended_image_candidates,
    realized_oracle_fraction,
    validate_group_sizes,
)


class M26Tests(unittest.TestCase):
    def test_variable_groups_accept_unpadded_users(self):
        audit = validate_group_sizes([100, 173, 300])
        self.assertEqual(audit["groups"], 3)
        self.assertEqual(audit["rows"], 573)
        self.assertEqual(audit["min_group_rows"], 100)
        self.assertEqual(audit["max_group_rows"], 300)

    def test_variable_groups_reject_out_of_contract_size(self):
        with self.assertRaisesRegex(ValueError, "100..300"):
            validate_group_sizes([99, 100])
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_group_sizes([])

    def test_oracle_realization_handles_non_positive_ceiling(self):
        self.assertAlmostEqual(
            realized_oracle_fraction(0.024, 0.020, 0.140, 0.120), 0.2
        )
        self.assertIsNone(
            realized_oracle_fraction(0.024, 0.020, 0.120, 0.120)
        )

    def test_cross_seed_aggregates_preserve_best_and_add_mean(self):
        connection = duckdb.connect()
        try:
            seeds = pd.DataFrame(
                {
                    "customer_id": ["u1", "u1"],
                    "row_index": [0, 1],
                    "seed_rank": [1, 2],
                }
            )
            indices = np.array(
                [[2, 3], [2, 3], [0, 1], [0, 1]], dtype=np.int32
            )
            scores = np.array(
                [[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]],
                dtype=np.float32,
            )
            _build_extended_image_candidates(
                connection,
                seeds,
                indices,
                scores,
                np.array(["a", "b", "c", "d"]),
                max_candidates=2,
            )
            row = connection.execute(
                "SELECT article_id,image_rank,image_score,image_cosine,"
                "image_mean_cosine,image_top3_mean_cosine,"
                "image_mean_weighted_score,image_top3_mean_weighted_score,"
                "best_seed_rank,best_neighbor_rank,seed_support "
                "FROM image_candidates ORDER BY image_rank LIMIT 1"
            ).fetchone()
            self.assertEqual(row[0:2], ("c", 1))
            self.assertAlmostEqual(row[2], 0.9, places=6)
            self.assertAlmostEqual(row[3], 0.9, places=6)
            self.assertAlmostEqual(row[4], 0.8, places=6)
            self.assertAlmostEqual(row[5], 0.8, places=6)
            self.assertAlmostEqual(row[6], (0.9 + 0.7 / 1.1) / 2, places=6)
            self.assertAlmostEqual(row[7], (0.9 + 0.7 / 1.1) / 2, places=6)
            self.assertEqual(row[8:], (1, 1, 2))
        finally:
            connection.close()

if __name__ == "__main__":
    unittest.main()
