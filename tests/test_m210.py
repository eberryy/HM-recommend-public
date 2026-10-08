from __future__ import annotations

import unittest

import numpy as np

from hm_recsys.m210 import (
    ALL_FEATURES,
    M29_FEATURES,
    TARGET_AWARE_FEATURES,
    _swap_delta_fast,
    average_precision_at_k,
    map_pair_gradients,
    map_swap_delta_bruteforce,
)


class M210Tests(unittest.TestCase):
    def test_target_aware_features_extend_m29_without_duplicates(self) -> None:
        self.assertEqual(ALL_FEATURES[:len(M29_FEATURES)], M29_FEATURES)
        self.assertEqual(ALL_FEATURES[len(M29_FEATURES):], TARGET_AWARE_FEATURES)
        self.assertEqual(len(ALL_FEATURES), len(set(ALL_FEATURES)))
        self.assertEqual(len(TARGET_AWARE_FEATURES), 28)

    def test_fast_swap_delta_matches_bruteforce(self) -> None:
        cases = [
            np.array([1, 0, 1, 0, 0], dtype=np.uint8),
            np.array([0, 0, 1, 0, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
            np.array([0]*12+[1, 0, 1], dtype=np.uint8),
        ]
        for labels in cases:
            cumulative = np.cumsum(labels)
            limit = min(12, len(labels))
            reciprocal = np.zeros(len(labels), dtype=np.float64)
            reciprocal[:limit] = (
                labels[:limit]/np.arange(1, limit+1, dtype=np.float64)
            )
            prefix = np.concatenate(([0.0], np.cumsum(reciprocal)))
            for positive_rank in np.flatnonzero(labels == 1):
                for negative_rank in np.flatnonzero(labels == 0):
                    expected = map_swap_delta_bruteforce(
                        labels, int(positive_rank), int(negative_rank), 12
                    )
                    actual = _swap_delta_fast(
                        labels,
                        cumulative,
                        prefix,
                        int(positive_rank),
                        int(negative_rank),
                        12,
                    )
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_map_pair_gradients_push_positive_scores_up(self) -> None:
        predictions = np.zeros(5, dtype=np.float64)
        labels = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
        gradients, hessians, stats = map_pair_gradients(
            predictions, labels, [5], k=12
        )
        self.assertLess(gradients[1], 0)
        self.assertGreater(float(gradients[[0, 2, 3, 4]].sum()), 0)
        self.assertAlmostEqual(float(gradients.sum()), 0.0, places=12)
        self.assertTrue(np.isfinite(gradients).all())
        self.assertTrue((hessians > 0).all())
        self.assertGreater(stats["pairs"], 0)

    def test_map_pair_ignores_zero_positive_group(self) -> None:
        gradients, hessians, stats = map_pair_gradients(
            np.zeros(4), np.zeros(4, dtype=np.uint8), [4], k=12
        )
        self.assertTrue((gradients == 0).all())
        self.assertTrue((hessians > 0).all())
        self.assertEqual(stats["weighted_groups"], 0)

    def test_ap_denominator_uses_min_truth_and_k(self) -> None:
        labels = np.array([1, 0, 1, 0], dtype=np.uint8)
        self.assertAlmostEqual(
            average_precision_at_k(labels, 12),
            (1.0+2.0/3.0)/2.0,
        )



class M210SamplingTests(unittest.TestCase):
    def test_stratified_sampling_balances_two_candidate_strata(self) -> None:
        import tempfile
        from pathlib import Path

        import duckdb
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"features.parquet"
            rows = []
            for rank in range(1, 213):
                rows.append(
                    {
                        "customer_id": "u1",
                        "article_id": f"{rank:010d}",
                        "candidate_rank": rank,
                        "target": 1 if rank in (5, 150) else 0,
                    }
                )
            connection = duckdb.connect()
            frame = pd.DataFrame(rows)
            connection.register("fixture", frame)
            connection.execute(
                f"COPY fixture TO '{path.as_posix()}' (FORMAT PARQUET)"
            )
            connection.close()
            from hm_recsys.m210 import load_training_data

            sampled, sizes, evidence = load_training_data(
                [path], ["candidate_rank"], "stratified_30x"
            )
            negatives = sampled[sampled["target"] == 0]
            self.assertEqual(sizes, [62])
            self.assertEqual(len(negatives), 60)
            self.assertEqual(int((negatives["candidate_rank"] <= 100).sum()), 30)
            self.assertEqual(int((negatives["candidate_rank"] > 100).sum()), 30)
            self.assertEqual(evidence["unobserved_per_positive"], 30.0)


if __name__ == "__main__":
    unittest.main()
