from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hm_recsys.m2 import FULL_FEATURES
from hm_recsys.m26 import IMAGE_FEATURES
from hm_recsys.m213 import (
    BASE_CONTROL,
    IMAGE_CONTROL,
    NO_DECAY_TARGET_FEATURES,
    PRIMARY_MODEL,
    evaluate_image_conversion,
    feature_sets,
    load_image_distribution_sample,
)


class M213FeatureTests(unittest.TestCase):
    def test_feature_sets_add_image_then_no_decay_target_aware(self) -> None:
        sets = feature_sets()
        self.assertEqual(sets[BASE_CONTROL], list(FULL_FEATURES))
        self.assertEqual(
            set(sets[IMAGE_CONTROL]) - set(sets[BASE_CONTROL]), set(IMAGE_FEATURES)
        )
        self.assertEqual(
            set(sets[PRIMARY_MODEL]) - set(sets[IMAGE_CONTROL]),
            set(NO_DECAY_TARGET_FEATURES),
        )
        self.assertEqual(len(NO_DECAY_TARGET_FEATURES), 21)


class M213SamplingTests(unittest.TestCase):
    def _fixture(self, directory: str) -> Path:
        path = Path(directory) / "features.parquet"
        rows = []
        for rank in range(1, 301):
            rows.append(
                {
                    "customer_id": "u1",
                    "article_id": f"{rank:010d}",
                    "candidate_rank": rank,
                    "image_is_new": int(rank > 100),
                    "target": int(rank in (5, 150)),
                }
            )
        connection = duckdb.connect()
        frame = pd.DataFrame(rows)
        connection.register("fixture", frame)
        connection.execute(f"COPY fixture TO '{path.as_posix()}' (FORMAT PARQUET)")
        connection.close()
        return path

    def test_sampling_covers_baseline_image_and_rank_tertiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._fixture(temporary)
            sample, sizes, evidence = load_image_distribution_sample(
                [path], ["candidate_rank", "image_is_new"], seed=17
            )
            negatives = sample[sample["target"] == 0]
            self.assertEqual(sizes, [62])
            self.assertEqual(len(negatives), 60)
            cells = set(
                zip(
                    negatives["_m213_source_layer"],
                    negatives["_m213_rank_bucket"],
                    strict=True,
                )
            )
            self.assertEqual(
                cells,
                {
                    (source, bucket)
                    for source in ("baseline_top100", "image_only")
                    for bucket in (1, 2, 3)
                },
            )
            self.assertEqual(evidence["unobserved_per_positive"], 30.0)

    def test_sampling_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._fixture(temporary)
            first = load_image_distribution_sample(
                [path], ["candidate_rank", "image_is_new"], seed=19
            )
            second = load_image_distribution_sample(
                [path], ["candidate_rank", "image_is_new"], seed=19
            )
            pd.testing.assert_frame_equal(first[0], second[0])
            self.assertEqual(first[1:], second[1:])


class M213ConversionTests(unittest.TestCase):
    def test_image_only_cold_truth_conversion_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "evaluation.duckdb"
            transactions = root / "transactions.parquet"
            connection = duckdb.connect(str(database))
            predictions = pd.DataFrame(
                [
                    {
                        "customer_id": "u1",
                        "article_id": "cold_image",
                        "candidate_rank": 101,
                        "target": 1,
                        "user_history_events_12w": 1,
                        "image_present": 1,
                        "image_is_new": 1,
                        "score_model": 0.9,
                    },
                    {
                        "customer_id": "u1",
                        "article_id": "warm_base",
                        "candidate_rank": 1,
                        "target": 0,
                        "user_history_events_12w": 1,
                        "image_present": 0,
                        "image_is_new": 0,
                        "score_model": 0.1,
                    },
                ]
            )
            connection.register("frame", predictions)
            connection.execute("CREATE TABLE predictions AS SELECT * FROM frame")
            connection.close()
            tx = pd.DataFrame(
                [
                    {
                        "customer_id": "u1",
                        "article_id": "warm_base",
                        "t_dat": pd.Timestamp("2020-07-01"),
                    },
                    {
                        "customer_id": "u1",
                        "article_id": "cold_image",
                        "t_dat": pd.Timestamp("2020-07-22"),
                    },
                ]
            )
            connection = duckdb.connect()
            connection.register("transactions", tx)
            connection.execute(
                f"COPY transactions TO '{transactions.as_posix()}' (FORMAT PARQUET)"
            )
            connection.close()
            result = evaluate_image_conversion(
                evaluation_db=database,
                transactions_path=transactions,
                cutoff="2020-07-22",
                variant_names=["model"],
                metric_k=12,
            )["orderings"]["model__inactive_rrf"]
            self.assertEqual(result["image_only_truth_pairs"], 1)
            self.assertEqual(result["image_only_top12_truth_pairs"], 1)
            self.assertEqual(result["image_only_top12_cold_truth_pairs"], 1)
            self.assertAlmostEqual(
                result["image_only_ap_contribution_to_overall_map@12"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
