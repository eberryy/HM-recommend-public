from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hm_recsys.m210 import TARGET_AWARE_FEATURES
from hm_recsys.m211 import (
    FEATURE_FAMILIES,
    IPW_MODEL,
    PRIMARY_MODEL,
    feature_sets,
    load_distribution_sample,
)


class M211FeatureTests(unittest.TestCase):
    def test_feature_families_are_disjoint_target_aware_subsets(self) -> None:
        flattened = [feature for values in FEATURE_FAMILIES.values() for feature in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(set(flattened).issubset(TARGET_AWARE_FEATURES))
        sets = feature_sets()
        self.assertEqual(sets[PRIMARY_MODEL], sets[IPW_MODEL])
        self.assertEqual(
            set(sets["ablation_all_target_aware"]),
            set(sets[PRIMARY_MODEL]) - set(TARGET_AWARE_FEATURES),
        )
        for family, removed in FEATURE_FAMILIES.items():
            self.assertEqual(
                set(sets[f"ablation_without_{family}"]),
                set(sets[PRIMARY_MODEL]) - set(removed),
            )


class M211SamplingTests(unittest.TestCase):
    def _fixture(self, directory: str) -> Path:
        path = Path(directory) / "features.parquet"
        rows = []
        for rank in range(1, 301):
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
        connection.execute(f"COPY fixture TO '{path.as_posix()}' (FORMAT PARQUET)")
        connection.close()
        return path

    def test_sampling_covers_both_sources_and_all_rank_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._fixture(temporary)
            sample, sizes, evidence = load_distribution_sample(
                [path], ["candidate_rank"], seed=7
            )
            negatives = sample[sample["target"] == 0]
            self.assertEqual(sizes, [62])
            self.assertEqual(len(negatives), 60)
            cells = set(
                zip(
                    negatives["_m211_source_layer"],
                    negatives["_m211_rank_bucket"],
                    strict=True,
                )
            )
            self.assertEqual(
                cells,
                {
                    (source, bucket)
                    for source in ("baseline_top100", "item2vec_only")
                    for bucket in (1, 2, 3)
                },
            )
            self.assertEqual(evidence["unobserved_per_positive"], 30.0)
            self.assertEqual(len(evidence["strata"]), 6)
            self.assertEqual(
                sum(row["source_rows"] for row in evidence["strata"]),
                298,
            )
            self.assertEqual(
                sum(row["selected_rows"] for row in evidence["strata"]),
                60,
            )

    def test_sampling_is_deterministic_and_ipw_is_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self._fixture(temporary)
            first, first_sizes, first_evidence = load_distribution_sample(
                [path], ["candidate_rank"], seed=11
            )
            second, second_sizes, second_evidence = load_distribution_sample(
                [path], ["candidate_rank"], seed=11
            )
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first_sizes, second_sizes)
            self.assertEqual(first_evidence, second_evidence)
            probabilities = first.loc[
                first["target"] == 0, "_m211_inclusion_probability"
            ].to_numpy(dtype=np.float64)
            inverse = first.loc[
                first["target"] == 0, "_m211_inverse_probability_weight"
            ].to_numpy(dtype=np.float64)
            np.testing.assert_allclose(probabilities * inverse, 1.0)
            self.assertAlmostEqual(
                float(first["_m211_normalized_ipw"].mean()), 1.0
            )


if __name__ == "__main__":
    unittest.main()
