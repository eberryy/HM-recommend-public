import unittest

import numpy as np
import pandas as pd

from hm_recsys.m31 import (
    SOURCE_NAMES,
    ablation_feature_sets,
    apply_source_outage,
    feature_groups,
)
from hm_recsys.m3 import ANCHOR_NAME
from hm_recsys.m212 import feature_sets


class M31FeatureTests(unittest.TestCase):
    def test_groups_partition_anchor_exactly(self):
        groups = feature_groups()
        flattened = [feature for values in groups.values() for feature in values]
        self.assertEqual(len(groups), 10)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(feature_sets()[ANCHOR_NAME]))
        for name, values in ablation_feature_sets().items():
            self.assertTrue(name.startswith("without_"))
            self.assertLess(len(values), len(flattened))

    def test_m1_outage_removes_only_route_unique_non_item2vec_rows(self):
        frame = pd.DataFrame({
            "repurchase_present": [1, 1, 1, 0],
            "repurchase_rank": [1, 2, 3, np.nan],
            "repurchase_score": [1.0, 0.5, 0.2, np.nan],
            "repurchase_rrf_contribution": [0.1, 0.08, 0.07, np.nan],
            "source_count": [1, 2, 1, 1],
            "item2vec_present": [0, 0, 1, 0],
            "fused_score": [0.1, 0.2, 0.3, 0.4],
        })
        modified, keep = apply_source_outage(frame, "repurchase")
        self.assertEqual(keep.tolist(), [False, True, True, True])
        self.assertEqual(modified["repurchase_present"].tolist(), [0, 0, 0, 0])
        self.assertEqual(modified["source_count"].tolist(), [0, 1, 0, 1])
        self.assertAlmostEqual(float(modified.loc[1, "fused_score"]), 0.12)

    def test_item2vec_outage_removes_only_appended_rows(self):
        frame = pd.DataFrame({
            "item2vec_present": [1, 1, 0],
            "item2vec_is_new": [1, 0, 0],
            "item2vec_rank": [1, 2, np.nan],
            "item2vec_score": [0.9, 0.8, np.nan],
            "item2vec_cosine": [0.9, 0.8, np.nan],
            "item2vec_best_seed_rank": [1, 1, np.nan],
            "item2vec_best_neighbor_rank": [1, 2, np.nan],
            "item2vec_seed_support": [1, 1, np.nan],
            "item2vec_vocab_count": [10, 10, 10],
        })
        modified, keep = apply_source_outage(frame, "item2vec")
        self.assertEqual(keep.tolist(), [False, True, True])
        self.assertEqual(modified["item2vec_present"].tolist(), [0, 0, 0])
        self.assertEqual(modified["item2vec_vocab_count"].tolist(), [0, 0, 0])

    def test_source_names_are_seven_pre_registered_routes(self):
        self.assertEqual(len(SOURCE_NAMES), 7)


if __name__ == "__main__":
    unittest.main()
