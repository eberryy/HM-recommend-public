import unittest

import numpy as np
import pandas as pd

from hm_recsys.m311 import (
    DEPTHS,
    MAX_REPLACEMENTS,
    ORACLE_DEPTHS,
    PROTECTED_BASE_RANK,
    _average_precision,
    _ordering_metrics,
)


class M311AuditTests(unittest.TestCase):
    def test_protocol_constants_are_bounded(self):
        self.assertEqual(DEPTHS, (1, 3, 5, 10, 20, 50, 100))
        self.assertEqual(ORACLE_DEPTHS, (1, 3, 5, 10))
        self.assertEqual(MAX_REPLACEMENTS, (1, 2))
        self.assertEqual(PROTECTED_BASE_RANK, 7)

    def test_average_precision_uses_truth_denominator(self):
        labels = np.array([1, 0, 1], dtype=np.int8)
        self.assertAlmostEqual(_average_precision(labels, 4), (1.0 + 2.0 / 3.0) / 4.0)

    def test_ordering_metrics_fix_same_candidate_pool(self):
        frame = pd.DataFrame({
            "evaluation_group": ["w|u1"] * 3,
            "article_id": ["a", "b", "c"],
            "target": [0, 1, 0],
            "soft_rank": [1, 2, 3],
            "soft_personalized_present": [1, 1, 1],
            "soft_personalized_rank": [1, 2, 3],
            "soft_global_present": [1, 1, 1],
            "soft_global_rank": [1, 2, 3],
            "image_expert_score": [0.1, 0.9, 0.2],
            "direct_visual_decay_max": [0.1, 0.8, 0.2],
        })
        source = _ordering_metrics(frame, "all_soft")
        expert = _ordering_metrics(frame, "image_expert")
        self.assertEqual(source["candidate_rows"], expert["candidate_rows"])
        self.assertEqual(source["depths"]["1"]["positive_pairs"], 0)
        self.assertEqual(expert["depths"]["1"]["positive_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
