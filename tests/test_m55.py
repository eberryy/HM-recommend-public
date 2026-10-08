from __future__ import annotations

import unittest
import inspect

import numpy as np
import pandas as pd

from hm_recsys.m55 import (
    DUAL_FEATURES,
    DUAL_WITHOUT_EXPERT_FEATURES,
    NEGATIVES_PER_POSITIVE,
    WARM_FEATURES,
    _exact_map_from_arrays,
    _render_report,
)


class M55Tests(unittest.TestCase):
    def test_feature_contracts_are_nested(self) -> None:
        self.assertTrue(set(WARM_FEATURES) < set(DUAL_FEATURES))
        self.assertTrue(set(DUAL_WITHOUT_EXPERT_FEATURES) < set(DUAL_FEATURES))
        self.assertNotIn("cold_expert_score", DUAL_WITHOUT_EXPERT_FEATURES)
        self.assertEqual(NEGATIVES_PER_POSITIVE, 30)

    def test_exact_map_uses_complete_truth_denominator(self) -> None:
        predictions = np.asarray([0.9, 0.8, 0.7])
        labels = np.asarray([1, 0, 0], dtype=np.uint8)
        score = _exact_map_from_arrays(
            predictions, labels, [3], [2], np.asarray([False, False, False]),
            np.asarray([1.0, 2.0, 3.0]), np.asarray([1, 2, 3]),
        )
        self.assertAlmostEqual(score, 0.5)

    def test_inactive_order_ignores_model_score(self) -> None:
        score = _exact_map_from_arrays(
            np.asarray([0.1, 0.9]), np.asarray([1, 0], dtype=np.uint8), [2], [1],
            np.asarray([True, True]), np.asarray([1.0, 2.0]), np.asarray([1, 2]),
        )
        self.assertAlmostEqual(score, 1.0)

    def test_report_requires_cold_source_diagnostics(self) -> None:
        text = inspect.getsource(_render_report)
        self.assertIn("来源感知特征增益", text)
        self.assertIn("负采样分层覆盖", text)
        self.assertIn("最终 Top12 来源构成", text)
