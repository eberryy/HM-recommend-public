from __future__ import annotations

import unittest

import numpy as np

from hm_recsys.p33 import ViewNeighbors, _compare


def _ordering(strict: float, sparse: float, density: float, mrr: float, conversion: float) -> dict:
    return {
        "at_k": {
            "20": {"positive_density": density},
            "200": {"segments": {"strict_cold": {"recall": strict}, "sparse_1_5": {"recall": sparse}}},
        },
        "ranking": {"mrr": mrr, "conversion": {"top200_to_top20": conversion}},
    }


class P33Tests(unittest.TestCase):
    def test_view_neighbor_validation_rejects_self(self) -> None:
        view = ViewNeighbors(
            anchors=np.asarray([1], dtype=np.int32),
            neighbors=np.asarray([[1] + [-1] * 99], dtype=np.int32),
            scores=np.asarray([[1.0] + [0.0] * 99], dtype=np.float32),
        )
        with self.assertRaises(RuntimeError):
            view.validate("bad")

    def test_promotion_requires_representation_and_end_to_end(self) -> None:
        windows = ("winter_20200122", "spring_20200318", "early_summer_20200624", "late_summer_20200819")
        baseline_p31 = {"windows": {window: {"coarse_metrics": _ordering(0.1, 0.1, 0.1, 0.1, 0.1)} for window in windows}}
        multiview_p31 = {"windows": {window: {"coarse_metrics": _ordering(0.2, 0.2, 0.2, 0.2, 0.2)} for window in windows}}
        baseline_p32 = {"windows": {window: {"metrics": {"candidate_aware_order": _ordering(0.1, 0.1, 0.1, 0.1, 0.1)}} for window in windows}}
        multiview_p32 = {"windows": {window: {"metrics": {"candidate_aware_order": _ordering(0.2, 0.2, 0.2, 0.2, 0.2)}} for window in windows}}
        result = _compare(
            baseline_p31=baseline_p31,
            baseline_p32=baseline_p32,
            multiview_p31=multiview_p31,
            multiview_p32=multiview_p32,
        )
        self.assertTrue(result["promotion_passed"])


if __name__ == "__main__":
    unittest.main()
