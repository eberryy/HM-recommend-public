from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p35_audit import (
    PROVENANCE_NAMES,
    _age_bucket,
    _exclusive_lane_summary,
    _quartile_labels,
    _rank_by_score,
    _safe_cutoff,
    _teacher_union,
)


def _write_neighbor(path: Path, rows: dict[int, list[int]]) -> None:
    anchors = np.asarray(sorted(rows), dtype=np.int32)
    neighbors = np.full((len(anchors), 100), -1, dtype=np.int32)
    scores = np.zeros((len(anchors), 100), dtype=np.float32)
    for index, anchor in enumerate(anchors):
        values = rows[int(anchor)]
        neighbors[index, : len(values)] = values
        scores[index, : len(values)] = np.linspace(1, 0.1, len(values))
    np.savez_compressed(path, anchors=anchors, neighbors=neighbors, scores=scores)


class P35AuditTests(unittest.TestCase):
    def test_final_cutoff_is_rejected(self) -> None:
        _safe_cutoff("2020-08-19")
        with self.assertRaises(RuntimeError):
            _safe_cutoff("2020-09-16")


    def test_rank_by_score_is_group_local_and_deterministic(self) -> None:
        users = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
        items = np.asarray([3, 2, 1, 5, 4], dtype=np.int32)
        scores = np.asarray([0.5, 0.5, 0.7, 0.1, 0.1], dtype=np.float32)
        self.assertEqual(_rank_by_score(users, items, scores).tolist(), [3, 2, 1, 2, 1])


    def test_quartiles_and_age_bins_are_complete(self) -> None:
        labels, cuts = _quartile_labels(np.asarray([1, 2, 3, 4, np.nan], dtype=float))
        self.assertEqual(set(labels), {"Q1", "Q2", "Q3", "Q4", "ineligible"})
        self.assertEqual(len(cuts), 3)
        self.assertEqual([_age_bucket(value) for value in [1, 8, 29, 85, np.nan]], [
            "0_7", "8_28", "29_84", "over_84_or_none", "over_84_or_none"
        ])


    def test_lane_attribution_is_mutually_exclusive_and_conserved(self) -> None:
        frame = pd.DataFrame(
            {
                "customer_id": ["a", "a", "b", "c"],
                "article_id": ["1", "2", "3", "4"],
                "personalized": [1, 1, 0, 1],
                "global_current": [0, 1, 1, 1],
                "prior_direct": [0, 0, 0, 1],
            }
        )
        result = _exclusive_lane_summary(frame)
        self.assertTrue(result["conservation_passed"])
        self.assertEqual(sum(value["marginal_truth_pairs"] for value in result["exclusive_combinations"].values()), 4)

    def test_teacher_provenance_is_seven_way_exclusive_and_union_conserved(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_neighbor(root / "neighbors_item2vec.npz", {0: [1, 2, 3, 4]})
            _write_neighbor(root / "neighbors_direct_covisit.npz", {0: [2, 3, 5, 6]})
            _write_neighbor(root / "neighbors_deepwalk.npz", {0: [3, 4, 6, 7]})
            types = np.arange(8, dtype=np.int32)
            garments = np.arange(8, dtype=np.int32)
            frame, audit, _ = _teacher_union(root, 8, types, garments)
            self.assertTrue(audit["provenance_union_conserved"])
            self.assertTrue(set(frame["provenance"]).issubset(set(PROVENANCE_NAMES.values())))
            self.assertEqual(len(frame), 7)


if __name__ == "__main__":
    unittest.main()
