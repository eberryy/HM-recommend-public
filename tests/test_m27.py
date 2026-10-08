import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hm_recsys.m2 import CATEGORICAL_FEATURES
from hm_recsys.m27 import (
    ALL_FEATURES,
    STAGE1_FEATURES,
    _prepare_variable_groups,
    _score_and_rank_stage1,
)


class _FakeModel:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)

    def predict(self, frame):
        return self.scores


def _stage_frame() -> pd.DataFrame:
    rows = 4
    data = {feature: np.zeros(rows) for feature in ALL_FEATURES}
    data.update(
        {
            "target_cutoff": ["2020-07-22"] * rows,
            "customer_id": ["u1", "u1", "u1", "u2"],
            "article_id": ["a3", "a1", "a2", "a4"],
            "candidate_rank": [103, 101, 102, 101],
            "image_rank": [3, 1, 2, 1],
            "target": [1, 0, 1, 0],
        }
    )
    return pd.DataFrame(data)


class M27Tests(unittest.TestCase):
    def test_stage1_rank_is_per_user_and_deterministic(self):
        frame = _stage_frame()
        maps = {feature: {0: 1} for feature in CATEGORICAL_FEATURES}
        with tempfile.TemporaryDirectory() as temporary:
            evidence = _score_and_rank_stage1(
                frame=frame,
                model=_FakeModel([0.8, 0.8, 0.7, 0.1]),
                category_maps=maps,
                output_path=Path(temporary) / "scores.parquet",
                top_k=1,
            )
        ranked = frame[frame["customer_id"] == "u1"]
        self.assertEqual(ranked["article_id"].tolist(), ["a1", "a3", "a2"])
        self.assertEqual(ranked["stage1_rank"].tolist(), [1, 2, 3])
        self.assertEqual(evidence["selected_rows"], 2)
        self.assertEqual(evidence["selected_positives"], 0)
        self.assertEqual(evidence["positives"], 2)

    def test_variable_joint_groups_accept_strict_and_rescue_sizes(self):
        frame = pd.DataFrame(
            {
                "target_cutoff": ["w1"] * 3 + ["w2"] * 4,
                "customer_id": ["u1"] * 3 + ["u1"] * 4,
                "article_id": ["a1", "a2", "a3", "b1", "b2", "b3", "b4"],
                "candidate_rank": [1, 2, 101, 1, 2, 101, 150],
                "target": [0, 1, 0, 0, 0, 1, 0],
            }
        )
        ordered, sizes, evidence = _prepare_variable_groups(frame, 3, 4)
        self.assertEqual(sizes, [3, 4])
        self.assertEqual(evidence["positive_groups"], 2)
        self.assertEqual(ordered.iloc[0]["target_cutoff"], "w1")
        with self.assertRaisesRegex(ValueError, "invalid_groups=2"):
            _prepare_variable_groups(frame.iloc[:-1], 4, 4)


if __name__ == "__main__":
    unittest.main()
