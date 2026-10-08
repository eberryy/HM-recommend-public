from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hm_recsys.warm_v3_positive_query_fusion import FEATURES, select_positive_groups, three_view_scores


class PositiveQueryFusionTest(unittest.TestCase):
    def test_group_selection_keeps_all_rows_of_positive_group(self) -> None:
        frame = pd.DataFrame({"source_window": ["a"] * 4, "customer_id": ["u", "u", "v", "v"], "target": [1, 0, 0, 0]})
        selected = select_positive_groups(frame)
        self.assertEqual(selected.customer_id.tolist(), ["u", "u"])
        self.assertEqual(selected.target.tolist(), [1, 0])

    def test_target_is_not_a_feature(self) -> None:
        self.assertNotIn("target", FEATURES)

    def test_three_view_score_is_label_free_and_complete(self) -> None:
        frame = pd.DataFrame({"customer_id": ["u", "u"], "article_id": ["a", "b"]})
        score = three_view_scores(frame, [2.0, 1.0], [1.0, 2.0], [2.0, 1.0])
        self.assertEqual(len(score), 2)
        self.assertGreater(score.loc[score.article_id == "a", "ranking_score"].iloc[0], score.loc[score.article_id == "b", "ranking_score"].iloc[0])


if __name__ == "__main__":
    unittest.main()
