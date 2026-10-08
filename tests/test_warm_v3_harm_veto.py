import unittest
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hm_recsys.warm_v3_harm_veto import FEATURES, PAIR_KEYS


class HarmVetoContractTest(unittest.TestCase):
    def test_decision_features_exclude_labels(self) -> None:
        self.assertNotIn("relation_label", FEATURES)
        self.assertNotIn("challenger_target", FEATURES)
        self.assertNotIn("victim_target", FEATURES)

    def test_pair_identity_is_label_free(self) -> None:
        self.assertEqual(PAIR_KEYS, ["customer_id", "challenger_article_id", "victim_article_id", "challenger_rank", "victim_rank"])

    def test_harm_veto_is_subset_operation(self) -> None:
        actions = pd.DataFrame({"predicted_relation_class": [0, 1, 2], "x": [1, 2, 3]})
        retained = actions.loc[actions.predicted_relation_class != 0]
        self.assertEqual(retained.x.tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
