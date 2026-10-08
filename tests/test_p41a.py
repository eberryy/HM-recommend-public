import inspect
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p41a_contract import guard_cutoff
from hm_recsys.p41a_oracle import best_choices
from hm_recsys.p41a_stats import (
    LCM, ap_units, auc_metrics, normalize_cold50, replacement_units,
    reliability, unlabeled_bins,
)


class P41ATest(unittest.TestCase):
    def test_exact_delta_exhaustive(self):
        relevance = (np.arange(4096)[:, None] >> np.arange(12)) & 1
        for slot in range(1, 13):
            for cold in (0, 1):
                changed = relevance.copy()
                changed[:, slot-1] = cold
                observed = replacement_units(relevance, cold, slot)
                np.testing.assert_array_equal(observed, ap_units(changed)-ap_units(relevance))
                np.testing.assert_array_equal(np.sign(observed), cold-relevance[:, slot-1])

    def test_no_admission_zero_and_harmful(self):
        r = np.ones((1, 12), int)
        self.assertLess(replacement_units(r, 0, 12)[0], 0)
        self.assertEqual(replacement_units(r, 1, 12)[0], 0)
        self.assertEqual(ap_units(r)[0]/(12*LCM), 1)

    def test_cutoff_rejects_final_and_overlap(self):
        for cutoff in ("2020-09-16", "2020-09-17", "2021-01-01", "2020-09-10"):
            with self.assertRaises(ValueError):
                guard_cutoff(cutoff)
        self.assertEqual(guard_cutoff("2020-08-19"), "2020-08-19")

    def test_auc_ties_and_direction(self):
        self.assertEqual(auc_metrics([1, 2, 3], [0, 0, 1])["roc_auc"], 1)
        self.assertEqual(auc_metrics([1, 2, 3], [1, 0, 0])["roc_auc"], 0)
        tied = auc_metrics([1, 1, 1, 1], [1, 0, 0, 0])
        self.assertEqual(tied["roc_auc"], .5)
        self.assertEqual(tied["pr_auc"], .25)
        self.assertIsNone(auc_metrics([1, 2], [0, 0])["roc_auc"])
        self.assertEqual(auc_metrics([np.nan, 1, 2], [1, 0, 1])["eligible_rows"], 2)

    def test_auc_against_pairwise_enumeration(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.integers(-2, 4, 21)
            y = rng.integers(0, 2, 21).astype(bool)
            pairdiff = x[y][:, None]-x[~y][None, :]
            expected = ((pairdiff > 0)+.5*(pairdiff == 0)).mean()
            self.assertAlmostEqual(auc_metrics(x, y)["roc_auc"], expected, places=15)

    def test_unlabeled_bins_keep_ties(self):
        self.assertEqual(list(inspect.signature(unlabeled_bins).parameters), ["score", "bins"])
        x = np.array([0, 0, 0, 0, 0, 1, 1, 3, 3, 3])
        q, ids = unlabeled_bins(x, 4)
        for v in np.unique(x):
            self.assertEqual(len(set(ids[x == v])), 1)
        y = np.arange(10) % 2
        first, second = reliability(x, y, 4), reliability(x, 1-y, 4)
        self.assertEqual(first["boundaries"], second["boundaries"])
        self.assertEqual(sum(row["rows"] for row in first["bins"]), len(x))

    def test_normalization_complete_population_and_missing(self):
        frame = pd.DataFrame({"customer_id": ["u"]*5+["v"]*5,
            "b0_rank": [1, 2, 3, 4, 5]*2, "b0_score": [5, 4, 3, 2, 1]+[1]*5,
            "b0_score_available": [1]*10, "b0_delta_vs_m4": [0]*10,
            "b0_delta_available": [1]*10})
        n = normalize_cold50(frame)
        np.testing.assert_allclose(n.b0_user_percentile[:5], [1, .75, .5, .25, 0])
        self.assertEqual(n.margin_to_rank2.iloc[1], 0)
        self.assertEqual(n.margin_to_rank5.iloc[4], 0)
        self.assertAlmostEqual(n.b0_user_zscore[:5].std(ddof=0), 1)
        self.assertTrue(n.b0_user_zscore_available[5:].eq(0).all())
        frame.loc[1, "b0_score_available"] = 0
        self.assertTrue(normalize_cold50(frame).margin_to_rank2_available[:5].eq(0).all())

    def test_tie_break_and_frozen_primary(self):
        f = pd.DataFrame({"user_index": [0, 0, 0, 0, 1], "article_id": ["b", "a", "a", "z", "x"],
            "cold_rank": [2, 1, 1, 20, 1], "warm_slot_rank": [10, 11, 10, 1, 12],
            "delta_ap_units": [20, 20, 20, 100, 2]})
        selected = best_choices(f, 10, [10, 11, 12])
        self.assertEqual(selected.iloc[0].article_id, "a")
        self.assertEqual(selected.iloc[0].warm_slot_rank, 10)
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main()
