import inspect
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p42_data import (
    STATE_COLUMNS, load_cutoff, normalize_cold_confidence,
    normalize_warm_confidence, validate_lineages, validate_warm_identity,
)


class P42DataTest(unittest.TestCase):
    def cold(self):
        return pd.DataFrame({"customer_id": ["cold_only_user"] * 50,
            "b0_rank": np.arange(1, 51), "b0_score": np.arange(50., 0., -1),
            "b0_score_available": 1, "b0_delta_vs_m4": 0., "b0_delta_available": 1})

    def test_cold_normalization_complete_population(self):
        frame = self.cold()
        out = normalize_cold_confidence(frame)
        self.assertEqual(len(out), 50)
        self.assertEqual(out.customer_id.iloc[0], "cold_only_user")
        std = np.std(frame.b0_score.to_numpy(), ddof=0)
        np.testing.assert_allclose(out.normalized_margin_to_rank2, (frame.b0_score - 49.) / std)
        np.testing.assert_allclose(out.normalized_margin_to_rank5, (frame.b0_score - 46.) / std)
        np.testing.assert_allclose(out.normalized_margin_to_user_median, (frame.b0_score - 25.5) / std)
        self.assertEqual(out.b0_user_percentile.iloc[0], 1.)
        self.assertEqual(out.b0_user_percentile.iloc[-1], 0.)

    def test_constant_and_missing_references(self):
        frame = self.cold()
        frame["b0_score"] = 1.
        out = normalize_cold_confidence(frame)
        self.assertTrue(out.normalized_margin_to_rank2.isna().all())
        self.assertTrue(out.normalized_margin_to_rank2_available.eq(0).all())
        frame = self.cold()
        frame.loc[1, "b0_score_available"] = 0
        out = normalize_cold_confidence(frame)
        self.assertTrue(out.normalized_margin_to_rank2.isna().all())
        self.assertTrue(out.normalized_margin_to_rank5.notna().iloc[4])

    def test_warm_top12_population_and_no_score_fallback(self):
        frame = pd.DataFrame({"customer_id": ["a"]*12 + ["b"]*12,
            "warm_model_score": list(range(12))*2,
            "warm_model_score_available": [1]*12 + [0]*12})
        out = normalize_warm_confidence(frame)
        np.testing.assert_allclose(out.warm_user_percentile[:12], np.arange(12)/11.)
        self.assertAlmostEqual(out.warm_user_zscore[:12].std(ddof=0), 1.)
        self.assertTrue(out.warm_user_percentile[12:].isna().all())
        self.assertTrue(out.warm_user_zscore_available[12:].eq(0).all())

    def test_warm_identity_rejects_missing_duplicate_reorder(self):
        frame = pd.DataFrame({"customer_id": ["a"]*12,
            "article_id": [f"{i:010d}" for i in range(12)], "warm_rank": np.arange(1, 13)})
        validate_warm_identity(frame)
        for invalid in (frame.iloc[:-1], frame.iloc[::-1], frame.assign(article_id="same")):
            with self.assertRaises(ValueError):
                validate_warm_identity(invalid)

    def test_lineage_future_and_b1_rejected(self):
        a = {"b0_lineage": {"available": True, "lineage_safe": True,
                "model_training_cutoffs": ["2019-12-25"], "model_label_end": "2020-01-01",
                "model": {"path": "B0_outer.pt"}},
             "warm_lineage": {"available": False, "safe": True}}
        validate_lineages(a, "2020-01-22")
        with self.assertRaises(ValueError):
            validate_lineages(a, "2019-12-31")
        a["b0_lineage"]["model"]["path"] = "B1_outer.pt"
        with self.assertRaises(ValueError):
            validate_lineages(a, "2020-01-22")

    def test_final_week_rejected_before_any_input_read(self):
        for cutoff in ("2020-09-16", "2020-09-17", "2020-09-10"):
            with self.assertRaises(ValueError):
                load_cutoff({}, cutoff)

    def test_six_state_fields_and_no_training_api(self):
        self.assertEqual(len(STATE_COLUMNS), 6)
        self.assertFalse(any("p33" in c or "cosine" in c for c in STATE_COLUMNS))
        self.assertNotIn("fit(", inspect.getsource(load_cutoff))


if __name__ == "__main__":
    unittest.main()
