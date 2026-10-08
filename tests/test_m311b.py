import unittest

import numpy as np
import pandas as pd

from hm_recsys.m311b import (
    M33_MODEL_KEY,
    MAX_REPLACEMENTS,
    _apply_replacements,
    _assign_active_base_rank,
    _build_pair_frame,
    _gate,
    _normalize_evaluation_rank,
    _select_high_precision_subpool,
    _select_replacements,
)


class M311BTests(unittest.TestCase):
    def test_m33_model_key_is_distinct_from_feature_variant_name(self):
        self.assertEqual(M33_MODEL_KEY, "anchor")

    def test_outer_mutable_slots_use_model_score_not_raw_candidate_rank(self):
        frame = pd.DataFrame({
            "customer_id": ["u"] * 13,
            "article_id": [f"i{x:02d}" for x in range(1, 14)],
            "candidate_rank": list(range(1, 14)),
            "target": [0] * 13,
            "score_anchor": [float(x) for x in range(1, 14)],
        })
        mutable = _assign_active_base_rank(frame)
        self.assertEqual(mutable.base_rank.tolist(), [8, 9, 10, 11, 12])
        self.assertEqual(mutable.article_id.tolist(), ["i06", "i05", "i04", "i03", "i02"])
        self.assertNotIn("i08", set(mutable.article_id))

    def test_evaluation_uses_final_base_rank_when_present(self):
        frame = pd.DataFrame({
            "candidate_rank": [99, 2],
            "base_final_rank": [1, 2],
            "target": [1, 0],
        })
        normalized = _normalize_evaluation_rank(frame)
        self.assertEqual(normalized.candidate_rank.tolist(), [1, 2])

    def test_subpool_is_personalized_bounded_union(self):
        rows = []
        for rank in range(1, 21):
            rows.append({
                "customer_id": "u1",
                "article_id": f"i{rank:02d}",
                "base_present": 0,
                "soft_present": 1,
                "soft_personalized_present": 1,
                "soft_personalized_rank": rank,
                "direct_visual_decay_max": float(rank),
                "user_history_events_12w": 1,
            })
        rows.append({
            "customer_id": "u2", "article_id": "not-personalized", "base_present": 0,
            "soft_present": 1, "soft_personalized_present": 0,
            "soft_personalized_rank": np.nan, "direct_visual_decay_max": 99.0,
            "user_history_events_12w": 1,
        })
        selected = _select_high_precision_subpool(pd.DataFrame(rows))
        self.assertEqual(selected.customer_id.unique().tolist(), ["u1"])
        self.assertEqual(set(selected.article_id), {*(f"i{x:02d}" for x in range(1, 11)), *(f"i{x:02d}" for x in range(16, 21))})
        self.assertEqual(len(selected), 15)

    def test_pair_labels_keep_only_directional_examples(self):
        from hm_recsys.m311b import COMPARISON_VALUE_COLUMNS, DIRECT_FEATURES, IMAGE_SOURCE_COLUMNS

        image = {"customer_id": "u", "article_id": "img", "target": 1, "anchor_score": 0.2, "subpool_rank": 1, "personalized_decay_rank": 1}
        base_good = {"customer_id": "u", "article_id": "base-good", "target": 1, "anchor_score": 0.3, "base_rank": 8}
        base_bad = {"customer_id": "u", "article_id": "base-bad", "target": 0, "anchor_score": 0.1, "base_rank": 9}
        for name in IMAGE_SOURCE_COLUMNS + DIRECT_FEATURES:
            image[name] = 1.0
        for name in COMPARISON_VALUE_COLUMNS:
            image[name] = 2.0
            base_good[name] = 1.0
            base_bad[name] = 1.0
        pairs = _build_pair_frame(pd.DataFrame([image]), pd.DataFrame([base_good, base_bad]), training=True)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(int(pairs.iloc[0].pair_target), 1)
        self.assertAlmostEqual(float(pairs.iloc[0].anchor_score_gap), 0.1)

    def test_bounded_matching_is_unique_and_recovers_two_compatible_edges(self):
        pairs = pd.DataFrame({
            "customer_id": ["u"] * 4,
            "image_article_id": ["i1", "i1", "i2", "i3"],
            "base_article_id": ["b12", "b11", "b12", "b10"],
            "base_rank": [12, 11, 12, 10],
            "image_target": [1, 1, 0, 0],
            "base_target": [0, 0, 0, 0],
        })
        selected = _select_replacements(pairs, np.array([0.9, 0.8, 0.85, 0.4]))
        self.assertEqual(len(selected), MAX_REPLACEMENTS)
        self.assertEqual(selected.image_article_id.nunique(), 2)
        self.assertEqual(selected.base_article_id.nunique(), 2)
        self.assertTrue((selected.comparison_score > 0.5).all())

    def test_apply_replacements_preserves_protected_slots(self):
        base = pd.DataFrame({
            "customer_id": ["u"] * 12,
            "article_id": [f"b{x}" for x in range(1, 13)],
            "base_final_rank": list(range(1, 13)),
            "target": [0] * 12,
        })
        selected = pd.DataFrame({
            "customer_id": ["u"], "image_article_id": ["img"], "base_article_id": ["b10"],
            "base_rank": [10], "image_target": [1], "base_target": [0], "comparison_score": [0.9],
        })
        images = pd.DataFrame({"customer_id": ["u"], "article_id": ["img"], "target": [1]})
        final = _apply_replacements(base, selected, images)
        self.assertEqual(final.loc[final.candidate_rank <= 7, "article_id"].tolist(), [f"b{x}" for x in range(1, 8)])
        self.assertEqual(final.loc[final.candidate_rank == 10, "article_id"].item(), "img")
        self.assertNotIn("b10", set(final.article_id))

    def test_baseline_gate_requires_stable_map_and_mechanism(self):
        accepted = _gate({"a": 0.001, "b": 0.001, "c": 0.001, "d": 0.0}, {"a": 3, "b": 2, "c": 0, "d": 0})
        self.assertTrue(accepted["accepted_as_new_baseline"])
        rejected = _gate({"a": 0.001, "b": 0.001, "c": 0.001, "d": -0.0001}, {"a": 3, "b": 2, "c": 0, "d": 0})
        self.assertFalse(rejected["accepted_as_new_baseline"])


if __name__ == "__main__":
    unittest.main()
