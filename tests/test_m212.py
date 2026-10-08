from __future__ import annotations

import unittest

import numpy as np

from hm_recsys.m210 import M29_FEATURES, TARGET_AWARE_FEATURES
from hm_recsys.m212 import (
    DECAY_FEATURES,
    INNER_PROTOCOL,
    REMAINING_FAMILIES,
    exact_group_map_at_k,
    feature_sets,
    select_inner_variant,
    validate_inner_protocol,
)


class M212FeatureTests(unittest.TestCase):
    def test_remaining_families_partition_no_decay_target_features(self) -> None:
        flattened = [
            feature for values in REMAINING_FAMILIES.values() for feature in values
        ]
        self.assertEqual(len(flattened), 21)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            set(flattened), set(TARGET_AWARE_FEATURES) - set(DECAY_FEATURES)
        )
        sets = feature_sets()
        self.assertEqual(sets["m29_only"], list(M29_FEATURES))
        for family, removed in REMAINING_FAMILIES.items():
            self.assertEqual(
                set(sets[f"no_decay_without_{family}"]),
                set(sets["no_decay_anchor"]) - set(removed),
            )

    def test_inner_protocol_is_strictly_temporal_and_matches_outer(self) -> None:
        validate_inner_protocol()
        self.assertEqual(INNER_PROTOCOL["dev_a"]["inner_validation"], "2020-06-24")
        self.assertEqual(INNER_PROTOCOL["dev_b"]["outer_validation"], "2020-08-19")


class M212MetricTests(unittest.TestCase):
    def test_exact_map_uses_full_truth_denominator(self) -> None:
        # group 1 top-2 labels are [1, 0], AP=(1/1)/2 because truth_count=2;
        # group 2 top-2 labels are [1, 1], AP=(1 + 2/2)/2 = 1.
        predictions = np.array([0.9, 0.8, 0.7, 0.1, 0.9])
        labels = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
        value = exact_group_map_at_k(
            predictions,
            labels,
            [3, 2],
            np.array([2, 1]),
            k=2,
        )
        self.assertAlmostEqual(value, 0.75)

    def test_exact_map_ties_follow_candidate_order(self) -> None:
        value = exact_group_map_at_k(
            np.array([0.5, 0.5, 0.5]),
            np.array([0, 1, 0], dtype=np.uint8),
            [3],
            np.array([1]),
            k=2,
        )
        self.assertAlmostEqual(value, 0.5)

    def test_inner_selector_prefers_score_then_fewer_rounds(self) -> None:
        evidence = {
            "a": {"best_score": 0.2, "best_iteration": 15},
            "b": {"best_score": 0.2, "best_iteration": 10},
            "c": {"best_score": 0.1, "best_iteration": 1},
        }
        self.assertEqual(select_inner_variant(evidence), "b")


if __name__ == "__main__":
    unittest.main()
