import unittest

from hm_recsys.m32 import (
    ALL_NEW_FEATURES,
    REPURCHASE_ROUTE_FEATURES,
    REPEAT_DYNAMICS_FEATURES,
    SEASONAL_PRIOR_FEATURES,
    TREND_ACCELERATION_FEATURES,
    variant_feature_sets,
)
from hm_recsys.m3 import ANCHOR_NAME
from hm_recsys.m212 import feature_sets


class M32FeatureTests(unittest.TestCase):
    def test_new_families_are_disjoint_and_do_not_overlap_anchor(self):
        anchor = set(feature_sets()[ANCHOR_NAME])
        flattened = [
            *REPEAT_DYNAMICS_FEATURES,
            *TREND_ACCELERATION_FEATURES,
            *SEASONAL_PRIOR_FEATURES,
        ]
        self.assertEqual(flattened, ALL_NEW_FEATURES)
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertFalse(anchor.intersection(flattened))

    def test_seven_variants_are_exactly_pre_registered(self):
        variants = variant_feature_sets()
        self.assertEqual(len(variants), 7)
        self.assertEqual(
            list(variants),
            [
                "anchor",
                "without_repurchase_route_features",
                "repeat_dynamics",
                "trend_acceleration",
                "seasonal_prior",
                "all_temporal",
                "repeat_substitute_without_repurchase_route_features",
            ],
        )

    def test_repurchase_retrain_controls_remove_only_route_features(self):
        variants = variant_feature_sets()
        anchor = set(variants["anchor"])
        without = set(variants["without_repurchase_route_features"])
        self.assertEqual(anchor - without, set(REPURCHASE_ROUTE_FEATURES))
        substitute = set(variants["repeat_substitute_without_repurchase_route_features"])
        self.assertEqual(substitute, without | set(REPEAT_DYNAMICS_FEATURES))

    def test_plain_calendar_features_are_excluded(self):
        self.assertFalse(any("month" in feature for feature in ALL_NEW_FEATURES))
        self.assertFalse(any("week_of_year" in feature for feature in ALL_NEW_FEATURES))


if __name__ == "__main__":
    unittest.main()
