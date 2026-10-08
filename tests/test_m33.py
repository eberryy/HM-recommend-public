import unittest

from hm_recsys.m212 import feature_sets
from hm_recsys.m3 import ANCHOR_NAME
from hm_recsys.m33 import (
    ADAPTIVE_SEASONAL_FEATURES,
    ALL_CUTOFFS,
    ALL_SEASONAL_FEATURES,
    NORMALIZED_PRIOR_FEATURES,
    RAW_PRIOR_FEATURES,
    ROLLING_PROTOCOL,
    validate_rolling_protocol,
    variant_feature_sets,
)


class M33ProtocolTests(unittest.TestCase):
    def test_four_cross_season_protocols_are_strict(self) -> None:
        validate_rolling_protocol()
        self.assertEqual(len(ROLLING_PROTOCOL), 4)
        self.assertTrue(all(value < "2020-09-16" for value in ALL_CUTOFFS))
        self.assertEqual(
            [row["season"] for row in ROLLING_PROTOCOL.values()],
            ["winter", "spring", "early_summer", "late_summer"],
        )

    def test_four_variants_are_exactly_pre_registered(self) -> None:
        variants = variant_feature_sets()
        self.assertEqual(
            list(variants),
            ["anchor", "raw_prior", "normalized_prior", "adaptive_seasonal"],
        )
        anchor = set(feature_sets()[ANCHOR_NAME])
        self.assertEqual(set(variants["anchor"]), anchor)
        self.assertEqual(set(variants["raw_prior"]), anchor | set(RAW_PRIOR_FEATURES))
        self.assertEqual(
            set(variants["normalized_prior"]),
            anchor | set(RAW_PRIOR_FEATURES) | set(NORMALIZED_PRIOR_FEATURES),
        )
        self.assertEqual(
            set(variants["adaptive_seasonal"]), anchor | set(ALL_SEASONAL_FEATURES)
        )

    def test_seasonal_families_are_disjoint_and_new(self) -> None:
        anchor = set(feature_sets()[ANCHOR_NAME])
        families = [
            set(RAW_PRIOR_FEATURES),
            set(NORMALIZED_PRIOR_FEATURES),
            set(ADAPTIVE_SEASONAL_FEATURES),
        ]
        self.assertFalse(anchor.intersection(ALL_SEASONAL_FEATURES))
        self.assertFalse(families[0].intersection(families[1]))
        self.assertFalse(families[0].intersection(families[2]))
        self.assertFalse(families[1].intersection(families[2]))

    def test_plain_calendar_columns_are_not_features(self) -> None:
        self.assertFalse(
            {"month", "week_of_year", "day_of_year"}.intersection(ALL_SEASONAL_FEATURES)
        )


if __name__ == "__main__":
    unittest.main()
