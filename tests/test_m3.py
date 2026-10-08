import unittest

from hm_recsys.m3 import (
    ANCHOR_NAME,
    FALLBACK_NAME,
    ROLLING_PROTOCOL,
    summarize_m3,
    validate_rolling_protocol,
)


def _ordering(value: float) -> dict:
    segment = {
        "map@12": value,
        "candidate_recall@expanded_pool": 0.15,
        "candidate_hit_rate@expanded_pool": 0.30,
        "oracle_map@12": 0.16,
    }
    return {
        "segments": {
            "overall": dict(segment),
            "warm": {"map@12": value + 0.001},
            "cold": {"map@12": 0.0},
        },
        "activity_segments": [
            {"activity_segment": "inactive_12w", "map@12": 0.006},
            {"activity_segment": "low_1_5", "map@12": value},
        ],
        "popularity_segments": {
            "head_top20pct_items": {"map@12": value + 0.002},
            "tail_bottom40pct_items": {"map@12": value - 0.002},
        },
    }


class M3ProtocolTests(unittest.TestCase):
    def test_three_window_protocol_is_strictly_temporal(self):
        validate_rolling_protocol()
        self.assertEqual(len(ROLLING_PROTOCOL), 3)
        self.assertEqual(
            [row["outer_validation"] for row in ROLLING_PROTOCOL.values()],
            ["2020-06-24", "2020-07-22", "2020-08-19"],
        )

    def test_summary_requires_positive_gain_in_every_window_and_checks_parity(self):
        anchor_values = {
            "roll_20200624": 0.023,
            "roll_20200722": 0.024,
            "roll_20200819": 0.025,
        }
        development = {}
        rounds = {"roll_20200624": 20, "roll_20200722": 37, "roll_20200819": 67}
        for name, value in anchor_values.items():
            development[name] = {
                "evaluation": {
                    "orderings": {
                        "expanded_append_order": _ordering(value - 0.003),
                        ANCHOR_NAME: _ordering(value - 0.001),
                        FALLBACK_NAME: _ordering(value),
                    }
                },
                "outer_model": {"selected_rounds_from_inner": rounds[name]},
            }
        m212 = {
            "development": {
                "dev_a": {
                    "evaluation": {"orderings": {FALLBACK_NAME: _ordering(0.024)}},
                    "outer_models": {ANCHOR_NAME: {"selected_rounds_from_inner": 37}},
                },
                "dev_b": {
                    "evaluation": {"orderings": {FALLBACK_NAME: _ordering(0.025)}},
                    "outer_models": {ANCHOR_NAME: {"selected_rounds_from_inner": 67}},
                },
            }
        }
        result = summarize_m3(development, m212, 12)
        self.assertTrue(result["stability_gate"]["accepted"])
        self.assertTrue(result["m2_12_reproduction"]["all_passed"])
        self.assertAlmostEqual(
            result["orderings"][FALLBACK_NAME]["mean_map@12"], 0.024
        )


if __name__ == "__main__":
    unittest.main()
