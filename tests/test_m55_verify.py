from __future__ import annotations

import unittest

from hm_recsys.m53 import choose_k
from hm_recsys.m55_verify import _recompute_m55_gates


class M55VerificationTests(unittest.TestCase):
    def test_m53_window_shape_is_projected_before_choose_k(self) -> None:
        windows = {
            "a": {"k_metrics": {str(k): {"warm_positive_retention@k": 0.96} for k in (50, 100, 150, 200, 300)}},
            "b": {"k_metrics": {str(k): {"warm_positive_retention@k": 0.96} for k in (50, 100, 150, 200, 300)}},
        }
        selected, _ = choose_k({name: row["k_metrics"] for name, row in windows.items()})
        self.assertEqual(selected, 50)

    def test_gate_recompute_preserves_strict_promotion_logic(self) -> None:
        windows = {}
        for index, delta in enumerate((0.001, 0.001, 0.001, -0.0001)):
            windows[str(index)] = {
                "final_week": "not_run",
                "evaluations": {
                    "dual_fusion": {
                        "segments": {
                            "overall": {"delta_vs_warm_map@12": delta},
                            "warm_21_plus": {"delta_vs_warm_map@12": delta},
                        },
                        "cold_only_truth_conversion": {"top12": 1 if index < 2 else 0},
                        "inserted_cold_sparse_positive_pairs": 2,
                        "removed_warm_positive_pairs": 1,
                    }
                },
            }
        gates = _recompute_m55_gates({"windows": windows}, True)
        self.assertTrue(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
