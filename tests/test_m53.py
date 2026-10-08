from __future__ import annotations

import unittest

from hm_recsys.m53 import K_VALUES, choose_k


def _windows(values: dict[int, float]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        f"w{index}": {
            str(k): {"warm_positive_retention@k": value}
            for k, value in values.items()
        }
        for index in range(4)
    }


class M53Tests(unittest.TestCase):
    def test_choose_smallest_passing_k(self) -> None:
        values = {50: 0.8, 100: 0.94, 150: 0.95, 200: 0.98, 300: 1.0}
        self.assertEqual(choose_k(_windows(values))[0], 150)

    def test_choose_300_fallback_when_200_fails(self) -> None:
        values = {50: 0.8, 100: 0.9, 150: 0.94, 200: 0.949, 300: 1.0}
        k, reason = choose_k(_windows(values))
        self.assertEqual(tuple(K_VALUES), (50, 100, 150, 200, 300))
        self.assertEqual(k, 300)
        self.assertIn("fallback", reason)
