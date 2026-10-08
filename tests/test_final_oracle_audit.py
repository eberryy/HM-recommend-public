import unittest
import numpy as np
from hm_recsys.final_oracle_audit import best_single
from hm_recsys.metrics import apk
from hm_recsys.p42d_stats import single_ap_delta


class OracleTest(unittest.TestCase):
    def test_exhaustive_single(self):
        rng = np.random.default_rng(123)
        warm, cold = list(range(12)), list(range(12, 18))
        for _ in range(50):
            truth = set(np.flatnonzero(rng.random(22) < .3))
            rel = np.array([a in truth for a in warm])
            y = single_ap_delta(np.repeat(rel[None], 6, axis=0),
                                np.array([a in truth for a in cold]), np.repeat(len(truth), 6))
            for limit in (5, 50):
                aps = [apk(list(truth), warm)]
                for i, c in enumerate(cold):
                    for slot in range(12):
                        items = warm.copy()
                        items[slot] = c
                        self.assertAlmostEqual(apk(list(truth), items)-aps[0], y[i, slot])
                        if i+1 <= limit:
                            aps.append(apk(list(truth), items))
                chosen = best_single(warm, cold, np.arange(1, 7), truth, y, limit)
                self.assertAlmostEqual(apk(list(truth), chosen), max(aps))

    def test_no_positive_no_gain(self):
        warm, cold, truth = list(range(12)), [12, 13], {0, 4, 8, 20}
        rel = np.array([i in truth for i in warm])
        y = single_ap_delta(np.repeat(rel[None], 2, axis=0), np.zeros(2), np.repeat(4, 2))
        self.assertTrue((y <= 0).all())
        self.assertEqual(best_single(warm, cold, [1, 2], truth, y), warm)
