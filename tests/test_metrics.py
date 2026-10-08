import unittest

from hm_recsys.metrics import apk, hit_rate_at_k, mapk, oracle_mapk, recall_at_k


class MetricTests(unittest.TestCase):
    def test_apk_rewards_early_hits(self):
        self.assertAlmostEqual(apk([1, 2], [1, 9, 2], 3), (1 + 2 / 3) / 2)

    def test_apk_does_not_double_count_duplicates(self):
        self.assertAlmostEqual(apk([1, 2], [1, 1, 2], 3), (1 + 2 / 3) / 2)

    def test_candidate_metrics_and_oracle(self):
        truth = {"u1": [1, 2], "u2": [3]}
        candidates = {"u1": [9, 1], "u2": [8, 7]}
        self.assertAlmostEqual(recall_at_k(truth, candidates, 2), 0.25)
        self.assertAlmostEqual(hit_rate_at_k(truth, candidates, 2), 0.5)
        self.assertAlmostEqual(oracle_mapk(truth, candidates, 2, 2), 0.25)
        self.assertAlmostEqual(mapk(truth, {"u1": [1], "u2": []}, 2), 0.25)


if __name__ == "__main__":
    unittest.main()
