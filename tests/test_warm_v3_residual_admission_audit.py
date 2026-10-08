import unittest

import numpy as np

from hm_recsys.warm_v3_residual_admission_audit import apk_from_targets, best_one_swap


class ResidualAdmissionAuditTests(unittest.TestCase):
    def test_best_swap_preserves_head_and_improves_tail(self):
        targets = np.zeros(50, dtype=np.uint8)
        targets[0] = 1
        targets[12] = 1
        result = best_one_swap(targets, truth_count=2)
        self.assertEqual(result["victim_rank"], 8)
        self.assertEqual(result["challenger_rank"], 13)
        self.assertGreater(result["oracle_ap"], result["baseline_ap"])
        self.assertAlmostEqual(result["baseline_ap"], apk_from_targets(targets, 2))

    def test_no_truth_challenger_is_noop(self):
        targets = np.zeros(50, dtype=np.uint8)
        targets[0] = 1
        result = best_one_swap(targets, truth_count=1)
        self.assertEqual(result["delta_ap"], 0.0)
        self.assertIsNone(result["victim_rank"])
        self.assertIsNone(result["challenger_rank"])


if __name__ == "__main__":
    unittest.main()
