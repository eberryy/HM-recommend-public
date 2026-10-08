import unittest

import numpy as np

from hm_recsys.warm_v3_bpr_safe_admission import replace_rank12


class BprSafeAdmissionTest(unittest.TestCase):
    def test_replacing_empty_rank12_with_truth_is_positive(self):
        values = np.zeros(15, dtype=np.uint8)
        self.assertAlmostEqual(replace_rank12(values, 1, 2), 1 / 24)

    def test_replacing_existing_truth_with_nontruth_is_negative(self):
        values = np.zeros(15, dtype=np.uint8)
        values[11] = 1
        self.assertAlmostEqual(replace_rank12(values, 0, 1), -1 / 12)

    def test_same_target_is_neutral(self):
        values = np.zeros(15, dtype=np.uint8)
        self.assertEqual(replace_rank12(values, 0, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
