import unittest
import numpy as np
import pandas as pd
from hm_recsys.final_binary_admission_e5 import sampled_mask


class BinaryAdmissionTest(unittest.TestCase):
    def test_sampling_always_retains_positives(self):
        keys = pd.DataFrame({"customer_id": ["u1", "u2"], "article_id": ["a", "b"]})
        mask = sampled_mask(keys, np.array([1, 0]), 10**9)
        self.assertTrue(mask[0])


if __name__ == "__main__":
    unittest.main()
