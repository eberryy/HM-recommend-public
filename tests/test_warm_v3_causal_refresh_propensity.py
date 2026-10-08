import unittest
from datetime import date

from hm_recsys.warm_v3_causal_refresh_propensity import label_end


class CausalRefreshPropensityTests(unittest.TestCase):
    def test_label_week_ends_seven_days_after_cutoff(self):
        self.assertEqual(label_end("2020-01-01"), date(2020, 1, 8))


if __name__ == "__main__":
    unittest.main()
