import numpy as np
from hm_recsys.final_b0_admission_e4 import stable_percentile


def test_stable_percentile_is_deterministic_and_ascending():
    result = stable_percentile(np.array([2.0, 1.0, 2.0]))
    np.testing.assert_allclose(result, [2 / 3, 1 / 3, 1.0])
