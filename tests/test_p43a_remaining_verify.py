import unittest
import numpy as np
from hm_recsys.p43a_remaining_verify import verify_pairs


class RemainingAuditTests(unittest.TestCase):
    def test_rebuild_and_noop(self):
        warm=list('abcdefghijkl'); articles=np.array(['x','y']); owners=np.array([0,1])
        pred,pairs=verify_pairs(warm,articles,owners,0,[[0,11],[-1,-1]])
        self.assertEqual(pred,warm[:11]+['x']);self.assertEqual(len(pairs),1)
        self.assertEqual(verify_pairs(warm,articles,owners,0,[[-1,-1]])[0],warm)

    def test_reject_other_user_duplicate_and_invalid_slot(self):
        for pairs in ([[1,0]],[[0,0],[0,1]],[[0,12]],[[2,0]]):
            with self.assertRaises(AssertionError):
                verify_pairs(list('abcdefghijkl'),np.array(['x','y']),np.array([0,1]),0,pairs)


if __name__=='__main__':unittest.main()
