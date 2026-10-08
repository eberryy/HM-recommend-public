import unittest
import numpy as np
from hm_recsys.warm_v3_joint_representation_audit import swap_gain


class JointTests(unittest.TestCase):
    def test_swap_gain_matches_exact_AP(self):
        for y in ([0,1,0,1,0,1],[1,0,1,0,1,0],[0,0,0,0,1,0]):
            y=np.asarray(y)
            for k in (3,6):
                truth=int(y.sum()+2)
                def ap(a):return float((a[:k]*np.cumsum(a)[:k]/np.arange(1,k+1)).sum()/min(truth,k))
                for i in np.flatnonzero(y):
                    for j in np.flatnonzero(y[:i]==0):
                        z=y.copy();z[i],z[j]=z[j],z[i]
                        self.assertAlmostEqual(swap_gain(y,i,j,truth,k),ap(z)-ap(y))


if __name__=='__main__':unittest.main()
