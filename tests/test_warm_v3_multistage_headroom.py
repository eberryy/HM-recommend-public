import unittest
import numpy as np
from hm_recsys.warm_v3_multistage_headroom_audit import pair_counts,safe_prior


class HeadroomTests(unittest.TestCase):
    def test_temporal_boundary(self):
        self.assertTrue(safe_prior('2019-11-20','2019-11-27'))
        self.assertFalse(safe_prior('2019-11-20','2019-11-26'))

    def test_analytical_sensitive_pairs_match_exact_swaps(self):
        for y in ([1,0,0,1,0,1],[0,0,1,0,1,0],[0,0,0,0,0,1],[1,1,1,0,0,0]):
            labels=np.asarray(y)
            k=3
            def ap(a):
                return float((a[:k]*np.cumsum(a)[:k]/np.arange(1,k+1)).sum()/min(a.sum(),k))
            actual=0
            total=0
            before=ap(labels)
            for i in np.flatnonzero(labels):
                for j in np.flatnonzero(labels==0):
                    z=labels.copy();z[i],z[j]=z[j],z[i]
                    total+=1
                    actual+=abs(ap(z)-before)>1e-12
            self.assertEqual(pair_counts(int(labels[:k].sum()),int(labels.sum()),k,len(labels)),(total,actual))


if __name__=='__main__':
    unittest.main()
