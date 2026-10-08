import unittest
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from hm_recsys.metrics import apk
from hm_recsys.p42d_stats import single_ap_delta, discrimination, summary, correlations, rank_metrics
from hm_recsys.p42d import logit, clipped


class DiagnosticTests(unittest.TestCase):
    def test_all_relevance_patterns(self):
        r=((np.arange(4096)[:,None] >> np.arange(12))&1).astype(int)
        for y in (0,1):
            n=r.sum(axis=1)+2
            delta=single_ap_delta(r,np.full(len(r),y),n)
            inv=1/np.arange(1,13)
            base=(r*np.cumsum(r,axis=1)*inv).sum(axis=1)/np.minimum(n,12)
            for slot in range(12):
                replaced=r.copy(); replaced[:,slot]=y
                expected=(replaced*np.cumsum(replaced,axis=1)*inv).sum(axis=1)/np.minimum(n,12)-base
                np.testing.assert_allclose(delta[:,slot],expected,rtol=0,atol=4e-16)

    def test_auc_ties(self):
        x=np.array([1,1,2,4,4,4,5,8]); y=np.array([0,1,1,0,1,0,1,0])
        got=discrimination(x,y)
        self.assertAlmostEqual(got['roc_auc'],roc_auc_score(y,x))
        self.assertAlmostEqual(got['pr_auc'],average_precision_score(y,x))

    def test_auc_oneclass(self):
        self.assertIsNone(discrimination([1,2],[0,0])['roc_auc'])

    def test_distribution(self):
        self.assertEqual(summary([1,2,3])['median'],2)
        self.assertIsNone(summary([])['mean'])

    def test_correlations(self):
        self.assertAlmostEqual(correlations([1,2,3],[3,2,1])['spearman'],-1)
        self.assertIsNone(correlations([1,1],[0,0])['pearson'])

    def test_ranking_denominator(self):
        x=rank_metrics([1,5,50]); self.assertAlmostEqual(x['recall']['5'],2/3)

    def test_utility_clipping(self):
        np.testing.assert_array_equal(clipped([0,1]),[1e-6,1-1e-6])
        self.assertEqual(logit([.5])[0],0)

    def test_ap_direct_list(self):
        original=list(range(12)); truth=[1,3,99]
        relevance=np.array([[int(x in truth) for x in original]])
        delta=single_ap_delta(relevance,[1],np.array([3]))
        for slot in range(12):
            new=original.copy(); new[slot]=99
            self.assertAlmostEqual(delta[0,slot],apk(truth,new)-apk(truth,original))


if __name__=='__main__': unittest.main()
