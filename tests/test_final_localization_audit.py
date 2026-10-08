import unittest
import numpy as np
from scipy.stats import rankdata
from hm_recsys.final_localization_audit import percentile, oracle_edge, decisions
from hm_recsys.p42d_stats import single_ap_delta
from hm_recsys.metrics import apk


class LocalizationTest(unittest.TestCase):
    def test_percentile_ties(self):
        for a in [np.array([1.,1.,2.,4.,4.]),np.ones(3),np.array([3.])]:
            expected=(rankdata(a)-1)/(len(a)-1) if len(a)>1 else np.array([.5])
            np.testing.assert_allclose([percentile(np.sort(a),x) for x in a],expected)

    def test_tail_exhaustive(self):
        rng=np.random.default_rng(13)
        warm=list(range(12));cold=[12,13,14]
        for _ in range(30):
            truth=set(np.flatnonzero(rng.random(18)<.4))
            rel=np.array([a in truth for a in warm])
            y=single_ap_delta(np.repeat(rel[None],3,axis=0),np.array([a in truth for a in cold]),np.repeat(len(truth),3))
            for rows in [np.arange(3),np.array([0,2]),np.array([],dtype=int)]:
                candidates=[0.]
                for i in rows:
                    for j in range(7,12):
                        items=warm.copy();items[j]=cold[i]
                        candidates.append(apk(list(truth),items)-apk(list(truth),warm))
                edge=oracle_edge(y,rows,7)
                self.assertAlmostEqual(0 if edge is None else y[edge],max(candidates))

    def test_truth_override_is_not_slot_oracle(self):
        scores=np.zeros((2,12));scores[0,1]=10;scores[1,10]=9
        y=np.ones((2,12));y[1,10]=0
        actions,_=decisions(scores,np.array([1,6]),np.array([0,1]),y,np.sort(scores.ravel()))
        self.assertEqual(actions['truth_candidate_model_slot'],(1,10))
        self.assertIsNone(actions['truth_candidate_model_slot_gated'])
        self.assertEqual(actions['model_candidate_oracle_slot'],(0,0))
