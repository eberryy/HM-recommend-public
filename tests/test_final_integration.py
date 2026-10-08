import unittest
import numpy as np
from hm_recsys.final_integration import CONTRACT, ROOT, read, relevance, sample_edges, earlier
from hm_recsys.p43a_policy import policies


class FinalIntegrationTests(unittest.TestCase):
    def test_relevance_mapping(self):
        np.testing.assert_array_equal(relevance(np.array([-.1,0,.01,.1,.5])),[0,1,3,4,4])

    def test_all_nonzero_actions_retained(self):
        y=np.array([[0,.1,-.1,0,.0001,-.0001,0,0,0,0,0,0]])
        mask,_=sample_edges('2019-12-25',np.array(['u']),np.array(['i']),y)
        self.assertTrue(mask[y!=0].all())
        again,_=sample_edges('2019-12-25',np.array(['u']),np.array(['i']),y)
        np.testing.assert_array_equal(mask,again)

    def test_strict_label_end(self):
        self.assertEqual(earlier('2020-01-01'),[])
        self.assertEqual(earlier('2020-01-02'),['2019-12-25'])

    @unittest.skipUnless(CONTRACT.exists(),'Local contract required')
    def test_exact_winner_and_policy(self):
        c=read(CONTRACT)
        parent=read(ROOT/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json')
        self.assertEqual(c['model'],next(x for x in parent['model_configs'] if x['id']=='A1-L15-D6-M50'))
        self.assertEqual(c['policy'],next(x for x in policies() if x['id']=='p43a-policy-0332'))
        self.assertEqual(len(c['features']),70)
        self.assertEqual(c['final_week'],'not_run')
        self.assertNotIn('2020-09-16',c['windows'].values())


if __name__=='__main__': unittest.main()
