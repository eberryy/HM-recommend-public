import unittest
from datetime import date,timedelta
from hm_recsys.warm_v2_fresh import allowed,FreshEngine
from hm_recsys.warm_v2_fresh_preflight import CHAINS
from hm_recsys.warm_v2_engine import Engine
from hm_recsys.warm_v2_bpr import PARAMS
from hm_recsys.warm_v2_fresh_review import verdict


class FreshTests(unittest.TestCase):
    def test_fixed_fresh_gate_and_strong_gate(self):
        self.assertEqual(verdict([.0005]*4),'strong_pass')
        self.assertEqual(verdict([.0008,.0008,.0008,-.0002]),'robust_pass')
        self.assertEqual(verdict([.0004]*4),'weak_generalization')
        self.assertEqual(verdict([.005,-.001,-.001,-.001]),'weak_generalization')
        self.assertEqual(verdict([-.0001]*4),'fresh_robustness_failed')
        self.assertEqual(verdict([0.]*4),'fresh_robustness_failed')

    def test_exact_explicit_chains(self):
        self.assertEqual(len({d for chain in CHAINS for d in chain}),12)
        self.assertEqual([v[2] for v in CHAINS],['2019-02-20','2019-05-22','2019-08-21','2019-11-20'])
        for t,v,o in CHAINS:
            self.assertEqual(date.fromisoformat(v)-date.fromisoformat(t),timedelta(days=28))
            self.assertEqual(date.fromisoformat(o)-date.fromisoformat(v),timedelta(days=28))
            for d in (t,v,o):allowed(d)

    def test_no_unregistered_or_final_cutoff(self):
        for d in ('2020-09-16','2020-01-22','2019-12-18'):
            with self.assertRaises(ValueError):allowed(d)

    def test_inherits_exact_ranker_and_sampler(self):
        for method in ('cached_data','dataset','train_inner','train_outer','score'):
            self.assertIs(getattr(FreshEngine,method),getattr(Engine,method))
        self.assertEqual(PARAMS,{'factors':100,'learning_rate':.01,'regularization':.01,'iterations':100,
            'num_threads':8,'verify_negative_samples':True,'random_state':20260908})


if __name__=='__main__':unittest.main()
