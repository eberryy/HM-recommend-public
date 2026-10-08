import unittest

from hm_recsys.warm_v3_multiview_joint import CANONICAL_PARAMS, EXTRA, FAMILIES, MultiViewEngine


class MultiViewJointTests(unittest.TestCase):
    def test_frozen_architecture_contract(self):
        self.assertEqual(FAMILIES, ['bpr_match', 'sequence', 'lightgcn'])
        self.assertEqual(len(EXTRA), 6)
        self.assertEqual(CANONICAL_PARAMS['feature_count'], 90)
        self.assertFalse(CANONICAL_PARAMS['fusion_threeway'])

    @unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
    def test_all_inner_lightgcn_caches_are_exact_and_complete(self):
        engine = MultiViewEngine(2020)
        for protocol in engine.contract['rolling_protocol'].values():
            for cutoff in [*protocol['inner_train'], protocol['inner_validation']]:
                path, meta = engine.signal_path('lightgcn', cutoff)
                self.assertTrue(path.is_file())
                self.assertTrue(meta['candidate_identity_unchanged'])


if __name__ == '__main__':
    unittest.main()
