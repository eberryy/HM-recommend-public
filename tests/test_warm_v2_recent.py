import unittest
from hm_recsys.warm_v2_contract import REPORT, read
from hm_recsys.warm_v2_recent_train import recent_protocol
from hm_recsys.warm_v2_freshness import week_before


class RecentTests(unittest.TestCase):
    @unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
    def test_shift_only_training_no_overlap(self):
        old=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')['rolling_protocol']
        new=recent_protocol(old)
        for key,p in new.items():
            self.assertEqual(p['inner_validation'],old[key]['inner_validation'])
            self.assertEqual(p['outer_validation'],old[key]['outer_validation'])
            self.assertEqual(len(p['inner_train']),1)
            self.assertEqual(len(p['outer_train']),2)
            self.assertEqual(p['inner_train'][0],week_before(p['inner_validation'],1))
            self.assertEqual(p['outer_train'][-1],week_before(p['outer_validation'],1))
            self.assertEqual(p['outer_train'][0],p['inner_train'][0])
        self.assertEqual(old['winter_20200122']['inner_train'],['2019-11-27'])

    def test_final_guard(self):
        with self.assertRaises(ValueError):week_before('2020-09-16',1)


if __name__=='__main__':unittest.main()
