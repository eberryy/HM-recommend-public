import unittest
from datetime import date
from hm_recsys.warm_v3_temporal_ensemble import assert_causal,predecessor_map
from hm_recsys import warm_v3_common as common


@unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
class TemporalEnsembleTests(unittest.TestCase):
    def test_predecessor_is_adjacent_and_winter_is_noop(self):
        engine=common.Engine(2020);m=predecessor_map(engine.contract['rolling_protocol']);windows=list(m)
        self.assertIsNone(m[windows[0]])
        for i in range(1,len(windows)):self.assertEqual(m[windows[i]],windows[i-1])

    def test_real_previous_checkpoints_end_before_targets(self):
        engine=common.Engine(2020);m=predecessor_map(engine.contract['rolling_protocol'])
        for window,p in engine.contract['rolling_protocol'].items():
            if m[window] is None:continue
            for suffix in ('000','501'):
                evidence=assert_causal(common.OLD/f'WV2-{suffix}'/m[window],p['inner_validation'])
                self.assertLessEqual(date.fromisoformat(evidence['latest_previous_training_label_end_exclusive']),
                    date.fromisoformat(p['inner_validation']))


if __name__=='__main__':unittest.main()
