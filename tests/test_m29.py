from __future__ import annotations

import unittest

from hm_recsys.m29 import (
    FEATURE_SETS,ITEM2VEC_FEATURES,_summary,validate_group_sizes,
)


class M29Tests(unittest.TestCase):
    def test_feature_control_uses_same_base_and_only_full_adds_item2vec(self)->None:
        control=FEATURE_SETS["expanded_full_no_item2vec"]
        full=FEATURE_SETS["expanded_full_plus_item2vec"]
        self.assertEqual(full[:len(control)],control)
        self.assertEqual(full[len(control):],ITEM2VEC_FEATURES)
        self.assertEqual(len(full),len(set(full)))

    def test_variable_groups_enforce_100_to_300(self)->None:
        self.assertEqual(validate_group_sizes([100,201,300])["rows"],601)
        with self.assertRaises(ValueError): validate_group_sizes([99,100])
        with self.assertRaises(ValueError): validate_group_sizes([301])

    def test_summary_promotes_only_two_window_improvement(self)->None:
        names=["expanded_append_order"]+[name for variant in FEATURE_SETS for name in (variant,f"{variant}__inactive_rrf")]
        scores={name:(0.09,0.19) for name in names}
        scores["expanded_full_plus_item2vec__inactive_rrf"]=(0.11,0.21)
        development={}
        frozen={}
        for index,dev in enumerate(("dev_a","dev_b")):
            orderings={}
            for name in names:
                value=scores[name][index]
                orderings[name]={"segments":{"overall":{"map@12":value,"oracle_map@12":0.3,
                    "candidate_recall@expanded_pool":0.25}},
                    "activity_segments":[{"activity_segment":"low","map@12":value}],
                    "popularity_segments":{"head":{"map@12":value,"oracle_map@12":0.3,
                        "candidate_recall@expanded_pool":0.25}}}
            development[dev]={"evaluation":{"orderings":orderings}}
            base=0.1 if dev=="dev_a" else 0.2
            frozen[dev]={"segments":{"overall":{"map@12":base,"oracle_map@12":0.2}},
                "activity_segments":[{"activity_segment":"low","map@12":base}]}
        result=_summary(development,frozen,12)
        self.assertTrue(result["selection_gate"]["accepted"])
        self.assertEqual(result["selection_gate"]["selected_variant"],
            "expanded_full_plus_item2vec__inactive_rrf")


if __name__=="__main__": unittest.main()
