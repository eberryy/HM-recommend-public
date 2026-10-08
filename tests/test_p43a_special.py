"""Synthetic exact-matching and label-order tests for the frozen F adapter."""
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hm_recsys.p43a_contract import F_FEATURES,f_configs
from hm_recsys.p43a_data import META_COLUMNS
from hm_recsys import p43a_special as special
from hm_recsys.p41a_contract import read_json


def brute_weight(weights,allowed,cap):
    edges=list(zip(*np.nonzero(allowed)))
    best=0.
    for k in range(1,min(cap,len(edges))+1):
        for subset in itertools.combinations(edges,k):
            if len({a for a,b in subset})==k and len({b for a,b in subset})==k:
                best=max(best,sum(weights[a,b] for a,b in subset))
    return best


def test_augmented_at_most_k_matches_independent_exhaustive_small_graphs():
    rng=np.random.default_rng(17)
    for _ in range(25):
        weights=rng.integers(1,20,size=(3,3)).astype(float)
        allowed=rng.random((3,3))>.35
        for cap in range(4):
            matches=special.at_most_k_matching(weights,allowed,cap)
            assert sum(weights[a,b] for a,b in matches)==brute_weight(weights,allowed,cap)
            assert len(matches)<=cap
            assert len({a for a,b in matches})==len(matches)
            assert len({b for a,b in matches})==len(matches)


def test_f_cap_solver_does_not_force_larger_cardinality_or_use_forbidden_edges():
    weights=np.array([[100.,1.],[1.,0.]])
    allowed=np.array([[True,True],[True,False]])
    assert special.at_most_k_matching(weights,allowed,2)==[(0,0)]
    assert special.at_most_k_matching(weights,np.zeros_like(allowed),2)==[]
    assert special.at_most_k_matching(np.empty((0,12)),np.empty((0,12),bool),12)==[]
    with unittest.TestCase().assertRaises(ValueError):
        special.at_most_k_matching(np.zeros((1,1)),np.ones((1,1),bool),1)


def test_f_matching_tie_is_deterministic_without_truth():
    weights=np.ones((4,12));allowed=np.ones_like(weights,bool)
    a=special.at_most_k_matching(weights,allowed,3)
    assert all(special.at_most_k_matching(weights,allowed,3)==a for _ in range(5))


def test_cross_user_rank_keeps_real_zero_and_separates_missing_denominator():
    values,audit=special.cross_user_percentile([0.,.5,np.nan])
    np.testing.assert_array_equal(values,[0.,1.,0.])
    assert audit['total_users']==3 and audit['available_users']==2 and audit['missing_users']==1
    values,audit=special.cross_user_percentile([3.,3.,3.])
    np.testing.assert_array_equal(values,[.5,.5,.5])
    values,audit=special.cross_user_percentile([np.nan,1.])
    np.testing.assert_array_equal(values,[0.,.5])


def test_f_features_use_all_w0_users_before_broadcast_and_no_truth_fields():
    data=dict(users=np.array(['u0','u1','u2']),
        cold=pd.DataFrame(dict(user_index=[0,0,1,1],b0_user_percentile=[.9,.1,.8,.2],
            qC_within_user_percentile=[.9,.1,.8,.2],qC_relative_available=[1,1,1,1])),
        warm=pd.DataFrame(dict(qW_within_user_percentile=np.full(36,.5),qW_relative_available=np.ones(36))),
        state=pd.DataFrame(dict(customer_id=['u0','u1','u2'],novel_purchase_share_0_5=[0.,.5,1.],
            user_past_purchase_count=[0,10,20])))
    meta=np.zeros((48,len(META_COLUMNS)),np.float32)
    x,audit=special.blend_feature_matrix(data,meta)
    assert x.shape==(48,10)
    np.testing.assert_array_equal(x[:24,F_FEATURES.index('novelty')],np.zeros(24))
    np.testing.assert_array_equal(x[24:,F_FEATURES.index('novelty')],np.full(24,.5))
    assert audit['novelty']['total_users']==3
    np.testing.assert_array_equal(x[:,F_FEATURES.index('old_U')],np.zeros(48))
    np.testing.assert_array_equal(x[:,F_FEATURES.index('warm_vulnerability')],np.full(48,.5))
    np.testing.assert_allclose(x[:12,F_FEATURES.index('B0')],17.5/23)
    np.testing.assert_allclose(x[12:24,F_FEATURES.index('B0')],5.5/23)


def _evaluation_data():
    users=np.array(['u0','u1']);warm=np.array([[f'a{i}' for i in range(12)],[f'b{i}' for i in range(12)]])
    return dict(cutoff='2020-01-22',users=users,warm_lists=warm,
        cold=pd.DataFrame(dict(user_index=[0,1],customer_id=users,article_id=['c0','c1'],b0_rank=[1,1])),
        truth=pd.DataFrame(dict(customer_id=users,article_id=['c0','b0'],interaction_count_before_cutoff=[0,21])),
        truthsets={'u0':{'c0'},'u1':{'b0'}})


def _config():
    return dict(id='F-synthetic',arm='F',trainable=False,weights={k:.1 for k in F_FEATURES},
        candidate_topK=50,replaceable_slot_floor=1,max_admissions=1,score_percentile_gate=.90)


def test_f_selects_all_matching_before_truth_and_recomputes_exact_ap():
    data=_evaluation_data();features=np.zeros((24,10),np.float32);features[11]=1
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)
        original=special.contexts
        def guarded_contexts(current):
            assert (folder/'SELECTION_COMPLETED_BEFORE_TRUTH.json').exists()
            assert read_json(folder/'SELECTION_PROGRESS.json')['next_user']==2
            return original(current)
        with patch.object(special,'contexts',side_effect=guarded_contexts):
            rows=special.evaluate_f(data,features,_config(),folder)
        row=rows[0]
        np.testing.assert_allclose(row['delta_map'],1/24,rtol=0,atol=1e-15)
        assert row['inserted_positives']==1 and row['removed_positives']==0
        assert row['replacements']==1 and row['admitted_users']==1
        assert row['all_matches_fixed_before_truth_access']
        saved=np.load(row['saved_decisions'])
        np.testing.assert_array_equal(saved[0,0],[0,11])
        assert (saved[1]==-1).all()
        assert special.evaluate_f(data,features,_config(),folder)==rows


def test_constant_zero_blend_passes_no_percentile_gate_and_does_not_force_admission():
    with TemporaryDirectory() as tmp:
        row=special.evaluate_f(_evaluation_data(),np.zeros((24,10),np.float32),_config(),tmp)[0]
        assert row['delta_map']==0 and row['replacements']==0
        assert row['global_score_threshold'] is None


def test_actual_frozen_f_config_fraction_is_converted_only_at_helper_boundary():
    config=f_configs()[0]
    assert .90<=config['score_percentile_gate']<=.999
    features=np.zeros((24,10),np.float32);features[11]=1
    with TemporaryDirectory() as tmp:
        with patch.object(special,'global_midrank_thresholds',wraps=special.global_midrank_thresholds) as helper:
            row=special.evaluate_f(_evaluation_data(),features,config,tmp)[0]
        assert helper.call_args.args[1]==[100.*config['score_percentile_gate']]
        assert row['policy']==config and row['policy_id']==config['id']
        assert row['replacements']==1 and row['inserted_positives']==1
        np.testing.assert_allclose(row['delta_map'],1/24,rtol=0,atol=1e-15)


def test_all_registered_f_configs_validate_and_wrong_percent_units_are_rejected():
    for config in f_configs():
        assert special._validate_f_config(config).shape==(10,)
    config=f_configs()[0];config['score_percentile_gate']=90
    with unittest.TestCase().assertRaises(ValueError):
        special._validate_f_config(config)


def test_f_rejects_final_week_before_writing_output():
    data=_evaluation_data();data['cutoff']='2020-09-16'
    with TemporaryDirectory() as tmp:
        folder=Path(tmp)/'never-created'
        with unittest.TestCase().assertRaises(ValueError):
            special.evaluate_f(data,np.zeros((24,10)),_config(),folder)
        assert not folder.exists()


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
        for name,function in sorted(globals().items()) if name.startswith('test_'))


if __name__=='__main__':
    unittest.main()
