"""Synthetic-only tests: no research artifacts, model fits or final-week reads."""
from datetime import date, timedelta
import time
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd

from hm_recsys.p42f_contract import GLOBAL
from hm_recsys.p43a_data import (source_for, recover_edge_indices, _group_arrays,
    _guard, edge_percentiles, hard_conditions, neutral_hash_keep, _e_view, META_COLUMNS)
from hm_recsys.p42f_core import neutral_keep
from hm_recsys.p41a_contract import write_json,read_json
from hm_recsys import p43a_data


def test_strict_source_matrix():
    expected = {
        '2019-12-25':(None,None),
        '2020-01-22':('winter_20200122','2020-01-01'),
        '2020-02-19':('winter_20200122','2020-01-01'),
        '2020-03-18':('spring_20200318','2020-02-26'),
        '2020-04-29':('spring_20200318','2020-02-26'),
        '2020-05-27':('spring_20200318','2020-02-26'),
        '2020-06-24':('early_summer_20200624','2020-06-03'),
        '2020-07-22':('early_summer_20200624','2020-06-03'),
        '2020-08-19':('late_summer_20200819','2020-07-29'),
    }
    for t,(window,end) in expected.items():
        source = source_for(t)
        assert (source['window'],source['training_label_end']) == (window,end)
        assert source['availability'] == int(window is not None)
        if window:
            assert end < t
            assert all((date.fromisoformat(d)+timedelta(days=7)).isoformat() < t
                       for d in source['training_cutoffs'])


def test_final_week_rejected_before_io():
    with unittest.TestCase().assertRaises(ValueError):
        source_for('2020-09-16')


def test_deadline_guard():
    with unittest.TestCase().assertRaises(TimeoutError):
        _guard(time.time()-1)


def _example():
    data = dict(users=np.array(['u0','u1','u2']),
        cold=pd.DataFrame(dict(user_index=[0,0,2],b0_rank=[2,4,50])))
    ui = np.array([0,0,0,2],np.int32)
    x = np.zeros((4,len(GLOBAL)),np.float32)
    x[:,GLOBAL.index('b0_rank')] = [2,2,4,50]
    x[:,GLOBAL.index('cold_rank_times_warm_slot')] = [2,24,4,600]
    return data,x,ui


def test_exact_edge_recovery_keeps_original_full50_ranks_after_overlap():
    data,x,ui = _example()
    np.testing.assert_array_equal(recover_edge_indices(data,x,ui),[0,11,12,35])


def test_recovery_rejects_corrupt_noninteger_slot():
    data,x,ui = _example()
    x[0,GLOBAL.index('cold_rank_times_warm_slot')] = 3
    with unittest.TestCase().assertRaises(ValueError):
        recover_edge_indices(data,x,ui)


def test_recovery_rejects_unknown_original_rank():
    data,x,ui = _example()
    x[0,GLOBAL.index('b0_rank')] = 1
    with unittest.TestCase().assertRaises(ValueError):
        recover_edge_indices(data,x,ui)


def test_groups_exclude_zero_row_users_but_preserve_user_identity():
    group, users = _group_arrays([0,0,0,2,2])
    np.testing.assert_array_equal(group,[3,2])
    np.testing.assert_array_equal(users,[0,2])
    empty, empty_users = _group_arrays([])
    assert len(empty) == len(empty_users) == 0
    with unittest.TestCase().assertRaises(ValueError):
        _group_arrays([0,2,0])


def test_neutral_hash_threshold_is_original_F_rule_and_point_five_subset():
    for i in range(500):
        args=('2020-02-19','u'+str(i%19),'000'+str(i),i%12+1)
        assert neutral_hash_keep(*args,threshold=200) == neutral_keep(*args)
        if neutral_hash_keep(*args):
            assert neutral_hash_keep(*args,threshold=200)


def test_group_percentile_is_after_overlap_and_preserves_average_ties():
    # Two Cold items for user0, one for user2; all 12 slots participate.
    values=np.r_[np.ones((1,12)),np.zeros((1,12)),np.arange(12).reshape(1,12)]
    out=edge_percentiles(values,[0,0,2])
    np.testing.assert_allclose(out[0],17.5/23)
    np.testing.assert_allclose(out[1],5.5/23)
    np.testing.assert_allclose(out[2],np.arange(12)/11,rtol=1e-7)
    np.testing.assert_array_equal(edge_percentiles(np.zeros((1,12)),[0]),np.full((1,12),.5))


def _hard_example():
    cold=pd.DataFrame(dict(user_index=np.arange(5),b0_rank=[5,20,20,20,20],
        qC_relative_available=[1]*5,qC_within_user_percentile=[.1,.95,.1,.1,.1]))
    qw=np.full((5,12),.9);qw[3,0]=.1
    warm=pd.DataFrame(dict(qW_within_user_percentile=qw.ravel(),qW_relative_available=np.ones(60)))
    old=np.zeros((5,12));old[4,0]=.95
    return dict(warm=warm),cold,old


def test_five_hard_conditions_use_correct_sides_and_or():
    data,cold,old=_hard_example()
    conditions=hard_conditions(data,cold,old,1)
    assert conditions['b0_top10'][0,0]
    assert conditions['qC_top_decile'][1,0]
    assert conditions['warm_slots_9_12'][:,8:].all()
    assert not conditions['warm_slots_9_12'][:,:8].any()
    assert conditions['qW_vulnerable_top_quintile'][3,0]
    assert not conditions['qW_vulnerable_top_quintile'][2,0]
    assert conditions['old_U_top_decile'][4,0]
    assert np.logical_or.reduce(list(conditions.values()))[:,0].tolist()==[True,True,False,True,True]


def test_missing_propensity_meta_does_not_create_hard_negatives():
    data,cold,old=_hard_example()
    cold['qC_relative_available']=0
    data['warm']['qW_relative_available']=0
    conditions=hard_conditions(data,cold,old,0)
    assert not conditions['qC_top_decile'].any()
    assert not conditions['qW_vulnerable_top_quintile'].any()
    assert not conditions['old_U_top_decile'].any()


def test_e_shared_view_removes_only_extra_easy_neutrals():
    audit=dict(rows=10,counts=dict(B=2,N=6,H=2),E_rows=7,
               E_counts=dict(B=2,N=3,H=2),paths=dict(X='same.npy',hard_neutral='hard.npy'))
    e=_e_view(audit)
    assert e['storage']=='A2' and e['rows']==7 and e['counts']['N']==3
    assert e['paths']['X']==audit['paths']['X']
    assert audit['rows']==10
    assert len(META_COLUMNS)==14 and len(set(META_COLUMNS))==14


def test_a1_end_to_end_exact_f_copy_group_identity_and_weight_change():
    data,x,ui=_example()
    x[0,GLOBAL.index('qC_within_user_percentile')]=np.nan
    y=np.array([0.,.1,-.2,0.],np.float64)
    with TemporaryDirectory() as tmp:
        repo=Path(tmp);root=repo/'new-artifacts';t='2019-12-25'
        source=repo/'artifacts/phase4'/p43a_data.F_RUN/'prepared'/t
        source.mkdir(parents=True)
        np.savez(source/'training.npz',X=x,y=y,user_index=ui,weight=np.where(y==0,50.,1.))
        reports=repo/'reports/phase4';reports.mkdir(parents=True)
        write_json(reports/'p4_2g_data_parity.json',dict(cutoffs={t:dict(retained_rows=4,B=1,N=34,H=1,edges=36)}))
        with patch.object(p43a_data,'load_f_data',return_value=(data,dict(path='synthetic-data'))), \
             patch.object(p43a_data,'trusted_source',return_value=dict(path='synthetic-F-npz')):
            audit=p43a_data.prepare_date(repo,root,t,sampling='A1')
            np.testing.assert_array_equal(np.load(audit['paths']['X']),x)
            np.testing.assert_array_equal(np.load(audit['paths']['y']),y)
            np.testing.assert_array_equal(np.load(audit['paths']['group']),[3,1])
            np.testing.assert_array_equal(np.load(audit['paths']['group_user_index']),[0,2])
            np.testing.assert_array_equal(np.load(audit['paths']['edge_index']),[0,11,12,35])
            assert audit['training_weight']==1 and audit['counts']==dict(B=1,N=2,H=1)
            assert audit['original_IPW_removed'] and audit['source_row_parity']
            second=p43a_data.prepare_date(repo,root,t,sampling='D')
            assert second['paths']==audit['paths']
            assert read_json(root/'prepared'/t/'AUDIT.json')['variants']['A1']['rows']==4


def test_prepare_final_week_or_expired_deadline_creates_no_output():
    with TemporaryDirectory() as tmp:
        root=Path(tmp)/'new-artifacts'
        with unittest.TestCase().assertRaises(ValueError):
            p43a_data.prepare_date(tmp,root,'2020-09-16')
        with unittest.TestCase().assertRaises(TimeoutError):
            p43a_data.prepare_date(tmp,root,'2019-12-25',deadline_epoch=time.time()-1)
        assert not root.exists()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
        for name,function in sorted(globals().items()) if name.startswith('test_'))


if __name__ == '__main__':
    unittest.main()
