"""Pure report arithmetic and completeness gates; no artifact rendering/fits."""
import unittest

from hm_recsys.p43a_report import (
    COLUMNS, INDEX, SEGMENTS, WINDOW_KEYS, compact_row, ranked_completed,
    oracle_summary, search_plan, selection, simple_policy, sampling_comparison, canonical_e_row, one_knob_comparison,
)


def completed(arm, model, mean, delta=.0002, nondegrade=3, worst=0., warm=0.):
    row = [None]*len(COLUMNS)
    for key, value in dict(arm=arm, model_config_id=model, policy_id='p',
        status='completed', mean_map=mean, mean_delta=delta, nondegrade_windows=nondegrade,
        worst_delta=worst, warm_mean_delta=warm).items():
        row[INDEX[key]] = value
    return row, (1,5,-12,1,-.99)


def observed(delta):
    return dict(status='completed', overall_map=.03+delta, delta_map=delta,
        segments={s:dict(delta_vs_w0=delta) for s in SEGMENTS},
        inserted_positives=1, removed_positives=2, replacements=3,
        coverage=.1, seconds=.5)


class P43AReportTests(unittest.TestCase):
    def test_one_knob_keeps_other_knobs_and_model_fixed(self):
        rows=[completed('A','a',.04),completed('A','a',.03),completed('A','a',.02),completed('A','b',.01)]
        for i,(row,_) in enumerate(rows):row[INDEX['policy_id']]=str(i)
        lookup={('a','0'):dict(id='0',top_edge=1,slot_floor=1),
                ('a','1'):dict(id='1',top_edge=2,slot_floor=1),
                ('a','2'):dict(id='2',top_edge=2,slot_floor=12),
                ('b','3'):dict(id='3',top_edge=1,slot_floor=1)}
        result=one_knob_comparison(rows,lookup)
        self.assertEqual({r['policy_id'] for r in result['rows']},{'0','1'})
        self.assertEqual(result['new_policy_evaluations'],0)

    def test_e_alias_is_parameter_checked_and_does_not_mutate_receipt(self):
        ids = {f'E-K{k}-S{s}' for k in (5,10,20,50) for s in (12,10,7,1)}
        for k in (5,10,20,50):
            for s in (12,10,7,1):
                alias = f'p43a-E-k{k}-floor{s}'
                row = dict(policy_id=alias, policy=dict(id=alias,candidate_topk=k,slot_floor=s),
                           delta_map=.000123, decision_policy_index=7)
                mapped = canonical_e_row(row,ids)
                self.assertEqual(mapped['policy_id'],f'E-K{k}-S{s}')
                self.assertEqual(mapped['execution_policy_id'],alias)
                self.assertEqual(row['policy_id'],alias)
                self.assertEqual(mapped['delta_map'],row['delta_map'])
                self.assertEqual(mapped['decision_policy_index'],7)
        bad = dict(row,policy=dict(row['policy'],slot_floor=12))
        with self.assertRaises(ValueError):
            canonical_e_row(bad,ids)
        with self.assertRaises(ValueError):
            canonical_e_row(dict(row,policy=dict(row['policy'],extra_threshold=.5)),ids)

    def test_sampling_comparison_requires_same_parameters_and_complete_policy(self):
        a = completed('A', 'A1-L15-D4-M200', .0301, delta=.0001)
        b = completed('A', 'A2-L15-D4-M200', .0300, delta=0.)
        unrelated = completed('A', 'A2-L31-D4-M200', .04, delta=.01)
        partial = completed('A', 'A2-L15-D4-M50', .04, delta=.01)
        partial[0][INDEX['status']] = 'partial'
        result = sampling_comparison([a,b,unrelated,partial])
        self.assertEqual(len(result['rows']),1)
        row = result['rows'][0]
        self.assertEqual(row['paired_complete_policies'],1)
        self.assertEqual(row['hard_lower'],1)
        self.assertEqual(row['hard_higher'],0)
        self.assertEqual(row['reference_same_policy_mean_delta'],.0001)
        self.assertEqual(result['new_policy_evaluations'],0)

    def test_sampling_comparison_absent_pair_is_not_failure(self):
        result = sampling_comparison([completed('A','A1-L15-D4-M200',.03)])
        self.assertEqual(result['rows'],[])
        self.assertEqual(result['status'],'not_run_no_complete_pairs')

    def test_ledger_all107690_and_C_not_tripled(self):
        models = [dict(arm=a,id=f'{a}-{i}') for a,n in [('A',16),('B',32),('D',8),('E',1)] for i in range(n)]
        models += [dict(arm='C',id='C-'+role) for role in ('count','cold','warm')]
        contract = dict(model_configs=models,policy_grid=[dict(id=str(i)) for i in range(1920)],
            E=dict(policies=[dict(id=str(i)) for i in range(16)]),
            C=dict(blends=[dict(id=str(i)) for i in range(26)]),
            F=dict(configs=[dict(id='F-'+str(i)) for i in range(128)]))
        plan = search_plan(contract)
        self.assertEqual(sum(len(p) for _,_,p in plan),107690)
        self.assertEqual(sum(len(p) for a,_,p in plan if a=='C'),26)

    def test_unrun_and_partial_are_null_not_zero(self):
        evidence = dict(attempted=False,fit_seconds=0.)
        row = compact_row('A','a','p',{}, {}, evidence)
        self.assertEqual(row[INDEX['status']],'not_run')
        self.assertIsNone(row[INDEX['mean_map']])
        self.assertIsNone(row[INDEX['inserted_positives']])
        self.assertIsNone(row[INDEX['observed_inserted_positives']])
        windows = {WINDOW_KEYS[0]:dict(p=observed(.0002))}
        row = compact_row('A','a','p',windows, {}, evidence)
        self.assertEqual(row[INDEX['status']],'partial')
        self.assertIsNone(row[INDEX['mean_map']])
        self.assertEqual(row[INDEX['observed_inserted_positives']],1)

    def test_four_window_means_and_denominators(self):
        ds = [.001,0.,-.001,.002]
        windows = {w:dict(p=observed(d)) for w,d in zip(WINDOW_KEYS,ds)}
        row = compact_row('A','a','p',windows,{},dict(attempted=True,fit_seconds=5.))
        self.assertEqual(row[INDEX['status']],'completed')
        self.assertAlmostEqual(row[INDEX['mean_delta']],.0005)
        self.assertEqual(row[INDEX['nondegrade_windows']],3)
        self.assertEqual(row[INDEX['worst_delta']],-.001)
        self.assertEqual(row[INDEX['inserted_positives']],4)
        self.assertEqual(row[INDEX['replacements']],12)
        self.assertEqual(row[INDEX['policy_seconds']],2)

    def test_pruned_rows_do_not_get_fake_four_window_mean(self):
        windows = {w:dict(p=observed(-.002)) for w in WINDOW_KEYS[:2]}
        record = dict(policies=dict(p=dict(status='pruned_bad_config')))
        row = compact_row('B','b','p',windows,record,dict(attempted=True,fit_seconds=2.))
        self.assertEqual(row[INDEX['status']],'pruned_bad_config')
        self.assertIsNone(row[INDEX['mean_delta']])
        windows[WINDOW_KEYS[1]]['p']['delta_map']=.01
        with self.assertRaises(ValueError):
            compact_row('B','b','p',windows,record,dict(attempted=True,fit_seconds=2.))

    def test_ranking_recomputes_current_maximum_tolerance_band(self):
        a=completed('A','a',.5,nondegrade=4,worst=.002)
        b=completed('B','b',.5-.9e-8,nondegrade=2,worst=0.)
        c=completed('C','c',.5-1.5e-8,nondegrade=3,worst=.001)
        ranked=ranked_completed([c,b,a])
        self.assertEqual([row[1] for row,_ in ranked],['a','c','b'])

    def test_incomplete_tournament_never_authorizes_scaleup(self):
        ranking=ranked_completed([completed('A','a',.05),completed('B','b',.04)])
        pick=selection(ranking,False)
        self.assertFalse(pick['full_scale_allowed'])
        self.assertFalse(pick['top2_full_scale_allowed'])
        self.assertEqual(len(pick['provisional_top2']),2)

    def test_F_keeps_rank_but_skips_trainable_slots(self):
        ranking=ranked_completed([completed('F','f',.06),completed('A','a',.05),
                                  completed('A','a2',.04),completed('B','b',.03)])
        pick=selection(ranking,True)
        self.assertTrue(pick['top2_full_scale_allowed'])
        self.assertEqual([r['arm'] for r in pick['provisional_top2']],['A','B'])
        self.assertEqual(ranking[0][0][0],'F')

    def test_pilot_gate_strict_mean_and_inclusive_worst(self):
        ranking=ranked_completed([completed('A','a',.05,delta=.0001),
                                  completed('B','b',.04,delta=.00010001,worst=-.001,nondegrade=2)])
        pick=selection(ranking,True)
        self.assertTrue(pick['top1_full_scale_allowed'])
        self.assertEqual(pick['provisional_top2'][0]['arm'],'B')

    def test_policy_simplicity_handles_fraction_and_percentage_units(self):
        a=dict(max_admissions=1,candidate_topk=5,slot_floor=12,top_edge=1,global_percentile=99)
        b=dict(max_admissions=1,candidate_topK=5,replaceable_slot_floor=12,top_edge=1,score_percentile_gate=.99)
        self.assertEqual(simple_policy(a),simple_policy(b))

    def test_incomplete_oracle_does_not_report_mean_zero(self):
        value=oracle_summary(dict(status='partial',dates={}))
        self.assertFalse(value['complete'])
        self.assertIsNone(value['mean_oracle_delta'])
        self.assertTrue(all(row['oracle_map'] is None for row in value['rows']))


if __name__=='__main__':
    unittest.main()
