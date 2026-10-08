"""Fixed INNER-only rank redundancy and AP-weighted raw-score complementarity."""
from __future__ import annotations
import time
import numpy as np
from . import warm_v3_common as common
from .warm_v2_contract import read,write,now
from .warm_v2_engine import literal
from .warm_v3_pool import connection,INNER_CUTOFFS
from .warm_v3_pool_failure import contributions,decompose

TRIALS={'sequence':'WV3-301','graph':'WV3-201','userknn':'WV3-221'}
COLUMNS={'bpr':'wv2_bpr_user_item_score','sequence':'wv3_sequence_score',
    'graph':'wv3_graph_user_item_score','userknn':'wv3_userknn_neighbor_score'}


def swap_gain(labels,positive,negative,truth,k=12):
    """Exact AP@k gain for moving a misordered positive above one negative."""
    y=np.asarray(labels,dtype=np.float64)
    i,j=int(positive),int(negative)
    assert i>j and y[i]==1 and y[j]==0 and truth>=y.sum()
    if j>=k:return 0.0
    hits=np.cumsum(y)
    gain=(hits[j]+1)/(j+1)
    if i<k:gain-=hits[i]/(i+1)
    stop=min(i,k)
    gain+=(y[j+1:stop]/np.arange(j+2,stop+1)).sum()
    return float(gain/min(truth,k))


def correction_audit(frame,n):
    total_pairs=0
    sensitive_pairs=0
    total_weight=0.0
    out={k:{'correct_weight':0.,'available_weight':0.,'tie_weight':0.,'bpr_missed_correct_weight':0.,
            'common_available_weight':0.,'bpr_missed_weight':0.,'bpr_correct_but_source_not_weight':0.} for k in COLUMNS}
    tri={'all_available_weight':0.,'bpr_and_lightgcn_both_wrong_weight':0.,
        'sequence_correct_when_bpr_and_lightgcn_wrong_weight':0.,
        'bpr_wrong_lightgcn_correct_weight':0.,'sequence_agrees_lightgcn_weight':0.,
        'bpr_correct_lightgcn_wrong_weight':0.,'sequence_agrees_bpr_weight':0.}
    for _,g in frame.groupby('customer_id',sort=False):
        y=g.target.to_numpy(np.int8)
        truth=int(g.truth_count.iloc[0])
        scores={k:g[k].to_numpy(np.float64) for k in COLUMNS}
        for i in np.flatnonzero(y):
            for j in np.flatnonzero(y[:i]==0):
                total_pairs+=1
                weight=swap_gain(y,i,j,truth)
                if weight<=0:continue
                sensitive_pairs+=1
                total_weight+=weight
                b_available=np.isfinite(scores['bpr'][[i,j]]).all()
                b_correct=b_available and scores['bpr'][i]>scores['bpr'][j]
                for family,s in scores.items():
                    v=out[family]
                    available=np.isfinite(s[[i,j]]).all()
                    correct=available and s[i]>s[j]
                    if available:
                        v['available_weight']+=weight
                        if correct:v['correct_weight']+=weight
                        elif s[i]==s[j]:v['tie_weight']+=weight
                    if available and b_available:
                        v['common_available_weight']+=weight
                        if not b_correct:
                            v['bpr_missed_weight']+=weight
                            if correct:v['bpr_missed_correct_weight']+=weight
                        elif not correct:v['bpr_correct_but_source_not_weight']+=weight
                if {'bpr','lightgcn','sequence'} <= scores.keys():
                    b,g,s=(scores[name] for name in ('bpr','lightgcn','sequence'))
                    if all(np.isfinite(x[[i,j]]).all() for x in (b,g,s)):
                        bc,gc,sc=b[i]>b[j],g[i]>g[j],s[i]>s[j]
                        tri['all_available_weight']+=weight
                        if not bc and not gc:
                            tri['bpr_and_lightgcn_both_wrong_weight']+=weight
                            if sc:tri['sequence_correct_when_bpr_and_lightgcn_wrong_weight']+=weight
                        elif not bc and gc:
                            tri['bpr_wrong_lightgcn_correct_weight']+=weight
                            if sc:tri['sequence_agrees_lightgcn_weight']+=weight
                        elif bc and not gc:
                            tri['bpr_correct_lightgcn_wrong_weight']+=weight
                            if sc:tri['sequence_agrees_bpr_weight']+=weight
    for v in out.values():
        v['correct_fraction_all_AP_weight']=v['correct_weight']/total_weight
        v['available_fraction_all_AP_weight']=v['available_weight']/total_weight
        v['correct_fraction_available_AP_weight']=v['correct_weight']/v['available_weight'] if v['available_weight'] else 0
        v['correct_fraction_BPR_missed_common_available_AP_weight']=v['bpr_missed_correct_weight']/v['bpr_missed_weight'] if v['bpr_missed_weight'] else 0
        v['additional_correct_vs_BPR_fraction_all_AP_weight']=v['bpr_missed_correct_weight']/total_weight
        v['lost_correct_vs_BPR_fraction_all_AP_weight']=v['bpr_correct_but_source_not_weight']/total_weight
    if tri['all_available_weight']:
        tri['sequence_rescue_fraction_when_bpr_and_lightgcn_both_wrong']=(
            tri['sequence_correct_when_bpr_and_lightgcn_wrong_weight']/tri['bpr_and_lightgcn_both_wrong_weight']
            if tri['bpr_and_lightgcn_both_wrong_weight'] else 0.)
        tri['sequence_agreement_when_only_lightgcn_correct']=(
            tri['sequence_agrees_lightgcn_weight']/tri['bpr_wrong_lightgcn_correct_weight']
            if tri['bpr_wrong_lightgcn_correct_weight'] else 0.)
        tri['sequence_agreement_when_only_bpr_correct']=(
            tri['sequence_agrees_bpr_weight']/tri['bpr_correct_lightgcn_wrong_weight']
            if tri['bpr_correct_lightgcn_wrong_weight'] else 0.)
        tri['both_wrong_fraction_all_available_AP_weight']=tri['bpr_and_lightgcn_both_wrong_weight']/tri['all_available_weight']
    return {'misordered_pairs_in_E1Top50':total_pairs,'AP_sensitive_misordered_pairs':sensitive_pairs,
        'sum_independent_swap_AP_weight':total_weight,'sum_weight_div_full_users_not_joint_MAP_gain':total_weight/n,
        'signals':out,'three_signal_conditions':tri}


def audit_window(w,cutoff,screens):
    assert cutoff in INNER_CUTOFFS
    baseline=read(common.ART/'gate_data'/f'2020_inner_{cutoff}'/'DATA.json')
    n=baseline['total_users']
    with connection() as con:
        con.execute(f'CREATE TEMP TABLE base AS SELECT * FROM read_parquet({literal(baseline["ranks_path"])})')
        ranks={}
        for family,screen in screens.items():
            rec=screen['windows'][w]
            assert rec['total_users']==n and rec['fit']['cutoff']==cutoff
            assert 'wv2_bpr_user_item_score' not in screen['producer_features']
            con.execute(f'CREATE TEMP TABLE expert AS SELECT * FROM read_parquet({literal(rec["ranks_path"])})')
            assert con.execute('''SELECT count(*) FROM expert e FULL JOIN base b USING(customer_id,article_id)
                WHERE e.customer_id IS NULL OR b.customer_id IS NULL OR e.target<>b.target OR e.r0<>b.r0 OR e.r1<>b.r1 OR e.rf<>b.rf''').fetchone()[0]==0
            stats=con.execute('''WITH g AS (SELECT customer_id,
                avg(abs(r2-r0))*1.0/(count(*)-1) normalized_distance_E0,
                avg(abs(r2-r1))*1.0/(count(*)-1) normalized_distance_E1,
                count(*) FILTER(WHERE r2<=12 AND r0<=12)/12.0 overlap_E0,
                count(*) FILTER(WHERE r2<=12 AND r1<=12)/12.0 overlap_E1
                FROM expert GROUP BY customer_id)
                SELECT avg(normalized_distance_E0),avg(normalized_distance_E1),avg(overlap_E0),avg(overlap_E1) FROM g''').fetchone()
            con.execute('CREATE TEMP TABLE A AS SELECT customer_id,article_id,target,truth_count,ap_rf rf FROM expert')
            con.execute('CREATE TEMP TABLE Z AS SELECT customer_id,article_id,target,truth_count,ap_r3 rf FROM expert')
            contributions(con,'A','ha');contributions(con,'Z','hz')
            pieces=decompose(con,'ha','hz',n)
            ranks[family]={'normalized_rank_distance_E0':stats[0],'normalized_rank_distance_E1':stats[1],
                'Top12_overlap_E0':stats[2],'Top12_overlap_E1':stats[3],
                'threeway_minus_original601_MAP':sum(x['population_delta'] for x in pieces),
                'AP_truth_decomposition':pieces,'source':rec['ranks_path']}
            for table in ('expert','A','Z','ha','hz'):con.execute('DROP TABLE '+table)
        con.execute('''CREATE TEMP TABLE head AS SELECT customer_id,article_id,target,truth_count,r1
            FROM base WHERE r1<=50 AND user_history_events_12w>0''')
        paths={}
        bpr,_=common.bpr_path(cutoff);paths['bpr']=bpr
        for family in TRIALS:
            path=(common.ART/'lightgcn/formal1000'/cutoff/'features.parquet'
                  if family=='lightgcn' else common.ART/family/cutoff/'features.parquet')
            meta=read(path.with_name('features.json' if family=='graph' else 'FEATURES.json'))
            assert meta['cutoff']==cutoff
            assert path.is_file()
            paths[family]=path
        sql='SELECT h.*,'+','.join(f's_{f}.{COLUMNS[f]} "{f}"' for f in COLUMNS)+' FROM head h '
        for f,p in paths.items():
            assert con.execute(f'''SELECT count(*) FROM head h LEFT JOIN read_parquet({literal(p)}) s USING(customer_id,article_id)
                WHERE s.customer_id IS NULL''').fetchone()[0]==0,'Signal cache missing candidate identities'
            sql+=f'LEFT JOIN read_parquet({literal(p)}) s_{f} USING(customer_id,article_id) '
        frame=con.execute(sql+'ORDER BY customer_id,r1').fetchdf()
        assert len(frame)==con.execute('SELECT count(*) FROM head').fetchone()[0]
        assert not frame.duplicated(['customer_id','article_id']).any()
    return {'cutoff':cutoff,'full_user_denominator':n,'expert_distances_and_existing_fusion':ranks,
        'raw_signal_correction':correction_audit(frame,n),'signal_paths':{k:str(p) for k,p in paths.items()},
        'final_week':'not_run'}


def run(output_name='JOINT_REPRESENTATION_AUDIT.json'):
    common.setup();common.budget(5)
    start=time.perf_counter()
    screens={f:read(common.REPORT/(t+'_SCREEN.json')) for f,t in TRIALS.items()}
    protocol=common.contracts()['rolling_protocol']
    rows={w:audit_window(w,p['inner_validation'],screens) for w,p in protocol.items()}
    out={'created_at':now(),'role':'read_only_original_inner','windows':rows,'new_fits':0,'new_outer':0,
        'candidate_pool':'unchanged original100-300; raw-score pair audit restricted to originalE1Top50',
        'warning':'single-pair delta AP weights overlap and cannot be summed into attainable joint MAP; no raw-score scaling comparison',
        'runtime_seconds':time.perf_counter()-start,'final_week':'not_run','automatic_joint_model_authorization':False}
    write(common.REPORT/output_name,out)
    print({'joint_audit_seconds':out['runtime_seconds']},flush=True)
    return out


if __name__=='__main__':run()
