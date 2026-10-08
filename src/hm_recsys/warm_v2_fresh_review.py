"""Independent fresh ranking audit and fixed-gate decision; no fitting or tuning."""
from __future__ import annotations
from pathlib import Path
import argparse
import time
import numpy as np
import pandas as pd
import duckdb
from .metrics import apk
from .warm_v2_contract import read,write,now,evidence_id
from .warm_v2_engine import literal,save_parquet
from .warm_v2_fresh import ART,CONTRACT,setup

REPORT=Path('reports/warm_v2')
TRIALS=['FRESH-000','FRESH-501','FRESH-601']


def verdict(deltas):
    a=np.asarray(deltas,float)
    assert len(a)==4 and np.isfinite(a).all()
    robust=a.mean()>=.0005 and (a>=0).sum()>=3 and a.min()>=-.0003
    return 'strong_pass' if robust and (a>0).all() else 'robust_pass' if robust else 'weak_generalization' if a.mean()>0 else 'fresh_robustness_failed'


def window_review(w,c):
    root=ART/'review'/w;root.mkdir(parents=True,exist_ok=True)
    if (root/'REVIEW.json').exists():return read(root/'REVIEW.json')
    cutoff=c['rolling_protocol'][w]['outer_validation']
    assert (ART/'chain_status'/w/'COMPLETE.json').exists()
    source='./data/interim/audit/transactions.parquet'
    threshold=int(c['candidate_source_config']['sample_rate']*1_000_000)
    with duckdb.connect() as con:
        con.execute("SET threads=8; SET memory_limit='3GB'")
        con.execute(f'SET temp_directory={literal(ART/"spill") }')
        for alias,trial in zip(('b','n','f'),TRIALS):
            con.execute(f'ATTACH {literal(ART/trial/w/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
        # Independent population from raw truth hash, NOT restricted to predicted users.
        truth=con.execute(f"""SELECT DISTINCT customer_id,article_id FROM read_parquet('{source}')
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
            AND hash(customer_id)%1000000 < {threshold} ORDER BY customer_id,article_id""").fetchdf()
        truthsets={u:g.article_id.tolist() for u,g in truth.groupby('customer_id',sort=False)};users=sorted(truthsets)
        con.register('t',truth)
        checks={}
        for alias in ('b','n','f'):
            rows,duplicates=con.execute(f'SELECT count(*),count(*)-count(DISTINCT (customer_id,article_id)) FROM {alias}.predictions').fetchone()
            mismatch=con.execute(f'''SELECT count(*) FROM b.predictions b FULL JOIN {alias}.predictions x USING(customer_id,article_id)
                WHERE b.customer_id IS NULL OR x.customer_id IS NULL OR b.target<>x.target OR b.candidate_rank<>x.candidate_rank OR b.user_history_events_12w<>x.user_history_events_12w''').fetchone()[0]
            label_errors=con.execute(f'''SELECT count(*) FROM {alias}.predictions x LEFT JOIN t USING(customer_id,article_id)
                WHERE x.target<>CASE WHEN t.article_id IS NOT NULL THEN 1 ELSE 0 END''').fetchone()[0]
            assert duplicates==mismatch==label_errors==0
            sizes=con.execute(f'SELECT count(*) n FROM {alias}.predictions GROUP BY customer_id').fetchnumpy()['n']
            assert sizes.min()>=100 and sizes.max()<=300
            checks[alias]={'candidate_rows':rows,'duplicate_pairs':duplicates,'pair_label_rank_history_mismatch':mismatch,'raw_label_errors':label_errors}
        ranked=con.execute('''SELECT b.customer_id,b.article_id,b.candidate_rank,b.user_history_events_12w,b.score score_b,n.score score_n,f.score score_f
            FROM b.predictions b JOIN n.predictions n USING(customer_id,article_id) JOIN f.predictions f USING(customer_id,article_id)
            ORDER BY b.customer_id,b.candidate_rank,b.article_id''').fetchdf()
        oldrank=ranked.groupby('customer_id',sort=False).score_b.rank(ascending=False,method='first')
        newrank=ranked.groupby('customer_id',sort=False).score_n.rank(ascending=False,method='first')
        expected=1/(60+oldrank.to_numpy())+1/(60+newrank.to_numpy())
        diff=float(np.max(np.abs(expected-ranked.score_f.to_numpy())));assert diff<1e-15
        activities=ranked.groupby('customer_id',sort=False).user_history_events_12w.first().to_dict()
        preds={};aps={};tops={}
        for alias,trial in zip(('b','n','f'),TRIALS):
            top=con.execute(f'SELECT customer_id,article_id,final_rank FROM {alias}.top12 ORDER BY customer_id,final_rank').fetchdf()
            scores=ranked['score_'+alias].to_numpy()
            order=ranked[['customer_id','article_id','candidate_rank']].copy()
            order['s']=np.where(ranked.user_history_events_12w.to_numpy()==0,-ranked.candidate_rank.to_numpy(),scores)
            order['final_rank']=order.groupby('customer_id',sort=False).s.rank(ascending=False,method='first').astype(np.int32)
            expected_top=order[order.final_rank<=12][['customer_id','article_id','final_rank']].sort_values(['customer_id','final_rank']).reset_index(drop=True)
            pd.testing.assert_frame_equal(top,expected_top,check_dtype=False)
            p={u:g.article_id.tolist() for u,g in top.groupby('customer_id',sort=False)}
            assert set(p)==set(truthsets) and all(len(v)==len(set(v))==12 for v in p.values())
            preds[trial]=p;tops[trial]=set(zip(top.customer_id,top.article_id))
            aps[trial]=np.array([apk(truthsets[u],p[u]) for u in users])
        hits=dict(con.execute('SELECT customer_id,sum(target) FROM b.predictions GROUP BY customer_id').fetchall())
        recalls=[hits[u]/len(truthsets[u]) for u in users]
        oracle=[min(hits[u],12)/min(len(truthsets[u]),12) for u in users]
        shared={'users':len(users),'truth_pairs':len(truth),'candidate_rows':len(ranked),'positive_candidate_pairs':sum(hits.values()),
            'candidate_recall':float(np.mean(recalls)),'candidate_oracle_map@12':float(np.mean(oracle)),
            'candidate_hit_rate':float(np.mean([hits[u]>0 for u in users]))}
    frame=pd.DataFrame({'customer_id':users,'history_events_12w':[activities[u] for u in users],
        'truth_count':[len(truthsets[u]) for u in users],'candidate_hits':[hits[u] for u in users]})
    frame['activity']=pd.cut(frame.history_events_12w,[-1,0,5,20,np.inf],labels=['inactive_0','low_1_5','medium_6_20','high_21_plus']).astype(str)
    results={}
    for trial in TRIALS:
        ap=aps[trial];delta=ap-aps['FRESH-000'];frame['ap_'+trial]=ap
        inactive=[u for u in users if activities[u]==0]
        parity=sum(preds[trial][u]!=preds['FRESH-000'][u] for u in inactive);assert parity==0
        if trial!='FRESH-601':
            trained=read(ART/trial/w/'outer_metrics.json')
            assert abs(ap.mean()-trained['evaluation']['map@12'])<1e-12
            assert all(shared[k]==trained['evaluation'][k] for k in ('users','truth_pairs','candidate_rows','candidate_recall','candidate_oracle_map@12'))
            params='lightgbm_outer_params';assert trained['parameters']==c[params]
            assert trained['features']==c['frozen_feature_columns']+([] if trial=='FRESH-000' else c['extra_features'])
            inner=read(ART/trial/w/'inner_metrics.json');assert inner['parameters']==c['lightgbm_inner_params']
            assert inner['train_cutoffs']==c['rolling_protocol'][w]['inner_train']
            assert trained['train_cutoffs']==c['rolling_protocol'][w]['outer_train']
            model_info={'inner_rounds':inner['best_iteration'],'train_sampling':trained['sampling'],
                'feature_usage':[v for v in trained['feature_importance'] if v['feature'].startswith('wv2_')]}
        else:model_info={'fits':0,'parents':TRIALS[:2]}
        results[trial]={**shared,**model_info,'map@12':float(ap.mean()),'delta_vs_FRESH-000':float(delta.mean()),
            'users_improved':int((delta>1e-15).sum()),'users_harmed':int((delta< -1e-15).sum()),'users_unchanged':int((np.abs(delta)<=1e-15).sum()),
            'gross_positive_map_contribution':float(delta[delta>1e-15].sum()/len(users)),
            'gross_negative_map_contribution':float(delta[delta< -1e-15].sum()/len(users)),
            'top12_new_pairs_vs_base':len(tops[trial]-tops['FRESH-000']),
            'inactive_users':len(inactive),'inactive_ordered_top12_mismatches':parity}
    segments={}
    for group,g in frame.groupby('activity',sort=True):
        segments[group]={'users':len(g),'systems':{t:{'map@12':float(g['ap_'+t].mean()),
            'delta_vs_base':float((g['ap_'+t]-g['ap_FRESH-000']).mean()),
            'contribution_to_all_users_delta':float((g['ap_'+t]-g['ap_FRESH-000']).sum()/len(frame))} for t in TRIALS}}
    fp=root/'user_ap.parquet';save_parquet(frame,fp)
    out={'window':w,'cutoff':cutoff,'systems':results,'activity_segments':segments,
        'checks':checks,'all_candidate_rrf_max_difference':diff,'independent_full_top12_replay':True,
        'full_raw_hash_sample_population':True,'user_ap':evidence_id(fp,reason='explicit_registry_evidence'),'final_week':'not_run'}
    write(root/'REVIEW.json',out);return out


def main():
    c=setup();results={w:window_review(w,c) for w in c['rolling_protocol']}
    complementarity={w:read(ART/'diagnostics'/w/'inner_ranks.json') for w in results}
    historical=read(REPORT/'WV2_600_COMPLEMENTARITY_AUDIT.json')
    write(REPORT/'WARM_V2_FRESH_COMPLEMENTARITY.json',{'windows':complementarity,
        'historical_development_context':historical,'diagnostic_only':True,
        'comparison_caution':'different dates and sample sizes; exclusive positive pairs are model-Top12 overlap, not full-truth density or causal effects',
        'final_week':'not_run'})
    d={t:[r['systems'][t]['delta_vs_FRESH-000'] for r in results.values()] for t in TRIALS}
    output={'created_at':now(),'status':verdict(d['FRESH-601']),'windows':results,
        'summary':{t:{'mean_map':float(np.mean([r['systems'][t]['map@12'] for r in results.values()])),
            'mean_delta':float(np.mean(v)),'worst_delta':min(v),'nondegrade_windows':sum(x>=0 for x in v),
            'positive_windows':sum(x>0 for x in v),'sign_pattern':['+' if x>0 else '-' if x<0 else '0' for x in v]} for t,v in d.items()},
        'development_evidence':read(REPORT/'WARM_V2_RESEARCH_MILESTONE.json'),
        'fresh_architecture_selected_on_these_windows':False,'no_adaptive_rescue':True,'final_week':'not_run'}
    write(REPORT/'WARM_V2_FRESH_ROBUSTNESS.json',output)
    write(REPORT/'WARM_V2_FRESH_REVIEW.json',{'created_at':now(),'status':'passed','windows':{w:{k:r[k] for k in ('checks','all_candidate_rrf_max_difference','independent_full_top12_replay','full_raw_hash_sample_population')} for w,r in results.items()},
        'scope':'independent raw population, raw labels, all candidate identity, all score and orderedTop12 replay, complete truth AP; verification pass is not promotion gate',
        'final_week':'not_run'})
    print({'status':output['status'],'summary':output['summary']},flush=True)


if __name__=='__main__':main()
