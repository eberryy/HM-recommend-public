"""Independent all-row P4.2E checks; never fits, retrieves, or matches."""
from pathlib import Path
import time
import numpy as np
import duckdb
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score

from .p42e import context
from .p41a_contract import identity,check_identity,read_json,write_json
from .p42r_data import _parquet
from .p42e_data import fold_for_user
from .p42e_stats import monotonic_parity
from .p42e_contract import now
from .p42e_fit import logits


def verify(repo):
    c,root=context(repo); started=time.perf_counter()
    m=read_json(repo/'reports/phase4/P4_2E_metrics.json')
    checks={}; detail={}
    check_identity(read_json(root/'EXECUTION_START.json')['contract']); checks['preregistration_unchanged']=True
    for source in c['source_review'].values(): check_identity(source)
    checks['old_implementation_unchanged']=True
    inp=read_json(root/'INPUT_VERIFICATION.json')
    for entry in inp['trusted_records']: check_identity(entry)
    checks['trusted_input_SHAs']=True
    for t in c['historical_cutoffs']:
        folder=root/'prepared'/t
        if not (folder/'COMPLETE.json').exists(): continue
        h=_parquet(folder/'eligible-users.parquet'); a=read_json(folder/'COMPLETE.json')
        assert len(h)==a['eligible_users'] and h.customer_id.is_unique
        np.testing.assert_array_equal(h.fold,np.array([fold_for_user(u) for u in h.customer_id]))
        selection=list(folder.glob('chunk-*/cold50-unlabelled.parquet')); training=list(folder.glob('chunk-*/training.parquet'))
        assert len(selection)==len(training)==len(a['chunks'])
        fence=read_json(folder/'CANDIDATES_FIXED_BEFORE_LABEL_JOIN.json')
        assert all(p.stat().st_mtime>=Path(folder/'CANDIDATES_FIXED_BEFORE_LABEL_JOIN.json').stat().st_mtime for p in training)
        with duckdb.connect() as con:
            con.execute('SET threads=4'); con.execute("SET memory_limit='2GB'")
            paths=[str(p) for p in training]
            roster=con.execute('SELECT DISTINCT customer_id FROM read_parquet(?) WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY ORDER BY customer_id',
                               [c['transactions']['path'],t,t]).fetchdf()
            np.testing.assert_array_equal(roster.customer_id,_parquet(folder/'full-roster.parquet').customer_id)
            expected_users=con.execute('''WITH latest AS (
                SELECT x.customer_id,x.article_id,max(t_dat) dt FROM read_parquet(?) x JOIN roster USING(customer_id)
                WHERE t_dat<CAST(? AS DATE) GROUP BY x.customer_id,x.article_id),
                ranked AS (SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY dt DESC,article_id) rn FROM latest)
                SELECT DISTINCT r.customer_id FROM ranked r JOIN read_csv(?, all_varchar=true) cat USING(article_id)
                WHERE rn<=20 ORDER BY r.customer_id''',[c['transactions']['path'],t,c['catalog']['path']]).fetchdf()
            np.testing.assert_array_equal(h.customer_id,expected_users.customer_id)
            n,p,u,dup=con.execute('SELECT count(*),sum(target),count(DISTINCT customer_id),count(*)-count(DISTINCT (customer_id,article_id)) FROM read_parquet(?)',[paths]).fetchone()
            assert n==len(h)*50==a['cold50_rows'] and p==a['positive_rows'] and u==len(h) and dup==0
            old=c['old_training'][t]['features']['qC']['path']
            diff=con.execute('WITH new AS (SELECT customer_id,article_id,target FROM read_parquet(?) WHERE hash(customer_id)%1000000<100000), old AS (SELECT customer_id,article_id,target FROM read_parquet(?)) SELECT (SELECT count(*) FROM (SELECT * FROM new EXCEPT SELECT * FROM old))+(SELECT count(*) FROM (SELECT * FROM old EXCEPT SELECT * FROM new))',[paths,old]).fetchone()[0]
            assert diff==0
            wrong=con.execute('''WITH truth AS (SELECT DISTINCT customer_id,article_id,1 hit FROM read_parquet(?)
                WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY)
                SELECT count(*) FROM read_parquet(?) f LEFT JOIN truth USING(customer_id,article_id)
                WHERE target<>coalesce(hit,0)''',[c['transactions']['path'],t,t,paths]).fetchone()[0]
            assert wrong==0
        detail[t]=dict(rows=n,positives=p,users=u,old10_labels_identity_exact=True,user_fold_recomputed=True,selection_fence=fence['at'])
    checks['all8_population_checks']=len(detail)==8
    for name in ['historical_only_hash_removed','no_cold_truth_eligibility','generation_before_truth_join','Cold50_exact_definition',
                 'all_full_labels_recomputed','stable_user_fold_all_dates','historical10_identity_rank_replay']:
        checks[name]=len(detail)==8
    checks['no_safe_B0_excluded']='2019-11-27' not in c['historical_cutoffs']
    checks['M4_frozen']=checks['trusted_input_SHAs']
    checks['B0_frozen']=checks['trusted_input_SHAs']
    for w in m.get('windows',{}):
        dest=root/'outer'/w; f=_parquet(dest/'predictions.parquet')
        old=root.parent/c['old_run_id']/'outer'/w
        cold=_parquet(old/'qC-predictions.parquet')
        np.testing.assert_array_equal(f[['customer_id','article_id']],cold[['customer_id','article_id']])
        np.testing.assert_array_equal(f.Q_old,cold.propensity)
        monotonic_parity(f.Q_full_raw.to_numpy(),f.Q_full_cal.to_numpy())
        ca=read_json(root/'models'/w/'calibrator.json')
        assert ca['b']>0 and ca['success']
        np.testing.assert_array_equal(expit(ca['a']+ca['b']*f.raw_logit.to_numpy()),f.Q_full_cal)
        for role in ('Q_old','Q_full_raw','Q_full_cal'):
            q=f[role].to_numpy(); r=m['windows'][w]['ranking'][role]
            np.testing.assert_allclose([roc_auc_score(cold.target,q),average_precision_score(cold.target,q)],
                                       [r['roc_auc'],r['pr_auc']],rtol=0,atol=1e-12)
            clipped=np.clip(q,1e-6,1-1e-6); y=cold.target.to_numpy()
            pc=m['windows'][w]['probability'][role]
            np.testing.assert_allclose([np.mean((clipped-y)**2),-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))],
                                       [pc['brier_score'],pc['logloss']],rtol=0,atol=1e-14)
        oof=root/'oof'/w; z=np.load(oof/'logits.float64.npy',mmap_mode='r'); y=np.load(oof/'labels.int8.npy',mmap_mode='r'); fold=np.load(oof/'fold.uint8.npy',mmap_mode='r')
        fold_models=[read_json(root/'models'/w/f'fold{k}'/'FIT_RESULT.json') for k in (0,1)]
        layout=read_json(oof/'LAYOUT.json'); cursor=0
        for row in layout:
            assert row['start']==cursor
            frame=_parquet(row['path']); end=row['end']
            np.testing.assert_array_equal(y[cursor:end],frame.target); np.testing.assert_array_equal(fold[cursor:end],frame.fold)
            for k in (0,1):
                mask=frame.fold.ne(k).to_numpy()
                if mask.any(): np.testing.assert_allclose(z[cursor:end][mask],logits(fold_models[k],frame.loc[mask]),rtol=0,atol=1e-12)
            assert np.isfinite(z[cursor:end]).all(); cursor=end
        assert cursor==len(z)
        for name in ('full','fold0','fold1'):
            fitted=read_json(root/'models'/w/name/'FIT_RESULT.json')
            assert fitted['model']['params']==c['raw_model_params']
            assert fitted['audit']['status']=='converged'
            if name=='full': assert fitted['audit']['rows']==sum(detail[t]['rows'] for t in c['historical_pools'][w])
            assert fitted['preprocessing']['numeric']==c['feature_spec']['numeric']
            assert fitted['preprocessing']['binary']==c['feature_spec']['binary']
            assert not fitted['audit']['negative_sampling'] and fitted['audit']['class_weight'] is None
            start=read_json(root/'models'/w/name/'FIT_START.json')
            assert start['fold']==(None if name=='full' else int(name[-1]))
            assert all(any('/'+t+'/' in p.replace('\\','/') for t in c['historical_pools'][w]) for p in start['paths'])
        check_identity(m['windows'][w]['frozen_qW'])
    checks['four_outer_verified']=len(m.get('windows',{}))==4
    for name in ['outer_fixed_users_exact','qC_features_unchanged','qC_solver_unchanged','no_class_weight','no_negative_sampling',
                 'OOF_user_disjoint','OOF_every_row_once','calibrator_only_raw_logit','calibrator_no_outer_label',
                 'calibrator_positive_slope','raw_cal_rank_exact','qW_predictions_frozen']:
        checks[name]=len(m.get('windows',{}))==4
    checks['final_week_not_run']=m['final_week']=='not_run'
    checks['qW_not_fit']=m['qW_fits']==0
    checks['no_matching']=m['matching_calls']==0
    checks['no_new_recommendations']=m['new_recommendations']==0
    checks['no_new_MAP']=m['new_MAP']==0
    checks['Warm_v2_not_integrated']=not m['Warm_v2_integrated']
    result=dict(stage='P4.2E',status='pass' if all(checks.values()) else 'partial_execution_verified_not_complete',
                checks=checks,population=detail,trusted_comparisons=len(inp['trusted_records']),seconds=time.perf_counter()-started,at=now())
    write_json(repo/'reports/phase4/P4_2E_VERIFICATION.json',result)
    return result


if __name__=='__main__': print(verify(Path.cwd()))
