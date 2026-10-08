"""Single frozen 10% Cold-A integration on reconstructed WV3-741.

No parameter/policy search and no final-week execution. Writes isolated outputs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .final_history_rebuild import ROOT, WROOT, OUT as WARM_ROOT, read, write, CONTRACT as HISTORY
from .mind_warm_final_confirmation import wv3_matrix, WV3_FEATURES, EXTRA_FEATURES, PAIR_TRANSFORM
from .warm_v3_clean_model_replay import fused_candidate_scores
from .warm_v2_engine import literal
from .p42f_contract import DATES, WINDOWS, GLOBAL, USER, earlier
from .p42f_core import batches, sample_edges
from .p42f_data import paths_for, frame, save_frame, records
from .p42f_features import relative_predictions
from .p42_data import normalize_cold_confidence, normalize_warm_confidence
from .p42_contract import FEATURE_SPEC
from .p43a_policy import policies, evaluate_grid, contexts, exact_ap
from .p43a_run import relevance, guard
from .p41a_contract import check_identity

RUN = ROOT/'artifacts/final/integration-10pct-v1'
REPORT = ROOT/'reports/final'
CONTRACT = REPORT/'FINAL_INTEGRATION_CONTRACT.json'
FROOT = ROOT/'artifacts/phase4/p4-2f-v1-relative-utility-hash10-user-oof'


def register():
    if CONTRACT.exists(): raise FileExistsError('Immutable integration contract exists')
    source=read(ROOT/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json')
    winner=next(x for x in source['model_configs'] if x['id']=='A1-L15-D6-M50')
    policy=next(x for x in policies() if x['id']=='p43a-policy-0332')
    assert winner['params']['num_leaves']==15 and winner['params']['min_child_samples']==50
    assert read(REPORT/'FINAL_HISTORY_REBUILD_VERIFICATION.json')['status']=='pass'
    c=dict(status='preregistered_before_new_action_labels',git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        warm='WV3-741',model=winner,policy=policy,historical_dates=DATES,windows=WINDOWS,features=GLOBAL,
        sampling=source['sampling']['A1'],hash=source['sampling']['hash'],graded_relevance=source['graded_relevance'],
        population='same original fixed hash10 W0 user cohort and frozen mapped-history Cold50; no future-Cold condition',
        warm_score='WV3-741 equal RRF60 of frozen pointwise/LambdaRank ranks over active Top50; inactive score unavailable',
        warm_rank_pct='new final position divided by complete original user candidate count, not by12 or150',
        warm150_membership='new WV2-601 fused ranks<=150, inactive original candidate ranks; WV3 swaps stay within50',
        auxiliary='exact original PIT qC E-full and qW R3 models, inference only on complete Cold50/new Warm12; normalize before overlap',
        prepare_order=['unlabeled full Cold50/new Warm12 features','relative scores and Warm150 membership','exclude overlaps and persist action population','join scoped next-week labels','exact new single replacement AP deltas','fixed A1 sampling'],
        gates=dict(mean_delta_gt=0,nondegrade_min=3,nondegrade_tolerance=-1e-5,worst_min=-5e-5,warm_mean_min=-2e-5,cold_sparse_mean_min=0,inserted_min=1,inserted_ge_removed=True),
        fullscale_this_session='not_run_user_requested_report_after10pct',final_week='not_run',no_retune=True,
        budget=dict(expected_minutes=[20,60],max_seconds=7200,threads=4,min_disk_gib=15,min_free_ram_gib=1.5),
        stop='integrity, temporal, feature, population, numerical or budget failure; preserve evidence; no silent downsampling')
    write(CONTRACT,c);return c


def unlabeled(db,p):
    cols=[x[0] for x in db.execute('DESCRIBE SELECT * FROM read_parquet(?)',[str(p)]).fetchall()]
    cols=[x for x in cols if x not in ['target','truth_count','already_w0']]
    return db.execute('SELECT '+','.join('"'+x+'"' for x in cols)+' FROM read_parquet(?)',[str(p)]).fetchdf()


def check_prior_inputs(t,paths):
    names=['P4_2_OUTPUT_MANIFEST.json','P4_2R_OUTPUT_MANIFEST.json','P4_2F_OUTPUT_MANIFEST.json']
    known={str(Path(r['path']).resolve()).lower():r for n in names for r in records(read(ROOT/'reports/phase4'/n))}
    result=[]
    for p in paths:
        key=str(Path(p).resolve()).lower()
        if key not in known: raise ValueError('No prior input receipt '+str(p))
        check_identity(known[key]);result.append(known[key])
    return result


def prepare(t,deadline):
    folder=RUN/'prepared'/t
    if (folder/'READY.json').exists(): return read(folder/'READY.json')
    if folder.exists(): raise FileExistsError('Incomplete preparation must be explicitly recovered: '+t)
    folder.mkdir(parents=True);started=time.perf_counter();guard(ROOT,deadline)
    fc=read(ROOT/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json');spec=read(HISTORY)['mappings'][t]
    paths=paths_for(ROOT,fc,t)
    inputs=check_prior_inputs(t,[paths['cold'],paths['selection'],paths['warm'],FROOT/f'user-state-{t}.parquet'])
    db=duckdb.connect(config={'threads':4,'memory_limit':'2GB'})
    cold=unlabeled(db,paths['cold']);selection=unlabeled(db,paths['selection'])
    missing=[k for k in selection if k not in cold]
    cold=cold.merge(selection[['customer_id','article_id',*missing]],on=['customer_id','article_id'],validate='one_to_one')
    if 'b0_score' in cold: cold=normalize_cold_confidence(cold)
    cold=cold.sort_values(['customer_id','b0_rank','article_id'],ignore_index=True)
    assert cold.groupby('customer_id').size().eq(50).all()
    warm=frame(WARM_ROOT/t/'warm-top12.parquet').rename(columns={'ap_rf':'warm_rank'})
    warm=warm.sort_values(['customer_id','warm_rank'],ignore_index=True)
    users=warm.customer_id.drop_duplicates().to_numpy()
    old_users=db.execute('SELECT DISTINCT customer_id FROM read_parquet(?) ORDER BY customer_id',[str(paths['warm'])]).fetchnumpy()['customer_id']
    np.testing.assert_array_equal(users,old_users)
    assert set(cold.customer_id)<=set(users)
    db.execute('ATTACH '+literal(WARM_ROOT/t/'stage.duckdb')+' AS ws (READ_ONLY)')
    # Score the full active Top50 BEFORE picking values for the new Top12.
    extras=list(dict.fromkeys(['user_unique_items_12w','user_days_since_last_purchase',*PAIR_TRANSFORM,*EXTRA_FEATURES]))
    active=db.execute('SELECT r.*,'+','.join('f.'+x for x in extras)+' FROM ws.top50 r JOIN read_parquet(?) f USING(customer_id,article_id) WHERE r.user_history_events_12w>0',[spec['base']]).fetchdf()
    matrix=wv3_matrix(active);values=[]
    for name in ['WV3-661','WV3-680']:
        b=lgb.Booster(model_file=str(WROOT/f'artifacts/warm_v3/{name}/MODEL.txt'))
        assert b.feature_name()==WV3_FEATURES;values.append(b.predict(matrix,num_threads=4))
    scoring=fused_candidate_scores(active,*values).rename(columns={'ranking_score':'warm_model_score'})
    warm=warm.merge(scoring,on=['customer_id','article_id'],how='left',validate='one_to_one')
    warm['warm_model_score_available']=warm.warm_model_score.notna().astype(int)
    db.register('keys',warm[['customer_id','article_id']])
    needed=list(dict.fromkeys(FEATURE_SPEC['qW']['numeric']+FEATURE_SPEC['qW']['binary']))
    derived={'warm_rank','warm_rank_pct','warm_user_percentile','warm_user_zscore','warm_model_score_available'}
    selected=[n for n in needed if n not in derived]
    features=db.execute('SELECT f.customer_id,f.article_id,'+','.join('f.'+x for x in selected)+' FROM read_parquet(?) f JOIN keys USING(customer_id,article_id)',[spec['base']]).fetchdf()
    warm=warm.merge(features,on=['customer_id','article_id'],validate='one_to_one')
    sizes=db.execute('SELECT customer_id,max(candidate_count) n FROM ws.ranks GROUP BY customer_id').fetchdf().set_index('customer_id').n
    warm['warm_rank_pct']=warm.warm_rank/warm.customer_id.map(sizes)
    warm=normalize_warm_confidence(warm)
    membership=db.execute('''SELECT customer_id,article_id FROM ws.ranks WHERE
      CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rf END<=150''').fetchdf()
    member=pd.MultiIndex.from_frame(membership)
    cold['also_in_Warm150']=pd.MultiIndex.from_frame(cold[['customer_id','article_id']]).isin(member).astype(int)
    cold['cold_only']=1-cold.also_in_Warm150
    # Helper strips target before inference; dummy has no outcome information.
    cold['target']=0;warm['target']=0
    relative=relative_predictions(ROOT,fc,t,cold,warm)
    cold=cold.drop(columns='target');warm=warm.drop(columns='target')
    ui={u:i for i,u in enumerate(users)};cold['user_index']=cold.customer_id.map(ui).astype(int)
    overlap=pd.MultiIndex.from_frame(cold[['customer_id','article_id']]).isin(pd.MultiIndex.from_frame(warm[['customer_id','article_id']]))
    cold=cold.loc[~overlap].reset_index(drop=True)
    state=frame(FROOT/f'user-state-{t}.parquet').set_index('customer_id').loc[users].reset_index()
    save_frame(cold,folder/'cold-before-labels.parquet');save_frame(warm,folder/'warm-before-labels.parquet')
    write(folder/'ACTION_POPULATION_FROZEN.json',dict(cutoff=t,users=len(users),cold_rows=len(cold),edges=len(cold)*12,
        full50_rows=len(selection),overlap_removed=int(overlap.sum()),relative=relative,inputs=inputs,labels_joined=False))
    guard(ROOT,deadline)
    # Only NOW read outcome labels, scoped to the registered cutoff and cohort.
    db.register('wanted',pd.DataFrame({'customer_id':users}))
    truth=db.execute('SELECT DISTINCT customer_id,article_id FROM read_parquet(?) JOIN wanted USING(customer_id) WHERE t_dat>=?::DATE AND t_dat<?::DATE+INTERVAL 7 DAY ORDER BY customer_id,article_id',[fc['transactions']['path'],t,t]).fetchdf()
    count=db.execute('SELECT article_id,count(*) n FROM read_parquet(?) WHERE t_dat<?::DATE GROUP BY article_id',[fc['transactions']['path'],t]).fetchdf()
    countmap=dict(zip(count.article_id,count.n));truth['interaction_count_before_cutoff']=truth.article_id.map(countmap).fillna(0).astype(int)
    ts={u:set(g.article_id) for u,g in truth.groupby('customer_id',sort=False)}
    assert set(ts)==set(users)
    for f in [cold,warm]:
        f['target']=[int(i in ts[u]) for u,i in zip(f.customer_id,f.article_id)]
        actual=f.article_id.map(countmap).fillna(0).to_numpy(int)
        if f is cold: np.testing.assert_array_equal(f.interaction_count_before_cutoff,actual)
        f['interaction_count_before_cutoff']=actual
    lists=warm.article_id.to_numpy().reshape(-1,12)
    base=np.array([exact_ap(x,ts[u]) for u,x in zip(users,lists)])
    data=dict(cutoff=t,users=users,cold=cold,warm=warm,state=state,truth=truth,truthsets=ts,countmap=countmap,
        warm_lists=lists,relevance=warm.target.to_numpy(int).reshape(-1,12),truth_count=np.array([len(ts[u]) for u in users]),baseline_ap=base,baseline_map=float(base.mean()))
    joblib.dump(data,folder/'data.joblib',compress=3)
    rows=0;tot=np.zeros(3,np.int64);kept=np.zeros(3,np.int64)
    if t in DATES:
        parts=[];ys=[];ids=[]
        for _,cc,x,y in batches(data):
            guard(ROOT,deadline);keep,_=sample_edges(t,cc.customer_id.to_numpy(),cc.article_id.to_numpy(),y)
            yy=y[keep];parts.append(x[keep.ravel()]);ys.append(yy);ids.append(np.broadcast_to(cc.user_index.to_numpy()[:,None],y.shape)[keep])
            tot+=np.array([(y>0).sum(),(y<0).sum(),(y==0).sum()]);kept+=np.array([(yy>0).sum(),(yy<0).sum(),(yy==0).sum()])
        x=np.concatenate(parts);y=np.concatenate(ys);idx=np.concatenate(ids);rows=len(y)
        assert np.array_equal(tot[:2],kept[:2]) and (np.diff(idx)>=0).all()
        groups=np.diff(np.r_[0,np.flatnonzero(np.diff(idx))+1,len(idx)]).astype(np.int32)
        np.save(folder/'X.npy',x);np.save(folder/'y.npy',y);np.save(folder/'groups.npy',groups)
    db.close()
    result=dict(status='completed',cutoff=t,users=len(users),cold_rows=len(cold),baseline_map=float(base.mean()),
        training_rows=rows,full_B_H_N=tot.tolist(),retained_B_H_N=kept.tolist(),row_weight=1,
        seconds=time.perf_counter()-started,final_week='not_run')
    write(folder/'READY.json',result);print(json.dumps(result),flush=True);return result


def run_window(w,deadline):
    c=read(CONTRACT);t=WINDOWS[w];folder=RUN/'models'/w
    if (folder/'RESULT.json').exists(): return read(folder/'RESULT.json')
    if folder.exists(): raise FileExistsError('Incomplete model run must be explicitly recovered')
    folder.mkdir(parents=True);start=time.perf_counter();guard(ROOT,deadline)
    dates=earlier(t)
    x=np.concatenate([np.load(RUN/'prepared'/s/'X.npy') for s in dates])
    target=np.concatenate([np.load(RUN/'prepared'/s/'y.npy') for s in dates])
    groups=np.concatenate([np.load(RUN/'prepared'/s/'groups.npy') for s in dates])
    assert groups.sum()==len(target)
    write(folder/'FIT_START.json',dict(params=c['model']['params'],features=GLOBAL,dates=dates,rows=len(target),groups=len(groups),row_weight=1))
    model=lgb.LGBMRanker(**c['model']['params'])
    with threadpool_limits(limits=4): model.fit(x,relevance(target),group=groups,feature_name=GLOBAL)
    model.booster_.save_model(str(folder/'model.txt'))
    del x,target,groups;gc.collect()
    data=joblib.load(RUN/'prepared'/t/'data.joblib')
    scores=np.empty((len(data['cold']),12),float)
    for lo,cc,x,_ in batches(data,labels=False):
        guard(ROOT,deadline);scores[lo:lo+len(cc)]=model.predict(x).reshape(-1,12)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite prediction')
    np.save(folder/'scores.npy',scores)
    row=evaluate_grid(data,scores,[c['policy']],folder/'policy',deadline)[0]
    row['baseline_map']=data['baseline_map'];row['window']=w;row['total_seconds']=time.perf_counter()-start
    # Independently reconstruct chosen single-edge decisions using old AP helper.
    from .metrics import apk
    masks=np.load(folder/'policy/policy-user-selected-mask.npy')[0]
    edges=np.load(folder/'policy/user-top10-global-edge.npy')
    lists=data['warm_lists'].copy();events=[];deltas=[]
    for i in np.flatnonzero(masks):
        assert masks[i]==1
        ci,slot=divmod(int(edges[i,0]),12);candidate=data['cold'].iloc[ci]
        assert candidate.user_index==i
        old=lists[i,slot];lists[i,slot]=candidate.article_id
        delta=apk(list(data['truthsets'][data['users'][i]]),list(lists[i]))-data['baseline_ap'][i]
        deltas.append(delta);events.append(dict(customer_id=data['users'][i],cold_article_id=candidate.article_id,
            removed_article_id=old,slot=slot+1,delta_ap=delta,inserted=int(candidate.target),removed=int(old in data['truthsets'][data['users'][i]])))
    ap=np.array([apk(list(data['truthsets'][u]),list(items)) for u,items in zip(data['users'],lists)])
    assert abs(ap.mean()-row['overall_map'])<1e-12
    assert sum(x['inserted'] for x in events)==row['inserted_positives']
    assert sum(x['removed'] for x in events)==row['removed_positives']
    row['executed_actions']=dict(beneficial=int(sum(d>0 for d in deltas)),harmful=int(sum(d<0 for d in deltas)),neutral=int(sum(d==0 for d in deltas)))
    row['independent_ap_replay_max_abs']=float(abs(ap.mean()-row['overall_map']))
    save_frame(pd.DataFrame(events,columns=['customer_id','cold_article_id','removed_article_id','slot','delta_ap','inserted','removed']),folder/'executed.parquet')
    save_frame(pd.DataFrame(dict(customer_id=np.repeat(data['users'],12),article_id=lists.ravel(),rank=np.tile(np.arange(1,13),len(lists)))),folder/'final-lists.parquet')
    write(folder/'RESULT.json',row);print(json.dumps(row),flush=True);return row


def finish():
    rows={w:read(RUN/'models'/w/'RESULT.json') for w in WINDOWS}
    delta=np.array([r['delta_map'] for r in rows.values()]);warm=np.mean([r['segments']['warm_21_plus']['delta'] for r in rows.values()]);cold=np.mean([r['segments']['all_cold_sparse']['delta'] for r in rows.values()])
    ins=sum(r['inserted_positives'] for r in rows.values());rem=sum(r['removed_positives'] for r in rows.values())
    tests=dict(mean_positive=bool(delta.mean()>0),three_nondegrade=bool((delta>=-1e-5).sum()>=3),worst=bool(delta.min()>=-5e-5),warm=bool(warm>=-2e-5),cold_sparse=bool(cold>=0),inserted_positive=ins>=1,efficiency=ins>=rem)
    result=dict(status='completed_10pct',windows=rows,mean_delta=float(delta.mean()),mean_warm_delta=float(warm),mean_cold_sparse_delta=float(cold),
        inserted_positives=ins,removed_positives=rem,gates=tests,fullscale_allowed=all(tests.values()),fullscale_status='not_run',final_week='not_run',evidence='development_selected_frozen_policy_on_new_warm; not independent holdout')
    write(REPORT/'FINAL_INTEGRATION_10PCT.json',result);print(json.dumps(result));return result


def recover_summary(w):
    """Only finish JSON serialization after a completed model/policy/list run."""
    folder=RUN/'models'/w
    if (folder/'RESULT.json').exists(): raise FileExistsError('Completed result must not be rewritten')
    row=read(folder/'policy/EVALUATION.json')['rows'][0]
    assert row['status']=='completed'
    executed=frame(folder/'executed.parquet')
    output=frame(folder/'final-lists.parquet')
    assert len(output)==row['users']*12
    row['baseline_map']=read(RUN/'prepared'/WINDOWS[w]/'READY.json')['baseline_map']
    row['window']=w;row['total_seconds']=None
    row['executed_actions']=dict(beneficial=int(executed.delta_ap.gt(0).sum()),harmful=int(executed.delta_ap.lt(0).sum()),neutral=int(executed.delta_ap.eq(0).sum()))
    assert abs(executed.delta_ap.sum()/row['users']-row['delta_map'])<1e-12
    row['serialization_recovery']='Only convert NumPy counts to JSON ints; reuse completed model, scores, policy and final lists; no refit or new decision'
    write(folder/'RESULT.json',row)
    write(REPORT/'FINAL_INTEGRATION_SERIALIZATION_RECOVERY.json',dict(window=w,model_retrained=False,policy_rerun=False,lists_changed=False,reason='np.int64 action-count JSON encoding'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','prepare','window','finish','recover-summary']);p.add_argument('--cutoff');p.add_argument('--window');a=p.parse_args()
    deadline=time.time()+7200
    try:
        if a.action=='register': register()
        elif a.action=='prepare': prepare(a.cutoff,deadline)
        elif a.action=='window':run_window(a.window,deadline)
        elif a.action=='recover-summary':recover_summary(a.window)
        else:finish()
    except Exception:
        write(REPORT/('FINAL_INTEGRATION_FAILURE_'+str(time.time_ns())+'.json'),dict(action=a.action,cutoff=a.cutoff,window=a.window,traceback=traceback.format_exc(),final_week='not_run'))
        raise
