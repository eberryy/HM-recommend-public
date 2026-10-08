"""Cutoff-safe adapters for unchanged Cold50 and exact W0."""
from pathlib import Path
from datetime import date,timedelta
import numpy as np
import pandas as pd
import duckdb
from .metrics import apk
from .p41a_contract import read_json,check_identity,write_json
from .p42_data import normalize_cold_confidence,validate_lineages,validate_warm_identity
from .p42_contract import WINDOWS,guard_cutoff
from .p42e_fit import logits
from .p42r_propensity import predict_repaired_propensity
from .p42f_contract import DATES,earlier
from .p42f_data import frame,paths_for,records,E_RUN

def relative_predictions(repo,c,t,cold,warm):
    choices=[(w,s) for w,s in WINDOWS.items() if s<=t and earlier(s) and
             max(date.fromisoformat(d)+timedelta(days=7) for d in earlier(s))<date.fromisoformat(t)]
    result={}
    if not choices:
        for side,f in [('qC',cold),('qW',warm)]:
            for k in ['within_user_rank','within_user_rank_pct','within_user_percentile']: f[side+'_'+k]=np.nan
            f[side+'_relative_available']=0
        return {'qC':None,'qW':None}
    w,_=choices[-1]
    for side,f,run,sub,manifest in [
        ('qC',cold,E_RUN,'full','P4_2E_OUTPUT_MANIFEST.json'),
        ('qW',warm,'p4-2r3-v1-lbfgs-solver-boundary-repair','qW-R2','P4_2R3_OUTPUT_MANIFEST.json')]:
        folder=repo/'artifacts/phase4'/run/'models'/w/sub
        known={str(Path(r['path']).resolve()).lower():r for r in records(read_json(repo/'reports/phase4'/manifest))}
        fitted={}
        for key in ['model','preprocessing']:
            p=folder/(key+'.json'); check_identity(known[str(p.resolve()).lower()]); fitted[key]=read_json(p)
        score=logits(fitted,f.drop(columns='target')) if side=='qC' else predict_repaired_propensity(fitted,f.drop(columns='target'))
        assert np.isfinite(score).all()
        series=pd.Series(score,index=f.index); group=series.groupby(f.customer_id,sort=False)
        rank=group.rank(ascending=False,method='first')
        f[side+'_within_user_rank']=rank
        f[side+'_within_user_rank_pct']=rank/group.transform('size')
        f[side+'_within_user_percentile']=(group.rank(method='average')-1)/(group.transform('size')-1)
        f[side+'_relative_available']=1
        result[side]={'model':str(folder/'model.json'),'training_cutoffs':earlier(WINDOWS[w]),'scoring_cutoff':t,
                      'latest_label_end':str(max(date.fromisoformat(d)+timedelta(days=7) for d in earlier(WINDOWS[w])))}
    return result

def load(repo,c,t,root):
    guard_cutoff(t); validate_lineages(c['inputs'][t],t)
    p=paths_for(repo,c,t); cold=frame(p['cold']); warm=frame(p['warm'])
    selection=frame(p['selection'])
    missing=[k for k in selection if k not in cold and k!='target']
    if missing: cold=cold.merge(selection[['customer_id','article_id',*missing]],on=['customer_id','article_id'],validate='one_to_one')
    cold=normalize_cold_confidence(cold)
    cold=cold.sort_values(['customer_id','b0_rank','article_id'],ignore_index=True)
    warm=warm.sort_values(['customer_id','warm_rank'],ignore_index=True); validate_warm_identity(warm)
    users=warm.customer_id.drop_duplicates().to_numpy(); ui={u:i for i,u in enumerate(users)}
    assert set(cold.customer_id)<=set(users)
    assert cold.groupby('customer_id').size().eq(50).all()
    with duckdb.connect(config={'threads':4,'memory_limit':'2GB'}) as db:
        original=db.execute('SELECT customer_id,article_id,warm_rank FROM read_parquet(?) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank',[str(p['warm150'])]).fetchdf()
        pd.testing.assert_frame_equal(warm[list(original.columns)],original,check_dtype=False)
        membership=db.execute('SELECT customer_id,article_id FROM read_parquet(?)',[str(p['warm150'])]).fetchdf()
        wanted=pd.DataFrame({'customer_id':users}); db.register('wanted',wanted)
        truth=db.execute('SELECT DISTINCT customer_id,article_id FROM read_parquet(?) JOIN wanted USING(customer_id) WHERE t_dat>=?::DATE AND t_dat<?::DATE+INTERVAL 7 DAY ORDER BY customer_id,article_id',[c['transactions']['path'],t,t]).fetchdf()
        counts=db.execute('SELECT article_id,count(*) n FROM read_parquet(?) WHERE t_dat<?::DATE GROUP BY article_id',[c['transactions']['path'],t]).fetchdf()
    ts={u:set(g.article_id) for u,g in truth.groupby('customer_id',sort=False)}; countmap=dict(zip(counts.article_id,counts.n))
    truth['interaction_count_before_cutoff']=truth.article_id.map(countmap).fillna(0).astype(int)
    for f in [cold,warm]:
        actual=np.array([i in ts[u] for u,i in zip(f.customer_id,f.article_id)],int)
        np.testing.assert_array_equal(actual,f.target)
        np.testing.assert_array_equal(f.interaction_count_before_cutoff,f.article_id.map(countmap).fillna(0))
    rel=relative_predictions(repo,c,t,cold,warm)
    member=set(zip(membership.customer_id,membership.article_id))
    cold['also_in_Warm150']=[int((u,i) in member) for u,i in zip(cold.customer_id,cold.article_id)]
    cold['cold_only']=1-cold.also_in_Warm150
    cold['user_index']=cold.customer_id.map(ui).astype(int)
    lists=warm.article_id.to_numpy().reshape(-1,12); ws={u:set(x) for u,x in zip(users,lists)}
    overlap=np.array([i in ws[u] for u,i in zip(cold.customer_id,cold.article_id)])
    cold=cold.loc[~overlap].reset_index(drop=True)
    state=frame(root/f'user-state-{t}.parquet').set_index('customer_id').loc[users].reset_index()
    assert np.array_equal(state.customer_id,users)
    base=np.array([apk(list(ts[u]),list(x)) for u,x in zip(users,lists)])
    audit=dict(cutoff=t,W0_users=len(users),Cold50_users=len(set(cold.customer_id)),main_cold_rows=len(cold),
               overlap_removed=int(overlap.sum()),W0_map=float(base.mean()),relative=rel,labels_recomputed=True,
               W0_identity_exact=True,full50_before_overlap=True)
    write_json(root/f'INPUT_AUDIT_{t}.json',audit)
    return dict(cutoff=t,users=users,cold=cold,warm=warm,state=state,truth=truth,truthsets=ts,countmap=countmap,
                warm_lists=lists,relevance=warm.target.to_numpy(int).reshape(-1,12),truth_count=np.array([len(ts[u]) for u in users]),
                baseline_ap=base,baseline_map=float(base.mean()),audit=audit)
