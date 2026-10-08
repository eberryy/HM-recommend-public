"""P4.2D: read frozen predictions and executed edges; never fit or re-match."""
from __future__ import annotations

import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .p41a_contract import identity, check_identity, read_json, write_json
from .p41a_data import identities, parquet, save_frame
from .p42_contract import WINDOWS, TAUS, branch_guard, guard_cutoff
from .p42d_contract import RUN_ID, FRACTIONS, BINS, SCORES, preregister
from .p42d_stats import summary, discrimination, correlations, rate, rank_metrics, single_ap_delta


def now():
    return datetime.now(timezone.utc).isoformat()


def clipped(q):
    q = np.asarray(q,float)
    assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    return np.clip(q,1e-6,1-1e-6)


def logit(q):
    q=clipped(q)
    return np.log(q/(1-q))


def census(f):
    p=f.loc[f.target.eq(1)]
    return dict(rows=len(f), positives=len(p), users=int(f.customer_id.nunique()),
                positive_users=int(p.customer_id.nunique()),positive_rate=rate(len(p),len(f)))


def pool_census(f):
    return {name:census(f.loc[mask]) for name,mask in {
        'all':np.ones(len(f),bool),'strict_cold':f.strict_cold_flag.eq(1),
        'sparse1_5':f.sparse1_5_flag.eq(1),'cold_only':f.source_branch.eq('cold_only'),
        'also_in_Warm150':~f.source_branch.eq('cold_only')}.items()}


def tail(f, base):
    out=census(f)
    out.update(lift=out['positive_rate']/base if base and len(f) else None,
               qC=summary(f.propensity),b0_rank=summary(f.b0_rank),
               strict_rows=int(f.strict_cold_flag.sum()),sparse_rows=int(f.sparse1_5_flag.sum()))
    return out


def rank_audit(cold, eligible):
    pos=eligible.loc[eligible.target.eq(1)].copy()
    fields=['b0_rank','b0_rank_pct','b0_user_percentile','b0_user_zscore','normalized_margin_to_rank2',
            'normalized_margin_to_rank5','normalized_margin_to_user_median','m4_coarse_rank',
            'qc_rank','qc_rank_pct','qc_percentile']
    delta=pos.qc_rank-pos.b0_rank
    return pos, dict(positive_distributions={k:summary(pos[k]) for k in fields},
        b0=rank_metrics(pos.b0_rank),qC=rank_metrics(pos.qc_rank),
        paired=dict(improved=int((delta<0).sum()),same=int((delta==0).sum()),worsened=int((delta>0).sum()),
                    rank_delta=summary(delta)),
        discrimination={k:discrimination(-cold.b0_rank if k=='negative_b0_rank' else
                       cold.propensity if k=='qC' else cold[k],cold.target) for k in SCORES})


def incidence_choice(cold):
    # Canonical complete50 arrays; every candidate retained including W0 overlap.
    assert cold.groupby('customer_id',sort=False).size().eq(50).all()
    q=cold.propensity.to_numpy().reshape(-1,50)
    y=cold.target.to_numpy().reshape(-1,50)
    z=cold.b0_user_zscore.to_numpy().reshape(-1,50)
    margin=cold.normalized_margin_to_user_median.to_numpy().reshape(-1,50)
    label=y.sum(axis=1)>0
    agg={'max_qC':q.max(axis=1),'sum_qC':q.sum(axis=1),'top5_sum_qC':np.sort(q,axis=1)[:,-5:].sum(axis=1),
         'noisy_or_qC':-np.expm1(np.log1p(-q).sum(axis=1)),
         'max_B0_user_zscore':z.max(axis=1),'max_margin_to_user_median':margin.max(axis=1)}
    out={'users':len(y),'positive_users':int(label.sum()),'incidence_rate':float(label.mean()),
         'aggregates':{k:discrimination(v,label) for k,v in agg.items()},'conditional':{}}
    for name,column in [('B0','b0_rank'),('qC','qc_rank')]:
        ranks=cold[column].to_numpy().reshape(-1,50)[label]
        yp=y[label].astype(bool)
        n=yp.sum(axis=1)
        first=np.where(yp,ranks,51).min(axis=1)
        out['conditional'][name]={'users':len(n),'positive_candidates':int(n.sum()),
            'macro_recall':{str(k):float(((yp&(ranks<=k)).sum(axis=1)/n).mean()) for k in (1,5,10,20,50)},
            'mrr':float((1/first).mean()),'pooled_candidate':rank_metrics(ranks[yp])}
    frame=pd.DataFrame(agg)
    frame.insert(0,'customer_id',cold.customer_id.to_numpy().reshape(-1,50)[:,0])
    frame['incidence_label']=label
    return out,frame


def qc_tails(cold):
    n=len(cold); base=float(cold.target.mean())
    order=np.argsort(cold.propensity.to_numpy(),kind='stable')
    bins=[]
    for low,high in zip(BINS[:-1],BINS[1:]):
        part=cold.iloc[order[math.floor(n*low):math.floor(n*high)]]
        t=tail(part,base)
        observed=t['positive_rate']; predicted=t['qC']['mean']
        t.update(low=low,high=high,predicted_over_observed=predicted/observed if observed else None,
                 ratio_status='finite' if observed else 'infinite' if predicted and predicted>0 else 'undefined')
        bins.append(t)
    return {'global':{str(f):tail(cold.iloc[order[-math.ceil(n*f):]],base) for f in FRACTIONS},
            'within_user':{str(k):tail(cold.loc[cold.qc_rank<=k],base) for k in (1,2,5,10)},
            'bins':bins,'full':tail(cold,base)}


def availability(cold,eligible,u,first,cutoff):
    n=len(cold); order=np.argsort(cold.propensity.to_numpy(),kind='stable')
    strict=cold.strict_cold_flag.eq(1).to_numpy()
    masks={'all_strict':strict,'strict_truth':strict&cold.target.eq(1).to_numpy()}
    for f in FRACTIONS:
        mask=np.zeros(n,bool); mask[order[-math.ceil(n*f):]]=True
        masks['global_top_'+str(f)]=mask&strict
    mask=np.zeros(n,bool)
    mask[eligible.loc[(u.max(axis=1)>0)&eligible.strict_cold_flag.eq(1),'cold_row_index']]=True
    masks['strict_any_U_gt0']=mask
    dates=pd.to_datetime(cold.article_id.map(first))
    days=(dates-pd.Timestamp(cutoff)).dt.days.to_numpy()
    assert not np.any(days[strict]<0)
    cold_proxy=cold[['customer_id','article_id','strict_cold_flag']].copy()
    cold_proxy['audit_only_first_sale_days']=days
    output={}
    for name,mask in masks.items():
        counts={bucket:int(np.sum(mask&cond)) for bucket,cond in {
            'within_next_7d':(days>=0)&(days<7),'8_28d':(days>=7)&(days<28),
            '29_84d':(days>=28)&(days<84),'over84d':days>=84,
            'not_observed_before_embargo':~np.isfinite(days)}.items()}
        assert sum(counts.values())==int(mask.sum())
        output[name]={'rows':int(mask.sum()),'unique_items':int(cold.loc[mask,'article_id'].nunique()),
                      'buckets':counts,'shares':{k:rate(v,mask.sum()) for k,v in counts.items()}}
    output['observation_days']=(pd.Timestamp('2020-09-16')-pd.Timestamp(cutoff)).days
    output['right_censored']=True
    return output,cold_proxy.loc[strict]


def executed_summary(e):
    return {'rows':len(e),'beneficial':int((e.net_positive>0).sum()),'harmful':int((e.net_positive<0).sum()),
            'neutral':int((e.net_positive==0).sum()),
            'distributions':{k:summary(e[k]) for k in ('qC','qW','utility','b0_cold_rank','warm_slot_rank')}}


def pair_audit(eligible,warm,u,executed,truth_count):
    ui=eligible.user_index.to_numpy(dtype=int)
    wr=warm.target.to_numpy(dtype=np.int8).reshape(-1,12)[ui]
    cy=eligible.target.to_numpy(dtype=np.int8)
    labels=cy[:,None]-wr
    assert labels.shape==u.shape and np.isin(labels,[-1,0,1]).all()
    ap=single_ap_delta(wr,cy,truth_count[ui])
    pos=np.flatnonzero(cy==1)
    p=eligible.iloc[pos].copy()
    best=u[pos].argmax(axis=1)
    safe=np.where(wr[pos]==0,u[pos],-np.inf)
    sbest=safe.argmax(axis=1); has_safe=np.isfinite(safe.max(axis=1))
    wq=clipped(warm.propensity).reshape(-1,12)[ui]
    p['best_U_all']=u[pos,best]; p['best_warm_rank']=best+1
    p['best_qW']=wq[pos,best]
    p['oracle_best_U_safe']=np.where(has_safe,safe.max(axis=1),np.nan)
    p['oracle_safe_warm_rank']=np.where(has_safe,sbest+1,np.nan)
    kinds={'beneficial':labels==1,'harmful':labels==-1,'neutral':labels==0,
        'strict_beneficial':(labels==1)&eligible.strict_cold_flag.to_numpy(dtype=bool)[:,None],
        'sparse_beneficial':(labels==1)&eligible.sparse1_5_flag.to_numpy(dtype=bool)[:,None]}
    distributions={k:summary(u[mask]) for k,mask in kinds.items()}
    significant=labels!=0
    comparison={'primary':discrimination(u[significant],labels[significant]==1),
                'secondary':discrimination(u,labels==1)}
    alignment={'overall':correlations(u,ap),'by_warm_rank':{}}
    alternatives={}
    for name,score in [('D1',clipped(eligible.qC)[:,None]-wq),('D2',clipped(eligible.qC)[:,None]-2*wq)]:
        alternatives[name]={'primary':discrimination(score[significant],labels[significant]==1),
                             'alignment':correlations(score,ap)}
    slots={}
    for j in range(12):
        lab=labels[:,j]; mask=lab!=0
        alignment['by_warm_rank'][str(j+1)]={**correlations(u[:,j],ap[:,j]),
             'primary':discrimination(u[mask,j],lab[mask]==1)}
        slots[str(j+1)]={'edges':len(lab),'beneficial':int((lab==1).sum()),'harmful':int((lab==-1).sum()),
             'beneficial_rate':float((lab==1).mean()),'harmful_rate':float((lab==-1).mean()),
             'mean_qW':float(wq[:,j].mean()),'mean_U':float(u[:,j].mean()),
             'mean_single_delta_ap':float(ap[:,j].mean()),'executed':{}}
    lookup={int(v):i for i,v in enumerate(eligible.cold_row_index)}
    survival={}; losses={}; selected_audit={}
    for variant,tau in TAUS.items():
        ex=executed.loc[executed.variant.eq(variant)]
        selected=np.zeros(u.shape,bool)
        for e in ex.itertuples():
            ci=lookup[int(e.cold_row_index)]; wi=int(e.warm_slot_rank)-1
            assert eligible.iloc[ci].article_id==e.cold_article_id
            assert u[ci,wi]==e.utility and u[ci,wi]>tau
            assert labels[ci,wi]==e.net_positive
            np.testing.assert_allclose(ap[ci,wi],e.individual_delta_ap,rtol=1e-13,atol=1e-15)
            selected[ci,wi]=True
        assert selected.sum()==len(ex)
        surviving=(u>tau)&(labels==1)
        any_edge=(u[pos]>tau).any(axis=1)
        any_safe=surviving[pos].any(axis=1)
        p['any_edge_'+variant]=any_edge
        p['any_beneficial_edge_'+variant]=any_safe
        p['selected_'+variant]=selected[pos].any(axis=1)
        p['beneficial_selected_'+variant]=(selected[pos]&(labels[pos]==1)).any(axis=1)
        groups={'all':np.ones(len(pos),bool),'strict':p.strict_cold_flag.eq(1).to_numpy(),
                'sparse':p.sparse1_5_flag.eq(1).to_numpy()}
        for lo,hi,name in [(1,1,'1'),(2,5,'2_5'),(6,10,'6_10'),(11,20,'11_20'),(21,50,'21_50')]:
            groups['b0_'+name]=p.b0_rank.between(lo,hi).to_numpy()
        survival[variant]={k:{'positive_candidates':int(mask.sum()),'any_edge':int((mask&any_edge).sum()),
              'any_beneficial_edge':int((mask&any_safe).sum()),'survival_recall':rate((mask&any_edge).sum(),mask.sum())}
              for k,mask in groups.items()}
        selected_cold=selected.any(axis=1)
        # Node occupancy checked against same USER, not global slot number.
        occupied={}
        for ci,wi in zip(*np.nonzero(selected)):
            occupied[(int(ui[ci]),int(wi))]=int(ci)
        conflicts=dict(same_cold_only=0,same_warm_only=0,both=0,neither=0,
                       higher_weight_nonbeneficial_node=0)
        detail=[]
        for ci,wi in zip(*np.nonzero(surviving&~selected)):
            sc=bool(selected_cold[ci]); oc=occupied.get((int(ui[ci]),int(wi)))
            sw=oc is not None
            key='both' if sc and sw else 'same_cold_only' if sc else 'same_warm_only' if sw else 'neither'
            conflicts[key]+=1
            blockers=[]
            if sc: blockers.extend((int(ci),int(w)) for w in np.flatnonzero(selected[ci]))
            if sw: blockers.append((oc,int(wi)))
            higher=any(labels[c,w]!=1 and u[c,w]>u[ci,wi] for c,w in blockers)
            conflicts['higher_weight_nonbeneficial_node']+=int(higher)
            detail.append({'cold_row_index':int(eligible.iloc[ci].cold_row_index),'warm_slot_rank':int(wi+1),
                           'conflict':key,'higher_weight_nonbeneficial_node':higher})
        losses[variant]={'beneficial_edges_above_tau':int(surviving.sum()),
            'beneficial_candidates_above_tau':int(surviving.any(axis=1).sum()),
            'beneficial_edges_selected':int((selected&(labels==1)).sum()),
            'beneficial_candidates_selected':int((selected&(labels==1)).any(axis=1).sum()),
            'lost_candidates':int((surviving.any(axis=1)&~(selected&(labels==1)).any(axis=1)).sum()),
            'unselected_surviving_edges':int((surviving&~selected).sum()),'conflicts':conflicts,
            'edge_details':detail}
        selected_audit[variant]={'all':executed_summary(ex),
            'beneficial':executed_summary(ex.loc[ex.net_positive>0]),
            'harmful':executed_summary(ex.loc[ex.net_positive<0])}
        for j in range(12):
            slots[str(j+1)]['executed'][variant]=executed_summary(ex.loc[ex.warm_slot_rank.eq(j+1)])
    return dict(positive_rows=p,labels=labels,ap=ap,utility_distributions=distributions,
                comparison=comparison,alignment=alignment,alternatives=alternatives,slots=slots,
                survival=survival,losses=losses,executed=selected_audit)


def warm_audit(warm,executed):
    order=np.argsort(warm.propensity.to_numpy(),kind='stable'); base=float(warm.target.mean()); n=len(warm)
    def describe(f):
        positive=int(f.target.sum())
        return {'rows':len(f),'positives':positive,'positive_rate':rate(positive,len(f)),
                'rate_vs_full':rate(positive,len(f))/base if len(f) and base else None,
                'qW':summary(f.propensity),'warm_rank':summary(f.warm_rank)}
    return {'full':describe(warm),'discrimination':discrimination(warm.propensity,warm.target),
        'bottom':{str(f):describe(warm.iloc[order[:math.ceil(n*f)]]) for f in (.01,.05,.1,.2)},
        'removed':{v:{'rows':len(e),'positives':int(e.removed_positive.sum()),
            'positive_rate':rate(e.removed_positive.sum(),len(e)),
            'qW':summary(e.qW),'warm_rank':summary(e.warm_slot_rank)}
            for v in TAUS for e in [executed.loc[executed.variant.eq(v)]]}}


def load_saved(repo,c,m,window,db):
    cutoff=guard_cutoff(WINDOWS[window]); folder=repo/'artifacts/phase4'/c['run_id']/'outer'/window
    cold,warm,eligible,executed,truth=[parquet(folder/(k+'.parquet')) for k in
        ('qC-predictions','qW-predictions','eligible_cold','executed','truth')]
    u=np.load(folder/'pair-utility.float64.npy',mmap_mode='r',allow_pickle=False)
    assert not cold.duplicated(['customer_id','article_id']).any()
    assert cold.groupby('customer_id',sort=False).size().eq(50).all()
    users=warm.customer_id.to_numpy().reshape(-1,12)[:,0]
    np.testing.assert_array_equal(warm.warm_rank,np.tile(np.arange(1,13),len(users)))
    for frame,key,cols in [(cold,'cold50',['customer_id','article_id','cold_rank','b0_rank','b0_score','m4_coarse_rank']),
                            (warm,'warm150',['customer_id','article_id','warm_rank'])]:
        sql='SELECT '+','.join(cols)+' FROM read_parquet(?)'
        sql+= ' WHERE warm_rank<=12 ORDER BY customer_id,warm_rank' if key=='warm150' else ' ORDER BY customer_id,cold_rank,article_id'
        old=db.execute(sql,[c['inputs'][cutoff][key]['path']]).fetchdf()
        pd.testing.assert_frame_equal(frame[cols],old,check_dtype=False,check_exact=True)
    qrank=cold.sort_values(['customer_id','propensity','b0_rank','article_id'],ascending=[True,False,True,True]).groupby('customer_id',sort=False).cumcount()+1
    cold['qc_rank']=qrank.reindex(cold.index).to_numpy()
    cold['qc_rank_pct']=cold.qc_rank/50
    cold['qc_percentile']=(cold.groupby('customer_id',sort=False).propensity.rank(method='average')-1)/49
    cold['cold_row_index']=np.arange(len(cold))
    eidx=eligible.cold_row_index.to_numpy(dtype=int)
    pd.testing.assert_frame_equal(cold.iloc[eidx][['customer_id','article_id']].reset_index(drop=True),
                                  eligible[['customer_id','article_id']],check_dtype=False,check_exact=True)
    assert not cold.iloc[eidx].already_w0.any()
    np.testing.assert_array_equal(eidx,np.flatnonzero(~cold.already_w0.to_numpy()))
    for key in ('qc_rank','qc_rank_pct','qc_percentile'):
        eligible[key]=cold.iloc[eidx][key].to_numpy()
    np.testing.assert_array_equal(clipped(cold.iloc[eidx].propensity),eligible.qC)
    wrq=clipped(warm.propensity).reshape(-1,12)
    np.testing.assert_array_equal(u,logit(eligible.qC)[:,None]-logit(wrq)[eligible.user_index.to_numpy()])
    actual=db.execute('SELECT DISTINCT customer_id,article_id FROM read_parquet(?) WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY',
                      [c['transactions']['path'],cutoff,cutoff]).fetchdf()
    actual=actual.loc[actual.customer_id.isin(users)].sort_values(['customer_id','article_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(truth[['customer_id','article_id']].sort_values(['customer_id','article_id']).reset_index(drop=True),actual,check_dtype=False)
    truthsets={uid:set(g.article_id) for uid,g in actual.groupby('customer_id',sort=False)}
    for f in (cold,warm):
        np.testing.assert_array_equal(f.target,[int(a in truthsets[uid]) for uid,a in zip(f.customer_id,f.article_id)])
    assert int(cold.target.sum())==m['calibration'][window]['qC']['positives']
    assert len(cold)==m['calibration'][window]['qC']['rows']
    truth_count=np.array([len(truthsets[uid]) for uid in users])
    return cold,warm,eligible,executed,u,truth_count


def verdicts(windows):
    ws=list(windows.values())
    def classify(n): return 'supported' if n>=3 else 'rejected' if n<=1 else 'weak'
    order=sum(w['ranking']['paired']['rank_delta']['mean']<=0 for w in ws)
    choice=sum(w['incidence']['conditional']['B0']['mrr']>w['incidence']['conditional']['qC']['mrr'] for w in ws)
    incident=sum(w['incidence']['aggregates']['max_qC']['roc_auc']>.55 for w in ws)
    tail_ok=sum(all(w['tails']['global'][str(f)]['positive_rate']>=w['tails']['global'][str(f)]['qC']['mean']/2
                    and w['tails']['global'][str(f)]['lift']>1 for f in (.001,.0001)) for w in ws)
    warm_ok=sum(w['warm']['discrimination']['roc_auc']>.55 and w['warm']['bottom']['0.1']['rate_vs_full']<1 for w in ws)
    u_ok=sum(w['pair']['comparison']['primary']['roc_auc']>.55 and w['pair']['alignment']['overall']['pearson']>0 for w in ws)
    u_bad=sum(w['pair']['comparison']['primary']['roc_auc']<=.5 or w['pair']['alignment']['overall']['pearson']<=0 for w in ws)
    n=sum(w['census']['eligible']['all']['positives'] for w in ws)
    survive=sum(w['pair']['losses']['A_tau0']['beneficial_candidates_above_tau'] for w in ws)
    lost=sum(w['pair']['losses']['A_tau0']['lost_candidates'] for w in ws)
    availability_ok=sum((w['availability']['all_strict']['shares']['within_next_7d'] or 0)<.5 for w in ws)
    incidence_verdict='supported' if incident>=3 and choice>=3 else 'rejected' if sum(w['incidence']['aggregates']['max_qC']['roc_auc']<=.5 for w in ws)>=3 or choice<=1 else 'weak'
    return {'qc_tail_reliability':classify(tail_ok),
        'qc_preserves_b0_positive_order':'supported' if order>=3 else 'rejected' if order<=1 else 'mixed',
        'qw_removal_risk_model':classify(warm_ok),
        'logodds_pair_utility_alignment':'supported' if u_ok>=3 else 'rejected' if u_bad>=3 else 'weak',
        'matching_is_primary_bottleneck':False if survive<=n/2 else True if lost>survive/2 else 'inconclusive',
        'user_incidence_factorization_hypothesis':incidence_verdict,'conditional_b0_choice_hypothesis':classify(choice),
        'strict_cold_availability_asymmetry':'supported' if availability_ok>=3 else 'inconclusive',
        'full_history_training_scaleup':'optional' if incidence_verdict=='supported' or choice>=3 else 'not_justified'}


def run(repo):
    repo=Path(repo).resolve(); branch_guard(repo)
    contract=preregister(repo); reports=repo/'reports/phase4'; root=repo/'artifacts/phase4'/RUN_ID
    if (root/'EXECUTION_START.json').exists(): raise ValueError('attempt already exists; do not overwrite')
    root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); write_json(root/'EXECUTION_START.json',{'at':now(),'contract':identity(reports/'P4_2D_EXPERIMENT_CONTRACT.json')})
    m=read_json(reports/'P4_2R3_metrics.json'); c=read_json(reports/'P4_2R3_EXPERIMENT_CONTRACT.json')
    previous=read_json(reports/'P4_2R3_OUTPUT_MANIFEST.json')
    records={r['path']:r for r in identities(previous)}
    records[c['transactions']['path']]=c['transactions']
    # Existing trusted manifest answers whether saved inputs changed across tasks.
    for r in records.values(): check_identity(r)
    write_json(root/'INPUT_VERIFICATION.json',{'trusted_records':list(records.values()),'comparisons':len(records),'pass':True})
    for a in contract['source_review'].values(): check_identity(a)
    out={'stage':'P4.2D','run_id':RUN_ID,'windows':{},'final_week':'not_run','Warm_v2_integrated':False,
         'new_model_fits':0,'optimizer_calls':0,'matching_calls':0,'new_candidate_generation':0}
    forbidden=[]
    def call_guard(frame,event,arg):
        if event=='call':
            name=frame.f_code.co_name; module=frame.f_globals.get('__name__','')
            if (name in ('fit','fit_transform','fit_predict','minimize','fmin_l_bfgs_b','exact_matching','predict_propensity','predict_repaired_propensity')
                or ('hm_recsys' in module and name.startswith(('fit_','train_','predict_')))):
                forbidden.append(module+'.'+name); raise RuntimeError('forbidden diagnostic call: '+forbidden[-1])
    sys.setprofile(call_guard)
    try:
        with duckdb.connect() as db:
            db.execute("SET threads=4"); db.execute("SET memory_limit='4GB'")
            first=db.execute("SELECT article_id,min(t_dat) first_sale FROM read_parquet(?) WHERE t_dat<DATE '2020-09-16' GROUP BY article_id",[c['transactions']['path']]).fetchdf().set_index('article_id').first_sale
            for window,cutoff in WINDOWS.items():
                if time.perf_counter()-started>1800: raise RuntimeError('budget exhausted')
                print('audit '+window,flush=True)
                cold,warm,eligible,executed,u,nt=load_saved(repo,c,m,window,db)
                _,ranking=rank_audit(cold,eligible)
                inc,incrows=incidence_choice(cold)
                pair=pair_audit(eligible,warm,u,executed,nt)
                avail,proxyrows=availability(cold,eligible,u,first,cutoff)
                folder=root/window
                save_frame(pair.pop('positive_rows'),folder/'positive-candidates.parquet',cutoff,{'audit_only':True})
                save_frame(incrows,folder/'user-incidence.parquet',cutoff,{'audit_only':True})
                save_frame(proxyrows,folder/'strict-first-sale-proxy.parquet',cutoff,{'audit_only':True,'excluded_final_week':True})
                np.save(folder/'edge-labels.int8.npy',pair.pop('labels'),allow_pickle=False)
                np.save(folder/'single-delta-ap.float64.npy',pair.pop('ap'),allow_pickle=False)
                result={'cutoff':cutoff,'census':{'full50':pool_census(cold),'excluded_overlap':pool_census(cold.loc[cold.already_w0]),
                    'eligible':pool_census(eligible)},'ranking':ranking,'incidence':inc,'tails':qc_tails(cold),
                    'pair':pair,'availability':avail,'warm':warm_audit(warm,executed)}
                out['windows'][window]=result
                write_json(folder/'window-metrics.json',result)
                print(json.dumps({'window':window,'positive_candidates':result['census']['eligible']['all']['positives'],
                     'survival_tau0':pair['survival']['A_tau0']['all'],'paired_rank':ranking['paired']},ensure_ascii=False),flush=True)
                del cold,warm,eligible,executed,u,pair,result; gc.collect()
            scale={}
            for cutoff,p in c['prepared_reuse'].items():
                guard_cutoff(cutoff)
                r=p['features']['qC']; w=p['features']['qW']
                check_identity(r); check_identity(w)
                stats=db.execute('SELECT count(DISTINCT customer_id) users,count(*) candidate_rows,sum(target) positives,count(DISTINCT CASE WHEN target=1 THEN customer_id END) positive_users FROM read_parquet(?)',[r['path']]).fetchone()
                assert stats[0]==r['users']==w['users'] and stats[1]==r['row_count'] and stats[2]==r['positive_rows']
                scale[cutoff]={'exact_hash10':dict(zip(['users','candidate_rows','positive_rows','positive_users'],map(int,stats))),
                    'estimated_hash100':dict(zip(['users','candidate_rows','positive_rows','positive_users'],[int(x)*10 for x in stats])),
                    'exact_hash100':None,'candidate_regeneration':False,'label_end':r['label_end'],
                    'estimate_warning':'10x deterministic user-cluster sample expansion, not independent-row confidence interval or exact positive support; retrieval/ranking100% not generated'}
            out['scaleup']=scale
        out['verdicts']=verdicts(out['windows']); out['status']='measured_pending_verification'
    except Exception:
        out['status']='engineering_failure'; out['traceback']=traceback.format_exc()
        raise
    finally:
        sys.setprofile(None)
        out['runtime_forbidden_calls']=forbidden; out['seconds']=time.perf_counter()-started; out['finished_at_utc']=now()
        write_json(reports/'P4_2D_metrics.json',out)
    write_components(reports,out)
    print(json.dumps({'status':out['status'],'seconds':out['seconds'],'verdicts':out['verdicts']},ensure_ascii=False),flush=True)
    return out


def write_components(reports,out):
    components={'cold_positive_census':'census','b0_vs_qc_positive_ranking':'ranking','qc_extreme_tail':'tails',
        'pair_label_utility':'pair','threshold_survival':'pair.survival','matching_loss_decomposition':'pair.losses',
        'warm_removal_risk':'warm','position_ap_alignment':'pair.alignment','alternative_utility_diagnostic':'pair.alternatives',
        'user_incidence_choice_audit':'incidence','strict_cold_availability_proxy':'availability'}
    for name,key in components.items():
        def extract(value):
            for part in key.split('.'): value=value[part]
            return value
        write_json(reports/('p4_2d_'+name+'.json'),{'stage':'P4.2D','unit_contract':str(reports/'P4_2D_EXPERIMENT_CONTRACT.json'),
             'windows':{k:extract(v) for k,v in out['windows'].items()}})
    write_json(reports/'p4_2d_training_scaleup_feasibility.json',out['scaleup'])


def finish_scaleup(repo):
    """Resume only after the documented final SQL alias error; no window rerun."""
    repo=Path(repo).resolve(); branch_guard(repo); reports=repo/'reports/phase4'; root=repo/'artifacts/phase4'/RUN_ID
    m=read_json(reports/'P4_2D_metrics.json'); c=read_json(reports/'P4_2R3_EXPERIMENT_CONTRACT.json')
    assert m['status']=='engineering_failure' and len(m['windows'])==4
    assert 'ParserException' in m['traceback'] and 'count(*) rows' in m['traceback']
    failure=root/'FAILURE_attempt01.json'
    if failure.exists(): raise ValueError('resume already attempted')
    write_json(failure,m)
    start=time.perf_counter(); scale={}
    with duckdb.connect() as db:
        for cutoff,p in c['prepared_reuse'].items():
            guard_cutoff(cutoff); r=p['features']['qC']; w=p['features']['qW']
            check_identity(r); check_identity(w)
            values=db.execute('SELECT count(DISTINCT customer_id),count(*),sum(target),count(DISTINCT CASE WHEN target=1 THEN customer_id END) FROM read_parquet(?)',[r['path']]).fetchone()
            assert values[0]==r['users']==w['users'] and values[1]==r['row_count'] and values[2]==r['positive_rows']
            names=['users','candidate_rows','positive_rows','positive_users']
            scale[cutoff]={'exact_hash10':dict(zip(names,map(int,values))),
                'estimated_hash100':dict(zip(names,[int(x)*10 for x in values])),
                'exact_hash100':None,'candidate_regeneration':False,'label_end':r['label_end'],
                'estimate_warning':'10x deterministic user-cluster expansion; not exact support or independent-row CI'}
    m['scaleup']=scale; m['verdicts']=verdicts(m['windows']); m['status']='measured_pending_verification'
    m['initial_failure_preserved']=str(failure); m['initial_failure']=m.pop('traceback')
    m['resume_seconds']=time.perf_counter()-start; m['resumed_at_utc']=now()
    write_json(reports/'P4_2D_metrics.json',m); write_components(reports,m)
    print(json.dumps({'verdicts':m['verdicts'],'resume_seconds':m['resume_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['preregister','run','finish-scaleup']); p.add_argument('--repo',default='.')
    a=p.parse_args()
    if a.action=='preregister': print(preregister(a.repo)['created_at_utc'])
    elif a.action=='finish-scaleup': finish_scaleup(a.repo)
    else: run(a.repo)
