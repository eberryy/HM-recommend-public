"""Exact final-list evaluation of only W0, G and H, complete outer denominator."""
import numpy as np
import pandas as pd
from .metrics import apk
from .p42_evaluate import _segment_truth,_segment_metrics
from .p42_matching import exact_matching,apply_admissions
from .p42d_stats import discrimination,correlations,summary
from .p42f_contract import BASE
from .p42f_core import batches
from .p42f_train import thresholds
from .p42f_data import save_frame
from .p41a_contract import write_json

VARIANTS=['G_expected_deltaAP','H_hierarchical_threshold']

def user_summary(mask,delta,count,ins,rem):
    n=int(mask.sum()); admits=int((count[mask]>0).sum()); reps=int(count[mask].sum())
    return dict(users=n,users_with_admission=admits,coverage=admits/n if n else None,
        replacements=reps,inserted_cold_positives=int(ins[mask].sum()),removed_warm_positives=int(rem[mask].sum()),
        net_positives=int((ins[mask]-rem[mask]).sum()),map_delta=float(delta[mask].mean()) if n else None,
        cold_positive_insertion_rate=float(ins[mask].sum()/reps) if reps else None,
        warm_positive_removal_rate=float(rem[mask].sum()/reps) if reps else None)

def evaluate(data,G,S,threshold,history,folder,guard):
    folder.mkdir(parents=True,exist_ok=False); cold=data['cold']; n=len(data['users'])
    fixed,hn,b=thresholds(threshold,history,data['state'],data['cutoff']); tau=fixed+b
    scores={v:np.empty((len(cold),12),float) for v in VARIANTS}; target=np.empty((len(cold),12),float)
    for start,cc,x,y in batches(data):
        guard(); end=start+len(cc); idx=cc.user_index.to_numpy(int)
        scores[VARIANTS[0]][start:end]=G.predict(x).reshape(-1,12)
        scores[VARIANTS[1]][start:end]=S.predict(x[:,:len(BASE)]).reshape(-1,12)-tau[idx,None]
        target[start:end]=y
    np.save(folder/'single_delta.npy',target)
    segments=_segment_truth(data); base_segments=_segment_metrics(data,data['warm_lists'],segments)
    out={'W0':dict(map12=data['baseline_map'],delta_vs_w0=0.,segments=base_segments)}
    # Quantile definitions learned exclusively from the historical training population.
    novelty_cuts=np.quantile(history.novel_purchase_share_0_5.fillna(0),[.25,.5,.75])
    richness_cuts=np.quantile(history.user_past_purchase_count,[1/3,2/3])
    nq=np.searchsorted(novelty_cuts,data['state'].novel_purchase_share_0_5.fillna(0),side='left')
    rq=np.searchsorted(richness_cuts,data['state'].user_past_purchase_count,side='left')
    distributions={'tau_fixed':summary(fixed),'b_u':summary(b),'tau_u':summary(tau),
        'support':{label:dict(users=int(mask.sum()),mean_abs_b=float(np.abs(b[mask]).mean()) if mask.any() else None)
          for label,mask in [('0',hn==0),('1',hn==1),('2',hn==2),('3+',hn>=3)]},
        'n0_b_exact_zero':bool((b[hn==0]==0).all()),'novelty_cutpoints':novelty_cuts.tolist(),'richness_cutpoints':richness_cuts.tolist()}
    save_frame(pd.DataFrame(dict(customer_id=data['users'],tau_fixed=fixed,n_u=hn,b_u=b,tau_u=tau)),folder/'thresholds.parquet')
    groups=list(cold.groupby('user_index',sort=False).indices.items())
    for variant in VARIANTS:
        utility=scores[variant]; assert np.isfinite(utility).all(); eligible=utility>0
        y=target.ravel(); u=utility.ravel(); nonneutral=y!=0
        action=dict(correlation=correlations(u,y),beneficial_vs_harmful=discrimination(u[nonneutral],y[nonneutral]>0),
            beneficial_vs_all_nonbeneficial=discrimination(u,y>0),edges=len(u),
            eligible_edges=int(eligible.sum()),eligible_beneficial=int(((target>0)&eligible).sum()),
            eligible_harmful=int(((target<0)&eligible).sum()),eligible_neutral=int(((target==0)&eligible).sum()),
            eligible_beneficial_ppv=float(((target>0)&eligible).sum()/eligible.sum()) if eligible.any() else None)
        lists=data['warm_lists'].copy(); counts=np.zeros(n,int); ins=np.zeros(n,int); rem=np.zeros(n,int); single_sum=np.zeros(n)
        strict_ins=np.zeros(n,int); sparse_ins=np.zeros(n,int); coldonly=np.zeros(n,int); rows=[]
        for ui,ix in groups:
            ix=np.asarray(ix); cc=cold.iloc[ix]; matches=exact_matching(utility[ix],0.)
            lists[ui]=apply_admissions(list(lists[ui]),cc.article_id.tolist(),matches)
            counts[ui]=len(matches)
            for ci,wi in matches:
                item=cc.iloc[ci]; ip=int(item.target); rp=int(data['relevance'][ui,wi])
                ins[ui]+=ip; rem[ui]+=rp; strict_ins[ui]+=ip*int(item.strict_cold_flag); sparse_ins[ui]+=ip*int(item.sparse1_5_flag)
                coldonly[ui]+=ip*int(item.cold_only); single_sum[ui]+=target[ix[ci],wi]
                rows.append(dict(customer_id=data['users'][ui],cold_article_id=item.article_id,warm_article_id=data['warm_lists'][ui,wi],
                    warm_slot=wi+1,cold_row=int(ix[ci]),utility=float(utility[ix[ci],wi]),inserted_positive=ip,removed_positive=rp,
                    single_delta=float(target[ix[ci],wi]),strict_cold=int(item.strict_cold_flag),sparse1_5=int(item.sparse1_5_flag),cold_only=int(item.cold_only)))
        ap=np.array([apk(list(data['truthsets'][user]),list(pred)) for user,pred in zip(data['users'],lists)])
        delta=ap-data['baseline_ap']; allmask=np.ones(n,bool)
        admission=user_summary(allmask,delta,counts,ins,rem)
        admission.update(mean_per_admitted=float(counts[counts>0].mean()) if (counts>0).any() else 0.,
            mean_per_all_users=float(counts.mean()),max_replacements=int(counts.max()),strict_positive_inserted=int(strict_ins.sum()),
            sparse_positive_inserted=int(sparse_ins.sum()),cold_only_positive_inserted=int(coldonly.sum()))
        buckets={}
        for label,mask in [('0',counts==0),('1',counts==1),('2',counts==2),('3',counts==3),('4+',counts>=4)]:
            buckets[label]=user_summary(mask,delta,counts,ins,rem)
            buckets[label]['mean_exact_minus_single_sum']=float((delta-single_sum)[mask].mean()) if mask.any() else None
        mechanisms={'novelty':{f'Q{k+1}':user_summary(nq==k,delta,counts,ins,rem) for k in range(4)},
                    'richness':{name:user_summary(rq==k,delta,counts,ins,rem) for k,name in enumerate(['low','medium','high'])},
                    'b_sign':{name:user_summary(mask,delta,counts,ins,rem) for name,mask in [('positive',b>1e-12),('near_zero',np.abs(b)<=1e-12),('negative',b<-1e-12)]}}
        out[variant]=dict(map12=float(ap.mean()),delta_vs_w0=float(ap.mean()-data['baseline_map']),
                          segments=_segment_metrics(data,lists,segments,base_segments),action=action,admission=admission,
                          buckets=buckets,mechanisms=mechanisms)
        np.save(folder/(variant+'-utility.npy'),utility)
        save_frame(pd.DataFrame(rows,columns=['customer_id','cold_article_id','warm_article_id','warm_slot','cold_row','utility','inserted_positive','removed_positive','single_delta','strict_cold','sparse1_5','cold_only']),folder/(variant+'-executed.parquet'))
        save_frame(pd.DataFrame({'customer_id':np.repeat(data['users'],12),'rank':np.tile(np.arange(1,13),n),'article_id':lists.ravel()}),folder/(variant+'-lists.parquet'))
        save_frame(pd.DataFrame(dict(customer_id=data['users'],ap=ap,baseline_ap=data['baseline_ap'],delta=delta,admissions=counts,
                                    inserted=ins,removed=rem,single_sum=single_sum)),folder/(variant+'-users.parquet'))
    write_json(folder/'EVALUATION.json',dict(variants=out,hierarchical=distributions))
    return dict(variants=out,hierarchical=distributions)

def decision(windows):
    gates={}
    for v in VARIANTS:
        rows=[r['variants'][v] for r in windows.values()]
        overall=np.array([r['delta_vs_w0'] for r in rows]); warm=np.array([r['segments']['warm_21_plus']['delta_vs_w0'] for r in rows]); cold=np.array([r['segments']['all_cold_sparse']['delta_vs_w0'] for r in rows])
        tests=dict(overall_mean=bool(overall.mean()>0),overall_nondegrade=bool((overall>=0).sum()>=3),overall_worst=bool(overall.min()>=-.0002),
                   warm_mean=bool(warm.mean()>=-.0001),warm_windows=bool((warm>=-.0002).sum()>=3),
                   cold_mean=bool(cold.mean()>0),cold_windows=bool((cold>=0).sum()>=3),
                   cold_only_windows=bool(sum(r['admission']['cold_only_positive_inserted']>0 for r in rows)>=2),
                   efficiency=bool(sum(r['admission']['inserted_cold_positives'] for r in rows)>sum(r['admission']['removed_warm_positives'] for r in rows)))
        gates[v]=dict(pass_all=all(tests.values()),tests=tests,mean_map=float(np.mean([r['map12'] for r in rows])),
                      mean_delta=float(overall.mean()),mean_cold_delta=float(cold.mean()),mean_warm_delta=float(warm.mean()))
    winners=[v for v in VARIANTS if gates[v]['pass_all']]
    selected='W0'
    if len(winners)==1:
        selected=winners[0]; machine='promote_global_expected_deltaAP' if selected==VARIANTS[0] else 'promote_hierarchical_user_threshold'
    elif len(winners)==2:
        g,h=[gates[v] for v in VARIANTS]
        selected=VARIANTS[0] if (g['mean_map']>h['mean_map']+1e-12 or abs(g['mean_map']-h['mean_map'])<=1e-12 and g['mean_cold_delta']>=h['mean_cold_delta']) else VARIANTS[1]
        machine='both_safe_global_selected' if selected==VARIANTS[0] else 'both_safe_hierarchical_selected'
    elif gates[VARIANTS[0]]['mean_delta']>0: machine='global_signal_but_gate_failure'
    elif gates[VARIANTS[1]]['mean_delta']>0: machine='hierarchical_signal_but_gate_failure'
    else: machine='relative_utility_not_sufficient'
    return dict(machine_decision=machine,selected=selected,gates=gates)
