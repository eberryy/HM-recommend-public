"""Risk-only matching; classification-only scores never construct lists."""
import numpy as np
import pandas as pd
from .p41a_contract import read_json,write_json
from .p42f_core import batches
from .p42f_evaluate import user_summary
from .p42f_data import save_frame
from .p42_evaluate import _segment_truth,_segment_metrics
from .p42_matching import exact_matching,apply_admissions
from .metrics import apk
from .p42g_core import utilities,action,tail,survival,calibration

def evaluate(repo,data,models,folder,ffolder,guard):
    folder.mkdir(parents=True,exist_ok=False);cold=data['cold'];n=len(data['users']);nc=len(cold)
    # Float64 memory-mapped outputs preserve exact gate signs with bounded RSS.
    shape=(nc,12)
    outputs={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype='float64',shape=(nc,12,3) if k=='probability' else shape)
             for k in ['probability','m_B_raw','m_H_raw','m_B','m_H','U_R','U_C']}
    target=np.load(ffolder/'single_delta.npy',mmap_mode='r')
    for start,cc,x,y in batches(data):
        guard();end=start+len(cc);p=models[0].predict(x,num_threads=4)
        mbraw=models[1].predict(x,num_threads=4);mhraw=models[2].predict(x,num_threads=4)
        ur,uc,mb,mh=utilities(p,mbraw,mhraw)
        np.testing.assert_array_equal(y,target[start:end])
        for k,value in [('probability',p.reshape(-1,12,3)),('m_B_raw',mbraw.reshape(-1,12)),('m_H_raw',mhraw.reshape(-1,12)),
                        ('m_B',mb.reshape(-1,12)),('m_H',mh.reshape(-1,12)),('U_R',ur.reshape(-1,12)),('U_C',uc.reshape(-1,12))]:outputs[k][start:end]=value
    for a in outputs.values():a.flush()
    guard();ur=outputs['U_R'];uc=outputs['U_C'];p=outputs['probability'].reshape(-1,3)
    calibration_audit=calibration(p,target.ravel())
    action_r=action(ur,target)
    # Only prescribed action statistics for U_C, with no gate or policy variant.
    action_c=action(uc,target);action_c.pop('beneficial_vs_all_nonbeneficial')
    g=np.load(ffolder/'G_expected_deltaAP-utility.npy',mmap_mode='r')
    fresult=read_json(ffolder/'EVALUATION.json')
    old=read_json(repo/'reports/phase4/P4_2D_metrics.json')['windows'][folder.name]['pair']
    assert old['comparison']['secondary']['rows']==target.size
    assert old['comparison']['primary']['positives']==int((target>0).sum())
    magnitude={}
    for name,mask,key in [('benefit',target>0,'m_B'),('harm',target<0,'m_H')]:
        pred=outputs[key][mask];truth=np.abs(target[mask]);raw=outputs[key+'_raw']
        magnitude[name]=dict(edges=int(mask.sum()),observed_mean=float(truth.mean()),predicted_mean=float(pred.mean()),
            conditional_MAE=float(np.abs(pred-truth).mean()),conditional_RMSE=float(np.sqrt(np.square(pred-truth).mean())),
            clipped_low_edges=int((raw<0).sum()),clipped_high_edges=int((raw>1).sum()))
    lists=data['warm_lists'].copy();counts=np.zeros(n,int);ins=np.zeros(n,int);rem=np.zeros(n,int);sums=np.zeros(n)
    strict=np.zeros(n,int);sparse=np.zeros(n,int);coldonly=np.zeros(n,int);rows=[]
    for ui,ix in cold.groupby('user_index',sort=False).indices.items():
        ix=np.asarray(ix);cc=cold.iloc[ix];matches=exact_matching(ur[ix],0.)
        lists[ui]=apply_admissions(list(lists[ui]),cc.article_id.tolist(),matches);counts[ui]=len(matches)
        for ci,j in matches:
            cr=int(ix[ci]);item=cc.iloc[ci];ip=int(item.target);rp=int(data['relevance'][ui,j]);v=float(target[cr,j])
            ins[ui]+=ip;rem[ui]+=rp;sums[ui]+=v;strict[ui]+=ip*int(item.strict_cold_flag);sparse[ui]+=ip*int(item.sparse1_5_flag);coldonly[ui]+=ip*int(item.cold_only)
            rows.append(dict(customer_id=data['users'][ui],cold_row=cr,cold_article_id=item.article_id,
                warm_article_id=data['warm_lists'][ui,j],warm_slot=j+1,utility=float(ur[cr,j]),single_delta=v,
                inserted_positive=ip,removed_positive=rp,strict_cold=int(item.strict_cold_flag),sparse1_5=int(item.sparse1_5_flag)))
    ap=np.array([apk(list(data['truthsets'][u]),list(pred)) for u,pred in zip(data['users'],lists)])
    delta=ap-data['baseline_ap'];allmask=np.ones(n,bool)
    admission=user_summary(allmask,delta,counts,ins,rem)
    admission.update(mean_per_admitted=float(counts[counts>0].mean()) if (counts>0).any() else 0.,mean_per_all_users=float(counts.mean()),
        max_replacements=int(counts.max()),strict_positive_inserted=int(strict.sum()),sparse_positive_inserted=int(sparse.sum()),cold_only_positive_inserted=int(coldonly.sum()))
    buckets={}
    for name,mask in [('0',counts==0),('1',counts==1),('2',counts==2),('3',counts==3),('4+',counts>=4)]:
        buckets[name]=user_summary(mask,delta,counts,ins,rem)
        buckets[name]['mean_exact_minus_single_sum']=float((delta-sums)[mask].mean()) if mask.any() else None
    cuts=fresult['hierarchical'];nq=np.searchsorted(cuts['novelty_cutpoints'],data['state'].novel_purchase_share_0_5.fillna(0),side='left')
    rq=np.searchsorted(cuts['richness_cutpoints'],data['state'].user_past_purchase_count,side='left')
    mechanisms=dict(novelty={f'Q{k+1}':user_summary(nq==k,delta,counts,ins,rem) for k in range(4)},
                    richness={name:user_summary(rq==k,delta,counts,ins,rem) for k,name in enumerate(['low','medium','high'])})
    segments=_segment_truth(data);base=_segment_metrics(data,data['warm_lists'],segments)
    result=dict(W0=dict(map12=data['baseline_map'],segments=base),
        R=dict(map12=float(ap.mean()),delta_vs_w0=float(ap.mean()-data['baseline_map']),segments=_segment_metrics(data,lists,segments,base),
               action=action_r,admission=admission,buckets=buckets,mechanisms=mechanisms),
        classifier=calibration_audit,magnitude=magnitude,classification_only_diagnostic=action_c,
        references=dict(G=fresult['variants']['G_expected_deltaAP']['action'],old_U=dict(beneficial_vs_harmful=old['comparison']['primary'],
            beneficial_vs_all_nonbeneficial=old['comparison']['secondary'],correlation=old['alignment']['overall'])),
        gate=dict(R=tail(ur,target),G=tail(g,target)),
        survival=dict(R=survival(cold,ur,target),G=survival(cold,g,target),old_U=old['survival']['A_tau0']),
        cutpoints=dict(novelty=cuts['novelty_cutpoints'],richness=cuts['richness_cutpoints']))
    save_frame(pd.DataFrame(rows,columns=['customer_id','cold_row','cold_article_id','warm_article_id','warm_slot','utility','single_delta','inserted_positive','removed_positive','strict_cold','sparse1_5']),folder/'executed.parquet')
    save_frame(pd.DataFrame(dict(customer_id=data['users'],ap=ap,baseline_ap=data['baseline_ap'],delta=delta,admissions=counts,
               inserted=ins,removed=rem,single_sum=sums,novelty_group=nq,richness_group=rq)),folder/'users.parquet')
    save_frame(pd.DataFrame(dict(customer_id=np.repeat(data['users'],12),rank=np.tile(np.arange(1,13),n),article_id=lists.ravel())),folder/'lists.parquet')
    write_json(folder/'EVALUATION.json',result)
    return result
