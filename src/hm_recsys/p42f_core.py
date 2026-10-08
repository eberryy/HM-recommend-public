"""Pure P4.2F edge features, deterministic sampling and historical residuals."""
import hashlib
import json
import numpy as np
import pandas as pd
from .p42f_contract import COLD,WARM,USER,PAIR,PERSONAL,GLOBAL,BASE,earlier
from .p42d_stats import single_ap_delta

def fold(user): return int.from_bytes(hashlib.sha256(str(user).encode()).digest()[:8],'big')%2

def neutral_keep(cutoff,user,item,slot):
    key=json.dumps([cutoff,str(user),str(item),int(slot)],ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%10000<200

def sample_edges(t,users,items,y):
    """No stochastic state; ALL nonzero targets retained, neutral weight50."""
    keep=y!=0
    for i,j in zip(*np.nonzero(~keep)):
        keep[i,j]=neutral_keep(t,users[i],items[i],j+1)
    weight=np.where(y[keep]==0,50.,1.)
    return keep,weight

def prior_residual(history,users,cutoff):
    h=history.loc[history.cutoff.isin(earlier(cutoff,sorted(history.cutoff.unique())))]
    g=h.groupby('customer_id').residual.agg(['count','mean']).reindex(users)
    n=g['count'].fillna(0).to_numpy(int)
    return n,n/(n+2)*g['mean'].fillna(0).to_numpy(float)

def batches(data,size=4096,labels=True):
    c,w,u=data['cold'],data['warm'],data['state']; xw=w[WARM].to_numpy(float).reshape(-1,12,len(WARM)); xu=u[USER].to_numpy(float)
    for start in range(0,len(c),size):
        cc=c.iloc[start:start+size]; idx=cc.user_index.to_numpy(int); n=len(cc)
        xc=np.repeat(cc[COLD].to_numpy(float),12,axis=0)
        ww=xw[idx].reshape(-1,len(WARM)); uu=np.repeat(xu[idx],12,axis=0)
        rank=cc.b0_rank.to_numpy(float)[:,None]; pct=cc.b0_rank_pct.to_numpy(float)[:,None]
        wp=w.warm_rank_pct.to_numpy().reshape(-1,12)[idx]
        vuln=1-w.warm_user_percentile.to_numpy().reshape(-1,12)[idx]
        vuln=np.where(np.isfinite(vuln),vuln,np.arange(12)[None,:]/11)
        slots=np.arange(1,13)[None,:]
        novelty=xu[idx,USER.index('novel_purchase_share_0_5')][:,None]
        rich=xu[idx,USER.index('user_past_purchase_count')][:,None]
        pair=np.stack([pct-wp,cc.b0_user_percentile.to_numpy()[:,None]-vuln,rank*slots],axis=-1).reshape(-1,len(PAIR))
        personal=np.stack([np.broadcast_to(cc.strict_cold_flag.to_numpy()[:,None]*novelty,(n,12)),
             np.broadcast_to(cc.sparse1_5_flag.to_numpy()[:,None]*novelty,(n,12)),
             np.broadcast_to(pct*novelty,(n,12)),slots*rich],axis=-1).reshape(-1,len(PERSONAL))
        x=np.concatenate([xc,ww,pair,uu,personal],axis=1).astype(np.float32)
        x[~np.isfinite(x)]=np.nan
        assert x.shape==(n*12,len(GLOBAL)) and GLOBAL[:len(BASE)]==BASE
        y=single_ap_delta(data['relevance'][idx],cc.target.to_numpy(),data['truth_count'][idx]) if labels else None
        yield start,cc,x,y

def empty_safe_max(scores,idx,nusers):
    out=np.zeros(nusers,float); np.maximum.at(out,idx,np.maximum(0,scores.max(axis=1)))
    return out
