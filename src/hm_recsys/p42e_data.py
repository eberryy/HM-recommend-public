"""Full historical roster, chunked frozen retrieval, and exact old10 replay."""
from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from .p41a_contract import read_json, write_json, guard_cutoff
from .p42_data import STATE_COLUMNS, normalize_cold_confidence, validate_lineages
from .p42r_data import (history_context, score_b0, shortlist, join_labels_last,
                       _parquet, _save_frame, _verify_declared, _find_declared)
from .p3 import _retrieve_top200
from .p37b_features import USER_STATE_COLUMNS, compute_user_state_batch
from .p42e_contract import now


def fold_for_user(user):
    return int.from_bytes(hashlib.sha256(str(user).encode('utf-8')).digest()[:8],'big')%2


def old_top200(repo,c,t,old_users):
    source=c['inputs'][t]; wanted=set(old_users)
    orig_users=pd.read_csv(source['p31_assets']['users']['path'],dtype=str).customer_id.to_numpy()
    z=np.load(source['p31_assets']['candidates']['path'],allow_pickle=False)
    keep=np.isin(orig_users[z['user_index']],list(wanted))
    parts=[pd.DataFrame(dict(customer_id=orig_users[z['user_index'][keep]],
                           catalog_row=z['catalog_row'][keep],rank=z['rank'][keep],
                           coarse_score=z['coarse_score'][keep]))]
    present=set(parts[0].customer_id)
    folder=repo/'artifacts/phase4/p4-2r-v1-population-numerical-repair/prepared'/t
    if wanted-present:
        gu=_parquet(folder/'generation-users.parquet').customer_id.to_numpy()
        nz=np.load(folder/'new-m4-top200-unlabelled.npz',allow_pickle=False)
        keep=np.isin(gu[nz['user_index']],list(wanted-present))
        parts.append(pd.DataFrame(dict(customer_id=gu[nz['user_index'][keep]],catalog_row=nz['catalog_row'][keep],
                                       rank=nz['rank'][keep],coarse_score=nz['coarse_score'][keep])))
    result=pd.concat(parts,ignore_index=True).sort_values(['customer_id','rank'],ignore_index=True)
    assert set(result.customer_id)==wanted and not result.duplicated(['customer_id','catalog_row']).any()
    assert result.groupby('customer_id').size().eq(200).all()
    return result


def parity(new,old,keys,score_fields=()):
    new=new.sort_values(keys,ignore_index=True); old=old.sort_values(keys,ignore_index=True)
    for key in keys:
        np.testing.assert_array_equal(new[key],old[key],err_msg='P4.2E identity/rank parity: '+key)
    errors={}
    for key in score_fields:
        errors[key]=float(np.max(np.abs(new[key].to_numpy(float)-old[key].to_numpy(float))))
        np.testing.assert_allclose(new[key],old[key],rtol=0,atol=1e-5,err_msg=key)
    return dict(rows=len(new),exact=True,score_max_abs=errors)


def generate(repo,c,t,root,resource_guard):
    guard_cutoff(t); assert t in c['historical_cutoffs']
    source=c['inputs'][t]; validate_lineages(source,t)
    assert source['b0_lineage']['available']
    folder=root/'prepared'/t
    if (folder/'COMPLETE.json').exists(): raise FileExistsError('completed cutoff must not be rerun')
    folder.mkdir(parents=True,exist_ok=True)
    def persist(frame,path):
        if path.exists():
            pd.testing.assert_frame_equal(frame.reset_index(drop=True),_parquet(path),check_dtype=False,check_exact=True)
        else: _save_frame(frame,path)
    started=time.perf_counter(); tx=c['transactions']['path']
    # Eligibility sees any next-week purchase, never item-level cold truth.
    with duckdb.connect() as con:
        con.execute('SET threads=4'); con.execute("SET memory_limit='4GB'")
        roster=con.execute('SELECT DISTINCT customer_id FROM read_parquet(?) WHERE t_dat>=CAST(? AS DATE) '
            'AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY ORDER BY customer_id',[tx,t,t]).fetchdf()
        roster['hash10']=con.execute('SELECT hash(customer_id)%1000000<100000 AS keep FROM roster').fetchdf()['keep'].to_numpy()
    catalog=pd.read_csv(c['catalog']['path'],dtype={'article_id':str}).sort_values('catalog_row')
    items=catalog.article_id.to_numpy()
    counts,hr,hd=history_context(tx,t,roster.customer_id.to_numpy(),items)
    active=(hr>=0).any(axis=1)
    roster['has_mapped_history']=active
    persist(roster,folder/'full-roster.parquet')
    h=roster.loc[active].reset_index(drop=True); users=h.customer_id.to_numpy()
    hr,hd=hr[active],hd[active]
    h['fold']=np.array([fold_for_user(u) for u in users],np.uint8)
    persist(h,folder/'eligible-users.parquet')
    # Original training table intentionally omits raw B0 score/rank. Replay
    # against the label-free selected-candidate evidence instead.
    old=_parquet(repo/'artifacts/phase4/p4-2r-v1-population-numerical-repair/prepared'/t/'cold50-selection-before-labels.parquet')
    np.testing.assert_array_equal(np.sort(old.customer_id.unique()),users[h.hash10])
    expected=old_top200(repo,c,t,users[h.hash10])
    p31=read_json(repo/'reports/phase3/P3_1_metrics.json')
    asset=p31['all_assets'][source['role']+':'+source['p31_label']]
    _verify_declared(asset['artifacts']['manifest'])
    pm=read_json(Path(asset['artifacts']['manifest']['path']))
    assert pm['cutoff']==t
    _verify_declared(pm['student_embedding'])
    ep=Path(pm['student_embedding']['path']); emb=np.load(ep,mmap_mode='r')
    p37=_find_declared(read_json(repo/'reports/phase3/P3_7B_OUTPUT_MANIFEST.json'),'P3_7B_metrics.json')
    _verify_declared(p37)
    art=read_json(Path(p37['path']))['authoritative_inputs']['articles']; _verify_declared(art)
    attrs=pd.read_csv(art['path'],dtype={'article_id':str},usecols=['article_id','product_type_no','garment_group_no'])
    aligned=catalog[['article_id']].merge(attrs,on='article_id',how='left',validate='one_to_one')
    product=aligned.product_type_no.fillna(-1).to_numpy(np.int32)
    garment=aligned.garment_group_no.fillna(-1).to_numpy(np.int32)
    state=np.empty((len(users),len(USER_STATE_COLUMNS)),np.float32)
    for start in range(0,len(users),2048):
        end=min(start+2048,len(users))
        state[start:end]=compute_user_state_batch(history_rows=hr[start:end],history_days=hd[start:end],
            history_mask=hr[start:end]>=0,m4_embeddings=emb,catalog_product_type=product)
    device=torch.device(c['generation']['device'])
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    audits=[]; shard_paths=[]
    for start in range(0,len(users),c['generation']['chunk_users']):
        resource_guard(); end=min(start+c['generation']['chunk_users'],len(users)); us=users[start:end]
        shard=folder/f'chunk-{start:07d}'
        if (shard/'top200-unlabelled.npz').exists():
            z=np.load(shard/'top200-unlabelled.npz',allow_pickle=False)
            arrays={k:z[k] for k in z.files if k not in ('b0_score','b0_rank')}
            scores,ranks=z['b0_score'],z['b0_rank']
            ra={'resumed_saved_unlabelled_chunk':True,'original_inference_seconds':'not_recorded_before_exception'}
        else:
            arrays,ra=_retrieve_top200(embeddings_path=ep,candidate_rows=np.flatnonzero(counts<=5).astype(np.int32),
                                      history_rows=hr[start:end],history_days=hd[start:end],truth={},device=device)
            assert not arrays['target'].any()
            scores,ranks=score_b0(arrays,hr[start:end],hd[start:end],state[start:end],counts,emb,
                                  product,garment,source['b0_lineage'],device)
            shard.mkdir()
            np.savez_compressed(shard/'top200-unlabelled.npz',**{k:v for k,v in arrays.items() if k!='target'},
                                b0_score=scores,b0_rank=ranks)
        selection=shortlist(arrays,scores,ranks,us,items)
        persist(pd.DataFrame({'customer_id':us}),shard/'users.parquet')
        persist(selection,shard/'cold50-unlabelled.parquet')
        subset=set(us)&set(old.customer_id)
        top=pd.DataFrame(dict(customer_id=us[arrays['user_index']],catalog_row=arrays['catalog_row'],
                               rank=arrays['rank'],coarse_score=arrays['coarse_score']))
        if subset:
            a=parity(top.loc[top.customer_id.isin(subset)],expected.loc[expected.customer_id.isin(subset)],
                     ['customer_id','rank','catalog_row'],['coarse_score'])
            b=parity(selection.loc[selection.customer_id.isin(subset)],old.loc[old.customer_id.isin(subset)],
                     ['customer_id','b0_rank','article_id'],['b0_score','m4_coarse_score'])
        else: a=b={'rows':0,'exact':True,'score_max_abs':{}}
        audits.append(dict(start=start,end=end,m4=a,cold50=b,retrieval=ra))
        write_json(shard/'PARITY.json',audits[-1]); shard_paths.append(shard)
        print(f'P4.2E {t}: frozen generation {end}/{len(users)}; old10 parity passed',flush=True)
        del arrays,scores,ranks,top,selection; gc.collect()
    # No historical item labels are joined until ALL chunks are fixed.
    write_json(folder/'CANDIDATES_FIXED_BEFORE_LABEL_JOIN.json',dict(at=now(),users=len(users),
        top200_rows=len(users)*200,cold50_rows=len(users)*50,future_item_labels_read=False))
    total_positive=0; positives=set()
    for shard in shard_paths:
        resource_guard()
        selection=_parquet(shard/'cold50-unlabelled.parquet')
        us=_parquet(shard/'users.parquet').customer_id.to_numpy()
        cold,_=join_labels_last(selection,tx,t,us)
        ix=np.searchsorted(users,us)
        sf=pd.DataFrame(state[ix,:6],columns=STATE_COLUMNS); sf.insert(0,'customer_id',us)
        cold=cold.merge(sf,on='customer_id',how='left',validate='many_to_one')
        cold['interaction_count_before_cutoff']=cold.article_id.map(dict(zip(items,counts))).astype(np.int64)
        cold['strict_cold_flag']=cold.interaction_count_before_cutoff.eq(0).astype(np.uint8)
        cold['sparse1_5_flag']=cold.interaction_count_before_cutoff.between(1,5).astype(np.uint8)
        cold=normalize_cold_confidence(cold)
        cold['target_cutoff']=t
        cold['fold']=cold.customer_id.map(dict(zip(us,h.iloc[ix].fold))).astype(np.uint8)
        cols=['target_cutoff','customer_id','article_id','target','fold',*c['feature_spec']['numeric'],*c['feature_spec']['binary']]
        _save_frame(cold[cols],shard/'training.parquet')
        total_positive+=int(cold.target.sum()); positives.update(cold.loc[cold.target.eq(1),'customer_id'])
    audit=dict(cutoff=t,full_roster_users=len(roster),eligible_users=len(users),cold50_rows=len(users)*50,
        top200_rows=len(users)*200,positive_rows=total_positive,positive_users=len(positives),
        old10_users=int(h.hash10.sum()),old10_user_exact=True,old10_m4_exact=True,old10_cold50_exact=True,
        parity_rows=sum(a['m4']['rows'] for a in audits),chunks=audits,
        generation_before_labels=True,seconds=time.perf_counter()-started,finished_at_utc=now())
    assert audit['parity_rows']==len(old.customer_id.unique())*200
    write_json(folder/'COMPLETE.json',audit)
    return audit
