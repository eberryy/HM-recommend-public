"""Frozen temporal checkpoint inventory and label-free Warm rebuild pilot.

Does not train models, select by MAP, or open integration/final-week labels.
"""
from pathlib import Path
from datetime import date, timedelta
import argparse
import hashlib
import json
import time
import shutil

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .p42f_contract import DATES, WINDOWS
from .warm_v2_engine import prepare
from .mind_warm_final_confirmation import wv3_matrix, choose_wv3_swaps, WV3_FEATURES, EXTRA_FEATURES, PAIR_TRANSFORM
from .warm_v3_clean_model_replay import fused_candidate_scores

ROOT = Path(__file__).resolve().parents[2]
WROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/final/history-rebuild-v1'
CONTRACT = ROOT / 'reports/final/FINAL_HISTORY_REBUILD_CONTRACT.json'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def end(t):
    return (date.fromisoformat(t) + timedelta(days=7)).isoformat()


def register():
    if CONTRACT.exists():
        raise FileExistsError('Frozen registration already exists')
    candidates = []
    for root, prefix in [(ROOT/'artifacts/warm_v2', 'WV2'),
                         (WROOT/'artifacts/warm_v2/fresh_robustness', 'FRESH')]:
        for p in sorted((root/f'{prefix}-000').glob('*/outer_metrics.json')):
            pair = []
            for arm in ['000', '501']:
                folder = root/f'{prefix}-{arm}'/p.parent.name
                outer, inner = read(folder/'outer_metrics.json'), read(folder/'inner_metrics.json')
                if outer['rounds'] != inner['best_iteration']:
                    raise ValueError('Round selection provenance differs')
                pair.append(dict(folder=str(folder), role=arm, features=outer['features'],
                    parameters=outer['parameters'], rounds=outer['rounds'], model=outer['model'],
                    training_cutoffs=outer['train_cutoffs'], selection_cutoff=inner['cutoff'],
                    latest_label_end=max([end(t) for t in outer['train_cutoffs']]+[end(inner['cutoff'])])))
            candidates.append(dict(origin_cutoff=read(p)['cutoff'], models=pair,
                latest_label_end=max(x['latest_label_end'] for x in pair)))
    mappings = {}
    for t in sorted(set(DATES+list(WINDOWS.values()))):
        allowed = [c for c in candidates if c['latest_label_end'] < t and c['origin_cutoff'] <= t]
        if not allowed:
            raise ValueError('No historical checkpoint available for '+t)
        chosen = max(allowed, key=lambda c: (c['origin_cutoff'],c['latest_label_end']))
        base = ROOT/f'artifacts/m3_3/seasonal-feature-cache-v1/{t}/features.parquet'
        bp = ROOT/f'artifacts/warm_v2/bpr-match-v1/{t}/features.parquet'
        bm = read(bp.with_suffix('.json'))
        if bm['cutoff'] != t or not bm['pit_safe'] or bm['future_labels_used'] or bm['latest_history_date'] >= t:
            raise ValueError('Unsafe BPR history')
        mappings[t] = dict(**chosen, base=str(base), bpr=str(bp), base_exists=base.is_file(),
            bpr_exists=bp.is_file(), bpr_metadata=str(bp.with_suffix('.json')),
            action='replay_existing_outer_checkpoint' if chosen['origin_cutoff']==t else 'rebuild_prediction_using_latest_safe_checkpoint')
    for arm in ['WV3-661','WV3-680']:
        audit = read(WROOT/f'artifacts/warm_v3/{arm}/TRAINING_AUDIT.json')
        assert audit['latest_label_end_exclusive'] < min(DATES)
    result = dict(status='preregistered_before_pilot', historical_dates=DATES, development_dates=WINDOWS,
        checkpoint_rule='latest origin_cutoff <= target with ALL training and early-stopping label ends < target; never compare target MAP',
        mappings=mappings, model_training_allowed=False,
        input_scope='full existing hash10 candidate feature rows; do not reuse inner covered-positive-only row filtering',
        labels='explicit projection excludes target and truth_count; no transaction label query in pilot',
        unchanged=['WV3-661','WV3-680','RRF60 equal','max2 disjoint swaps 13-50 into8-12','Cold A config and policy'],
        pilot_cutoff='2019-12-25', pilot_limits=dict(seconds=900, threads=4, duckdb_memory='2GB', min_disk_gib=15),
        stop='provenance, feature order, population, uniqueness or temporal failure; preserve partial outputs; no training or policy rescue',
        final_week_integration='not_run')
    write(CONTRACT,result)
    return result


def verify_model(folder):
    meta=read(folder/'outer_metrics.json'); p=folder/'outer_model.txt'
    # Compare mutable historical checkpoints against their recorded model receipts.
    if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['model']['sha256']:
        raise ValueError('Historical model integrity differs: '+str(p))
    model=lgb.Booster(model_file=str(p))
    if model.feature_name()!=meta['features']:
        raise ValueError('Model feature identity differs')
    maps=read(folder/'outer_category_maps.json')
    maps={k:{int(a):int(b) for a,b in v.items()} for k,v in maps.items()}
    return model,maps


def pilot(t, continuation=False):
    contract=read(CONTRACT)
    if t!=contract['pilot_cutoff'] and not continuation:
        raise ValueError('Only the registered cost pilot is authorized here')
    if continuation:
        approved=read(ROOT/'reports/final/FINAL_HISTORY_REBUILD_CONTINUATION.json')
        if t not in approved['cutoffs']: raise ValueError('Unregistered continuation cutoff')
    m=contract['mappings'][t]; dest=OUT/t
    if dest.exists(): raise FileExistsError('Preserve prior pilot')
    dest.mkdir(parents=True); started=time.perf_counter()
    bm=read(m['bpr_metadata'])
    for path,expected in [(m['base'],bm['source_sha256']), (m['bpr'],bm['artifact']['sha256'])]:
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
        if h.hexdigest()!=expected: raise ValueError('Historical feature differs from prior BPR receipt')
    def guard():
        if time.perf_counter()-started>900 or shutil.disk_usage(ROOT).free<15*2**30:
            raise RuntimeError('Pilot cost/resource boundary')
    db=duckdb.connect(str(dest/'stage.duckdb'))
    db.execute("SET threads=4"); db.execute("SET memory_limit='2GB'")
    source=duckdb.connect(); source.execute("SET threads=4"); source.execute("SET memory_limit='2GB'")
    models=[verify_model(Path(x['folder'])) for x in m['models']]
    base_names=models[0][0].feature_name()
    columns=list(dict.fromkeys(['customer_id','article_id',*base_names]))
    query='SELECT '+','.join('b.'+n for n in columns)+',p.wv2_bpr_user_item_score,p.wv2_bpr_unavailable FROM read_parquet(?) b LEFT JOIN read_parquet(?) p USING(customer_id,article_id)'
    cursor=source.execute(query,[m['base'],m['bpr']]); rows=0
    while True:
        f=cursor.fetch_df_chunk(16)
        if f.empty: break
        guard()
        if f.wv2_bpr_unavailable.isna().any(): raise ValueError('Missing BPR candidate keys')
        out=f[['customer_id','article_id','candidate_rank','user_history_events_12w']].copy()
        for (model,maps),name in zip(models,['score_base','score_bpr']):
            out[name]=model.predict(prepare(f,model.feature_name(),maps),num_threads=4)
        out['latent']=f.wv2_bpr_user_item_score; out['missing']=f.wv2_bpr_unavailable
        db.register('batch',out)
        if rows==0: db.execute('CREATE TABLE predictions AS SELECT * FROM batch WHERE FALSE')
        db.execute('INSERT INTO predictions SELECT * FROM batch'); db.unregister('batch'); rows+=len(f)
    db.execute('''CREATE TABLE ranked AS SELECT *,
      row_number() OVER(PARTITION BY customer_id ORDER BY score_base DESC,candidate_rank,article_id)::INTEGER r0,
      row_number() OVER(PARTITION BY customer_id ORDER BY score_bpr DESC,candidate_rank,article_id)::INTEGER r1,
      count(*) OVER(PARTITION BY customer_id)::INTEGER candidate_count,
      avg(latent) OVER(PARTITION BY customer_id)::FLOAT latent_mean,
      stddev_samp(latent) OVER(PARTITION BY customer_id)::FLOAT latent_std FROM predictions''')
    db.execute('''CREATE TABLE ranks AS SELECT *,row_number() OVER(PARTITION BY customer_id
      ORDER BY rrf_score DESC,candidate_rank,article_id)::INTEGER rf FROM
      (SELECT *,1.0/(60+r0)+1.0/(60+r1) rrf_score FROM ranked)''')
    db.execute('''CREATE TABLE top50 AS SELECT * FROM (SELECT *,CASE WHEN user_history_events_12w=0
      THEN candidate_rank ELSE rf END::INTEGER ap_rf FROM ranks) WHERE ap_rf<=50''')
    stats=db.execute('SELECT count(*),count(DISTINCT customer_id) FROM top50').fetchone()
    needed=list(dict.fromkeys(['user_unique_items_12w','user_days_since_last_purchase',*PAIR_TRANSFORM,*EXTRA_FEATURES]))
    query='SELECT r.*,'+','.join('f.'+n for n in needed)+' FROM top50 r JOIN read_parquet(?) f USING(customer_id,article_id) WHERE r.user_history_events_12w>0'
    f=db.execute(query,[m['base']]).fetchdf(); guard()
    boosters=[]
    for name in ['WV3-661','WV3-680']:
        folder=WROOT/f'artifacts/warm_v3/{name}'; meta=read(folder/'MODEL.json')
        if hashlib.sha256((folder/'MODEL.txt').read_bytes()).hexdigest()!=meta['model']['sha256']:
            raise ValueError('WV3 model differs from prior receipt')
        booster=lgb.Booster(model_file=str(folder/'MODEL.txt')); assert booster.feature_name()==WV3_FEATURES
        boosters.append(booster)
    matrix=wv3_matrix(f)
    scores=fused_candidate_scores(f,*[b.predict(matrix,num_threads=4) for b in boosters])
    scores=scores.merge(f[['customer_id','article_id','rf']],validate='one_to_one')
    swaps=choose_wv3_swaps(scores)
    top=db.execute('SELECT customer_id,article_id,ap_rf FROM top50 WHERE ap_rf<=12 ORDER BY customer_id,ap_rf').fetchdf()
    changes=swaps[['customer_id','victim_rank','challenger_article_id']].rename(columns={'victim_rank':'ap_rf'})
    top=top.merge(changes,on=['customer_id','ap_rf'],how='left',validate='one_to_one')
    assert top.challenger_article_id.notna().sum()==len(swaps)
    top['article_id']=top.challenger_article_id.fillna(top.article_id)
    top=top.drop(columns='challenger_article_id')
    assert not top.duplicated(['customer_id','article_id']).any()
    assert top.groupby('customer_id').size().eq(12).all()
    for frame,name in [(top,'warm-top12'),(swaps,'swaps')]:
        db.register('output',frame); db.execute('COPY output TO ? (FORMAT PARQUET)',[str(dest/(name+'.parquet'))]); db.unregister('output')
    result=dict(status='completed_label_free_pilot',cutoff=t,seconds=time.perf_counter()-started,
        scored_rows=rows,users=stats[1],top50_rows=stats[0],swaps=len(swaps),unique_top12=True,
        model_fits=0,labels_read=False,cold_training_started=False,final_week='not_run',checkpoint=m,
        bytes=sum(p.stat().st_size for p in dest.glob('*') if p.is_file()))
    source.close();db.close();write(dest/'RESULT.json',result)
    if not continuation: write(ROOT/'reports/final/FINAL_HISTORY_REBUILD_PILOT.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','pilot','rebuild']);p.add_argument('--cutoff');a=p.parse_args()
    result=register() if a.action=='register' else pilot(a.cutoff or '2019-12-25',a.action=='rebuild')
    print(json.dumps({k:result[k] for k in ['status','cutoff','seconds','scored_rows','users','swaps'] if k in result},ensure_ascii=False))
