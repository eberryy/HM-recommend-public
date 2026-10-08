"""Frozen 2019 robustness backtest. Only output routing/temporal chains are new."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import time
import traceback
import shutil
import numpy as np
import lightgbm as lgb

from . import warm_v2_engine as we
from . import warm_v2_bpr as bpr
from . import warm_v2_recent_data as builder
from .warm_v2_contract import REPORT,read,write,guard,git,now,evidence_id
from .warm_v2_fresh_preflight import ROOT,CHAINS
from .warm_v2_rank_fusion import rank_sql,final_rank_sql

ART=ROOT/'artifacts/warm_v2/fresh_robustness'
CONTRACT=ROOT/'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS_CONTRACT.json'
SYSTEMS=['FRESH-000','FRESH-501']


def assert_fresh():
    assert Path.cwd().resolve()==ROOT and git('branch','--show-current')=='warm-v2-fresh-robustness'
    assert git('merge-base','5fbe34b','HEAD')==git('rev-parse','5fbe34b')


def allowed(cutoff):
    guard(cutoff)
    if cutoff not in {v for chain in CHAINS for v in chain}:raise ValueError('not an authorized fresh cutoff')


def setup():
    assert_fresh();c=read(CONTRACT)
    assert c['no_search'] and c['preflight']['history_pass'] and c['preflight']['freshness_pass']
    assert c['bpr_params']==bpr.PARAMS
    import implicit,torch
    versions={'implicit':implicit.__version__,'lightgbm':lgb.__version__,'torch':torch.__version__}
    assert versions==c['versions'], 'shared conda environment drift; preserve artifacts and review, never silently change runtime'
    # Process-local routing ONLY. Imported original algorithm functions stay unchanged.
    we.ARTIFACT=ART
    bpr.ROOT=ART/'bpr';bpr.assert_branch=assert_fresh
    builder.ROOT=ART/'candidates';builder.assert_branch=assert_fresh
    assert builder.RAW.resolve()==Path(__file__).resolve().parents[2] / 'data/raw'
    return c


class FreshEngine(we.Engine):
    def __init__(self,c):
        super().__init__(c)
        original=deepcopy(self.history['development'])
        template=next(iter(original.values()))
        for v in original.values():
            assert v['inner_models']['anchor']['parameters']==c['lightgbm_inner_params']
            assert v['outer_models']['anchor']['parameters']==c['lightgbm_outer_params']
        self.history['development']={w:deepcopy(template) for w in c['rolling_protocol']}
        self.history['feature_cache']={}
        assert len(self.features)==84

    def base_path(self,cutoff):
        allowed(cutoff)
        if cutoff not in self.history['feature_cache']:
            if shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('disk below 10GiB; preserve all evidence')
            r=builder.build(cutoff)
            self.history['feature_cache'][cutoff]=r['target']
            if r['wall_seconds']>1200:raise RuntimeError('candidate build exceeded 20min; engineering cost review required')
        return Path(self.history['feature_cache'][cutoff]['artifact']['path'])


def inner_diagnostic(w,e):
    p=e.contract['rolling_protocol'][w];fp,stats=e.cached_data(p['inner_validation'],'inner')
    f=we.load_parquet(fp);out=f[['customer_id','article_id','candidate_rank','target','truth_count']].copy()
    from .warm_v2_features import attach_features
    for trial,label in zip(SYSTEMS,['base','bpr']):
        model=lgb.Booster(model_file=str(ART/trial/w/'inner_model.txt'))
        maps=read(ART/trial/w/'inner_category_maps.json');maps={k:{int(a):int(b) for a,b in v.items()} for k,v in maps.items()}
        ff=attach_features(f,p['inner_validation'],['bpr_match'],e) if label=='bpr' else f
        out['score_'+label]=model.predict(we.prepare(ff,model.feature_name(),maps),num_threads=8)
    path=ART/'diagnostics'/w/'inner_ranks.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    with we.connection() as con:
        con.register('f',out)
        con.execute(f'CREATE TEMP TABLE ranks AS SELECT *,{rank_sql("score_base")} base_rank,{rank_sql("score_bpr")} bpr_rank FROM f')
        v=con.execute('''SELECT count(*) FILTER(WHERE target=1 AND base_rank<=12 AND bpr_rank>12),
            count(*) FILTER(WHERE target=1 AND bpr_rank<=12 AND base_rank>12),
            count(*) FILTER(WHERE target=1 AND bpr_rank<=12 AND base_rank<=12),sum(target) FROM ranks''').fetchone()
        con.execute(f'COPY ranks TO {we.literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    r={'cutoff':p['inner_validation'],'base_only_correct_top12_positive_pairs':v[0],
        'bpr_only_correct_top12_positive_pairs':v[1],'shared_correct_top12_positive_pairs':v[2],
        'positive_candidate_pairs':v[3],'groups':stats['groups'],'diagnostic_only_no_selection':True,
        'population':'covered-active historical validation users; counts are positive user-item candidate pairs',
        'artifact':evidence_id(path,reason='explicit_registry_evidence'),'final_week':'not_run'}
    write(path.with_suffix('.json'),r);return r


def fusion(w):
    root=ART/'FRESH-601'/w;root.mkdir(parents=True,exist_ok=True)
    if (root/'READY.json').exists():return read(root/'READY.json')
    assert not (root/'evaluation.duckdb').exists(),'preserve partial fusion evidence'
    with we.connection() as con:
        for alias,trial in [('base','FRESH-000'),('learned','FRESH-501')]:
            con.execute(f'ATTACH {we.literal(ART/trial/w/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
        n=con.execute('''SELECT count(*) FROM base.predictions b FULL JOIN learned.predictions n USING(customer_id,article_id)
            WHERE b.customer_id IS NULL OR n.customer_id IS NULL OR b.target<>n.target OR b.candidate_rank<>n.candidate_rank OR b.user_history_events_12w<>n.user_history_events_12w''').fetchone()[0]
        assert n==0
        con.execute('''CREATE TEMP TABLE merged AS SELECT b.* EXCLUDE(score),b.score score_base,n.score score_bpr
            FROM base.predictions b JOIN learned.predictions n USING(customer_id,article_id)''')
        con.execute(f'CREATE TEMP TABLE ranks AS SELECT *,{rank_sql("score_base")} base_rank,{rank_sql("score_bpr")} bpr_rank FROM merged')
        con.execute(f'ATTACH {we.literal(root/"evaluation.duckdb")} AS outdb')
        con.execute('''CREATE TABLE outdb.predictions AS SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w,
            1.0/(60+base_rank)+1.0/(60+bpr_rank) score FROM ranks''')
        con.execute(f'CREATE TABLE outdb.top12 AS SELECT * FROM (SELECT *,{final_rank_sql()} final_rank FROM outdb.predictions) WHERE final_rank<=12')
        con.execute(f'COPY (SELECT * FROM outdb.top12 ORDER BY customer_id,final_rank) TO {we.literal(root/"top12.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    r={'window':w,'parents':SYSTEMS,'fits':0,'constant':60,'weights':[1,1],
        'top12':evidence_id(root/'top12.parquet',reason='explicit_registry_evidence'),'final_week':'not_run'}
    write(root/'READY.json',r);return r


def run(chain=None):
    c=setup();e=FreshEngine(c);start=time.perf_counter()
    windows=list(c['rolling_protocol'])
    if chain is not None:windows=[windows[chain-1]]
    for w in windows:
        root=ART/'chain_status'/w
        if (root/'COMPLETE.json').exists():continue
        p=c['rolling_protocol'][w]
        # Build once at its own cutoff, never late-cutoff backfill.
        for cutoff in [*p['inner_train'],p['inner_validation'],p['outer_validation']]:
            e.base_path(cutoff)
        for trial in SYSTEMS:
            families=[] if trial=='FRESH-000' else ['bpr_match']
            extra=[] if trial=='FRESH-000' else c['extra_features']
            inner_path=ART/trial/w/'inner_metrics.json'
            print(f'{w} {trial}: inner fit',flush=True)
            inner=read(inner_path) if inner_path.exists() else e.train_inner(w,trial,families,extra)
            outer_path=ART/trial/w/'outer_metrics.json'
            print(f'{w} {trial}: outer refit/score',flush=True)
            if not outer_path.exists():
                # Build BPR for outer before fitting the ranker, avoiding partial scoring DB on cost pause.
                if families:bpr.build_features(p['outer_validation'],e)
                e.train_outer(w,trial,families,extra,inner)
        diagnostic=ART/'diagnostics'/w/'inner_ranks.json'
        if not diagnostic.exists():inner_diagnostic(w,e)
        fusion(w)
        write(root/'COMPLETE.json',{'window':w,'completed_at':now(),'final_week':'not_run'})
        print(f'{w}: three frozen systems complete; no feedback-driven changes',flush=True)
    print({'invocation_seconds':time.perf_counter()-start,'final_week':'not_run'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chain',type=int,choices=range(1,5));a=p.parse_args()
    try:run(a.chain)
    except Exception:
        write(REPORT/f'FRESH_FAILURE_{time.time_ns()}.json',{'created_at':now(),'traceback':traceback.format_exc(),'final_week':'not_run'})
        raise
