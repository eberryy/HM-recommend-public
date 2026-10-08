"""WV3-341: one-swap unanimous raw-signal admission on a frozen baseline list."""
from __future__ import annotations
import time

import numpy as np

from . import warm_v3_common as common
from .metrics import apk
from .warm_v2_contract import evidence_id, read, write
from .warm_v2_engine import connection, literal, save_parquet
from .warm_v2_rank_fusion import ap_table, rank_sql
from .warm_v3_expert import screening_gate
from .warm_v3_multiview_joint import MultiViewEngine

TRIAL = 'WV3-341'
SIGNALS = {
    'bpr': ('wv2_bpr_user_item_score', 'wv2_bpr_unavailable'),
    'sequence': ('wv3_sequence_score', 'wv3_sequence_unavailable'),
    'lightgcn': ('wv3_lightgcn_score', 'wv3_lightgcn_unavailable'),
}


def build_admission_ranks(con, source='signals'):
    """Create a full rank permutation with at most one unanimous Top50-to-Top12 exchange."""
    con.execute(f'''CREATE TEMP TABLE eligible_swaps AS SELECT c.customer_id,
        c.article_id challenger,c.baseline_rank challenger_rank,v.article_id victim,v.baseline_rank victim_rank,
        greatest(c.bpr_rank,c.sequence_rank,c.lightgcn_rank) challenger_worst_signal_rank,
        row_number() OVER(PARTITION BY c.customer_id ORDER BY
          greatest(c.bpr_rank,c.sequence_rank,c.lightgcn_rank),c.baseline_rank,c.article_id,
          v.baseline_rank DESC,v.article_id) selection_rank
        FROM {source} c JOIN {source} v ON c.customer_id=v.customer_id
        WHERE c.user_history_events_12w>0 AND c.baseline_rank BETWEEN 13 AND 50
          AND v.baseline_rank BETWEEN 1 AND 12
          AND c.bpr_unavailable=0 AND v.bpr_unavailable=0
          AND c.sequence_unavailable=0 AND v.sequence_unavailable=0
          AND c.lightgcn_unavailable=0 AND v.lightgcn_unavailable=0
          AND c.bpr_rank<v.bpr_rank AND c.sequence_rank<v.sequence_rank
          AND c.lightgcn_rank<v.lightgcn_rank''')
    con.execute('''CREATE TEMP TABLE chosen_swap AS SELECT * FROM eligible_swaps WHERE selection_rank=1''')
    con.execute(f'''CREATE TEMP TABLE admitted AS SELECT s.*,
        CASE WHEN s.article_id=c.challenger THEN c.victim_rank
             WHEN s.article_id=c.victim THEN c.challenger_rank ELSE s.baseline_rank END final_rank
        FROM {source} s LEFT JOIN chosen_swap c USING(customer_id)''')
    return 'admitted'


def evaluate(engine, window, cutoff, base_path, base_rank_column, total_users, stage):
    root = common.ART/TRIAL/window/stage
    output = root/'REVIEW.json'
    if output.is_file():
        return read(output)
    root.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    paths = {name:engine.signal_path(name if name!='bpr' else 'bpr_match',cutoff)[0] for name in SIGNALS}
    with connection() as con:
        con.execute(f'''CREATE TEMP TABLE base AS SELECT customer_id,article_id,candidate_rank,target,truth_count,
            user_history_events_12w,{base_rank_column} baseline_rank
            FROM read_parquet({literal(base_path)})''')
        for name,path in paths.items():
            missing = con.execute(f'''SELECT count(*) FROM base b LEFT JOIN read_parquet({literal(path)}) s
                USING(customer_id,article_id) WHERE s.customer_id IS NULL''').fetchone()[0]
            assert missing == 0
        con.execute(f'''CREATE TEMP TABLE raw AS SELECT b.*,
            p.wv2_bpr_user_item_score bpr_score,p.wv2_bpr_unavailable bpr_unavailable,
            s.wv3_sequence_score sequence_score,s.wv3_sequence_unavailable sequence_unavailable,
            g.wv3_lightgcn_score lightgcn_score,g.wv3_lightgcn_unavailable lightgcn_unavailable
            FROM base b JOIN read_parquet({literal(paths['bpr'])}) p USING(customer_id,article_id)
            JOIN read_parquet({literal(paths['sequence'])}) s USING(customer_id,article_id)
            JOIN read_parquet({literal(paths['lightgcn'])}) g USING(customer_id,article_id)''')
        con.execute(f'''CREATE TEMP TABLE signals AS SELECT *,{rank_sql('bpr_score')} bpr_rank,
            {rank_sql('sequence_score')} sequence_rank,{rank_sql('lightgcn_score')} lightgcn_rank FROM raw''')
        table = build_admission_ranks(con)
        rows, users = con.execute(f'SELECT count(*),count(DISTINCT customer_id) FROM {table}').fetchone()
        assert 0 < users <= total_users
        permutation_errors = con.execute(f'''SELECT count(*) FROM (SELECT customer_id,count(*) n,
            count(DISTINCT final_rank) nr,min(final_rank) lo,max(final_rank) hi FROM {table} GROUP BY customer_id)
            WHERE n<>nr OR lo<>1 OR hi<>n''').fetchone()[0]
        inactive_errors = con.execute(f'''SELECT count(*) FROM {table}
            WHERE user_history_events_12w=0 AND final_rank<>baseline_rank''').fetchone()[0]
        assert permutation_errors == inactive_errors == 0
        aps = None
        for name,rank in [('baseline','baseline_rank'),('admission','final_rank')]:
            one=ap_table(con,table,rank).rename(columns={'ap':'ap_'+name})
            aps=one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
        assert len(aps) == users
        delta = aps.ap_admission-aps.ap_baseline
        swaps,eligible = con.execute('SELECT count(*),(SELECT count(*) FROM eligible_swaps) FROM chosen_swap').fetchone()
        truth_change = con.execute('''SELECT sum(c.target),sum(v.target) FROM chosen_swap x
            JOIN admitted c ON x.customer_id=c.customer_id AND x.challenger=c.article_id
            JOIN admitted v ON x.customer_id=v.customer_id AND x.victim=v.article_id''').fetchone()
        rank_path = root/'ranks.parquet'
        con.execute(f'COPY {table} TO {literal(rank_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    save_parquet(aps,root/'user_ap.parquet')
    result={'window':window,'cutoff':cutoff,'stage':stage,'candidate_rows':rows,'total_users':total_users,
        'included_users':users,'baseline_MAP':float(aps.ap_baseline.sum()/total_users),
        'MAP':float(aps.ap_admission.sum()/total_users),
        'population_delta':float(delta.sum()/total_users),'positive_user_count':int((delta>1e-15).sum()),
        'harmed_user_count':int((delta< -1e-15).sum()),'gross_positive_MAP':float(delta[delta>0].sum()/total_users),
        'gross_negative_MAP':float(delta[delta<0].sum()/total_users),
        'users_with_swap':swaps,'eligible_user_item_pairs':eligible,
        'truth_challengers':int(truth_change[0] or 0),'truth_victims':int(truth_change[1] or 0),
        'candidate_pool_changed':False,'maximum_swaps_per_user':1,'permutation_errors':permutation_errors,
        'inactive_rank_errors':inactive_errors,'signal_paths':{k:str(v) for k,v in paths.items()},
        'ranks_path':str(rank_path),'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(output,result)
    print({'consensus':stage,'window':window,'delta':result['population_delta'],'swaps':swaps},flush=True)
    return result


def screen():
    common.setup(); common.budget(5)
    entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['outer_exposures']==0 and entry['decision']=='preregistered'
    engine=MultiViewEngine(2020); started=time.perf_counter(); windows={}
    for window,p in engine.contract['rolling_protocol'].items():
        cutoff=p['inner_validation']; meta=read(common.ART/'gate_data'/f'2020_inner_{cutoff}'/'DATA.json')
        windows[window]=evaluate(engine,window,cutoff,meta['ranks_path'],'ap_rf',meta['total_users'],'inner')
        assert abs(windows[window]['baseline_MAP']-meta['baseline_map_population_component'])<1e-12
    gate=screening_gate(v['population_delta'] for v in windows.values())
    result={'experiment_id':TRIAL,'architecture':'one_swap_unanimous_raw_signal_admission',
        'windows':windows,'screening':gate,'candidate_pool_changed':False,'new_training':False,
        'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(common.REPORT/(TRIAL+'_SCREEN.json'),result); print(gate,flush=True); return result


def confirm():
    common.setup(); common.budget(10)
    entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['outer_exposures']==1 and read(common.REPORT/(TRIAL+'_SCREEN.json'))['screening']['passed']
    destination=common.REPORT/(TRIAL+'_OUTER.json')
    if destination.is_file():return read(destination)
    engine=MultiViewEngine(2020); started=time.perf_counter(); windows={}
    for window,p in engine.contract['rolling_protocol'].items():
        base=common.ART/'WV3-331'/window/'outer_multiview_ranks.parquet'
        prior=read(common.ART/'WV3-331'/window/'MULTIVIEW_OUTER_REVIEW.json')
        windows[window]=evaluate(engine,window,p['outer_validation'],base,'baseline_rank',prior['full_users'],'outer')
        assert abs(windows[window]['baseline_MAP']-prior['baseline_full_MAP'])<1e-12
    maps={w:r['MAP'] for w,r in windows.items()}
    result={'experiment_id':TRIAL,'architecture':'one_swap_unanimous_raw_signal_admission',
        'windows':windows,**common.summary(maps,2020),'candidate_pool_changed':False,'new_training':False,
        'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(destination,result); return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['screen','confirm'])
    globals()[parser.parse_args().command]()
