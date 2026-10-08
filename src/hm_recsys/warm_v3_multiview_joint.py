"""90-feature BPR+sequence+LightGCN joint expert and fixed two-way fusion."""
import gc
from pathlib import Path
import time

import duckdb
import lightgbm as lgb
import numpy as np

from . import warm_v3_common as common
from . import warm_v3_joint as joint
from . import warm_v2_engine as we
from .metrics import apk
from .warm_v2_contract import evidence_id, guard, read, write
from .warm_v2_rank_fusion import ap_table, rank_sql
from .warm_v3_lightgcn import FORMAL_PARAMS

STANDALONE = 'WV3-331'
FUSION = 'WV3-332'
FAMILIES = ['bpr_match', 'sequence', 'lightgcn']
EXTRA = [
    'wv2_bpr_user_item_score', 'wv2_bpr_unavailable',
    'wv3_sequence_score', 'wv3_sequence_unavailable',
    'wv3_lightgcn_score', 'wv3_lightgcn_unavailable',
]
CANONICAL_PARAMS = {
    'architecture': 'joint90_BPR_sequence_LightGCN_LambdaRank',
    'families': FAMILIES,
    'extra_columns': EXTRA,
    'feature_count': 90,
    'source_selection': ('four-inner AP-sensitive audit: LightGCN corrects 46.7-50.3% of BPR-missed '
                         'weight; sequence corrects 29.6-36.1% where BPR and LightGCN are both wrong'),
    'new_representation_training': False,
    'candidate_pool': 'original frozen100-300',
    'tree_parameters': 'exact original84 anchor parameters; no overrides',
    'sampling': 'original30:1 two-strata distribution-aware, all positives, no IPW',
    'early_stopping': 'original inner exact MAP12, max200, patience20',
    'ordering_331': 'joint90 score',
    'ordering_332': '1/(60+E0rank)+1/(60+joint90rank)',
    'fusion_threeway': False,
    'inactive': 'exact original candidate_rank fallback',
    'full_truth_denominator': True,
    'inner_gate': {'mean_min': .0001, 'positive_min': 3, 'worst_min': -.0005},
    'only_root_selects_and_exposes_outer': True,
    'final_week': 'not_run',
}


class MultiViewEngine(joint.JointEngine):
    def signal_path(self, family, cutoff):
        if family != 'lightgcn':
            return super().signal_path(family, cutoff)
        root = common.ART / 'lightgcn/formal1000' / cutoff
        path, feature_meta, model_meta = root / 'features.parquet', root / 'FEATURES.json', root / 'MODEL.json'
        if not path.is_file() or not feature_meta.is_file() or not model_meta.is_file():
            raise RuntimeError('Multi-view joint expert requires a completed frozen LightGCN cache')
        meta, model = read(feature_meta), read(model_meta)
        assert meta['cutoff'] == model['cutoff'] == cutoff
        assert meta['candidate_identity_unchanged'] and meta['features'] == EXTRA[-2:]
        assert Path(meta['source']).resolve() == Path(self.base_path(cutoff)).resolve()
        assert model['params'] == FORMAL_PARAMS and model['completed_updates'] == 1000
        assert model['point_in_time_safe'] and model['latest_history_date'] < cutoff
        return path, meta


def configure_joint_runner():
    # The reused runner is process-local; no registry or prior artifact is mutated.
    joint.STANDALONE = STANDALONE
    joint.FUSION = FUSION
    joint.FAMILIES = FAMILIES
    joint.EXTRA = EXTRA
    joint.CANONICAL_PARAMS = CANONICAL_PARAMS
    joint.JointEngine = MultiViewEngine


def screen():
    configure_joint_runner()
    return joint.screen()


def outer_window(engine, window, protocol):
    root = common.ART / STANDALONE / window
    review_path = root / 'MULTIVIEW_OUTER_REVIEW.json'
    if review_path.is_file():
        return read(review_path)
    start = time.perf_counter()
    cutoff = guard(protocol['outer_validation'])
    for source_cutoff in [*protocol['outer_train'], cutoff]:
        for family in FAMILIES:
            engine.signal_path(family, source_cutoff)
    inner = read(root / 'JOINT_INNER_REVIEW.json')['fit']
    metadata = root / 'outer_metrics.json'
    we.ARTIFACT = common.ART
    fit = read(metadata) if metadata.is_file() else engine.train_outer(
        window, STANDALONE, FAMILIES, EXTRA, inner)
    assert fit['features'] == engine.features + EXTRA and fit['feature_count'] == 90
    assert fit['families'] == FAMILIES and fit['cutoff'] == cutoff

    fusion_root = common.ART / FUSION / window
    fusion_root.mkdir(parents=True, exist_ok=True)
    output_db = fusion_root / 'evaluation.duckdb'
    if output_db.exists():
        raise RuntimeError('Incomplete WV3-332 fusion database exists; preserve for review')
    base_root = common.OLD
    with we.connection() as con:
        for alias, source in [('b', base_root/'WV2-000'/window),
                              ('n', base_root/'WV2-501'/window), ('x', root)]:
            con.execute(f'ATTACH {we.literal(source/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
        for alias in ('n', 'x'):
            errors = con.execute(f'''SELECT count(*) FROM b.predictions b FULL JOIN {alias}.predictions n
                USING(customer_id,article_id) WHERE b.customer_id IS NULL OR n.customer_id IS NULL
                OR b.target<>n.target OR b.candidate_rank<>n.candidate_rank
                OR b.user_history_events_12w<>n.user_history_events_12w''').fetchone()[0]
            assert errors == 0
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id
            FROM read_parquet({we.literal(engine.transactions)}) WHERE t_dat>=DATE '{cutoff}'
            AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        con.execute('''CREATE TEMP TABLE truth_users AS SELECT customer_id,count(*) truth_count
            FROM truth GROUP BY customer_id''')
        con.execute('''CREATE TEMP TABLE merged AS SELECT b.* EXCLUDE(score),b.score score_base,
            n.score score_bpr,x.score score_joint,t.truth_count FROM b.predictions b
            JOIN n.predictions n USING(customer_id,article_id)
            JOIN x.predictions x USING(customer_id,article_id)
            JOIN truth_users t USING(customer_id)''')
        rows, users = con.execute('SELECT count(*),count(DISTINCT customer_id) FROM merged').fetchone()
        population_errors = con.execute('''SELECT count(*) FROM
            (SELECT DISTINCT customer_id FROM merged) p FULL JOIN truth_users t USING(customer_id)
            WHERE p.customer_id IS NULL OR t.customer_id IS NULL''').fetchone()[0]
        label_errors = con.execute('''SELECT count(*) FROM merged p LEFT JOIN truth t USING(customer_id,article_id)
            WHERE p.target<>CAST(t.article_id IS NOT NULL AS INTEGER)''').fetchone()[0]
        duplicates = con.execute('SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM merged').fetchone()[0]
        assert population_errors == label_errors == duplicates == 0
        con.execute(f'''CREATE TEMP TABLE model_ranks AS SELECT *,{rank_sql('score_base')} r0,
            {rank_sql('score_bpr')} r1,{rank_sql('score_joint')} rj FROM merged''')
        con.execute('''CREATE TEMP TABLE rank_scores AS SELECT *,
            1.0/(60+r0)+1.0/(60+r1) baseline_score,
            1.0/(60+r0)+1.0/(60+rj) fusion_score FROM model_ranks''')
        con.execute(f'''CREATE TEMP TABLE raw_ranks AS SELECT *,{rank_sql('baseline_score')} rb,
            {rank_sql('fusion_score')} rf FROM rank_scores''')
        con.execute('''CREATE TEMP TABLE ranked AS SELECT *,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rb END baseline_rank,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rj END standalone_rank,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rf END fusion_rank FROM raw_ranks''')
        aps = None
        for name, rank in [('baseline','baseline_rank'),('standalone','standalone_rank'),('fusion','fusion_rank')]:
            one = ap_table(con, 'ranked', rank).rename(columns={'ap':'ap_'+name})
            aps = one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
        assert len(aps) <= users
        systems = {}
        for name in ('standalone','fusion'):
            delta = aps['ap_'+name] - aps.ap_baseline
            systems[name] = {'full_MAP': float(aps['ap_'+name].sum()/users),
                'population_delta': float(delta.sum()/users),
                'improved_users': int((delta>1e-15).sum()), 'harmed_users': int((delta< -1e-15).sum()),
                'gross_positive_MAP': float(delta[delta>0].sum()/users),
                'gross_negative_MAP': float(delta[delta<0].sum()/users)}
        expected = read(common.ROOT/'reports/warm_v2/WV2-601_OUTER.json')['per_window_MAP'][window]
        baseline_map = float(aps.ap_baseline.sum()/users)
        assert abs(baseline_map-expected) < 1e-12
        con.execute(f'ATTACH {we.literal(output_db)} AS output')
        con.execute('''CREATE TABLE output.predictions AS SELECT customer_id,article_id,candidate_rank,target,
            user_history_events_12w,fusion_score score FROM ranked''')
        con.execute('''CREATE TABLE output.top12 AS SELECT customer_id,article_id,candidate_rank,target,
            user_history_events_12w,fusion_score score,fusion_rank final_rank
            FROM ranked WHERE fusion_rank<=12''')
        top = con.execute('SELECT customer_id,article_id,fusion_rank FROM ranked WHERE fusion_rank<=12 ORDER BY customer_id,fusion_rank').fetchdf()
        raw_truth = con.execute('SELECT * FROM truth ORDER BY customer_id,article_id').fetchdf()
        truthsets = {u:set(g.article_id) for u,g in raw_truth.groupby('customer_id',sort=False)}
        predicted = {u:g.article_id.tolist() for u,g in top.groupby('customer_id',sort=False)}
        assert set(predicted) == set(truthsets) and all(len(v)==len(set(v))==12 for v in predicted.values())
        independent = float(np.mean([apk(list(t),predicted[u]) for u,t in truthsets.items()]))
        assert abs(independent-systems['fusion']['full_MAP']) < 1e-12
        inactive_errors = con.execute('''SELECT count(*) FROM ranked
            WHERE user_history_events_12w=0 AND fusion_rank<>candidate_rank''').fetchone()[0]
        assert inactive_errors == 0
        rank_path = root / 'outer_multiview_ranks.parquet'
        con.execute(f'COPY ranked TO {we.literal(rank_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        top_path = fusion_root / 'top12.parquet'
        con.execute(f'''COPY (SELECT * FROM output.top12 ORDER BY customer_id,final_rank)
            TO {we.literal(top_path)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
    we.save_parquet(aps, root/'outer_user_ap.parquet')
    result = {'window':window,'cutoff':cutoff,'fit':fit,'systems':systems,'baseline_full_MAP':baseline_map,
        'candidate_rows':rows,'full_users':users,'truth_pairs':len(raw_truth),'candidate_pool_changed':False,
        'candidate_identity_errors':0,'population_errors':population_errors,'label_errors':label_errors,
        'duplicate_pairs':duplicates,'inactive_rank_errors':inactive_errors,
        'independent_fusion_MAP_error':abs(independent-systems['fusion']['full_MAP']),
        'ranks_path':str(rank_path),'fusion_top12':evidence_id(top_path,reason='explicit_registry_evidence'),
        'runtime_seconds':time.perf_counter()-start,'final_week':'not_run'}
    write(review_path,result)
    print({'joint90_outer':window,'systems':{k:v['population_delta'] for k,v in systems.items()}},flush=True)
    del aps
    gc.collect()
    return result


def confirm():
    configure_joint_runner(); common.setup(); common.budget(10)
    registry = read(common.REGISTRY)
    selected = next(t for t in registry['trials'] if t['experiment_id']==FUSION)
    assert selected['outer_exposures']==1
    screened = read(common.REPORT/(FUSION+'_SCREEN.json'))
    assert screened['screening']['passed'] and screened['ordering']=='fusion'
    destination = common.REPORT/(FUSION+'_OUTER.json')
    if destination.is_file():
        return read(destination)
    engine = MultiViewEngine(2020); started = time.perf_counter()
    windows = {w:outer_window(engine,w,p) for w,p in engine.contract['rolling_protocol'].items()}
    maps = {w:r['systems']['fusion']['full_MAP'] for w,r in windows.items()}
    result = {'experiment_id':FUSION,'producer_model_trial':STANDALONE,
        'architecture':CANONICAL_PARAMS['architecture'],'chosen_ordering':'fusion',
        'windows':windows,**common.summary(maps,2020),
        'mechanism_control':{'ordering':'standalone','per_window_MAP':
            {w:r['systems']['standalone']['full_MAP'] for w,r in windows.items()},
            'posthoc_promotion_allowed':False},
        'candidate_pool_changed':False,'feature_count':90,'runtime_seconds':time.perf_counter()-started,
        'no_registry_mutation':True,'final_week':'not_run'}
    write(destination,result)
    write(common.REPORT/(STANDALONE+'_OUTER_CONTROL.json'),{
        'experiment_id':STANDALONE,'ordering':'standalone','role':'mechanism_control_only',
        'posthoc_promotion_allowed':False,'selected_trial':FUSION,
        'per_window_MAP':result['mechanism_control']['per_window_MAP'],'windows':windows,'final_week':'not_run'})
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['screen','confirm'],nargs='?',default='screen')
    args = parser.parse_args()
    globals()[args.command]()
