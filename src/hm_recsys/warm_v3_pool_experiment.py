"""WV3-401 fixed expanded-pool E0/E1 refits and fixed RRF; INNER screen only."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from . import warm_v2_engine as we
from .warm_v2_contract import read, write, now
from .warm_v2_rank_fusion import rank_sql, ap_table
from .warm_v3_pool import PoolEngine, INNER_CUTOFFS, ART, connection
from .warm_v3_retrieval_audit import pool_metrics

TRAIN_CUTOFFS = ('2019-11-27', '2020-01-22', '2020-04-29', '2020-06-24')
APPROVED = tuple(sorted(set(INNER_CUTOFFS) | set(TRAIN_CUTOFFS)))
MODELS = ART / 'models'
BASELINE_DIR = common.ART / 'gate_data'
MODEL_NAMES = ('WV3-401-E0', 'WV3-401-E1')
EXTRA = ['wv2_bpr_user_item_score', 'wv2_bpr_unavailable']


def preregister():
    path = ART / 'SCREEN_CONTRACT.json'
    if path.exists():
        return read(path)
    value = {'created_at': now(), 'experiment_id': 'WV3-401', 'role': 'original_inner_only',
        'approved_cutoffs': list(APPROVED), 'inner_train_cutoffs': list(TRAIN_CUTOFFS),
        'inner_validation_cutoffs': list(INNER_CUTOFFS),
        'models': list(MODEL_NAMES), 'only_fusion_is_candidate': True,
        'parameters': 'unchanged original84/86LambdaRank; max200,patience20,MAP12; originaltwo-strata30:1',
        'fusion': '1/(60+E0_rank)+1/(60+E1_rank), ranks over entireexpandeduserpool, tiesoriginalcandidate_rank/article_id',
        'baseline': 'savedWV2-601original-inner E0/E1RRF; newlycoveredactiveusers absentfromoldcoveredset haveoldAP0',
        'denominator': 'full frozen10percent truth-user cohort; nonactive contributions unchanged and cancel; alltruth denominator min(ntruth,12)',
        'gate': {'mean_delta_min': 0.0001, 'positive_windows_min': 3, 'worst_delta_min': -0.0005},
        'artifact_root': str(ART), 'no_outer_training_or_evaluation': True, 'final_week': 'not_run'}
    write(path, value)
    return value


def screen_gate(deltas):
    ds = np.asarray(list(deltas), dtype=np.float64)
    if len(ds) != 4 or not np.isfinite(ds).all():
        raise ValueError('Screen requires four finite INNER deltas')
    return {'mean_population_delta': float(ds.mean()), 'positive_windows': int((ds > 0).sum()),
        'worst_delta': float(ds.min()), 'passed': bool(ds.mean() >= .0001 and (ds > 0).sum() >= 3 and ds.min() >= -.0005)}


def compare_ap(new, old, total_users):
    if new.customer_id.duplicated().any() or old.customer_id.duplicated().any():
        raise ValueError('Duplicate user AP rows')
    if not set(old.customer_id) <= set(new.customer_id):
        raise ValueError('Expanded eligible pool lost originally covered users')
    merged = new.merge(old[['customer_id', 'ap_rrf']].rename(columns={'ap_rrf': 'ap_old_rrf'}), on='customer_id', how='left', validate='one_to_one')
    merged['newly_covered'] = merged.ap_old_rrf.isna()
    merged['ap_old_rrf'] = merged.ap_old_rrf.fillna(0.0)
    if not len(merged) <= total_users or total_users <= 0:
        raise ValueError('Wrong full-user denominator')
    summaries = {}
    for label in ('e0', 'e1', 'rrf'):
        delta = merged['ap_' + label] - merged.ap_old_rrf
        summaries[label] = {'population_delta': float(delta.sum() / total_users),
            'population_map_component': float(merged['ap_' + label].sum() / total_users),
            'improved_users': int((delta > 1e-15).sum()), 'harmed_users': int((delta < -1e-15).sum()),
            'gross_positive': float(delta[delta > 0].sum() / total_users),
            'gross_negative': float(delta[delta < 0].sum() / total_users)}
    return merged, summaries


def window_screen(engine, window, protocol):
    root = ART / 'screen' / window
    ready = root / 'SCREEN.json'
    if ready.exists():
        return read(ready)
    common.budget(5)
    cutoff = protocol['inner_validation']
    assert cutoff in INNER_CUTOFFS and set(protocol['inner_train']) <= set(TRAIN_CUTOFFS)
    started = time.perf_counter()
    fits = {}
    for model_name, families, columns in ((MODEL_NAMES[0], [], []), (MODEL_NAMES[1], ['bpr_match'], EXTRA)):
        # Only a process-local destination override: original trainer function and parameters unchanged.
        we.ARTIFACT = MODELS
        metadata = MODELS / model_name / window / 'inner_metrics.json'
        fits[model_name] = read(metadata) if metadata.exists() else engine.train_inner(window, model_name, families, columns)
    fp, cache_meta = engine.cached_data(cutoff, 'inner')
    f = we.load_parquet(fp)
    scored = f[['customer_id', 'article_id', 'candidate_rank', 'target', 'truth_count', 'user_history_events_12w']].copy()
    for model_name, label, families in ((MODEL_NAMES[0], 'e0', []), (MODEL_NAMES[1], 'e1', ['bpr_match'])):
        model = lgb.Booster(model_file=str(MODELS / model_name / window / 'inner_model.txt'))
        maps = read(MODELS / model_name / window / 'inner_category_maps.json')
        maps = {key: {int(a): int(b) for a, b in value.items()} for key, value in maps.items()}
        frame = common.attach_features(f, cutoff, families, engine) if families else f
        scored['score_' + label] = model.predict(we.prepare(frame, model.feature_name(), maps), num_threads=4)
    old_meta = read(BASELINE_DIR / f'2020_inner_{cutoff}' / 'DATA.json')
    assert old_meta['cutoff'] == cutoff and old_meta['total_users'] == cache_meta['source_users']
    old_ap = we.load_parquet(old_meta['users_path'])[['customer_id', 'ap_rrf']]
    root.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.register('scored', scored)
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id FROM read_parquet({we.literal(common.TX)})
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        con.execute('CREATE TEMP TABLE truth_users AS SELECT customer_id,count(*) truth_count FROM truth GROUP BY customer_id')
        con.execute(f'CREATE VIEW full_pool AS SELECT * FROM read_parquet({we.literal(engine.base_path(cutoff))})')
        con.execute(f'CREATE VIEW original_pool AS SELECT * FROM read_parquet({we.literal(engine.original.base_path(cutoff))})')
        expected = con.execute('SELECT customer_id FROM truth_users ORDER BY customer_id').fetchdf()
        observed = con.execute('SELECT DISTINCT customer_id FROM full_pool ORDER BY customer_id').fetchdf()
        pd.testing.assert_frame_equal(expected, observed)
        assert len(expected) == cache_meta['source_users']
        assert con.execute('''SELECT count(*) FROM full_pool p LEFT JOIN truth t USING(customer_id,article_id)
            WHERE p.target<>CAST(t.article_id IS NOT NULL AS INTEGER)''').fetchone()[0] == 0
        assert con.execute('''SELECT count(*) FROM scored s JOIN truth_users t USING(customer_id)
            WHERE s.truth_count<>t.truth_count''').fetchone()[0] == 0
        con.execute(f'CREATE VIEW old_ranks AS SELECT * FROM read_parquet({we.literal(old_meta["ranks_path"])})')
        assert con.execute('''SELECT count(*) FROM old_ranks o LEFT JOIN scored n USING(customer_id,article_id)
            WHERE n.customer_id IS NULL OR o.target<>n.target OR o.candidate_rank<>n.candidate_rank
            OR o.truth_count<>n.truth_count''').fetchone()[0] == 0
        con.execute(f'''CREATE TEMP TABLE x AS SELECT *,{rank_sql('score_e0')} r0,{rank_sql('score_e1')} r1 FROM scored''')
        con.execute('CREATE TEMP TABLE y AS SELECT *,1.0/(60+r0)+1.0/(60+r1) rrf_score FROM x')
        con.execute(f'CREATE TEMP TABLE z AS SELECT *,{rank_sql("rrf_score")} rf FROM y')
        ap = None
        for label, rank in (('e0', 'r0'), ('e1', 'r1'), ('rrf', 'rf')):
            one = ap_table(con, 'z', rank).rename(columns={'ap': 'ap_' + label})
            ap = one if ap is None else ap.merge(one, on='customer_id', validate='one_to_one')
        merged, values = compare_ap(ap, old_ap, cache_meta['source_users'])
        newly_covered = merged.loc[merged.newly_covered, ['customer_id']]
        con.register('new_users', newly_covered)
        assert con.execute('''SELECT coalesce(sum(p.target),0) FROM original_pool p JOIN new_users n USING(customer_id)''').fetchone()[0] == 0
        con.register('user_ap', merged)
        con.execute(f'COPY user_ap TO {we.literal(root / "user_ap.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        con.execute(f'COPY z TO {we.literal(root / "ranks.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        current = pool_metrics(con, 'SELECT customer_id,article_id FROM original_pool')
        expanded = pool_metrics(con, 'SELECT customer_id,article_id FROM full_pool')
    result = {'window': window, 'cutoff': cutoff, 'role': 'inner_only', 'producer_models': fits,
        'systems': values, 'population_delta': values['rrf']['population_delta'],
        'full_users': cache_meta['source_users'], 'expanded_covered_active_users': len(ap),
        'original_covered_active_users': len(old_ap), 'newly_covered_active_users': int(merged.newly_covered.sum()),
        'candidate_current': current, 'candidate_expanded': expanded,
        'old_baseline': old_meta, 'independent_cohort_truth_and_old_pair_parity': True,
        'runtime_seconds': time.perf_counter() - started,
        'artifacts': {'user_ap': str(root / 'user_ap.parquet'), 'ranks': str(root / 'ranks.parquet')}, 'final_week': 'not_run'}
    write(ready, result)
    print({'window': window, '401_fusion_delta': result['population_delta'], 'seconds': result['runtime_seconds']}, flush=True)
    del f, frame, scored, merged
    gc.collect()
    return result


def run(command='screen'):
    common.setup()
    assert any(t['experiment_id'] == 'WV3-401' for t in read(common.REGISTRY)['trials'])
    preregister()
    engine = PoolEngine(allowed_cutoffs=APPROVED, device='cpu')
    started = time.perf_counter()
    profile = {}
    for cutoff in TRAIN_CUTOFFS:
        before = time.perf_counter()
        engine.base_path(cutoff)
        profile[cutoff] = {'observed_call_seconds': time.perf_counter() - before,
            'pool_build': read(ART / cutoff / 'BUILD.json')}
    profile_path = ART / 'INNER_TRAIN_POOL_PROFILE.json'
    if not profile_path.exists():
        write(profile_path, {'created_at': now(), 'windows': profile,
            'wall_seconds': time.perf_counter() - started, 'device': 'cpu', 'new_fits': 0, 'final_week': 'not_run'})
    if command == 'build':
        return profile
    # Training starts only after all fixed input pools have passed identity/profile checks.
    common.budget(10)
    results = {w: window_screen(engine, w, p) for w, p in engine.contract['rolling_protocol'].items()}
    gate = screen_gate([r['population_delta'] for r in results.values()])
    output = {'experiment_id': 'WV3-401', 'role': 'inner_only', 'windows': results, 'screening': gate,
        'runtime_seconds': time.perf_counter() - started, 'candidate_pool_changed': True,
        'formal_candidate': 'fixedRRF60 ofsame-expanded-pool84/86models', 'final_week': 'not_run',
        'outer_authorized': False}
    write(ART / 'SCREEN.json', output)
    print({'WV3-401': gate, 'outer': 'not_run'}, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['build', 'screen'], default='screen', nargs='?')
    args = parser.parse_args()
    try:
        run(args.command)
    except Exception:
        write(ART / 'SCREEN_FAILURE.json', {'created_at': now(), 'traceback': traceback.format_exc(), 'final_week': 'not_run'})
        raise
