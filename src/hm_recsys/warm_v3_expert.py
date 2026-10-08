"""Frozen-pool 84+2 learned experts, inner screens and fixed three-way fusion.

This module never registers, exposes, selects, or promotes a trial. The root
orchestrator must preregister the chosen ordering before calling confirm().
"""
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
from .warm_v2_contract import evidence_id, guard, now, read, write
from .warm_v2_engine import connection, literal, load_parquet, prepare, save_parquet
from .warm_v2_rank_fusion import ap_table, final_rank_sql, rank_sql
from .metrics import apk

ART, REPORT = common.ART, common.REPORT
ORDERINGS = ('standalone', 'fusion')
FEATURES = {
    'graph': ['wv3_graph_user_item_score', 'wv3_graph_unavailable'],
    'sequence': ['wv3_sequence_score', 'wv3_sequence_unavailable'],
    'lightgcn': ['wv3_lightgcn_score', 'wv3_lightgcn_unavailable'],
    'userknn': ['wv3_userknn_neighbor_score', 'wv3_userknn_unavailable'],
}


def report_path(trial, stage, year):
    if year not in (2019, 2020):
        raise ValueError('Only registered development and cross-year replay years')
    return REPORT/f'{trial}{"_2019" if year == 2019 else ""}_{stage}.json'


def screening_gate(deltas):
    values = list(deltas)
    if len(values) != 4 or not np.isfinite(values).all():
        raise ValueError('Screen gate requires four finite population deltas')
    return {'mean_population_delta': float(np.mean(values)),
        'positive_windows': sum(v > 0 for v in values), 'worst_delta': min(values),
        'passed': bool(np.mean(values) >= .0001 and sum(v > 0 for v in values) >= 3
                       and min(values) >= -.0005)}


def rank_bundle(con, source):
    """Complete candidate-pool ranks with fixed tie order; no source-rank fusion."""
    con.execute(f'''CREATE TEMP TABLE expert_model_ranks AS SELECT *,
        {rank_sql('score_base')} r0, {rank_sql('score_bpr')} r1,
        {rank_sql('score_expert')} r2 FROM {source}''')
    con.execute('''CREATE TEMP TABLE expert_scores AS SELECT *,
        1.0/(60+r0)+1.0/(60+r1) rrf_score,
        1.0/(60+r0)+1.0/(60+r1)+1.0/(60+r2) fusion_score
        FROM expert_model_ranks''')
    con.execute(f'''CREATE TEMP TABLE expert_raw_ranks AS SELECT *,
        {rank_sql('rrf_score')} rf, {rank_sql('fusion_score')} r3 FROM expert_scores''')
    con.execute('''CREATE TEMP TABLE expert_ranked AS SELECT *,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r0 END ap_r0,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r1 END ap_r1,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rf END ap_rf,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r2 END ap_r2,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r3 END ap_r3
        FROM expert_raw_ranks''')
    return 'expert_ranked'


def diagnose(con, table, total_users):
    aps = None
    for name, rank in [('base', 'ap_r0'), ('bpr', 'ap_r1'), ('rrf', 'ap_rf'),
                       ('standalone', 'ap_r2'), ('fusion', 'ap_r3')]:
        a = ap_table(con, table, rank).rename(columns={'ap': 'ap_'+name})
        aps = a if aps is None else aps.merge(a, on='customer_id', validate='one_to_one')
    if total_users < len(aps) or total_users <= 0:
        raise ValueError('Invalid complete evaluation-user denominator')
    systems = {}
    for name in ORDERINGS:
        delta = aps['ap_'+name]-aps.ap_rrf
        systems[name] = {'covered_map': float(aps['ap_'+name].mean()),
            'population_map_component': float(aps['ap_'+name].sum()/total_users),
            'population_delta': float(delta.sum()/total_users),
            'users_improved': int((delta > 1e-15).sum()),
            'users_harmed': int((delta < -1e-15).sum()),
            'users_unchanged': int((np.abs(delta) <= 1e-15).sum()),
            'gross_positive_MAP_contribution': float(delta[delta > 0].sum()/total_users),
            'gross_negative_MAP_contribution': float(delta[delta < 0].sum()/total_users)}
    unique = con.execute(f'''SELECT
        count(*) FILTER(WHERE target=1 AND ap_r2<=12 AND ap_r0>12 AND ap_r1>12),
        count(*) FILTER(WHERE target=1 AND ap_r2<=12 AND (ap_r0<=12 OR ap_r1<=12)),
        count(*) FILTER(WHERE target=1 AND ap_r2>12 AND (ap_r0<=12 OR ap_r1<=12)),
        count(*) FILTER(WHERE target=1 AND ap_r2<=12 AND ap_rf>12),
        count(*) FILTER(WHERE target=1 AND ap_r3<=12 AND ap_rf>12),
        count(*) FILTER(WHERE target=1 AND ap_r3>12 AND ap_rf<=12),
        corr(score_expert, score_base), corr(score_expert, score_bpr)
        FROM {table}''').fetchone()
    complementarity = {'new_expert_only_correct_top12_pairs_vs_E0_E1_union': unique[0],
        'new_expert_correct_top12_shared_with_E0_or_E1': unique[1],
        'E0_or_E1_correct_top12_missed_by_new_expert': unique[2],
        'new_expert_only_correct_top12_pairs_vs_rrf': unique[3],
        'fusion_new_correct_top12_pairs_vs_rrf': unique[4],
        'fusion_lost_correct_top12_pairs_vs_rrf': unique[5],
        'pearson_new_score_vs_E0': unique[6], 'pearson_new_score_vs_E1': unique[7],
        'definition': '独有正确Top12按未来购买命中的用户—商品对计数，参照另一模型最终Top12集合；不是召回源独有正例。相关系数在同一完整候选行集合上计算。'}
    return {'systems': systems, 'complementarity': complementarity,
        'included_users': len(aps), 'total_users': total_users,
        'population_weight': len(aps)/total_users,
        'baseline_population_component': float(aps.ap_rrf.sum()/total_users)}, aps


def inner_baseline(window, engine):
    """Reuse root's 2020 gate ranks; 2019 gets cutoff-matched FRESH checkpoints."""
    year = engine.year
    cutoff = guard(engine.contract['rolling_protocol'][window]['inner_validation'])
    existing = ART/'gate_data'/f'{year}_inner_{cutoff}'/'DATA.json'
    if existing.exists():
        m = read(existing)
        return Path(m['ranks_path']), m['total_users']
    root = ART/'expert_baselines'/f'{year}_inner_{cutoff}'
    path, meta = root/'ranks.parquet', root/'DATA.json'
    if meta.exists():
        m = read(meta)
        return path, m['total_users']
    fp, stats = engine.cached_data(cutoff, 'inner')
    f = load_parquet(fp)
    out = f[['customer_id', 'article_id', 'candidate_rank', 'target', 'truth_count',
             'user_history_events_12w']].copy()
    old = common.FRESH if year == 2019 else common.OLD
    prefix = 'FRESH' if year == 2019 else 'WV2'
    for suffix, label, families in [('000', 'base', []), ('501', 'bpr', ['bpr_match'])]:
        modelroot = old/f'{prefix}-{suffix}'/window
        model = lgb.Booster(model_file=str(modelroot/'inner_model.txt'))
        maps = {k: {int(a): int(b) for a, b in v.items()}
                for k, v in read(modelroot/'inner_category_maps.json').items()}
        ff = common.attach_features(f, cutoff, families, engine) if families else f
        out['score_'+label] = model.predict(prepare(ff, model.feature_name(), maps), num_threads=8)
    save_parquet(out, path)
    write(meta, {'window': window, 'year': year, 'cutoff': cutoff,
        'total_users': stats['source_users'], 'included_users': stats['groups'],
        'rows': len(out), 'reference': f'{prefix}-000 / {prefix}-501 frozen inner checkpoints',
        'final_week': 'not_run'})
    return path, stats['source_users']


def train_inner_once(engine, window, trial, family):
    root = ART/trial/window
    meta = root/'inner_metrics.json'
    if meta.exists():
        value = read(meta)
        assert value['families'] == [family] and value['features'] == engine.features+FEATURES[family]
        assert value['cutoff'] == engine.contract['rolling_protocol'][window]['inner_validation']
        return value
    if (root/'inner_model.txt').exists():
        raise RuntimeError('Incomplete inner model evidence; preserve and inspect before retry')
    common.budget(2)
    return engine.train_inner(window, trial, [family], FEATURES[family])


def inner_window(engine, window, family, standalone_trial, fusion_trial):
    root = ART/standalone_trial/window
    outpath = root/'EXPERT_INNER_REVIEW.json'
    if outpath.exists():
        result = read(outpath)
        assert result['family'] == family and result['fusion_trial'] == fusion_trial
        return result
    start = time.perf_counter()
    fit = train_inner_once(engine, window, standalone_trial, family)
    cutoff = guard(engine.contract['rolling_protocol'][window]['inner_validation'])
    reference, total_users = inner_baseline(window, engine)
    path, stats = engine.cached_data(cutoff, 'inner')
    frame = load_parquet(path)
    model = lgb.Booster(model_file=str(root/'inner_model.txt'))
    maps = {k: {int(a): int(b) for a, b in v.items()}
            for k, v in read(root/'inner_category_maps.json').items()}
    ff = common.attach_features(frame, cutoff, [family], engine)
    scored = frame[['customer_id', 'article_id', 'candidate_rank', 'target', 'truth_count',
                    'user_history_events_12w']].copy()
    scored['score_expert'] = model.predict(prepare(ff, model.feature_name(), maps), num_threads=8)
    with connection() as con:
        con.register('newscore', scored)
        con.execute(f'''CREATE TEMP VIEW oldscore AS SELECT customer_id, article_id,
            candidate_rank, target, truth_count, user_history_events_12w, score_base, score_bpr
            FROM read_parquet({literal(reference)})''')
        errors = con.execute('''SELECT count(*) FROM newscore n FULL JOIN oldscore b USING(customer_id,article_id)
            WHERE n.customer_id IS NULL OR b.customer_id IS NULL OR n.target<>b.target
            OR n.candidate_rank<>b.candidate_rank OR n.truth_count<>b.truth_count
            OR n.user_history_events_12w<>b.user_history_events_12w''').fetchone()[0]
        assert errors == 0 and total_users == stats['source_users']
        con.execute('''CREATE TEMP TABLE merged AS SELECT n.*, b.score_base, b.score_bpr
            FROM newscore n JOIN oldscore b USING(customer_id,article_id)''')
        assert con.execute('SELECT count(*) FROM merged').fetchone()[0] == len(scored)
        table = rank_bundle(con, 'merged')
        review, aps = diagnose(con, table, total_users)
        assert review['included_users'] == stats['groups']
        assert abs(review['systems']['standalone']['covered_map']-fit['inner_covered_active_map']) < 1e-12
        ranks = root/'inner_expert_ranks.parquet'
        con.execute(f'COPY {table} TO {literal(ranks)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    save_parquet(aps, root/'inner_user_ap.parquet')
    result = {'family': family, 'standalone_trial': standalone_trial, 'fusion_trial': fusion_trial,
        'window': window, 'year': engine.year, 'cutoff': cutoff, 'fit': fit, **review,
        'identity_mismatches': errors, 'ranks_path': str(ranks),
        'runtime_seconds': time.perf_counter()-start, 'final_week': 'not_run'}
    write(outpath, result)
    print({'expert_inner': family, 'window': window,
        'standalone_delta': review['systems']['standalone']['population_delta'],
        'fusion_delta': review['systems']['fusion']['population_delta']}, flush=True)
    del frame, ff, scored, model
    gc.collect()
    return result


def screen(family, standalone_trial, fusion_trial, year=2020):
    """No registry effects. 2019 screening only selects fresh inner iterations."""
    if family not in FEATURES or standalone_trial == fusion_trial:
        raise ValueError('Distinct model and fusion IDs with a supported learned family required')
    common.setup()
    common.budget(5)
    e = common.Engine(year)
    started = time.perf_counter()
    windows = {w: inner_window(e, w, family, standalone_trial, fusion_trial)
               for w in e.contract['rolling_protocol']}
    result = {}
    for ordering, trial in zip(ORDERINGS, (standalone_trial, fusion_trial)):
        reduced = {w: {**r['systems'][ordering], 'fit': r['fit'],
            'complementarity': r['complementarity'], 'included_users': r['included_users'],
            'total_users': r['total_users'], 'ranks_path': r['ranks_path']}
            for w, r in windows.items()}
        gate = screening_gate(r['population_delta'] for r in reduced.values())
        entry = {'experiment_id': trial, 'family': family, 'year': year, 'ordering': ordering,
            'producer_model_trial': standalone_trial, 'producer_feature_count': 86,
            'producer_features': e.features+FEATURES[family], 'windows': reduced, 'screening': gate,
            'runtime_seconds': time.perf_counter()-started, 'candidate_pool_changed': False,
            'fusion': '1/(60+r0)+1/(60+r1)+1/(60+r2)' if ordering == 'fusion' else None,
            'sampling': 'original 30:1 distribution-aware negative:positive; no change',
            'final_week': 'not_run', 'no_registry_mutation': True,
            '2019_policy': 'new inner rounds only; no new architecture or ordering selection' if year == 2019 else None}
        write(report_path(trial, 'SCREEN', year), entry)
        result[ordering] = entry
    return result


def verify_outer_permission(family, standalone_trial, fusion_trial, ordering, year):
    if ordering not in ORDERINGS:
        raise ValueError('Choose standalone or fusion before outer exposure')
    trial = standalone_trial if ordering == 'standalone' else fusion_trial
    registry = read(common.REGISTRY)
    entry = next(t for t in registry['trials'] if t['experiment_id'] == trial)
    assert entry['outer_exposures'] == 1, 'Root must preregister/expose this trial before confirm()'
    if year == 2020:
        screened = read(report_path(trial, 'SCREEN', 2020))
        assert screened['family'] == family and screened['screening']['passed']
    else:
        previous = read(report_path(trial, 'OUTER', 2020))
        assert previous['chosen_ordering'] == ordering and previous['family'] == family
    return trial


def outer_window(engine, window, family, standalone_trial, fusion_trial):
    root = ART/standalone_trial/window
    reviewed = root/'EXPERT_OUTER_REVIEW.json'
    if reviewed.exists():
        value = read(reviewed)
        assert value['family'] == family and value['fusion_trial'] == fusion_trial
        return value
    start = time.perf_counter()
    p = engine.contract['rolling_protocol'][window]
    cutoff = guard(p['outer_validation'])
    inner = train_inner_once(engine, window, standalone_trial, family)
    # Materialize the signal before opening a scorer DB so representation errors
    # cannot leave a partly populated prediction database.
    common.signal_path(family, cutoff, engine)
    metadata = root/'outer_metrics.json'
    if metadata.exists():
        fit = read(metadata)
        assert fit['families'] == [family] and fit['features'] == engine.features+FEATURES[family]
        assert fit['rounds'] == inner['best_iteration'] and fit['cutoff'] == cutoff
    else:
        if (root/'evaluation.duckdb').exists():
            raise RuntimeError('Incomplete outer score database; preserve for explicit engineering review')
        common.budget(3)
        fit = engine.train_outer(window, standalone_trial, [family], FEATURES[family], inner)
    original = common.FRESH if engine.year == 2019 else common.OLD
    prefix = 'FRESH' if engine.year == 2019 else 'WV2'
    fusion_root = ART/fusion_trial/window
    fusion_root.mkdir(parents=True, exist_ok=True)
    if (fusion_root/'evaluation.duckdb').exists():
        raise RuntimeError('Incomplete fusion review; preserve existing database')
    with connection() as con:
        for alias, source in [('b', original/f'{prefix}-000'/window),
                              ('n', original/f'{prefix}-501'/window), ('x', root)]:
            con.execute(f'ATTACH {literal(source/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
        for alias in ('n', 'x'):
            errors = con.execute(f'''SELECT count(*) FROM b.predictions b FULL JOIN {alias}.predictions n USING(customer_id,article_id)
                WHERE b.customer_id IS NULL OR n.customer_id IS NULL OR b.target<>n.target
                OR b.candidate_rank<>n.candidate_rank OR b.user_history_events_12w<>n.user_history_events_12w''').fetchone()[0]
            assert errors == 0
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id
            FROM read_parquet({literal(engine.transactions)}) WHERE t_dat>=DATE '{cutoff}'
            AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        population_error = con.execute('''SELECT count(*) FROM
            (SELECT DISTINCT customer_id FROM truth) t FULL JOIN
            (SELECT DISTINCT customer_id FROM x.predictions) p USING(customer_id)
            WHERE t.customer_id IS NULL OR p.customer_id IS NULL''').fetchone()[0]
        label_error = con.execute('''SELECT count(*) FROM x.predictions p LEFT JOIN truth t USING(customer_id,article_id)
            WHERE p.target<>CASE WHEN t.article_id IS NULL THEN 0 ELSE 1 END''').fetchone()[0]
        duplicates = con.execute('SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM x.predictions').fetchone()[0]
        assert population_error == label_error == duplicates == 0
        con.execute('''CREATE TEMP TABLE merged AS SELECT b.* EXCLUDE(score), b.score score_base,
            n.score score_bpr, x.score score_expert, t.truth_count
            FROM b.predictions b JOIN n.predictions n USING(customer_id,article_id)
            JOIN x.predictions x USING(customer_id,article_id)
            JOIN (SELECT customer_id,count(*) truth_count FROM truth GROUP BY customer_id) t USING(customer_id)''')
        rows, users = con.execute('SELECT count(*),count(DISTINCT customer_id) FROM merged').fetchone()
        assert rows == fit['evaluation']['candidate_rows']
        table = rank_bundle(con, 'merged')
        review, aps = diagnose(con, table, users)
        assert abs(review['systems']['standalone']['covered_map']-fit['evaluation']['map@12']) < 1e-12
        con.execute(f'ATTACH {literal(fusion_root/"evaluation.duckdb")} AS output')
        con.execute(f'''CREATE TABLE output.predictions AS SELECT customer_id,article_id,candidate_rank,
            target,user_history_events_12w,fusion_score score FROM {table}''')
        con.execute(f'''CREATE TABLE output.top12 AS SELECT * FROM (SELECT *,{final_rank_sql()} final_rank
            FROM output.predictions) WHERE final_rank<=12''')
        top = con.execute('SELECT customer_id,article_id,final_rank FROM output.top12 ORDER BY customer_id,final_rank').fetchdf()
        rawtruth = con.execute('SELECT * FROM truth ORDER BY customer_id,article_id').fetchdf()
        truthsets = {u: set(g.article_id) for u, g in rawtruth.groupby('customer_id', sort=False)}
        predicted = {u: g.article_id.tolist() for u, g in top.groupby('customer_id', sort=False)}
        assert set(predicted) == set(truthsets) and all(len(v) == len(set(v)) == 12 for v in predicted.values())
        independent = float(np.mean([apk(list(t), predicted[u]) for u, t in truthsets.items()]))
        assert abs(independent-review['systems']['fusion']['covered_map']) < 1e-12
        inactive_errors = con.execute('''SELECT count(*) FROM
            (SELECT customer_id,article_id,final_rank FROM output.top12 WHERE user_history_events_12w=0) f
            FULL JOIN (SELECT customer_id,article_id,final_rank FROM x.top12 WHERE user_history_events_12w=0) b
            USING(customer_id,article_id,final_rank) WHERE f.customer_id IS NULL OR b.customer_id IS NULL''').fetchone()[0]
        assert inactive_errors == 0
        candidate = con.execute('''SELECT avg(hits*1.0/truth_count),avg(least(hits,12)*1.0/least(truth_count,12))
            FROM (SELECT customer_id,sum(target) hits,max(truth_count) truth_count FROM merged GROUP BY customer_id)''').fetchone()
        baseline_map = float(aps.ap_rrf.mean())
        expected = (read(common.ROOT/'reports/warm_v2/WV2-601_OUTER.json')['per_window_MAP'][window]
                    if engine.year == 2020 else read(common.ROOT/'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS.json')['windows'][window]['systems']['FRESH-601']['map@12'])
        assert abs(expected-baseline_map) < 1e-12
        top_path = fusion_root/'top12.parquet'
        con.execute(f'COPY (SELECT * FROM output.top12 ORDER BY customer_id,final_rank) TO {literal(top_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        rank_path = root/'outer_expert_ranks.parquet'
        con.execute(f'COPY {table} TO {literal(rank_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    save_parquet(aps, root/'outer_user_ap.parquet')
    result = {'family': family, 'standalone_trial': standalone_trial, 'fusion_trial': fusion_trial,
        'window': window, 'year': engine.year, 'cutoff': cutoff, 'fit': fit, **review,
        'candidate_rows': rows, 'truth_pairs': len(rawtruth), 'candidate_recall': candidate[0],
        'candidate_oracle_map@12': candidate[1], 'baseline_full_MAP': baseline_map,
        'raw_population_errors': population_error, 'raw_label_errors': label_error,
        'candidate_identity_errors': 0, 'duplicate_pairs': duplicates, 'inactive_top12_errors': inactive_errors,
        'independent_fusion_ap_error': abs(independent-review['systems']['fusion']['covered_map']),
        'fusion_top12': evidence_id(top_path, reason='explicit_registry_evidence'),
        'ranks_path': str(rank_path), 'runtime_seconds': time.perf_counter()-start, 'final_week': 'not_run'}
    write(reviewed, result)
    return result


def confirm(family, standalone_trial, fusion_trial, chosen_ordering='fusion', year=2020):
    """Confirm one preselected ordering; alternate ordering is mechanism control."""
    common.setup()
    selected_trial = verify_outer_permission(family, standalone_trial, fusion_trial, chosen_ordering, year)
    destination = report_path(selected_trial, 'OUTER', year)
    if destination.exists():
        prior = read(destination)
        assert prior['chosen_ordering'] == chosen_ordering and prior['family'] == family
        return prior
    common.budget(10)
    engine = common.Engine(year)
    start = time.perf_counter()
    # In 2019 this only obtains cutoff-safe inner rounds under the same model;
    # no architecture, ordering or hyperparameter is selected from replay scores.
    if year == 2019:
        screen(family, standalone_trial, fusion_trial, year=2019)
    windows = {w: outer_window(engine, w, family, standalone_trial, fusion_trial)
               for w in engine.contract['rolling_protocol']}
    maps = {w: r['systems'][chosen_ordering]['covered_map'] for w, r in windows.items()}
    alternative = 'standalone' if chosen_ordering == 'fusion' else 'fusion'
    result = {'experiment_id': selected_trial, 'family': family, 'year': year,
        'chosen_ordering': chosen_ordering, 'producer_model_trial': standalone_trial,
        'windows': windows, **common.summary(maps, year),
        'mechanism_control': {'ordering': alternative,
            'per_window_MAP': {w: r['systems'][alternative]['covered_map'] for w, r in windows.items()},
            'posthoc_promotion_allowed': False},
        'candidate_pool_changed': False, 'new_tree_feature_count': 86,
        'fusion': 'equal three-way rank RRF, fixed60, zero additional fusion fits',
        'runtime_seconds': time.perf_counter()-start, 'no_registry_mutation': True,
        'final_week': 'not_run'}
    write(destination, result)
    # Separate control file is explicitly not a registered promotion candidate.
    alternate_trial = standalone_trial if alternative == 'standalone' else fusion_trial
    write(report_path(alternate_trial, 'OUTER_CONTROL', year), {
        'experiment_id': alternate_trial, 'family': family, 'year': year,
        'ordering': alternative, 'role': 'mechanism_control_only', 'posthoc_promotion_allowed': False,
        'selected_trial': selected_trial, 'per_window_MAP': result['mechanism_control']['per_window_MAP'],
        'windows': windows, 'final_week': 'not_run'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['screen', 'confirm'])
    parser.add_argument('--family', choices=list(FEATURES), required=True)
    parser.add_argument('--standalone-trial', required=True)
    parser.add_argument('--fusion-trial', required=True)
    parser.add_argument('--ordering', choices=ORDERINGS, default='fusion')
    parser.add_argument('--year', type=int, choices=[2019, 2020], default=2020)
    args = parser.parse_args()
    try:
        kwargs = {'family': args.family, 'standalone_trial': args.standalone_trial,
            'fusion_trial': args.fusion_trial, 'year': args.year}
        if args.command == 'confirm':
            kwargs['chosen_ordering'] = args.ordering
        result = globals()[args.command](**kwargs)
        print({'command': args.command, 'family': args.family, 'year': args.year,
            'completed': True}, flush=True)
    except Exception:
        write(ART/f'EXPERT_FAILURE_{time.time_ns()}.json', {'created_at': now(),
            'family': args.family, 'stage': args.command, 'traceback': traceback.format_exc(),
            'final_week': 'not_run', 'algorithm_rejection': False})
        raise
