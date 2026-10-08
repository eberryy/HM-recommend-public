"""CPU-only mechanics pilot for B; reads the original server bundle unchanged.

No training, submission export, label evaluation, or automatic full-scale run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rank_scores(frame):
    """Exact E0/E1 RRF60, with frozen candidate-rank/article-id tie breaks."""
    frame = frame.copy()
    for score, rank in [('score_base', 'r0'), ('score_bpr', 'r1')]:
        ordered = frame.sort_values(
            ['customer_id', score, 'candidate_rank', 'article_id'],
            ascending=[True, False, True, True])
        frame[rank] = ordered.groupby('customer_id', sort=False).cumcount() + 1
    frame['rrf_score'] = 1.0 / (60 + frame.r0) + 1.0 / (60 + frame.r1)
    frame = frame.sort_values(['customer_id', 'rrf_score', 'candidate_rank', 'article_id'],
                             ascending=[True, False, True, True])
    frame['output_rank'] = frame.groupby('customer_id', sort=False).cumcount() + 1
    return frame


def run(args):
    # Limit native thread pools before importing NumPy or the frozen runtime.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[name] = str(args.threads)
    import numpy as np
    import pandas as pd
    import lightgbm as lgb

    bundle, old, raw, out = [p.resolve() for p in
                            [args.bundle_dir, args.original_run, args.data_dir, args.output_dir]]
    for protected in [bundle, old, raw]:
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out),
                'Output must be disjoint from all original input directories')
    require(not out.exists(), 'Output already exists; preserve it and select a new output directory')
    batch_mode = getattr(args, 'batch_mode', False)
    offset = getattr(args, 'offset', 0)
    require(1 <= args.users <= (8192 if batch_mode else 2048), 'User batch exceeds bounded limit')
    require(offset >= 0 and (batch_mode or offset == 0), 'Offset requires explicit batch mode')
    require(1 <= args.threads <= 8, 'Pilot limited to at most 8 threads')
    sys.path.insert(0, str(bundle / 'src'))
    spec = importlib.util.spec_from_file_location('original_wv3', bundle / 'scripts/run_wv3_741_kaggle.py')
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    r, f = runtime, runtime.frozen
    out.mkdir(parents=True)
    started = time.perf_counter()
    timings = {}
    contract = {
        'stage': 'B-INACTIVE-CPU-BATCH-v1' if batch_mode else 'B-INACTIVE-CPU-PILOT-v1', 'cutoff': r.CUTOFF,
        'users': args.users, 'offset': offset, 'sample': 'lexical inactive customer IDs, OFFSET offset LIMIT users',
        'purpose': 'mechanics and timing only; not a representative MAP estimate',
        'candidate_budget': 100, 'global_statistics': 'all pre-cutoff transactions',
        'change': 'inactive original candidate order -> E0/E1 equal RRF60',
        'training': False, 'labels_read': False, 'submission_export': False,
        'active_recommendations_changed': False, 'automatic_full_scale': False,
        'threads': args.threads, 'duckdb_memory_limit': '32GB',
        'paths': {'bundle': str(bundle), 'original_run': str(old), 'raw': str(raw)},
        'stop': 'any identity, cutoff, candidate, feature, score or replay mismatch; external timeout 3600s',
    }
    r.write_json(out / 'CONTRACT.json', contract)
    try:
        freeze = r.read_json(bundle / 'FREEZE_MANIFEST.json')
        previous = r.read_json(old / 'RUN_MANIFEST.json')
        bpr_dir = old / 'artifacts/bpr'
        bpr_meta = r.read_json(bpr_dir / 'MODEL.json')
        require(freeze['inference_cutoff'] == previous['cutoff'] == bpr_meta['cutoff'] == r.CUTOFF,
                'Cutoff mismatch')
        require(previous['status'] == 'completed', 'Original run is incomplete')
        require(bpr_meta['latest_history_date'] == '2020-09-22' and bpr_meta['params'] == f.BPR_PARAMS,
                'BPR date/parameter mismatch')
        # Historical remote reuse: compare model files to trusted packaged digests.
        for relative, expected in r.MODEL_SHA256.items():
            if relative.startswith(('models/wv2_base/', 'models/wv2_bpr/')):
                require(r.sha256(bundle / relative) == expected, 'Frozen model identity mismatch: ' + relative)
        layout = r.Layout(raw, out)
        r.CFG.update(layout=layout, threads=args.threads, memory_limit='32GB')
        r.configure_frozen_runtime(layout, args.users)
        tx = old / 'work/input/transactions.parquet'
        f.TX = tx
        # Feature builders are reused exactly; replace only IO and the old GPU gate.
        f.resource_gate = lambda: None

        def feature_connection(raw_dir, work_dir):
            work_dir.mkdir(parents=True, exist_ok=True)
            db = r.tuned_connection(work_dir / 'feature-build.duckdb')
            db.execute(f"CREATE VIEW transactions AS SELECT * FROM read_parquet('{r.sql(tx)}') WHERE t_dat<DATE '{r.CUTOFF}'")
            for table, filename in r.audit.TABLE_FILES.items():
                if table != 'transactions':
                    db.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv_auto('{r.sql(raw_dir / filename)}',header=true,all_varchar=true)")
            return db

        r.m2.prepare_tabular_connection = feature_connection
        old_inactive = old / 'artifacts/m1-active/inactive-top12.parquet'
        stage = time.perf_counter()
        with feature_connection(raw, out / 'retrieval-work') as db:
            stats = db.execute(f"SELECT count(*),min(t_dat),max(t_dat) FROM read_parquet('{r.sql(tx)}')").fetchone()
            require(tuple(map(str, stats)) == ('31788324', '2018-09-20', '2020-09-22'), 'Transaction identity mismatch')
            db.execute(f"CREATE TEMP VIEW old_inactive AS SELECT * FROM read_parquet('{r.sql(old_inactive)}')")
            require(db.execute('SELECT count(DISTINCT customer_id) FROM old_inactive').fetchone()[0] == 882427,
                    'Inactive population mismatch')
            db.execute(f"CREATE TEMP TABLE m1_users AS SELECT DISTINCT customer_id FROM old_inactive ORDER BY customer_id LIMIT {args.users} OFFSET {offset}")
            require(db.execute(f"SELECT count(*) FROM transactions SEMI JOIN m1_users USING(customer_id) WHERE t_dat>=DATE '{r.CUTOFF}'-INTERVAL 12 WEEK").fetchone()[0] == 0,
                    'Pilot includes a recent-active user')
            db.execute('CREATE TEMP TABLE m1_warm_catalog AS SELECT DISTINCT article_id FROM transactions')
            db.execute('CREATE TEMP TABLE m1_eligible_catalog AS SELECT DISTINCT article_id FROM articles')
            config = r.m1.M1Config(cutoff=r.CUTOFF, sample_rate=1., fusion_profile='collaborative')
            counts = {}
            for name, builder in [
                ('repurchase', r.m1._create_repurchase),
                ('recent_popularity', r.m1._create_recent_popularity),
                ('product_family', r.m1._create_product_family),
                ('attribute_content', r.m1._create_attribute_content),
                ('user_day_covisit', r.m1._create_covisit),
                ('age_popularity', r.m1._create_age_popularity),
            ]:
                t = time.perf_counter()
                builder(db, "DATE '2020-09-23'", config)
                counts[name] = db.execute(f'SELECT count(*) FROM {r.m15.SOURCE_TABLES[name]}').fetchone()[0]
                timings[name] = time.perf_counter() - t
                r.progress(f'{name}: {counts[name]} rows, {timings[name]:.1f}s')
            require(all(counts[n] == 0 for n in ['repurchase', 'product_family', 'attribute_content', 'user_day_covisit']),
                    'Unexpected behavior-seeded candidates for inactive users')
            r.m15._create_candidates_long(db)
            r.m15._create_collaborative_fusion(db, config)
            db.execute('CREATE TEMP VIEW m16_reference AS SELECT * FROM m1_candidates')
            for name, table in r.m15.SOURCE_TABLES.items():
                db.execute(f'CREATE TEMP VIEW m16_src_{name} AS SELECT * FROM {table}')
            db.execute(r.m16._direct_wide_sql('collaborative', 60))
            mismatch = db.execute('''WITH a AS (SELECT customer_id,article_id,candidate_rank rank
                FROM m1_candidates WHERE candidate_rank<=12), b AS (
                SELECT customer_id,article_id,fallback_rank rank FROM old_inactive SEMI JOIN m1_users USING(customer_id))
                SELECT count(*) FROM a FULL JOIN b USING(customer_id,rank)
                WHERE a.article_id IS DISTINCT FROM b.article_id''').fetchone()[0]
            require(mismatch == 0, 'Rebuilt original Top12 differs from old fallback')
            source_root = old / 'artifacts/item2vec-source'
            source_meta = r.read_json(source_root / 'source-manifest.json')
            require(source_meta['cutoff'] == r.CUTOFF, 'Item2Vec vocabulary cutoff mismatch')
            vocab = Path(source_meta['artifacts']['items']['path'])
            require(vocab.is_file() and vocab.resolve().is_relative_to(source_root.resolve()), 'Vocabulary outside verified source directory')
            db.execute(f"CREATE TEMP TABLE vocab AS SELECT article_id,token_count FROM read_csv_auto('{r.sql(vocab)}',header=true)")
            f.EXPANDED.parent.mkdir(parents=True, exist_ok=True)
            # Inactive users have no recent seeds; vocabulary counts still apply.
            query = f'''SELECT b.customer_id,b.article_id,{r.m29._base_retrieval_projection('b')},
                0::INTEGER item2vec_present,0::INTEGER item2vec_is_new,
                NULL::BIGINT item2vec_rank,NULL::DOUBLE item2vec_score,NULL::DOUBLE item2vec_cosine,
                NULL::BIGINT item2vec_best_seed_rank,NULL::BIGINT item2vec_best_neighbor_rank,
                NULL::BIGINT item2vec_seed_support,coalesce(v.token_count,0)::BIGINT item2vec_vocab_count
                FROM m16_direct_wide b LEFT JOIN vocab v USING(article_id)'''
            db.execute(f"COPY ({query} ORDER BY customer_id,candidate_rank,article_id) TO '{r.sql(f.EXPANDED)}' (FORMAT PARQUET)")
            old_top = db.execute('SELECT * FROM old_inactive SEMI JOIN m1_users USING(customer_id)').fetchdf()
        timings['retrieval_total'] = time.perf_counter() - stage
        stage = time.perf_counter()
        features = f.build_base_features({'rows': args.users * 100})
        timings['features'] = time.perf_counter() - stage
        model0 = lgb.Booster(model_file=str(bundle / 'models/wv2_base/outer_model.txt'))
        model1 = lgb.Booster(model_file=str(bundle / 'models/wv2_bpr/outer_model.txt'))
        names0, names1 = model0.feature_name(), model1.feature_name()
        require(len(names0) == 84 and names1 == names0 + ['wv2_bpr_user_item_score', 'wv2_bpr_unavailable'],
                'Feature contract mismatch')
        maps0 = f.category_maps(bundle / 'models/wv2_base/outer_category_maps.json')
        maps1 = f.category_maps(bundle / 'models/wv2_bpr/outer_category_maps.json')
        with np.load(bpr_dir / 'factors.npz', allow_pickle=False) as data:
            uf, itf = data['user_factors'], data['item_factors']
        with r.tuned_connection() as db:
            users = db.execute(f"SELECT * FROM read_parquet('{r.sql(bpr_dir / 'users.parquet')}') ORDER BY user_index").fetchdf()
            items = db.execute(f"SELECT * FROM read_parquet('{r.sql(bpr_dir / 'items.parquet')}') ORDER BY item_index").fetchdf()
            frame = db.execute(f"SELECT * FROM read_parquet('{r.sql(f.BASE_FEATURES)}')").fetchdf()
        require(np.array_equal(users.user_index, np.arange(len(users))) and len(users) == len(uf), 'BPR user index mismatch')
        require(np.array_equal(items.item_index, np.arange(len(items))) and len(items) == len(itf), 'BPR item index mismatch')
        uindex, iindex = pd.Index(users.customer_id), pd.Index(items.article_id)
        require(uindex.is_unique and iindex.is_unique and uf.shape[1] == itf.shape[1] == 101, 'BPR shape/ID mismatch')

        def score(frame):
            ui, ii = uindex.get_indexer(frame.customer_id), iindex.get_indexer(frame.article_id)
            good = (ui >= 0) & (ii >= 0)
            latent = np.full(len(frame), np.nan, np.float32)
            latent[good] = np.einsum('ij,ij->i', uf[ui[good]], itf[ii[good]])
            require(np.isfinite(latent[good]).all(), 'Nonfinite available BPR score')
            frame = frame.copy()
            frame['wv2_bpr_user_item_score'] = latent
            frame['wv2_bpr_unavailable'] = (~good).astype(np.float32)
            result = frame[['customer_id', 'article_id', 'candidate_rank']].copy()
            result['score_base'] = model0.predict(r.m2._prepare_frame(frame, names0, maps0), num_threads=args.threads)
            result['score_bpr'] = model1.predict(r.m2._prepare_frame(frame, names1, maps1), num_threads=args.threads)
            require(np.isfinite(result[['score_base', 'score_bpr']]).all().all(), 'Nonfinite predictions')
            return result

        require((frame.user_history_events_12w == 0).all(), 'Inactive feature state mismatch')
        stage = time.perf_counter()
        scores = score(frame)
        ranking = rank_scores(scores)
        timings['pilot_scoring_and_ranking'] = time.perf_counter() - stage
        # Independent SQL replay of the pandas ordering.
        with r.tuned_connection() as db:
            db.register('scores', scores)
            replay = db.execute('''WITH ranked AS (SELECT *,
                row_number() OVER(PARTITION BY customer_id ORDER BY score_base DESC,candidate_rank,article_id) r0,
                row_number() OVER(PARTITION BY customer_id ORDER BY score_bpr DESC,candidate_rank,article_id) r1 FROM scores)
                SELECT customer_id,article_id,row_number() OVER(PARTITION BY customer_id
                ORDER BY 1.0/(60+r0)+1.0/(60+r1) DESC,candidate_rank,article_id) output_rank FROM ranked''').fetchdf()
            compare = ranking.merge(replay, on=['customer_id', 'article_id'], suffixes=('', '_sql'), validate='one_to_one')
            require((compare.output_rank == compare.output_rank_sql).all(), 'Independent rank replay failed')
            # Read-only active-score check: same models/transforms versus saved old predictions.
            oldr = old / 'artifacts/ranks-top50.parquet'
            oldf = old / 'artifacts/base-features.parquet'
            db.execute(f"CREATE TEMP TABLE active_keys AS SELECT DISTINCT customer_id FROM read_parquet('{r.sql(oldr)}') ORDER BY customer_id LIMIT 16")
            check = db.execute(f"SELECT f.*,r.score_base saved_base,r.score_bpr saved_bpr FROM read_parquet('{r.sql(oldf)}') f JOIN read_parquet('{r.sql(oldr)}') r USING(customer_id,article_id) SEMI JOIN active_keys USING(customer_id)").fetchdf()
        require(len(check) == 800, 'Active control incomplete')
        control = score(check)
        error = max(float(np.max(abs(control.score_base - check.saved_base))),
                    float(np.max(abs(control.score_bpr - check.saved_bpr))))
        require(error <= 1e-10, 'Original active score replay mismatch')
        top = ranking[ranking.output_rank <= 12].copy()
        require(len(top) == args.users * 12 and top.groupby('customer_id').article_id.nunique().eq(12).all(), 'Top12 invariant failed')
        changed = top.merge(old_top, left_on=['customer_id', 'output_rank'],
                            right_on=['customer_id', 'fallback_rank'], suffixes=('_b', '_a'), validate='one_to_one')
        require(len(changed) == len(top), 'Old/new user-rank coverage mismatch')
        changed_users = changed.loc[changed.article_id_b != changed.article_id_a, 'customer_id'].nunique()
        with r.tuned_connection() as db:
            db.register('ranking', ranking)
            db.execute(f"COPY ranking TO '{r.sql(out / 'pilot-ranking.parquet')}' (FORMAT PARQUET)")
        result = {
            'status': 'completed', 'contract': contract, 'users': args.users,
            'candidate_rows': len(frame), 'source_rows': counts, 'timings_seconds': timings,
            'total_seconds': time.perf_counter() - started,
            'changed_top12_users': int(changed_users), 'original_top12_reconstruction_mismatches': mismatch,
            'active_control_users': 16, 'active_score_max_abs_error': error,
            'independent_rank_replay': 'passed', 'feature_evidence': features,
            'pilot_feature_bytes': f.BASE_FEATURES.stat().st_size,
            'full_scale_started': batch_mode, 'MAP_evaluated': False,
            'scaling_note': 'Global aggregation cost is fixed; do not multiply total pilot duration by full/pilot users.',
        }
        if sys.platform == 'linux':
            import resource
            result['peak_process_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        result['output_file_bytes'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        r.write_json(out / 'RESULT.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        r.write_json(out / 'FAILURE.json', {'status': 'failed', 'type': type(error).__name__,
                     'error': str(error), 'seconds': time.perf_counter() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--original-run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--users', type=int, default=1024)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--batch-mode', action='store_true', help='Bounded batch worker for explicit full-run orchestrator')
    parser.add_argument('--offset', type=int, default=0)
    run(parser.parse_args())
