"""Fixed BPR Top100 append pool; original 84 features and ranking machinery preserved."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import os
from pathlib import Path
import time
import traceback

import duckdb
import numpy as np
import pandas as pd
import torch

from . import warm_v3_common as common
from .warm_v2_contract import read, write, now, guard, evidence_id
from .warm_v2_engine import literal
from .warm_v2_recent_data import raw_views
from .warm_v2_bpr import match_scores
from .warm_v3_retrieval_audit import topk_exact, CUTOFFS as INNER_CUTOFFS
from .m2 import M2Config, RETRIEVAL_FEATURES, _create_static_dimensions, build_point_in_time_dataset
from .m26 import _empty_retrieval_projection
from .m29 import ITEM2VEC_FEATURES, FEATURE_SCHEMA_VERSION
from .m210 import build_target_aware_cache
from .m211 import _distribution_relation_sql, SAMPLING_SEED

ART = common.ART / 'learned_pool'
CONTRACT = ART / 'CONTRACT.json'
K = 100
ALL_CUTOFFS = ('2019-11-27', '2019-12-25', '2020-01-22', '2020-02-19', '2020-03-18',
               '2020-04-29', '2020-05-27', '2020-06-24', '2020-07-22', '2020-08-19')


def allowed(cutoff, approved=INNER_CUTOFFS):
    guard(cutoff)
    if cutoff not in ALL_CUTOFFS or cutoff not in approved:
        raise ValueError(f'Cutoff not explicitly approved for this pool run: {cutoff}')


def contract():
    assert Path.cwd().resolve() == common.ROOT
    assert common.git('branch', '--show-current') == 'warm-v3-architecture-lab'
    if CONTRACT.exists():
        old = read(CONTRACT)
        assert old['retrieval_k'] == K and old['maximum_pool'] == 400
        return old
    common.setup()
    value = {'created_at': now(), 'experiment': 'WV3-401', 'status': 'preregistered_before_pool_build',
        'initial_build_cutoffs': list(INNER_CUTOFFS), 'subsequent_cutoffs_require_explicit_caller_approval': list(ALL_CUTOFFS),
        'retrieval_k': K, 'maximum_pool': 400,
        'candidate_rule': 'Keep exact original100-300 rows/ranks; append unseen-in-pool items from fixed full-historyBPRTop100. New ranks=max(original rank)+contiguous BPR rank order; no padding.',
        'retrieval_population': 'original full10percent truth-user cohort; BPR queries onlyhistory_events_12w>0; keep previously boughtitems',
        'catalog': 'same-cutoffBPRvocabulary intersect originalarticles; all101factorcolumns includingbias; FP32TF32off tiesarticle_id',
        'features': 'same frozen84; copy ALL old numeric values, missingness, labels andcandidate_rank exactly. ConstructnewpairPIT features usingoriginalM2+M210 SQL; discardM33seasonalextras anddecay from model.',
        'six_source_features_new_rows': 'same M26 empty retrieval projection: present0,rank/score/contributionNULL,fused0,source_count0; these represent retainedsixsourceTop100 provenance, not allrawretrievalunion',
        'item2vec_new_rows': 'join same-cutofforiginalrawTop300source anditemvocabulary; retainrank/scores ifpresent; is_new=present andnotinsixsourceTop100; vocab_count availableevenwhen notinusersource',
        'sampler': 'exact original30:1 deterministictwo-strata<=100 versus>100 andranktertile/hash; latterrenamedexpanded_tail, noBPRspecificquota',
        'ranking': 'same84/86LambdaRankandfixedRRF60; samepoolrefits required, originalcutoffprotocol/earlystopping/fallback/denominator',
        'bpr_features': 'oldpairs copiedfromsavedfeatures; newpairs np.einsum ofsame101factorcolumns; noBPRfit',
        'safety': 'own cache namespace; originalassetsreadonly; groups100-400; diskfloor10GiB; noautomaticouterfit/eval',
        'final_week': 'not_run'}
    write(CONTRACT, value)
    return value


def connection():
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='3GB'")
    spill = ART / 'spill' / str(os.getpid())
    spill.mkdir(parents=True, exist_ok=True)
    con.execute(f'SET temp_directory={literal(spill)}')
    return con


def sampled_relation(path):
    # The literal rename preserves partitions, relative source order and every hash.
    return _distribution_relation_sql(path, seed=SAMPLING_SEED).replace("'item2vec_only'", "'expanded_tail'")


def append_relation(base='original', retrieved='retrieved'):
    return f'''WITH n AS (SELECT r.*,row_number() OVER(PARTITION BY customer_id
        ORDER BY retrieval_rank,article_id) appended_rank FROM {retrieved} r
        WHERE NOT EXISTS (SELECT 1 FROM {base} b WHERE b.customer_id=r.customer_id AND b.article_id=r.article_id)),
        maxima AS (SELECT customer_id,max(candidate_rank) old_max FROM {base} GROUP BY customer_id)
        SELECT n.*,CAST(m.old_max+n.appended_rank AS BIGINT) candidate_rank FROM n JOIN maxima m USING(customer_id)'''


def partial_topk(scores, k):
    """Exact top-k indices; input columns already in ascending article-ID order."""
    output = []
    for row in scores:
        if not np.isfinite(row).all() or k > len(row):
            raise ValueError('Invalid score vector')
        threshold = np.partition(row, len(row) - k)[len(row) - k]
        above = np.flatnonzero(row > threshold)
        equal = np.flatnonzero(row == threshold)[:k - len(above)]
        chosen = np.concatenate([above, equal])
        output.append(chosen[np.lexsort((chosen, -row[chosen]))])
    return np.asarray(output, dtype=np.int64)


def cpu_topk_exact(query, item_factors, article_ids, k=K, batch=128):
    """Same FP32 exhaustive inner product, partial exact sorting to bound CPU cost."""
    order = np.argsort(article_ids, kind='stable')
    ids = np.asarray(article_ids)[order]
    output, values = [], []
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        factors_tensor = torch.as_tensor(np.ascontiguousarray(item_factors[order], dtype=np.float32))
        with torch.no_grad():
            for start in range(0, len(query), batch):
                q = torch.as_tensor(np.ascontiguousarray(query[start:start + batch], dtype=np.float32))
                scores = (q @ factors_tensor.T).numpy()
                selected = partial_topk(scores, k)
                output.append(ids[selected])
                values.append(np.take_along_axis(scores, selected, axis=1))
    finally:
        torch.set_num_threads(before)
    return np.concatenate(output), np.concatenate(values)


def factors(cutoff, original):
    bp, bm = common.bpr_path(cutoff)
    root = bp.parent
    mm = read(root / 'model.json')
    assert mm['cutoff'] == cutoff and mm['latest_history_date'] < cutoff
    assert mm['params']['factors'] == 100 and mm['params']['iterations'] == 100
    assert bm['source_sha256'] == original.history['feature_cache'][cutoff]['artifact']['sha256']
    for name, value in mm['artifacts'].items():
        assert (root / name).stat().st_size == value['bytes']
    with connection() as con:
        users = con.execute(f'SELECT * FROM read_parquet({literal(root / "users.parquet")}) ORDER BY user_index').fetchdf()
        items = con.execute(f'SELECT * FROM read_parquet({literal(root / "items.parquet")}) ORDER BY item_index').fetchdf()
    assert np.array_equal(users.user_index, np.arange(mm['users']))
    assert np.array_equal(items.item_index, np.arange(mm['items']))
    with np.load(root / 'factors.npz', allow_pickle=False) as z:
        uf, itf = z['user_factors'], z['item_factors']
    assert uf.shape == (len(users), 101) and itf.shape == (len(items), 101)
    return bp, bm, mm, users, items, uf, itf


def retrieve(cutoff, original, destination, device):
    if destination.exists():
        return destination
    prior = common.ART / 'bpr_retrieval' / cutoff / 'AUDIT.json'
    if prior.exists():
        audit = read(prior)
        assert audit['cutoff'] == cutoff and audit['retrieval_k'] == K and audit['score_parity']['passed']
        assert audit['source']['sha256'] == original.history['feature_cache'][cutoff]['artifact']['sha256']
        return Path(audit['output']['path'])
    _, _, _, users, items, uf, itf = factors(cutoff, original)
    with connection() as con:
        active = con.execute(f'''SELECT customer_id FROM read_parquet({literal(original.base_path(cutoff))})
            GROUP BY customer_id HAVING max(user_history_events_12w)>0 ORDER BY customer_id''').fetchdf()
        articles = con.execute(f'''SELECT article_id FROM read_csv({literal(common.SHARED / 'data/raw/articles.csv')},
            header=true,all_varchar=true)''').fetchdf()
        selected = items[items.article_id.isin(set(articles.article_id))].sort_values('article_id')
        idx = pd.Index(users.customer_id).get_indexer(active.customer_id)
        assert (idx >= 0).all()
        if device == 'cpu':
            ids, scores = cpu_topk_exact(uf[idx], itf[selected.item_index.to_numpy()], selected.article_id.to_numpy(), k=K)
        else:
            ids, scores = topk_exact(uf[idx], itf[selected.item_index.to_numpy()], selected.article_id.to_numpy(), k=K, device=device)
        frame = pd.DataFrame({'customer_id': np.repeat(active.customer_id.to_numpy(), K),
            'article_id': ids.reshape(-1), 'bpr_score': scores.reshape(-1),
            'retrieval_rank': np.tile(np.arange(1, K + 1), len(active))})
        con.register('out', frame)
        con.execute(f'COPY out TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    del uf, itf
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return destination


def build(cutoff, original, approved=INNER_CUTOFFS, device='cuda'):
    allowed(cutoff, approved)
    contract()
    common.budget(5)
    root = ART / cutoff
    ready = root / 'BUILD.json'
    if ready.exists():
        result = read(ready)
        assert result['cutoff'] == cutoff and result['old_features_exact']
        assert Path(result['artifact']['path']).stat().st_size == result['artifact']['bytes']
        return result
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    source = original.base_path(cutoff)
    old_identity = original.history['feature_cache'][cutoff]['artifact']
    prereq = original.history['prerequisite_cache']['item2vec_sources'][cutoff]['artifacts']
    retrieved = retrieve(cutoff, original, root / 'bpr_top100.parquet', device)
    cp = root / 'candidate_features.parquet'
    extras = RETRIEVAL_FEATURES + ITEM2VEC_FEATURES
    if not cp.exists():
        with connection() as con:
            con.execute(f'CREATE VIEW original AS SELECT * FROM read_parquet({literal(source)})')
            con.execute(f'CREATE VIEW retrieved AS SELECT * FROM read_parquet({literal(retrieved)})')
            assert con.execute('''SELECT count(*) FROM (SELECT customer_id,count(*) n,count(DISTINCT article_id) d
                FROM retrieved GROUP BY customer_id) WHERE n<>100 OR d<>100''').fetchone()[0] == 0
            con.execute('CREATE TEMP TABLE bpr_new AS ' + append_relation())
            con.execute(f'COPY bpr_new TO {literal(root / "new_pairs.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
            con.execute(f'CREATE VIEW i2v AS SELECT * FROM read_parquet({literal(prereq["candidates"]["path"])})')
            con.execute(f'''CREATE VIEW vocab AS SELECT article_id,CAST(token_count AS BIGINT) token_count
                FROM read_csv({literal(prereq['items']['path'])},header=true,all_varchar=true)''')
            sql = f'''SELECT customer_id,article_id,{','.join(extras)},0::INTEGER bpr_is_new FROM original
                UNION ALL BY NAME
                SELECT n.customer_id,n.article_id,n.candidate_rank,{_empty_retrieval_projection()},
                    CAST(i.article_id IS NOT NULL AS INTEGER) item2vec_present,
                    CAST(i.article_id IS NOT NULL AS INTEGER) item2vec_is_new,
                    i.item2vec_rank,i.item2vec_score,i.item2vec_cosine,
                    i.best_seed_rank item2vec_best_seed_rank,i.best_neighbor_rank item2vec_best_neighbor_rank,
                    i.seed_support item2vec_seed_support,coalesce(v.token_count,0)::BIGINT item2vec_vocab_count,
                    1::INTEGER bpr_is_new
                FROM bpr_new n LEFT JOIN i2v i USING(customer_id,article_id) LEFT JOIN vocab v USING(article_id)'''
            con.execute(f'COPY ({sql} ORDER BY customer_id,candidate_rank,article_id) TO {literal(cp)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    export_root = ART / 'm2' / cutoff
    export_root.mkdir(parents=True, exist_ok=True)
    m2path = export_root / 'features.parquet'
    m2meta = export_root / 'feature-manifest.json'
    if not m2meta.exists():
        with connection() as con:
            raw_views(con, original)
            _create_static_dimensions(con)
            rows = int(con.execute(f'SELECT count(*) FROM read_parquet({literal(cp)})').fetchone()[0])
            evidence = build_point_in_time_dataset(con, cp, {'cutoff': cutoff, 'declared_rows': rows}, m2path,
                M2Config(**{**original.contract['config'], 'candidate_k': 400}),
                retrieval_features=extras + ['bpr_is_new'], candidate_group_range=(100, 400))
        evidence.update(schema_version=FEATURE_SCHEMA_VERSION, item2vec_features=ITEM2VEC_FEATURES)
        write(m2meta, evidence)
    target = build_target_aware_cache(raw_dir=common.SHARED / 'data/raw', transactions_path=common.TX,
        m29_cache_dir=ART / 'm2', cache_dir=ART / 'target', cutoffs=(cutoff,))[cutoff]
    assert target['inputs']['transactions']['sha256'] == original.history['inputs']['transactions']['sha256']
    expected_articles = original.history['prerequisite_cache']['target_features'][cutoff]['inputs']['articles']['sha256']
    assert target['inputs']['articles']['sha256'] == expected_articles
    final = root / 'features.parquet'
    columns = ['customer_id', 'article_id', *original.features, 'target']
    with connection() as con:
        con.execute(f'CREATE VIEW original AS SELECT * FROM read_parquet({literal(source)})')
        con.execute(f'CREATE VIEW generated AS SELECT * FROM read_parquet({literal(target["artifact"]["path"])})')
        types = dict((r[0], r[1]) for r in con.execute('DESCRIBE original').fetchall())
        projection = ','.join(columns)
        casts = ','.join(f'CAST({col} AS {types[col]}) AS {col}' for col in columns)
        if not final.exists():
            con.execute(f'''COPY (SELECT {projection},0::INTEGER bpr_is_new FROM original
                UNION ALL SELECT {casts},1::INTEGER bpr_is_new FROM generated WHERE bpr_is_new=1)
                TO {literal(final)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        con.execute(f'CREATE VIEW output AS SELECT * FROM read_parquet({literal(final)})')
        mismatch = ' OR '.join(f'o.{col} IS DISTINCT FROM n.{col}' for col in columns[2:])
        assert con.execute(f'''SELECT count(*) FROM original o LEFT JOIN output n USING(customer_id,article_id)
            WHERE n.customer_id IS NULL OR {mismatch}''').fetchone()[0] == 0
        assert con.execute('''SELECT count(*) FROM (SELECT customer_id,count(*) n,count(DISTINCT article_id) d,
            min(candidate_rank) lo,max(candidate_rank) hi,count(DISTINCT candidate_rank) nr FROM output GROUP BY customer_id)
            WHERE n<>d OR n<100 OR n>400 OR lo<>1 OR hi<>n OR nr<>n''').fetchone()[0] == 0
        rows, users, positives, new_rows, raw_i2v_new = con.execute('''SELECT count(*),count(DISTINCT customer_id),sum(target),
            sum(bpr_is_new),count(*) FILTER(WHERE bpr_is_new=1 AND item2vec_present=1) FROM output''').fetchone()
        old_rows = int(con.execute('SELECT count(*) FROM original').fetchone()[0])
        assert rows == old_rows + new_rows
        assert con.execute('''SELECT count(*) FROM generated g JOIN output o USING(customer_id,article_id)
            WHERE g.target<>o.target OR g.user_history_events_12w<>o.user_history_events_12w''').fetchone()[0] == 0
    bp, bm, mm, busers, bitems, uf, itf = factors(cutoff, original)
    bpr_final = root / 'bpr_features.parquet'
    with connection() as con:
        nf = con.execute(f'SELECT customer_id,article_id FROM read_parquet({literal(final)}) WHERE bpr_is_new=1').fetchdf()
        values, missing = match_scores(nf, pd.Index(busers.customer_id), pd.Index(bitems.article_id), uf, itf)
        assert (missing == 0).all() and np.isfinite(values).all()
        nf['wv2_bpr_user_item_score'] = values
        nf['wv2_bpr_unavailable'] = missing
        con.register('new_bpr', nf)
        if not bpr_final.exists():
            con.execute(f'''COPY (SELECT * FROM read_parquet({literal(bp)}) UNION ALL BY NAME SELECT * FROM new_bpr)
                TO {literal(bpr_final)} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        assert con.execute(f'SELECT count(*) FROM read_parquet({literal(bpr_final)})').fetchone()[0] == rows
        assert con.execute(f'''SELECT count(*) FROM read_parquet({literal(bp)}) a
            LEFT JOIN read_parquet({literal(bpr_final)}) b USING(customer_id,article_id)
            WHERE b.customer_id IS NULL OR a.wv2_bpr_user_item_score IS DISTINCT FROM b.wv2_bpr_user_item_score
            OR a.wv2_bpr_unavailable IS DISTINCT FROM b.wv2_bpr_unavailable''').fetchone()[0] == 0
    artifact = evidence_id(final, reason='explicit_registry_evidence')
    bpr_meta = {'cutoff': cutoff, 'features': ['wv2_bpr_user_item_score', 'wv2_bpr_unavailable'],
        'source_sha256': artifact['sha256'], 'source': str(final), 'rows': int(rows),
        'latest_history_date': mm['latest_history_date'], 'pit_safe': True, 'future_labels_used': False,
        'original_bpr_features': str(bp), 'old_pairs_exact': True, 'new_pairs_numpy_dot_101': True,
        'artifact': evidence_id(bpr_final, reason='explicit_registry_evidence'), 'final_week': 'not_run'}
    write(root / 'bpr_features.json', bpr_meta)
    result = {'cutoff': cutoff, 'contract': str(CONTRACT), 'artifact': artifact,
        'audit': {'source_rows': int(rows), 'users': int(users), 'positive_pairs': int(positives),
                  'old_rows': old_rows, 'new_rows': int(new_rows), 'new_bpr_also_in_raw_item2vec': int(raw_i2v_new)},
        'original_source': old_identity, 'old_features_exact': True, 'old_bpr_exact': True,
        'candidate_protocol': 'old100-300 + fixedTop100BPRonly; max400',
        'runtime_seconds': time.perf_counter() - started, 'new_bpr_model_fits': 0, 'final_week': 'not_run'}
    write(ready, result)
    del uf, itf, nf
    gc.collect()
    print({'pool': cutoff, 'rows': rows, 'new_rows': new_rows, 'seconds': result['runtime_seconds'], 'old84_exact': True}, flush=True)
    return result


class PoolEngine(common.Engine):
    def __init__(self, allowed_cutoffs=INNER_CUTOFFS, device='cuda'):
        if not set(allowed_cutoffs) <= set(ALL_CUTOFFS):
            raise ValueError('PoolEngine does not authorize new/final cutoffs')
        super().__init__(2020)
        self.original = common.Engine(2020)
        self.allowed_cutoffs = tuple(allowed_cutoffs)
        self.device = device
        self.contract = deepcopy(self.contract)
        self.contract['config']['candidate_k'] = 400
        contract()

    def base_path(self, cutoff):
        allowed(cutoff, self.allowed_cutoffs)
        result = build(cutoff, self.original, self.allowed_cutoffs, self.device)
        self.history['feature_cache'][cutoff] = result
        return Path(result['artifact']['path'])

    def signal_path(self, family, cutoff):
        if family != 'bpr_match':
            return None
        self.base_path(cutoff)
        return ART / cutoff / 'bpr_features.parquet', read(ART / cutoff / 'bpr_features.json')

    def cached_data(self, cutoff, role):
        allowed(cutoff, self.allowed_cutoffs)
        source_path = self.base_path(cutoff)
        destination = ART / 'frozen_data' / cutoff / f'{role}.parquet'
        meta = destination.with_suffix('.json')
        if destination.exists() and meta.exists():
            result = read(meta)
            assert result['source_sha256'] == self.history['feature_cache'][cutoff]['artifact']['sha256']
            return destination, result
        destination.parent.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(['customer_id', 'article_id', *self.features, 'target', 'user_history_events_12w']))
        source = f'read_parquet({literal(source_path)})'
        with connection() as con:
            if role == 'sample':
                query = f"SELECT {','.join(columns)} FROM ({sampled_relation(source_path)})"
            elif role == 'inner':
                query = f'''WITH eligible AS (SELECT customer_id FROM {source} GROUP BY customer_id
                    HAVING sum(target)>0 AND max(user_history_events_12w)>0), truth AS (
                    SELECT customer_id,count(DISTINCT article_id)::BIGINT truth_count FROM read_parquet({literal(common.TX)})
                    WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY GROUP BY customer_id)
                    SELECT {','.join('s.' + c for c in columns)},t.truth_count FROM {source} s
                    JOIN eligible USING(customer_id) JOIN truth t USING(customer_id)'''
            else:
                raise ValueError(role)
            frame = con.execute(query + ' ORDER BY customer_id,candidate_rank,article_id').fetchdf()
            groups = frame.groupby('customer_id', sort=False).target.agg(['size', 'sum'])
            assert groups['sum'].gt(0).all() and groups['size'].between(2, 400).all()
            assert not frame.duplicated(['customer_id', 'article_id']).any()
            if role == 'sample':
                assert int((frame.target == 0).sum()) <= 30 * int(frame.target.sum())
            else:
                assert frame.user_history_events_12w.gt(0).all()
            con.register('out', frame)
            con.execute(f'COPY out TO {literal(destination)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        audit = self.history['feature_cache'][cutoff]['audit']
        stats = {'cutoff': cutoff, 'role': role, 'rows': len(frame), 'groups': len(groups),
            'positive_rows': int(frame.target.sum()), 'source_rows': audit['source_rows'], 'source_users': audit['users'],
            'min_group_rows': int(groups['size'].min()), 'max_group_rows': int(groups['size'].max()),
            'source_path': str(source_path), 'source_sha256': self.history['feature_cache'][cutoff]['artifact']['sha256'],
            'candidate_feature_contract': 'same frozen84; originalrows exact; fixedBPRappend; two-strata30:1 expanded_tail',
            'artifact': evidence_id(destination, reason='explicit_registry_evidence')}
        write(meta, stats)
        return destination, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--cutoff', choices=INNER_CUTOFFS)
    args = parser.parse_args()
    try:
        common.setup()
        registered = read(common.REGISTRY)
        assert any(t['experiment_id'] == 'WV3-401' for t in registered['trials']), 'central preregistration required'
        engine = PoolEngine(device=args.device)
        for cutoff in ([args.cutoff] if args.cutoff else INNER_CUTOFFS):
            engine.base_path(cutoff)
        print('Original INNER feature pools complete; no ranker fit or outer evaluation.', flush=True)
    except Exception:
        write(ART / 'BUILD_FAILURE.json', {'created_at': now(), 'traceback': traceback.format_exc(), 'final_week': 'not_run'})
        raise
