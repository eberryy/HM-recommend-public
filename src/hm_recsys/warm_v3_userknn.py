"""Bounded explicit-frequency User-KNN, not a complete TIFU-KNN reproduction.

Only neighbor support is exposed. User's own frequency is not mixed into it.
Same-day purchases are sets; repeated purchase dates retain frequency.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
from pathlib import Path
import time
import traceback

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from .warm_v2_contract import evidence_id, git, guard, now, read, write

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'artifacts/warm_v3/userknn'
REPORT = ROOT/'reports/warm_v3'
TX = Path(__file__).resolve().parents[2] / 'data/interim/audit/transactions.parquet'
PARAMS = {'architecture': 'bounded_temporal_frequency_userknn_v1',
    'history_days': 84, 'half_life_days': 28, 'neighbor_pool_cap': 2000,
    'neighbors_k': 100, 'threads': 4, 'device': 'cpu', 'self_score_weight': 0,
    'profile': 'sum_of_calendar_decayed_daily_basket_presence',
    'similarity': 'cosine_of_full_sparse_frequency_vectors',
    'neighbor_score': 'similarity_weighted_mean_L1_normalized_neighbor_frequency',
    'pool_generation': 'equal_per_seed_budget_postings_recent_first_no_self',
    'tie_order': 'decayed_seed_weight_desc_article_id; neighbor_similarity_desc_customer_id',
    'new_dependency': False}
FEATURES = ['wv3_userknn_neighbor_score', 'wv3_userknn_unavailable']


def literal(path):
    return "'"+str(path).replace("'", "''")+"'"


def assert_workspace():
    assert Path.cwd().resolve() == ROOT
    assert git('branch', '--show-current') == 'warm-v3-architecture-lab'


@contextmanager
def connection():
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='2GB'")
    con.execute('SET enable_progress_bar=false')
    (ART/'scratch').mkdir(parents=True, exist_ok=True)
    con.execute(f'SET temp_directory={literal(ART/"scratch")}')
    try:
        yield con
    finally:
        con.close()


def frame_read(path):
    with connection() as con:
        return con.execute(f'SELECT * FROM read_parquet({literal(path)})').fetchdf()


def frame_save(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.register('f', frame)
        con.execute(f'COPY f TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')


def resource_contract():
    assert_workspace()
    path = REPORT/'USERKNN_RESOURCE_CONTRACT.json'
    if path.exists():
        prior = read(path)
        assert prior['params'] == PARAMS
        return prior
    value = {'created_at_utc': now(), 'status': 'registered_before_resource_pilot',
        'params': PARAMS, 'pilot_cutoff': '2019-11-27', 'query_users_max': 1000,
        'query_selection': '原固定候选池用户按hash(customer_id)、customer_id排序取前1000；不读取target列。',
        'hypothesis': '显式商品购买频次与相似用户的直接邻居支持，可能提供不同于BPR低维因子/图后处理/GRU隐状态的协同信息；本阶段不评测MAP。',
        'representation': '只用截止前84天交易。用户同日购买同商品在购物篮表示中取一次，跨日重复购买保留；权重为0.5**(距截止日天数/28)。原交易文件不改。',
        'pool': '查询用户非零商品按时间加权频次降序、商品ID升序排列；将2000个倒排项读取额度平均分给各商品，余数依序分配。每商品倒排表按最后购买日降序、用户ID升序；先排除自身，再取该商品配额。各表所取用户并集最多2000人，重复用户只计一次且不补足空出的额度。',
        'similarity': '仅对这个有界用户池计算完整稀疏频次向量的余弦相似度，再按相似度降序/用户ID升序取最多100人；不是全用户精确最近邻。',
        'score': '仅邻居分数：每个邻居的频次向量先除以自身L1和，再按余弦相似度加权平均；不混入查询用户自己的频次或自身分数。',
        'features': {FEATURES[0]: '项目特征；一个用户—候选商品对获得的邻居加权频次支持，非负、无购买概率解释。已知用户/商品没有邻居投票时为0。',
            FEATURES[1]: '项目特征；用户无84天历史、没有非自身近邻或候选商品不在历史词表时为1，此时分数NaN；分母为原候选行。'},
        'resource_gate': {'query_seconds_max': 20, 'pilot_total_seconds_max': 300,
            'process_peak_GiB_max': 4, 'projected_eight_cutoff_features_seconds_max': 2100},
        'stop': '超出资源门槛则停止，不减少模型定义继续沿用同一试验；只向根线程报告审计。未获内层许可前不运行全8截止日、不计算MAP。',
        'sources': ['https://arxiv.org/abs/2006.00556', 'https://github.com/HaojiHu/TIFUKNN'],
        'borrowed': '只借显式个性化商品频次、时间动态及直接用户近邻信号。',
        'not_copied': '不同于原TIFU-KNN的分组衰减及自己的/邻居分数混合；本实现是日历半衰期、有界倒排近邻、仅邻居分数；不宣称完整复现。',
        'gpu': False, 'model_training': False, 'MAP_evaluation': False, 'final_week': 'not_run'}
    write(path, value)
    return value


def postings(user_indices, item_indices, last_day, n_items):
    """Per-item recent-first customer postings; arrays contain only observed pairs."""
    user_indices = np.asarray(user_indices, np.int32)
    item_indices = np.asarray(item_indices, np.int32)
    last_day = np.asarray(last_day, np.int32)
    order = np.lexsort((user_indices, -last_day, item_indices))
    offsets = np.r_[0, np.cumsum(np.bincount(item_indices, minlength=n_items))].astype(np.int64)
    return offsets, user_indices[order]


def bounded_pool(profile, user_index, posting_offsets, posting_users, cap=2000):
    """Never returns more than cap users or reads more than cap + seed_count slots."""
    if cap != PARAMS['neighbor_pool_cap']:
        raise ValueError('Neighbor pool budget is frozen; new value needs new version')
    lo, hi = profile.indptr[user_index:user_index+2]
    indices, weights = profile.indices[lo:hi], profile.data[lo:hi]
    if not len(indices):
        return np.empty(0, np.int32), 0
    order = np.lexsort((indices, -weights))
    q, remainder = divmod(cap, len(indices))
    pools, read_slots = [], 0
    for ordinal, item in enumerate(indices[order]):
        quota = q+int(ordinal < remainder)
        if quota == 0:
            continue
        begin, end = posting_offsets[item:item+2]
        # A user occurs at most once per posting list. One extra entry suffices
        # to remove self while preserving the registered non-self quota.
        prefix = posting_users[begin:min(end, begin+quota+1)]
        read_slots += len(prefix)
        take = prefix[prefix != user_index][:quota]
        pools.append(take)
    pool = np.unique(np.concatenate(pools)) if pools else np.empty(0, np.int32)
    assert len(pool) <= cap and user_index not in pool
    assert read_slots <= cap+len(indices)
    return pool, read_slots


class UserKNN:
    def __init__(self, profile, posting_offsets, posting_users):
        self.profile = profile.tocsr()
        self.posting_offsets = posting_offsets
        self.posting_users = posting_users
        self.l1 = np.asarray(self.profile.sum(axis=1)).ravel().astype(np.float32)
        self.l2 = np.sqrt(np.asarray(self.profile.multiply(self.profile).sum(axis=1)).ravel()).astype(np.float32)

    def query(self, user_index):
        if user_index < 0 or user_index >= self.profile.shape[0] or self.l2[user_index] <= 0:
            return csr_matrix((1, self.profile.shape[1]), dtype=np.float32), {
                'pool_users': 0, 'neighbors': 0, 'posting_entries_read': 0, 'unavailable': True}
        pool, scanned = bounded_pool(self.profile, user_index, self.posting_offsets, self.posting_users)
        if len(pool) == 0:
            return csr_matrix((1, self.profile.shape[1]), dtype=np.float32), {
                'pool_users': 0, 'neighbors': 0, 'posting_entries_read': scanned, 'unavailable': True}
        selected = self.profile[pool]
        # The only dense similarity result has <=2000 entries. We never form a
        # query-by-all-users matrix; the sparse product itself has <=2000 rows.
        dot = (selected @ self.profile[user_index].T).toarray().ravel()
        sim = dot/(self.l2[pool]*self.l2[user_index])
        order = np.lexsort((pool, -sim))
        order = order[sim[order] > 0][:PARAMS['neighbors_k']]
        neighbor = pool[order]
        similarity = sim[order]
        if not len(neighbor):
            return csr_matrix((1, self.profile.shape[1]), dtype=np.float32), {
                'pool_users': len(pool), 'neighbors': 0, 'posting_entries_read': scanned, 'unavailable': True}
        weight = similarity/(similarity.sum()*self.l1[neighbor])
        score = (csr_matrix(weight[None, :]) @ self.profile[neighbor]).tocsr()
        score.sort_indices()
        assert np.isfinite(score.data).all() and np.min(score.data) >= 0
        assert abs(float(score.sum())-1) < 2e-5
        return score, {'pool_users': len(pool), 'neighbors': len(neighbor),
            'posting_entries_read': scanned, 'unavailable': False,
            'neighbor_ids': neighbor.tolist(), 'neighbor_similarity': similarity.tolist()}

    def score_pairs(self, user_index, item_indices):
        score, audit = self.query(user_index)
        good = (item_indices >= 0) & (item_indices < self.profile.shape[1])
        out = np.full(len(item_indices), np.nan, np.float32)
        if not audit['unavailable']:
            out[good] = score[:, item_indices[good]].toarray().ravel()
        missing = (~good | bool(audit['unavailable'])).astype(np.float32)
        return out, missing, audit


def build_model(cutoff, transactions=TX):
    assert_workspace()
    guard(cutoff)
    resource_contract()
    root = ART/cutoff
    path = root/'MODEL.json'
    if path.exists():
        result = read(path)
        assert result['params'] == PARAMS and result['cutoff'] == cutoff
        assert result['transactions_path'] == str(Path(transactions))
        for v in result['artifacts'].values():
            assert Path(v['path']).stat().st_size == v['bytes']
        return result
    if (root/'profiles.npz').exists():
        raise RuntimeError('Incomplete UserKNN model preserved; explicit review required')
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with connection() as con:
        con.execute(f'''CREATE TEMP TABLE daily AS SELECT DISTINCT customer_id,article_id,t_dat
            FROM read_parquet({literal(transactions)}) WHERE t_dat<DATE '{cutoff}'
            AND t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY''')
        bounds = con.execute('SELECT min(t_dat),max(t_dat),count(*) FROM daily').fetchone()
        assert str(bounds[1]) < cutoff
        con.execute('''CREATE TEMP TABLE users AS SELECT customer_id,
            (row_number() OVER(ORDER BY customer_id)-1)::INTEGER user_index
            FROM (SELECT DISTINCT customer_id FROM daily)''')
        con.execute('''CREATE TEMP TABLE items AS SELECT article_id,
            (row_number() OVER(ORDER BY article_id)-1)::INTEGER item_index
            FROM (SELECT DISTINCT article_id FROM daily)''')
        pair = con.execute(f'''SELECT user_index,item_index,
            sum(pow(0.5,date_diff('day',t_dat,DATE '{cutoff}')/28.0))::FLOAT frequency,
            max(date_diff('day',DATE '1970-01-01',t_dat))::INTEGER last_day
            FROM daily JOIN users USING(customer_id) JOIN items USING(article_id)
            GROUP BY user_index,item_index ORDER BY user_index,item_index''').fetchdf()
        users = con.execute('SELECT * FROM users ORDER BY user_index').fetchdf()
        items = con.execute('SELECT * FROM items ORDER BY item_index').fetchdf()
    profile = csr_matrix((pair.frequency.to_numpy(np.float32),
        (pair.user_index.to_numpy(), pair.item_index.to_numpy())), shape=(len(users), len(items)))
    assert profile.nnz == len(pair) and (profile.data > 0).all()
    offset, inverted = postings(pair.user_index, pair.item_index, pair.last_day, len(items))
    save_npz(root/'profiles.npz', profile, compressed=True)
    np.savez(root/'postings.npz', offsets=offset, users=inverted)
    frame_save(users, root/'users.parquet')
    frame_save(items, root/'items.parquet')
    from .m3 import _peak_working_set_bytes
    output = {'created_at_utc': now(), 'cutoff': cutoff, 'params': PARAMS,
        'transactions_path': str(Path(transactions)), 'oldest_history_date': str(bounds[0]),
        'latest_history_date': str(bounds[1]), 'daily_basket_item_presences': int(bounds[2]),
        'user_item_pairs': len(pair), 'users': len(users), 'items': len(items),
        'sparse_profile_bytes': int(profile.data.nbytes+profile.indices.nbytes+profile.indptr.nbytes),
        'posting_bytes': int(offset.nbytes+inverted.nbytes),
        'runtime_seconds': time.perf_counter()-started,
        'process_peak_working_set_bytes': _peak_working_set_bytes(),
        'artifacts': {name: evidence_id(root/name, reason='explicit_registry_evidence')
            for name in ['profiles.npz', 'postings.npz', 'users.parquet', 'items.parquet']},
        'future_labels_used': False, 'pit_safe': True, 'final_week': 'not_run'}
    write(path, output)
    del profile, pair, offset, inverted, users, items
    gc.collect()
    return output


def load_model(cutoff):
    root = ART/cutoff
    metadata = read(root/'MODEL.json')
    assert metadata['params'] == PARAMS and metadata['latest_history_date'] < cutoff
    profile = load_npz(root/'profiles.npz')
    with np.load(root/'postings.npz', allow_pickle=False) as values:
        offset, inverted = values['offsets'], values['users']
    users = pd.Index(frame_read(root/'users.parquet').customer_id)
    items = pd.Index(frame_read(root/'items.parquet').article_id)
    return UserKNN(profile, offset, inverted), users, items, metadata


def build_features(cutoff, engine):
    """Only call after root authorizes a formal frozen-pool inner screen."""
    assert_workspace()
    guard(cutoff)
    source = engine.base_path(cutoff)
    root = ART/cutoff
    path, meta_path = root/'features.parquet', root/'FEATURES.json'
    expected = engine.history['feature_cache'][cutoff]['artifact']['sha256']
    if meta_path.exists():
        meta = read(meta_path)
        assert meta['params'] == PARAMS and meta['source_sha256'] == expected
        assert path.stat().st_size == meta['artifact']['bytes']
        return path, meta
    started = time.perf_counter()
    build_model(cutoff, engine.transactions)
    model, users, items, metadata = load_model(cutoff)
    with connection() as con:
        pairs = con.execute(f'''SELECT customer_id,article_id FROM read_parquet({literal(source)})
            ORDER BY customer_id,article_id''').fetchdf()
    assert not pairs.duplicated(['customer_id', 'article_id']).any()
    values = np.full(len(pairs), np.nan, np.float32)
    missing = np.ones(len(pairs), np.float32)
    user_index = users.get_indexer(pairs.customer_id)
    item_index = items.get_indexer(pairs.article_id)
    offsets = np.r_[0, np.flatnonzero(pairs.customer_id.to_numpy()[1:] != pairs.customer_id.to_numpy()[:-1])+1, len(pairs)]
    audits = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        values[lo:hi], missing[lo:hi], a = model.score_pairs(int(user_index[lo]), item_index[lo:hi])
        audits.append({k: a[k] for k in ['pool_users', 'neighbors', 'posting_entries_read', 'unavailable']})
    pairs[FEATURES[0]], pairs[FEATURES[1]] = values, missing
    frame_save(pairs, path)
    result = {'cutoff': cutoff, 'params': PARAMS, 'features': FEATURES, 'rows': len(pairs),
        'users': len(audits), 'source_sha256': expected, 'source': str(source),
        'unavailable_pairs': int(missing.sum()), 'maximum_neighbor_pool': max(a['pool_users'] for a in audits),
        'runtime_seconds': time.perf_counter()-started, 'model_metadata': str(root/'MODEL.json'),
        'future_labels_used': False, 'pit_safe': True, 'latest_history_date': metadata['latest_history_date'],
        'artifact': evidence_id(path, reason='explicit_registry_evidence'), 'final_week': 'not_run'}
    write(meta_path, result)
    return path, result


def pilot():
    contract = resource_contract()
    cutoff = contract['pilot_cutoff']
    started = time.perf_counter()
    metadata = build_model(cutoff)
    model, users, items, _ = load_model(cutoff)
    history = read(ROOT/'reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json')
    source = history['feature_cache'][cutoff]['artifact']['path']
    with connection() as con:
        queries = con.execute(f'''SELECT DISTINCT customer_id FROM read_parquet({literal(source)})
            ORDER BY hash(customer_id),customer_id LIMIT 1000''').fetchdf()
        # Structural size reads only. No target/truth columns or validation labels.
        contract2020 = read(ROOT/'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')
        cutoffs = sorted(set(c for p in contract2020['rolling_protocol'].values()
                             for c in [*p['inner_train'], p['inner_validation']]))
        target_counts = {c: con.execute(f'SELECT count(DISTINCT customer_id) FROM read_parquet({literal(history["feature_cache"][c]["artifact"]["path"])})').fetchone()[0] for c in cutoffs}
    query_index = users.get_indexer(queries.customer_id)
    results = []
    query_start = time.perf_counter()
    for u in query_index:
        v, a = model.query(int(u))
        assert a['pool_users'] <= 2000 and a['neighbors'] <= 100
        results.append({k: a[k] for k in ['pool_users', 'neighbors', 'posting_entries_read', 'unavailable']})
    query_seconds = time.perf_counter()-query_start
    from .m3 import _peak_working_set_bytes
    peak = _peak_working_set_bytes()
    # Conservatively reserve twice the measured model preparation per cutoff;
    # query count is actual frozen source-user count, never a MAP denominator change.
    projected = metadata['runtime_seconds']*2*len(cutoffs)+query_seconds/len(queries)*sum(target_counts.values())
    elapsed = time.perf_counter()-started
    gate = contract['resource_gate']
    checks = {'query_within20s': query_seconds <= gate['query_seconds_max'],
        'whole_pilot_within300s': elapsed <= gate['pilot_total_seconds_max'],
        'peak_within4GiB': peak <= gate['process_peak_GiB_max']*1024**3,
        'projected_eight_cutoffs_within35min': projected <= gate['projected_eight_cutoff_features_seconds_max']}
    result = {'created_at_utc': now(), 'cutoff': cutoff, 'params': PARAMS, 'resource_contract': str(REPORT/'USERKNN_RESOURCE_CONTRACT.json'),
        'model': metadata, 'query_users': len(queries), 'query_users_with84d_history': int((query_index >= 0).sum()),
        'query_users_with_nonself_neighbors': sum(not a['unavailable'] for a in results),
        'maximum_pool_users': max(a['pool_users'] for a in results),
        'pool_quantiles': np.quantile([a['pool_users'] for a in results], [0, .5, .9, .99, 1]).tolist(),
        'neighbor_quantiles': np.quantile([a['neighbors'] for a in results], [0, .5, .9, .99, 1]).tolist(),
        'posting_entries_read_max': max(a['posting_entries_read'] for a in results),
        'query_seconds': query_seconds, 'pilot_total_seconds': elapsed,
        'process_peak_working_set_bytes': peak, 'target_user_counts_for_cost_projection': target_counts,
        'projected_eight_cutoff_feature_seconds': projected, 'checks': checks,
        'resource_passed': all(checks.values()), 'MAP_evaluation': False,
        'neighbor_and_item_scoring_dense_matrix': '仅最多2000个近邻用户的相似度列向量可转稠密；不创建查询用户×全部历史用户或用户×商品稠密矩阵。',
        'source_integrity': '模型和查询均只读已登记截止日之前交易及原候选用户ID；所有新产物写本项目UserKNN目录，旧输入未改。',
        'next_action': '向根线程报告；正式221/222与8截止日内层训练未启动。', 'gpu': False, 'final_week': 'not_run'}
    write(REPORT/'USERKNN_RESOURCE_AUDIT.json', result)
    print({k: result[k] for k in ['query_users', 'query_seconds', 'pilot_total_seconds',
        'maximum_pool_users', 'projected_eight_cutoff_feature_seconds', 'resource_passed']}, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['pilot'])
    a = p.parse_args()
    try:
        pilot()
    except Exception:
        write(ART/f'FAILURE_{time.time_ns()}.json', {'created_at': now(),
            'traceback': traceback.format_exc(), 'algorithm_failure': False, 'final_week': 'not_run'})
        raise
