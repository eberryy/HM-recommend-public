"""Cutoff-safe post-hoc graph propagation of frozen BPR latent factors.

This is NOT trained LightGCN: the binary graph changes representations by the
LightGCN propagation operator, but no graph-aware gradient update is performed.
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
from scipy.sparse import csr_matrix

from .warm_v2_bpr import PARAMS as BPR_PARAMS, relation
from .warm_v2_contract import evidence_id, git, guard, now, read, write

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT/'artifacts/warm_v3/graph'
REPORT = ROOT/'reports/warm_v3'
OLD_BPR = Path(__file__).resolve().parents[2] / 'artifacts/warm_v2/bpr-match-v1'
FRESH_BPR = ROOT/'artifacts/warm_v2/fresh_robustness/bpr'
PARAMS = {'architecture': 'posthoc_bpr_graph_mean012_v1', 'layers': 2,
          'layer_weights': [1/3, 1/3, 1/3], 'latent_dimensions': 100,
          'normalization': 'symmetric_binary_degree', 'bias_propagated': False,
          'score_has_original_item_bias': False, 'graph_training_steps': 0,
          'dtype': 'float32', 'device': 'cpu', 'threads': 4}
FEATURES = ['wv3_graph_user_item_score', 'wv3_graph_unavailable']
DEFINITIONS = {
    FEATURES[0]: '项目特征：截至特征日以前的二部购买图上，将冻结BPR的100维潜因子做0/1/2层对称度归一化传播后等权平均，再计算用户与候选商品的内积；每行是一个用户—候选商品对，无概率含义，不包含商品偏置。',
    FEATURES[1]: '项目特征：用户或候选商品不在截至该日的购买图词表时为1，否则为0；缺失匹配分数为NaN。分母为当前原候选池的所有用户—商品行。',
}


def assert_workspace():
    if Path.cwd().resolve() != ROOT or git('branch', '--show-current') != 'warm-v3-architecture-lab':
        raise RuntimeError('Graph writes require the isolated Warm-v3 worktree')


def literal(path):
    return "'"+str(path).replace("'", "''")+"'"


@contextmanager
def connection():
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='2GB'")
    scratch = ART/'_scratch'
    scratch.mkdir(parents=True, exist_ok=True)
    con.execute(f'SET temp_directory={literal(scratch)}')
    try:
        yield con
    finally:
        con.close()


def read_frame(path):
    with connection() as con:
        return con.execute(f'SELECT * FROM read_parquet({literal(path)})').fetchdf()


def save_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.register('frame', frame)
        con.execute(f'COPY frame TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')


def normalized_relation(matrix):
    """S = D_user^-1/2 R D_item^-1/2; degrees count distinct graph edges."""
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(matrix.data == 1):
        raise ValueError('Expected binary/set relation, not event-frequency weights')
    du = np.diff(matrix.indptr).astype(np.float32)
    di = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    iu = np.zeros_like(du)
    ii = np.zeros_like(di)
    np.divide(1., np.sqrt(du), out=iu, where=du > 0)
    np.divide(1., np.sqrt(di), out=ii, where=di > 0)
    matrix.data *= np.repeat(iu, np.diff(matrix.indptr))
    matrix.data *= ii[matrix.indices]
    return matrix, du, di


def latent_factors(user_factors, item_factors):
    """implicit BPR stores an extra bias coordinate, which is not a latent axis."""
    if user_factors.shape[1] != 101 or item_factors.shape[1] != 101:
        raise ValueError('Frozen implicit BPR must have 100 latent + 1 bias columns')
    if not np.allclose(user_factors[:, -1], 1., rtol=0, atol=1e-6):
        raise ValueError('Unexpected BPR user bias-coordinate semantics')
    if not np.isfinite(user_factors).all() or not np.isfinite(item_factors).all():
        raise ValueError('Nonfinite frozen factors')
    return (np.ascontiguousarray(user_factors[:, :100], dtype=np.float32),
            np.ascontiguousarray(item_factors[:, :100], dtype=np.float32))


def propagate(matrix, user_factors, item_factors, layers=2):
    """Return uniform mean of layers 0..K on the bipartite graph, no gradients."""
    if layers != PARAMS['layers']:
        raise ValueError('Changing graph depth needs a new registered architecture')
    s, du, di = normalized_relation(matrix)
    st = s.T.tocsr()
    u = np.ascontiguousarray(user_factors, dtype=np.float32)
    i = np.ascontiguousarray(item_factors, dtype=np.float32)
    if u.shape[0] != s.shape[0] or i.shape[0] != s.shape[1] or u.shape[1] != i.shape[1]:
        raise ValueError('Graph vocabulary and factor shape disagree')
    us = u.copy()
    isum = i.copy()
    timings = []
    for layer in range(layers):
        start = time.perf_counter()
        unext = s @ i
        inext = st @ u
        u, i = unext, inext
        us += u
        isum += i
        timings.append({'layer': layer+1, 'seconds': time.perf_counter()-start})
    us /= layers+1
    isum /= layers+1
    assert np.isfinite(us).all() and np.isfinite(isum).all()
    audit = {'layers': timings, 'sparse_csr_plus_transpose_bytes': int(
        sum(a.nbytes for m in (s, st) for a in (m.data, m.indices, m.indptr))),
        'user_degree_quantiles': np.quantile(du, [0, .5, .9, .99, 1]).tolist(),
        'item_degree_quantiles': np.quantile(di, [0, .5, .9, .99, 1]).tolist(),
        'zero_degree_users': int((du == 0).sum()), 'zero_degree_items': int((di == 0).sum())}
    return us, isum, audit


def match_scores(frame, users, items, uf, itf, chunk_size=32768):
    ui = users.get_indexer(frame.customer_id)
    ii = items.get_indexer(frame.article_id)
    good = (ui >= 0) & (ii >= 0)
    scores = np.full(len(frame), np.nan, np.float32)
    positions = np.flatnonzero(good)
    for start in range(0, len(positions), chunk_size):
        p = positions[start:start+chunk_size]
        scores[p] = np.einsum('ij,ij->i', uf[ui[p]], itf[ii[p]])
    return scores, (~good).astype(np.float32)


def source_model(cutoff):
    guard(cutoff)
    for base in (OLD_BPR, FRESH_BPR):
        path = base/cutoff/'model.json'
        if path.exists():
            meta = read(path)
            assert meta['cutoff'] == cutoff and meta['latest_history_date'] < cutoff
            assert meta['params'] == BPR_PARAMS and meta['pit_safe']
            return path.parent, meta
    raise FileNotFoundError(f'No frozen cutoff-matched BPR model for {cutoff}; graph does not retrain BPR')


def build_model(cutoff, engine):
    assert_workspace()
    guard(cutoff)
    destination = ART/cutoff
    meta_path = destination/'model.json'
    original_root, original = source_model(cutoff)
    if meta_path.exists():
        cached = read(meta_path)
        assert cached['params'] == PARAMS and cached['cutoff'] == cutoff
        assert cached['source_bpr_sha256'] == original['artifacts']['factors.npz']['sha256']
        assert cached['transactions_sha256'] == engine.history['inputs']['transactions']['sha256']
        for artifact in cached['artifacts'].values():
            assert Path(artifact['path']).stat().st_size == artifact['bytes']
        return cached
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'factors.npz').exists():
        raise RuntimeError('Preserve incomplete graph factors; no implicit overwrite')
    started = time.perf_counter()
    # Concrete reuse risk: old shared mutable factors after migration; compare with
    # original saved BPR evidence, once, before graph transformation.
    source_checks = {}
    for name in ('factors.npz', 'users.parquet', 'items.parquet'):
        current = evidence_id(original_root/name, reason='explicit_registry_evidence')
        expected = original['artifacts'][name]
        assert current['sha256'] == expected['sha256'] and current['bytes'] == expected['bytes']
        source_checks[name] = current
    assert original['transactions']['sha256'] == engine.history['inputs']['transactions']['sha256']
    prep = time.perf_counter()
    with connection() as con:
        matrix, users, items, latest = relation(con, engine.transactions, cutoff)
    assert matrix.nnz == original['binary_pairs']
    oldusers = read_frame(original_root/'users.parquet')
    olditems = read_frame(original_root/'items.parquet')
    pd.testing.assert_frame_equal(users, oldusers)
    pd.testing.assert_frame_equal(items, olditems)
    relation_seconds = time.perf_counter()-prep
    print(f'Graph {cutoff}: {matrix.shape}, {matrix.nnz:,} binary edges; propagation', flush=True)
    with np.load(original_root/'factors.npz', allow_pickle=False) as arrays:
        uf, itf = latent_factors(arrays['user_factors'], arrays['item_factors'])
    t = time.perf_counter()
    gu, gi, audit = propagate(matrix, uf, itf)
    propagation_seconds = time.perf_counter()-t
    np.savez(destination/'factors.npz', user_factors=gu, item_factors=gi)
    from .m3 import _peak_working_set_bytes
    result = {'cutoff': cutoff, 'created_at_utc': now(), 'params': PARAMS,
        'model_name': 'post-hoc graph-smoothed BPR; NOT trained LightGCN',
        'source_model': str(original_root/'model.json'),
        'source_bpr_sha256': original['artifacts']['factors.npz']['sha256'],
        'source_checks': source_checks, 'transactions_sha256': original['transactions']['sha256'],
        'latest_history_date': latest, 'pit_safe': latest < cutoff, 'future_labels_used': False,
        'users': len(users), 'items': len(items), 'binary_pairs': matrix.nnz,
        'factor_shapes': [list(gu.shape), list(gi.shape)], 'relation_seconds': relation_seconds,
        'propagation_seconds': propagation_seconds, 'runtime_seconds': time.perf_counter()-started,
        'peak_process_working_set_bytes': _peak_working_set_bytes(), 'propagation_audit': audit,
        'vocabulary': {'users': str(original_root/'users.parquet'), 'items': str(original_root/'items.parquet')},
        'artifacts': {'factors.npz': evidence_id(destination/'factors.npz', reason='explicit_registry_evidence')},
        'gpu_used': False, 'final_week': 'not_run'}
    write(meta_path, result)
    del matrix, uf, itf, gu, gi, users, items, oldusers, olditems
    gc.collect()
    return result


def build_features(cutoff, engine):
    """Same cutoff/engine API as Warm-v2 BPR, but entirely new output storage."""
    assert_workspace()
    guard(cutoff)
    source = Path(engine.base_path(cutoff))
    source_hash = engine.history['feature_cache'][cutoff]['artifact']['sha256']
    destination = ART/cutoff
    path = destination/'features.parquet'
    meta_path = destination/'features.json'
    if meta_path.exists():
        meta = read(meta_path)
        assert meta['params'] == PARAMS and meta['source_sha256'] == source_hash
        assert path.stat().st_size == meta['artifact']['bytes']
        return path, meta
    started = time.perf_counter()
    model = build_model(cutoff, engine)
    with np.load(destination/'factors.npz', allow_pickle=False) as arrays:
        uf, itf = arrays['user_factors'], arrays['item_factors']
    users = pd.Index(read_frame(model['vocabulary']['users']).customer_id)
    items = pd.Index(read_frame(model['vocabulary']['items']).article_id)
    chunks = []
    with connection() as con:
        cursor = con.execute(f'SELECT customer_id, article_id FROM read_parquet({literal(source)})')
        while True:
            frame = cursor.fetch_df_chunk(16)
            if frame.empty:
                break
            frame[FEATURES[0]], frame[FEATURES[1]] = match_scores(frame, users, items, uf, itf)
            chunks.append(frame)
    frame = pd.concat(chunks, ignore_index=True)
    assert not frame.duplicated(['customer_id', 'article_id']).any()
    save_frame(frame, path)
    result = {'cutoff': cutoff, 'params': PARAMS, 'rows': len(frame), 'features': FEATURES,
        'definitions': DEFINITIONS, 'source': str(source), 'source_sha256': source_hash,
        'model_metadata': str(destination/'model.json'), 'source_bpr_sha256': model['source_bpr_sha256'],
        'unavailable_pairs': int(frame[FEATURES[1]].sum()), 'runtime_seconds': time.perf_counter()-started,
        'artifact': evidence_id(path, reason='explicit_registry_evidence'),
        'pit_safe': True, 'future_labels_used': False, 'final_week': 'not_run'}
    write(meta_path, result)
    return path, result


def no_label_score_audit(cutoff):
    """Signal redundancy only: inspect numeric features, never target/labels."""
    original, _ = source_model(cutoff)
    graph_path = ART/cutoff/'features.parquet'
    bpr_path = original/'features.parquet'
    with connection() as con:
        con.execute(f'CREATE VIEW g AS SELECT * FROM read_parquet({literal(graph_path)})')
        con.execute(f'CREATE VIEW b AS SELECT * FROM read_parquet({literal(bpr_path)})')
        invalid = con.execute('''SELECT count(*) FROM g FULL JOIN b USING(customer_id, article_id)
            WHERE g.customer_id IS NULL OR b.customer_id IS NULL
            OR g.wv3_graph_unavailable <> b.wv2_bpr_unavailable''').fetchone()[0]
        assert invalid == 0
        row = con.execute('''SELECT count(*), corr(wv3_graph_user_item_score, wv2_bpr_user_item_score),
            stddev_pop(wv3_graph_user_item_score), stddev_pop(wv2_bpr_user_item_score),
            min(wv3_graph_user_item_score), max(wv3_graph_user_item_score)
            FROM g JOIN b USING(customer_id, article_id) WHERE wv3_graph_unavailable=0''').fetchone()
    return {'definition': '仅在原候选池中两种表示都可用的用户—商品对上计算分数分布与Pearson相关系数；不使用未来购买标签。相关性不能证明正例互补或MAP增益。',
        'jointly_available_pairs': row[0], 'pearson_graph_vs_bpr_score': row[1],
        'graph_score_std': row[2], 'bpr_score_std': row[3], 'graph_score_min': row[4],
        'graph_score_max': row[5], 'pair_identity_or_availability_mismatches': invalid,
        'labels_read': False, 'bpr_score_includes_original_item_bias': True}


def pilot(cutoff='2019-11-27'):
    if cutoff != '2019-11-27':
        raise ValueError('Mechanics pilot is restricted to first historical inner-training cutoff')
    from .warm_v2_engine import Engine
    engine = Engine(read(ROOT/'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json'))
    t = time.perf_counter()
    path, features = build_features(cutoff, engine)
    model = read(ART/cutoff/'model.json')
    old_sizes = []
    for meta_path in sorted(OLD_BPR.glob('*/model.json')):
        v = read(meta_path)
        scale = v['binary_pairs']/model['binary_pairs']
        old_sizes.append({'cutoff': v['cutoff'], 'binary_pairs': v['binary_pairs'],
            'nodes': v['users']+v['items'], 'factor_bytes': (v['users']+v['items'])*100*4,
            'projected_propagation_seconds_linear_edges': model['propagation_seconds']*scale})
    output = {'created_at_utc': now(), 'status': 'mechanics_pilot_completed_no_outer_evaluation',
        'hypothesis': '冻结BPR潜因子加入归一化高阶邻居传播后，可给原固定候选池提供超出直接矩阵分解的协同匹配信号；是否有MAP增益尚待根线程的固定inner筛选。',
        'architecture': PARAMS, 'pilot_cutoff': cutoff, 'pilot': model, 'features': features,
        'pilot_total_seconds': time.perf_counter()-t, 'gpu_used': False, 'gpu_released': True,
        'graph_fully_trained': False, 'new_bpr_training': False, 'labels_accessed': False,
        'projections': old_sizes, 'no_label_score_audit': no_label_score_audit(cutoff),
        'final_week': 'not_run',
        'borrowed_idea': 'LightGCN的二部图对称度归一化传播、保留第0层并跨层等权平均。没有复制训练循环，也没有声称复现其端到端图训练收益。',
        'sources': ['https://arxiv.org/abs/2002.02126',
            'https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py'],
        'resource_caveat': '线性外推只估计离线传播，不是LightGCN训练时间。官方全图前向每次BPR小批次重算传播；全量图训练的总成本还依赖梯度反传和批次数，本轮未测训练性能。',
        'representation_risks': ['传播后未经图目标重新训练，可能过度平滑或丢失个人细分偏好。',
            '全历史图不含显式时间衰减；与已有BPR共享输入，可能高度冗余。',
            '删除偏置坐标防止把常数用户偏置传播成虚假的潜特征；原BPR分数仍由对照保留。'],
        'next_gate': '不读取outer；先检查各inner窗口的匹配增量及专家独有Top12正例，统一预注册后由根线程决定是否做一次outer确认。'}
    write(REPORT/'GRAPH_RESOURCE_AUDIT.json', output)
    print({'pilot_seconds': output['pilot_total_seconds'], 'model_seconds': model['runtime_seconds'],
        'propagation_seconds': model['propagation_seconds'], 'feature_rows': features['rows'],
        'feature_path': str(path)}, flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['pilot'])
    args = parser.parse_args()
    try:
        pilot()
    except Exception:
        write(ART/f'FAILURE_{time.time_ns()}.json', {'time': now(), 'traceback': traceback.format_exc(),
            'final_week': 'not_run', 'algorithm_failure': False})
        raise
