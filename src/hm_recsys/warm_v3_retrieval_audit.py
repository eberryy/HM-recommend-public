"""Frozen BPR Top100 retrieval diagnostic on four original INNER weeks only."""
from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from . import warm_v3_common as common
from .warm_v2_contract import read, write, guard, now
from .warm_v2_engine import literal

CUTOFFS = ('2019-12-25', '2020-02-19', '2020-05-27', '2020-07-22')
ART = common.ART / 'bpr_retrieval'
CONTRACT = common.REPORT / 'BPR_RETRIEVAL_CONTRACT.json'
K = 100


def allowed(cutoff):
    guard(cutoff)
    if cutoff not in CUTOFFS:
        raise ValueError('BPR retrieval diagnostic authorizes original INNER dates only')


def topk_exact(query, factors, article_ids, k=K, batch=128, device='cpu'):
    """FP32 exhaustive dot products; stable descending sort over ascending item IDs."""
    article_ids = np.asarray(article_ids)
    if len(article_ids) < k or len(np.unique(article_ids)) != len(article_ids):
        raise ValueError('Insufficient or duplicate catalog IDs')
    if query.shape[1] != factors.shape[1]:
        raise ValueError('Factor dimensions differ')
    order = np.argsort(article_ids, kind='stable')
    ids = article_ids[order]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    it = torch.as_tensor(np.ascontiguousarray(factors[order], dtype=np.float32), device=device)
    output, scores = [], []
    with torch.no_grad():
        for start in range(0, len(query), batch):
            q = torch.as_tensor(np.ascontiguousarray(query[start:start + batch], dtype=np.float32), device=device)
            dot = q @ it.T
            if not bool(torch.isfinite(dot).all()):
                raise ValueError('Nonfinite BPR score')
            indices = torch.argsort(dot, dim=1, descending=True, stable=True)[:, :k]
            output.append(ids[indices.cpu().numpy()])
            scores.append(torch.gather(dot, 1, indices).cpu().numpy())
    del it
    if not output:
        return np.empty((0, k), dtype=article_ids.dtype), np.empty((0, k), np.float32)
    return np.concatenate(output), np.concatenate(scores)


def pool_metrics(con, relation):
    counts = con.execute(f'''SELECT u.customer_id,u.truth_count,
        coalesce(c.n,0) candidates,coalesce(c.hits,0) hits FROM truth_users u LEFT JOIN (
        SELECT p.customer_id,count(*) n,count(t.article_id) hits FROM ({relation}) p
        LEFT JOIN truth t USING(customer_id,article_id) GROUP BY p.customer_id
        ) c USING(customer_id) ORDER BY customer_id''').fetchdf()
    h = counts.hits.to_numpy(np.float64)
    t = counts.truth_count.to_numpy(np.float64)
    rows = int(counts.candidates.sum())
    result = {'users_denominator': len(counts), 'candidate_pairs': rows,
              'positive_pairs': int(h.sum()), 'macro_recall': float(np.mean(h / t)),
              'oracle_map12': float(np.mean(np.minimum(h, 12) / np.minimum(t, 12))),
              'pair_density': float(h.sum() / rows) if rows else 0.0,
              'users_with_covered_truth': int((h > 0).sum())}
    return result


def one(cutoff, engine, device):
    allowed(cutoff)
    common.budget(3)
    root = ART / cutoff
    ready = root / 'AUDIT.json'
    if ready.exists():
        return read(ready)
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = engine.base_path(cutoff)
    source_meta = engine.history['feature_cache'][cutoff]['artifact']
    bp, feature_meta = common.bpr_path(cutoff)
    factor_root = bp.parent
    model_meta = read(factor_root / 'model.json')
    assert model_meta['cutoff'] == cutoff and model_meta['latest_history_date'] < cutoff
    assert model_meta['params']['factors'] == 100 and model_meta['params']['iterations'] == 100
    assert feature_meta['source_sha256'] == source_meta['sha256']
    assert model_meta['transactions']['sha256'] == engine.history['inputs']['transactions']['sha256']
    for name, old in model_meta['artifacts'].items():
        assert (factor_root / name).stat().st_size == old['bytes']
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='2GB'")
    con.execute(f'SET temp_directory={literal(root / "spill")}')
    try:
        con.execute(f'''CREATE VIEW base AS SELECT customer_id,article_id,target,candidate_rank,
            user_history_events_12w FROM read_parquet({literal(source)})''')
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id
            FROM read_parquet({literal(common.TX)}) WHERE t_dat>=DATE '{cutoff}'
            AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        con.execute('CREATE TEMP TABLE truth_users AS SELECT customer_id,count(*) truth_count FROM truth GROUP BY customer_id')
        customers = con.execute('''SELECT customer_id,max(user_history_events_12w) events,
            count(*) n,count(DISTINCT article_id) nd FROM base GROUP BY customer_id ORDER BY customer_id''').fetchdf()
        expected = con.execute('SELECT customer_id FROM truth_users ORDER BY customer_id').fetchdf()
        assert customers.customer_id.tolist() == expected.customer_id.tolist(), 'full original truth cohort differs'
        assert customers.n.between(100, 300).all() and (customers.n == customers.nd).all()
        assert con.execute('''SELECT count(*) FROM base b LEFT JOIN truth t USING(customer_id,article_id)
            WHERE b.target<>CAST(t.article_id IS NOT NULL AS INTEGER)''').fetchone()[0] == 0
        users = con.execute(f'SELECT * FROM read_parquet({literal(factor_root / "users.parquet")}) ORDER BY user_index').fetchdf()
        items = con.execute(f'SELECT * FROM read_parquet({literal(factor_root / "items.parquet")}) ORDER BY item_index').fetchdf()
        assert np.array_equal(users.user_index, np.arange(model_meta['users']))
        assert np.array_equal(items.item_index, np.arange(model_meta['items']))
        articles = con.execute(f'''SELECT article_id FROM read_csv({literal(common.SHARED / 'data/raw/articles.csv')},
            header=true,all_varchar=true)''').fetchdf()
        catalog = items[items.article_id.isin(set(articles.article_id))].sort_values('article_id').copy()
        active = customers[customers.events > 0].copy()
        user_index = pd.Index(users.customer_id)
        query_index = user_index.get_indexer(active.customer_id)
        assert (query_index >= 0).all(), 'recently active user missing from full-history factors'
        with np.load(factor_root / 'factors.npz', allow_pickle=False) as arrays:
            full_users = arrays['user_factors']
            item_factors = arrays['item_factors']
            assert full_users.shape == (len(users), 101) and item_factors.shape == (len(items), 101)
            queries = full_users[query_index].copy()
            sample = con.execute(f'''SELECT * FROM read_parquet({literal(bp)})
                WHERE wv2_bpr_unavailable=0 ORDER BY customer_id,article_id LIMIT 2048''').fetchdf()
            sui = user_index.get_indexer(sample.customer_id)
            sii = pd.Index(items.article_id).get_indexer(sample.article_id)
            sq, si = full_users[sui].copy(), item_factors[sii].copy()
            del full_users
        expected_score = sample.wv2_bpr_user_item_score.to_numpy(np.float32)
        numpy_score = np.einsum('ij,ij->i', sq, si)
        np.testing.assert_allclose(numpy_score, expected_score, rtol=2e-5, atol=2e-5)
        # Same matmul primitive as exhaustive retrieval, sampled diagonal entries.
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            gpu_check = torch.diagonal(torch.tensor(sq, device=device) @ torch.tensor(si, device=device).T).cpu().numpy()
        np.testing.assert_allclose(gpu_check, expected_score, rtol=2e-4, atol=5e-5)
        score_check = {'pairs': len(sample), 'numpy_max_absolute_error': float(np.max(np.abs(numpy_score - expected_score))),
                       'matmul_max_absolute_error': float(np.max(np.abs(gpu_check - expected_score))), 'passed': True}
        gc.collect()
        item_ids, scores = topk_exact(queries, item_factors[catalog.item_index.to_numpy()], catalog.article_id.to_numpy(), device=device)
        retrieved = pd.DataFrame({'customer_id': np.repeat(active.customer_id.to_numpy(), K),
            'article_id': item_ids.reshape(-1), 'bpr_score': scores.reshape(-1),
            'retrieval_rank': np.tile(np.arange(1, K + 1, dtype=np.int32), len(active))})
        assert len(retrieved) == len(active) * K and not retrieved.duplicated(['customer_id', 'article_id']).any()
        con.register('retrieved', retrieved)
        path = root / 'top100.parquet'
        con.execute(f'COPY retrieved TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        current = pool_metrics(con, 'SELECT customer_id,article_id FROM base')
        bpr_only = pool_metrics(con, 'SELECT customer_id,article_id FROM retrieved')
        union = pool_metrics(con, 'SELECT customer_id,article_id FROM base UNION SELECT customer_id,article_id FROM retrieved')
        overlap = con.execute('SELECT count(*) FROM retrieved r JOIN base b USING(customer_id,article_id)').fetchone()[0]
        marginal = con.execute('''SELECT count(*),count(DISTINCT r.customer_id) FROM retrieved r
            JOIN truth t USING(customer_id,article_id) WHERE NOT EXISTS
            (SELECT 1 FROM base b WHERE b.customer_id=r.customer_id AND b.article_id=r.article_id)''').fetchone()
        assert union['candidate_pairs'] == current['candidate_pairs'] + bpr_only['candidate_pairs'] - overlap
        assert union['positive_pairs'] - current['positive_pairs'] == marginal[0]
        top_item_users = con.execute('SELECT max(n) FROM (SELECT article_id,count(*) n FROM retrieved GROUP BY article_id)').fetchone()[0]
        unique_items = int(retrieved.article_id.nunique())
        result = {'cutoff': cutoff, 'role': 'original_inner', 'retrieval_k': K, 'device': device,
            'truth_users': len(customers), 'truth_pairs': int(con.execute('SELECT count(*) FROM truth').fetchone()[0]),
            'active_query_users': len(active), 'inactive_no_bpr_users': int((customers.events == 0).sum()),
            'warm_catalog_items': len(catalog), 'bpr_vocabulary_items': len(items),
            'current': current, 'bpr_only': bpr_only, 'union': union,
            'marginal_truth_pairs': int(marginal[0]), 'users_with_marginal_truth': int(marginal[1]),
            'recall_delta': union['macro_recall'] - current['macro_recall'],
            'oracle_delta': union['oracle_map12'] - current['oracle_map12'],
            'overlap_pairs': int(overlap), 'bpr_overlap_fraction': float(overlap / len(retrieved)),
            'retrieved_distinct_items': unique_items, 'most_common_retrieved_item_query_fraction': float(top_item_users / len(active)),
            'score_parity': score_check, 'source': source_meta, 'bpr_metadata': str(factor_root / 'model.json'),
            'latest_factor_history_date': model_meta['latest_history_date'], 'factor_columns': 101,
            'output': {'path': str(path), 'bytes': path.stat().st_size},
            'runtime_seconds': time.perf_counter() - started, 'fits': 0, 'final_week': 'not_run'}
        write(ready, result)
        print({'cutoff': cutoff, 'marginal_truth_pairs': marginal[0], 'recall_delta': result['recall_delta'],
               'oracle_delta': result['oracle_delta'], 'seconds': result['runtime_seconds']}, flush=True)
        return result
    finally:
        con.close()
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()


def report(results, seconds, device):
    for r in results:
        extra = r['union']['candidate_pairs'] - r['current']['candidate_pairs']
        r['marginal_candidate_pairs'] = extra
        r['marginal_pair_density'] = r['marginal_truth_pairs'] / extra if extra else 0.0
        r['mean_extra_candidates_per_active_user'] = extra / r['active_query_users']
        r['candidate_pair_growth_fraction'] = extra / r['current']['candidate_pairs']
    positives = sum(r['recall_delta'] > 0 and r['oracle_delta'] > 0 for r in results)
    output = {'created_at': now(), 'contract': str(CONTRACT), 'windows': {r['cutoff']: r for r in results},
        'mean_recall_delta': float(np.mean([r['recall_delta'] for r in results])),
        'mean_oracle_delta': float(np.mean([r['oracle_delta'] for r in results])),
        'mechanism_gate': positives >= 2, 'positive_windows': positives, 'device': device,
        'wall_seconds': seconds, 'fits': 0, 'candidate_pool_changed': False,
        'next_step_authorization': 'diagnostic only; candidate changes require a new same-pool baseline contract',
        'final_week': 'not_run'}
    write(common.REPORT / 'BPR_RETRIEVAL_AUDIT.json', output)
    lines = ['# BPR 学习式召回诊断（原开发内层四周）', '',
        'BPR（行业通用贝叶斯个性化排序）在本项目中是全历史用户—商品二值关系矩阵分解。本次直接复用保存的 100 隐维及 1 列商品偏置，穷举候选目录内积，取每位活跃用户前 100 件；不重训、不排除已购买商品、不修改原候选池。', '',
        '本审计的真值是各起始日期随后 7 天购买的不同商品；总体分母是冻结的 10% 真值用户队列，包含不活跃和原候选未覆盖真值的用户。只对最近 12 周有交易的用户执行 BPR 召回。商品目录是原 articles 与该截止日前 BPR 商品词表的交集。', '',
        'Recall（行业召回率）先计算每用户命中的不同真实商品数除以完整真值商品数，再对全部用户取平均。Oracle MAP@12（行业理想排序上界）假设池中命中真值都排在最前，使用 min(命中数,12)/min(完整真值数,12)；它不是实际排序器 MAP。新增真值对指 BPR 找到、原候选池没找到的用户—商品购买对，不是跨时间窗口互斥的贡献。', '',
        '| 验证周起点 | 原池 Recall | BPR 单路 Recall | 合并 Recall | 新增真值对 | 合并 Oracle 增量 | BPR 与原池重合比例 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['cutoff']} | {r['current']['macro_recall']:.6f} | {r['bpr_only']['macro_recall']:.6f} | {r['union']['macro_recall']:.6f} | {r['marginal_truth_pairs']} | {r['oracle_delta']:+.6f} | {r['bpr_overlap_fraction']:.2%} |")
    lines += ['', '| 验证周起点 | 全部用户 | 活跃查询用户 | 目录商品 | BPR 候选对正例密度 | 新增真值用户 | 实际秒数 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['cutoff']} | {r['truth_users']} | {r['active_query_users']} | {r['warm_catalog_items']} | {r['bpr_only']['pair_density']:.4%} | {r['users_with_marginal_truth']} | {r['runtime_seconds']:.1f} |")
    lines += ['', '| 验证周起点 | 新增候选对 | 新增候选对正例密度 | 每活跃用户平均新增商品 | 候选总行数增幅 |',
              '|---|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['cutoff']} | {r['marginal_candidate_pairs']} | {r['marginal_pair_density']:.4%} | {r['mean_extra_candidates_per_active_user']:.1f} | {r['candidate_pair_growth_fraction']:.2%} |")
    lines += ['', '候选对正例密度是该来源命中的真实购买对数除以该来源实际返回的候选用户—商品对数；BPR 与原池重合比例以 BPR 返回的候选对数为分母。上述比例不改变整体 Recall/Oracle 的完整用户分母。', '',
        '新增候选对正例密度以 BPR 找到、原池没有的候选对为分母。新增覆盖存在，但新增池仍稀疏；合并后需排序更多商品，能否兑现实际 MAP 必须通过同池排序器重训验证。候选总行数增幅以原候选用户—商品对数为分母。', '',
        f"机制门槛：{positives}/4 窗口同时改善 Recall 和 Oracle，预注册至少 2 窗门槛{'通过' if positives >= 2 else '未通过'}。平均 Recall 增量 {output['mean_recall_delta']:+.6f}，平均 Oracle 增量 {output['mean_oracle_delta']:+.6f}。这是候选覆盖证据，不能等同于 MAP 已兑现或允许部署新候选池。", '',
        '旧 WV2-502 只取商品偏置的内层消融也有平均 +0.000294176 增益，但完整用户匹配四窗均优于偏置、平均再高 +0.000373821。这说明旧排序增益不只来自商品偏置；不能据此假定新的全目录召回必然有效，也不能按本次结果临时删除偏置或屏蔽热门商品。', '',
        f'计算使用 {device} FP32（32 位浮点），关闭 TF32（较低有效精度的 TensorFloat-32 计算）；同分固定按 article_id 升序，候选 K=100 没有搜索。逐窗用确定性 2,048 对与旧 BPR 特征对照，numpy 与矩阵乘法均在合同容差内；完整用户队列、标签、无重复候选及合并集合恒等式均验证。实际新增计算 {seconds:.1f} 秒，模型拟合 0 次。', '',
        '未运行任何外层时间验证窗口（outer）、新排序器、提交文件或最终周。2020-09-16 = not_run。下一步若改变候选池，必须另行注册候选协议并建立同池 WV2-601 对照，不能直接把本诊断上界当实际 MAP 增益。']
    (common.REPORT / 'BPR_RETRIEVAL_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


def run(device):
    common.setup()
    common.budget(15)
    contract = read(CONTRACT)
    assert tuple(contract['cutoffs']) == CUTOFFS and contract['retrieval_k'] == K
    assert device != 'cuda' or torch.cuda.is_available()
    engine = common.Engine(2020)
    started = time.perf_counter()
    results = []
    for cutoff in CUTOFFS:
        if time.perf_counter() - started > 900:
            raise TimeoutError('15min retrieval diagnostic budget exceeded')
        results.append(one(cutoff, engine, device))
    return report(results, time.perf_counter() - started, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    try:
        run(args.device)
    except Exception:
        write(common.REPORT / 'BPR_RETRIEVAL_FAILURE.json', {'created_at': now(), 'traceback': traceback.format_exc(), 'final_week': 'not_run'})
        raise
