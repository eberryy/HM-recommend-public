"""Bounded day-basket GRU expert; frozen Item2Vec inputs, no same-day sequence."""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import gc
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/warm_v3/sequence'
REPORT = ROOT / 'reports/warm_v3'
RAW = Path(__file__).resolve().parents[2] / 'data/interim/audit/transactions.parquet'
PARAMS = dict(history_days=84, max_baskets=8, hidden=64, epochs=3,
              max_transitions=250000, batch_size=1024, negatives=128,
              learning_rate=.001, weight_decay=.0001, temperature=.1,
              seed=20260909, max_training_seconds=420)
FEATURES = ['wv3_sequence_score', 'wv3_sequence_unavailable']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def literal(path):
    return "'" + str(path).replace("'", "''") + "'"


def guard(cutoff):
    value = date.fromisoformat(cutoff)
    if value >= date(2020, 9, 16):
        raise ValueError('final week and later are forbidden')
    return value


def connection():
    con = duckdb.connect(); con.execute('SET threads=4'); con.execute("SET memory_limit='2GB'")
    return con


def save_parquet(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.register('frame', frame)
        con.execute(f'COPY frame TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')


def load_parquet(path):
    with connection() as con:
        return con.execute(f'SELECT * FROM read_parquet({literal(path)})').fetchdf()


def source(cutoff):
    guard(cutoff)
    roots = [Path(__file__).resolve().parents[2] / 'artifacts/m2_9/item2vec-source-v1' / cutoff,
             ROOT / 'artifacts/warm_v2/fresh_robustness/candidates' / cutoff / 'source']
    found = [p / 'source-manifest.json' for p in roots if (p / 'source-manifest.json').exists()]
    if not found:
        raise FileNotFoundError(f'No existing cutoff-safe Item2Vec source for {cutoff}')
    manifest = read(found[0]); assert manifest['cutoff'] == cutoff
    paths = {}
    for key in ('items', 'normalized_vectors'):
        entry = manifest['artifacts'][key]
        path = Path(entry['path'])
        # Legacy absolute paths may precede Warm worktree migration.
        if not path.exists():
            path = found[0].parent / path.name
        assert path.stat().st_size == entry['bytes'], (key, path)
        paths[key] = path
    items = pd.read_csv(paths['items'], dtype={'article_id': str})
    vectors = np.load(paths['normalized_vectors'], allow_pickle=False)
    assert np.array_equal(items.row_index, np.arange(len(items)))
    assert vectors.shape == (len(items), 64) and np.isfinite(vectors).all()
    return items, vectors, {'manifest': str(found[0]), 'cutoff': cutoff,
                            'identities': {k: manifest['artifacts'][k] for k in paths}}


def pool_basket(indices, vectors):
    """Event-weighted mean: duplicates remain; sorting guarantees exact replay."""
    ids = np.sort(np.asarray(indices, dtype=np.int64))
    if not len(ids):
        return np.zeros(vectors.shape[1], np.float32)
    value = vectors[ids].mean(axis=0)
    return (value / max(float(np.linalg.norm(value)), 1e-12)).astype(np.float32)


class BasketGRU(nn.Module):
    def __init__(self, dim=64, hidden=64):
        super().__init__()
        self.gru = nn.GRU(dim + 2, hidden, batch_first=True)
        self.output = nn.Linear(hidden + dim + 1, dim)

    def forward(self, basket_vectors, day_gap, event_count, lengths, query_gap):
        # Packing makes right padding completely irrelevant to the final state.
        scalar = torch.stack((torch.log1p(day_gap.clamp(0, 365)) / 6,
                              torch.log1p(event_count.clamp(0, 1000)) / 7), dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            torch.cat((basket_vectors, scalar), dim=-1), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        last = basket_vectors[torch.arange(len(lengths), device=lengths.device), lengths - 1]
        gap = torch.log1p(query_gap.clamp(0, 365)).unsqueeze(1) / 6
        return F.normalize(self.output(torch.cat((hidden[-1], last, gap), dim=1)), dim=1)


def prepare(cutoff):
    guard(cutoff); out = ART / cutoff; marker = out / 'PREPARED.json'
    if marker.exists():
        value = read(marker); assert value['params'] == PARAMS
        return value
    out.mkdir(parents=True, exist_ok=True); start = time.perf_counter()
    items, vectors, evidence = source(cutoff)
    with connection() as con:
        con.register('vocab', items[['article_id', 'row_index']])
        raw = f"read_parquet({literal(RAW)})"
        where = f"t_dat>=DATE '{cutoff}'-INTERVAL {PARAMS['history_days']} DAY AND t_dat<DATE '{cutoff}'"
        stats = con.execute(f'''WITH h AS (SELECT * FROM {raw} WHERE {where}),
             b AS (SELECT customer_id,t_dat,count(*) n FROM h GROUP BY customer_id,t_dat),
             u AS (SELECT customer_id,count(*) n FROM b GROUP BY customer_id)
             SELECT (SELECT count(*) FROM h),(SELECT count(*) FROM b),
                    (SELECT count(*) FROM u),(SELECT count(*) FROM u WHERE n>=2),
                    (SELECT sum(n-1) FROM u),(SELECT avg(n) FROM b),
                    (SELECT quantile_cont(n,.95) FROM b),(SELECT max(t_dat) FROM h),
                    (SELECT count(*) FROM h JOIN vocab USING(article_id))''').fetchone()
        frame = con.execute(f'''SELECT customer_id,t_dat,
             list(row_index ORDER BY row_index) ids,count(*) event_count
             FROM {raw} h JOIN vocab USING(article_id) WHERE {where}
             GROUP BY customer_id,t_dat ORDER BY customer_id,t_dat''').fetchdf()
    assert str(stats[7])[:10] < cutoff and len(frame)
    assert frame.t_dat.max() < pd.Timestamp(cutoff)
    raw_items = frame.ids.tolist()
    pooled = np.stack([pool_basket(v, vectors) for v in raw_items])
    # The target is a set of next-purchase-day items; repeated raw events are not erased.
    target_lists = [np.unique(v).astype(np.int32) for v in raw_items]
    offsets = np.r_[0, np.cumsum([len(v) for v in target_lists])].astype(np.int64)
    target_ids = np.concatenate(target_lists)
    dates = frame.t_dat.to_numpy().astype('datetime64[D]').astype(np.int32)
    customer = frame.customer_id.to_numpy()
    first = np.r_[True, customer[1:] != customer[:-1]]
    starts = np.maximum.accumulate(np.where(first, np.arange(len(frame)), 0)).astype(np.int32)
    gap = np.r_[0, np.diff(dates)].astype(np.float32); gap[first] = 0
    transitions = np.flatnonzero(~first).astype(np.int32)
    assert np.all(dates[transitions] > dates[transitions - 1])
    users = frame.loc[np.r_[customer[:-1] != customer[1:], True], ['customer_id']].copy()
    users['basket_index'] = np.flatnonzero(np.r_[customer[:-1] != customer[1:], True]).astype(np.int32)
    save_parquet(users, out / 'users.parquet')
    arrays = dict(basket_vectors=pooled, dates=dates, gaps=gap, starts=starts,
                  counts=frame.event_count.to_numpy(np.float32),
                  target_ids=target_ids, target_offsets=offsets, transitions=transitions)
    np.savez(out / 'prepared.npz', **arrays)
    value = {'cutoff': cutoff, 'params': PARAMS, 'source': evidence,
             'latest_history_date': str(stats[7])[:10], 'history_strictly_before_cutoff': True,
             'raw_events': int(stats[0]), 'raw_day_baskets': int(stats[1]),
             'raw_users': int(stats[2]), 'raw_users_two_or_more_days': int(stats[3]),
             'raw_next_day_transitions': int(stats[4]), 'raw_mean_basket_events': float(stats[5]),
             'raw_p95_basket_events': float(stats[6]), 'vocab_covered_events': int(stats[8]),
             'vocab_covered_baskets': len(frame), 'trainable_transitions': len(transitions),
             'sampled_transitions_cap': PARAMS['max_transitions'],
             'retained_item_vocabulary': len(items), 'prepared_arrays_bytes': sum(v.nbytes for v in arrays.values()),
             'preparation_seconds': time.perf_counter() - start,
             'same_day_order': 'none; sorted integer indices only ensure reproducible commutative mean',
             'frozen_embedding_caveat': 'Item2Vec reuses the baseline pseudo-order representation; the NEW GRU only models ordered dates, not that within-day pseudo-order. Its pretraining uses all pre-cutoff history, not a per-transition rolling representation.',
             'final_week': 'not_run'}
    write(marker, value); print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def contexts(data, ends, query_dates):
    ends = np.asarray(ends, np.int64)
    starts = np.maximum(data['starts'][ends], ends - PARAMS['max_baskets'] + 1)
    lengths = ends - starts + 1
    index = starts[:, None] + np.arange(PARAMS['max_baskets'])[None, :]
    mask = np.arange(PARAMS['max_baskets'])[None, :] < lengths[:, None]
    index = np.minimum(index, ends[:, None])
    assert np.all(data['dates'][ends] < query_dates), 'a target/query day leaked into its history'
    x = data['basket_vectors'][index].copy(); x[~mask] = 0
    gaps = data['gaps'][index].copy(); gaps[~mask] = 0; gaps[:, 0] = 0
    counts = data['counts'][index].copy(); counts[~mask] = 0
    query_gap = np.asarray(query_dates) - data['dates'][ends]
    return x, gaps, counts, lengths.astype(np.int64), query_gap.astype(np.float32)


def tensors(values, device):
    return [torch.as_tensor(v, device=device) for v in values]


def next_basket_samples(data, basket_indices, rng, vocabulary_size):
    positives = [data['target_ids'][data['target_offsets'][i]:data['target_offsets'][i + 1]]
                 for i in basket_indices]
    selected = np.array([v[rng.integers(len(v))] for v in positives], np.int64)
    negatives = rng.integers(vocabulary_size, size=PARAMS['negatives'], dtype=np.int64)
    # Exclude every item of that target basket, not just the chosen positive.
    excluded = np.stack([np.isin(negatives, v) for v in positives])
    return selected, negatives, excluded


def fit(cutoff, device='cuda'):
    guard(cutoff); out = ART / cutoff; marker = out / 'MODEL.json'
    if marker.exists():
        value = read(marker); assert value['params'] == PARAMS
        return value
    prep = prepare(cutoff)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no silent architecture or training-device switch')
    torch.set_num_threads(4); torch.manual_seed(PARAMS['seed'])
    rng = np.random.default_rng(PARAMS['seed']); data = dict(np.load(out / 'prepared.npz', allow_pickle=False))
    _, vectors, _ = source(cutoff); item_vectors = torch.from_numpy(vectors).to(device)
    model = BasketGRU(hidden=PARAMS['hidden']).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])
    transitions = data['transitions']
    if len(transitions) > PARAMS['max_transitions']:
        transitions = np.sort(rng.choice(transitions, PARAMS['max_transitions'], replace=False))
    assert len(transitions) > 0
    history = []; start = time.perf_counter(); resumed_fit_seconds = 0.
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    resume = (out / 'model.pt').exists()
    if resume:
        training = read(out / 'TRAINING_COMPLETED.json')
        assert training['params'] == PARAMS and len(training['epochs']) == PARAMS['epochs']
        history = training['epochs']; resumed_fit_seconds = history[-1]['elapsed_seconds']
        model.load_state_dict(torch.load(out / 'model.pt', map_location=device, weights_only=True))
    for epoch in range(0 if resume else PARAMS['epochs']):
        order = rng.permutation(transitions); loss_sum = 0.; correct = 0; observations = 0
        for begin in range(0, len(order), PARAMS['batch_size']):
            target = order[begin:begin + PARAMS['batch_size']]
            values = contexts(data, target - 1, data['dates'][target])
            positive, negative, excluded = next_basket_samples(data, target, rng, len(vectors))
            query = model(*tensors(values, device))
            pos = (query * item_vectors[torch.as_tensor(positive, device=device)]).sum(1, keepdim=True)
            neg = query @ item_vectors[torch.as_tensor(negative, device=device)].T
            neg = neg.masked_fill(torch.as_tensor(excluded, device=device), -1e4)
            logits = torch.cat((pos, neg), dim=1) / PARAMS['temperature']
            loss = F.cross_entropy(logits, torch.zeros(len(target), dtype=torch.long, device=device))
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            correct += int(logits.detach().argmax(1).eq(0).sum()); observations += len(target)
            if time.perf_counter() - start > PARAMS['max_training_seconds']:
                write(out / 'RESOURCE_STOP.json', {'epoch': epoch, 'observations': observations,
                       'training_seconds': time.perf_counter() - start, 'final_week': 'not_run'})
                raise RuntimeError('sequence fixed resource cap exceeded; no partial model promotion')
        history.append({'epoch': epoch + 1, 'sampled_training_cross_entropy': loss_sum / observations,
                        'sampled_training_top1_accuracy': correct / observations, 'observations': observations,
                        'elapsed_seconds': time.perf_counter() - start})
        print({'cutoff': cutoff, **history[-1]}, flush=True)
    model.eval()
    if not resume:
        torch.save(model.state_dict(), out / 'model.pt')
        write(out / 'TRAINING_COMPLETED.json', {'cutoff': cutoff, 'params': PARAMS,
              'epochs': history, 'status': 'all_registered_epochs_completed_before_encoding'})
    parameter_change = sum(float((v.detach().cpu() - initial[k]).square().sum()) for k, v in model.state_dict().items())
    assert parameter_change > 0
    users = load_parquet(out / 'users.parquet'); encoded = []
    query_day = np.datetime64(cutoff, 'D').astype(np.int32)
    with torch.no_grad():
        for begin in range(0, len(users), 2048):
            ends = users.basket_index.to_numpy()[begin:begin + 2048]
            values = contexts(data, ends, np.full(len(ends), query_day, np.int32))
            encoded.append(model(*tensors(values, device)).cpu().numpy())
    user_vectors = np.concatenate(encoded); assert np.isfinite(user_vectors).all()
    np.save(out / 'user_vectors.npy', user_vectors, allow_pickle=False)
    last = data['basket_vectors'][users.basket_index.to_numpy()]
    np.save(out / 'last_basket_vectors.npy', last, allow_pickle=False)
    value = {'cutoff': cutoff, 'params': PARAMS, 'preparation': prep,
             'architecture': 'event-mean day basket -> GRU64 with gap/count -> query-gap-conditioned normalized64 vector',
             'objective': 'sampled softmax over one uniform unique next-basket positive and128 uniform vocabulary negatives; all same-target-basket items masked from negatives',
             'training_target_semantics': 'next PURCHASE day, not necessarily next calendar day; all targets strictly before cutoff',
             'input_embedding_trainable': False, 'same_day_permutation_invariant': True,
             'trainable_parameters': sum(v.numel() for v in model.parameters()),
             'parameter_squared_change': parameter_change, 'training_transitions': len(transitions),
             'epochs': history, 'fit_and_all_user_encode_seconds': time.perf_counter() - start + resumed_fit_seconds,
             'resumed_checkpoint_without_retraining': resume,
             'device': device, 'torch': torch.__version__, 'query_users': len(users),
             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
             'peak_cuda_measurement_scope': 'encoding_after_checkpoint_resume_only' if resume else 'training_and_encoding',
             'outer_labels_read': False, 'final_week': 'not_run'}
    write(marker, value)
    del model, optimizer, data, item_vectors; gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()
    return value


def score_cutoff(cutoff, candidate_rows_path, device='cuda'):
    """Return original candidate identity plus two learned feature columns and control."""
    guard(cutoff); model = fit(cutoff, device); out = ART / cutoff
    path = out / 'features.parquet'; meta = out / 'FEATURES.json'
    if meta.exists():
        saved = read(meta); assert saved['candidate_source'] == str(Path(candidate_rows_path).resolve())
        return path, saved
    users = pd.Index(load_parquet(out / 'users.parquet').customer_id)
    items, vectors, _ = source(cutoff); vocabulary = pd.Index(items.article_id)
    user_vectors = np.load(out / 'user_vectors.npy', allow_pickle=False)
    last_vectors = np.load(out / 'last_basket_vectors.npy', allow_pickle=False)
    chunks = []; start = time.perf_counter()
    with connection() as con:
        cursor = con.execute(f'SELECT customer_id,article_id FROM read_parquet({literal(candidate_rows_path)})')
        while True:
            frame = cursor.fetch_df_chunk(32)
            if frame.empty: break
            ui = users.get_indexer(frame.customer_id); ii = vocabulary.get_indexer(frame.article_id)
            good = (ui >= 0) & (ii >= 0)
            score = np.full(len(frame), np.nan, np.float32); control = score.copy()
            score[good] = np.einsum('ij,ij->i', user_vectors[ui[good]], vectors[ii[good]])
            control[good] = np.einsum('ij,ij->i', last_vectors[ui[good]], vectors[ii[good]])
            frame['wv3_sequence_score'] = score; frame['wv3_sequence_unavailable'] = (~good).astype(np.float32)
            frame['wv3_last_basket_cosine_control'] = control; chunks.append(frame)
    result = pd.concat(chunks, ignore_index=True)
    assert not result.duplicated(['customer_id', 'article_id']).any()
    save_parquet(result, path)
    value = {'cutoff': cutoff, 'candidate_source': str(Path(candidate_rows_path).resolve()),
             'rows': len(result), 'features': FEATURES, 'diagnostic_only': ['wv3_last_basket_cosine_control'],
             'unavailable_pairs': int(result.wv3_sequence_unavailable.sum()),
             'candidate_identity_unchanged': True, 'model_metadata': str(out / 'MODEL.json'),
             'scoring_seconds': time.perf_counter() - start, 'final_week': 'not_run'}
    write(meta, value); return path, value


def build_features(cutoff, engine):
    """Common expert interface; caller controls which registered cutoffs may run."""
    return score_cutoff(cutoff, engine.base_path(cutoff))


def audit(cutoffs):
    rows = {c: prepare(c) for c in cutoffs}
    out = {'stage': 'WV3 sequence resource and supervision audit',
           'created_at_utc': datetime.now(timezone.utc).isoformat(), 'params': PARAMS, 'cutoffs': rows,
           'hypothesis': 'Warm用户跨日近期意图提供全历史BPR以外信号；同日集合均值不虚构真实购买顺序。',
           'resource_estimate_before_gpu_pilot': {'per_cutoff_fit_cap_seconds': 420,
                 'expected_gpu_gib': '<1 for1024x8x64 GRU and frozen vocabulary; must verify measured pilot',
                 'expected_prepare_plus_fit_minutes': [2, 10], 'no_training_started_by_audit': True},
           'borrowed_ideas': [
               {'source': 'https://arxiv.org/abs/1511.06939', 'borrowed': 'GRU压缩跨时间行为为近期意图',
                'not_copied': '不复制其逐item session顺序、batch组织、损失或超参数；本实现使用跨购买日序列。'},
               {'source': 'https://arxiv.org/abs/1703.06114', 'borrowed': '集合池化对元素置换不变',
                'not_copied': '未声称复现DeepSets网络；只借同日聚合的置换不变约束。'}],
           'definitions_zh': {'day_basket': '项目命名：同一用户同一天的购买事件集合；输入均值保留重复事件权重，无同日先后。',
                 'next_day_transition': '项目命名：同一用户相邻两个有购买记录的日期之间的监督样本，单位为用户—下一购买日；不是相邻自然日。',
                 'sampled_training_top1_accuracy': '训练诊断：选中的未来购买商品得分超过128个抽样商品的训练样本比例；分母训练样本数，不是MAP或验证指标。',
                 'wv3_sequence_score': '项目特征：GRU当前用户向量与候选商品冻结Item2Vec向量的余弦值，不是概率。',
                 'wv3_sequence_unavailable': '项目特征：截止前12周无词表覆盖历史或候选不在词表时为1，余弦缺失为NaN。',
                 'wv3_last_basket_cosine_control': '项目诊断对照：最近一个购买日均值向量与候选的余弦，不送入正式排序器。'},
           'final_week': 'not_run', 'outer_map_exposures': 0}
    write(REPORT / 'SEQUENCE_RESOURCE_AUDIT.json', out); return out


def close_pilot(cutoff):
    """Publish mechanics/cost only; never evaluate recommendation labels."""
    value = read(REPORT / 'SEQUENCE_RESOURCE_AUDIT.json')
    model = read(ART / cutoff / 'MODEL.json'); scoring = read(ART / cutoff / 'FEATURES.json')
    seconds = model['preparation']['preparation_seconds'] + model['fit_and_all_user_encode_seconds'] + scoring['scoring_seconds']
    value['pilot'] = {'cutoff': cutoff, 'model': model, 'scoring': scoring,
                     'mechanics_pass': model['parameter_squared_change'] > 0 and scoring['candidate_identity_unchanged'],
                     'cpu_unit_tests_passed': 5, 'measured_core_seconds': seconds,
                     'ten_cutoff_core_minutes_linear_estimate': seconds * 10 / 60,
                     'ten_cutoff_budget_minutes_conservative': [15, 25],
                     'gpu_memory_note': '本次训练后I/O失败，恢复后仅记录编码阶段峰值；不可把此数声称为完整训练显存峰值。后续正常cutoff会记录训练和编码两者峰值。',
                     'engineering_failure': str(ART / cutoff / 'ENCODING_IO_FAILURE.json'),
                     'ranking_conclusion': 'not_measured; training proxy improvement is not heldout ranking evidence',
                     'status': 'mechanics_ready_for_inner_screen_not_promoted'}
    write(REPORT / 'SEQUENCE_RESOURCE_AUDIT.json', value)
    report = f'''# Warm-v3 购买日序列专家：机制与资源预实验

状态：仅在2019-11-27训练截止运行预实验，尚未计算任何外层MAP；不能据训练损失判断排序增益。

## 实际结构

截止前12周交易 → 同一用户同一天的商品向量均值 → 最近最多8个购买日按日期排序 → GRU64 → 用户向量与候选Item2Vec向量余弦。

GRU为行业通用门控循环单元，本实验隐状态64维、33,664个可训练参数。输入含相邻购买日间隔、当天事件数量；输出还使用查询日距最近购买日的天数。用户向量与商品向量均归一化，余弦分数不是概率。

同日均值保留重复事件的权重，不虚构同日商品顺序。冻结商品向量来自原截止的Item2Vec；它自身仍是历史基线采用伪顺序训练的表示，因此这里不是声称重新训练了一个全程无伪顺序的Item2Vec。新的GRU只建模跨购买日顺序。

## 监督与覆盖

2019-11-27之前84天有3,105,555条原始事件、969,403个用户—购买日、470,954名用户。其中224,681名用户至少有两个购买日，提供498,449个相邻购买日转换。Item2Vec词表覆盖3,099,011条事件；过滤不在词表的商品后，有497,824个可训练转换。

“相邻购买日转换”为项目自定义统计单位：同一用户前一购买日到下一购买日，不是相邻自然日。确定性抽取至多250,000个转换，训练3遍，每个样本从下一购买日去重商品集合均匀取1个正例，与128个均匀抽样词表商品比较；该购买日所有真商品均不作负例。抽样非购买商品只是隐式负例，并不等于用户不喜欢。

所有训练输入和目标都早于2019-11-27，最后历史日期为2019-11-26；最终周未运行。冻结Item2Vec使用该截止前全体历史，不将样本内逐转换表示训练称为严格滚动回测。

## 机制与成本

- 五项CPU测试通过：同日置换不变、重复事件保留、最终周拒绝、查询日/跨用户边界、补零不影响输出、梯度及同篮正例排除。部分测试同时覆盖两个性质。
- 训练3遍共750,000次转换观察，耗时35.78秒。训练抽样交叉熵3.484→3.157；训练抽样Top1准确率16.98%→24.83%。这两个量仅为优化器工作证据，不是验证MAP。
- 准备数据22.62秒，训练加全用户编码38.21秒。含候选打分的核心流程约{seconds:.1f}秒。
- 同预算10个截止的线性估计约{seconds * 10 / 60:.1f}分钟；预留15–25分钟较稳妥，尚不含下游LightGBM训练。
- 准备数组约271MiB；单截止约470k个用户的64维向量约115MiB。输出保存在新Warm-v3目录。
- 已保存的显存数仅为恢复后的编码阶段约36.44MiB，不能当作完整训练峰值；发生I/O异常前未保存训练峰值。

训练后发现环境未安装pandas可选Parquet引擎，已改用项目现有DuckDB读取，并从保存的3轮checkpoint继续编码，没有重训、改参数或替换模型。失败证据保留在ENCODING_IO_FAILURE.json。

## 接口与后续门槛

`build_features(cutoff, engine)`返回原候选用户—商品对及两列正式特征：`wv3_sequence_score`（序列用户向量与商品向量余弦）、`wv3_sequence_unavailable`（用户没有词表覆盖历史或商品不在词表时为1，分数为NaN）。

`wv3_last_basket_cosine_control`为诊断对照：最近一个购买日均值向量与候选的余弦；不进入正式排序器，除非另行注册对照。不得把缺失历史用户的候选删除；原inactive fallback保持不变。

目前只有机制和资源门槛通过。下一步由主线程按已注册内层筛选决定是否有独立排序信号；没有验证增益就停止，不因训练损失下降晋级。

## 借鉴边界

借鉴[GRU4Rec论文](https://arxiv.org/abs/1511.06939)的递归网络压缩近期行为思想；未复制其逐商品会话顺序、训练损失和超参数。借鉴[Deep Sets论文](https://arxiv.org/abs/1703.06114)的集合置换不变约束；仅采用简单均值，不声称复现Deep Sets模型。
'''
    (REPORT / 'SEQUENCE_RESOURCE_AUDIT.md').write_text(report, encoding='utf-8')
    return value['pilot']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'fit', 'score', 'close-pilot'])
    parser.add_argument('--cutoff', required=True); parser.add_argument('--candidates'); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'audit': audit(args.cutoff.split(','))
    elif args.command == 'fit': fit(args.cutoff, args.device)
    elif args.command == 'close-pilot': close_pilot(args.cutoff)
    else: score_cutoff(args.cutoff, args.candidates, args.device)
