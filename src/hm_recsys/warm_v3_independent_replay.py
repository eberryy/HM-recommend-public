"""Independent full-rank replay of fixed three-expert Warm-v3 RRF results.

Reads saved predictions and raw truth; never trains, promotes, or changes scores.
Does not import or call the producer runner's rank or AP-decomposition helpers.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from .metrics import apk
from .warm_v2_contract import guard, now, read, write
from .warm_v3_common import ROOT, ART, REPORT, OLD, FRESH, TX


def literal(path):
    return "'"+str(path).replace("'", "''")+"'"


def stable_ranks(frame, score):
    """Independent pandas ranking of the FULL candidate set with original ties."""
    ordered = frame.sort_values(['customer_id', score, 'candidate_rank', 'article_id'],
        ascending=[True, False, True, True], kind='stable')
    result = np.empty(len(frame), np.int32)
    result[ordered.index.to_numpy()] = ordered.groupby('customer_id', sort=False).cumcount().to_numpy()+1
    return result


def ap_contributions(top, truthsets):
    result = {}
    for user, group in top.groupby('customer_id', sort=False):
        hits = 0
        denominator = min(len(truthsets[user]), 12)
        for row in group.itertuples():
            if row.article_id in truthsets[user]:
                hits += 1
                result[(user, row.article_id)] = (hits/row.final_rank)/denominator
    return result


def decomposition(before, after, users):
    old, new = set(before), set(after)
    return {'introduced_true_pairs': sum(after[k] for k in new-old)/users,
        'lost_true_pairs': -sum(before[k] for k in old-new)/users,
        'shared_true_pairs_rank_and_precision_change': sum(after[k]-before[k] for k in new & old)/users}


def replay_window(window, meta, family, selected_trial, producer_trial, year):
    cutoff = guard(meta['cutoff'])
    started = time.perf_counter()
    original = FRESH if year == 2019 else OLD
    prefix = 'FRESH' if year == 2019 else 'WV2'
    sources = {'b': original/f'{prefix}-000'/window,
               'n': original/f'{prefix}-501'/window,
               'x': ART/producer_trial/window,
               'f': ART/selected_trial/window}
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        con.execute("SET memory_limit='2GB'")
        con.execute("SET temp_directory=''")
        con.execute('SET enable_progress_bar=false')
        for alias, path in sources.items():
            con.execute(f'ATTACH {literal(path/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
        counts = {a: con.execute(f'''SELECT count(*),count(DISTINCT (customer_id,article_id)),
            count(DISTINCT customer_id) FROM {a}.predictions''').fetchone() for a in sources}
        assert len(set(counts.values())) == 1 and counts['b'][0] == counts['b'][1]
        for alias in ['n', 'x', 'f']:
            mismatch = con.execute(f'''SELECT count(*) FROM b.predictions b FULL JOIN {alias}.predictions z
                USING(customer_id,article_id) WHERE b.customer_id IS NULL OR z.customer_id IS NULL
                OR b.target<>z.target OR b.candidate_rank<>z.candidate_rank
                OR b.user_history_events_12w<>z.user_history_events_12w''').fetchone()[0]
            assert mismatch == 0
        con.execute(f'''CREATE TEMP TABLE raw_truth AS SELECT DISTINCT customer_id,article_id
            FROM read_parquet({literal(TX)}) WHERE t_dat>=DATE '{cutoff}'
            AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        cohort_error = con.execute('''SELECT count(*) FROM
            (SELECT DISTINCT customer_id FROM raw_truth) t FULL JOIN
            (SELECT DISTINCT customer_id FROM f.predictions) p USING(customer_id)
            WHERE t.customer_id IS NULL OR p.customer_id IS NULL''').fetchone()[0]
        label_error = con.execute('''SELECT count(*) FROM f.predictions p LEFT JOIN raw_truth t
            USING(customer_id,article_id) WHERE p.target<>CASE WHEN t.article_id IS NULL THEN 0 ELSE 1 END''').fetchone()[0]
        assert cohort_error == label_error == 0
        frame = con.execute('''SELECT b.customer_id,b.article_id,b.candidate_rank,b.target,
            b.user_history_events_12w,b.score e0,n.score e1,x.score e2,f.score stored_fusion
            FROM b.predictions b JOIN n.predictions n USING(customer_id,article_id)
            JOIN x.predictions x USING(customer_id,article_id)
            JOIN f.predictions f USING(customer_id,article_id)
            ORDER BY customer_id,candidate_rank,article_id''').fetchdf()
        actual = {a: con.execute(f'''SELECT customer_id,article_id,final_rank
            FROM {a}.top12 ORDER BY customer_id,final_rank''').fetchdf() for a in ['x', 'f']}
        truth = con.execute('SELECT * FROM raw_truth ORDER BY customer_id,article_id').fetchdf()
    assert len(frame) == counts['b'][0]
    sizes = frame.groupby('customer_id', sort=False).size()
    assert sizes.between(100, 300).all()
    for name in ['e0', 'e1', 'e2']:
        frame['r_'+name] = stable_ranks(frame, name)
    frame['baseline'] = 1/(60+frame.r_e0)+1/(60+frame.r_e1)
    frame['replayed_fusion'] = frame.baseline+1/(60+frame.r_e2)
    error = float(np.abs(frame.replayed_fusion-frame.stored_fusion).max())
    assert error < 1e-15
    frame['r_baseline'] = stable_ranks(frame, 'baseline')
    frame['r_fusion'] = stable_ranks(frame, 'replayed_fusion')
    inactive = frame.user_history_events_12w.to_numpy() == 0
    truthsets = {u: set(g.article_id) for u, g in truth.groupby('customer_id', sort=False)}
    users = list(truthsets)
    predictions, positive_sets, contributions, aps = {}, {}, {}, {}
    for name in ['e0', 'e1', 'e2', 'baseline', 'fusion']:
        rank = np.where(inactive, frame.candidate_rank.to_numpy(), frame['r_'+name].to_numpy())
        top = frame.loc[rank <= 12, ['customer_id', 'article_id', 'target', 'candidate_rank']].copy()
        top['final_rank'] = rank[rank <= 12]
        top = top.sort_values(['customer_id', 'final_rank'], kind='stable').reset_index(drop=True)
        if name in ['e2', 'fusion']:
            pd.testing.assert_frame_equal(top[['customer_id', 'article_id', 'final_rank']],
                actual['x' if name == 'e2' else 'f'], check_dtype=False)
        predictions[name] = {u: g.article_id.tolist() for u, g in top.groupby('customer_id', sort=False)}
        assert set(predictions[name]) == set(truthsets)
        assert all(len(v) == len(set(v)) == 12 for v in predictions[name].values())
        aps[name] = np.array([apk(list(truthsets[u]), predictions[name][u]) for u in users])
        positive_sets[name] = set(zip(top.loc[top.target == 1, 'customer_id'], top.loc[top.target == 1, 'article_id']))
        contributions[name] = ap_contributions(top, truthsets)
    inactive_users = set(frame.loc[inactive, 'customer_id'])
    inactive_error = sum(predictions['fusion'][u] != predictions['baseline'][u] for u in inactive_users)
    assert inactive_error == 0
    assert abs(aps['fusion'].mean()-meta['systems']['fusion']['covered_map']) < 1e-12
    assert abs(aps['baseline'].mean()-meta['baseline_full_MAP']) < 1e-12
    positive_counts = frame.groupby('customer_id', sort=False).target.sum().to_dict()
    recall = np.mean([positive_counts[u]/len(truthsets[u]) for u in users])
    oracle = np.mean([min(positive_counts[u], 12)/min(len(truthsets[u]), 12) for u in users])
    delta = aps['fusion']-aps['baseline']
    parts = decomposition(contributions['baseline'], contributions['fusion'], len(users))
    assert abs(sum(parts.values())-delta.mean()) < 1e-12
    old, new = positive_sets['baseline'], positive_sets['fusion']
    gained, lost = new-old, old-new
    feature_gain = meta['fit']['feature_importance']
    new_features = meta['fit']['features'][84:]
    assert len(new_features) == 2
    family_gain = sum(v['gain'] for v in feature_gain if v['feature'] in new_features)
    total_gain = sum(v['gain'] for v in feature_gain)
    result = {'candidate_rows': len(frame), 'users': len(users), 'truth_pairs': len(truth),
        'candidate_size_range': [int(sizes.min()), int(sizes.max())],
        'candidate_identity_errors': 0, 'duplicate_pairs': 0,
        'raw_cohort_errors': cohort_error, 'raw_label_errors': label_error,
        'full_fusion_score_max_error': error, 'standalone_top12_exact': True,
        'fusion_top12_exact': True, 'inactive_users': len(inactive_users),
        'inactive_user_order_errors': inactive_error, 'candidate_recall': float(recall),
        'candidate_oracle_MAP': float(oracle), 'MAP': {k: float(v.mean()) for k, v in aps.items()},
        'delta': float(delta.mean()), 'users_improved': int((delta > 1e-15).sum()),
        'users_harmed': int((delta < -1e-15).sum()), 'users_unchanged': int((np.abs(delta) <= 1e-15).sum()),
        'gross_positive_MAP': float(delta[delta > 0].sum()/len(users)),
        'gross_negative_MAP': float(delta[delta < 0].sum()/len(users)),
        'new_expert_only_correct_top12_pairs_vs_E0_E1_union': len(positive_sets['e2']-(positive_sets['e0'] | positive_sets['e1'])),
        'new_expert_shared_correct_top12_pairs': len(positive_sets['e2'] & (positive_sets['e0'] | positive_sets['e1'])),
        'new_expert_missed_union_correct_top12_pairs': len((positive_sets['e0'] | positive_sets['e1'])-positive_sets['e2']),
        'fusion_gained_correct_pairs': len(gained), 'fusion_lost_correct_pairs': len(lost),
        'AP_contribution_decomposition': parts, 'new_expert_score_Pearson_vs_BPR_tree': float(frame.e2.corr(frame.e1)),
        'new_feature_gain_share': family_gain/total_gain if total_gain else 0,
        'new_feature_names': new_features,
        'tree_rounds': meta['fit']['rounds'], 'runtime_seconds': time.perf_counter()-started}
    print({'independent_replay': window, 'rows': len(frame), 'score_error': error, 'delta': result['delta']}, flush=True)
    return result


def replay(family, selected_trial, producer_trial, year=2020, output_prefix=None):
    suffix = '_2019' if year == 2019 else ''
    source = REPORT/f'{selected_trial}{suffix}_OUTER.json'
    report = read(source)
    assert report['family'] == family and report['chosen_ordering'] == 'fusion'
    assert report['producer_model_trial'] == producer_trial and not report['candidate_pool_changed']
    output_prefix = output_prefix or f'{selected_trial}{suffix}_INDEPENDENT_REPLAY'
    if Path(output_prefix).name != output_prefix or '/' in output_prefix or '\\' in output_prefix:
        raise ValueError('Output prefix must be a filename stem, not a path')
    started = time.perf_counter()
    windows = {}
    for window, metadata in report['windows'].items():
        windows[window] = replay_window(window, metadata, family, selected_trial, producer_trial, year)
        gc.collect()
    assert len(windows) == 4
    value = {'created_at_utc': now(), 'family': family, 'selected_trial': selected_trial,
        'producer_trial': producer_trial, 'year': year, 'source_report': str(source),
        'review_type': 'independent_pandas_full_rank_and_raw_truth_replay_no_training',
        'windows': windows, 'candidate_rows_total': sum(v['candidate_rows'] for v in windows.values()),
        'baseline_mean_MAP': float(np.mean([v['MAP']['baseline'] for v in windows.values()])),
        'fusion_mean_MAP': float(np.mean([v['MAP']['fusion'] for v in windows.values()])),
        'mean_delta': float(np.mean([v['delta'] for v in windows.values()])),
        'nondegrade': sum(v['delta'] >= 0 for v in windows.values()), 'mechanics_passed': True,
        'runtime_seconds': time.perf_counter()-started, 'final_week': 'not_run',
        'definitions': {'positive_pairs': '单位为命中原验证周真值的用户—商品对，模型独有指Top12集合差而非召回源差。',
            'AP_contribution_decomposition': '新进入/移出/共享Top12真值三类AP贡献；共享项含名次与前方累计精度变化，所有贡献都除以完整评测用户数。',
            'MAP': '每用户以min(完整真值商品数,12)归一化，再对原哈希10%完整用户集合平均；不是只对命中用户平均。'},
        'limits': ['Saved predictions replayed; no repeat model inference or factor rehash.',
                   'Only unchanged100-300candidate pool and fixed three-expert reciprocal-rank fusion supported.'],
        'reproduce': f'python -m hm_recsys.warm_v3_independent_replay --family {family} --selected-trial {selected_trial} --producer-trial {producer_trial} --year {year} --output-prefix {output_prefix}'}
    write(REPORT/f'{output_prefix}.json', value)
    render_report(value, output_prefix)
    return value


def render_report(value, output_prefix):
    """Render already audited evidence; permits report-only edits without rereading labels."""
    windows = value['windows']
    selected_trial, producer_trial, family = (value[k] for k in ['selected_trial', 'producer_trial', 'family'])
    rows = '\n'.join(f"| {w} | {v['candidate_rows']:,} | {v['users']:,} | {v['MAP']['baseline']:.9f} | {v['MAP']['fusion']:.9f} | {v['delta']:+.9f} |" for w, v in windows.items())
    parts = '\n'.join(
        f"| {w} | {v['new_expert_only_correct_top12_pairs_vs_E0_E1_union']} | "
        f"{v['fusion_gained_correct_pairs']} | {v['fusion_lost_correct_pairs']} | "
        f"{v['AP_contribution_decomposition']['introduced_true_pairs']:+.9f} | "
        f"{v['AP_contribution_decomposition']['lost_true_pairs']:+.9f} | "
        f"{v['AP_contribution_decomposition']['shared_true_pairs_rank_and_precision_change']:+.9f} |"
        for w, v in windows.items())
    replacement = float(np.mean([v['AP_contribution_decomposition']['introduced_true_pairs']+
        v['AP_contribution_decomposition']['lost_true_pairs'] for v in windows.values()]))
    shared = float(np.mean([v['AP_contribution_decomposition']['shared_true_pairs_rank_and_precision_change']
        for v in windows.values()]))
    text = f'''# {selected_trial} 独立完整排序重放

生产专家：{producer_trial}；机制：{family}。只读保存预测与原始交易真值，不训练、不晋级。

复核 {value['candidate_rows_total']:,} 个用户—候选商品对，分数、Top12、完整真值分母和无近期历史用户回退全部通过。用户与商品对按每窗独立计数。

| 窗口 | 候选行 | 完整评测用户 | 冻结基线 MAP | 新融合 MAP | 增量 |
|---|---:|---:|---:|---:|---:|
{rows}

平均增量：{value['mean_delta']:+.9f}。这是机制核对，不是自动晋级决定。

机器报告中的 AP_contribution_decomposition（本项目AP贡献分解）将新进入、移出、共享Top12真值分别计数并累加AP贡献；共享项含自身名次及前方累计精度变化。分母均为完整评测用户数。

## 互补性与得失分解

“新专家独有”指新专家单独排序进入Top12、但原84特征排序器与BPR特征排序器的Top12并集未命中的正确用户—商品对；它不是召回源独有正例。新增/丢失则比较实际三专家融合与冻结双专家融合的Top12集合，单位也为用户—商品对。两种参照集合不同。

| 窗口 | 新专家独有正确对 | 融合新增正确对 | 融合丢失正确对 | 新增AP贡献 | 丢失AP贡献 | 共享正例重排贡献 |
|---|---:|---:|---:|---:|---:|---:|
{parts}

四窗等权平均：Top12正确对替换的净贡献为 {replacement:+.9f}，原已命中正例的名次与累计精度变化贡献为 {shared:+.9f}，两项相加为总增量 {value['mean_delta']:+.9f}。这一区分说明增加正确商品个数不必然提高MAP，因为原有正确商品位置和前方命中精度同样重要。

此为分数与排序结果的描述性分解，不证明季节性、表示压缩或模型容量是因果原因；不得据此调整已经暴露外层结果的方案权重。

## 口径和边界

- MAP@12（行业评测指标）：每个用户只看前12名，按每次命中的当前精度累加，除以该用户完整未来一周真值商品数与12的较小者，再对原哈希10%完整用户群等权平均。
- candidate_recall（机器字段）：先对每个用户计算完整候选中命中的真值数除以完整真值商品数，再对完整用户群等权平均；candidate_oracle_MAP是只重排现有候选时、把命中的真值全部排在最前的理论MAP上限，不是模型实际指标。
- new_expert_score_Pearson_vs_BPR_tree：在该窗全部候选行上，新专家与BPR特征树模型原始分数的皮尔逊相关系数；不表示按用户等权，也不说明独立因果贡献。
- new_feature_gain_share：新专家两列信号占该专家所有树分裂增益的比例，只是训练模型的分裂统计，不是验证MAP贡献。
- 完整候选数固定为每用户100–300个。没有近期历史的用户仍按原候选名次回退；他们没有从MAP分母删除。审计耗时 {value['runtime_seconds']:.2f} 秒。

使用 pandas 稳定排序重建完整候选模型名次，以独立官方APK函数核对最终推荐；未复用生产运行器的排名函数。最终周始终未运行。

复现命令（在已配置环境、PYTHONPATH=src 下执行）：

```text
{value['reproduce']}
```
'''
    (REPORT/f'{output_prefix}.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True)
    parser.add_argument('--selected-trial', required=True)
    parser.add_argument('--producer-trial', required=True)
    parser.add_argument('--year', type=int, choices=[2019, 2020], default=2020)
    parser.add_argument('--output-prefix')
    args = parser.parse_args()
    replay(args.family, args.selected_trial, args.producer_trial, args.year, args.output_prefix)
