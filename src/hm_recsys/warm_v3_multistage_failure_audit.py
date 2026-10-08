"""Saved501 checkpoint diagnosis: IN-SAMPLE2019 fit and original INNER only, no fit."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from . import warm_v3_common as c
from . import warm_v3_multistage as model_code
from .warm_v2_engine import literal


def read_frame(path):
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        return con.execute(f'SELECT * FROM read_parquet({literal(path)})').fetchdf()


def save_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.register('frame', frame)
        con.execute(f'COPY frame TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')


def aps(labels, truth_count, ordering):
    y = np.take_along_axis(labels, ordering[:, :12], axis=1).astype(np.float64)
    return (y*np.cumsum(y, axis=1)/np.arange(1, 13)).sum(axis=1)/np.minimum(truth_count, 12)


def infer(top, model, mean, std):
    residuals = []
    train_eval_errors = []
    losses = {'initial_dynamic': [], 'final_dynamic': [], 'final_frozen_initial_weights': []}
    per_user_rows = []
    with torch.no_grad():
        for lo in range(0, len(top['x']), 256):
            x = ((top['x'][lo:lo+256]-mean)/std).astype(np.float32)
            tx = torch.from_numpy(x)
            rank = torch.from_numpy(top['stage1_rank'][lo:lo+256])
            labels = torch.from_numpy(top['target'][lo:lo+256])
            truth = torch.from_numpy(top['truth_count'][lo:lo+256])
            model.eval()
            score, residual = model(tx, rank)
            model.train()
            train_score, train_residual = model(tx, rank)
            model.eval()
            train_eval_errors.append(float((train_score-score).abs().max()))
            assert torch.equal(train_residual, residual)
            initial = -torch.log(rank)
            wi = model_code.delta_ap_weights(labels, initial, truth)
            wf = model_code.delta_ap_weights(labels, score, truth)
            initial_comparison = F.softplus(-10*(initial[:, :, None]-initial[:, None, :]))
            final_comparison = F.softplus(-10*(score[:, :, None]-score[:, None, :]))
            penalty = .001*residual.square().mean(dim=1)
            values = {'initial_dynamic': (wi*initial_comparison).sum(dim=(1, 2))/50,
                      'final_dynamic': (wf*final_comparison).sum(dim=(1, 2))/50+penalty,
                      'final_frozen_initial_weights': (wi*final_comparison).sum(dim=(1, 2))/50+penalty}
            for name, value in values.items():
                losses[name].append(value.numpy())
            residuals.append(residual.numpy())
    return np.concatenate(residuals), {k: np.concatenate(v) for k, v in losses.items()}, max(train_eval_errors)


def audit_window(meta, role, model, mean, std, feature_names, stored_inner=None):
    cutoff = meta['cutoff']
    c.guard(cutoff)
    # Never create/rebuild a source cache as a side effect of this read-only audit.
    assert (c.ART/'candidate_gate_data'/cutoff/'DATA.json').exists()
    top, info, keys = model_code.top50_data(meta)
    assert info['features'] == feature_names
    residual, losses, mode_error = infer(top, model, mean, std)
    scores = model_code.residual_scores(top['stage1_rank'], residual)
    candidate_rank = keys.candidate_rank.to_numpy()[top['array_row']]
    ordering = np.lexsort((top['article_id'], candidate_rank, -scores), axis=1)
    original_order = np.tile(np.arange(50), (len(ordering), 1))
    baseline_ap = aps(top['target'], top['truth_count'], original_order)
    final_ap = aps(top['target'], top['truth_count'], ordering)
    # Existing source includes the full cohort for2019, and covered-active only
    # for2020INNER. Inactive contribution is unchanged, never removed from denominator.
    inactive = keys[keys.user_history_events_12w == 0]
    inactive_ap = 0.
    for _, group in inactive.groupby('customer_id', sort=False):
        y = group.sort_values(['candidate_rank', 'article_id']).target.to_numpy()[:12]
        inactive_ap += float(np.sum(y*np.cumsum(y)/np.arange(1, len(y)+1))/min(group.truth_count.iloc[0], 12))
    baseline_population = float((baseline_ap.sum()+inactive_ap)/meta['total_users'])
    final_population = float((final_ap.sum()+inactive_ap)/meta['total_users'])
    assert abs(baseline_population-meta['baseline_map_population_component']) < 1e-12
    replay = {'train_eval_score_max_error': mode_error,
              'same_feature_order': True, 'complete_truth_normalizer': True,
              'source_identity': info['source_identity']}
    if stored_inner is not None:
        stored = read_frame(stored_inner['rank_path']).set_index(['customer_id', 'article_id'], verify_integrity=True)
        index = pd.MultiIndex.from_arrays([np.repeat(top['customer_id'], 50), top['article_id'].reshape(-1)])
        actual = stored.reindex(index)
        assert actual.score.notna().all()
        replay['stored_inner_score_max_error'] = float(np.abs(actual.score.to_numpy()-scores.reshape(-1)).max())
        replay['stored_inner_residual_max_error'] = float(np.abs(actual.residual.to_numpy()-residual.reshape(-1)).max())
        assert replay['stored_inner_score_max_error'] < 1e-12
        inverse = np.empty_like(ordering)
        np.put_along_axis(inverse, ordering, np.tile(np.arange(1, 51), (len(ordering), 1)), axis=1)
        assert np.array_equal(actual.final_rank.to_numpy(), inverse.reshape(-1))
        assert abs(final_population-stored_inner['map_component']) < 1e-12
        replay['stored_inner_top50_rank_exact'] = True
    positive = top['target'].astype(bool)
    zero_groups = positive.sum(axis=1) == 0
    selected = np.zeros_like(positive)
    np.put_along_axis(selected, ordering[:, :12], True, axis=1)
    old_selected = np.zeros_like(positive)
    old_selected[:, :12] = True
    gained = (positive & selected & ~old_selected).sum(axis=1)
    lost = (positive & ~selected & old_selected).sum(axis=1)
    delta = final_ap-baseline_ap
    standardized = ((top['x']-mean)/std).astype(np.float32)
    feature_shift = [{'feature': feature_names[i],
        'mean_standardized_value': float(standardized[:, :, i].mean()),
        'standardized_std': float(standardized[:, :, i].std()),
        'abs_standardized_over3_share': float((np.abs(standardized[:, :, i]) > 3).mean())}
        for i in range(len(feature_names))]
    user_report = pd.DataFrame({'customer_id': top['customer_id'], 'cutoff': cutoff,
        'full_truth_count': top['truth_count'], 'top50_positives': positive.sum(axis=1),
        'baseline_AP': baseline_ap, 'saved_model_AP': final_ap, 'AP_delta': delta,
        'residual_mean': residual.mean(axis=1), 'residual_std': residual.std(axis=1),
        'gained_correct_top12_pairs': gained, 'lost_correct_top12_pairs': lost,
        **{'loss_'+k: v for k, v in losses.items()}})
    output = c.ART/'multistage_failure_audit'/cutoff/'users.parquet'
    save_frame(user_report, output)
    result = {'cutoff': cutoff, 'role': role, 'full_user_denominator': meta['total_users'],
        'active_users': len(top['x']), 'zero_positive_active_groups': int(zero_groups.sum()),
        'zero_positive_active_group_share': float(zero_groups.mean()),
        'baseline_population_MAP_or_component': baseline_population,
        'saved501_population_MAP_or_component': final_population,
        'delta_complete_population': final_population-baseline_population,
        'baseline_active_mean_AP': float(baseline_ap.mean()), 'saved501_active_mean_AP': float(final_ap.mean()),
        'users_improved': int((delta > 1e-15).sum()), 'users_harmed': int((delta < -1e-15).sum()),
        'users_unchanged': int((np.abs(delta) <= 1e-15).sum()),
        'gained_correct_top12_pairs': int(gained.sum()), 'lost_correct_top12_pairs': int(lost.sum()),
        'mean_top12_candidate_turnover': float((selected & ~old_selected).sum(axis=1).mean()),
        'residual': {'quantiles_0_1_10_50_90_99_100': np.quantile(residual, [0, .01, .1, .5, .9, .99, 1]).tolist(),
            'abs_at_least1_9_share': float((np.abs(residual) >= 1.9).mean()),
            'abs_at_least1_98_share': float((np.abs(residual) >= 1.98).mean()),
            'mean_at_each_stage1_rank': residual.mean(axis=0).tolist(),
            'mean_at_stage1_top12': float(residual[:, :12].mean()),
            'mean_at_stage1_13_50': float(residual[:, 12:].mean()),
            'positive_pair_mean': float(residual[positive].mean()),
            'negative_pair_mean': float(residual[~positive].mean()),
            'zero_positive_group_mean_squared': float(np.square(residual[zero_groups]).mean()),
            'positive_group_mean_squared': float(np.square(residual[~zero_groups]).mean())},
        'objective_at_saved_parameters': {k: float(v.mean()) for k, v in losses.items()},
        'standardized_feature_stats': feature_shift, 'replay_checks': replay,
        'user_audit_path': str(output), 'final_week': 'not_run'}
    print({'cutoff': cutoff, 'role': role, 'delta': result['delta_complete_population'],
           'dynamic_initial': result['objective_at_saved_parameters']['initial_dynamic'],
           'dynamic_final': result['objective_at_saved_parameters']['final_dynamic']}, flush=True)
    x = top['x'] if role == 'in_sample_2019_meta_training_not_generalization' else None
    del top, keys, standardized, user_report
    gc.collect()
    return result, x


def run():
    c.setup()
    c.budget(8)
    started = time.perf_counter()
    directory = c.ART/'WV3-501'
    train = c.read(directory/'TRAINING.json')
    screen = c.read(c.REPORT/'WV3-501_SCREEN.json')
    assert train['params'] == model_code.PARAMS == screen['params'] and not screen['passed']
    registry = next(v for v in c.read(c.REGISTRY)['trials'] if v['experiment_id'] == 'WV3-501')
    assert registry['outer_exposures'] == 0
    model = model_code.Top50Residual()
    model.load_state_dict(torch.load(directory/'model.pt', map_location='cpu', weights_only=True))
    torch.set_num_threads(4)
    with np.load(directory/'scale.npz', allow_pickle=False) as z:
        mean, std = z['mean'].copy(), z['std'].copy()
    assert mean.dtype == std.dtype == np.float32 and mean.shape == std.shape == (21,)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std >= .01).all()
    source = c.read(c.REPORT/'WV3_100_GATE_ORACLE_AUDIT.json')
    training = [m for m in source['historical_meta'] if m['cutoff'] in train['meta_cutoffs']]
    assert [m['cutoff'] for m in training] == train['meta_cutoffs']
    model_code.validate_training_metadata(training, train['prediction_cutoff'])
    windows = []
    xx = []
    for meta in training:
        result, x = audit_window(meta, 'in_sample_2019_meta_training_not_generalization',
                                 model, mean, std, train['features'])
        windows.append(result)
        xx.append(x)
    x = np.concatenate(xx)
    exact_mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    exact_std = np.maximum(x.std(axis=(0, 1), dtype=np.float64).astype(np.float32), .01)
    scale_error = {'mean_max_error': float(np.abs(exact_mean-mean).max()),
                   'std_max_error': float(np.abs(exact_std-std).max()),
                   'rows_reconstructed': int(np.prod(x.shape[:2])),
                   'training_population_only': True}
    assert np.array_equal(exact_mean, mean) and np.array_equal(exact_std, std)
    assert len(x) == train['users']
    del x, xx
    gc.collect()
    for meta in source['inner']:
        result, _ = audit_window(meta, 'original_inner_screen_not_outer', model, mean, std,
                                 train['features'], stored_inner=screen['windows'][meta['window']])
        windows.append(result)
    old, new = windows[:4], windows[4:]
    train_delta = float(np.mean([v['delta_complete_population'] for v in old]))
    inner_delta = float(np.mean([v['delta_complete_population'] for v in new]))
    objective = {k: float(np.average([v['objective_at_saved_parameters'][k] for v in old],
                                      weights=[v['active_users'] for v in old]))
                 for k in old[0]['objective_at_saved_parameters']}
    conclusion = ('training_AP_improved_but_original_inner_failed_transfer_or_overfitting_supported'
        if train_delta > 0 else 'training_AP_already_failed_not_explainable_only_by_cross_year_transfer')
    out = {'created_at': c.now(), 'experiment_id': 'WV3-501', 'scope': 'read_only_checkpoint_diagnostic_no_training',
        'saved_training': train, 'scale_reconstruction': scale_error, 'windows': windows,
        'in_sample2019_mean_delta': train_delta, 'original_inner_mean_delta': inner_delta,
        'in_sample_final_objective': objective,
        'optimization_loss_relative_change': (objective['final_dynamic']/objective['initial_dynamic']-1),
        'mechanism_classification': conclusion,
        'outer_exposures': 0, 'new_model_fits': 0, 'parameter_changes': False,
        'runtime_seconds': time.perf_counter()-started, 'final_week': 'not_run',
        'limits': ['2019replay uses the same supervised model on its training labels; it is in-sample, NOT cross-year robustness evidence.',
                   'Saved model exactly replayed, but no intermediate epoch checkpoints exist, so when AP degradation began cannot be reconstructed.',
                   'Frozen-initial-weight objective is diagnostic only; it was never used for fitting or selecting a model.'],
        'reproduce': 'python -m hm_recsys.warm_v3_multistage_failure_audit'}
    c.write(c.REPORT/'MULTISTAGE_FAILURE_AUDIT.json', out)
    render(out)
    return out


def render(out):
    lines = '\n'.join(f"| {v['cutoff']} | {'训练内重放' if 'in_sample' in v['role'] else '原内层'} | "
        f"{v['baseline_population_MAP_or_component']:.9f} | {v['saved501_population_MAP_or_component']:.9f} | "
        f"{v['delta_complete_population']:+.9f} | {v['users_improved']} | {v['users_harmed']} |"
        for v in out['windows'])
    residual = '\n'.join(f"| {v['cutoff']} | {v['residual']['mean_at_stage1_top12']:+.4f} | "
        f"{v['residual']['mean_at_stage1_13_50']:+.4f} | {v['residual']['abs_at_least1_9_share']:.3%} | "
        f"{v['mean_top12_candidate_turnover']:.3f} | {v['gained_correct_top12_pairs']} | {v['lost_correct_top12_pairs']} |"
        for v in out['windows'])
    objective = out['in_sample_final_objective']
    interpretation = ('同一已保存模型在元训练数据上改善AP，但原内层失败，支持迁移/过拟合问题仍需区分；不能据此宣称模型具备泛化收益。'
        if out['in_sample2019_mean_delta'] > 0 else
        '同一已保存模型在元训练数据上已经降低AP，因此不能把失败只归因于跨年迁移。若优化损失同时下降，应首先记录为当前代理目标与实际排序指标的失配，而不是“完全没有学动”。')
    text = f'''# WV3-501失败机制：已保存模型的只读诊断

状态：501已按原内层门槛关闭，未运行外层。本审计不重训、不修改参数、不搜索残差界限/正则/训练轮数。

## 首先区分训练内效果与迁移

{interpretation}

训练内重放（行业in-sample诊断）使用同一个训练完成的模型，重新预测曾给它提供监督的4个2019窗口。这不是2019跨年检验：模型已经看过这些标签，早期窗口还会被该模型使用的更晚元训练标签影响。它只能回答“训练数据上的AP是否提高”。原内层是原先用于筛选的2020路线窗口，不是新外层。

| 截止 | 用途 | 基线MAP或完整人口分量 | 保存模型MAP或分量 | 增量 | 活跃用户改善数 | 活跃用户受损数 |
|---|---|---:|---:|---:|---:|---:|
{lines}

四个2019训练内窗口等权平均增量 {out['in_sample2019_mean_delta']:+.9f}；四个原内层增量 {out['original_inner_mean_delta']:+.9f}。2019使用完整原用户群；原内层缓存只保留候选覆盖且活跃用户，但其AP总和仍除以完整原验证用户数，表中该项因此是完整MAP的对应分量。两种人群不能直接比较绝对AP水平。

完整真值分母始终是每用户 `min(原完整未来一周真值商品数,12)`，不是前50名正例数。无近期历史用户使用原回退，其贡献保持不变。新排序只在原前50候选里发生，没有新召回。

## 损失是否优化

保存的8轮在线训练损失从 {out['saved_training']['loss_curve'][0]:.9f} 降至 {out['saved_training']['loss_curve'][-1]:.9f}。在线轮均值混合该轮不同时间的模型参数，不等于最后模型的精确完整训练目标。

本审计在完全相同的元训练用户组上重新计算：

- 零残差初始模型的动态ΔAP加权损失：{objective['initial_dynamic']:.9f}。
- 已保存模型的动态ΔAP加权损失（含残差正则）：{objective['final_dynamic']:.9f}；相对变化 {out['optimization_loss_relative_change']:.2%}。
- 使用初始名次固定交换权重、仅代入最终分数的诊断损失：{objective['final_frozen_initial_weights']:.9f}。这不是训练目标，也没有用于模型选择。

动态ΔAP权重（本项目命名）是当前名次中交换正负商品对所造成的AP@12绝对变化，随着排序改变而更新；双方在第12名以后时权重为0。因此训练损失下降并不数学保证AP单调上升。固定初始权重的附加计算用于观察损失变化是否只来自权重改动，但也不是实际AP。

共有 {out['saved_training']['users']:,} 个活跃用户—周训练组，其中 {out['saved_training']['zero_positive_groups']:,} 个前50名没有正例，占 {out['saved_training']['zero_positive_group_share']:.2%}。这些组只贡献残差平方正则，没有伪造正负监督。用户—周是统计单位，不一定为互不重复的人。

## 残差是否饱和、改动了哪些位置

残差为加到 `-log(601名次)` 上的有界修正，范围−2至+2。表中“近界比例”为绝对残差≥1.9的候选占全部活跃前50候选的比例，是描述性阈值，不是调参门槛；“替换数”为每个活跃用户前12名新增候选个数的平均值。新增/丢失正确对的单位为用户—商品对。

| 截止 | 原前12平均残差 | 原13–50平均残差 | 近界比例 | 每用户前12平均替换数 | 新增正确对 | 丢失正确对 |
|---|---:|---:|---:|---:|---:|---:|
{residual}

完整分位数、逐原名次均值、正负候选残差均值、零正例组平方残差及标准化后的逐特征分布均保存在JSON，不将均值差直接解释为因果机制。

## 实现一致性

从4个真实元训练前50集合重建 {out['scale_reconstruction']['rows_reconstructed']:,} 行输入，重新计算保存的21列标准化参数：均值最大误差 {out['scale_reconstruction']['mean_max_error']}，标准差最大误差 {out['scale_reconstruction']['std_max_error']}，两者逐位一致。未用任何内层输入拟合标准化参数。

同一批输入在训练模式与评估模式下输出逐位一致；该网络没有Dropout或BatchNorm。四个原内层重新推理的分数、残差和最终前50名次与保存结果逐位一致。故现有证据不支持“训练/推断标准化不同”或“评估模式改变输出”作为原因。

这次只读诊断耗时 {out['runtime_seconds']:.2f} 秒，没有中间轮次checkpoint，所以不能事后重建哪一轮开始掉AP；不能以最后损失曲线替代这项缺失证据。也不能从501失败推出神经网络排序整体不可行。

结论只关闭本次具体结构/目标/输入组合，不救火调参。基线继续是601。复现命令：`{out['reproduce']}`。输出只进入新的失败审计目录；最终周2020-09-16仍未运行。
'''
    (c.REPORT/'MULTISTAGE_FAILURE_AUDIT.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    run()
