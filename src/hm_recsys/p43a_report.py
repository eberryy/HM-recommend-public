"""Streaming development tournament ledger; partial work is never a winner.

This module only reads completed evaluation receipts and writes aggregate
reports. It does not train, evaluate new policies, touch raw data or hash files.
"""
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from .p41a_contract import read_json, write_json
from .p42f_contract import WINDOWS, now


ARMS = ('A', 'B', 'C', 'D', 'E', 'F')
NAMES = {'winter_20200122': '冬季', 'spring_20200318': '春季',
         'early_summer_20200624': '初夏', 'late_summer_20200819': '夏末'}
WINDOW_KEYS = tuple(WINDOWS)
SEGMENTS = ('warm_21_plus', 'strict_cold', 'sparse1_5', 'all_cold_sparse')
COLUMNS = ['arm', 'model_config_id', 'policy_id', 'status', 'completed_windows'] + [
    name+suffix for name in WINDOW_KEYS for suffix in ('_map', '_delta')] + [
    'mean_map', 'mean_delta', 'std_map', 'std_delta', 'worst_delta', 'nondegrade_windows',
    'warm_mean_delta', 'strict_mean_delta', 'sparse_mean_delta', 'cold_sparse_mean_delta',
    'inserted_positives', 'removed_positives', 'coverage', 'replacements',
    'policy_seconds', 'fit_seconds_shared', 'observed_inserted_positives',
    'observed_removed_positives', 'observed_replacements', 'observed_policy_seconds']
INDEX = {name: i for i, name in enumerate(COLUMNS)}
ARM_FILES = dict(A='lambdarank', B='classifier', C='oracle_imitation',
                 D='stacking', E='residual_challenge', F='blend_search')


def fmt(value):
    if value is None:
        return '未运行/不可用'
    if isinstance(value, (float, np.floating)):
        return f'{value:.9g}'
    return str(value).replace('|', '\\|').replace('\n', '；')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(fmt(x) for x in row)+' |' for row in rows])


def search_plan(contract):
    """One item per independently evaluated model/policy family, no C tripling."""
    policies = contract['policy_grid']
    plan = []
    for model in contract['model_configs']:
        if model['arm'] in ('A', 'B', 'D'):
            plan.append((model['arm'], model['id'], policies))
        elif model['arm'] == 'E':
            plan.append(('E', model['id'], contract['E']['policies']))
    plan.append(('C', 'C-oracle', contract['C']['blends']))
    plan.extend(('F', config['id'], [config]) for config in contract['F']['configs'])
    expected = 56*1920+16+26+128
    if sum(len(policies) for _, _, policies in plan) != expected:
        raise ValueError('Frozen tournament must contain exactly107690 model-policy rows')
    return plan


def simple_policy(policy):
    percentile = policy.get('global_percentile', policy.get('score_percentile_gate', 0.))
    if percentile > 1:
        percentile /= 100.
    return (policy.get('max_admissions', 12),
            policy.get('candidate_topk', policy.get('candidate_topK', 50)),
            -policy.get('slot_floor', policy.get('replaceable_slot_floor', 1)),
            policy.get('top_edge', 600), -percentile)


def ranked_completed(entries):
    """Exact current-max tolerance band; no non-transitive pairwise comparator.

Each entry is (compact row, simplicity tuple). The current maximum remaining
mean defines the band anew at every rank. A heap expands its eligible window
monotonically as the remaining maximum decreases.
"""
    entries = [entry for entry in entries if entry[0][INDEX['status']] == 'completed']
    ordered = sorted(range(len(entries)), key=lambda i: -entries[i][0][INDEX['mean_map']])
    used = set()
    top = 0
    frontier = 0
    heap = []
    answer = []
    while len(answer) < len(entries):
        while top < len(ordered) and ordered[top] in used:
            top += 1
        maximum = entries[ordered[top]][0][INDEX['mean_map']]
        while frontier < len(ordered):
            i = ordered[frontier]
            row, simplicity = entries[i]
            if maximum-row[INDEX['mean_map']] > 1e-8:
                break
            tie = (-int(row[INDEX['nondegrade_windows']] >= 3),
                   -row[INDEX['worst_delta']], -row[INDEX['warm_mean_delta']],
                   simplicity, row[INDEX['model_config_id']], row[INDEX['policy_id']])
            heapq.heappush(heap, (tie, i))
            frontier += 1
        _, selected = heapq.heappop(heap)
        if selected in used:
            raise AssertionError('duplicate rank selection')
        used.add(selected)
        answer.append(entries[selected])
    return answer


def pilot_gate(row):
    return (row[INDEX['status']] == 'completed' and row[INDEX['mean_delta']] > .0001 and
            row[INDEX['nondegrade_windows']] >= 2 and row[INDEX['worst_delta']] >= -.001)


def one_knob_comparison(ranking, policy_lookup):
    """Reuse existing completed rows; fix best model and all other policy knobs."""
    if not ranking:
        return dict(status='not_run', rows=[])
    best=ranking[0][0]; chosen=policy_lookup[(best[1],best[2])]
    knobs=[k for k in ('top_edge','global_percentile','candidate_topk','slot_floor','max_admissions') if k in chosen]
    output=[]
    for knob in knobs:
        for row,_ in ranking:
            if row[1]!=best[1]:continue
            candidate=policy_lookup[(row[1],row[2])]
            if all(candidate.get(k)==v for k,v in chosen.items() if k not in ('id',knob)):
                output.append(dict(knob=knob,value=candidate[knob],policy_id=row[2],
                    mean_delta=row[INDEX['mean_delta']],nondegrade_windows=row[INDEX['nondegrade_windows']],
                    replacements=row[INDEX['replacements']]))
    return dict(status='completed_read_only', model=best[1],reference_policy=chosen,rows=output,
        new_fits=0,new_policy_evaluations=0,
        interpretation='Only one policy knob changes around the development-selected best; same fitted model. Not independent test evidence or proof that search itself generalizes.')


def selection(ranking, tournament_complete):
    family = {}
    for row, _ in ranking:
        arm = row[INDEX['arm']]
        if arm not in family and arm != 'F' and pilot_gate(row):
            family[arm] = row
    top = list(family.values())[:2]
    count = len(top) if tournament_complete else 0
    return dict(status='final_development_selection' if tournament_complete else 'provisional_not_a_winner',
                tournament_complete=bool(tournament_complete),
                any_positive_completed_map=any(row[INDEX['mean_delta']] > 0 for row, _ in ranking),
                F_has_qualifying_policy=any(row[0] == 'F' and pilot_gate(row) for row, _ in ranking),
                qualifying_trainable_families=list(family),
                provisional_top2=[dict(zip(COLUMNS, row)) for row in top],
                top2_full_scale_allowed=bool(count >= 2), top1_full_scale_allowed=bool(count == 1),
                full_scale_allowed=bool(count),
                machine_decision=('tournament_incomplete' if not tournament_complete else
                    'tournament_top2_scaleup' if count >= 2 else
                    'tournament_one_arm_scaleup' if count == 1 else 'tournament_no_positive_map'),
                F_rule='F keeps rank but cannot consume a trainable-family scaleup slot',
                no_scaleup_machine_semantics='no qualifying trainable family; not a claim that every completed MAP delta is nonpositive',
                final_week='not_run', auto_final_week_authorized=False)


def _resolve_receipt(root, value):
    if isinstance(value, dict):
        return value, None
    path = Path(value)
    if not path.is_absolute():
        path = root/path
    return read_json(path), str(path)


def canonical_e_row(row, policy_ids):
    """Report-only alias, validated against frozen E parameters; receipts stay intact."""
    original = row['policy_id']
    if original in policy_ids:
        return row
    policy = row.get('policy', {})
    k, floor = policy.get('candidate_topk'), policy.get('slot_floor')
    alias = f'p43a-E-k{k}-floor{floor}'
    canonical = f'E-K{k}-S{floor}'
    if (original != alias or policy != dict(id=alias, candidate_topk=k, slot_floor=floor)
            or canonical not in policy_ids):
        raise ValueError('E receipt alias does not match registered candidate/slot parameters')
    return dict(row, policy_id=canonical, execution_policy_id=original)


def _model_evidence(root, state, model_id, arm, policy_ids):
    candidates = [model_id]
    if arm == 'C':
        candidates += ['C-count']
    configs = state.get('configs', {})
    record = next((configs[name] for name in candidates if name in configs), {})
    windows = {}
    receipt_paths = {}
    fit_seconds = 0.
    prediction_seconds = 0.
    attempted = record.get('status', 'not_run') != 'not_run'
    for w in WINDOW_KEYS:
        saved = record.get('windows', {}).get(w)
        receipt = None
        if saved:
            attempted = True
            if saved.get('receipt'):
                receipt, path = _resolve_receipt(root, saved['receipt'])
                receipt_paths[w] = path
        if receipt is None:
            fallback = root/'evaluation'/model_id/w/'policy/EVALUATION.json'
            if fallback.exists():
                receipt, path = _resolve_receipt(root, str(fallback))
                receipt_paths[w] = path
                attempted = True
        if receipt is not None and receipt.get('status') == 'completed':
            observed = ([canonical_e_row(r, policy_ids) for r in receipt['rows']]
                        if arm == 'E' else receipt['rows'])
            rows = {r['policy_id']: r for r in observed}
            if len(rows) != len(receipt['rows']):
                raise ValueError('duplicate completed policy rows')
            unknown = set(rows)-policy_ids
            if unknown:
                raise ValueError(f'unregistered policy ids for {model_id}: {sorted(unknown)[:3]}')
            windows[w] = rows
        fit_ids = ['C-count', 'C-cold', 'C-warm'] if arm == 'C' else [model_id]
        for fitted_id in fit_ids:
            ff = root/'models'/fitted_id/w
            if (ff/'FIT_START.json').exists():
                attempted = True
            if (ff/'FIT_RESULT.json').exists():
                fitted = read_json(ff/'FIT_RESULT.json')
                if fitted.get('status') == 'completed':
                    # A completed fit still costs time if prediction/evaluation
                    # was paused before its window receipt was finalized.
                    fit_seconds += float(fitted.get('seconds', 0.) or 0.)
        predicted = root/'evaluation'/model_id/w/'PREDICTION.json'
        if predicted.exists():
            pr = read_json(predicted)
            if pr.get('status') == 'completed':
                prediction_seconds += float(pr.get('seconds', 0.) or 0.)
    return record, windows, dict(receipts=receipt_paths, fit_seconds=fit_seconds,
        prediction_seconds=prediction_seconds, attempted=attempted)


def compact_row(arm, model_id, policy_id, windows, record, evidence):
    observations = {w: rows[policy_id] for w, rows in windows.items() if policy_id in rows and
                    rows[policy_id].get('status', 'completed') == 'completed'}
    saved_status = record.get('policies', {}).get(policy_id, {}).get('status')
    if len(observations) == 4:
        status = 'completed'
    elif saved_status == 'pruned_bad_config':
        first_two = [observations.get(w) for w in WINDOW_KEYS[:2]]
        if any(x is None for x in first_two):
            raise ValueError('pruned policy lacks the registered first-two-window evidence')
        ds = [x['delta_map'] for x in first_two]
        if not (all(d < 0 for d in ds) and np.mean(ds) < -.0015):
            raise ValueError('pruned policy violates the frozen early-pruning gate')
        status = 'pruned_bad_config'
    elif saved_status in ('failed', 'engineering_failure'):
        status = 'failed'
    elif observations or evidence['attempted']:
        status = 'partial'
    else:
        status = 'not_run'
    row = [None]*len(COLUMNS)
    def put(key, value):
        row[INDEX[key]] = value
    for key, value in [('arm', arm), ('model_config_id', model_id), ('policy_id', policy_id),
                       ('status', status), ('completed_windows', len(observations)),
                       ('fit_seconds_shared', evidence['fit_seconds'] if evidence['attempted'] else None)]:
        put(key, value)
    for w, x in observations.items():
        put(w+'_map', x['overall_map'])
        put(w+'_delta', x['delta_map'])
    for output, source in [('observed_inserted_positives','inserted_positives'),
                            ('observed_removed_positives','removed_positives'),
                            ('observed_replacements','replacements'),
                            ('observed_policy_seconds','seconds')]:
        put(output, sum(x.get(source, 0.) or 0. for x in observations.values()) if observations else None)
    if status != 'completed':
        return row
    maps = np.array([observations[w]['overall_map'] for w in WINDOW_KEYS])
    deltas = np.array([observations[w]['delta_map'] for w in WINDOW_KEYS])
    assert np.isfinite(maps).all() and np.isfinite(deltas).all()
    for key, value in [('mean_map', float(maps.mean())), ('mean_delta', float(deltas.mean())),
                       ('std_map', float(maps.std())), ('std_delta', float(deltas.std())),
                       ('worst_delta', float(deltas.min())), ('nondegrade_windows', int((deltas >= 0).sum()))]:
        put(key, value)
    for segment, key in zip(SEGMENTS, ('warm_mean_delta','strict_mean_delta','sparse_mean_delta','cold_sparse_mean_delta')):
        values = [observations[w]['segments'][segment].get('delta_vs_w0', observations[w]['segments'][segment].get('delta')) for w in WINDOW_KEYS]
        put(key, float(np.mean(values)) if all(v is not None for v in values) else None)
    for key, source in [('inserted_positives','inserted_positives'), ('removed_positives','removed_positives'),
                         ('replacements','replacements'), ('policy_seconds','seconds')]:
        put(key, sum(x.get(source, 0.) or 0. for x in observations.values()))
    put('coverage', float(np.mean([observations[w]['coverage'] for w in WINDOW_KEYS])))
    return row


def _write_placeholder(path, content):
    if not path.exists():
        write_json(path, content)
    return read_json(path)


def oracle_summary(oracle):
    rows = []
    for window, cutoff in WINDOWS.items():
        value = oracle.get('dates', {}).get(cutoff, {})
        done = value.get('status') == 'completed'
        rows.append(dict(window=window, cutoff=cutoff, status=value.get('status', 'not_run'),
            users=value.get('users') if done else None,
            w0_map=value.get('w0_map') if done else None,
            oracle_map=value.get('overall_map') if done else None,
            oracle_delta=value.get('delta_map') if done else None,
            K_distribution=value.get('k_star_distribution') if done else None))
    complete = all(row['status'] == 'completed' for row in rows)
    return dict(rows=rows, complete=complete,
        mean_oracle_map=float(np.mean([r['oracle_map'] for r in rows])) if complete else None,
        mean_oracle_delta=float(np.mean([r['oracle_delta'] for r in rows])) if complete else None)


def sampling_comparison(completed):
    """Matched existing policies only; no new scores, tuning, or fits."""
    lookup = {(row[1], row[2]): (row, simple) for row, simple in completed
              if row[INDEX['status']] == 'completed'}
    groups = defaultdict(list)
    for (model, policy), (row, simple) in lookup.items():
        if not model.startswith(('A1-', 'B1-')):
            continue
        other = model[0]+'2'+model[2:]
        if (other, policy) in lookup:
            groups[(model, other)].append(((row, simple), lookup[(other, policy)]))
    results = []
    for (reference, hard), pairs in sorted(groups.items()):
        changes = np.array([b[0][INDEX['mean_delta']]-a[0][INDEX['mean_delta']] for a,b in pairs])
        winner = ranked_completed([b for _,b in pairs])[0][0]
        old = lookup[(reference, winner[2])][0]
        results.append(dict(reference_model=reference, hard_model=hard, paired_complete_policies=len(pairs),
            hard_higher=int((changes>0).sum()), equal=int((changes==0).sum()), hard_lower=int((changes<0).sum()),
            hard_best_policy=winner[2], hard_best_mean_delta=winner[INDEX['mean_delta']],
            reference_same_policy_mean_delta=old[INDEX['mean_delta']]))
    return dict(stage='P4.3A', status='completed_pairs_only' if results else 'not_run_no_complete_pairs',
        rows=results, final_week='not_run', new_model_fits=0, new_policy_evaluations=0,
        interpretation='Same model parameters and same policy, both with complete four-window receipts only. Policies are correlated search configurations, not independent statistical samples. Incomplete families remain undecided.')


def write_report(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    dest = repo/'reports/phase4'
    c = read_json(dest/'P4_3A_EXPERIMENT_CONTRACT.json')
    state_path = root/'TOURNAMENT_STATE.json'
    state = read_json(state_path) if state_path.exists() else dict(status='not_run', configs={}, final_week='not_run')
    from .p43a_session import execution_budget
    budget = execution_budget(c,state)
    assert c['final_week'] == state.get('final_week', 'not_run') == 'not_run'
    plan = search_plan(c)
    completed = []
    counts = Counter()
    arm_counts = defaultdict(Counter)
    policy_lookup = {}
    sources = {}
    fit_seconds_total = 0.
    prediction_seconds_total = 0.
    policy_seconds_total = 0.
    total = 0
    json_path = dest/'p4_3a_tournament_ledger.json'
    csv_path = dest/'p4_3a_tournament_ledger.csv'
    json_part, csv_part = json_path.with_suffix('.json.part'), csv_path.with_suffix('.csv.part')
    with json_part.open('w', encoding='utf8') as js, csv_part.open('w', encoding='utf8', newline='') as cs:
        js.write('{"stage":"P4.3A","schema":"columns_plus_rows_v1","columns":')
        json.dump(COLUMNS, js, ensure_ascii=False, separators=(',', ':'))
        js.write(',"rows":[')
        writer = csv.writer(cs)
        writer.writerow(COLUMNS)
        first = True
        for arm, model_id, policy_list in plan:
            ids = {p['id'] for p in policy_list}
            record, windows, evidence = _model_evidence(root, state, model_id, arm, ids)
            sources[model_id] = evidence
            fit_seconds_total += evidence['fit_seconds']
            prediction_seconds_total += evidence['prediction_seconds']
            for policy in policy_list:
                row = compact_row(arm, model_id, policy['id'], windows, record, evidence)
                if not first:
                    js.write(',')
                json.dump(row, js, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
                first = False
                writer.writerow(row)
                total += 1
                status = row[INDEX['status']]
                counts[status] += 1
                arm_counts[arm][status] += 1
                policy_seconds_total += row[INDEX['observed_policy_seconds']] or 0.
                if status == 'completed':
                    completed.append((row, simple_policy(policy)))
                    policy_lookup[(model_id, policy['id'])] = policy
        js.write('],"status_counts":')
        json.dump(dict(counts), js, ensure_ascii=False, separators=(',', ':'))
        js.write(',"final_week":"not_run"}\n')
    if total != 107690:
        raise AssertionError('full fixed search ledger not preserved')
    json_part.replace(json_path)
    csv_part.replace(csv_path)
    ranking = ranked_completed(completed)
    paired = sampling_comparison(completed)
    write_json(dest/'p4_3a_sampling_comparison.json', paired)
    terminal = {'completed', 'pruned_bad_config'}
    tournament_complete = all(status in terminal for status in counts) and all(
        sum(arm_counts[a].values()) > 0 and not any(status not in terminal for status in arm_counts[a]) for a in ARMS)
    chosen = selection(ranking, tournament_complete)
    best = ranking[0][0] if ranking else None
    top20 = [dict(rank=i+1, **dict(zip(COLUMNS, row)), policy=policy_lookup[(row[1], row[2])])
             for i, (row, _) in enumerate(ranking[:20])]
    sensitivity=one_knob_comparison(ranking,policy_lookup)
    write_json(dest/'p4_3a_policy_sensitivity.json',sensitivity)
    elapsed = max(0., time.time()-budget['start_epoch'])
    summary = dict(stage='P4.3A', status='tournament_completed' if tournament_complete else 'partial',
        run_state=state.get('status'), snapshot_at=now(), counts=dict(counts), expected_policy_rows=107690,
        arm_counts={a:dict(arm_counts[a]) for a in ARMS}, completed_four_window_rows=len(completed),
        development_only=True, untouched_confirmatory=False, selection=chosen,
        execution_session=state.get('active_session'),candidate_numeric_gate=state.get('candidate_numeric_gate'),
        timing=dict(session_elapsed_seconds=elapsed, approved_session_seconds=budget['total_seconds'],
            deadline_epoch=budget['deadline_epoch'], deadline_utc=budget.get('deadline_utc'),
            original_budget=c['budget'],
            paused_at=state.get('paused_at'), reason=state.get('reason',state.get('error')),
            recorded_fit_seconds=fit_seconds_total, allocated_policy_seconds=policy_seconds_total,
            recorded_prediction_seconds=prediction_seconds_total,
            component_timings_are_cumulative_across_sessions=True,
            note='fit seconds counted once per model-window; per-policy grid shares are allocations, not independent runtime; excludes work without completed receipts'),
        oracle=None, final_week='not_run', Warm_v2_integrated=False, full_history_run=state.get('full_history_run',False),
        selected_baseline='W0', automatic_next_stage=False)
    oracle_path = dest/'p4_3a_oracle_headroom.json'
    oracle = read_json(oracle_path) if oracle_path.exists() else dict(status='not_run')
    oracle_view = oracle_summary(oracle)
    summary['oracle'] = oracle
    summary['oracle_summary'] = oracle_view
    write_json(dest/'P4_3A_metrics.json', summary)
    write_json(dest/'p4_3a_top20.json', dict(stage='P4.3A', status='final_development_ranking' if tournament_complete else 'provisional',
               rows=top20, tournament_complete=tournament_complete, final_week='not_run'))
    write_json(dest/'p4_3a_top2_selection.json', chosen)
    arm_summaries = {}
    for arm in ARMS:
        winner = next((dict(zip(COLUMNS, row)) for row, _ in ranking if row[0] == arm), None)
        arm_status = ('not_run' if set(arm_counts[arm]) == {'not_run'} else
                      'completed' if all(s in terminal for s in arm_counts[arm]) else 'partial')
        arm_summaries[arm] = dict(stage='P4.3A', arm=arm,
            status=arm_status,
            counts=dict(arm_counts[arm]), provisional_best=winner, final_week='not_run')
        write_json(dest/f'p4_3a_arm_{arm}_{ARM_FILES[arm]}.json', arm_summaries[arm])
    full = _write_placeholder(dest/'p4_3a_fullscale.json', dict(stage='P4.3A', status='not_run',
        reason='no complete eligible tournament authorization has been executed', final_week='not_run'))
    _write_placeholder(dest/'p4_3a_fullscale_comparison.json', dict(stage='P4.3A', status='not_run',
        reason='no completed full-history model comparison', final_week='not_run'))
    verified = _write_placeholder(dest/'P4_3A_VERIFICATION.json', dict(stage='P4.3A', status='not_run',
        reason='report construction is not independent verification', final_week='not_run'))
    write_json(root/'REPORT_SNAPSHOT.json', dict(at=now(), status=summary['status'],
        sources=sources, ledger_rows=total, ledger_schema=COLUMNS, final_week='not_run'))
    old_h_path = dest/'P4_2H_metrics.json'
    old_h = read_json(old_h_path) if old_h_path.exists() else None
    historical_rows = []
    if old_h is not None:
        for label, key in [('W0冻结参照','W0'),('P4.2H条件期望值参照','H_conditional_BH')]:
            values = [old_h['windows'][w][key]['map12'] for w in WINDOW_KEYS]
            delta = 0. if key == 'W0' else old_h['verdicts']['mean_delta']
            historical_rows.append([label,float(np.mean(values)),delta,'历史已测，不重新拟合或选择'])
    text = ['# P4.3A：冷侧信号MAP兑现竞赛',
        '状态：'+('四窗竞赛已完成；独立核验的具体范围和限制见第8节及补充核验记录。' if tournament_complete else
                 '阶段未完成。以下是当前证据快照，未运行或中途暂停不等于方案失败。'),
        '## 1. 当前能回答什么',
        ('当前已有完成四个开发窗口的配置：最优暂列'+str(best[1])+'、'+str(best[2])+
         '，平均MAP差'+fmt(best[INDEX['mean_delta']])+'。'+
         (('其开发MAP为正，扩量门槛与不同家族选择见第5节；仍不是最终周确认。' if tournament_complete else
           '其开发MAP为正，但在完整竞赛完成前不能宣布最终获胜或扩大训练。') if best[INDEX['mean_delta']] > 0 else
          '目前最好的完整四窗配置还没有正MAP；不等于未运行家族已失败。')
         if best is not None else '目前尚无完成四个开发窗口的模型—策略配置，不能报告四窗均值、赢家或扩量结论。'),
        '本阶段将2020-01-22、2020-03-18、2020-06-24、2020-08-19重新分类为development windows（行业开发验证窗口）：'
        '允许据其结果选择模型、门槛和混合权重，因此不再属于未触碰确认测试。历史P4.0–P4.2H报告不改写。'
        '最终周2020-09-16始终not_run（未运行）；Warm-v2未整合。',
        table(['固定历史参照','四窗平均MAP','相对W0平均差','边界'],historical_rows),
        '## 2. 术语、统计单位与时间',
        'Arm（实验通用术语）指六个方法家族：A操作边LambdaRank、B有益对其余分类、C真值知情参考策略模仿、'
        'D历史分数堆叠、E保留原暖排序的冷侧挑战、F固定随机混合。model_config_id是冻结训练配置ID；'
        'policy_id是将分数变成替换列表的规则ID。完整参数引用本轮冻结合同，不在每行重复复制。',
        'A1/B1为本项目采样版本名：历史训练保留全部有益与有害操作边，加确定性的2%中性边；'
        'A2/B2保留全部有益、有害、规则定义的困难中性边，再加剩余中性边的确定性0.5%。'
        '中性指单次替换AP变化为0，不表示商品已曝光但被用户拒绝。A/D不沿用旧中性样本50倍权重；B只按合同提高有益边权重。',
        '策略参数均为本项目配置名：top_edge是每用户完整操作集合中保留的最高分边数；global_percentile是同窗全部操作分数的全局百分位门槛；'
        'candidate_topk/candidate_topK是可用冷商品的原B0名次上限（E按其挑战分数取候选上限）；'
        'slot_floor/replaceable_slot_floor是允许替换的原暖位置下界，12只允许最后一位，1允许全部位置；'
        'max_admissions是每用户最多替换件数。实际件数还受候选、阈值和一对一匹配约束，不保证等于上限。',
        '操作边为“历史日期—用户—冷商品—原暖位置”，每用户最多600条。W0是冻结的原Top12及位置；'
        'Cold50为冻结冷侧候选，先排除与W0重合。MAP@12为所有固定用户的AP均值，AP分母min(该用户不同真值商品数,12)。'
        '无冷候选用户也保留并计入；跨窗人数是用户—窗口观察，不是去重自然人数。',
        'mean/std/worst分别是完整四窗等权均值、总体标准差(ddof=0)、最差窗；std_map与std_delta分开保存。'
        '部分配置的四窗均值保持null，CSV空白，不用0冒充未测。observed_*只累计已完成窗口，不能与四窗总量直接比较。'
        'warm_21_plus、strict_cold、sparse1_5、all_cold_sparse按截止日前商品全站交易事件数≥21、0、1–5、0–5分组；'
        '分组MAP以该组真值重算，仅对有该组真值的用户平均。',
        'coverage为四窗准入用户占该窗全部用户比例的等权平均；inserted/removed是插入冷侧真值或移除原推荐真值的用户—商品—窗口对数；'
        'replacements是替换操作次数。剪枝只按最先两窗均负且平均差<−0.0015，保留记录，不进入四窗排名。',
        f'当前会话记录的预算为{budget["total_seconds"]}秒（含准备实现），截止{budget.get("deadline_utc",budget["deadline_epoch"])}。'
        f'到本次快照已过{elapsed/60:.2f}分钟，这是时间来源记录，不是预计完成工期。'
        f'跨会话累计的训练完成收据计时{fit_seconds_total:.2f}秒、预测完成收据计时{prediction_seconds_total:.2f}秒、策略共享计算分摊计时{policy_seconds_total:.2f}秒。'
        '后者不是每个策略独立运行耗时；这些计时不包含无完成收据的准备、失败或暂停工作，不能相加冒充端到端成本。',
        '运行状态：'+fmt(state.get('status'))+'；暂停时间：'+fmt(state.get('paused_at'))+'。',
        '如有新的会话授权，原两小时合同与第一轮结果保持不变，以追加的会话授权记录新截止时间。'
        '本次若因候选数值门槛暂停，只表示至少一套完整四窗可训练方案满足平均差>0.0001、至少2窗不退化、最差差≥−0.001；'
        '不等于未完成竞赛已产生最终Top2，也不自动执行全量训练。',
        '## 3. 完整搜索台账',
        table(['家族','completed四窗完成','pruned已剪枝','partial部分进度','not_run未运行','failed工程失败'],
            [[a,*[arm_counts[a][s] for s in ('completed','pruned_bad_config','partial','not_run','failed')]] for a in ARMS]),
        f'冻结107690个模型—策略配置全部保留，已完成四窗{len(completed)}条。'
        '[完整JSON台账](p4_3a_tournament_ledger.json)使用columns+rows紧凑结构；'
        '[CSV台账](p4_3a_tournament_ledger.csv)逐行含模型、策略、四窗指标和状态。未删除失败或相同结果配置。',
        '## 4. 当前Top20（仅四窗完整配置）',
        '排序先按当前最大平均MAP的1e-8候选带，再优先至少3窗不退化、更好的最差窗和暖组、更简单冻结策略，最后稳定ID。'
        '四窗不完整或已剪枝者不参赛；竞赛未完成时这里只是provisional（临时名次）。',
        table(['名次','家族','模型','策略',*[''+NAMES[w]+' MAP；差值' for w in WINDOW_KEYS],
               'meanMAP','mean差','std差','最差','暖均差','冷稀疏均差','插入','移除','覆盖率','替换','策略秒','状态'],
            [[x['rank'],x['arm'],x['model_config_id'],x['policy_id'],
              *[fmt(x[w+'_map'])+' ; '+fmt(x[w+'_delta']) for w in WINDOW_KEYS],
              *[x[k] for k in ('mean_map','mean_delta','std_delta','worst_delta','warm_mean_delta','cold_sparse_mean_delta',
                  'inserted_positives','removed_positives','coverage','replacements','policy_seconds','status')]] for x in top20]),
        '## 5. 候选上限与扩量状态',
        'oracle（行业真值知情参考上限）只在冻结候选空间比较K=0..12个单边总权最优匹配方案后重算列表AP；'
        '并非所有组合的全局AP最优。该参考不作为特征；当前记录状态：'+fmt(oracle.get('status'))+'。'
        '完整结果见[p4_3a_oracle_headroom.json](p4_3a_oracle_headroom.json)。',
        table(['窗口','状态','固定用户数','W0 MAP','参考方案MAP','相对W0上限差','K=0用户','K=1用户','K≥2用户'],
            [[NAMES[r['window']],r['status'],r['users'],r['w0_map'],r['oracle_map'],r['oracle_delta'],
              r['K_distribution'].get('0',0) if r['K_distribution'] is not None else None,
              r['K_distribution'].get('1',0) if r['K_distribution'] is not None else None,
              sum(v for k,v in r['K_distribution'].items() if int(k)>=2) if r['K_distribution'] is not None else None]
             for r in oracle_view['rows']]),
        '完整四窗参考方案平均MAP='+fmt(oracle_view['mean_oracle_map'])+'，相对W0平均差='
        +fmt(oracle_view['mean_oracle_delta'])+'。只有全部四窗完成才给均值；该真值知情差是候选机会，不保证可学习兑现。',
        '扩量选择状态：'+chosen['status']+'；full_scale_allowed='+str(chosen['full_scale_allowed'])+'。'
        '只有完整六族竞赛完成后，才按平均差>0.0001、至少2窗不退化、最差≥−0.001筛选不同可训练家族。'
        'F保留名次但不占100%训练名额。当前100%执行状态：'+fmt(full.get('status'))+'。',
        '## 6. 17个机制问题：已知与未测']
    best_policy = policy_lookup.get((best[1],best[2]),{}) if best else {}
    answers = [
        (1,'哪个家族最强？', '当前完整配置暂列'+str(best[0]) if best else '无完整四窗配置，未定。'),
        (2,'排序模型是否优于概率/期望值？','只能将已完成四窗结果与历史F/G/H参照比较；完整竞赛未完成时不作总架构胜负结论。'),
        (3,'困难负样本是否有效？','已有'+str(len(paired['rows']))+'组相同模型参数的完整采样对照，见第9节；仅适用于表中参数和双方未剪枝的完整策略，不能把相关配置数当独立样本量。'),
        (4,'类别加权是否有效？','已完成B最佳四窗平均差为'+fmt((arm_summaries['B']['provisional_best'] or {}).get('mean_delta'))+'。这回答已测加权配置的最佳表现；没有权重1对照，不能将结果单独归因于加权。'),
        (5,'阈值搜索是否为主增益？','第11节固定最终最佳模型和其他规则，仅改变一个已测策略参数。可以定位准入强度的影响，但没有预注册的无搜索对照，不能宣称搜索本身是主因或可泛化收益。'),
        (6,'最优max_admissions？',fmt(best_policy.get('max_admissions'))+'；若只允许用户最高1条操作边，该上限本来不生效，不能解释为模型证明“最多替换1件”最优。第11节给出其余参数不变的对照。'),
        (7,'暖位置限制是否必要？',fmt(best_policy.get('slot_floor',best_policy.get('replaceable_slot_floor')))+'；是否必要需同模型政策对照。'),
        (8,'candidate_topK最优？',fmt(best_policy.get('candidate_topk',best_policy.get('candidate_topK')))+'；仅当前暂列。'),
        (9,'堆叠是否更稳定？','D已完成配置的暂列最佳平均差为'+fmt((arm_summaries['D']['provisional_best'] or {}).get('mean_delta'))+'，不退化窗口数为'+fmt((arm_summaries['D']['provisional_best'] or {}).get('nondegrade_windows'))+'；部分进度不参与均值或胜负，见第10节。'),
        (10,'模仿oracle的K有效？','C已完成最佳策略的替换次数为'+fmt((arm_summaries['C']['provisional_best'] or {}).get('replacements'))+'；逐窗替换数量模型输出与训练类别分布见本轮诊断。参考策略有收益不证明学习已成功。'),
        (11,'残差挑战优于完整边排序？','E已完成配置的最佳平均差为'+fmt((arm_summaries['E']['provisional_best'] or {}).get('mean_delta'))+'；其冷侧命中和原推荐损失必须一起看，见第10节。'),
        (12,'10%真的正MAP？',('当前完整策略最佳mean差='+fmt(best[INDEX['mean_delta']])+'，只属于开发证据。') if best else '尚无完整四窗均值。'),
        (13,'Top2值得扩100%？',str(chosen['full_scale_allowed'])+'；未完成竞赛不提前授予。'),
        (14,'100%是否继续提升？','状态'+fmt(full.get('status'))+'；没有完成对照时不能判断。'),
        (15,'扩量提升来自方差降低还是策略？','只有固定策略、仅改训练覆盖率的对照才可归于扩量相关变化；单次提升仍不能证明方差降低机制。'),
        (16,'还剩多少候选上限？','参考上限状态'+fmt(oracle.get('status'))+'，四窗平均相对W0差='
         +fmt(oracle_view['mean_oracle_delta'])+'；第5节逐窗。它不是全部组合的全局AP上限。'),
        (17,'最终周仍未运行？','2020-09-16=not_run；Warm-v2未整合，不启动P4.3B。')]
    text += [table(['序号','问题','当前证据'],answers),
             '## 7. 明确结论',
             ('目前至少有一个完整四窗方案取得正开发MAP：'+str(best[1])+' / '+str(best[2])+
              '，平均差'+fmt(best[INDEX['mean_delta']])+'。'+
              ('六族开发竞赛已完成，是否扩量以冻结门槛为准；没有最终周的独立确认。' if tournament_complete else
               '未完成的家族仍待比较，不能提前宣布最终赢家或独立确认成功。')
              if best and best[INDEX['mean_delta']] > 0 else
              '目前还没有证据支持“已有一个完成四窗的方案把冷信号兑现为正MAP”。未运行与部分进度不是失败，待完成的搜索不能被本次快照替代。')]
    text += ['## 8. 实现边界与独立核验范围',
        '训练与开发评分按已冻结的搜索范围执行。计算复用只合并相同允许操作集合、相同匹配及相同最终列表的重复计算；'
        '每个模型—策略记录仍保留，不能把台账行数当独立样本量。后续家族实现与正在运行的进程分开；'
        '实际模型绑定进程启动时的源码收据，旧版保存在忽略的大产物目录，不用当前磁盘版本冒充旧模型源码。',
        'F在正式执行前修正百分位接口单位：冻结配置0.90–0.999，仅传入共享函数时乘100；'
        '这不是门槛调整，也不改128套随机权重。C用固定4类的原生多分类接口，即便历史只出现部分替换数量类别，也保留0、1、2、3+的同一输出列含义。',
        '独立核验状态：'+fmt(verified.get('status'))+'；整个阶段已核验：'+str(verified.get('overall_stage_verified',False))+'。'
        '独立核验不是训练脚本自己重复输出同一指标。已核验模型—窗口：'+
        '、'.join(str(x.get('model_id'))+'@'+str(x.get('window')) for x in verified.get('scope_model_windows',[]))+ '。'
        '其余配置即使有训练评测收据，也不能借用这个局部核验状态宣称整个竞赛已通过。详情见[P4_3A_VERIFICATION.json](P4_3A_VERIFICATION.json)。']
    text += ['## 9. 已完成的采样对照',
        '只配对模型其他参数与决策策略完全相同、且双方均完成四窗的A1/A2或B1/B2。'
        '较高/相同/较低统计的是困难中性版本相对2%中性版本的四窗平均MAP差方向，单位是策略配置；'
        '这些规则高度相关，不能当作独立样本量或显著性检验。最佳策略按本阶段同一排序规则选择，仍属于开发选择。'
        '局部参数的负结果不能外推为整个未完成家族失败。',
        table(['2%中性版本','困难中性版本','配对策略数','困难版本较高','相同','较低','困难版本最佳策略','其平均差','同策略2%版本平均差'],
            [[r[k] for k in ('reference_model','hard_model','paired_complete_policies','hard_higher','equal','hard_lower',
                'hard_best_policy','hard_best_mean_delta','reference_same_policy_mean_delta')] for r in paired['rows']]),
        '机器结果见[p4_3a_sampling_comparison.json](p4_3a_sampling_comparison.json)。此表只汇总已有收据，没有新增训练或策略评测。']
    text += ['## 10. 各家族的已测最佳与未测边界',
        '每行只展示本家族已经完成四窗的最佳模型—策略，不代表全竞赛赢家。C/F没有结果时留空。'
        '插入、移除为四窗累计真值用户—商品—窗口对；总体MAP还受命中位置和每用户真值分母影响，不能只用两个计数相减推导MAP。',
        table(['家族','状态','最佳模型','平均MAP差','不退化窗数','严格零交互MAP差','暖组MAP差','插入真值对','移除真值对','替换次数'],
            [[a,arm_summaries[a]['status'],*[(arm_summaries[a]['provisional_best'] or {}).get(k)
                for k in ('model_config_id','mean_delta','nondegrade_windows','strict_mean_delta',
                          'warm_mean_delta','inserted_positives','removed_positives','replacements')]] for a in ARMS]),
        ('当前数值门槛达标的可训练家族：'+', '.join(chosen['qualifying_trainable_families'])
         if chosen['qualifying_trainable_families'] else
         '当前没有完整四窗可训练策略达到扩量数值门槛。不仅是竞赛未完成：已测最佳平均差也必须超过0.0001，不能以微小正增益替代该条件。'),
        'E若严格零交互组有正收益而总体为负，说明有冷商品进入Top12并命中，但代价超过总体收益；'
        '不能称为冷启动已经解决，也不能笼统称全部冷信号均无排序价值。'
        '目前的指标不能单独确定剩余瓶颈究竟是正例稀疏、匹配特征还是分数校准。',
        '本轮资源隔离、旧输入格式兼容及E编号映射的原因、验证范围和失败记录见'
        '[工程说明](P4_3A_ENGINEERING_NOTES.md)。已修复工程失败保留历史收据，不将其误记为模型负结果。']
    text += ['## 11. 最佳模型的单参数策略对照（只读）',
        '固定最佳模型及其他所有策略参数，仅列出已完成台账中一个参数变化的结果。'
        '没有新增试验；这里只解释开发集内局部变化，不构成新的未触碰测试。参数定义见第2节。',
        table(['变化参数','取值','已有策略编号','平均MAP差','不退化窗数','替换次数'],
            [[r[k] for k in ('knob','value','policy_id','mean_delta','nondegrade_windows','replacements')]
             for r in sensitivity['rows']]),
        '机器结果见[p4_3a_policy_sensitivity.json](p4_3a_policy_sensitivity.json)。']
    remaining=dest/'P4_3A_REMAINING_VERIFICATION.json'
    if remaining.exists():
        rv=read_json(remaining)
        text += ['## 12. 后续家族补充核验',
            '状态：'+rv['status']+'。对C、E、F所有已完成策略，按保存的用户—冷商品—原位置决策重建最终列表，'
            '使用项目旧AP函数重算总体及商品分组指标、插入/移除真值、覆盖率和替换次数；'
            '核对用户归属、位置限制及最终12件唯一性。另核对所有完整模型的合同参数与训练标签结束时间。'
            '不将这项核验称作独立重训，也不宣称逐个证明全部匹配最优性；A/B/D另有明确范围的快照核验。',
            '[补充核验记录](P4_3A_REMAINING_VERIFICATION.json)；'
            '[第三会话诊断](P4_3A_SESSION_03_DIAGNOSTICS.json)含C替换数量分布与D暂停模型前缀一致性。']
    diagnostic=dest/'P4_3A_SESSION_03_DIAGNOSTICS.json'
    if diagnostic.exists():
        cr=read_json(diagnostic)['C_count']['windows']
        text += ['## 13. C的替换数量入口诊断（只读）',
            'K为本项目每用户要执行的替换件数；模型固定四类0、1、2、3及以上，选分数最大的类别，'
            '并列优先较小类别。训练用户—窗口只计有可替换冷候选的人；预测保留全体固定评测用户。'
            '非零类别总分指模型对类别1–3的输出之和，不是校准后的真实购买概率。',
            table(['窗口','历史训练用户—窗口数','历史应替换≥1件数','其占训练分母比例','评测用户数','预测替换≥1件数','非零类别总分最大值'],
                [[NAMES[r['window']],r['training_rows'],sum(v for k,v in r['training_class_counts'].items() if k!='0'),
                  sum(v for k,v in r['training_class_counts'].items() if k!='0')/r['training_rows'],r['users'],
                  sum(v for k,v in r['predicted_K_counts'].items() if k!='0'),r['nonzero_class_score_quantiles']['max']] for r in cr]),
            '四窗都预测0件，故后续冷候选排序、暖位置排序和26个混合规则没有产生实际替换。'
            '这与历史非零类别很少、当前固定最大类别决策保守相容，但不能据此证明后两模型没有排序信号。'
            '本轮未改类别权重、决策门槛或训练规模；这些变化不属于此次补完任务。']
    (dest/'P4_3A_FINAL.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')
    return summary
