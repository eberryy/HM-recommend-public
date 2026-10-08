"""Label-aware diagnostic only; never used for fitting or policy selection."""
from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd

from .metrics import apk
from .p42d_stats import single_ap_delta
from .p42f_data import frame, save_frame
from .p43a_oracle import oracle_user
from .p43a_policy import contexts, exact_ap, SEGMENTS

WINDOWS = {'winter_20200122': '2020-01-22', 'spring_20200318': '2020-03-18',
           'early_summer_20200624': '2020-06-24', 'late_summer_20200819': '2020-08-19'}


def best_single(warm, cold, ranks, truth, weights, limit=50):
    eligible = np.flatnonzero(np.asarray(ranks) <= limit)
    if not len(eligible):
        return list(warm)
    local, slot = np.unravel_index(np.argmax(weights[eligible]), (len(eligible), 12))
    ci = eligible[local]
    result = list(warm)
    if weights[ci, slot] > 1e-15:
        result[slot] = cold[ci]
    return result


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def run(repo):
    start = time.perf_counter()
    repo = Path(repo)
    source = repo / 'artifacts/final/integration-10pct-v1'
    root = repo / 'artifacts/final/wv3-action-oracle-v1'
    report = repo / 'reports/final'
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Preserve previous evidence: output already exists')
    root.mkdir(parents=True, exist_ok=True)
    definitions = {
        'cold50_one': '原始 Cold50 去除 WV3 Top12 重合后，任意位置最多替换一次，允许不操作；精确单次替换上限。',
        'cold5_one': '原始 B0 名次<=5（不是去重后重新取5个），任意位置最多替换一次；移除模型分数门槛的机会上限。',
        'frozen_gate_reject_only': '保留已执行冻结策略全部门槛及动作，仅允许真值知情地拒绝无益动作；不能改选第二名。',
        'matching_reference': '沿用 P4.3A：逐一 K=0..12，最大化单边 AP 增量之和的一对一匹配，再用完整列表 AP 选 K；不是全局组合 AP 最大值。',
    }
    contract = dict(stage='WV3_ACTION_SPACE_ORACLE', windows=WINDOWS, sample='original fixed 10% evaluation users',
                    source=str(source), definitions=definitions, final_week='not_run', training='not_run',
                    policy_changes=False, baseline_promotion=False, time_budget_seconds=600,
                    tie_rule='candidate saved order then slot; equal AP keeps no-op/smaller K',
                    shortcut='No candidate positive => no improvement possible; K0 is optimal without enumerating K>0.',
                    verification='Replay all selected lists using independent metrics.apk; test single-action matrix by exhaustive enumeration.')
    contract_path = report / 'WV3_ACTION_SPACE_ORACLE_CONTRACT.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text(encoding='utf-8')) == contract
    else:
        dump(contract_path, contract)
    actual = json.loads((report / 'FINAL_INTEGRATION_10PCT.json').read_text(encoding='utf-8'))
    old = json.loads((repo / 'reports/phase4/p4_3a_oracle_headroom.json').read_text(encoding='utf-8'))
    results = {}
    max_error = 0.
    for window, cutoff in WINDOWS.items():
        if time.perf_counter() - start > 600:
            raise TimeoutError('Audit budget exhausted; existing outputs preserved')
        data = joblib.load(source / 'prepared' / cutoff / 'data.joblib')
        truths, valid, base = contexts(data)
        n = len(data['users'])
        np.testing.assert_allclose(base[:, 0].mean(), actual['windows'][window]['baseline_map'], atol=1e-12, rtol=0)
        assert n == actual['windows'][window]['users'] and data['cutoff'] == cutoff
        cold = data['cold']
        groups = cold.groupby('user_index', sort=False).indices
        executed = frame(source / 'models' / window / 'executed.parquet')
        by_user = {r.customer_id: r for r in executed.itertuples()}
        arrays = {name: base.copy() for name in definitions}
        selected = []
        positive_pairs = positive_users = positive_edges = positive_edges5 = 0
        kdist = np.zeros(13, dtype=int)
        for ui, user in enumerate(data['users']):
            warm = list(data['warm_lists'][ui])
            c = cold.iloc[groups.get(ui, [])]
            articles = c.article_id.tolist()
            assert len(articles) <= 50 and len(set(articles)) == len(articles) and not set(articles).intersection(warm)
            truth = truths[ui][0]
            targets = c.target.to_numpy()
            np.testing.assert_array_equal(targets, [int(a in truth) for a in articles])
            positive_pairs += int(targets.sum())
            positive_users += int(targets.any())
            lists = {name: warm.copy() for name in definitions}
            if targets.any():
                y = single_ap_delta(np.repeat(data['relevance'][ui:ui+1], len(c), axis=0), targets,
                                    np.repeat(data['truth_count'][ui], len(c)))
                positive_edges += int((y > 1e-15).sum())
                positive_edges5 += int((y[c.b0_rank.to_numpy() <= 5] > 1e-15).sum())
                lists['cold50_one'] = best_single(warm, articles, c.b0_rank, truth, y)
                lists['cold5_one'] = best_single(warm, articles, c.b0_rank, truth, y, 5)
                ref = oracle_user(warm, articles, truth, y)
                lists['matching_reference'] = ref['best_items']
                kdist[ref['k_star']] += 1
            else:
                kdist[0] += 1
            if user in by_user:
                action = by_user[user]
                candidate = warm.copy()
                slot = int(action.slot) - 1
                assert candidate[slot] == action.removed_article_id
                candidate[slot] = action.cold_article_id
                delta = apk(list(truth), candidate) - apk(list(truth), warm)
                np.testing.assert_allclose(delta, action.delta_ap, rtol=0, atol=1e-12)
                if delta > 1e-15:
                    lists['frozen_gate_reject_only'] = candidate
            for name, items in lists.items():
                ap = [exact_ap(items, t) for t in truths[ui]]
                replay = [apk(list(t), items) for t in truths[ui]]
                max_error = max(max_error, float(np.max(np.abs(np.array(ap) - replay))))
                arrays[name][ui] = ap
                for j, (before, after) in enumerate(zip(warm, items)):
                    if before != after:
                        selected.append(dict(method=name, customer_id=user, slot=j+1, removed=before,
                                             inserted=after, inserted_positive=after in truth, removed_positive=before in truth))
        entries = {}
        for name, a in arrays.items():
            entries[name] = dict(map=float(a[:, 0].mean()), delta=float((a[:, 0]-base[:, 0]).mean()),
                                 benefiting_users=int((a[:, 0]-base[:, 0] > 1e-15).sum()),
                                 segments={s: dict(truth_users=int(valid[:, j].sum()),
                                     map=float(a[valid[:, j], j].mean()), delta=float((a-base)[valid[:, j], j].mean()))
                                           for j, s in enumerate(SEGMENTS) if j})
        assert entries['cold50_one']['delta'] + 1e-12 >= entries['cold5_one']['delta']
        assert entries['matching_reference']['delta'] + 1e-12 >= entries['cold50_one']['delta']
        result = dict(users=n, cold_pairs=len(cold), actions=len(cold)*12, positive_candidate_pairs=positive_pairs,
                      positive_candidate_users=positive_users, beneficial_edges=positive_edges,
                      beneficial_top5_edges=positive_edges5, baseline_map=float(base[:, 0].mean()),
                      methods=entries, matching_k_distribution={str(k): int(v) for k, v in enumerate(kdist)},
                      actual_delta=actual['windows'][window]['delta_map'],
                      old_matching_delta=old['dates'][cutoff]['delta_map'])
        results[window] = result
        save_frame(pd.DataFrame(selected), root / (window + '-selected.parquet'))
        np.savez_compressed(root / (window + '-ap.npz'), baseline=base, **arrays)
        dump(root / (window + '.json'), result)
        print(window, entries, flush=True)
        del data, cold, arrays
        gc.collect()
    output = dict(status='completed', definitions=definitions, windows=results,
                  mean_baseline=float(np.mean([v['baseline_map'] for v in results.values()])),
                  means={name: dict(map=float(np.mean([v['methods'][name]['map'] for v in results.values()])),
                                    delta=float(np.mean([v['methods'][name]['delta'] for v in results.values()]))) for name in definitions},
                  old_matching_mean_delta=old['development_mean_delta'],
                  seconds=time.perf_counter()-start, final_week='not_run', training='not_run', promoted=False)
    dump(report / 'WV3_ACTION_SPACE_ORACLE.json', output)
    dump(report / 'WV3_ACTION_SPACE_ORACLE_VERIFICATION.json', dict(status='pass', independent_ap_max_abs_error=max_error,
         cohort_and_baseline_match=True, labels_rechecked=True, unique_nonoverlap_budget_checked=True, frozen_action_delta_replayed=True))
    print(json.dumps(output['means'], indent=2), flush=True)


if __name__ == '__main__':
    run(Path.cwd())
