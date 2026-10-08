"""Read-only localization of frozen WV3 x Cold50 decision failures."""
from pathlib import Path
import gc
import json
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from .p42e_resources import memory
from .final_oracle_audit import WINDOWS, dump
from .metrics import apk
from .p42d_stats import single_ap_delta
from .p42f_data import frame, save_frame

METHODS = {
    'actual': '原冻结策略：完整动作首选，再检查原B0前5和全窗99.9百分位，最多1次，无次选回退。',
    'model_candidate_oracle_slot': '门槛前模型首选动作所属商品固定；真值选择最佳位置或不操作；移除两个准入门槛。',
    'admitted_candidate_oracle_slot': '仅对原策略已准入用户，固定其商品，真值选位置或不操作。',
    'truth_candidate_model_slot': '有正例候选用户中，按模型每商品最佳位置分数选最高分正例；用该商品模型最佳位置，移除准入门槛；其他用户不操作。',
    'truth_candidate_model_slot_gated': '同上一项，但仍检查该商品原B0前5及原全窗99.9百分位；只绕过全用户动作首选竞争。',
    'oracle_any': '全Cold50任意位置最多替换1次的精确真值上限，允许不操作。',
    'oracle_tail': '全Cold50仅原位置8至12最多替换1次的精确真值上限，允许不操作。',
    'oracle_tail_top5': '原B0前5候选，仅原位置8至12最多替换1次的精确真值上限。',
}


def percentile(sorted_scores, value):
    n = len(sorted_scores)
    if n <= 1:
        return .5
    left = np.searchsorted(sorted_scores, value, side='left')
    right = np.searchsorted(sorted_scores, value, side='right')
    return float((left + right - 1) / (2 * (n - 1)))


def oracle_edge(y, rows=None, floor=0):
    rows = np.arange(len(y)) if rows is None else np.asarray(rows)
    if not len(rows):
        return None
    a, b = np.unravel_index(np.argmax(y[rows, floor:]), (len(rows), 12-floor))
    return (int(rows[a]), int(b+floor)) if y[rows[a], b+floor] > 1e-15 else None


def decisions(scores, ranks, target, y, global_sorted):
    """All label-aware interventions are diagnostic, never deployment policy."""
    ci, slot = map(int, np.unravel_index(np.argmax(scores), scores.shape))
    passes = lambda i, j: ranks[i] <= 5 and percentile(global_sorted, scores[i, j]) >= .999
    admitted = passes(ci, slot)
    actions = dict(actual=(ci, slot) if admitted else None,
                   model_candidate_oracle_slot=oracle_edge(y, [ci]),
                   admitted_candidate_oracle_slot=oracle_edge(y, [ci]) if admitted else None,
                   truth_candidate_model_slot=None, truth_candidate_model_slot_gated=None,
                   oracle_any=oracle_edge(y), oracle_tail=oracle_edge(y, floor=7),
                   oracle_tail_top5=oracle_edge(y, np.flatnonzero(ranks <= 5), 7))
    positive = np.flatnonzero(target)
    if len(positive):
        a, b = np.unravel_index(np.argmax(scores[positive]), (len(positive), 12))
        chosen = int(positive[a]), int(b)
        actions['truth_candidate_model_slot'] = chosen
        if passes(*chosen):
            actions['truth_candidate_model_slot_gated'] = chosen
    return actions, (ci, slot)


def summary_frame(f):
    return dict(count=len(f), mean=float(f.mean()) if len(f) else None,
                median=float(f.median()) if len(f) else None,
                q25=float(f.quantile(.25)) if len(f) else None,
                q75=float(f.quantile(.75)) if len(f) else None)


def run(repo):
    repo = Path(repo)
    root = repo/'artifacts/final/wv3-localization-v1'
    report = repo/'reports/final'
    source = repo/'artifacts/final/integration-10pct-v1'
    if root.exists():
        raise FileExistsError('Never overwrite an earlier diagnostic run')
    resources = dict(free_disk_gib=shutil.disk_usage(repo).free/2**30,
                     free_ram_gib=memory().available/2**30)
    assert resources['free_disk_gib'] >= 15 and resources['free_ram_gib'] >= 1.5
    contract = dict(stage='FINAL-D2: WV3 decision localization', windows=WINDOWS, source=str(source),
        methods=METHODS, population='all original fixed 10% users; full saved post-overlap Cold50',
        model='frozen A1-L15-D6-M50; reuse saved scores.npy; no prediction or fitting',
        gate='original B0 rank<=5 AND global average-tie action percentile>=.999; no score>0 gate',
        ordering='saved candidate row then slot; stable descending score; candidate score=max over12 slots',
        truth_choice='among positive candidates select model highest edge, NOT true highest AP candidate',
        ranks='positive candidate best edge rank among all user edges; max-slot candidate rank among all user candidates; global percentile among all window edges',
        item_summary='descriptive unique articles and positive concentration only; no new availability statistics',
        hypothesis_tests=['candidate-first failure', 'position failure conditional on correct candidate', 'gates discard identifiable positives', 'tail-safe headroom'],
        historical_no_read=True, final_week='not_run', training='not_run', policy_change=False, promotion=False,
        budget_seconds=600, resources=resources, stop='identity/cohort/score/AP invariant or budget failure; preserve outputs and WV3; no rescue search',
        preregistration='written before loading labels/scores for this diagnostic; prior oracle outcomes already known')
    root.mkdir(parents=True)
    dump(report/'WV3_LOCALIZATION_CONTRACT.json', contract)
    start = time.perf_counter()
    actual = json.loads((report/'FINAL_INTEGRATION_10PCT.json').read_text(encoding='utf-8'))
    old_oracle = json.loads((report/'WV3_ACTION_SPACE_ORACLE.json').read_text(encoding='utf-8'))
    results, positive_all = {}, []
    max_ap_error = 0.
    for window, cutoff in WINDOWS.items():
        assert cutoff < '2020-09-16'
        data = joblib.load(source/'prepared'/cutoff/'data.joblib')
        cold, users = data['cold'], data['users']
        scores = np.load(source/'models'/window/'scores.npy', mmap_mode='r')
        assert scores.shape == (len(cold), 12) and np.isfinite(scores).all()
        assert data['cutoff'] == cutoff and len(users) == actual['windows'][window]['users']
        global_sorted = np.sort(np.asarray(scores).ravel())
        saved_edges = np.load(source/'models'/window/'policy/user-top10-global-edge.npy')
        saved_masks = np.load(source/'models'/window/'policy/policy-user-selected-mask.npy')[0]
        executed = frame(source/'models'/window/'executed.parquet')
        events = {r.customer_id: r for r in executed.itertuples()}
        assert len(events) == len(executed)
        groups = cold.groupby('user_index', sort=False).indices
        gains = {k: np.zeros(len(users)) for k in METHODS}
        counts = {k: dict(actions=0, beneficial=0, harmful=0, neutral=0, inserted_positive=0, removed_positive=0) for k in METHODS}
        records, positives, user_records = [], [], []
        chosen_positive = opportunity_users = model_hits_in_opportunity = 0
        for ui, user in enumerate(users):
            if ui % 256 == 0 and time.perf_counter()-start > 600:
                raise TimeoutError('Diagnostic budget exceeded')
            idx = np.asarray(groups.get(ui, []), dtype=int)
            warm = list(data['warm_lists'][ui])
            truth = data['truthsets'][user]
            baseline = apk(list(truth), warm)
            np.testing.assert_allclose(baseline, data['baseline_ap'][ui], atol=1e-12, rtol=0)
            if not len(idx):
                assert not saved_masks[ui] and user not in events
                continue
            c = cold.iloc[idx]
            articles = c.article_id.tolist()
            assert len(c) <= 50 and len(set(articles)) == len(c) and not set(articles).intersection(warm)
            target = c.target.to_numpy()
            np.testing.assert_array_equal(target, [int(a in truth) for a in articles])
            rel = np.array([a in truth for a in warm], dtype=int)
            np.testing.assert_array_equal(rel, data['relevance'][ui])
            s, ranks = np.asarray(scores[idx]), c.b0_rank.to_numpy()
            y = single_ap_delta(np.repeat(rel[None], len(c), axis=0), target, np.repeat(len(truth), len(c)))
            actions, (ci, slot) = decisions(s, ranks, target, y, global_sorted)
            assert int(saved_edges[ui, 0]) == int(idx[ci])*12+slot
            assert bool(saved_masks[ui]) == (actions['actual'] is not None)
            if actions['actual']:
                event = events[user]
                assert event.cold_article_id == articles[ci] and event.slot == slot+1
            chosen_positive += int(target[ci])
            opportunity_users += int(target.any())
            model_hits_in_opportunity += int(target[ci])
            user_records.append(dict(customer_id=user, candidate_count=len(c), positives=int(target.sum()),
                model_candidate=articles[ci], model_candidate_positive=int(target[ci]),
                model_slot=slot+1, model_b0_rank=int(ranks[ci]), model_global_percentile=percentile(global_sorted,s[ci,slot]),
                actual_admitted=actions['actual'] is not None))
            candidate_order = np.argsort(-s.max(axis=1), kind='stable')
            candidate_rank = np.empty(len(c), int); candidate_rank[candidate_order] = np.arange(1, len(c)+1)
            edge_order = np.argsort(-s.ravel(), kind='stable')
            edge_rank = np.empty(s.size, int); edge_rank[edge_order] = np.arange(1, s.size+1)
            for p in np.flatnonzero(target):
                j = int(np.argmax(s[p]))
                best = int(np.argmax(y[p]))
                positives.append(dict(window=window, customer_id=user, article_id=articles[p],
                    strict_cold=int(c.iloc[p].strict_cold_flag), b0_rank=int(ranks[p]),
                    model_candidate_rank=int(candidate_rank[p]), best_edge_user_rank=int(edge_rank[p*12+j]),
                    best_edge_global_percentile=percentile(global_sorted, s[p,j]),
                    original_top1_candidate=bool(p==ci), pass_b0_top5=bool(ranks[p]<=5),
                    pass_global=bool(percentile(global_sorted,s[p,j])>=.999),
                    model_slot=j+1, oracle_slot=best+1, model_slot_delta=float(y[p,j]), oracle_delta=float(y[p,best]),
                    model_removes_positive=bool(rel[j]), positive_slot_available=bool((y[p]>1e-15).any())))
            for method, action in actions.items():
                if action is None:
                    continue
                i, j = action
                items = warm.copy(); items[j] = articles[i]
                delta = apk(list(truth), items)-baseline
                max_ap_error = max(max_ap_error, abs(delta-float(y[i,j])))
                assert abs(delta-y[i,j]) < 1e-12
                gains[method][ui] = delta
                count = counts[method]; count['actions'] += 1
                count['beneficial' if delta>1e-15 else 'harmful' if delta < -1e-15 else 'neutral'] += 1
                count['inserted_positive'] += int(target[i]); count['removed_positive'] += int(rel[j])
                records.append(dict(method=method, customer_id=user, inserted=articles[i], removed=warm[j], slot=j+1, delta_ap=delta))
        p = pd.DataFrame(positives)
        positive_all.append(p)
        funnel = dict(positive_pairs=len(p), opportunity_users=opportunity_users, model_top1_positive_users=chosen_positive,
            positive_pairs_pass_b0=int(p.pass_b0_top5.sum()), positive_pairs_pass_global=int(p.pass_global.sum()),
            positive_pairs_pass_both=int((p.pass_b0_top5 & p.pass_global).sum()),
            model_top1_positive_pass_b0=int((p.original_top1_candidate & p.pass_b0_top5).sum()),
            model_top1_positive_pass_global=int((p.original_top1_candidate & p.pass_global).sum()),
            model_top1_positive_pass_both=int((p.original_top1_candidate & p.pass_global & p.pass_b0_top5).sum()))
        rank_summary = {k:summary_frame(p[k]) for k in ['b0_rank','model_candidate_rank','best_edge_user_rank','best_edge_global_percentile']}
        rank_summary['candidate_positive_recall'] = {str(k):float((p.model_candidate_rank<=k).mean()) for k in [1,5,10,20,50]}
        rank_summary['global_positive_survival'] = {str(q):int((p.best_edge_global_percentile>=q).sum()) for q in [.9,.95,.99,.995,.999]}
        metrics = {k:dict(delta_map=float(g.mean()), map=float(data['baseline_map']+g.mean()), **counts[k]) for k,g in gains.items()}
        np.testing.assert_allclose(metrics['actual']['delta_map'], actual['windows'][window]['delta_map'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(metrics['oracle_any']['delta_map'], old_oracle['windows'][window]['methods']['cold50_one']['delta'], atol=1e-12, rtol=0)
        assert metrics['actual']['actions'] == len(events)
        result = dict(users=len(users), candidates=len(cold), baseline_map=float(data['baseline_map']), methods=metrics,
            funnel=funnel, positive_ranks=rank_summary,
            positive_position=dict(model_slot_distribution={str(k):int((p.model_slot==k).sum()) for k in range(1,13)},
                removes_positive=int(p.model_removes_positive.sum()), beneficial=int((p.model_slot_delta>1e-15).sum()),
                positive_pairs=len(p)),
            positive_items=dict(unique=int(p.article_id.nunique()), largest_item_pairs=int(p.article_id.value_counts().max()),
                top5_item_pairs=int(p.article_id.value_counts().head(5).sum())))
        results[window] = result
        save_frame(p,root/f'{window}-positive-candidates.parquet')
        save_frame(pd.DataFrame(records),root/f'{window}-diagnostic-actions.parquet')
        save_frame(pd.DataFrame(user_records),root/f'{window}-users.parquet')
        np.savez_compressed(root/f'{window}-gains.npz', **gains)
        dump(root/f'{window}.json', result)
        print(window, json.dumps(dict(funnel=funnel,methods=metrics)),flush=True)
        del data,cold,scores,global_sorted
        gc.collect()
    all_p = pd.concat(positive_all)
    result = dict(status='completed',stage=contract['stage'],methods=METHODS,windows=results,
        means={k:dict(delta_map=float(np.mean([r['methods'][k]['delta_map'] for r in results.values()])),
                      map=float(np.mean([r['methods'][k]['map'] for r in results.values()]))) for k in METHODS},
        positive_unique_items_across_windows=int(all_p.article_id.nunique()),
        seconds=time.perf_counter()-start,final_week='not_run',training='not_run',promotion=False)
    dump(report/'WV3_LOCALIZATION.json', result)
    dump(report/'WV3_LOCALIZATION_VERIFICATION.json',dict(status='pass',max_independent_ap_error=max_ap_error,
        actual_top1_masks_actions_replayed=True,previous_oracle_replayed=True,all_labels_verified=True,
        cohort_no_overlap_unique_budget=True,score_shape_finite=True))
    print(json.dumps(result['means'],indent=2),flush=True)


if __name__ == '__main__':
    run(Path.cwd())
