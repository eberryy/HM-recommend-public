"""Research-led supervision audit and isolated reopened Warm experiments."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import time

import numpy as np

from .warm_v2_contract import ARTIFACT, REPORT, assert_branch, now, read, write
from .warm_v2_engine import Engine, connection, literal

SOURCES=[
    {'url':'https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/hello-world-2nd-place-solution',
     'finding':'Second-place authors discuss multi-stage ranking and MAP-oriented LambdaRank; one component uses 2-3 months of transactions. This does not establish the exact number of supervised target weeks.'},
    {'url':'https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/sirius-3rd-place-solution',
     'finding':'Third-place writeup uses recalled positives and30x negative sampling; BPR user-item similarity improves their model. Our30x ratio alone is therefore not proven wrong; sampling construction and learned semantics need controlled evidence.'},
    {'url':'https://blog.recruit.co.jp/data/articles/kaggle-h-and-m/',
     'finding':'The11th-place authors found YetiRank best and LightGBM xendcg better than lambdarank; the outcome is architecture/data-specific, not a promised improvement.'},
    {'url':'https://lightgbm.readthedocs.io/en/stable/Parameters.html',
     'finding':'rank_xendcg is an available listwise ranking objective; it remains NDCG-oriented, NOT a direct MAP objective.'},
    {'url':'https://arxiv.org/abs/1911.09798',
     'finding':'Bruch proposes a cross-entropy ranking loss linked to NDCG. Borrow the alternative listwise supervision idea, not a claim that this is a new2026 paper.'},
]


def audit():
    assert_branch()
    c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')
    engine=Engine(c)
    windows={}
    for w,p in c['rolling_protocol'].items():
        cutoff=p['inner_train'][0]
        sample,meta=engine.cached_data(cutoff,'sample')
        with connection() as con:
            con.execute(f'CREATE VIEW sampled AS SELECT * FROM read_parquet({literal(sample)})')
            con.execute('CREATE TEMP TABLE selected_users AS SELECT DISTINCT customer_id FROM sampled')
            con.execute(f'CREATE TEMP TABLE full_groups AS SELECT f.customer_id,f.article_id,f.target,f.candidate_rank FROM read_parquet({literal(engine.base_path(cutoff))}) f SEMI JOIN selected_users USING(customer_id)')
            full=con.execute('SELECT count(*),sum(target),count(DISTINCT customer_id) FROM full_groups').fetchone()
            coverage=con.execute('''SELECT count(*) FILTER(WHERE f.target=0 AND f.candidate_rank<=12),
                count(*) FILTER(WHERE f.target=0 AND f.candidate_rank<=12 AND s.customer_id IS NOT NULL),
                count(*) FILTER(WHERE f.target=0 AND f.candidate_rank<=20),
                count(*) FILTER(WHERE f.target=0 AND f.candidate_rank<=20 AND s.customer_id IS NOT NULL)
                FROM full_groups f LEFT JOIN sampled s USING(customer_id,article_id)''').fetchone()
            lengths=con.execute('''SELECT avg(n),median(n),min(n),max(n) FROM
                (SELECT count(*) n FROM full_groups GROUP BY customer_id)''').fetchone()
            zero=con.execute(f'''SELECT count(*) FILTER(WHERE npos=0),count(*) FROM
                (SELECT customer_id,sum(target) npos FROM read_parquet({literal(engine.base_path(cutoff))}) GROUP BY customer_id)''').fetchone()
            assert full[1]==meta['positive_rows'] and full[2]==meta['groups']
        windows[w]={'train_cutoff':cutoff,'inner_validation':p['inner_validation'],
            'gap_from_last_training_label_to_inner_days':(date.fromisoformat(p['inner_validation'])-(date.fromisoformat(cutoff)+timedelta(days=6))).days,
            'outer_train_cutoffs':p['outer_train'],
            'gap_from_last_training_label_to_outer_days':(date.fromisoformat(p['outer_validation'])-(date.fromisoformat(max(p['outer_train']))+timedelta(days=6))).days,
            'sample_rows':meta['rows'],'sample_groups':meta['groups'],'positive_pairs':meta['positive_rows'],
            'full_same_group_rows':int(full[0]),'row_expansion_factor':full[0]/meta['rows'],
            'sample_average_group_size':meta['rows']/meta['groups'],'full_average_group_size':lengths[0],
            'rrf_top12_negative_pairs':coverage[0],'rrf_top12_negative_pairs_retained':coverage[1],
            'rrf_top12_negative_retention':coverage[1]/coverage[0],
            'rrf_top20_negative_pairs':coverage[2],'rrf_top20_negative_pairs_retained':coverage[3],
            'rrf_top20_negative_retention':coverage[3]/coverage[2],
            'zero_positive_groups':zero[0],'all_training_cutoff_groups':zero[1],
            'dense_float32_matrix_lower_bound_bytes':int(full[0])*84*4}
    passed=max(v['row_expansion_factor'] for v in windows.values())<=8 and min(1-v['rrf_top12_negative_retention'] for v in windows.values())>=.4
    result={'audited_at':now(),'sources':SOURCES,'windows':windows,
        'training_targets':'one historical week per inner fit, two spaced historical weeks per outer refit;12-week features are NOT12-week supervised training',
        'hard_negative_proxy':'Project-defined: unpurchased candidates ranked<=12 or20 by frozen weighted-RRF/expanded order, conditional on unchanged positive-containing training users. Not oracle labels for feature construction; not calibrated model-hard negatives.',
        'decision':'run fixed-pool supervision factorial' if passed else 'reassess smallest discriminating intervention',
        'audit_gate_passed':passed,'audit_gate':'all four training cutoffs omit>=40% of top12 RRF negatives; full group row expansion<=8x',
        'caveat':'Missing top negatives is an observed mechanism, not proof that full negatives improve MAP. Zero-positive groups yield no pairwise ordering signal; do not silently add them.',
        'final_week':'not_run','outer_labels_used_for_this_audit':False}
    write(REPORT/'WV2_300_SUPERVISION_AUDIT.json',result)
    print({w:{k:v[k] for k in ('row_expansion_factor','rrf_top12_negative_retention','gap_from_last_training_label_to_inner_days')} for w,v in windows.items()},flush=True)
    return result


def contract():
    path=REPORT/'WARM_RESEARCH_REOPEN_CONTRACT.json'
    if path.exists():
        return read(path)
    a=read(REPORT/'WV2_300_SUPERVISION_AUDIT.json')
    assert a['audit_gate_passed']
    result={'registered_at':now(),'authorization':'User reopened autonomous research->audit->experiment workflow after prior local feature plateau.',
        'status':'registered_before_new_fits','stage':'WV2.4 fixed-pool supervision geometry',
        'supersedes':'Prior stop recommendation and old first-feature-stage trial budget only; all previous evidence remains immutable.',
        'frozen':'same84 Warm-v1 features; exact existing candidate IDs/budget; original10% users, full-history feature statistics; original inner/outer cutoffs; same MAP@12 denominator and fallback; default31leaves; max200rounds inner patience20.',
        'trials':[
            {'id':'WV2-301','sampling':'full_positive_groups','objective':'lambdarank','hypothesis':'补全原正例用户组中的全部候选负例，检验训练时漏掉高排名竞争者是否限制Top12排序。'},
            {'id':'WV2-302','sampling':'distribution_30x','objective':'rank_xendcg','hypothesis':'保持完全相同的30:1样本，仅用另一种面向整组候选的排序损失，检验损失形式的影响。'},
            {'id':'WV2-303','sampling':'full_positive_groups','objective':'rank_xendcg','hypothesis':'与301和302形成2x2对照，区分补全候选竞争者与整组损失的交互作用。'},
        ],
        'baseline':'WV2-000 = distribution30x + lambdarank',
        'screen_gate':{'mean_population_delta_min':.0001,'positive_windows_min':3,'worst_population_delta_min':-.0005},
        'outer_selection':'Only highest mean passing inner configuration receives formal four-window confirmation; tie by id. Maximum1 confirmation for this factorial; no parameter sweep. Other passing arms retained for historical attribution.',
        'milestone_gate':{'mean_delta_min':.0005,'nondegrade_min':3,'worst_delta_min':-.0003,'large_gain_review':.001},
        'cost':'CPU8threads;3x4 inner fits estimated10-30min; selected4outerfits5-15min; full cached rows bounded by audit<=8x. Stop any fit at >15min or observed memory >12GiB for diagnosis; noGPU/sharedserver.',
        'failure_path':'If sampling/objective factor does not generalize, next audit is train-target freshness or learned collaborative score; do not return to nearly identical feature search and do not infer candidate ceiling.',
        'final_week':'not_run','Cold_Admission':'read_only','automatic_merge_or_push':False}
    write(path,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['audit','register'])
    a=p.parse_args()
    audit() if a.command=='audit' else contract()
