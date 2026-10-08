"""P4.2F fixed relative-utility experiment; registration precedes labels/fits."""
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
import subprocess
from .p41a_contract import identity, read_json, write_json
from .p42_contract import WINDOWS, SOURCES, branch_guard

RUN_ID = 'p4-2f-v1-relative-utility-hash10-user-oof'
DATES = ['2019-12-25','2020-01-22','2020-02-19','2020-03-18','2020-04-29','2020-05-27','2020-06-24','2020-07-22']
COLD = ['b0_rank','b0_rank_pct','b0_user_percentile','b0_user_zscore',
        'normalized_margin_to_rank2','normalized_margin_to_rank5','normalized_margin_to_user_median',
        'm4_coarse_rank','m4_coarse_rank_pct','strict_cold_flag','sparse1_5_flag','cold_only','also_in_Warm150',
        'qC_within_user_rank','qC_within_user_rank_pct','qC_within_user_percentile','qC_relative_available']
WARM = ['warm_rank','warm_rank_pct','warm_user_percentile','warm_user_zscore','warm_model_score_available','source_count',
        *[s+x for s in SOURCES for x in ['_present','_rank','_rrf_contribution']],
        'item2vec_present','item2vec_rank','qW_within_user_rank','qW_within_user_percentile','qW_relative_available']
USER = ['user_past_purchase_count','novel_purchase_count_0_5','novel_purchase_share_0_5',
        'low_pop_purchase_count_0_20','low_pop_purchase_share_0_20','novel_purchase_count_0_5_recent84d',
        'novel_purchase_share_0_5_recent84d','days_since_last_0_5_purchase',
        'median_item_interaction_count_at_purchase','median_item_popularity_percentile_at_purchase',
        'user_purchase_events_12w','user_unique_items_12w','user_unique_items_all',
        'active_purchase_days_12w','active_purchase_days_all','days_since_last_purchase',
        'historical_action_windows_available']
PAIR = ['cold_rank_minus_warm_rank_pct','cold_percentile_minus_warm_vulnerability','cold_rank_times_warm_slot']
PERSONAL = ['strict_times_novelty','sparse_times_novelty','cold_rank_pct_times_novelty','warm_rank_times_history_richness']
BASE = COLD + WARM + PAIR
GLOBAL = BASE + USER + PERSONAL
PARAMS = dict(objective='regression_l2',learning_rate=.05,n_estimators=250,num_leaves=31,max_depth=6,
              min_child_samples=200,subsample=1.,colsample_bytree=1.,reg_lambda=1.,reg_alpha=0.,
              random_state=20260912,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)

def now(): return datetime.now(timezone.utc).isoformat()
def earlier(t, cutoffs=DATES):
    return [s for s in cutoffs if date.fromisoformat(s)+timedelta(days=7)<date.fromisoformat(t)]

def register(repo):
    repo=Path(repo).resolve(); branch_guard(repo); dest=repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json'
    if dest.exists(): raise FileExistsError('immutable registration already exists')
    e=read_json(repo/'reports/phase4/P4_2E_EXPERIMENT_CONTRACT.json')
    paths=[repo/'docs/ROADMAP_PHASE4.zh-CN.md',(repo / 'docs/research-contracts/private-approval.txt')]
    paths+=sorted(p for p in (repo/'reports/phase4').glob('*.json') if p.name.lower().startswith(('p4_2d_','p4_2e_')))
    paths += [repo/'reports/phase4'/f'P4_2{s}_FINAL.md' for s in ['D','E']]
    c=dict(stage='P4.2F',run_id=RUN_ID,status='preregistered_before_formal_computation',created_at_utc=now(),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),branch='main',
        authority=[identity(p) for p in paths],user_approval='2026-09-12: historical10%; H user-level 2-fold OOF',
        historical_cutoffs=DATES,outer_windows=WINDOWS,historical_pools={w:earlier(t) for w,t in WINDOWS.items()},
        population='exact prior hash10 H_t: frozen W0 users with mapped pre-cutoff M4 history; no future cold condition',
        outer_population='complete frozen W0 S_t, including users with no Cold candidate',
        excluded_cutoffs={'2019-11-27':'no safe B0'},inputs=e['inputs'],transactions=e['transactions'],catalog=e['catalog'],
        features=dict(G=GLOBAL,H_pair=BASE,user=USER,missing='LightGBM native NaN; Ridge train median(all missing0)+StandardScaler'),
        target='exact AP@12 after single in-place replacement minus exact W0 AP; denominator min(unique future truth count,12)',
        overlap='exclude Cold article already in W0 Top12 after full50 confidence normalization',
        neutral=dict(rate=.02,weight=50.,nonneutral_weight=1.,keep_all_nonzero=True,
                     hash='SHA256 UTF8 compact JSON [cutoff,customer_id,cold_article_id,warm_slot]; first8 big-endian integer % 10000 < 200'),
        G_params=PARAMS,H_pair_params=PARAMS,
        H=dict(oof='2 fixed customer folds; all dates same fold; sha256 UTF8 first8 big-endian %2; train opposite fold',
               oof_timeline='all labels earlier than outer; NOT forward-at-each-historical-date; no user seen by own scoring fold model',
               threshold='Ridge alpha=1 fit_intercept=True solver=svd; one equally weighted user-window; target=max(0,max_edge S_OOF)-max(0,max_edge true delta)',
               pair_max='score ALL historical action edges, not sampled edges',
               residual='r=d-tau_fixed; outer uses only label_end<outer; historical prior b(t) only rows label_end<t',
               shrink_kappa=2.,no_history='b=0, tau remains fixed user-state regression, not necessarily scalar intercept',
               fit_budget={'global':4,'pair_full':4,'pair_oof':8,'ridge':4}),
        novelty=dict(count='all global transaction events strictly before purchase DATE; duplicates retained; same-day events excluded',
                     percentile='ascending midrank of purchase-time item count among optimistic full static catalog; all zero-history items included; (rank-1)/(catalog size-1)',
                     history='all observed events before scoring cutoff; recent84d=[cutoff-84days,cutoff)',
                     unavailable_last_novel='NaN + native missing handling; no current count backfill'),
        relative=dict(qC='latest existing P4.2E full model with training label_end<scoring cutoff; no current-date OOF; absent=>NaN,available0',
                      qW='latest existing P4.2R3 qW with training label_end<scoring cutoff; absent=>NaN,available0',
                      probability='only transient normalization input; absolute probabilities/logits forbidden in G/H',
                      confidence='reuse frozen full50 B0 and exact Top12 Warm normalization before overlap removal',
                      rank='descending evidence, average ties for percentile; qC deterministic ties B0 rank/article; qW ties original slot',
                      warm_vulnerability='1-warm_user_percentile; fallback (warm_slot-1)/11 only if warm score unavailable'),
        matching=dict(eligibility='utility>0, equality rejected',weight='utility',algorithm='scipy linear_sum_assignment',
                      order='Cold B0 rank then article_id; Warm original slots then zero-cost dummy columns',max1=False,K_admit=None,
                      unique=True,approximation='sum predicted single-edge delta; exact final list AP recomputed'),
        gates=dict(overall_mean_delta_gt=0.,overall_nondegrade_min=3,overall_worst_delta_min=-.0002,
                   warm_mean_delta_min=-.0001,warm_window_delta_min=-.0002,warm_pass_windows_min=3,
                   cold_mean_delta_gt=0.,cold_nondegrade_min=3,cold_only_positive_windows_min=2,inserted_gt_removed=True),
        selection='only G/H; if both pass larger mean overall, tie<=1e-12 larger cold mean, exact tie G',
        diagnostics=dict(novelty_quartiles='historical training user-window 25/50/75 quantiles, ties lower bucket, apply unchanged outer',
                         richness='historical training all-history purchase-count tertiles, ties lower bucket',
                         near_zero_b=1e-12,admission_buckets=['0','1','2','3','4+'],
                         action='complete outer action space unweighted Pearson/Spearman, beneficial-vs-harmful and beneficial-vs-rest AUC/AP',
                         segments='reuse Phase4 segment truth sets and truth-user denominators; no redefinition'),
        budget=dict(pilot_seconds=600,formal_seconds=7200,threads=4,min_free_disk_gib=15,
                    target_process_RAM_gib=6,estimated_compute_minutes=[30,120],estimate_status='provisional; validate via fixed pilot',
                    stop='input drift, nonfinite output, contract/invariant violation, resource limit or >2h; preserve failure; W0 fallback; no downscale'),
        final_week='not_run',Warm_v2_integrated=False,P4_3_started=False,
        hash_purpose='explicit experiment evidence contract; compare reused inputs to trusted earlier manifests, not new hashes alone')
    write_json(dest,c)
    write_json(repo/'reports/phase4/p4_2f_feature_contract.json',dict(stage='P4.2F',**c['features'],novelty=c['novelty'],relative=c['relative']))
    return c
