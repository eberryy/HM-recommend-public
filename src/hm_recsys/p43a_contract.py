"""P4.3A immutable search definition; no model training at import/register.

The four historical outer windows are development data for this stage only.
Registration never revises earlier reports and never opens the final week.
"""
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import math
import subprocess
import time

import numpy as np

from .p41a_contract import identity, read_json, write_json
from .p42f_contract import GLOBAL, USER, COLD, WARM, DATES, WINDOWS, earlier, now
from .p42f_data import records


RUN_ID = 'p4-3a-v1-map-cashout-hash10-tournament'
SEED = 20260912
THREADS = 4
COMMON_PARAMS = dict(learning_rate=.05, n_estimators=300, num_leaves=31,
    max_depth=6, min_child_samples=100, subsample=1., colsample_bytree=1.,
    reg_lambda=1., reg_alpha=0., random_state=SEED, n_jobs=THREADS,
    deterministic=True, force_col_wise=True, verbosity=-1)
LABEL_GAIN = [0, 1, 4, 10, 20]
TOP_EDGES = [1, 2, 3, 5, 10]
GLOBAL_PERCENTILES = [.90, .95, .98, .99, .995, .999]
CANDIDATE_TOPK = [5, 10, 20, 50]
SLOT_FLOORS = [1, 7, 10, 12]
MAX_ADMISSIONS = [1, 2, 3, 12]
F_FEATURES = ['B0', 'qC', 'old_U', 'F_G', 'G_UR', 'H_q_raw', 'H_q_cal',
              'warm_vulnerability', 'novelty', 'richness']
META_BASES = ['old_U', 'F_G', 'G_U_R', 'H_q_raw', 'H_q_cal', 'H_G_BH', 'H_r']
META_FEATURES = [name+suffix for name in META_BASES for suffix in ('_percentile', '_available')]


def _model_configs():
    result = []
    for sample, leaves, depth, minimum in product(('A1', 'A2'), (15, 31), (4, 6), (50, 200)):
        result.append(dict(id=f'{sample}-L{leaves}-D{depth}-M{minimum}', arm='A',
            role='edge', sampling=sample, sampling_variant=sample, feature_family='base70',
            params=dict(COMMON_PARAMS, objective='lambdarank', num_leaves=leaves,
                        max_depth=depth, min_child_samples=minimum, label_gain=LABEL_GAIN.copy()),
            pos_weight=1., negative_weight=1., row_weight=1., group='user-window',
            target='graded_delta_AP', trainable=True))
    for variant, positive, leaves, depth in product(('B1', 'B2'), (20, 50, 100, 200), (15, 31), (4, 6)):
        result.append(dict(id=f'{variant}-P{positive}-L{leaves}-D{depth}', arm='B',
            role='edge', sampling='A1' if variant == 'B1' else 'A2', sampling_variant=variant,
            feature_family='base70',
            params=dict(COMMON_PARAMS, objective='binary', num_leaves=leaves, max_depth=depth),
            pos_weight=float(positive), negative_weight=1., row_weight=None,
            group=None, target='B_vs_rest', trainable=True))
    result.append(dict(id='E-hard', arm='E', role='edge', sampling='E',
        sampling_variant='all_B_H_plus_hard_N', feature_family='base70',
        params=dict(COMMON_PARAMS, objective='binary'), pos_weight=1., negative_weight=1.,
        row_weight=1., group=None, target='B_vs_rest', trainable=True))
    # Explicit constraints permit four structures, not the prose's claimed six.
    for (depth, leaves), minimum in product(((4, 15), (6, 15), (6, 31), (-1, 63)), (50, 200)):
        result.append(dict(id=f'D-L{leaves}-D{depth}-M{minimum}', arm='D', role='edge',
            sampling='A1', sampling_variant='same_F_2percent_neutral', feature_family='base70_plus_meta',
            params=dict(COMMON_PARAMS, objective='lambdarank', learning_rate=.03,
                        n_estimators=500, num_leaves=leaves, max_depth=depth,
                        min_child_samples=minimum, label_gain=LABEL_GAIN.copy()),
            pos_weight=1., negative_weight=1., row_weight=1., group='user-window',
            target='graded_delta_AP', trainable=True))
    for role in ('count', 'cold', 'warm'):
        params = dict(COMMON_PARAMS, objective='multiclass' if role == 'count' else 'binary')
        if role == 'count':
            params['num_class'] = 4
        result.append(dict(id='C-'+role, arm='C', role=role, family_config_id='C-oracle',
            sampling='oracle', sampling_variant='all_historical_oracle_target_units',
            feature_family='oracle_'+role, params=params, pos_weight=1., negative_weight=1.,
            row_weight=1., group=None,
            target={'count': 'min(K_star,3)', 'cold': 'oracle_selected_cold',
                    'warm': 'oracle_selected_warm_slot'}[role], trainable=True))
    return result


MODEL_CONFIGS = tuple(_model_configs())


def model_configs():
    """Independent dictionaries: callers cannot mutate the frozen grid."""
    return deepcopy(list(MODEL_CONFIGS))


def _f_configs():
    rng = np.random.default_rng(SEED)
    result = []
    for index in range(128):
        weights = rng.dirichlet(np.ones(len(F_FEATURES))).tolist()
        result.append(dict(id=f'F-{index+1:03d}', arm='F', trainable=False,
            weights=dict(zip(F_FEATURES, weights)),
            candidate_topK=int(rng.choice(CANDIDATE_TOPK)),
            replaceable_slot_floor=int(rng.choice(SLOT_FLOORS)),
            max_admissions=int(rng.choice(MAX_ADMISSIONS)),
            score_percentile_gate=float(rng.choice(GLOBAL_PERCENTILES))))
    return result


F_CONFIGS = tuple(_f_configs())


def f_configs():
    return deepcopy(list(F_CONFIGS))


def c_blends():
    result = []
    for weights in product((0., .5, 1.), repeat=3):
        if any(weights):
            result.append(dict(id=f'C-blend-{len(result)+1:02d}',
                weights=dict(zip(('old_U', 'G_UR', 'H_q_cal'), weights))))
    return result


def _deadline(start_epoch, deadline_epoch):
    start, stop = float(start_epoch), float(deadline_epoch)
    if not math.isfinite(start) or not math.isfinite(stop) or stop-start != 7200.:
        raise ValueError('P4.3A approved session budget is exactly two hours, including preparation')
    return dict(start_epoch=start, deadline_epoch=stop,
                start_utc=datetime.fromtimestamp(start, timezone.utc).isoformat(),
                deadline_utc=datetime.fromtimestamp(stop, timezone.utc).isoformat(),
                total_seconds=7200, includes='preparation, implementation, registration, training, search and verification',
                at_deadline='pause_and_report_partial_progress; preserve all completed/failed/pending configurations; no silent scope reduction',
                resume='manual direction required; no resetting this timer when starting a new process')


def _trusted_prepared(repo, f_contract):
    root = Path(f_contract['reuse_root']) if 'reuse_root' in f_contract else repo/'artifacts/phase4/p4-2f-v1-relative-utility-hash10-user-oof'
    known = {}
    for filename in ('P4_2F_OUTPUT_MANIFEST.json', 'P4_2H_OUTPUT_MANIFEST.json'):
        for record in records(read_json(repo/'reports/phase4'/filename)):
            known[str(Path(record['path']).resolve()).lower()] = record
    wanted = [root/'prepared'/cutoff/'data.joblib' for cutoff in sorted(set(DATES+list(WINDOWS.values())))]
    wanted += [root/'prepared'/cutoff/'training.npz' for cutoff in DATES]
    result = []
    for path in wanted:
        key = str(path.resolve()).lower()
        if key not in known:
            raise ValueError('prepared asset lacks trusted F/H manifest identity: '+str(path))
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append(deepcopy(known[key]))
    assert len(result) == 17
    return root, result


def register(repo, root=None, *, start_epoch, deadline_epoch):
    """Create the contract once, before any tournament fit or policy scoring."""
    repo = Path(repo).resolve()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=repo, text=True).strip()
    if branch != 'main':
        raise ValueError('P4.3A Cold workspace must remain on main; refusing '+branch)
    dest = repo/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json'
    if dest.exists():
        raise FileExistsError('immutable P4.3A contract already exists')
    budget = _deadline(start_epoch, deadline_epoch)
    if time.time() >= budget['deadline_epoch']:
        raise TimeoutError('approved two-hour session expired before registration')
    output_root = Path(root).resolve() if root is not None else repo/'artifacts/phase4'/RUN_ID
    if not output_root.is_relative_to(repo/'artifacts/phase4'):
        raise ValueError('P4.3A output root must stay inside this repository artifacts/phase4')
    # Imported only at registration, after the independent policy module exists.
    from .p43a_policy import policies
    policy_grid = policies()
    if isinstance(policy_grid, np.ndarray):
        policy_grid = policy_grid.tolist()
    if len(policy_grid) != 1920:
        raise ValueError('A/B/D policy grid must contain exactly1920 configurations')
    h = read_json(repo/'reports/phase4/P4_2H_EXPERIMENT_CONTRACT.json')
    f = read_json(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json')
    assert read_json(repo/'reports/phase4/P4_2H_VERIFICATION.json')['status'] == 'pass'
    assert h['features'] == GLOBAL and len(GLOBAL) == 70
    froot, assets = _trusted_prepared(repo, h)
    authorities = sorted(p for p in (repo/'reports/phase4').iterdir() if p.is_file() and
                         any(p.name.lower().startswith('p4_2'+stage+'_') for stage in 'defgh') and
                         p.suffix.lower() in ('.json', '.md'))
    authorities += [repo/'docs/ROADMAP_PHASE4.zh-CN.md',
                    repo/'reports/phase4/P4_3A_PREFLIGHT.zh-CN.md',
                    (repo / 'docs/research-contracts/private-approval.txt')]
    features = dict(base70=GLOBAL, cold_fields=COLD, warm_fields=WARM, user_fields=USER,
        D_extra14=META_FEATURES, D_exact84=GLOBAL+META_FEATURES,
        D_no_duplicate_base='B0/M4/qC/qW already represented in base70; append only seven named source percentiles and availability flags',
        permitted='only preregistered subsets/monotonic transforms/ranks/percentiles/aggregates; no later feature additions',
        forbidden=['future truth', 'future sale proxy', 'customer ID memorization', 'article ID memorization',
                   'final week 2020-09-16', 'Warm-v2', 'same-window truth-trained historical meta-score'],
        meta_sources=['B0','M4','qC_relative','qW_vulnerability','old_U','F_G','G_UR','H_q_raw','H_q_cal','H_G_BH','H_r'],
        meta_temporal_rule='model training_label_end < scoring_cutoff, independently of source model cutoff; latest eligible existing source',
        meta_required_lineage=['score_source_cutoff','training_label_end','availability'],
        missing_meta='availability0 and transformed value0; no unsafe user-only OOF backfill; no diagnostic outer calibrator',
        percentile_transform='post-overlap same-user complete edge average rank; (ascending_average_rank-1)/(n-1), singleton0.5; high raw score is high percentile',
        cold_norm='reuse original full50 B0/M4/qC normalizations before removing W0 overlap; do not renormalize overlap survivors',
        qW_vulnerability='1-qW_within_user_percentile; qW missing => condition false, not high vulnerability',
        old_U='original R qC-original and R3 qW-R2 clipped-logit difference; not E full-history qC; full50/Top12 before overlap then within-user edge average-tie percentile',
        old_F_H_OOF='user-only OOF is not historical forward-safe; forbidden as historical meta feature')
    features['C_exact'] = dict(
        count=list(USER)+[f'{s}_top{k}_mean' for s in ('B0','qC','old_U','H_q_raw') for k in (1,3,5)]
              +['qW_vulnerability_top1','qW_vulnerability_top2','warm_source_count_mean','warm_source_count_min',
                'warm_source_count_max','warm_zscore_min','warm_zscore_max','warm_zscore_std'],
        cold=list(COLD)+list(USER)+['max_pair_'+s for s in ('cold_rank_minus_warm_rank_pct',
             'cold_percentile_minus_warm_vulnerability','cold_rank_times_warm_slot')]
             +['max_meta_'+s for s in META_FEATURES],
        warm=list(WARM)+list(USER),
        count_aggregation='B0/qC use frozen Cold50-relative percentiles; old_U/H_q_raw use available edge percentiles maximized over12slots perCold; take highest1/3/5 candidate values then mean, empty0; qW largest/second vulnerability, missing0; warm source_count/zscore summaries over12slots, nonfinite excluded, allmissingNaN',
        cold_aggregation='base Cold and user fields once per candidate; max each named pair field and each14meta field over its12slots; fixed zero-filled meta availability rules',
        population='count trains one row per historical action user-window (has Cold); Cold trains every such candidate; Warm trains12originalslots of each action user; noCold inference forcedK0',
        degenerate_features='constant rank summaries remain declared; LightGBM may ignore them; no label-based feature selection')
    contract = dict(stage='P4.3A', run_id=RUN_ID, status='preregistered_before_tournament',
        created_at_utc=now(), git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        branch='main', authority=[identity(p) for p in authorities], historical_reports_immutable=True,
        trusted_F_prepared=assets, reuse_root=str(froot), output_root=str(output_root),
        user_approvals=dict(execute=True, pause_after_two_hours_including_preparation=True,
            F_keeps_tournament_rank_but_does_not_consume_trainable_scaleup_slot=True,
            F_novelty_richness_use_same_cutoff_cross_user_label_free_percentiles=True),
        development_windows=WINDOWS, development_reclassification='only P4.3A and later cash-out; four old outer windows now development/model-selection, not untouched confirmatory evidence',
        historical_cutoffs=DATES, historical_pools={w:earlier(t) for w,t in WINDOWS.items()},
        population=f['population'], evaluation_population=f['outer_population'],
        evaluation_denominator='same frozen ten-percent W0 users including users without Cold; unchanged during100percent training',
        frozen=['W0 Top12 and positions','M4','B0','Cold50'],
        action_space='Cold50 excluding W0 overlap x original Warm12; <=600 edges/user-window',
        labels='exact delta_AP12_single fixed observed-truth list difference; B>0 N==0 H<0; not online causal effect',
        features=features,
        graded_relevance=dict(strong_beneficial='delta>=.10 =>4',weak_beneficial='0<delta<.10 =>3',
            neutral='delta==0 =>1',harmful='delta<0 =>0', label_gain=LABEL_GAIN),
        sampling=dict(A1='all B/H plus exact previous F deterministic2percent N rows; ALL row weights1 (not old inverse-propensity50)',
            A2='all B/H plus ALL hard N from full action space plus deterministic0.5percent remaining N',
            B1='exact A1 rows, positive weight grid, every negative weight1',
            B2='exact A2 rows, positive weight grid, every negative weight1',
            D='A1 rows and every row weight1', E='all B/H and hard N only; no remaining-neutral random sample; every row weight1',
            hard_N='N and any(B0 rank<=10,available qC percentile>=.9,slot>=9,available qW vulnerability>=.8,available old_U within-user edge percentile>=.9)',
            hash='same F SHA256 compact UTF8 JSON [cutoff,customer_id,cold_article_id,warm_slot], first8 big-endian modulo10000; A1 N keep<200, A2 nonhard N keep<50',
            never='never mine hard neutrals from only the old2percent retained table'),
        model_configs=model_configs(), model_count=60,
        model_counts={'A':16,'B':32,'C':3,'D':8,'E':1,'F':0},
        D_grid_correction='explicit constraints =>(depth4,leaves15),(6,15),(6,31),(-1,63) x min_child50/200 =8; prose12 not followed',
        policy_grid=policy_grid, policy_count=1920,
        policy_semantics=dict(arms=['A','B','D'],
            gate_order='rank ALL action edges per user by stable descending score; take topEdge first; AND complete-window average-tie empirical score percentile>=threshold, Cold original B0 rank<=topK, original slot>=floor',
            score_sign='no implicit raw score>0 gate; negative raw model scores can qualify by rank',
            percentile='all scoring-cutoff action edges, labels unused: (L+R-1)/(2*(N-1)), L=count(score<s), R=count(score<=s); N<=1 =>.5; all equal=>.5, so no90percent gate; grid90..99.9 means threshold .90...999',
            weight='positive 1/(edge rank among complete user action space), unchanged after eligibility filtering',
            matching='exact maximum total positive rank-weight matching with cardinality<=K; one Cold and one Warm slot each; original Warm order otherwise unchanged',
            grids=dict(top_edges=TOP_EDGES,global_percentiles=GLOBAL_PERCENTILES,candidate_topK=CANDIDATE_TOPK,
                       slot_floor=SLOT_FLOORS,max_admissions=MAX_ADMISSIONS),
            canonical_order='original Cold order then original Warm slot for score ties; no truth tie-break',
            matching_tie='smaller unsigned selected Top10-edge bitmask among exactly equal float64 objective sums',
            duplicate_results='may cache identical allowed graphs/final lists, but preserve every config ledger row'),
        C=dict(models=['C-count','C-cold','C-warm'],
            oracle='K=0..min(12,nCold), true single-delta maximum-weight exact-cardinality matching; construct each resulting list and exact AP; max AP tie smallestK',
            oracle_boundary='matching optimizes sum single deltas at each K, NOT global exact-AP optimum over all matchings; not candidate feature',
            count_target='min(K_star,3) classes0/1/2/3; inference3plus=3; highest probability tie smallestK',
            count_features='PIT user state plus top1/3/5 B0/qC/old_U/H_q_raw aggregates; weakest/second qW-vulnerability and Warm source confidence aggregates',
            cold_features='Cold relative evidence plus user state plus max/best pair aggregates',
            warm_features='Warm relative evidence plus original slot plus user state',
            binary_weights=1., inference='top Khat Cold candidates and top Khat removable original Warm slots; exact matching between selected sides; preserve original slot positions',
            blend_features=['old_U','G_UR','H_q_cal'],blends=c_blends(),blend_count=26,
            blend_normalization='within-user percentiles only; weights0/.5/1 excludingall0, proportional duplicates retained'),
        E=dict(config='E-hard',policies=[dict(id=f'E-K{k}-S{floor}',candidate_topK=k,replaceable_slot_floor=floor)
            for k,floor in product(CANDIDATE_TOPK,SLOT_FLOORS)],policy_count=16,
            candidate_order='for each policy take max_j challenge(c,j) among that policy allowed original slots; descending max, ties original Cold order',
            inference='take challenger candidate budget then selected Cold x ALL allowed slots; use every binary score>0 edge with original positive probability matching weights, not bestslot-only edges; exact uniqueness matching; no extra reject threshold/max_admissions',
            Warm_order='in-place replacements only; never rerank remaining Warm'),
        F=dict(configs=f_configs(),config_count=128,seed=SEED,random_generator='numpy.default_rng PCG64',
            generation='for each of128: Dirichlet(ones10), then independent uniform choices candidateK/slotfloor/maxadmissions/globalgate, generated once before scoring',
            feature_order=F_FEATURES,
            directions_zh=dict(B0='原冷专家证据越高越好；原B0名次越小越好',qC='相对冷侧倾向百分位越高越好',
                old_U='原跨来源logit差越高越有利冷侧',F_G='F预测单边AP增量越高越好',G_UR='G风险分离净效用越高越好',
                H_q_raw='H原始有益对有害条件分数越高越好',H_q_cal='H历史正式校准条件分数越高越好',
                warm_vulnerability='1减qW相对百分位，越高表示原槽位越易移除',
                novelty='历史购买中新商品事件占比越高越好，仅同cutoff跨用户无标签百分位',
                richness='历史购买事件数越高越好，仅同cutoff跨用户无标签百分位'),
            normalization='other8 signals within-user action-edge percentiles; novelty/richness only same-cutoff across unique frozen users; no labels in rank transforms',
            percentile_transform='ascending average rank mapped(rank-1)/(n-1), singleton0.5; unavailable source=>0; novelty/richness one row per frozen W0 user before broadcasting to edges',
            source_mapping=dict(B0='base b0_user_percentile',qC='base qC_within_user_percentile',
                old_U='old_U_percentile',F_G='F_G_percentile',G_UR='G_U_R_percentile',
                H_q_raw='H_q_raw_percentile',H_q_cal='H_q_cal_percentile',
                warm_vulnerability='1-base qW_within_user_percentile, unavailable0',
                novelty='base novel_purchase_share_0_5',richness='base user_past_purchase_count'),
            gate='same-cutoff complete action-space average-tie empirical percentile of weighted blend >=selectedgate; identical L/R formula to universal policy; AND originalB0 rank<=K and slot>=floor; no topEdge truncation',
            matching='raw Dirichlet-weighted sum of normalized evidence as positive edge weight; cardinality<=selectedmax_admissions; assert eligible positive weights, no extra model score-sign threshold',
            fullscale='no fit; preserve tournament rank but skipF in trainable top2 capacity and advance next qualifying distinct family'),
        oracle_headroom=dict(replay='P4.1A frozen oracle reference',
            new='per development user K0..12 exact-cardinality single-delta matching followed by exact listAP; impossibleK skipped; tie smallestK',
            claim='best among13 deterministic matching proposals, not all-combination globalAP oracle',
            outputs=['oracle meanMAP','delta vsW0','Kstar distribution','cold selected','Warm removed'],
            order='before tournament',feature=False),
        selection=dict(objective='exact equal-weight mean MAP@12 over all4 development windows',
            eligible_status='only completed four-window configs; no partial/pruned means eligible',
            tie_band='find maximum remaining mean; candidates with max_mean-mean<=1e-8 form one comparison band; do not use nontransitive pairwise tolerance sort',
            tie_break=['atleast3/4 nondegrade first','higher worst-window delta','higher warm21plus mean delta',
                       'simpler frozen policy','stable model id then policy id'],
            simple_policy='lexicographically lower max_admissions, smaller candidate budget, higher slot floor, smaller top-edge budget, higher percentile gate; absent knobs use fixed neutral defaults',
            distinct_arms=True, F_keeps_rank_but_excluded_from_training_slots=True),
        pruning=dict(enabled=True,order=list(WINDOWS),minimum_windows=2,
            condition='mean(first2 MAP deltas)<-.0015 AND each of first2<0',
            status='pruned_bad_config',after='skip remaining two only for that model-policy config; keep row; no tournament mean; cannot win'),
        scaleup=dict(pilot_gate=dict(mean_delta_strictly_greater=.0001,nondegrade_windows_atleast=2,worst_delta_atleast=-.001),
            selection='best qualifying config per trainable family; choose best<=2 distinct families; F remains ranked but cannot consume100percentfit slots',
            machine_decisions=['tournament_no_positive_map','tournament_one_arm_scaleup','tournament_top2_scaleup'],
            authority='approved conditionally after gates; two-hour pause supersedes starting further work; resource preflight mandatory',
            full_population='P4.2E H_full semantics, remove historical10percent user hash only; candidate generation before truth; labels end before devcutoff',
            fixed=['development evaluation users','model structure','features','sampling rules','policy','blend','K','slots','percentile'],
            only_change='training historical eligible users10percent to100percent',retuning=False,
            promotion_gate=dict(mean_delta_strictly_greater=0.,nondegrade_windows_atleast=3,worst_delta_atleast=-.0005),
            promotion_boundary='development-selected improvement only, not sealed final-week confirmation',
            negative_overall='not_promoted even if cold improves'),
        budget=budget,resources=dict(threads=THREADS,min_free_disk_gib=15,
            hard_neutral_fullmatrix_lower_bound_gib=4.528,
            safety='bounded-memory construction; do not silently reduce approved samples; stop/pause resource failure and preserve ledger'),
        execution_priority=['oracle','A','B','E','D','C','F'],
        source_binding='append-only run-source manifest established before firstfit; each fit binds exact source identities; no need to pretend not-yet-written stages are implemented',
        failures='keep every failed/pruned/pending config; no overwrites, no silent search expansion',
        required_metrics=['perwindow MAP and delta','mean','std','worst','warm mean delta','cold/sparse mean delta',
                          'inserted positives','removed positives','coverage','replacements','runtime','status'],
        reports='full ledger JSON+CSV, top20 in Chinese FINAL, eachA-F summary, oracle, top2 selection, fullscale ornotrun, independentverification, outputmanifest',
        final_week='not_run',sealed_confirmatory_holdout='2020-09-16',Warm_v2_integrated=False,
        P4_3B_started=False,selected='W0 until fullscale gate evidence',
        hash_purpose='explicit user evidence contract: cached prepared arrays compared to trusted prior F/H identities, authority reports fixed for immutable history; source identities perfit')
    write_json(dest, contract)
    return contract
