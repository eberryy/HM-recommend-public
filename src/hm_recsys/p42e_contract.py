"""P4.2E immutable registration; no generation or fitting in this module."""
from __future__ import annotations

import shutil
import subprocess
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

from .p42e_resources import memory

from .p41a_contract import identity, read_json, write_json
from .p42_contract import WINDOWS, FEATURE_SPEC, branch_guard
from .p42_propensity import MODEL_PARAMS

RUN_ID = 'p4-2e-v1-full-history-oof-calibration'
DATES = ['2019-12-25','2020-01-22','2020-02-19','2020-03-18',
         '2020-04-29','2020-05-27','2020-06-24','2020-07-22']
FRACTIONS = [.01,.005,.001,.0005,.0001]
BINS = [0,.9,.99,.995,.999,.9995,.9999,1]


def now():
    return datetime.now(timezone.utc).isoformat()


def preregister(repo):
    repo=Path(repo).resolve(); branch_guard(repo)
    report=repo/'reports/phase4'; dest=report/'P4_2E_EXPERIMENT_CONTRACT.json'
    if dest.exists():
        raise FileExistsError('P4.2E registration exists; do not overwrite')
    old=read_json(report/'P4_2R3_EXPERIMENT_CONTRACT.json')
    assert old['feature_spec']['qC']==FEATURE_SPEC['qC']
    authority=['P4_2R3_FINAL.md','P4_2R3_EXPERIMENT_CONTRACT.json','P4_2R3_OUTPUT_MANIFEST.json',
               'P4_2D_FINAL.md','P4_2D_metrics.json','P4_2D_EXPERIMENT_CONTRACT.json','P4_2D_OUTPUT_MANIFEST.json',
               'p4_2d_qc_extreme_tail.json','p4_2d_b0_vs_qc_positive_ranking.json',
               'p4_2d_pair_label_utility.json','p4_2d_training_scaleup_feasibility.json']
    estimates=read_json(report/'p4_2d_training_scaleup_feasibility.json')
    n=sum(x['estimated_hash100']['users'] for x in estimates.values())
    c=dict(stage='P4.2E',run_id=RUN_ID,status='preregistered_before_formal_computation',created_at_utc=now(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),branch='main',
        authority={k:identity(report/k) for k in authority},
        roadmap=identity(repo/'docs/ROADMAP_PHASE4.zh-CN.md'),
        source_review={p.name:identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42*.py'))
                       if not p.name.startswith('p42e')},
        historical_cutoffs=DATES,outer_cutoffs=WINDOWS,
        historical_pools={w:[t for t in DATES if (date.fromisoformat(t)+timedelta(days=7)).isoformat()<d]
                          for w,d in WINDOWS.items()},
        excluded_cutoffs={'2019-11-27':'no safe earlier B0; no M4-only fallback'},
        inputs=old['inputs'],transactions=old['transactions'],catalog=old['catalog'],
        old_training=old['prepared_reuse'],old_run_id=old['run_id'],
        population=dict(full='next-week any purchase AND >=1 mapped item among pre-cutoff recent20 distinct items; remove only user hash',
                        outer='DuckDB hash(customer_id)%1000000<100000; exact frozen P4.2D users',
                        cold_truth_selection=False,new_user_generalization=False),
        generation=dict(order=['roster','past history','M4 Top200','safe B0','Cold50',
                               'persist all unlabelled selections','join future truth'],
                        chunk_users=1536,device='cuda',M4_B0_frozen=True,
                        parity='all old H users, M4 Top200 identities, Cold50 identities/ranks exact; no tolerate rank swaps',
                        score_atol=1e-5,score_rtol=0),
        feature_spec=FEATURE_SPEC['qC'],raw_model_params=MODEL_PARAMS,
        preprocessing='same p42_propensity: training-only numeric median and StandardScaler ddof0; binary and availability unscaled',
        model_fits=dict(max_raw_fits=12,per_outer='full + fold0 train + fold1 train',qW_fits=0,
                       no_sampling=True,no_class_weight=True,stop_on_nonconvergence=True),
        oof=dict(hash='int.from_bytes(sha256(customer_id UTF8).digest()[:8], big) % 2',
                 unit='customer_id across ALL historical dates',folds=2,each_row_predictions=1,
                 user_disjoint=True,temporal_rule='all historical label_end < outer; no inner temporal cross-fitting'),
        calibrator=dict(formula='q_cal=expit(a+b*z_raw); z_raw=raw logistic linear predictor=logit(q_raw)',
                        approval='Lyra approved unclipped raw linear predictor on 2026-09-10',
                        epsilon=1e-6,epsilon_scope='probability diagnostic numerical protection and cross-source U only; never calibration input',
                        fitting='unweighted unregularized binary logistic MLE; scipy L-BFGS-B; analytic gradient',
                        initial=[0.,1.],maxiter=200,ftol=1e-12,gtol=1e-8,
                        b_positive_required=True,outer_labels='diagnostic-only fits; never calibrator training',
                        parity='raw/cal probability ordering AND tie partitions exactly identical; no secondary-score repair'),
        evaluation=dict(roles=['Q_old','Q_full_raw','Q_full_cal'],population='full frozen outer Cold50 including W0 overlap',
                        ranking=['ROC-AUC','average_precision','positive mean/median rank','conditional-user MRR','candidate Recall@1/5/10/20/50'],
                        fractions=FRACTIONS,within_user=[1,2,5,10],bins=BINS,
                        probability=['mean','observed','ratio','Brier','logloss','ECE','diagnostic intercept/slope','10-bin reliability'],
                        probability_scope='mean/Brier/logloss/ECE on epsilon-clipped probability, additionally save unclipped mean; tails raw probabilities',
                        global_ties='canonical original row order, ascending stable argsort, take last ceil(n*f), identical to P4.2D',
                        user_ties='descending q, ascending B0 rank, ascending article_id',
                        correlation='average-rank Spearman including ties; assert tie partitions as stronger exact test'),
        cross_source=dict(qW='frozen P4.2R3 predictions only',U='clipped_logit(qC)-clipped_logit(qW)',
                          positive_population='264 P4.2D cutoff-user-item positives excluding W0 overlap',
                          thresholds=[0.,0.6931471805599453,1.3862943611198906],
                          metrics=['beneficial-vs-harmful AUC/AP','Pearson/Spearman with single replacement AP delta','positive threshold survival by strict/sparse'],
                          matching=False,new_recommendations=False,new_MAP=False),
        verdict_rules=dict(
            ranking='supported: mean ROC>=old AND >=3 windows ROC>=old AND mean AP>=old; rejected: both means<old AND <=1 windows ROC>=old; otherwise mixed',
            tail='supported: pooled top1% precision>=old AND pooled top0.5% precision>=old AND pooled top0.1% positives>=old AND >=2 metrics nondegrade >=3 windows; rejected if all three pooled metrics degrade; otherwise mixed',
            calibration='supported: >=3 windows improve abs(log(mean/observed)), >=3 logloss nonworse, >=3 Brier nonworse, all b>0, >=3 improve abs(diagnostic slope-1); rejected if any b<=0 or both mean logloss and mean Brier worsen; otherwise mixed',
            readiness='weak iff pooled calibrated tau0 positives>old AND mean pair AUC>=old-.01; supported also >=3 windows tau0 improve and >=1 strict tau0 positive; else rejected',
            admission='ranking!=rejected AND calibration=supported AND readiness in [weak,supported]; even true stop'),
        budget=dict(expected_user_date_observations=n,expected_cold50_rows=n*50,expected_m4_rows=n*200,
                    estimated_disk_gib=[8,25],estimated_peak_RAM_gib=[10,18],estimated_compute_minutes=[30,120],
                    estimate_basis='P4.2D hash10 x10, frozen reconstruction runtimes; not measured full counts',
                    disk_free_gib=shutil.disk_usage(repo).free/2**30,
                    physical_RAM_gib=memory().total/2**30,
                    available_RAM_gib=memory().available/2**30,
                    stop='identity/rank mismatch, nonfinite, raw nonconvergence, disk free<15GiB, insufficient exact-memory budget, formal runtime>2h; W0 fallback; never shrink sample'),
        final_week='not_run',Warm_v2_integrated=False,next_stage_started=False,
        hash_purpose='user requires evidence contract; compare frozen inputs to prior manifests; new output hashes bind later replay, not authenticity alone')
    write_json(dest,c)
    return c
