"""Bounded diagnostic preregistration; old stages are never written."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import subprocess

from .p41a_contract import identity, read_json, write_json
from .p42_contract import WINDOWS, TAUS, branch_guard

RUN_ID = 'p4-2d-v1-full-positive-selected-tail-audit'
FRACTIONS = [.01,.005,.001,.0005,.0001]
BINS = [0,.9,.99,.995,.999,.9995,.9999,1]
SCORES = ['negative_b0_rank','b0_user_percentile','b0_user_zscore',
          'normalized_margin_to_rank5','normalized_margin_to_user_median','qC']


def preregister(repo):
    repo = Path(repo).resolve()
    branch_guard(repo)
    report = repo/'reports/phase4'
    target = report/'P4_2D_EXPERIMENT_CONTRACT.json'
    if target.exists():
        return read_json(target)
    required = ['docs/ROADMAP_PHASE4.zh-CN.md'] + ['reports/phase4/'+n for n in (
        'P4_1A_FINAL.md','P4_2_FINAL.md','P4_2R_FINAL.md','P4_2R2_FINAL.md','P4_2R3_FINAL.md',
        'P4_2R3_metrics.json','P4_2R3_EXPERIMENT_CONTRACT.json','P4_2R3_OUTPUT_MANIFEST.json',
        'p4_2r3_propensity_calibration.json','p4_2r3_pair_utility_audit.json','p4_2r3_matching_audit.json',
        'p4_2r3_admission_risk.json','p4_2r3_segment_metrics.json')]
    authority = {}
    for name in required:
        p = repo/name
        content = p.read_text(encoding='utf-8')
        if p.suffix == '.json':
            read_json(p)
        authority[name] = {**identity(p), 'lines': len(content.splitlines())}
    source_review = {}
    for p in sorted((repo/'src/hm_recsys').glob('p42*.py')):
        if p.name.startswith('p42d'):
            continue
        code = p.read_text(encoding='utf-8')
        tree = ast.parse(code)
        source_review[p.name] = {**identity(p), 'module_doc': ast.get_docstring(tree),
            'functions': [{'name': n.name, 'line': n.lineno, 'doc': ast.get_docstring(n)}
                          for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]}
    c = dict(stage='P4.2D', run_id=RUN_ID, created_at_utc=datetime.now(timezone.utc).isoformat(),
        status='preregistered_before_computation', branch='main',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        authority=authority, source_review=source_review, windows=WINDOWS, taus=TAUS,
        no_training=True,no_refit=True,no_optimizer=True,no_candidate_regeneration=True,no_threshold_search=True,
        no_matching_calls=True,only_saved_original_matches=True,final_week='not_run',
        candidate_unit='unique cutoff,user,cold article; complete Cold50 minus W0 overlap for primary positives',
        calibration_population='full original Cold50 before overlap exclusion',
        edge_unit='eligible Cold candidate x original Warm slot; positive-negative=beneficial; negative-positive=harmful; rest neutral',
        rank_policy='qC raw probability descending; ties original B0 rank then article_id; within complete50 BEFORE overlap filter; rank_pct=rank/50; score percentile average ascending(q)-1 divided49',
        b0_policy='exact original b0_rank; no new inference; native summaries copied from frozen full50 normalization',
        candidate_recall_denominator='all primary positive candidates in Cold50 minus overlap',
        conditional_choice='incidence-positive users on full50 incl overlap; macro multi-positive recall; MRR reciprocal first truth rank; additionally pooled candidate recall',
        qc_global_top_fractions=FRACTIONS,qc_within_user_top=[1,2,5,10],qc_quantile_bins=BINS,
        quantile_policy='ascending raw qC stable canonical row order; bins integer floor(n*f) half-open, final endpoint n; top tail ceil(n*f)',
        warm_bottom_fractions=[.01,.05,.1,.2],score_directions=SCORES,
        ap='single in-place replacement; original full truth denominator min(n_truth,12); no new formal MAP',
        pair_scores=['U','D1=qC-qW','D2=qC-2*qW'],
        diagnostic_alternative_probability='same frozen clipped probabilities as original U; no matching or policy selection',
        matching_loss='above-threshold beneficial unselected edges: mutually exclusive same-Cold-only, same-Warm-only, both, neither; extra overlapping higher-weight-nonbeneficial-node flag; candidate lost only if no beneficial selected; not solver defect',
        availability={'exclusive_end':'2020-09-16','reason':'final-week embargo overrides full-future proxy; censoring explicit',
            'buckets':'0<=days<7;7<=days<28;28<=days<84;days>=84;not_observed_before_embargo',
            'populations':['all strict Cold50','global qC top1%,0.5%,0.1%,0.05%,0.01% intersect strict','eligible strict maxU>0','strict truth'],
            'not_inventory_ground_truth':True,'no_feature_use':True},
        scaleup='read8 frozen historical tables; exact10% users/positive users/rows; 10x expansion estimates only, no100% candidate generation; users uncertainty and hash clustering stated',
        verdict_rules={
            'qc_tail_reliability':'supported if >=3 windows top0.1% and0.01% observed>=meanq/2 and lift>1; rejected if >=3 fail this; otherwise weak (descriptive, sparse counts)',
            'qc_preserves_b0_positive_order':'supported if >=3windows mean positive rank delta<=0; rejected if >=3 >0; else mixed',
            'qw_removal_risk_model':'supported if >=3windows AUC>.55 and bottom10% rate<full; rejected if <=1; else weak; not safety guarantee',
            'logodds_pair_utility_alignment':'supported if >=3windows primary AUC>.55 and Pearson>0; rejected if >=3 AUC<=.5 or Pearson<=0; otherwise weak',
            'matching_is_primary_bottleneck':'true if >half all primary positives survive beneficial tau0 but most survivors not beneficial-selected; false if <=half survive; otherwise inconclusive',
            'user_incidence_factorization_hypothesis':'supported if fixed max_qC AUC>.55 in>=3windows AND B0 conditional MRR>qC in>=3; rejected if max_qC AUC<=.5 in>=3 or B0 MRR<=qC in>=3; otherwise weak',
            'conditional_b0_choice_hypothesis':'supported if B0 macroMRR>qC in>=3; rejected if <= in>=3; else weak',
            'strict_cold_availability_asymmetry':'supported descriptive if >=3windows less than half strict candidate rows first sell in next7d; otherwise inconclusive; never inventory proof',
            'full_history_training_scaleup':'optional if conditional choice or incidence hypothesis supported; otherwise not_justified; recommended not awarded without exact positive-support evidence'},
        budget={'CPU_only':True,'soft_seconds':1800,'estimate_minutes':[10,30],'max_peak_target_gib':6,
                'stop':'input mismatch, forbidden call, memory/time limit; retain W0 and partial failure evidence, no model fallback'})
    write_json(target,c)
    return c
