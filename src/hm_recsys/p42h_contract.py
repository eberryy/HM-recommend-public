"""Immutable conditional B/H pilot; all choices registered before any new fit."""
from pathlib import Path
import subprocess
from .p41a_contract import read_json, write_json, identity
from .p42_contract import branch_guard
from .p42f_contract import GLOBAL, DATES, WINDOWS, earlier, now, PARAMS
from .p42f_data import records
from .p42g_contract import RUN_ID as G_RUN, MAGNITUDE

RUN_ID = 'p4-2h-v1-conditional-bh-hash10-user-oof'
Q_PARAMS = dict(PARAMS, objective='binary', min_child_samples=100)
R_PARAMS = dict(PARAMS, objective='binary')
PLATT_PARAMS = dict(penalty=None, solver='lbfgs', tol=1e-8, max_iter=1000, fit_intercept=True)
AUTHORITY = ['P4_2D_FINAL.md','P4_2E_FINAL.md','P4_2F_FINAL.md','P4_2G_FINAL.md',
    'P4_2F_EXPERIMENT_CONTRACT.json','P4_2G_EXPERIMENT_CONTRACT.json',
    'P4_2F_OUTPUT_MANIFEST.json','P4_2G_OUTPUT_MANIFEST.json','p4_2f_feature_contract.json',
    'p4_2g_data_parity.json','p4_2g_multiclass_risk.json','p4_2g_benefit_magnitude.json',
    'p4_2g_harm_magnitude.json','p4_2g_action_metrics.json','p4_2g_gate_tail.json',
    'p4_2g_admission_risk.json','P4_2G_metrics.json','P4_2G_VERIFICATION.json',
    'p4_2f_global_utility.json','p4_2f_action_training_audit.json']

def register(repo):
    repo=Path(repo).resolve(); branch_guard(repo)
    dest=repo/'reports/phase4/P4_2H_EXPERIMENT_CONTRACT.json'
    if dest.exists(): raise FileExistsError('immutable contract exists')
    g=read_json(repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json')
    assert read_json(repo/'reports/phase4/P4_2G_VERIFICATION.json')['status']=='pass'
    feature=read_json(repo/'reports/phase4/p4_2f_feature_contract.json')
    assert g['features']==feature['G']==GLOBAL and len(GLOBAL)==70
    groot=repo/'artifacts/phase4'/G_RUN
    known={str(Path(r['path']).resolve()).lower():r for r in records(read_json(repo/'reports/phase4/P4_2G_OUTPUT_MANIFEST.json'))}
    wanted=[]
    for w in WINDOWS:
        for role in ['benefit','harm']:
            wanted.extend(groot/'models'/w/role/n for n in ['model.txt','FIT_START.json','FIT_RESULT.json','training-pool-row-indices.npy'])
        wanted.extend(groot/'outer'/w/n for n in ['m_B_raw.npy','m_H_raw.npy','m_B.npy','m_H.npy','U_R.npy','EVALUATION.json'])
    assets=[known[str(p.resolve()).lower()] for p in wanted]
    authorities=[repo/'reports/phase4'/n for n in AUTHORITY]+[repo/'docs/ROADMAP_PHASE4.zh-CN.md',
        (repo / 'docs/research-contracts/private-approval.txt')]
    c=dict(stage='P4.2H',run_id=RUN_ID,status='preregistered_before_formal_computation',created_at_utc=now(),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        authority=[identity(p) for p in authorities],trusted_F_assets=g['trusted_F_assets'],trusted_G_assets=assets,
        reuse_root=g['reuse_root'],g_reuse_root=str(groot),historical_cutoffs=DATES,outer_windows=WINDOWS,
        historical_pools={w:earlier(t) for w,t in WINDOWS.items()},features=GLOBAL,feature_contract=feature,
        feature_identity=identity(repo/'reports/phase4/p4_2f_feature_contract.json'),
        population=g['population'],outer_population=g['outer_population'],action_space=g['action_space'],neutral=g['neutral'],
        labels='B iff exact delta_AP_single>0; N iff ==0; H iff <0; no tolerance',
        q_target='all retained B/H only, B=1 H=0; unweighted, N=0 rows; no class_weight or sampling',
        r_target='B or H=1, N=0; exact F/G sampled rows; B/H weight1 sampled N weight50',
        q_params=Q_PARAMS,r_params=R_PARAMS,benefit_params=MAGNITUDE,harm_params=MAGNITUDE,
        magnitude='exact reuse of saved G outer-specific models; zero new magnitude fits; clip[0,1]; full outer prediction parity',
        calibration=dict(params=PLATT_PARAMS,epsilon=1e-6,input='logit(clip(q_raw,1e-6,1-1e-6))',
            formula='q_cal=expit(a+b*input)',OOF='sha256 UTF8 customer_id first8 big-endian mod2; same user all dates same fold',
            roles='q_fold0 trains fold0 predicts fold1; q_fold1 trains fold1 predicts fold0; every BH row exactly once',
            timeline='user-isolated OOF inside historical pool; NOT historical-date forward OOF; all label_end<outer',
            full='q_full on all historical B/H, then use historical OOF calibrator on outer full-q probabilities',
            weights='unweighted logistic, no regularization, no class weights',
            stop='any convergence warning or b<=0 stops entire run; preserve partial evidence, no formal affected outer, no retry'),
        utility='G_BH=q_cal*m_B-(1-q_cal)*m_H; U_H=r*G_BH',gate='G_BH>0',
        implied_threshold='m_H/(m_B+m_H); zero denominator=>1, G_BH=0, reject',
        numerical='exact boolean G_BH>0 == q_cal>threshold; eligible r>0 and U_H>0; violation=>engineering_failure, no epsilon rescue',
        matching=dict(g['matching'],eligibility='G_BH>0; equality rejected',weight='r*G_BH; mask ineligible to0 explicitly'),
        formal_variants=['W0','H_conditional_BH'],primary_action_score='U_H; G_BH fully reported but not alternative gate selection',
        diagnostics=dict(D1='raw-q gain: BH ROC/AP, positive survival, eligible B/H only; no matching/MAP',
            D2='q_cal: BH ROC/AP only',D3='G_BH vs U_H correlations all/eligible; same-user eligible top1/top5 overlap, no second matching/MAP',
            overlap='intersection/min(K,eligible edge count), mean over users with eligible edges; ties Cold canonical row then Warm slot',
            q_outer='unweighted outer B/H only: mean, Brier, logloss, ECE; diagnostic intercept/slope cannot enter policy',
            diagnostic_logistic='same unweighted PLATT_PARAMS; nonconvergence recorded not retried; b need not positive for diagnostic',
            probability_loss_clip='float64 machine epsilon for logloss; probability mean/Brier/ECE unmodified',
            ECE='10 equal-width probability bins; weight by BH row counts; raw and calibrated own probability bins',
            quantiles=[0,.5,.8,.9,.95,.99,1],quantile_ties='raw-q value quantiles np.quantile linear; shared raw/cal rows, ties lower bucket',
            distributions='all/B/H/N m_B,m_H,m_B/m_H on m_H>0 with undefined counts; q,threshold,gap; observed units action edges',
            user_groups='exact F historical novelty quartile and richness tertile cutpoints; no reslicing/router'),
        verdicts=dict(action_supported='U_H mean BH ROC>=G_R, >=3/4 ROC nondegrade, mean BH PR>=G_R',
            action_mixed='not supported but mean ROC>G_R OR mean PR>G_R; otherwise rejected',
            survival_supported='pooled>=8 AND >G_R in>=3/4 windows AND strict surviving>=1',
            survival_mixed='not supported but pooled>=2; at least twice reference1, descriptive not statistically significant',
            precision_supported='insert/remove>=.25 AND inserted>=3 AND removed<=20',
            precision_mixed='not supported but insert/remove>=.15 AND >1/15; otherwise rejected',
            zero_denominator='removed0 inserted>0=infinite; both0=undefined and rejected',
            safety_supported='mean overall delta>=-.0005 AND worst>=-.001 AND mean warm delta>=-.0005',
            safety_mixed='not supported but mean overall, worst, mean warm all strictly better than F_G',
            scaleup='action!=rejected AND survival in[supported,mixed] AND precision in[supported,mixed] AND safety==supported; stop even true'),
        budget=dict(estimated_compute_minutes=[20,40],formal_max_seconds=3600,threads=4,min_free_ram_gib=6,
            process_peak_gib=6,min_free_disk_gib=15,LGBM_fits=16,OOF_Platt_fits=4,magnitude_fits=0,
            outer_diagnostic_logistic_fits=8,stop='resource/time/invariant failure; preserve evidence+W0; no tuning/retry/downscale'),
        final_week='not_run',Warm_v2_integrated=False,P4_3_started=False,full_history_run=False,selected='W0',
        hashes='explicit user evidence contract: compare cached F/G assets to prior manifests at preflight; freeze implementation before fits; verify at completion')
    write_json(dest,c);return c
