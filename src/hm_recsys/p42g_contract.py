"""Immutable P4.2G pilot contract; references existing P4.2F assets directly."""
from pathlib import Path
import subprocess
from .p41a_contract import read_json, write_json, identity, check_identity
from .p42_contract import branch_guard
from .p42f_contract import RUN_ID as F_RUN, GLOBAL, DATES, WINDOWS, earlier, now, PARAMS
from .p42f_data import records

RUN_ID = 'p4-2g-v1-risk-separated-hash10'
CLASSIFIER = dict(PARAMS, objective='multiclass', num_class=3)
MAGNITUDE = dict(PARAMS, n_estimators=150, num_leaves=15, max_depth=4, min_child_samples=20)
AUTHORITY = ['P4_2D_FINAL.md','P4_2E_FINAL.md','P4_2F_FINAL.md',
    'P4_2F_EXPERIMENT_CONTRACT.json','P4_2F_OUTPUT_MANIFEST.json',
    'p4_2f_feature_contract.json','p4_2f_action_training_audit.json',
    'p4_2f_global_utility.json','p4_2f_admission_risk.json','p4_2f_segment_metrics.json',
    'P4_2F_metrics.json','P4_2F_VERIFICATION.json','P4_2D_metrics.json',
    'P4_2D_OUTPUT_MANIFEST.json']

def register(repo):
    repo=Path(repo).resolve(); branch_guard(repo)
    dest=repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json'
    if dest.exists(): raise FileExistsError('immutable contract already exists')
    f=read_json(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json')
    feature=read_json(repo/'reports/phase4/p4_2f_feature_contract.json')
    manifest=read_json(repo/'reports/phase4/P4_2F_OUTPUT_MANIFEST.json')
    assert feature['G']==f['features']['G']==GLOBAL
    assert read_json(repo/'reports/phase4/P4_2F_VERIFICATION.json')['status']=='pass'
    known={str(Path(r['path']).resolve()).lower():r for r in records(manifest)}
    root=repo/'artifacts/phase4'/F_RUN
    wanted=list((root/'prepared').glob('*/data.joblib'))+list((root/'prepared').glob('*/training.npz'))
    wanted+=list((root/'outer').glob('*/G_expected_deltaAP-utility.npy'))+list((root/'outer').glob('*/single_delta.npy'))
    wanted+=list((root/'outer').glob('*/EVALUATION.json'))
    wanted+=list((repo/'src/hm_recsys').glob('p42f*.py'))
    assets=[]
    for p in wanted:
        r=known[str(p.resolve()).lower()]; check_identity(r); assets.append(r)
    assert len(list((root/'prepared').glob('*/data.joblib')))==9
    assert len(list((root/'prepared').glob('*/training.npz')))==8
    authorities=[repo/'reports/phase4'/n for n in AUTHORITY]
    authorities += [repo/'docs/ROADMAP_PHASE4.zh-CN.md',(repo / 'docs/research-contracts/private-approval.txt')]
    c=dict(stage='P4.2G',run_id=RUN_ID,status='preregistered_before_formal_computation',created_at_utc=now(),
        git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        authority=[identity(p) for p in authorities],trusted_F_assets=assets,
        historical_cutoffs=DATES,outer_windows=WINDOWS,historical_pools={w:earlier(t) for w,t in WINDOWS.items()},
        reuse_root=str(root),features=GLOBAL,feature_contract=feature,neutral=f['neutral'],
        action_space=f['overlap'],population=f['population'],outer_population=f['outer_population'],
        class_labels={'B':0,'N':1,'H':2},class_definition='B iff exact y>0; N iff y==0; H iff y<0; no tolerance',
        classifier_params=CLASSIFIER,benefit_params=MAGNITUDE,harm_params=MAGNITUDE,
        magnitude='B uses y>0, target y; H uses y<0, target abs(y); all weights1; clip predictions[0,1]',
        utility='p_B*clip(m_B_raw,0,1)-p_H*clip(m_H_raw,0,1)',lambda_value=1.,gate='U_R>0',
        matching=f['matching'],formal_variants=['W0','R_risk_separated'],
        diagnostic='U_C=p_B-p_H; action ROC/AP and Pearson/Spearman only; no matching or MAP',
        probability_audit='unweighted complete outer edges; 10 fixed equal-width bins [0,.1),...,[.9,1]; logloss clip true-class p at float64 epsilon; Brier mean sum of 3 squared errors',
        metrics=['BH ROC-AUC/AP','B-vs-rest ROC-AUC/AP','Pearson/Spearman all edges','class rates and calibration',
                 'positive candidate survival','gate B/N/H','exact overall/segment MAP','admission counts and buckets','F novelty/richness buckets'],
        verdicts=dict(action_supported='mean BH ROC>G, >=3/4 ROC nondegrade, mean BH PR>=G',
            action_mixed='not supported, but mean BH ROC>G OR mean BH PR>G; otherwise rejected',
            precision_supported='pooled insert/remove>=.25 AND ratio>=3*(12/257) AND removed<257',
            precision_mixed='not supported but ratio>12/257 AND removed<257; otherwise rejected',
            zero_denominator='removed0 inserted>0: infinite ratio, explicit status; both0 undefined and precision rejected',
            safety_supported='mean overall delta>=-.0005 AND worst>=-.001 AND mean warm delta>=-.0005',
            safety_mixed='not supported but overall mean, worst window and warm mean all strictly better than G; otherwise rejected',
            scaleup='action!=rejected AND precision==supported AND safety in[supported,mixed]; readiness only, no execution authority'),
        promotion='pilot only: retain W0; no automatic baseline promotion',
        budget=dict(estimated_compute_minutes=[20,45],formal_max_seconds=3600,threads=4,min_free_ram_gib=6,
                    process_peak_gib=6,min_free_disk_gib=15,stop='parity violation, nonfinite output, missing class, resource/time limit; preserve failure and W0; no retries/tuning/downscale'),
        final_week='not_run',Warm_v2_integrated=False,P4_3_started=False,full_history_run=False,
        hashes='explicit user contract: reused F assets checked against previous manifest; new contract binds implementation before fits')
    write_json(dest,c)
    return c
