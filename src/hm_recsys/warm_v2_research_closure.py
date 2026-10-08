"""Read-only experiment closure and explicit evidence packaging; never trains."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

from .warm_v2_contract import ARTIFACT,REPORT,assert_branch,git,guard,now,read,write,evidence_id
from .warm_v2_lab import log


def collect(value,found):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:
            found[(value['path'],value['sha256'])]=value
        for child in value.values():collect(child,found)
    elif isinstance(value,list):
        for child in value:collect(child,found)


def main():
    assert_branch()
    baseline=read(REPORT/'WARM_V1_REPRODUCTION.json')
    result=read(REPORT/'WV2-601_OUTER.json')
    review=read(REPORT/'WV2-601_REVIEW.json')
    replay=read(REPORT/'WV2-601_FUSION_REPLAY.json')
    assert result['decision']=='milestone' and result['stable'] and review['status']==replay['status']=='passed'
    assert result['delta_vs_Warm_v1']>=.0005 and result['non_degrade_windows']>=3 and result['worst_window_delta']>=-.0003
    assert baseline['status']=='baseline_exact_reproduction_passed'
    found={};model_cost=[]
    for path in sorted((ARTIFACT/'bpr-match-v1').glob('*/model.json')):
        m=read(path);guard(m['cutoff'])
        assert m['latest_history_date']<m['cutoff'] and m['pit_safe'] and len(m['epochs'])==100
        assert m['factor_shapes'][0][1]==m['factor_shapes'][1][1]==101
        collect(m,found)
        f=read(path.parent/'features.json');collect(f,found)
        assert f['cutoff']==m['cutoff'] and f['pit_safe'] and not f['future_labels_used']
        model_cost.append({k:m[k] for k in ('cutoff','users','items','binary_pairs','prepare_seconds','fit_seconds','runtime_seconds')})
    assert len(model_cost)==10
    recent=[read(p) for p in sorted((ARTIFACT/'recent-target-v1').glob('*/BUILD.json'))]
    assert len(recent)==4
    for r in recent:assert r['final_week']=='not_run'
    registered=read(REPORT/'WARM_RESEARCH_REOPEN_CONTRACT.json')['registered_at']
    elapsed=(datetime.fromisoformat(now())-datetime.fromisoformat(registered)).total_seconds()
    costs={'created_at':now(),'scope':'current reopened research, not previous plateau; overlapping timers must NOT be added together',
        'registered_reopen_at':registered,'elapsed_wall_since_reopen_seconds_including_research_engineering_waits_closure':elapsed,
        'completed_bpr_models':model_cost,'completed_bpr_model_seconds_sum':sum(v['runtime_seconds'] for v in model_cost),
        'completed_bpr_fit_seconds_sum':sum(v['fit_seconds'] for v in model_cost),
        'recent_train_candidate_builds':[{'cutoff':r.get('cutoff'),'wall_seconds':r['wall_seconds']} for r in recent],
        'recent_train_candidate_build_seconds_sum':sum(r['wall_seconds'] for r in recent),
        'fusion_screen_seconds':read(REPORT/'WV2-601_SCREEN.json')['runtime_seconds'],
        'fusion_formal_seconds':result['runtime_seconds'],
        'bpr_bias_ablation_seconds':read(REPORT/'WV2-502_SCREEN.json')['runtime_seconds'],
        'timing_caveat':'BPR SCREEN includes on-demand builds but excludes pilot; resumed OUTER timer is only last invocation. Per-window timers may include builds. These are different scopes, not additive total CPU time.',
        'local_artifact_bytes':sum(p.stat().st_size for p in ARTIFACT.rglob('*') if p.is_file()),
        'bpr_artifact_bytes':sum(p.stat().st_size for p in (ARTIFACT/'bpr-match-v1').rglob('*') if p.is_file()),
        'current_free_disk_bytes':shutil.disk_usage('.').free,
        'compute':'local CPU8, no shared server; original Item2Vec GPU exact-neighbor computation used ONLY by rejected recent-target trial. BPR/LightGBM/fusion CPU.',
        'memory_scope':'per-fit process lifetime working set recorded in SCREEN/OUTER; no continuously instrumented end-to-end peak',
        'token_usage':'not_available; do not estimate from elapsed time',
        'final_week':'not_run'}
    write(REPORT/'WARM_V2_RESEARCH_COST_AUDIT.json',costs)
    clarifications={'created_at':now(),'scope':'clarifies legacy generic metadata without rewriting measured outputs',
        'WV2-501_OUTER.training_protocol_changed_false':'only original LightGBM loop unchanged; upstream BPR training and two new feature columns WERE added',
        'WV2-601_OUTER.training_protocol_changed_false':'fusion stage fits zero models, but depends on baseline84 and enhanced86 feature rankers plus upstream BPR',
        'WV2-601_registry_feature_count_84':'legacy generic registration field, NOT ensemble layout; use realized_feature_layout and producer_feature_counts',
        'timing':'invocation-only timers after resume are NOT total experiment cost; see WARM_V2_RESEARCH_COST_AUDIT.json',
        'candidate_metrics_in_601':'copied from baseline ONLY after complete candidate identity/target equality; independent MAP reviewer verifies identity and full truth MAP. Fusion replay independently validates every score.',
        'final_week':'not_run'}
    write(REPORT/'WARM_V2_METADATA_CLARIFICATIONS.json',clarifications)
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    assert registry['current_champion']=='WV2-601'
    registry.update(autonomous_state='milestone_complete_waiting_human_integration_review',
        broad_exploration_stopped=True,automatic_push_or_merge=False,
        current_delivery='reports/warm_v2/WARM_V2_RESEARCH_MILESTONE.md')
    for t in registry['trials']:
        if t['experiment_id']=='WV2-501':
            t['realized_feature_layout']='原84列+全历史BPR匹配分数及不可用标记，共86列；LightGBM原训练循环不变，但新增上游矩阵分解训练'
            t['additional_artifact_evidence']=list(found.values())
        elif t['experiment_id']=='WV2-601':
            t['realized_feature_layout']='两个已训练排序器：旧84列、新86列；融合输入仅为两模型在同一用户完整候选池中的名次，等权RRF常量60；本阶段无新拟合'
            t['producer_feature_counts']={'WV2-000':84,'WV2-501':86}
            t['learned_upstream_family']='bpr_match'
            t['independent_review_status']='passed'
    write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry);log()
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    exposure=[t['experiment_id'] for t in registry['trials'] if t.get('outer_confirmations') and t['experiment_id']!='WV2-000']
    assert len(exposure)==5
    ablation=read(REPORT/'WV2-502_SCREEN.json');full=read(REPORT/'WV2-501_SCREEN.json')
    out={'created_at':now(),'status':'provisional_milestone_for_human_review','branch':git('branch','--show-current'),
        'packaging_parent_HEAD':git('rev-parse','HEAD'),'frozen_baseline':'WV2-000','current_champion':'WV2-601',
        'baseline_mean_MAP':baseline['mean_MAP'],'champion_mean_MAP':result['mean_MAP'],
        'delta_mean_MAP':result['delta_vs_Warm_v1'],'relative_gain':result['delta_vs_Warm_v1']/baseline['mean_MAP'],
        'per_window_MAP':result['per_window_MAP'],'per_window_delta':result['per_window_delta'],
        'nondegrade_windows':result['non_degrade_windows'],'worst_delta':result['worst_window_delta'],
        'candidate_pool_changed':False,'evaluation_contract_changed':False,
        'pipeline_changed':True,'changes':['all pre-cutoff user-item BPR matching feature','second 86-feature LambdaRank ranker','equal RRF of two model ranks with fixed60'],
        'ranker_training_loop_changed':False,'upstream_feature_model_training_added':True,
        'new_ranker_feature_columns':['wv2_bpr_user_item_score','wv2_bpr_unavailable'],
        'ablation':{'scope':'four original historical validation windows only, same fitted BPR; no outer ablation',
            'full_bpr_inner_mean_delta':full['screening']['mean_population_delta'],
            'bias_only_inner_mean_delta':ablation['screening']['mean_population_delta'],
            'full_minus_bias_mean_delta':full['screening']['mean_population_delta']-ablation['screening']['mean_population_delta']},
        'review_status':review['status'],'all_candidate_fusion_replay':replay['status'],'mechanics_tests':19,
        'outer_trial_exposure':exposure,'adaptive_selection_risk':True,'final_week':'not_run',
        'integration_recommendation':'human-reviewed integration candidate; replay frozen Cold/Admission compatibility in separate integration branch, never silently replace producer scores',
        'stop_reason':'original milestone gates met; no further search in this turn',
        'automatic_merge_or_push':False,'cost_report':'WARM_V2_RESEARCH_COST_AUDIT.json',
        'historical_plateau_delivery_preserved':'WARM_V2_MILESTONE.md/json and WARM_V2_OUTPUT_MANIFEST.json are previous run snapshots'}
    write(REPORT/'WARM_V2_RESEARCH_MILESTONE.json',out)
    write(REPORT/'WARM_V2_CURRENT_DELIVERY.json',{'current':'WARM_V2_RESEARCH_MILESTONE.md',
        'metrics':'WARM_V2_RESEARCH_MILESTONE.json','manifest':'WARM_V2_RESEARCH_OUTPUT_MANIFEST.json',
        'previous_plateau':'WARM_V2_MILESTONE.md','champion':'WV2-601','final_week':'not_run'})
    # Explicit user evidence-package requirement, NOT a routine authenticity claim.
    # Reuse recorded large-artifact checksums; hash current small code/report files once.
    collect(registry,found)
    manifest_path=REPORT/'WARM_V2_RESEARCH_OUTPUT_MANIFEST.json'
    paths=sorted(Path('src/hm_recsys').glob('warm_v2_*.py'))+sorted(Path('tests').glob('test_warm_v2*.py'))+[Path('dependency.txt')]
    paths+=sorted(p for p in REPORT.iterdir() if p.is_file() and p.suffix in ('.json','.md') and p!=manifest_path and p.name!='WARM_V2_OUTPUT_MANIFEST.json')
    records=[evidence_id(p,reason='explicit_registry_evidence') for p in paths]
    for item in found.values():assert Path(item['path']).stat().st_size==item['bytes']
    write(manifest_path,{'created_at':now(),'branch':out['branch'],'packaging_parent_HEAD':out['packaging_parent_HEAD'],
        'purpose':'explicit user evidence manifest; current hashes establish package identity, not independent authenticity',
        'code_reports_tests':records,'recorded_large_artifact_evidence':list(found.values()),
        'large_artifact_check':'reuse recorded generation checksums, verify existence and byte size; content replay independently passed',
        'current_review':'WV2-601_REVIEW.json','fusion_replay':'WV2-601_FUSION_REPLAY.json',
        'gitignored_artifacts_stay_local':True,'final_week':'not_run'})
    print({'champion':out['current_champion'],'mean':out['champion_mean_MAP'],'gain':out['delta_mean_MAP'],
        'bpr_minutes':costs['completed_bpr_model_seconds_sum']/60,'recent_build_minutes':costs['recent_train_candidate_build_seconds_sum']/60,
        'elapsed_hours':elapsed/3600,'artifact_gib':costs['local_artifact_bytes']/1024**3},flush=True)


if __name__=='__main__':main()
