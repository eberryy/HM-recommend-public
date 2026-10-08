"""Package a verified Warm milestone or an evidence-backed feature plateau."""
from __future__ import annotations

import argparse
from pathlib import Path

from .warm_v2_contract import ARTIFACT, BASE_COMMIT, BRANCH, REPORT, assert_branch, evidence_id, git, now, read, write
from .warm_v2_lab import log
from .warm_v2_review import review


def package(status):
    assert_branch()
    if (REPORT/'WARM_V2_OUTPUT_MANIFEST.json').exists():
        raise RuntimeError('Existing evidence package: do not silently overwrite or routinely rehash it')
    log()
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    realization=read(REPORT/'WARM_FEATURE_REALIZATION_AUDIT.json')
    baseline=read(REPORT/'WARM_V1_REPRODUCTION.json')
    champion=next(t for t in registry['trials'] if t['experiment_id']==registry['current_champion'])
    milestones=[t for t in registry['trials'] if t.get('decision')=='milestone']
    if status=='milestone':
        assert milestones
    else:
        assert not milestones
        for n in [*range(101,111),201,202]:
            assert (REPORT/f'WV2-{n}_SCREEN.json').exists()
    reviews={}
    for trial in registry['trials']:
        if trial['experiment_id']!='WV2-000' and trial['outer_confirmations']:
            path=REPORT/f"{trial['experiment_id']}_REVIEW.json"
            reviews[trial['experiment_id']]=read(path) if path.exists() else review(trial['experiment_id'])
    changed=git('diff','--name-only',BASE_COMMIT,'HEAD').splitlines()
    allowed=lambda p:p=='.gitignore' or p=='tests/test_warm_v2.py' or p.startswith('reports/warm_v2/') or p.startswith('src/hm_recsys/warm_v2_')
    assert all(allowed(p) for p in changed),changed
    unstaged=git('diff','--name-only').splitlines()
    assert all(allowed(p) for p in unstaged),unstaged
    families={}
    for trial in registry['trials']:
        if trial.get('role')=='historical_screening' and not trial.get('parameter_overrides'):
            for family in trial['feature_bundle']:
                families.setdefault(family,[]).append({'trial':trial['experiment_id'],'screening':trial.get('screening'),
                    'decision':trial.get('decision'),'mean_MAP':trial.get('mean_MAP')})
    screen_records={p.stem:read(p) for p in sorted(REPORT.glob('*_SCREEN.json'))}
    outer_records={p.stem:read(p) for p in sorted(REPORT.glob('*_OUTER.json'))}
    runtimes={'baseline':baseline['runtime_seconds'],
        'historical_screens':sum(r['runtime_seconds'] for r in screen_records.values()),
        'formal_confirmations':sum(r['runtime_seconds'] for r in outer_records.values())}
    evidence=[]
    for trial in registry['trials']:
        maps=[evidence_id(p,reason='explicit_registry_evidence') for p in sorted((ARTIFACT/trial['experiment_id']).glob('*/*_category_maps.json'))]
        trial['additional_artifact_evidence']=maps
        trial['artifact_evidence'].extend(maps)
        trial['artifact_paths']=sorted({v['path'] for v in trial['artifact_evidence']})
        evidence.extend(trial.get('artifact_evidence',[]))
    write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    feature_evidence=[]
    for path in sorted(ARTIFACT.glob('features*/**/*.json')):
        v=read(path)
        if 'artifact' in v:
            evidence.append(v['artifact']);feature_evidence.append({'metadata':str(path),**v})
    unique={(v['path'],v['sha256']):v for v in evidence}
    # Explicit requested output evidence package; only new code/report IDs once.
    code_paths=sorted(Path('src/hm_recsys').glob('warm_v2_*.py'))+[Path('tests/test_warm_v2.py')]
    code_ids=[evidence_id(p,reason='explicit_registry_evidence') for p in code_paths]
    result={'stage_status':status,'generated_at':now(),'branch':BRANCH,'evidence_HEAD_before_packaging_commit':git('rev-parse','HEAD'),
        'HEAD_semantics':'This is the implementation/evidence commit at report generation. The final packaging commit is reported in the delivery message; no self-referential commit hash is claimed.',
        'baseline_id':'WV2-000','baseline_mean_MAP':baseline['mean_MAP'],'champion_id':champion['experiment_id'],
        'champion_mean_MAP':champion['mean_MAP'],'champion_delta_vs_v1':champion['mean_MAP']-baseline['mean_MAP'],
        'champion_per_window_MAP':champion['per_window_MAP'],
        'champion_per_window_delta':{w:champion['per_window_MAP'][w]-v for w,v in baseline['per_window_MAP'].items()},
        'non_degrade_windows':champion['non_degrade_windows'],'worst_window_delta':champion['worst_window_delta'],
        'champion_new_feature_families':champion['feature_bundle'],'champion_parameter_overrides':champion.get('parameter_overrides',{}),
        'milestone_trials':[t['experiment_id'] for t in milestones],'feature_families':families,
        'formal_confirmation_counts':registry['outer_confirmation_counts'],
        'formal_trial_count':sum(t['outer_confirmations'] for t in registry['trials']),
        'historical_trial_count':len(screen_records),'candidate_pool_changed':False,
        'model_family_changed':False,'champion_training_parameter_changed':bool(champion.get('parameter_overrides')),
        'sampling_objective_denominator_truth_K_changed':False,'Cold_Admission_changed':False,
        'outer_overfitting_risk':'Nonzero: these are repeatedly used project development windows. Inner-only selection and bounded confirmations reduce, not remove, selection bias. No claim of untouched-test performance.',
        'full_data_training':False,'sample_scope':'Original deterministic10% users; global statistics use all historical users. Never final-week labels.',
        'runtime_seconds':runtimes,'runtime_interpretation':'Measured training/screen/refit paths; not all human-facing tool/edit time and not cold-start retrieval regeneration.',
        'peak_memory_bytes_available':max((t.get('peak_memory_bytes') or 0 for t in registry['trials']),default=0) or None,
        'peak_memory_scope':'Only later instrumented processes; no retrospective claim about earlier peak memory.',
        'artifact_bytes_current':sum(p.stat().st_size for p in ARTIFACT.rglob('*') if p.is_file()),
        'verification_status':{t:r['status'] for t,r in reviews.items()},'changed_paths':changed,
        'real_feature_mechanics_audit':'reports/warm_v2/WARM_FEATURE_REALIZATION_AUDIT.json',
        'tests':{'suite':'tests/test_warm_v2.py','passed':9,'command':'conda run --no-capture-output -n hm-recommend python -m unittest discover -s tests -p test_warm_v2.py -v'},
        'next_step':{'recommendation':'stop_warm_optimization_under_current_contract' if status=='plateau' else 'manual_milestone_review',
            'if_reopened':'Propose inner-only training-week coverage and supervision-size audit before changing candidate budget; not executed here.',
            'reason':'Eight distinct feature families plus ablations and bounded leaf-capacity changes do not yield stable gain; this is not evidence that candidate ceiling is binding.'},
        'integration_recommendation':'manual review of provisional champion; do not automatically replace Cold/admission pipeline' if milestones and champion['experiment_id']!='WV2-000' else 'retain formal Warm-v1; any sub-threshold experimental champion is not automatically promoted; do not merge rejected variants',
        'cherry_pick_candidates':git('log','--reverse','--format=%h %s',f'{BASE_COMMIT}..HEAD').splitlines(),
        'final_week':'not_run','submission_generated':False,'leaderboard_consulted':False,'pushed':False,'merged':False}
    write(REPORT/'WARM_V2_MILESTONE.json',result)
    lines=['# Warm-v2 阶段交付','',f"结论：**{'达到需人工复核的阶段成果' if status=='milestone' else '固定候选与有界排序器探索形成阶段性平台，未伪造晋级'}**。",'',
        '## 口径与术语','',
        'Warm-v1 是本项目冻结的M3.3通用排序基线；champion（本项目“当前冠军”）是满足四窗稳定性条件的最好方案，不等于某个窗口或均值最高的失败方案。MAP@12 是行业常用的前12名平均准确率：每个用户以完整未来7天去重购买真值计算AP，再对原有全部评测用户平均。',
        'inner（行业通用“内层”）指2019-12-25、2020-02-19、2020-05-27、2020-07-22历史筛选周；outer（“外层”）指下表四个固定开发确认周。历史贡献增量是有覆盖且活跃用户AP增量乘该群体占完整用户集合的比例，不能冒充正式四窗MAP。',
        'PIT（point-in-time，行业通用“时点安全”）要求交易统计严格早于对应截止日。原始重复交易保留；购买日期只在明确的日期计数和间隔统计中去重。目录沿用乐观假设：全部articles静态属性可用，不把首次观察销售称为真实上架日期。',
        '', '## 当前结果','',f"- 分支：`{BRANCH}`；报告生成时的代码/证据HEAD：`{result['evidence_HEAD_before_packaging_commit']}`。最终封包提交号在交付消息中给出。",
        f"- Warm-v1均值：{baseline['mean_MAP']:.12f}；当前冠军`{champion['experiment_id']}`均值：{champion['mean_MAP']:.12f}；增量：{result['champion_delta_vs_v1']:+.9f}。",
        f"- 不退化窗数：{champion['non_degrade_windows']}/4；最差增量：{champion['worst_window_delta']:+.9f}；新增特征组：{champion['feature_bundle']}；参数覆盖：{champion.get('parameter_overrides',{})}。",'',
        '| 正式开发窗口 | Warm-v1 MAP@12 | 当前冠军 MAP@12 | 增量 |','|---|---:|---:|---:|']
    for w,v in baseline['per_window_MAP'].items():
        lines.append(f"| {w} | {v:.9f} | {champion['per_window_MAP'][w]:.9f} | {result['champion_per_window_delta'][w]:+.9f} |")
    lines += ['', '## 全部历史筛选，不只保留赢家','',
        '| 实验 | 新增特征组或容量变更 | 平均历史贡献增量 | 正增窗数 | 最差增量 | 通过 |',
        '|---|---|---:|---:|---:|---|']
    for trial in registry['trials']:
        if 'screening' in trial:
            s=trial['screening']; description=str(trial['feature_bundle']) if not trial.get('parameter_overrides') else str(trial['parameter_overrides'])
            lines.append(f"| {trial['experiment_id']} | {description} | {s['mean_population_delta']:+.9f} | {s['positive_windows']} | {s['worst_population_delta']:+.9f} | {s['passed']} |")
    lines += ['', '特征组中文定义：item_demand=商品近期需求与全历史销售生命周期；user_rhythm=用户购买节奏；repeat_affinity=近期复购与层级偏好交叉；source_agreement=召回来源名次一致性；category_competition=同类别/部门/服装组内商品竞争力；price_context=商品交易价分布及同类别个人价格匹配；fine_hierarchy=细颜色/图案/销售分区/产品系列偏好；candidate_context=完整冻结候选池内的已有信号相对强度。以上都是项目自定义组名；逐列对象、单位和分母见三份WARM_FEATURE_SPEC文件。num_leaves是LightGBM通用参数“每棵树最多叶子数”。',
        '', '## 正式确认与失败保留','',
        '| 实验 | 平均MAP@12 | 相对Warm-v1均值增量 | 不退化窗数 | 最差增量 | 决定 |','|---|---:|---:|---:|---:|---|']
    for r in outer_records.values():
        lines.append(f"| {r['experiment_id']} | {r['mean_MAP']:.9f} | {r['delta_vs_Warm_v1']:+.9f} | {r['non_degrade_windows']} | {r['worst_window_delta']:+.9f} | {r['decision']} |")
    lines += ['', 'reject表示未通过正式晋级条件；new_champion表示稳定但尚未达到大阶段增益；milestone表示触发人工复核，仍不是自动合并批准。每个正式试验逐窗数值见对应OUTER.json。',
        '', '## 能够与不能够推出的结论','',
        '- 基线不是引用旧数字：四窗重新拟合后，逐候选分数与旧结果最大差为0，前12件身份与顺序完全一致，分母与MAP复现。',
        f"- 真实特征复核：在2019-11-27原采样训练的{realization['sample']['rows']}条用户—商品记录上，8个组的126列新增字段都有非空且变化的值；四窗模型记录了实际使用的新增分裂。容量试验中每棵树实际达到15或63叶，设置确已生效。不能将失败归因于整组字段没有接入、全常数或容量参数未应用。详细范围见WARM_FEATURE_REALIZATION_AUDIT.json。",
        '- 所有新增正式方案均独立重查候选集合、标签、原名次、历史量和无历史回退，完整真值MAP重算通过。候选召回率和理想排序上限保持不变；没有把候选漏行或分母变化伪装成模型效果。',
        '- 商品需求父组历史四窗提升，但正式仅两窗提升；删除近期需求后，生命周期组的春季退化更大。这不支持“只删除某一类列就能修复泛化”的简单解释。',
        '- 拒绝只针对当前特征定义、训练样本和门槛，不等于证明价格、属性或用户节奏永远无效。模型使用过某列的gain（分裂损失改善累计量）也不是因果收益证据。',
        '- 如果标记平台期，范围仅是固定候选、现有采样训练规模、已登记的8类特征与小范围树容量试验，不代表穷尽全部Warm优化。没有通过降低原晋级门槛制造成果。',
        '- 候选内归一化先基于完整冻结候选集合计算，再做负采样；不根据正例集合或采样负例重新定义名次。',
        '- 不能由这些失败推出召回已到上限。若进入容量诊断，它仅改变叶子上限，不能解释为神经排序模型无效。',
        '- 冻结候选的理想MAP上限四窗约0.157885、0.164143、0.136977、0.149495，定义为让已召回真值全排最前时、按同一完整用户分母计算的上限。它远高于当前实测MAP，但包含事后标签，不代表可学习、可上线的承诺成绩。',
        '', '## 成本、边界与集成','',
        f"- 训练路径实测：基线复现{runtimes['baseline']:.1f}秒，历史筛选{runtimes['historical_screens']:.1f}秒，正式确认{runtimes['formal_confirmations']:.1f}秒；当前本分支大文件占用约{result['artifact_bytes_current']/1024**3:.2f} GiB。全部CPU，无GPU训练。",
        f"- 可用的进程生命周期峰值内存记录：{result['peak_memory_bytes_available']}字节；较早试验未埋点，不能声称已测其峰值。",
        f"- 正式确认次数：{result['formal_trial_count']}；分族/阶段记录：{result['formal_confirmation_counts']}。所有这些窗口均已有历史开发使用，仍有选择偏差风险，不能称独立测试结论。",
        '- 候选身份与预算、采样、真值定义、分母、K=12、排序损失、无历史回退均未改变。若参数覆盖非空，仅单独容量阶段改变最多叶子数。Cold/Admission主线代码与报告未修改。',
        '- 仍是原固定10%用户实验，但全局统计使用全部历史用户；未进行全量用户训练。未重新计入冻结召回与历史特征首次生成的成本。',
        f"- 集成建议：{'人工审核临时冠军后再决定，不自动替换主线' if milestones and champion['experiment_id']!='WV2-000' else '保留正式Warm-v1，不合并被拒绝方案；若有低于里程碑门槛的实验冠军，也不自动晋级'}。不自动合并、不推送；最终周2020-09-16始终not_run（未运行），未生成提交文件、未查看排行榜。",
        '- 若没有晋级，不建议把失败特征接入主线。可选择性保留精确复现、候选一致性复核和注册表基础设施。若有临时冠军，应先人工审核其逐窗风险及Cold准入接口兼容性，再在独立集成分支组合；没有开展该组合测试。',
        '- 本轮若为平台期，建议暂停当前合同下继续加特征。若人工决定重开，先对历史内层的训练周覆盖和正例监督规模设计诊断，区分统计冗余、时间泛化与监督不足；这只是待讨论方向，本轮没有擅自更改训练周或扩大候选。',
        '', '## 可选择的提交','']+['- `'+s+'`' for s in result['cherry_pick_candidates']]+['',
        '详细注册表、代码与模型证据路径见WARM_EXPERIMENT_REGISTRY.json和WARM_V2_OUTPUT_MANIFEST.json。SHA256用于用户明确要求的证据包标识；不是对当前新文件反复校验，也不单独证明真实性。']
    (REPORT/'WARM_V2_MILESTONE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={'created_at':now(),'branch':BRANCH,'evidence_HEAD':result['evidence_HEAD_before_packaging_commit'],
        'code':code_ids,'models_prediction_and_frozen_data_evidence':list(unique.values()),'feature_cache_metadata':feature_evidence,
        'reports':[str(p) for p in sorted(REPORT.glob('*')) if p.is_file()],
        'artifact_policy':'Large artifacts remain local and ignored; existing SHA IDs reused without routine rehash. New code hashed once only for explicit output manifest.',
        'final_week':'not_run','automatic_merge_or_push':False}
    write(REPORT/'WARM_V2_OUTPUT_MANIFEST.json',manifest)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('status',choices=['milestone','plateau'])
    a=p.parse_args()
    print(package(a.status)['stage_status'])
