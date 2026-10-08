"""Package frozen robustness evidence and a conditional, non-executing integration plan."""
from pathlib import Path
from datetime import datetime
import shutil
from .warm_v2_contract import read,write,now,git,evidence_id
from .warm_v2_fresh import ART,CONTRACT,setup
from .warm_v2_research_closure import collect

REPORT=Path('reports/warm_v2')


def main():
    c=setup();r=read(REPORT/'WARM_V2_FRESH_ROBUSTNESS.json');v=read(REPORT/'WARM_V2_FRESH_REVIEW.json')
    assert v['status']=='passed' and len(r['windows'])==4
    found={};costs=[];evidence=[]
    for cutoff in [d for p in c['rolling_protocol'].values() for d in (*p['inner_train'],p['inner_validation'],p['outer_validation'])]:
        build=read(ART/'candidates'/cutoff/'BUILD.json')
        model=read(ART/'bpr'/cutoff/'model.json');feature=read(ART/'bpr'/cutoff/'features.json')
        src=read(ART/'candidates'/cutoff/'source/source-manifest.json')
        manifest=read(ART/'candidates'/cutoff/'manifest.json')
        assert build['status']=='completed' and build['target']['audit']['latest_history_date']<cutoff
        assert model['params']==c['bpr_params'] and model['cutoff']==cutoff and model['latest_history_date']<cutoff
        assert len(model['epochs'])==100 and model['factor_shapes'][0][1]==model['factor_shapes'][1][1]==101
        assert src['cutoff']==cutoff and src['contract']==c['item2vec_config']
        expected={**c['candidate_source_config'],'cutoff':cutoff};assert manifest['config']==expected
        assert feature['cutoff']==cutoff and feature['pit_safe'] and not feature['future_labels_used']
        costs.append({'cutoff':cutoff,'candidate_build_seconds':build['wall_seconds'],'bpr_build_seconds':model['runtime_seconds'],
            'bpr_users':model['users'],'bpr_items':model['items'],'binary_pairs':model['binary_pairs']})
        evidence.append({'cutoff':cutoff,'candidate_config_exact':True,'item2vec_config_exact':True,'bpr_params_exact':True,
            'bpr_iterations':len(model['epochs']),'history_before_cutoff':True,'target_feature_audit':build['target']['audit']})
    # Load generated metadata, reuse generation SHA values; do NOT rehash large arrays.
    for path in ART.rglob('*.json'):collect(read(path),found)
    for item in found.values():
        if Path(item['path']).is_file():assert Path(item['path']).stat().st_size==item['bytes']
        else:raise FileNotFoundError(item['path'])
    data={'created_at':now(),'cutoffs':costs,'candidate_build_seconds_sum':sum(x['candidate_build_seconds'] for x in costs),
        'bpr_build_seconds_sum':sum(x['bpr_build_seconds'] for x in costs),'artifact_bytes':sum(p.stat().st_size for p in ART.rglob('*') if p.is_file()),
        'free_disk_bytes':shutil.disk_usage(ART).free,
        'elapsed_wall_since_registration_seconds':(datetime.fromisoformat(now())-datetime.fromisoformat(c['registered_at'])).total_seconds(),
        'scope':'component times may overlap outer fitting invocation timers; not additive with SCREEN/OUTER timers; no full-memory-peak instrumentation',
        'final_week':'not_run'}
    write(REPORT/'WARM_V2_FRESH_COST.json',data)
    v['cutoff_config_and_history_review']=evidence
    write(REPORT/'WARM_V2_FRESH_REVIEW.json',v)
    status=r['status'];passed=status in ('strong_pass','robust_pass')
    status_zh={'strong_pass':'稳健门槛通过且四窗均提升','robust_pass':'达到稳健门槛',
        'weak_generalization':'均值为正但未达到稳健门槛','fresh_robustness_failed':'新历史均值未提升'}[status]
    s=r['summary'];comp_report=read(REPORT/'WARM_V2_FRESH_COMPLEMENTARITY.json');comp=comp_report['windows']
    lines=['# Warm-v2：冻结方案的新历史稳健性确认','',
        f"结论：**{status}（{status_zh}）**。这些是本项目的审核状态名。2019新历史四窗平均增量为{s['FRESH-601']['mean_delta']:+.9f}，正增{s['FRESH-601']['positive_windows']}窗，最差{s['FRESH-601']['worst_delta']:+.9f}。本轮已停止，等待人工审核。",'',
        '## 口径与冻结边界','',
        'FRESH-000/501/601是项目实验名：分别指原Warm-v1、增加原两列BPR匹配特征的排序器、两个模型的固定等权名次融合。BPR是行业通用贝叶斯个性化排序，在这里用用户/商品ID因子内积学习隐式偏好，含商品偏置，分数不是概率。RRF是行业通用倒数名次融合，公式始终为1/(60+原模型名次)+1/(60+BPR模型名次)。没有搜索或调整权重。','',
        '召回仍为原六路加权RRF前100件，加至多200个Item2Vec独有商品，每用户100–300件。Item2Vec是通过购买历史学习商品向量的行业通用方法。原84列与原LightGBM LambdaRank学习排序保持不变；BPR增强仅增加项目字段wv2_bpr_user_item_score（用户—商品含偏置内积，无概率含义）和wv2_bpr_unavailable（用户或商品无历史因子为1，对应分数缺失，否则0）。','',
        'MAP@12先计算每名用户前12件推荐的平均准确率AP，真值分母为未来7天完整去重购买数与12的较小值，再对原确定性10%评测用户平均。未召回购买仍进入分母。每窗均值独立计算，四窗均值等权。训练负采样30:1指负例:正例，保持原分层哈希规则。BPR使用截止前全部用户的二值购买关系，不是只使用10%用户；未购买不代表明确负反馈。','',
        '候选召回率为每用户召回真值数/完整真值数，再对全体评测用户平均；Oracle MAP是让召回真值全部排最前的理想上限，不是可学习成绩。正例候选对以用户—商品为单位。Top12新增对数相对同窗FRESH-000列表，指新增商品身份，不统计仅名次变化。','',
        '每名用户近12周无购买历史时，三个系统仍按原候选名次回退。原始重复事件保留；二值BPR与用户日内商品集合的去重只属于模型输入表示。静态商品目录继续使用全部articles可用的乐观假设；客户快照中的动态会员/活跃字段仍不用。','',
        '## 新历史证据与原开发证据分开','',
        '原2020开发四窗经过多阶段适应性研究，WV2-601平均0.027880053，相对Warm-v1平均+0.000921410，四窗全部提升；这些不能当作未见测试。下表2019四窗未进入此前Warm外层选模注册表，本轮在模型/策略完全冻结后运行；没有用它们选择架构。最终2020-09-16始终not_run（未运行），本轮没有提交文件或排行榜访问。','',
        '| 新验证周 | 原基线MAP | BPR增强MAP | 固定融合MAP | BPR增量 | 融合增量 |',
        '|---|---:|---:|---:|---:|---:|']
    for w,x in r['windows'].items():
        z=x['systems'];lines.append(f"| {x['cutoff']} | {z['FRESH-000']['map@12']:.9f} | {z['FRESH-501']['map@12']:.9f} | {z['FRESH-601']['map@12']:.9f} | {z['FRESH-501']['delta_vs_FRESH-000']:+.9f} | {z['FRESH-601']['delta_vs_FRESH-000']:+.9f} |")
    lines += [f"| 四窗等权均值 | {s['FRESH-000']['mean_map']:.9f} | {s['FRESH-501']['mean_map']:.9f} | {s['FRESH-601']['mean_map']:.9f} | {s['FRESH-501']['mean_delta']:+.9f} | {s['FRESH-601']['mean_delta']:+.9f} |",'',
        f"BPR单模型逐窗符号{s['FRESH-501']['sign_pattern']}，最差{s['FRESH-501']['worst_delta']:+.9f}；融合逐窗符号{s['FRESH-601']['sign_pattern']}，最差{s['FRESH-601']['worst_delta']:+.9f}。501只负责机制对照，不负责晋级。",'',
        'robust_pass（项目“稳健通过”）必须均值增量至少+0.000500、至少3窗不退化、最差不低于−0.000300；额外四窗严格正增才是strong_pass（“全窗正增通过”）。均值正但未过门槛为weak_generalization（“弱迁移”），均值不正为fresh_robustness_failed（“新历史验证失败”）。没有用8窗混合均值替代新历史门槛。','',
        '## 时间、候选与独立复核','',
        '第一训练点2018-12-26所需12周从2018-10-03开始，raw从2018-09-20开始，所需84天全部有记录。四条链使用附件指定日期，外层重训目标周固定为本链的内层训练周和内层验证周；只通过原内层早停规则选树轮数，不移动训练周。详见预注册CONTRACT.json。','',
        '| 新验证周 | 用户数 | 候选对数 | 真值对数 | 正例候选对 | 候选召回率 | 理想MAP上限 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in r['windows'].values():
        z=x['systems']['FRESH-000'];lines.append(f"| {x['cutoff']} | {z['users']} | {z['candidate_rows']} | {z['truth_pairs']} | {z['positive_candidate_pairs']} | {z['candidate_recall']:.9f} | {z['candidate_oracle_map@12']:.9f} |")
    lines += ['', '三个系统的候选召回率、理想上限、候选对数和用户集合完全相同。独立复核从原始交易按原哈希规则重建全部评测用户，不将范围限制为已有预测；核对每个候选标签、排序名次及完整真值AP，并用Pandas独立重放全部候选的模型名次和融合分数。每个无近期历史用户的最终12件身份和顺序与原基线一致。','',
        '## 用户收益与损失','',
        '下表改善/受损/不变按用户AP与同窗原基线比较；正贡献是所有改善用户的AP增量之和除以该窗全部用户数，负贡献同理。两者之和为整体MAP增量，不以各自子组人数作为分母。','',
        '| 新验证周 | 系统 | 改善用户 | 受损用户 | 不变用户 | 正贡献 | 负贡献 | Top12新增对 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for x in r['windows'].values():
        for t in ('FRESH-501','FRESH-601'):
            z=x['systems'][t];lines.append(f"| {x['cutoff']} | {t} | {z['users_improved']} | {z['users_harmed']} | {z['users_unchanged']} | {z['gross_positive_map_contribution']:+.9f} | {z['gross_negative_map_contribution']:+.9f} | {z['top12_new_pairs_vs_base']} |")
    lines += ['', '## 新历史内层互补性（只诊断，不选策略）','',
        '统计对象是有候选正例且近12周活跃的历史验证用户的正例候选对，不是完整外层用户分母。旧独有表示进入原模型Top12但未进入BPR模型Top12，新独有反之；共享表示两者Top12都命中。这不是独有召回，也不是只在某季节出现。','',
        '| 内层验证周 | 旧模型独有正确对 | BPR独有正确对 | 共享正确对 |', '|---|---:|---:|---:|']
    for x in comp.values():lines.append(f"| {x['cutoff']} | {x['base_only_correct_top12_positive_pairs']} | {x['bpr_only_correct_top12_positive_pairs']} | {x['shared_correct_top12_positive_pairs']} |")
    lines += ['', '与旧开发诊断对比时，额外报告“独有正确对净优势”：(BPR独有−原模型独有)/(BPR独有+原模型独有+共享)。这是本报告自定义描述量，分母为两模型Top12命中真值的并集对数，不是全部真值，也不是MAP。它只帮助比较互补的方向，不用于选权重或证明统计显著性。','',
        '| 证据来源 | 内层验证周 | 独有正确对净优势 |', '|---|---|---:|']
    for x in comp.values():
        b=x['base_only_correct_top12_positive_pairs'];n=x['bpr_only_correct_top12_positive_pairs'];shared=x['shared_correct_top12_positive_pairs']
        lines.append(f"| 新历史 | {x['cutoff']} | {(n-b)/(n+b+shared):+.6f} |")
    for x in comp_report['historical_development_context']['windows'].values():
        b=x['base_only_correct_top12_pairs'];n=x['bpr_only_correct_top12_pairs'];shared=x['shared_correct_top12_pairs']
        lines.append(f"| 旧开发 | {x['cutoff']} | {(n-b)/(n+b+shared):+.6f} |")
    lines += ['', '不同日期的人数、商品与真值分布不同，不能将上述对数直接相减并归因为模型变弱。即使内层存在独有正确商品，也仍可能在外层混排中损失AP；是否迁移最终只由预注册外层门槛判断。']
    lines += ['', '## 只读结果分解','',
        '活动量分组按截止前12周原始购买事件数固定为0、1–5、6–20、21及以上；项目组名分别为inactive_0、low_1_5、medium_6_20、high_21_plus。不按模型结果重分组。分组AP和对完整用户MAP的贡献分别报告，避免将小组均值误当整体贡献。完整机器数据见ROBUSTNESS.json的activity_segments（活动量分组）字段。','',
        '| 新验证周 | 活动量组 | 用户数 | BPR整体贡献 | 融合整体贡献 |', '|---|---|---:|---:|---:|']
    for x in r['windows'].values():
        for name,g in x['activity_segments'].items():
            lines.append(f"| {x['cutoff']} | {name} | {g['users']} | {g['systems']['FRESH-501']['contribution_to_all_users_delta']:+.9f} | {g['systems']['FRESH-601']['contribution_to_all_users_delta']:+.9f} |")
    lines += ['', 'BPR特征的gain是树分裂损失改善的累计量，只证明模型使用情况，不是因果归因。下表同时记录原内层选出的树轮数及BPR分裂次数，供失败或收益机制判断；没有按这些结果改参数。','',
        '| 新验证周 | 原模型轮数 | BPR模型轮数 | BPR匹配分数分裂次数 |', '|---|---:|---:|---:|']
    for x in r['windows'].values():
        z=x['systems'];usage=z['FRESH-501']['feature_usage'];splits=next(a['splits'] for a in usage if a['feature']=='wv2_bpr_user_item_score')
        lines.append(f"| {x['cutoff']} | {z['FRESH-000']['inner_rounds']} | {z['FRESH-501']['inner_rounds']} | {splits} |")
    lines += ['', '分解只说明观察到的窗口、模型利用、误差互补、用户活动量和候选上限差异，不能自动断言唯一原因。候选Oracle高不等于BPR一定可学习；固定融合不一定在所有历史阶段优于两个单模型。拒绝时不会根据失败窗口重新调权重、换常量或重新特征探索。','',
        '本次具体观察：四个新历史内层窗口中，BPR独有正确对均多于原模型独有正确对，互补方向没有消失。外层融合相对BPR单模型，在四窗均减少了受损用户和负MAP贡献的绝对量，但也削弱部分正贡献；因此本次单BPR模型的平均增量更高，固定融合的最差窗口增量更好。不能据此宣称融合处处更优，也不根据新历史结果改冠军定义。整体四窗正增仍不意味着各活动量子组都提升，表中2月、11月的中等活动量组在融合下仍有负贡献。','',
        '## 成本与工程适配','',
        f"12个截止点候选/特征构建累计{data['candidate_build_seconds_sum']/60:.1f}分钟，BPR累计{data['bpr_build_seconds_sum']/60:.1f}分钟；从预注册到封包实际经过{data['elapsed_wall_since_registration_seconds']/3600:.2f}小时，含开发、验证和等待。新产物{data['artifact_bytes']/1024**3:.2f}GiB，D盘剩余{data['free_disk_bytes']/1024**3:.1f}GiB。组件计时与外层调用计时可能重叠，不重复累加。",'',
        '所有新大文件仅在新worktree的artifacts/warm_v2/fresh_robustness中；与主库共享Git对象，不clone大资产。主线只读，迁移备份未删除。原候选构建函数被复用，但旧函数的warm-recent字样只是历史命名；没有采用旧近期训练周移动策略。内存峰值不是全流程连续监控，不能把目标预算当实测峰值。','',
        '执行层处理：新目录不在旧会话默认写入范围，写报告/训练时使用明确授权执行；并发conda run曾发生临时文件冲突，测试改用同一conda环境内Python直接启动，未换依赖。构建/拟合仍使用原参数；进程启动时检查implicit、LightGBM与Torch版本一致。','',
        '## 复核入口','',
        '使用conda的hm-recommend环境，工作目录为.，并设置PYTHONPATH=src。python -m hm_recsys.warm_v2_fresh_review只审查已完成产物；python -m hm_recsys.warm_v2_fresh_close只封装证据和生成报告，两者都不训练。复核命令会复用本轮已完成的窗口复核缓存；全新独立重放需先人工确认缓存处理，不隐式删除证据。','',
        '23项Warm相关测试全部通过，pip check无依赖冲突，具体命令及输出见WARM_V2_FRESH_TESTS.json。训练编排、门槛单元测试与整批预测独立复核是不同证据，不以测试通过代替指标门槛。BPR多线程重训不保证逐位相同，本轮保存的实际因子与预测才是成绩对应的模型。','',
        '## 人工审核节点','',
        ('结果满足新历史门槛，将WV2-601标记为Warm-v2 integration candidate（项目“待人工接入主线候选”）；这不是合并批准。另见WARM_V2_INTEGRATION_PLAN.md。' if passed else '结果未满足新历史门槛，不提升为集成候选。保留原Warm-v1及旧开发证据；以上分解仅供人工决定后续，不启动自动救援实验。'),'',
        '本轮无论通过或失败均停止，不merge main，不执行Cold/Admission集成，不运行最终周，不自动推送。原2020成绩与原冠军注册表保持为历史事实，不用新证据覆盖旧失败记录。完整证据清单见WARM_V2_FRESH_OUTPUT_MANIFEST.json。','']
    (REPORT/'WARM_V2_FRESH_ROBUSTNESS.md').write_text('\n'.join(lines),encoding='utf-8')
    if passed:
        plan=f'''# Warm-v2 人工集成计划（本轮不执行）

Main当前HEAD：{git('rev-parse','main')}；Warm fresh分支HEAD：{git('rev-parse','HEAD')}。
Fresh实现尚未提交时，准确代码以本轮manifest为准，不能只引用父提交声称包含新代码。

建议main → 新integration分支 → clean-port（仅移植必要实现）/审阅后的cherry-pick → Cold/Admission兼容性重放。
禁止whole-branch blind merge；main与Warm有不同的迁移整理提交，不应把Warm旧迁移拓扑覆盖回主线。

1. 生产实现：截止前BPR二值关系/因子/匹配分数、原84列和新增两列的两个LambdaRank、完整候选池内等权RRF常量60及原无近期历史回退。参考warm_v2_bpr.py、warm_v2_engine.py、warm_v2_rank_fusion.py；抽出生产接口，不移植其历史试验注册副作用。
2. 依赖：implicit==0.7.3、threadpoolctl==3.6.0以及主线已验证的NumPy/SciPy/LightGBM/DuckDB环境；先查主线现有版本，不盲目覆写dependency.txt。
3. 保留但不进入生产路径：WV2-101至110统计特征、201/202容量、301–303损失采样、401近期目标、502偏置消融等失败/归因实验；Fresh脚本是离线验证工具，不是线上请求路径。
4. 必需测试：原候选/真值/分母、采样规则、BPR缺失ID、100隐维及商品偏置语义、cutoff截断、等权RRF同分顺序、无近期历史精确回退、独立完整候选分数及最终列表重放。
5. BPR刷新：每个线上可用截止点只能使用当时之前全部历史；保存用户/商品映射、实际因子、版本、参数、分数、SHA和bytes。多线程不承诺逐位重训一致。先制定刷新频率、计算/储存预算，不直接复用未来模型回填过去。
6. Warm输出语义由LightGBM原始分数变成两个排序器的倒数名次融合值，数值不是购买概率，也不是旧树分数。
7. Cold/Admission必须重新校准并重放兼容性，不能继续使用旧Warm raw-score尺度。Warm新Top12和末位商品也改变，不能只替换一列分数后宣称等价。校准只用获批训练/验证窗口，最终周仍锁定。
8. 推荐先人工审阅原BPR提交41d977b和融合提交6b4625a的必要函数及依赖，再从本轮证据中移植验证契约；不要把整个研究分支及迁移housekeeping盲目cherry-pick。

本轮只生成计划，不创建integration、不修改主线、不触碰Cold/B0、不解锁2020-09-16。
'''
        (REPORT/'WARM_V2_INTEGRATION_PLAN.md').write_text(plan,encoding='utf-8')
    write(REPORT/'WARM_V2_FRESH_STATUS.json',{'status':'Warm-v2 integration candidate' if passed else status,
        'human_review_required':True,'broad_exploration':False,'final_week':'not_run','automatic_merge_or_push':False})
    files=sorted(Path('src/hm_recsys').glob('warm_v2_fresh*.py'))+[Path('tests/test_warm_v2_fresh.py')]
    reused=['warm_v2_engine','warm_v2_bpr','warm_v2_rank_fusion','warm_v2_recent_data','warm_v2_features',
        'warm_v2_contract','m1','m15','m2','m28','m29','m210','m211','m212','metrics']
    files += [Path('src/hm_recsys')/(name+'.py') for name in reused]
    assert not git('diff','5fbe34b','--',*[str(Path('src/hm_recsys')/(name+'.py')) for name in reused]), 'reused algorithm code changed'
    files+=sorted(p for p in REPORT.glob('WARM_V2_FRESH*') if p.is_file() and p.name!='WARM_V2_FRESH_OUTPUT_MANIFEST.json')
    if passed:files.append(REPORT/'WARM_V2_INTEGRATION_PLAN.md')
    # Explicit evidence-package contract: preserve small model/configuration and
    # sampling metadata too, without rereading large factor or score arrays.
    metadata=sorted(ART.rglob('*.json'))
    manifest={'created_at':now(),'branch':git('branch','--show-current'),'parent_HEAD':git('rev-parse','HEAD'),
        'original_algorithm_commit':'5fbe34b1cb3a8993d3cfb5eb50b85bc3efacdae3','reused_algorithm_files_unchanged':True,
        'files':[evidence_id(p,reason='explicit_registry_evidence') for p in files],
        'artifact_metadata':[evidence_id(p,reason='explicit_registry_evidence') for p in metadata],
        'artifacts':list(found.values()),'large_asset_hash_policy':'reuse recorded generation hashes; existence/bytes checked, exact score/metric replay separate',
        'final_week':'not_run'}
    write(REPORT/'WARM_V2_FRESH_OUTPUT_MANIFEST.json',manifest)
    print({'status':status,'candidate':passed,'artifact_gib':data['artifact_bytes']/1024**3},flush=True)


if __name__=='__main__':main()
