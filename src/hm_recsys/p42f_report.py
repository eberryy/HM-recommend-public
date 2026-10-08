"""Chinese P4.2F report from verified evidence, not proposed gains."""
from pathlib import Path
import numpy as np
from .p41a_contract import read_json,write_json,identity
from .p42f_contract import RUN_ID,COLD,WARM,USER,PAIR,PERSONAL,now
from .p42f_evaluate import VARIANTS

def table(headers,rows):
    def f(x):return '不可用' if x is None else f'{x:.8g}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(f(x) for x in r)+' |' for r in rows])

def report(repo):
    repo=Path(repo).resolve(); root=repo/'artifacts/phase4'/RUN_ID; rp=repo/'reports/phase4'
    m=read_json(rp/'P4_2F_metrics.json');v=read_json(rp/'P4_2F_VERIFICATION.json');c=read_json(rp/'P4_2F_EXPERIMENT_CONTRACT.json')
    assert m['status']=='completed' and v['status']=='pass'
    windows=m['windows'];labels={'winter_20200122':'冬季','spring_20200318':'春季','early_summer_20200624':'初夏','late_summer_20200819':'夏末'}
    title={VARIANTS[0]:'G统一效用',VARIANTS[1]:'H用户阈值','W0':'W0原推荐'}
    training=m['training_audit'];h={w:read_json(root/'models'/w/'TRAINING.json') for w in windows}
    artifacts={
        'action_training_audit':dict(cutoffs=training,model_training=h),
        'global_utility':{w:r['variants'][VARIANTS[0]] for w,r in windows.items()},
        'hierarchical_threshold':{w:dict(**r['hierarchical'],metrics=r['variants'][VARIANTS[1]]) for w,r in windows.items()},
        'novelty_appetite_audit':{w:{k:r['variants'][k]['mechanisms'] for k in VARIANTS} for w,r in windows.items()},
        'admission_risk':{w:{k:dict(admission=r['variants'][k]['admission'],buckets=r['variants'][k]['buckets']) for k in VARIANTS} for w,r in windows.items()},
        'segment_metrics':{w:{k:r['variants'][k]['segments'] for k in ['W0',*VARIANTS]} for w,r in windows.items()}}
    for name,a in artifacts.items():write_json(rp/f'p4_2f_{name}.json',dict(stage='P4.2F',data=a,final_week='not_run'))
    text=['# P4.2F：相对证据驱动的替换效用与用户阈值','',
        f"状态：completed（完成），独立复核pass（通过）。机器结论：{m['decision']['machine_decision']}（本轮相对证据效用未满足全部晋级要求）；保留/选择：{m['decision']['selected']}。",'',
        '## 1. 先说结论与术语','',
        '本轮不再比较冷暖绝对概率。G直接回归单次替换的AP增量；H先预测配对得分，再减去用户状态阈值和收缩的历史用户残差。两者是预注册并列方案，不按外层结果调规则。',
        'W0是冻结Warm-v1原前12件及顺序；Cold50是冻结M4内容召回前200件经B0冷侧专家排序后的前50件。qC/qW为冷侧/暖侧观察购买倾向模型，仅使用各自用户内相对名次，不将绝对概率或其对数赔率输入G/H。以上简称是项目命名。',
        '操作边是同一用户的一件Cold候选与一个W0位置的配对；候选与W0重复则先排除。有益边的真实单次AP增量>0，有害边<0，中性边=0。没有观察购买不是已知不喜欢。',
        'AP@12为单个用户前12件的平均精度，分母为min(下一周不同真值商品数,12)；MAP为用户AP平均。整体分母是完整冻结W0评测用户，包括无Cold候选者；分组MAP沿用Phase4：只对该组有真值的用户求平均，以该组真值重新计算AP。不可与全用户分母的贡献值混用。',
        'warm_21_plus表示截止日前商品全站事件数至少21；strict_cold为0；sparse1_5为1–5；all_cold_sparse为0–5。不是冷用户。cold-only表示该Cold候选不在Warm150集合中，而不仅是没在W0前12。',
        'ROC-AUC衡量正负类排序；PR-AUC使用非插值平均精度。Pearson为线性相关，Spearman为带同分平均名次的秩相关。主操作指标只以有益为正、有害为负；次级指标以有益为正、其余全部为负，两个基础率不可混用。',
        '所有跨窗平均对四窗等权；跨窗累计用户计数是用户—窗口，不是去重人数。实验是离线观察购买评测，不证明线上转化或因果收益。','',
        '## 2. 数据、训练与防泄漏','',
        '经Lyra确认，G/H使用约10%历史共同用户，不是P4.2E的100%冷侧总体。全站商品购买计数仍使用所有用户的交易事件。保留全部非零收益边，中性边按固定SHA256哈希2%概率保留、权重50；实际保留比例允许围绕2%波动，不是每组强行凑2%。',
        table(['历史截止日','共同操作用户','完整操作边','有益边','有害边','中性边','保留中性边','训练总行'],
          [[t,a['historical_action_users'],a['full_edges'],a['beneficial'],a['harmful'],a['neutral'],a['retained_neutral'],a['training_rows']] for t,a in training.items()]),'',
        'H两折交叉预测按用户固定，同一用户全部日期在同一折。另一折模型对完整操作空间评分，产生用户最佳预测机会s；真实最佳单次机会为g；Ridge阈值回归目标d=s−g。不能只对抽样边取最大值。',
        '两折只保证用户折外，全部训练标签早于相应外层；不声称在每个历史评分日严格时间前推。qC/qW操作输入则必须严格时间前推，找不到合法模型时缺失并标记不可用，不用同日OOF模型补值。',
        'Ridge固定alpha=1，训练中位数填缺失后标准化；b_u=n/(n+2)×历史残差均值。外层只用更早标签已结束的历史窗口。n=0只保证b_u=0，阈值仍包含用户状态项，不等于所有人共享常数截距。',
        '辅助字段边界：threshold-training.parquet中的prior_n/prior_b是同一外层历史池内“先按日期筛早期残差行”的诊断统计，不是G、H配对模型或Ridge的输入，也没有用它生成历史推荐。其残差所用Ridge参数来自整个外层历史池，故不能把该辅助列当成在那个历史日期已可部署的PIT用户阈值。复核中的historical_residual_strict_prior只验证残差行的日期筛选；本轮未运行历史逐日前推H。真正用于四个外层的b_u，其全部标签、模型与残差均早于该外层。以后若复用辅助列做历史预测，必须另行前推模型，不能直接使用该列。',
        'H的效用不是直接校准过的单边期望AP：它减去的是从“最佳单次机会”误差学习的用户阈值。G和H的胜负不能单独证明随机残差b_u的因果价值；本轮没有H去掉b_u的消融。','',
        '## 3. 完整MAP与保护门槛','',
        table(['窗口','角色','整体MAP','相对W0差','暖商品组MAP','严格冷MAP','稀疏MAP','全部冷稀疏MAP'],
         [[labels[w],title[k],a['map12'],a['delta_vs_w0'],*[a['segments'][s]['map12'] for s in ['warm_21_plus','strict_cold','sparse1_5','all_cold_sparse']]] for w,r in windows.items() for k,a in r['variants'].items()]),'',
        table(['方案','平均MAP','平均整体差','平均暖组差','平均冷稀疏差','全部门槛通过'],
          [[title[k],a['mean_map'],a['mean_delta'],a['mean_warm_delta'],a['mean_cold_delta'],a['pass_all']] for k,a in m['decision']['gates'].items()]),'',
        table(['方案','门槛','通过'],[[title[k],key,value] for k,a in m['decision']['gates'].items() for key,value in a['tests'].items()]),'',
        '门槛字段：overall_mean为整体均值严格提升，overall_nondegrade为至少3窗不退化，overall_worst为最差差值≥−0.0002；warm_mean为暖组均值差≥−0.0001，warm_windows为至少3窗暖组差≥−0.0002；cold_mean为冷稀疏均值严格提升，cold_windows为至少3窗不退化；cold_only_windows为至少2窗实际插入来源独有正例；efficiency为累计插入正例多于移除正例。','',
        '## 4. 配对得分能否区分真实替换收益','',
        table(['窗口','方案','有益对有害AUC','主PR-AUC','有益对其余AUC','次级PR-AUC','Pearson','Spearman','效用>0边数','其中有益','其中有害','其中中性','合格边有益率'],
          [[labels[w],title[k],a['beneficial_vs_harmful']['roc_auc'],a['beneficial_vs_harmful']['pr_auc'],a['beneficial_vs_all_nonbeneficial']['roc_auc'],a['beneficial_vs_all_nonbeneficial']['pr_auc'],a['correlation']['pearson'],a['correlation']['spearman'],a['eligible_edges'],a['eligible_beneficial'],a['eligible_harmful'],a['eligible_neutral'],a['eligible_beneficial_ppv']] for w,r in windows.items() for k in VARIANTS for a in [r['variants'][k]['action']]]),
        '合格边有益率=效用>0且真实有益的边数/全部效用>0边数，不是实际匹配后的精度。多个边可共享候选或位置，不能当成独立正例商品计数。','',
        '## 5. 实际替换及多件加和边界','',
        table(['窗口','方案','准入用户','覆盖率','替换总数','每准入用户替换数','最大替换数','插入正例','移除正例','严格冷插入','稀疏插入','cold-only正例插入'],
          [[labels[w],title[k],a['users_with_admission'],a['coverage'],a['replacements'],a['mean_per_admitted'],a['max_replacements'],a['inserted_cold_positives'],a['removed_warm_positives'],a['strict_positive_inserted'],a['sparse_positive_inserted'],a['cold_only_positive_inserted']] for w,r in windows.items() for k in VARIANTS for a in [r['variants'][k]['admission']]]),'',
        table(['窗口','方案','替换数桶','用户','相同用户MAP差','插入正例','移除正例','实际AP差减单边AP差和'],
          [[labels[w],title[k],bucket,a['users'],a['map_delta'],a['inserted_cold_positives'],a['removed_warm_positives'],a['mean_exact_minus_single_sum']] for w,r in windows.items() for k in VARIANTS for bucket,a in r['variants'][k]['buckets'].items()]),
        '匹配最大化预测单边收益之和；多件同时替换的真实AP不严格可加。末列为同桶用户“真实列表AP增量−各实际执行边单独AP增量之和”的均值。该差异用于定位加和误差，不事后加入max1。不同替换数桶的用户也不同，桶间比较不是替换件数的随机因果效应。','',
        '## 6. 新鲜商品接受度、历史丰富度和用户阈值','',
        'novelty appetite（本项目新鲜商品接受度）主分组字段为用户历史购买中“购买日前全站交易数≤5”的事件占比。数据只有日期，因此同日交易全部不计入该次购买前的交易数；不按文件行顺序伪造日内先后。全历史与近84日分别统计，重复交易事件保留。',
        '购买时热度百分位以全静态目录105542件商品为参照，按购买日前累计交易数的同分平均名次计算，包含历史零交易商品；这是乐观目录口径，不是商品当时已上架的证明。',
        'Q1至Q4按历史训练用户—窗口主占比的25/50/75分位边界切分，低到高，同分落低桶；历史丰富度按全历史购买事件数的三分位边界切分为low/medium/high（低/中/高）。边界冻结后应用外层，不使用外层标签。相同值很多时允许空桶。',
        table(['窗口','方案','分组轴','分组','用户','准入覆盖','插入正例/替换数','移除正例/替换数','MAP差'],
          [[labels[w],title[k],axis,group,a['users'],a['coverage'],a['cold_positive_insertion_rate'],a['warm_positive_removal_rate'],a['map_delta']] for w,r in windows.items() for k in VARIANTS for axis,groups in r['variants'][k]['mechanisms'].items() for group,a in groups.items()]),'',
        'b_sign按b_u正、绝对值≤1e-12、负分组（positive/near_zero/negative）。阈值分布、历史窗口支持数以及各支持组平均|b_u|如下；绝对值小不等于稳定有效，稳定价值仍应由跨窗收益和对照证明。',
        table(['窗口','阈值部分','平均','中位数','最小','最大'],[[labels[w],key,r['hierarchical'][key]['mean'],r['hierarchical'][key]['median'],r['hierarchical'][key]['min'],r['hierarchical'][key]['max']] for w,r in windows.items() for key in ['tau_fixed','b_u','tau_u']]),'',
        table(['窗口','历史支持窗口数','用户','平均绝对残差'],[[labels[w],key,a['users'],a['mean_abs_b']] for w,r in windows.items() for key,a in r['hierarchical']['support'].items()]),'',
        '## 7. 23项问题答复','']
    def pooled(k,field):return sum(r['variants'][k]['admission'][field] for r in windows.values())
    answers=[
        'G逐窗主AUC/PR见第4节，主比较仅有益对有害边。','H对应指标见第4节。',
        'G完整操作空间Pearson/Spearman见第4节，不只计算非零收益边。','H对应相关性见第4节。',
        '第6节按历史分位边界展示；不能仅凭高接受度用户被更多准入就说更安全，必须同时看移除率与MAP差。',
        '历史丰富度与偏好可能混杂，本轮分组不能独立区分偏好与置信度的因果作用。',
        '已报告各n_u组的平均|b_u|；没有残差去除消融或独立置信区间，不能仅据收缩公式声称更稳定。',
        'n_u=0时b_u严格为0；回退到用户状态总体阈值，不是统一常数阈值。',
        f"G累计替换{pooled(VARIANTS[0],'replacements')}次；逐窗每准入用户均值与最大值见第5节。",
        f"H累计替换{pooled(VARIANTS[1],'replacements')}次；对应逐窗均值见第5节。",
        '数量由严格正效用和一对一匹配决定，未写死K；实际0/1/2/3/4+分布见第5节。',
        '第5节单独给出多件桶损伤和加和误差；不把跨桶用户构成差异全部归因于多件替换。',
        f"G插入正例{pooled(VARIANTS[0],'inserted_cold_positives')}，移除正例{pooled(VARIANTS[0],'removed_warm_positives')}。",
        f"H插入正例{pooled(VARIANTS[1],'inserted_cold_positives')}，移除正例{pooled(VARIANTS[1],'removed_warm_positives')}。",
        f"严格冷正例实际插入：G {pooled(VARIANTS[0],'strict_positive_inserted')}，H {pooled(VARIANTS[1],'strict_positive_inserted')}；逐窗见第5节，不等于彻底解决冷启动。",
        '稀疏组逐窗MAP见第3节；与整体保护分开判断。','整体保护按第3节三个overall门槛，不以局部冷收益替代。',
        '暖侧保护按第3节warm_mean与warm_windows两门槛同时判断。',
        f"G全部门槛通过={m['decision']['gates'][VARIANTS[0]]['pass_all']}。",
        f"H全部门槛通过={m['decision']['gates'][VARIANTS[1]]['pass_all']}。",
        '联合查看合格边正负组成、实际插入/移除以及多件加和误差。没有用户残差消融，不能把G/H差异唯一归因于个性化；有害单边被准入也不能全部归咎于匹配。',
        '最终周2020-09-16仍not_run，未生成提交文件或排行榜分数。','Warm-v2未整合，P4.3未启动；完成后等待人工决定。']
    gkey,hkey=VARIANTS
    for k,answer_index in [(gkey,8),(hkey,9)]:
        total_users=sum(r['variants'][k]['admission']['users'] for r in windows.values())
        admitted=pooled(k,'users_with_admission');replacements=pooled(k,'replacements')
        answers[answer_index]=f"{title[k]}累计替换{replacements}次；按全部{total_users}个用户—外层窗口平均{replacements/total_users:.6f}件，按{admitted}个准入用户—窗口平均{replacements/admitted if admitted else 0:.6f}件。"
    q4_better={k:sum(r['variants'][k]['mechanisms']['novelty']['Q4']['map_delta']>r['variants'][k]['mechanisms']['novelty']['Q1']['map_delta'] for r in windows.values()) for k in VARIANTS}
    answers[4]=f"未显示稳定更安全：Q4相对Q1的MAP差更好，G只有{q4_better[gkey]}/4窗、H只有{q4_better[hkey]}/4窗；两方案Q4四窗MAP差仍全部为负。这里是分组关联，不是新鲜偏好的因果证明。"
    answers[11]='不是唯一失败原因：G/H的恰好1件替换桶四窗MAP差均为负；多件桶的实际AP差减单边AP差之和总体为非负，意味着实际损失通常比单边损失简单加总更小，而不是加和误差制造了本次大部分额外损伤。'
    sparse=[]
    for k in VARIANTS:
        delta=[r['variants'][k]['segments']['sparse1_5']['delta_vs_w0'] for r in windows.values()]
        sparse.append(f"{title[k]}稀疏MAP平均差{np.mean(delta):+.8f}，不退化{sum(d>=0 for d in delta)}/4窗")
    answers[15]='；'.join(sparse)+'。不代替整体/暖侧保护门槛。'
    answers[20]='失败在匹配前已存在：G合格有益/有害边合计260/11186，H为1305/43683；实际匹配不能把不可靠正效用自动变成可靠决策。H两夏用户总阈值中位数为负，降低准入门槛，伴随更多替换与暖侧损伤；未做随机残差消融，不能把全部差异归因于b_u。恰好1件桶也失败，不能只怪多件加和。'
    text+=[table(['问题序号（对应用户任务第36节）','答复'],[[i+1,a] for i,a in enumerate(answers)]),'',
       '### 机制归纳','',
       f"G共插入{pooled(gkey,'inserted_cold_positives')}个Cold正例、移除{pooled(gkey,'removed_warm_positives')}个原正例；H分别为{pooled(hkey,'inserted_cold_positives')}与{pooled(hkey,'removed_warm_positives')}。严格冷实际插入分别为{pooled(gkey,'strict_positive_inserted')}与{pooled(hkey,'strict_positive_inserted')}。因此本轮不是“Cold始终进不了Top12”，而是“准入有了一些冷收益，但付出了大得多的暖侧代价”。",
       answers[20],answers[4],
       '特别要区分：H用最佳单次机会误差推断用户阈值，当真实最佳机会高于配对模型的预测最大值时，d为负。把这个用户级负阈值减到该用户每条边上，会同时放宽许多并非有益的边。两夏负阈值与大量准入的观察符合此机制，但尚未通过隔离对照证明它是唯一根因。本轮不改成截断阈值、max1或新增模型。','',
       '## 8. 特征字段注释','',
       '以下全部变量名属于本项目字段命名；rank/percentile/zscore/RRF是行业通用概念。rank越小越靠前，percentile为同分平均名次归一化，zscore为减同用户均值再除总体标准差；RRF为倒数名次融合。',
       table(['字段组及精确字段','构造与参照'],[
        [', '.join(COLD[:9]),'B0原50件中的名次、名次/50、同用户完整50件分数百分位与标准分；相对原第2、第5、中位分数之差除完整50件标准差。M4原200件粗排名次及名次/200。先标准化后排除W0重合。'],
        [', '.join(COLD[9:13]),'截止前交易数0与1–5标志；不在Warm150与同时在Warm150的来源标志。'],
        [', '.join(COLD[13:]),'合法更早全量qC模型在同用户完整Cold50的名次、名次/50、百分位；无合法模型则缺失和available=0。'],
        [', '.join(WARM[:6]),'原Top12位置；warm_rank_pct沿用冻结上游候选总体归一化值，不擅自改成/12；分数标准分和百分位以完整Top12为参照；availability表示原模型分数是否可用；source_count是原召回路支持数。'],
        [', '.join(WARM[6:-3]),'各已有召回路的支持标志、路内原名次、原RRF分数贡献；item2vec为商品共购序列嵌入召回的支持与原名次。repurchase=复购，recent_popularity=近期热度，product_family=商品家族，user_day_covisit=用户日篮子共现，age_popularity=年龄分群热度，attribute_content=属性内容。'],
        [', '.join(WARM[-3:]),'合法更早qW在该用户完整Top12的名次、百分位和可用标志；不用绝对概率。'],
        [', '.join(USER[:10]),'用户过去事件总数；购买日前商品全站事件数≤5/≤20的事件数及占全部过去事件比例；近84日≤5事件数及占近84日全部事件比例；最近一次≤5购买距截止日天数；每次购买日前商品事件数中位数及全目录热度百分位中位数。'],
        [', '.join(USER[10:]),'近84日事件数/不同商品数、全历史不同商品数、近84日与全历史活跃购买日数、最后购买距今天数、该用户更早且标签期已结束的合格操作窗口数。全历史事件总数使用user_past_purchase_count同一字段，不重复建列。'],
        [', '.join(PAIR),'冷名次比例减暖名次比例；冷百分位减暖脆弱百分位（1−暖分数百分位，无分数时用(位置−1)/11）；B0名次乘暖位置。'],
        [', '.join(PERSONAL),'严格冷/稀疏标志分别乘全历史≤5购买占比；冷名次比例乘该占比；暖位置乘全历史购买事件数。仅G使用，不进入H配对模型。']]),'',
       '## 9. 工程、验证与停止','',
       '准备阶段两次接口失败已保留：夏末外层原资产不在历史修复目录；环境没有pandas可选Parquet引擎，修复为现有DuckDB读写。均发生于新数据统计前，没有拟合失败重试、参数调整或新增依赖。',
       '正式推理日志出现LightGBM/sklearn“数组无列名”告警：训练已明确传入特征名，预测传入固定列序数组。没有据此改模型；独立复核检查模型列名与合同一致，并以保存模型逐行重放数值。',
       f"购买时统计耗时{read_json(root/'NOVELTY_PREPARATION.json')['seconds']:.2f}秒；正式准备/训练/评测{m['seconds']:.2f}秒；独立复核{v['seconds']:.2f}秒。均不包含阅读、实现与写报告的时间。16次LightGBM拟合和4次Ridge拟合，无qC/qW或上游模型重训，无GPU训练。",
       f"独立复核{len(v['checks'])}项通过。检查全部历史训练标签和抽样重放、全部历史用户折外最大机会、模型列序、外层预测与最终AP；精确匹配另用每窗每方案前16个用户图的独立线性规划检查最优目标（零合格边图不求解）。不把抽样最优性检查说成所有图的独立算法证明。",
       '补充复核见P4_2F_SUPPLEMENTAL_VERIFICATION.json：34份权威文件与登记身份一致；独立核对Ridge训练中位数/标准化器及一阶最优条件、两折拟合行数，没有额外拟合。辅助历史残差列的时点适用边界见第2节，不将行级截断核查扩大解释为历史端到端PIT验证。',
       '单次AP公式另经4096种Top12命中状态×12位置×2候选标签穷举测试。大型逐行数据、模型、预测均在忽略目录；没有提交推送。完成后停止，不启动新阶段。']
    (rp/'P4_2F_FINAL.md').write_text('\n\n'.join(s for s in text if s)+'\n',encoding='utf-8')
    payloads=[identity(p) for p in root.rglob('*') if p.is_file() and p.suffix not in ['.wal']]
    reports=[identity(p) for p in rp.glob('*') if p.name.lower().startswith(('p4_2f_')) and p.name!='P4_2F_OUTPUT_MANIFEST.json']
    write_json(rp/'P4_2F_OUTPUT_MANIFEST.json',dict(stage='P4.2F',at=now(),status='completed',verification='pass',
        reports=reports,source=[identity(p) for p in (repo/'src/hm_recsys').glob('p42f*.py')],tests=identity(repo/'tests/test_p42f.py'),
        artifacts=payloads,final_week='not_run',Warm_v2_integrated=False,selected=m['decision']['selected'],
        hash_purpose='bind generated outputs for later independent replay; not a standalone authenticity claim'))

if __name__=='__main__':report('.')
