"""Render P4.2G aggregate evidence; no fit, scoring or policy selection."""
from pathlib import Path
import numpy as np
from .p41a_contract import read_json,write_json,identity
from .p42g_contract import RUN_ID

NAMES={'winter_20200122':'冬季','spring_20200318':'春季','early_summer_20200624':'初夏','late_summer_20200819':'夏末'}

def fmt(x):
    if x is None:return '不可用'
    if isinstance(x,float):return f'{x:.9g}'
    return str(x)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in rows])+'\n'

def report(repo):
    repo=Path(repo).resolve();dest=repo/'reports/phase4';root=repo/'artifacts/phase4'/RUN_ID
    m=read_json(dest/'P4_2G_metrics.json');v=read_json(dest/'P4_2G_VERIFICATION.json');assert m['status']=='completed' and v['status']=='pass'
    c=read_json(dest/'P4_2G_EXPERIMENT_CONTRACT.json');ws=m['windows'];vd=m['verdicts']
    exports={
        'p4_2g_multiclass_risk':{w:x['classifier'] for w,x in ws.items()},
        'p4_2g_benefit_magnitude':{w:dict(training=m['training'][w]['benefit'],outer=x['magnitude']['benefit']) for w,x in ws.items()},
        'p4_2g_harm_magnitude':{w:dict(training=m['training'][w]['harm'],outer=x['magnitude']['harm']) for w,x in ws.items()},
        'p4_2g_action_metrics':{w:dict(R=x['R']['action'],U_C=x['classification_only_diagnostic'],**x['references']) for w,x in ws.items()},
        'p4_2g_gate_tail':{w:x['gate'] for w,x in ws.items()},
        'p4_2g_positive_survival':{w:x['survival'] for w,x in ws.items()},
        'p4_2g_admission_risk':{w:dict(admission=x['R']['admission'],buckets=x['R']['buckets']) for w,x in ws.items()},
        'p4_2g_segment_metrics':{w:dict(W0=x['W0']['segments'],R=x['R']['segments'],mechanisms=x['R']['mechanisms'],cutpoints=x['cutpoints']) for w,x in ws.items()}}
    for name,data in exports.items():write_json(dest/(name+'.json'),dict(stage='P4.2G',data=data,final_week='not_run'))
    text=['# P4.2G：风险分离准入效用（10%机制试验）',
          '状态：completed（已完成）；独立复核pass（通过）。本轮保留W0，不自动晋级、不扩量。',
          '## 1. 结论与适用边界',
          f"四窗等权平均MAP@12为 **{vd['mean_map']:.9f}**，相对W0为 **{vd['mean_delta']:+.9f}**。累计插入Cold正例 **{vd['inserted']}** 对，移除原推荐正例 **{vd['removed']}** 对；插入/移除比为 **{fmt(vd['ratio'])}**，是P4.2F G的 **{fmt(vd['ratio_improvement'])}** 倍。比值状态{vd['ratio_status']}（finite=分母非零的有限比值；infinite=仅分母为零；undefined=两者均零）。",
          table(['判定字段（项目自定义）','中文含义','结果'],[
              ['risk_separation_action_signal','有益/有害边排序信号是否优于G',vd['risk_separation_action_signal']],
              ['risk_separation_policy_precision','实际插入与移除正例的比例是否达到试验门槛',vd['risk_separation_policy_precision']],
              ['risk_separation_map_safety','整体及暖商品组损伤是否在试验容忍范围内',vd['risk_separation_map_safety']],
              ['full_history_scaleup_allowed','是否具备下一轮人工讨论扩量的条件，不是执行授权',vd['full_history_scaleup_allowed']]]),
          '本次是固定10%历史用户的机制筛选，不是100%训练、线上试验或最终模型晋级。supported/mixed/rejected分别表示满足预注册规则、部分比较改善、未支持；不表示统计显著性。所有判定使用未四舍五入值。',
          '## 2. 术语、单位与模型',
          '本报告项目命名：W0=冻结Warm-v1原Top12及位置；M4=冻结纯内容商品表示；B0=冻结Cold排序专家；Cold50=M4前200件经B0排序后的前50件，排除已在W0的商品后进入操作空间。G=P4.2F直接L2回归单次AP增量模型；R或R_risk_separated=本轮风险分离效用方案。',
          '操作边（图优化通用概念）是一条“截止日—用户—Cold商品—原Warm位置”记录，每用户最多50×12条。B/N/H为项目标签，分别指exact单次AP增量>0、=0、<0的有益/中性/有害边；不使用容差归零。主冷正例以排除W0重合后的真实购买用户—商品对计数，不因12个位置重复计数。',
          'p_B/p_N/p_H是上述三类的预测概率；m_B/m_H是条件于有益/有害时的预测AP绝对幅度，分别仅用B/H训练并裁剪到[0,1]。正式U_R=p_B×m_B−p_H×m_H，λ固定1；只有U_R>0可匹配，等于0拒绝。U_C=p_B−p_H只诊断排序与相关性，不匹配、不计算正式推荐MAP。三分类每轮3棵树，因此250轮对应750棵树，不是擅自增加轮数。',
          'AP@12是单用户前12件平均精度，分母min(下一周不同真值商品数,12)；MAP为用户AP均值。整体分母为冻结W0全部评测用户，无Cold候选者也保留。分组MAP仅对该组有真值的用户求均值、以该组真值重算AP。strict_cold、sparse1_5、warm_21_plus、all_cold_sparse分别为截止日前商品全站交易事件数0、1–5、≥21、0–5；这是商品冷暖，不是用户冷暖。',
          'ROC-AUC为行业二类排序指标；本报告PR-AUC统一使用非插值平均精度。主比较仅B对H，次级B对所有非B，分母不同不能横向比较数值大小。Pearson为线性相关，Spearman为同分平均名次的秩相关；均在完整外层操作空间计算。',
          '其他术语：L2回归使用平方误差（行业方法）；exact指按原公式精确重算而非近似标签；label_end是历史标签结束日，outer cutoff是待评测窗口起点，前者必须严格早于后者。OOF为行业折外预测，本轮不增加折外校准；class_weight指额外类别权重，本轮禁用。feature_contract是项目的精确字段与构造约定。float64 epsilon为双精度机器精度，约2.22×10⁻¹⁶，只用于对数数值保护。qC/qW分别为冷侧与暖侧历史购买倾向模型，本轮仅继承其原相对名次特征。',
          '多类logloss为真实类别概率的负对数均值，只在取对数时以float64 epsilon保护；多类Brier为每条边三个类别平方误差之和，再对边平均。可靠性表固定10个等宽概率区间[0,0.1)…[0.9,1]；空箱记不可用。概率很小时会集中在第一箱，因此等宽表不能单独证明极端低概率尾部已校准。',
          '准入覆盖率=至少替换一次的用户数/该窗全部W0用户数；插入、移除率的分母为实际替换总次数。插入/移除比则是两类正例对数相除。多窗累计均为用户—窗口或用户—商品—窗口观察，不是跨窗去重人数。未观察购买不是明确负反馈，离线倾向不等于曝光后的真实购买概率。',
          '## 3. 冻结、数据及成本',
          f"合同登记于{c['created_at_utc']}，起始提交{c['git_sha']}。直接复用P4.2F已保存训练数组、数据对象及外层名单，70列特征及列序不变；精确字段与语义见[P4.2F报告第8节](P4_2F_FINAL.md#8-特征字段注释)，本轮feature_contract完整嵌入合同。没有新增或删除特征、绝对qC/qW概率、商品ID或未来首售代理。",
          '全部B/H保留，中性仍使用P4.2F同一SHA256确定性2%样本、权重50；B/H权重1。三分类不加class_weight；两个幅度模型只训练各自符号行。每外层仅使用label_end<outer cutoff的原历史池，不做OOF新校准、早停、参数搜索或神经模型训练。全部全站历史统计沿用P4.2F，不重新按10%人口统计全局热度。',
          table(['外层','历史训练日期数','分类训练边数','B训练边数','H训练边数'],[[NAMES[w],len(c['historical_pools'][w]),m['training'][w]['classifier']['rows'],m['training'][w]['benefit']['rows'],m['training'][w]['harm']['rows']] for w in ws]),
          f"预先估计计算与核验20–45分钟；实际行一致性预检{read_json(root/'PREFLIGHT.json')['seconds']:.2f}秒、正式训练/评分/评测{m['seconds']:.2f}秒、独立复核{v['seconds']:.2f}秒。正式峰值进程工作集{m['peak_gib']:.3f} GiB。时间不含阅读、代码实现及报告编写。CPU 4线程，GPU不用；12次正式拟合，无上游重训。",
          '## 4. 三分类概率与可靠性',
          table(['窗口','类别','完整操作边数','实际类别率','平均预测概率'],[[NAMES[w],k,x['classifier']['edges'],z['observed_rate'],z['predicted_mean']] for w,x in ws.items() for k,z in x['classifier']['classes'].items()]),
          table(['窗口','多类logloss','多类Brier'],[[NAMES[w],x['classifier']['multiclass_logloss'],x['classifier']['multiclass_Brier']] for w,x in ws.items()])]
    for w,x in ws.items():
        text += [f"### {NAMES[w]}固定可靠性分箱",table(['概率字段','下界','上界','边数','实际该类边数','实际类别率','平均预测概率'],
           [[k,b['low'],b['high'],b['rows'],b['positives'],b['observed_rate'],b['predicted_mean']] for k,bins in x['classifier']['reliability'].items() for b in bins])]
    text += ['## 5. 操作区分与幅度模型',table(['窗口','分数','B对H AUC','B对H PR','B对其余AUC','B对其余PR','Pearson','Spearman'],[
        [NAMES[w],name,a['beneficial_vs_harmful']['roc_auc'],a['beneficial_vs_harmful']['pr_auc'],a.get('beneficial_vs_all_nonbeneficial',{}).get('roc_auc'),a.get('beneficial_vs_all_nonbeneficial',{}).get('pr_auc'),a['correlation']['pearson'],a['correlation']['spearman']]
        for w,x in ws.items() for name,a in [('U_R正式',x['R']['action']),('U_C只读',x['classification_only_diagnostic']),('G历史',x['references']['G']),('旧U历史',x['references']['old_U'])]]),
        '旧U是P4.2R3冻结log-odds差，比较来自P4.2D同一操作空间只读审计，不重新训练或匹配。U_C次级B对其余指标未按本轮合同要求报告，表内记不可用。',
        table(['窗口','幅度头','对应符号边数','实际幅度均值','预测幅度均值','条件MAE','条件RMSE','全操作空间负值裁剪数','超1裁剪数'],[
            [NAMES[w],name,a['edges'],a['observed_mean'],a['predicted_mean'],a['conditional_MAE'],a['conditional_RMSE'],a['clipped_low_edges'],a['clipped_high_edges']] for w,x in ws.items() for name,a in x['magnitude'].items()]),
        'MAE/RMSE分别是对应符号边上的平均绝对误差/均方根误差（行业指标），不能将有益与有害边数直接当独立正例商品数。幅度头的额外价值仅比较U_R与U_C的预注册诊断，不擅自给U_C生成推荐。',
        '## 6. 合格边与正例候选存活',
        table(['窗口','方案','效用>0边数','B边数','N边数','H边数','B占比','N占比','H占比','B/H边数比'],[
            [NAMES[w],name,z['eligible_edges'],z['eligible_B'],z['eligible_N'],z['eligible_H'],z['B_share'],z['N_share'],z['H_share'],z['B_H_ratio']] for w,x in ws.items() for name,z in x['gate'].items()]),
        table(['窗口','方案','主正例候选数','至少有一条合格有益边的正例数','存活比例','严格冷存活','稀疏存活'],[
            [NAMES[w],name,z['main_positive_candidates'],z['surviving_positive_candidates'],z['survival_rate'],z['strict_surviving'],z['sparse_surviving']] for w,x in ws.items() for name,z in x['survival'].items() if name!='old_U']),
        table(['窗口','旧U主正例','旧U存活','旧U严格冷存活','旧U稀疏存活'],[[NAMES[w],x['survival']['old_U']['all']['positive_candidates'],x['survival']['old_U']['all']['any_beneficial_edge'],x['survival']['old_U']['strict']['any_beneficial_edge'],x['survival']['old_U']['sparse']['any_beneficial_edge']] for w,x in ws.items()]),
        '存活是匹配前至少一个有益位置通过门槛，不等于最终实际插入；分母是Cold50内、排除W0重合后的主正例，不是全目录冷真值。',
        '## 7. 最终推荐及暖侧风险',
        table(['窗口','角色','整体MAP','暖≥21 MAP','严格冷MAP','稀疏1–5 MAP','冷稀疏0–5 MAP'],[[NAMES[w],name,z['map12'],*[z['segments'][k]['map12'] for k in ['warm_21_plus','strict_cold','sparse1_5','all_cold_sparse']]] for w,x in ws.items() for name,z in [('W0',x['W0']),('R',x['R'])]]),
        table(['窗口','整体MAP差','暖组MAP差','严格冷MAP差','稀疏MAP差','全部冷稀疏MAP差'],[[NAMES[w],x['R']['delta_vs_w0'],*[x['R']['segments'][k]['delta_vs_w0'] for k in ['warm_21_plus','strict_cold','sparse1_5','all_cold_sparse']]] for w,x in ws.items()]),
        table(['窗口','全部用户','准入用户','覆盖率','替换总数','每准入用户替换数','每全部用户替换数','最大替换数','插入正例','移除正例','严格冷插入','稀疏插入'],[[NAMES[w],*[x['R']['admission'][k] for k in ['users','users_with_admission','coverage','replacements','mean_per_admitted','mean_per_all_users','max_replacements','inserted_cold_positives','removed_warm_positives','strict_positive_inserted','sparse_positive_inserted']]] for w,x in ws.items()]),
        table(['窗口','替换数桶','用户数','相同用户MAP差','插入正例','移除正例','净正例','实际AP差减单边AP差和'],[[NAMES[w],name,*[b[k] for k in ['users','map_delta','inserted_cold_positives','removed_warm_positives','net_positives','mean_exact_minus_single_sum']]] for w,x in ws.items() for name,b in x['R']['buckets'].items()]),
        '多件替换的AP不严格可加。最后一列是每桶用户的实际最终AP增量减去各执行边单独AP增量之和；桶间用户不同，不当作件数的随机因果效应。不以多件误差为理由追加max1。',
        '## 8. 新鲜商品偏好及历史丰富度',
        'novelty（项目新鲜商品接受度）为用户历史购买中购买日前商品全站交易≤5的事件占比；Q1–Q4按P4.2F同一历史用户—窗口四分位边界，低到高、同分落低桶。richness（项目历史丰富度）按过去交易数同一历史三分位切成low/medium/high。全部原切点复制，空桶保留，不根据外层标签重分桶。',
        table(['窗口','分组轴','分组','用户数','覆盖率','插入正例/替换次数','移除正例/替换次数','MAP差'],[[NAMES[w],axis,name,*[b[k] for k in ['users','coverage','cold_positive_insertion_rate','warm_positive_removal_rate','map_delta']]] for w,x in ws.items() for axis,groups in x['R']['mechanisms'].items() for name,b in groups.items()]),
        '## 9. 判定规则与核验边界',
        'Action supported：平均B对H AUC>G、至少3窗AUC不退化、平均PR≥G；否则只要平均AUC或PR任一提升为mixed，其余rejected。Policy supported：插入/移除比≥0.25、≥G的3倍、移除数<257；未全满足但比值提升且移除下降为mixed。两者都0时不把拒绝所有候选当precision成功。',
        'MAP safety supported：整体平均差≥−0.0005、最差窗≥−0.001、暖组平均差≥−0.0005；未通过但整体均值、最差窗及暖均值均严格优于G为mixed。只有action不是rejected、policy为supported且safety为supported/mixed时，扩量准备度才true。以上补足的mixed口径均先于拟合冻结。',
        f"独立复核{len(v['checks'])}项通过；全部外层模型输出重放、最终12件重建、最终AP与AUC/PR独立核验；另对{v['LP_graphs']}个有合格边的用户图用独立线性规划复核最优目标（每窗最多前16图），不夸大为全图独立求解证明。6项无拟合单元测试通过。",
        '历史行、特征值、抽样和权重在训练前完整重放；拟合记录及输入身份在训练后复核。正式实现、模型参数和合同固定后未调优。收尾报告生成器曾有一处列表括号语法错误，已修复；它不在正式计算路径中，未重训、重预测或改指标，记录在REPORT_ENGINEERING_NOTE.json。大型模型、行级概率/匹配/推荐留在Git忽略的artifacts目录。没有运行最终周2020-09-16、没有100%扩量、没有Warm-v2整合、没有P4.3、没有提交推送。完成后停止。']
    total_edges=sum(x['gate']['R']['eligible_edges'] for x in ws.values())
    total_b=sum(x['gate']['R']['eligible_B'] for x in ws.values())
    total_n=sum(x['gate']['R']['eligible_N'] for x in ws.values())
    total_h=sum(x['gate']['R']['eligible_H'] for x in ws.values())
    reps=sum(x['R']['admission']['replacements'] for x in ws.values())
    pbrate=[x['classifier']['classes']['B']['predicted_mean']/x['classifier']['classes']['B']['observed_rate'] for x in ws.values()]
    text += ['## 10. 本轮解释：保护改善，但冷机会再次被压制',
        '以下是测量后解释，不用于修改合同、分数、门槛或推荐。',
        f"有益/有害平均AUC从G的{vd['G_mean_auc']:.6f}升到{vd['mean_auc']:.6f}，平均PR从{vd['G_mean_PR']:.6f}升到{vd['mean_PR']:.6f}；3窗AUC不退化，因此操作信号supported。幅度头相对U_C四窗AUC都更高、3窗PR更高，支持其具有额外排序信息，但夏末PR更差，且没有U_C正式MAP对照，不能声称幅度头已证明改善最终推荐。",
        f"实际替换从G的27701次缩到{reps}次，移除正例从257降到{vd['removed']}，插入正例也从12降到{vd['inserted']}。MAP安全门槛通过主要伴随大幅减少修改，不能把‘接近不改原推荐’写成‘冷通道兑现成功’。插入/移除比{fmt(vd['ratio'])}虽高于12/257，但未达到0.25或3倍门槛，policy precision为mixed，扩量准备度false。",
        '平均p_B只有实际有益边比例的'+ '、'.join(f'{r:.1%}' for r in pbrate)+'（冬、春、初夏、夏末）。有害概率平均冬春及夏末偏低，初夏偏高。三类总体概率没有跨窗同时对齐；p_B≥0.1的固定分箱合计83条边、0条有益，说明总体低估与局部高分误判可以同时出现。边共享用户/商品，不能将83条当独立样本推断显著性，也不能靠全局乘系数保证修复决策尾部。',
        f"正式正效用放出{total_edges}条边，其中B={total_b}、N={total_n}、H={total_h}。仅夏末有2条有益边，归属同1个稀疏正例商品；最终它被插入。四窗264个主冷正例中263个没有任何合格有益边，160个严格冷正例全部未存活。因此当前主要机会损失在效用门槛前，不在匹配冲突后。与概率及幅度相乘的抑制模式一致，但不能据此断言‘只补校准就能解决’，本轮未试校准。",
        '恰好1次替换桶不再四窗全部为负：冬季0、春季负、初夏负、夏末正。实际AP增量与已执行单边AP增量之和的差在所有桶仅为浮点舍入量级；本轮失败不能归因于多件非加和。夏末插1删2仍有微小MAP净增，是位置和用户真值分母决定AP贡献，而非所有正例价值相同。',
        '新鲜偏好最高Q4并未稳定更安全：相对最低Q1的MAP差冬季相同、春季与初夏更差、夏末更好。只在夏末插入1个稀疏正例；不能把高新鲜偏好直接作为安全准入人群。',
        '结论范围：风险分离比直接L2回归减少了伤害，并改善主操作区分，但没有跨过“既安全又能捕获冷机会”的门槛。接下来的问题是预测概率与选择尾部的可靠性，而不是已经证明必须扩量或改匹配。若要继续，必须人工授权下一份限定成本的诊断/实验；本轮不启动校准、排名损失或100%训练。',
        '## 11. 用户27项问题答复索引',
        table(['序号','回答'],[
            [1,'第4节：B均值四窗低于观测，H初夏偏高其余偏低，三类不能视为均已校准。'],
            [2,'第4节固定10箱全表；p_B高度集中最低箱，较高箱仍有误判；不新增校准器。'],
            [3,f"平均B/H AUC {vd['G_mean_auc']:.6f}→{vd['mean_auc']:.6f}，PR {vd['G_mean_PR']:.6f}→{vd['mean_PR']:.6f}。"],
            [4,'U_R相对U_C四窗AUC更高，3窗PR更高，见第5节。'],
            [5,'提供额外操作排序信息，不代表已证明最终MAP价值；U_C没有正式推荐。'],
            [6,total_edges],[7,f'B={total_b}，N={total_n}，H={total_h}。'],
            [8,'只有夏末的合格边有益率高于G，其他三窗为0。'],[9,'四窗合格有害边均下降，合计11186→139。'],
            [10,'主正例264，存活1，四窗为0、0、0、1；263个在匹配前被挡住。'],[11,'严格冷存活0，稀疏存活1。'],
            [12,vd['inserted']],[13,vd['removed']],[14,vd['ratio']],[15,vd['ratio_improvement']],
            [16,'不是四窗皆负：冬0、春负、初夏负、夏末正。'],[17,'非加和差仅浮点舍入，本轮不是主要损失来源。'],
            [18,vd['mean_delta']],[19,vd['mean_warm_delta']],
            [20,'严格冷四窗差均0；稀疏仅夏末+0.00018076645，平均+0.00004519161。'],
            [21,'没有稳定证据；Q4对Q1为冬相同、春/初夏更差、夏末更好。'],
            [22,vd['risk_separation_action_signal']],[23,vd['risk_separation_policy_precision']],[24,vd['risk_separation_map_safety']],
            [25,vd['full_history_scaleup_allowed']],[26,'2020-09-16持续not_run。'],[27,'Warm-v2仍未整合，P4.3未启动。']])]
    (dest/'P4_2G_FINAL.md').write_text('\n\n'.join(text)+'\n',encoding='utf8')
    manifest(repo)

def manifest(repo):
    repo=Path(repo).resolve();dest=repo/'reports/phase4';root=repo/'artifacts/phase4'/RUN_ID
    paths=sorted([p for p in dest.glob('*') if p.name.lower().startswith(('p4_2g_','p4_2g.')) and p.name!='P4_2G_OUTPUT_MANIFEST.json'])
    write_json(dest/'P4_2G_OUTPUT_MANIFEST.json',dict(stage='P4.2G',status='completed',verification='pass',
        reports=[identity(p) for p in paths],source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42g*.py'))],
        tests=[identity(repo/'tests/test_p42g.py')],artifacts=[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
        final_week='not_run',Warm_v2_integrated=False,full_history_run=False,selected='W0',
        hash_purpose='explicit evidence-package requirement; bind outputs for future comparison'))

if __name__=='__main__':report(Path('.'))
