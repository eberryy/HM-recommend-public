"""Render only measured P4.2D evidence. Does not train or select a policy."""
from pathlib import Path
from datetime import datetime, timezone
from .p41a_contract import read_json,write_json,identity
from .p42d_contract import RUN_ID


def table(headers,rows):
    def cell(v):
        if v is None: return '不可用'
        if isinstance(v,float): return f'{v:.6g}'
        return str(v).replace('|','／').replace('\n',' ')
    return '\n\n'+'| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(cell,r))+' |\n' for r in rows)+'\n'


def report(repo):
    repo=Path(repo).resolve(); reports=repo/'reports/phase4'; root=repo/'artifacts/phase4'/RUN_ID
    m=read_json(reports/'P4_2D_metrics.json'); v=read_json(reports/'P4_2D_VERIFICATION.json')
    assert v['status'] in ('pass','pass_with_execution_boundary_exception') and len(v['windows'])==4
    windows=m['windows']; names=['冬季','春季','初夏','夏末']; ws=list(windows.values())
    write_json(reports/'p4_2d_position_ap_alignment.json',{'stage':'P4.2D',
        'windows':{key:{'correlations':w['pair']['alignment'],'slots':w['pair']['slots']} for key,w in windows.items()}})
    n=sum(w['census']['eligible']['all']['positives'] for w in ws)
    strict=sum(w['census']['eligible']['strict_cold']['positives'] for w in ws)
    total=sum(w['census']['full50']['all']['rows'] for w in ws)
    survive=sum(w['pair']['survival']['A_tau0']['all']['any_edge'] for w in ws)
    text='''# P4.2D：完整冷正例与实际准入尾部失败审计

## 1. 结论先行

正式审计是只读诊断，不晋级模型。数值求解不再是主要嫌疑；冻结的冷侧和暖侧预测、候选与原匹配全部复用，未训练、未重新预测、未重新匹配。另有执行范围例外：收尾误跑了包含合成拟合的旧回归测试，不满足全流程零优化器调用；详见第14节。

'''
    text+=f'当前完整操作范围中有 **{n}个独立冷正例候选**，其中严格冷{strict}个、稀疏{n-strict}个。最宽松阈值只有{survive}个正例存在合格边，{n-survive}个在匹配前就已被挡住；3个中最终有益插入2个，另1个在匹配中失去机会。**匹配不是主要瓶颈。**\n\n'
    text+='''更准确的定位是：现有绝对倾向分数和跨来源比较，在最终决策尾部没有形成可靠的收益优势，而不是B0完全没有排序信号。

但这不自动证明“换分解式模型”或“扩大10倍数据”就能解决：B0相对qC的条件选择优势只出现于冬春；两种概率差值替代分数虽更贴近AP数值幅度，却降低了有益/有害边区分AUC。本轮没有据此改变效用、阈值或训练规模。

## 2. 术语、分母与冻结边界

以下W0、B0、Cold50、qC/qW和诊断判定字段均为项目自定义命名：

- W0：冻结Warm-v1每用户原前12件商品及位置；Warm150是原暖路150候选集合，不代表其中每件商品都不是冷商品。
- B0：冻结冷侧专家，重排M4纯内容表示召回的200件商品；Cold50是B0原前50件，每个有候选用户恰好50件。
- qC、qW：分别估计Cold50和原Top12商品下一周被观察购买的概率倾向。不是曝光后购买率；没观察到购买不等于用户明确拒绝。
- 完整Cold50口径：包括与W0重合的候选，供校准、qC名次、极端尾部和用户机会审计。主冷正例口径：排除已在W0前12的商品后，未来一周实际购买的唯一“截止日—用户—商品”对。一个正例不能因为有12个可替换位置而算12次。
- 边（图优化通用概念）：主候选与同用户原暖位置的配对。有益边=插入真值、移除非真值；有害边相反；双方都是真值或都不是则中性。所有标签只用于离线审计。
- 严格冷商品：截止日前全站交易事件数0；稀疏商品：1–5次。不是冷用户。cold-only表示不在Warm150中；also_in_Warm150表示同时存在于Warm150。
- U=logit(qC)−logit(qW)，logit为赔率的自然对数；沿原方案先将概率裁到[1e-6,1−1e-6]。三个阈值τ固定为0、ln2、ln4，要求严格U>τ。诊断D1=qC−qW、D2=qC−2qW使用相同裁剪概率，但不匹配、不产生新推荐或正式MAP。
- AUC为行业区分指标：ROC-AUC比较随机正负样本的相对名次；本报告PR-AUC使用非插值平均精度。主配对指标只比较有益与有害边，中性边不进入其负类；次级才比较有益与其他所有边。不同分母的PR-AUC不能直接比高低。
- 正例候选Recall@K：分母是当前Cold50中全部主冷正例，不是目录全部冷真值。正例倒数名次均值给每个正例等权；条件MRR则只在完整Cold50有真值的用户中，取每人第一个真值的倒数名次再平均。多正例用户的条件Recall先除该用户的Cold50真值数，再对用户平均，两者不可混用。
- 名次比例rank_pct为名次/50；用户百分位percentile为同用户分数升序平均名次减1再除49；zscore为相对完整50件均值的差除总体标准差。标准化分差margin为相对原第2、第5或中位分数的差除该标准差。M4名次是原200候选中的粗排位置。
- 极端高分尾部为按原始qC概率排序的最高固定比例。全局同分按原规范行顺序；用户内同分按B0名次再商品编号。分箱按无标签名次切割，箱为互斥集合；不同top比例则嵌套，不能相加。
- precision是该候选集合内实际正例数/候选行数；lift为该率除完整Cold50正例率。表格里的概率和率默认0–1小数，标%时才乘100。
- 单次ΔAP@12是仅执行该边、其他原位置不动时的用户AP变化，分母仍为min(用户下一周不同真值数,12)。相关性Pearson比较线性数值关系，Spearman比较带平均同分名次的单调关系。大量边ΔAP=0，相关不等于尾部决策安全。
- 跨窗相加均为用户—窗口观察或候选—窗口对，不是跨窗去重自然人数。推断单位相互相关，未做独立边假设的显著性检验。

最终周从2020-09-16起封存。未来首售代理也只观察到2020-09-15；不读取最终周标签计算任何统计。原方案、模型及历史失败记录不改，Warm-v2与P4.3均未启动。

## 3. 全部候选与正例普查
'''
    text+=table(['窗口','完整50行数','完整正例','排除重合行','排除重合正例','主候选行','主正例','正例用户','严格冷正例','稀疏正例'],[
        [name,w['census']['full50']['all']['rows'],w['census']['full50']['all']['positives'],
         w['census']['excluded_overlap']['all']['rows'],w['census']['excluded_overlap']['all']['positives'],
         w['census']['eligible']['all']['rows'],w['census']['eligible']['all']['positives'],w['census']['eligible']['all']['positive_users'],
         w['census']['eligible']['strict_cold']['positives'],w['census']['eligible']['sparse1_5']['positives']] for name,w in zip(names,ws)])
    text+='完整正例逐窗38、55、105、78，精确对齐P4.2R3 qC校准；减少的3、4、0、5对全部是已在W0的重合正例。完整分群、用户数、正例率及来源拆分见 `p4_2d_cold_positive_census.json`。\n'
    text+=table(['窗口','主候选分组','候选行','候选用户','正例','正例用户','正例率'],[
        [name,k,d['rows'],d['users'],d['positives'],d['positive_users'],d['positive_rate']]
        for name,w in zip(names,ws) for k,d in w['census']['eligible'].items()])
    text+='\n## 4. B0原排序与qC排序\n'
    text+=table(['窗口','B0正例名次均值','B0中位名次','qC正例名次均值','变好数','不变数','变差数','qC减B0平均名次'],[
        [name,w['ranking']['positive_distributions']['b0_rank']['mean'],w['ranking']['positive_distributions']['b0_rank']['median'],
         w['ranking']['positive_distributions']['qc_rank']['mean'],w['ranking']['paired']['improved'],w['ranking']['paired']['same'],
         w['ranking']['paired']['worsened'],w['ranking']['paired']['rank_delta']['mean']] for name,w in zip(names,ws)])
    text+=table(['窗口','排序','Recall@1','@5','@10','@20','@50','正例倒数名次均值'],[
        [name,order,*[w['ranking'][order]['recall'][str(k)] for k in (1,5,10,20,50)],w['ranking'][order]['mean_reciprocal_positive_rank']]
        for name,w in zip(names,ws) for order in ('b0','qC')])
    text+='平均名次3窗变差，但中位名次差四窗均为0；正例倒数名次均值与用户条件MRR均是冬春变差、两个夏季变好。故“qC不稳定地改变了B0顺序”成立，“qC四窗全面洗坏B0”不成立。\n'
    text+=table(['窗口','分数（方向预先固定）','全候选ROC-AUC','PR-AUC'],[
        [name,k,d['roc_auc'],d['pr_auc']] for name,w in zip(names,ws) for k,d in w['ranking']['discrimination'].items()])
    text+='negative_b0_rank就是−B0名次；其余分数均越大越好。完整正例分位数及配对名次差保存在排名审计JSON；每个正例的所有指标在忽略目录的 `positive-candidates.parquet`。\n'
    text+='\n## 5. qC极端尾部与概率偏差\n'
    text+=table(['窗口','全局最高比例','候选行','正例','实测precision','相对全池lift','平均qC','B0平均名次','严格冷行','稀疏行'],[
        [name,f'{float(f)*100:g}%',d['rows'],d['positives'],d['positive_rate'],d['lift'],d['qC']['mean'],d['b0_rank']['mean'],d['strict_rows'],d['sparse_rows']]
        for name,w in zip(names,ws) for f,d in w['tails']['global'].items()])
    text+='\n最高0.01%四窗合计144行、0正例；它们的预测期望正例数合计约1.90。**0/144不是统计上证明真实概率为0**：即使按独立同分布近似，0命中的95%单侧上界也约2.06%，还未考虑用户相关性。因此机器“rejected”表示未满足预注册的观测尾部可靠性标准，不是显著否定概率模型。\n\n'
    text+='大多数候选平均预测偏低与极端尾部观测低于预测可以同时发生；但尾部正例很少，不能把“观测到这种模式”升级为已确证系统性过度自信。最高0.1%在冬春各命中2个，在两个夏季为0，尾部越极端并不保证命中率单调提高。\n'
    text+=table(['窗口','用户内qC前K','候选行','正例','precision','平均qC','B0中位名次'],[
        [name,k,d['rows'],d['positives'],d['positive_rate'],d['qC']['mean'],d['b0_rank']['median']]
        for name,w in zip(names,ws) for k,d in w['tails']['within_user'].items()])
    text+=table(['窗口','无标签百分位箱','候选行','正例','平均预测','实测率','预测/实测'],[
        [name,f"{d['low']*100:g}–{d['high']*100:g}%",d['rows'],d['positives'],d['qC']['mean'],d['positive_rate'],
         d['predicted_over_observed'] if d['ratio_status']=='finite' else d['ratio_status']]
        for name,w in zip(names,ws) for d in w['tails']['bins']])
    text+='infinite表示预测正但该箱0命中，undefined表示比值无定义；机器JSON用null加状态字段，不填虚构比值。\n'
    text+='\n## 6. 全配对效用、阈值与匹配损失\n'
    text+=table(['窗口','边类型','边数','U均值','p10','p25','中位数','p75','p90','p95','p99','最大'],[
        [name,k,d['rows'],*[d[x] for x in ('mean','p10','p25','median','p75','p90','p95','p99','max')]]
        for name,w in zip(names,ws) for k,d in w['pair']['utility_distributions'].items()])
    text+=table(['窗口','有益对有害ROC-AUC','PR-AUC','主比较边数','有益对所有其他ROC-AUC','次级PR-AUC'],[
        [name,w['pair']['comparison']['primary']['roc_auc'],w['pair']['comparison']['primary']['pr_auc'],
         w['pair']['comparison']['primary']['rows'],w['pair']['comparison']['secondary']['roc_auc'],w['pair']['comparison']['secondary']['pr_auc']]
        for name,w in zip(names,ws)])
    text+=table(['窗口','阈值方案','主正例','有任意合格边正例','有合格有益边正例','合格有益边数','最终有益插入','匹配丢失正例'],[
        [name,k,w['census']['eligible']['all']['positives'],w['pair']['survival'][k]['all']['any_edge'],
         d['beneficial_candidates_above_tau'],d['beneficial_edges_above_tau'],d['beneficial_candidates_selected'],d['lost_candidates']]
        for name,w in zip(names,ws) for k,d in w['pair']['losses'].items()])
    text+='A_tau0、M_tau_ln2、C_tau_ln4分别是原激进、中等、保守方案，不是新搜索。严格冷160个正例无一存在U>0的边；3个存活者都是稀疏商品。匹配只能在已过阈值的边里选，无法找回那261个。\n'
    text+=table(['窗口','方案','未选中合格有益边','仅冷节点冲突','仅暖节点冲突','两者冲突','均未占用','被更高权重非有益边占节点'],[
        [name,k,d['unselected_surviving_edges'],*[d['conflicts'][x] for x in ('same_cold_only','same_warm_only','both','neither','higher_weight_nonbeneficial_node')]]
        for name,w in zip(names,ws) for k,d in w['pair']['losses'].items()])
    text+='前四冲突列互斥，合计为未选中边数；最后一列可与它们重叠，不能再次相加。边冲突不等于独立候选丢失，同一候选选中另一个安全位置并不算丢候选。这里描述原最优匹配的节点占用，不表示求解器算错。\n'
    text+='全部正例的最大U、最佳暖位置及qW、真值限定的安全位置最大U，逐行写入正例表；严格/稀疏和B0名次1、2–5、6–10、11–20、21–50的完整阈值存活率见 `p4_2d_threshold_survival.json`。\n'
    text+='\n## 7. 实际执行尾部与暖侧保护\n'
    text+=table(['窗口','方案','执行边','有益','有害','中性','平均qC','平均qW','平均U','平均B0名次','平均暖位置'],[
        [name,k,d['all']['rows'],d['all']['beneficial'],d['all']['harmful'],d['all']['neutral'],
         *[d['all']['distributions'][x]['mean'] for x in ('qC','qW','utility','b0_cold_rank','warm_slot_rank')]]
        for name,w in zip(names,ws) for k,d in w['pair']['executed'].items()])
    text+=table(['窗口','暖侧口径','行数','真值数','真值率','相对原Top12基础率'],[
        [name,k,d['rows'],d['positives'],d['positive_rate'],d['rate_vs_full']]
        for name,w in zip(names,ws) for k,d in [('全池',w['warm']['full']),*[(f'最低{float(f)*100:g}%',d) for f,d in w['warm']['bottom'].items()]]])
    text+='qW确实能找到平均风险更低的删除候选，但低于平均不等于低于对应冷候选的真实收益。激进方案最终移除11个原正例、仅插入2个，这比暖侧全池AUC更直接反映实际替换质量。执行有益/有害边各自的qC、qW、U、原名次全分位数，及实际被删暖商品的真值率，见配对审计与暖侧审计JSON。\n'
    text+='\n## 8. 位置价值及两种只读替代分数\n'
    text+=table(['窗口','分数','有益对有害AUC','PR-AUC','与单次ΔAP Pearson','Spearman'],[
        [name,k,d['primary']['roc_auc'],d['primary']['pr_auc'],d['alignment']['pearson'],d['alignment']['spearman']]
        for name,w in zip(names,ws) for k,d in [('U',{'primary':w['pair']['comparison']['primary'],'alignment':w['pair']['alignment']['overall']}),*w['pair']['alternatives'].items()]])
    text+='U对有益/有害有全局区分度，但与实际AP幅度的相关性较弱。D1/D2的相关性四窗更高，主AUC却四窗更低，PR-AUC也没有一致占优，不能宣布“概率差一定更合理”。大量AP为0以及负损失幅度共同影响相关性，不能据相关系数推出一个可部署阈值。\n\n'
    text+='U可分解为冷分数减暖分数，同一组被选中冷/暖节点间的某些位置排列可同权；当前匹配不显式学习位置价值，因此最优U总和不等于最优AP。具体每个位置的有益/有害率、U均值、执行次数、单次ΔAP均值和分层相关性均保存在配对与位置审计中。这是结构性目标差异，但261个正例阈值前被挡住说明它不是本轮唯一或首要损失来源。\n'
    text+='\n## 9. 用户是否有机会，与有机会时选哪件\n'
    text+=table(['窗口','有Cold50用户','机会用户','机会率','max_qC AUC','max_qC PR-AUC','B0条件MRR','qC条件MRR'],[
        [name,w['incidence']['users'],w['incidence']['positive_users'],w['incidence']['incidence_rate'],
         w['incidence']['aggregates']['max_qC']['roc_auc'],w['incidence']['aggregates']['max_qC']['pr_auc'],
         w['incidence']['conditional']['B0']['mrr'],w['incidence']['conditional']['qC']['mrr']] for name,w in zip(names,ws)])
    text+=table(['窗口','固定用户汇总分数','AUC','PR-AUC'],[[name,k,d['roc_auc'],d['pr_auc']]
        for name,w in zip(names,ws) for k,d in w['incidence']['aggregates'].items()])
    text+='max_qC是用户50件概率最大值；sum_qC是总和，top5_sum_qC为前5总和；noisy_or_qC=1−∏(1−qC)，这里只是无训练汇总分数，不假设商品购买独立就把它当准确发生概率。其余两个是用户内B0标准分/相对中位数标准化分差的最大值。\n\n'
    text+='用户层max_qC四窗AUC约0.587–0.654，存在可用方向，但不比候选层AUC更高；不同标签、基础率和统计单位本身也不能直接断言“更容易预测”。B0条件MRR只在2窗更好，因此“用户是否有机会×B0条件选择”的分解式模型假设目前为弱支持，不能作为已证实答案。本轮未训练这种模型。\n'
    text+='\n## 10. 严格冷商品的未来首售代理\n'
    text+=table(['窗口','严格冷候选行','7天内首次卖出','第8–28天','第29–84天','更晚','封存前未观察到','可观察天数'],[
        [name,w['availability']['all_strict']['rows'],*[w['availability']['all_strict']['buckets'][k] for k in
        ('within_next_7d','8_28d','29_84d','over84d','not_observed_before_embargo')],w['availability']['observation_days']]
        for name,w in zip(names,ws)])
    text+='7天内首次卖出的候选占比约3.45%、2.35%、10.96%、21.21%。大量“历史零交易”候选在更晚才首次出现销售，支持明显的未来销售时点差异；**首次观察销售不是库存、可售或上架真值**，也可能只是需求少。不能据此把未售商品当确定不可用。\n\n'
    text+='第1–7天对应[cutoff,cutoff+7天)，其余桶也采用左闭右开区间；最后一桶为观察被截断，不叫“永不再销售”。初夏仅能观察84天、夏末仅28天，后面时间桶为结构性不可观察，不是证明那些商品之后不卖。跨窗不应直接比较这些晚期桶。严格冷真值及固定qC尾部、U>0候选的分桶同见代理JSON，代理从不进入模型输入。\n'
    text+='\n## 11. 历史100%训练可行性与证据限制\n'
    text+=table(['历史截止日','实际10% H用户','实际正例用户','实际正例候选','估计100% H用户','估计正例用户','估计正例候选'],[
        [t,d['exact_hash10']['users'],d['exact_hash10']['positive_users'],d['exact_hash10']['positive_rows'],
         d['estimated_hash100']['users'],d['estimated_hash100']['positive_users'],d['estimated_hash100']['positive_rows']] for t,d in m['scaleup'].items()])
    text+='估计列只是固定用户哈希10%样本乘10，不是实际生成100% M4/B0候选后的精确计数。正例按用户聚集，同一用户跨截止日也相关，不能对候选行套独立抽样置信区间。未启动候选再生成；没有测量全量训练能否改善尾部或校准，故本轮不建议直接投入100%训练。这里的not_justified是“现有证据不足以授权该成本”，不是“更多数据一定无效”。\n\n'
    text+='未来100%历史训练、固定10%外层评测在时间推荐上可以成立：训练日期及标签结束严格早于外层；无外层或未来标签进入训练；历史100%用户遵守同一H资格；外层哈希规则确定且不按冷真值选择；qC/qW每个历史截止日使用相同用户总体。同一用户早期训练、晚期评测允许，研究的是已有用户下的商品冷启动，不得宣传为新用户泛化。H仍继承离线“下周有任意购买”的评测名单条件，不能偷换成线上所有用户。\n'
    text+='\n## 12. 预注册诊断判定（非模型晋级）\n'
    definitions={
      'qc_tail_reliability':'qC极端高分尾部达到观测可靠性标准',
      'qc_preserves_b0_positive_order':'qC保留B0正例平均名次',
      'qw_removal_risk_model':'qW能够识别平均更低风险的原商品',
      'logodds_pair_utility_alignment':'U有方向性有益/有害区分与正相关，非强幅度拟合',
      'matching_is_primary_bottleneck':'多数主正例是否主要在匹配阶段丢失',
      'user_incidence_factorization_hypothesis':'用户机会预测与B0条件选择分解的联合支持',
      'conditional_b0_choice_hypothesis':'B0条件MRR至少3窗优于qC',
      'strict_cold_availability_asymmetry':'首售时点代理差异，非真实库存证明',
      'full_history_training_scaleup':'当前是否有充分证据承担历史全量训练成本'}
    text+=table(['判定字段','中文解释','结果'],[[k,definitions[k],x] for k,x in m['verdicts'].items()])
    text+='这些是本项目计算前登记的描述性规则，supported=满足规则，weak=证据混合，rejected=未达到规则；不是统计显著性或因果结论。U的supported只要求方向一致，不得解释为实际准入已可靠。完整规则见合同，未根据有利窗口改门槛。\n'
    text+='\n## 13. 22项问题答复索引\n'
    answers=[
      ('主冷正例多少？',f'{n}，逐窗35、51、105、73。'),('严格冷与稀疏？',f'{strict}和{n-strict}，见第3节。'),
      ('B0通常排第几？','逐窗平均及中位数见第4节；不是只看Top10。'),('qC名次/MRR变好还是变坏？','平均名次3窗变差；MRR冬春降、两夏升。'),
      ('最高1%、0.1%、0.01%精度？','第5节全部固定比例逐窗列行数、正例、precision。'),('主体低估+尾部高估？','有该观测模式，极端箱样本少，不能作显著系统性结论。'),
      ('有益U分布？','第6节完整分位数。'),('有害U分布？','同表单列，不能与中性混淆。'),('有益对有害AUC/PR？','第6节主比较与次级分开。'),
      ('连τ0都不过的正例？','261/264。'),('过有益门槛又在匹配丢失？','3个中1个；另外2个插入。'),('匹配主瓶颈？','不是，98.86%正例此前已被挡住。'),
      ('qW找到较弱暖候选了吗？','找到低于平均风险的尾部，但不足以使替换净获益。'),('U与AP对齐？','方向正相关但幅度相关弱、未包含位置价值。'),
      ('差值效用更合理吗？','相关性更高，主AUC更低，不能直接替换。'),('用户机会更易预测？','有信号但不能跨标签单位比较难度，尚未证明更易。'),
      ('B0条件选择更强？','只有冬春，四窗不一致。'),('支持分解模型？','联合假设weak，不能直接启动。'),
      ('首售代理不对称？','是未来时点描述性差异，不是真实可售性；末期右删失。'),('扩大训练值得吗？','只有10倍数量估计，无解决尾部的证据，当前not_justified。'),
      ('最终周？','not_run，首售代理也排除。'),('Warm-v2？','未整合，P4.3未启动。')]
    text+=table(['序号','问题','回答'],[[i+1,*a] for i,a in enumerate(answers)])
    text+='\n## 14. 执行、验证、成本与停止\n'
    text+=f"合同先于正式计算登记。四窗审计及首次收尾耗时{m['seconds']:.2f}秒；最后一项历史规模SQL误用保留字rows而停止，原失败保存在 `FAILURE_attempt01.json`。仅修复该别名、复用已完成四窗，收尾{m.get('resume_seconds',0):.2f}秒；没有重训或重算四窗。独立复核{v['seconds']:.2f}秒，22项通过，131条可信源文件比较通过，24个旧实现文件未变。进程峰值内存本次未捕获，不补造。计算时间不含阅读、实现、报告。\n\n"
    text+='执行边界例外：收尾额外运行旧 `test_p42*.py` 回归套件，122项通过、2.924秒，但其中包含合成数据上的逻辑回归拟合与优化器调用。没有接触真实项目训练数据，也没有改动qC/qW；不过全流程“no optimizer call”不能判通过。22项数据核查的通过范围仅限正式只读审计，不覆盖该额外测试进程。证据记录在 `P4_2D_EXECUTION_BOUNDARY.json`，总体复核状态明确为 `pass_with_execution_boundary_exception`（数据通过但存在执行范围例外），不隐瞒或以测试名义豁免。\n\n'
    text+='本阶段专属8项测试不拟合模型；AP公式已对4096种原Top12命中模式、每个位置和两种候选标签穷举测试；复核另外对全部17,046,240条边逐位置重算AP差，并用独立指标库核对配对AUC/PR。正式审计只按保存匹配重建核对原推荐，没有运行匹配求解器。所有诊断不产生新正式MAP，W0仍保留。\n\n'
    text+='入口：`python -m hm_recsys.p42d run`（拒绝覆盖已有运行）、`python -m hm_recsys.p42d_verify`。继续使用原hm-recommend conda环境，无新依赖。逐行用户与商品证据留在忽略目录，报告/JSON只含汇总或不可直接识别的内部索引。\n\n'
    text+='本轮到此停止，等待人工决策。没有训练分解式模型，没有修改qC/qW、utility或阈值，没有扩大100%训练，没有最终周、P4.3、Warm-v2整合，也没有提交推送。\n'
    (reports/'P4_2D_FINAL.md').write_text(text,encoding='utf-8')
    manifest={'stage':'P4.2D','run_id':RUN_ID,'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'verification_status':v['status'],'selected_variant':'W0','final_week':'not_run','new_project_model_fits':0,
       'synthetic_regression_optimizer_exception':True,
       'reports':[identity(p) for p in sorted(reports.iterdir()) if p.is_file() and p.name.lower().startswith(('p4_2d_')) and p.name!='P4_2D_OUTPUT_MANIFEST.json'],
       'sources':[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42d*.py'))]+[identity(repo/'tests/test_p42d.py')],
       'artifacts':[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
       'reused_source_manifest':identity(reports/'P4_2R3_OUTPUT_MANIFEST.json'),
       'hash_purpose':'explicit evidence-package contract; trusted R3 comparisons in input receipt, not authenticity from self hashes'}
    write_json(reports/'P4_2D_OUTPUT_MANIFEST.json',manifest)
    return text


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--repo',default='.')
    report(p.parse_args().repo)
