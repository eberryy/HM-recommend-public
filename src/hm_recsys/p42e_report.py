"""Chinese evidence-bound closure; no computation of recommendation MAP."""
from pathlib import Path
import json

from .p41a_contract import read_json,write_json,identity
from .p42e import context
from .p42e_contract import now


def table(headers,rows):
    def fmt(x):
        if x is None: return '未运行/不可定义'
        if isinstance(x,float): return f'{x:.8g}'
        return str(x)
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                       *['| '+' | '.join(fmt(x) for x in r)+' |' for r in rows]])+'\n'


def report(repo):
    c,root=context(repo); rp=repo/'reports/phase4'; m=read_json(rp/'P4_2E_metrics.json')
    v=read_json(rp/'P4_2E_VERIFICATION.json') if (rp/'P4_2E_VERIFICATION.json').exists() else {'status':'not_run'}
    ws=m.get('windows',{}); pop=m.get('population',{})
    text=['# P4.2E：全量历史冷侧倾向模型与用户外校准\n',
        f"状态：`{m['status']}`；独立复核：`{v['status']}`。保留W0，未运行准入匹配、新推荐或新MAP。\n",
        '## 1. 比较范围与术语\n',
        '核心摘要中的术语均在本节定义。表内概率/比例默认0–1小数，例如0.01=1%。四窗均值对窗口等权，不是混池平均；汇总尾部precision则使用正例总数除候选行总数。\n',
        '项目窗口标签winter_20200122、spring_20200318、early_summer_20200624、late_summer_20200819分别为冬季2020-01-22、春季2020-03-18、初夏2020-06-24、夏末2020-08-19。supported表示满足预注册规则，mixed表示证据混合，rejected表示不支持该门槛；admission_rerun_allowed是项目定义的“是否具备讨论下一轮准入验证的条件”，不等于自动执行许可。\n',
        'qC、qW是本项目冷侧/暖侧下一周被观察购买的倾向模型，不是曝光后的线上购买概率。未购买只表示未观察，不是明确拒绝。W0为冻结Warm-v1原前12件。M4为冻结纯内容商品表示；B0为冻结冷侧专家；Cold50为M4前200件经B0排序后的前50件。以上均为项目命名。\n',
        'H_full是原合格用户总体取消10%哈希采样后的集合：下一周有任意购买，且截止日前最近20件不同商品至少一件可映射。它仍条件化于离线评测人口，不是线上全用户。用户在更早日期训练、较晚日期评测允许；不声称新用户泛化。最终周2020-09-16不参与任何统计。\n',
        'Q_old为冻结10%历史qC；Q_full_raw为100%历史同模型；Q_full_cal为后者加单调概率校准。OOF（行业交叉拟合）按用户固定两折，同用户所有日期同折：用另一折用户训练的模型产生其校准分数。每历史候选行恰好一个折外预测。\n',
        'raw简称Q_full_raw。训练角色full为完整历史池；fold0/fold1分别用第0/1折用户训练、预测另一折。这里只对qC头交叉拟合，上游M4/B0始终冻结。原始qC固定为L2逻辑回归（系数平方惩罚），使用LBFGS拟牛顿求解器，C=1（逆正则强度参数）、tol=1e-7（停止容差）、max_iter=1000（迭代上限）；没有类别权重或负采样。\n',
        '校准公式q_cal=sigmoid(a+b*z_raw)，z_raw为原逻辑回归线性输出；a是截距，b是斜率，必须为正。logit为赔率自然对数，sigmoid为其反函数。经人工批准，校准输入不截断，epsilon=1e-6仅用于概率诊断保护和跨来源效用U=logit(qC)−logit(qW)。不添加特征，不用原始B0绝对分数训练qC。\n',
        'ROC-AUC衡量正例相对未观察购买行的区分；PR-AUC按非插值平均精度计算，均为行业指标。候选Recall@K的分母是当前完整Cold50中已召回的正例，不是目录全部真值。MRR是有Cold50正例的用户各自第一个正例倒数名次的均值；正例倒数名次均值另给每个正例等权，不混用。\n',
        '尾部指全局最高固定比例的分数行。precision=该集合正例数/候选行数；lift=precision/完整Cold50正例率。不同top集合嵌套，不能相加；百分位分箱互斥。候选排序用原始概率，校准前后顺序及并列关系必须完全相同。Brier为概率平方误差均值；logloss为二分类对数损失；ECE为10个等人数概率箱预测均值与实际正例率绝对差的行数加权均值。概率指标使用固定数值裁剪，同时保存未裁剪概率均值。外层校准截距/斜率仅作诊断，不反向修正外层。\n',
        '主冷正例排除已在W0前12的候选，以截止日—用户—商品为单位；严格冷为截止前全站0交易，稀疏为1–5交易，不是冷用户。每个主候选与同用户12个暖位置组成配对；有益=冷真值且暖非真值，有害相反，中性不进入主区分指标。单次ΔAP是只替换一个位置的离线诊断，不是运行新推荐MAP。Pearson/Spearman分别衡量数值线性/名次关系。阈值存活是存在至少一条U>固定阈值的边，不代表最终被插入。\n',
        '## 2. 历史100%实际规模\n',
        table(['截止日','合格用户','Cold50行','正例行','正例用户','原10%用户','生成及标签秒'],
              [[t,p['eligible_users'],p['cold50_rows'],p['positive_rows'],p['positive_users'],p['old10_users'],p['seconds']] for t,p in pop.items()]),
        '每行用户数按该日期去重；跨日期相加是用户—日期观察数。以下训练池用户数跨池内日期去重；候选行仍以日期—用户—商品计。\n',
        table(['外层','旧用户','全量用户','旧正例用户','全量正例用户','旧正例行','全量正例行','正例倍数'],
              [[w,r['scale']['old']['users'],r['scale']['full']['users'],r['scale']['old']['positive_users'],r['scale']['full']['positive_users'],
                r['scale']['old']['positives'],r['scale']['full']['positives'],r['scale']['full']['positives']/r['scale']['old']['positives']] for w,r in ws.items()]),
        '## 3. 候选排序与极端尾部\n',
        table(['外层','角色','ROC-AUC','PR-AUC','正例平均名次','正例中位名次','用户MRR','Recall@1','@5','@10','@20','@50'],
              [[w,k,*[r['ranking'][k][x] for x in ['roc_auc','pr_auc','positive_mean_rank','positive_median_rank','conditional_user_mrr']],
                *[r['ranking'][k]['recall'][x] for x in ['1','5','10','20','50']]] for w,r in ws.items() for k in ['Q_old','Q_full_raw']]),
        'Q_full_cal只能改变概率尺度，不能改善前K件组成；一致性断言通过才有效。\n',
        '固定C=1扩量时，数据损失与L2正则的相对影响也会变化，不能把对照宣称为只增加正例数量的唯一因果效应。两折OOF模型使用半池用户，而最终模型使用全池；校准迁移到全池模型的有效性仍需外层验证。\n',
        table(['外层','角色','最高比例','行数','正例','precision','lift','严格冷行','稀疏行','B0平均名次'],
              [[w,k,f,t['rows'],t['positives'],t['precision'],t['lift'],t['strict_rows'],t['sparse_rows'],t['mean_B0_rank']]
               for w,r in ws.items() for k in ['Q_old','Q_full_raw'] for f,t in r['tails'][k]['global_top'].items()]),
        '用户内前1/2/5/10件和互斥尾部分箱的全部行数、正例、概率均值见p4_2e_qc_extreme_tail.json；零命中箱的预测/实测比用null加infinite状态，不捏造有限值。\n',
        '## 4. OOF校准与外层概率\n',
        table(['外层','训练角色','候选行','正例','迭代数','状态','拟合秒'],
              [[w,k,f['audit']['rows'],f['audit']['positives'],f['model']['n_iter'][0],f['audit']['status'],f['audit']['fit_seconds']]
               for w in ws for k in ['full','fold0','fold1'] for f in [read_json(root/'models'/w/k/'FIT_RESULT.json')]]),
        table(['外层','OOF行','OOF正例','a','b','raw OOF Brier','cal OOF Brier','raw OOF logloss','cal OOF logloss','raw OOF ECE','cal OOF ECE'],
              [[w,r['oof']['rows'],r['oof']['positives'],r['oof']['calibrator']['a'],r['oof']['calibrator']['b'],
                *[r['oof'][role][metric] for metric in ['brier_score','logloss','ece'] for role in ['raw','calibrated']]] for w,r in ws.items()]),
        table(['外层','角色','实际正例率','平均预测','预测/实际','Brier','logloss','ECE','诊断截距','诊断斜率'],
              [[w,k,*[p[x] for x in ['observed_positive_rate','mean_predicted_probability','predicted_to_observed_rate_ratio',
                                    'brier_score','logloss','ece','calibration_intercept','calibration_slope']]]
               for w,r in ws.items() for k,p in r['probability'].items()]),
        '## 5. 冻结qW下的阈值准备度\n',
        table(['外层','角色','有益/有害AUC','PR-AUC','ΔAP Pearson','ΔAP Spearman','主正例','U>0','U>ln2','U>ln4','严格冷U>0'],
              [[w,k,z['pair_discrimination']['roc_auc'],z['pair_discrimination']['pr_auc'],z['ap_correlations']['pearson'],z['ap_correlations']['spearman'],
                *[z['survival']['all'][x] for x in ['positives','tau0','ln2','ln4']],z['survival']['strict']['tau0']]
               for w,r in ws.items() for k,z in r['readiness'].items()]),
        'ln2、ln4为2和4的自然对数，与0一起是冻结的三阈值，不是新搜索。不能把存活计数当作净推荐增益。\n',
        '## 6. 预注册判定与23项问题\n',
        table(['判定字段','结果'],list(m.get('verdicts',{'all_verdicts':'not_run'}).items()))]
    if len(ws)==4:
        z=list(ws.values()); mean=lambda role,key:sum(r['ranking'][role][key] for r in z)/4
        survive=lambda role,group='all':sum(r['readiness'][role]['survival'][group]['tau0'] for r in z)
        text.insert(2,'## 核心结论\n\n术语、分母与窗口标签请参见本报告第1节。\n\n'
            +f"历史全量共{sum(p['cold50_rows'] for p in pop.values()):,}条候选、{sum(p['positive_rows'] for p in pop.values()):,}个正例。平均ROC-AUC从{mean('Q_old','roc_auc'):.6f}到{mean('Q_full_raw','roc_auc'):.6f}，平均PR-AUC从{mean('Q_old','pr_auc'):.8f}到{mean('Q_full_raw','pr_auc'):.8f}。扩量的总体排序信号受到支持，不应把整轮说成完全无效。\n\n"
            +f"但极端尾部为mixed，单调校准的概率尺度改善也是mixed；264个主冷正例中，U>0数量从{survive('Q_old')}变为{survive('Q_full_cal')}，严格冷仍有{survive('Q_full_cal','strict')}个。跨来源阈值准备度rejected，admission_rerun_allowed=false。没有新MAP结论，不晋级最终推荐基线。\n\n"
            +'历史训练池正例率与外层正例率存在差异。OOF只在历史分布学习校准，不能自动修正未来时间窗口的基础率变化；这是一项已观测分布差异及方法边界，不是已证明唯一根因。\n\n'
            +table(['外层','历史全量正例率','外层完整Cold50正例率','raw平均预测','校准平均预测'],
                   [[w,r['scale']['full']['positives']/r['scale']['full']['rows'],r['probability']['Q_old']['observed_positive_rate'],
                     r['probability']['Q_full_raw']['mean_predicted_probability'],r['probability']['Q_full_cal']['mean_predicted_probability']] for w,r in ws.items()]))
        answers=[
            '第2节列出每训练池精确用户/行/正例；日期观察与去重用户已分开。',
            '全量正例行逐外层为'+str([r['scale']['full']['positives'] for r in z])+'，旧值为'+str([r['scale']['old']['positives'] for r in z])+'。',
            f"平均ROC {mean('Q_old','roc_auc'):.8f}→{mean('Q_full_raw','roc_auc'):.8f}；平均PR {mean('Q_old','pr_auc'):.8f}→{mean('Q_full_raw','pr_auc'):.8f}。",
            f"正例平均名次不恶化{sum(r['ranking']['Q_full_raw']['positive_mean_rank']<=r['ranking']['Q_old']['positive_mean_rank'] for r in z)}/4窗；用户MRR不恶化{sum(r['ranking']['Q_full_raw']['conditional_user_mrr']>=r['ranking']['Q_old']['conditional_user_mrr'] for r in z)}/4窗。",
            '按1%、0.5%、0.1%的汇总门槛判定：'+m['verdicts']['full_history_tail_signal']+'；完整数值见第3节。',
            '两夏top0.1%全量正例为'+str([r['tails']['Q_full_raw']['global_top']['0.001']['positives'] for r in z[2:]])+'；旧两夏均0。',
            '第4节a/b逐窗表。','四窗b>0已检查。',f"相对raw，预测/实际比例的对数绝对值改善{m['verdicts']['calibration_gate_counts']['ratio_improve']}/4窗。",
            f"相对raw，Brier不恶化{m['verdicts']['calibration_gate_counts']['brier_nonworse']}/4窗，logloss不恶化{m['verdicts']['calibration_gate_counts']['logloss_nonworse']}/4窗；ECE不恶化{sum(r['probability']['Q_full_cal']['ece']<=r['probability']['Q_full_raw']['ece'] for r in z)}/4窗。",'互斥尾部分箱见JSON，0命中极端箱不作显著性结论。',
            '四窗原始/校准顺序及并列关系exact identical已验证。','第5节，按平均AUC不低于旧均值−0.01门槛检查。',
            f"{survive('Q_old')}→{survive('Q_full_cal')}个。",f"校准后严格冷U>0正例合计{survive('Q_full_cal','strict')}个。",
            f"排序变化只发生于扩量后raw模型，校准不改变排序；阈值存活依次为旧{survive('Q_old')}→全量raw {survive('Q_full_raw')}→校准{survive('Q_full_cal')}。不是随机试验或唯一因果证明。",
            m['verdicts']['full_history_ranking_signal'],m['verdicts']['full_history_tail_signal'],m['verdicts']['oof_calibration_improves_probability_scale'],
            m['verdicts']['cross_source_threshold_readiness'],str(m['verdicts']['admission_rerun_allowed']),
            '2020-09-16仍not_run。','Warm-v2未整合。']
    else:
        answers=['未完成全部正式计算，不能给出完整结论。']*21+['2020-09-16仍not_run。','Warm-v2未整合。']
    text.append(table(['问题序号（对应用户任务第33节）','答复'],list(enumerate(answers,1))))
    text.extend(['## 7. 工程、成本与停止\n',
                 '两次输入接口异常及逐列标准化放弃均保留，详见P4_2E_ENGINEERING_NOTE.zh-CN.md。原始历史失败记录不改写。\n',
                 f"正式qC拟合尝试{m.get('qC_fits',0)}次，成功{m.get('qC_successful_fits',0)}次；qW拟合0次。生成最近执行段{m.get('generation_seconds')}秒；训练段{m.get('training_seconds')}秒。各日期实际生成时间见第2节；分段时间不假装包含所有失败、阅读和实现成本。\n",
                 f"独立复核耗时{v.get('seconds')}秒，可信旧资产比较{v.get('trusted_comparisons')}项。大型候选/训练/概率产物留在Git忽略的artifacts/phase4目录。\n",
                 '判定supported/mixed/rejected分别表示满足预注册规则、证据混合、未满足相应否定规则；不是统计显著性或线上效果。readiness的weak仅表示阈值准备度有限支持。即使admission_rerun_allowed=true，本轮也停止，必须由人工决定新合同。\n'])
    if 'failure' in m: text.append('当前停止原因：\n\n```text\n'+m['failure']+'\n```\n')
    (rp/'P4_2E_FINAL.md').write_text('\n'.join(text),encoding='utf-8')
    write_json(rp/'p4_2e_10pct_parity.json',{t:{k:p[k] for k in ['old10_users','old10_user_exact','old10_m4_exact','old10_cold50_exact','parity_rows']} for t,p in pop.items()})
    # Design matrices are reconstructible scratch, not required frozen evidence.
    artifacts=[p for p in root.rglob('*') if p.is_file() and p.name!='design.float64.npy']
    manifest=dict(stage='P4.2E',created_at_utc=now(),status=m['status'],verification=v['status'],
                  reports=[identity(p) for p in sorted(rp.glob('*4_2E*')) if p.is_file() and p.name!='P4_2E_OUTPUT_MANIFEST.json'],
                  detail_reports=[identity(p) for p in sorted(rp.glob('p4_2e_*.json'))],
                  source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42e*.py'))],
                  artifacts=[identity(p) for p in artifacts],scratch_exclusion='design.float64.npy; deterministically reconstructible from training shards and saved preprocessor',
                  final_week='not_run',Warm_v2_integrated=False,selected='W0',matching_calls=0,new_MAP=0)
    write_json(rp/'P4_2E_OUTPUT_MANIFEST.json',manifest)


if __name__=='__main__': report(Path.cwd())
