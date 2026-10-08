"""Render verified P4.2H aggregate evidence; never fit or choose a new policy."""
from pathlib import Path

import numpy as np

from .p41a_contract import read_json, write_json, identity
from .p42h_contract import RUN_ID


NAMES = {'winter_20200122': '冬季', 'spring_20200318': '春季',
         'early_summer_20200624': '初夏', 'late_summer_20200819': '夏末'}
VARIANT = 'H_conditional_BH'
SEGMENTS = ('warm_21_plus', 'strict_cold', 'sparse1_5', 'all_cold_sparse')
VERDICTS = ('conditional_bh_action_signal', 'conditional_bh_survival',
            'conditional_bh_policy_precision', 'conditional_bh_map_safety')


def fmt(value):
    if value is None:
        return '不可用'
    if isinstance(value, (float, np.floating)):
        return f'{value:.10g}'
    return str(value).replace('|', '\\|').replace('\n', '；')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(fmt(value) for value in row)+' |' for row in rows])


def _window_values(windows, fn):
    return '；'.join(f'{NAMES[w]} {fmt(fn(x))}' for w, x in windows.items())


def _action_rows(windows):
    for w, x in windows.items():
        candidates = [('G_BH条件净收益', x['action']['G_BH']),
                      ('U_H正式权重', x['action']['U_H']),
                      ('G_raw只读', x['diagnostics']['D1_raw_q']['action']),
                      ('q_cal只读', x['diagnostics']['D2_classification_only']),
                      ('P4.2G U_R', x['references']['G_R']),
                      ('P4.2F G', x['references']['F_G']),
                      ('P4.2R3旧U', x['references']['old_U'])]
        for name, action in candidates:
            bh = action['beneficial_vs_harmful']
            corr = action.get('correlation', {})
            yield [NAMES[w], name, bh['rows'], bh['roc_auc'], bh['pr_auc'],
                   corr.get('pearson'), corr.get('spearman')]


def _diagnostic_summary(windows, verdict):
    improve = {metric: sum(x['q_calibration']['q_cal'][metric] <
                            x['q_calibration']['q_raw'][metric] for x in windows.values())
               for metric in ('Brier', 'logloss', 'ECE')}
    q_auc = sum(x['q_calibration']['q_cal_discrimination']['roc_auc'] >=
                x['references']['G_R']['beneficial_vs_harmful']['roc_auc'] for x in windows.values())
    q_mean_auc = float(np.mean([x['q_calibration']['q_cal_discrimination']['roc_auc'] for x in windows.values()]))
    q_mean_pr = float(np.mean([x['q_calibration']['q_cal_discrimination']['pr_auc'] for x in windows.values()]))
    separated_windows = sum(x['conditional_distributions']['B']['q_cal']['max'] <
                            x['conditional_distributions']['B']['implied_q_threshold']['min']
                            for x in windows.values())
    mag_auc = sum(x['action']['G_BH']['beneficial_vs_harmful']['roc_auc'] >=
                  x['diagnostics']['D2_classification_only']['beneficial_vs_harmful']['roc_auc']
                  for x in windows.values())
    mag_pr = sum(x['action']['G_BH']['beneficial_vs_harmful']['pr_auc'] >=
                 x['diagnostics']['D2_classification_only']['beneficial_vs_harmful']['pr_auc']
                 for x in windows.values())
    r_auc = sum(x['action']['U_H']['beneficial_vs_harmful']['roc_auc'] >=
                x['action']['G_BH']['beneficial_vs_harmful']['roc_auc'] for x in windows.values())
    raw_survival = sum(x['survival']['raw']['surviving_positive_candidates'] for x in windows.values())
    main = sum(x['survival']['H']['main_positive_candidates'] for x in windows.values())
    one = _window_values(windows, lambda x: x[VARIANT]['buckets']['1']['map_delta'])
    novelty_better = sum(
        x[VARIANT]['mechanisms']['novelty']['Q4']['map_delta'] >
        x[VARIANT]['mechanisms']['novelty']['Q1']['map_delta']
        for x in windows.values()
        if x[VARIANT]['mechanisms']['novelty']['Q4']['map_delta'] is not None and
           x[VARIANT]['mechanisms']['novelty']['Q1']['map_delta'] is not None)
    additive = max(abs(bucket['mean_exact_minus_single_sum'])
                   for x in windows.values() for bucket in x[VARIANT]['buckets'].values()
                   if bucket['mean_exact_minus_single_sum'] is not None)
    notes = [
        ('A. q ranking failure（条件分类排序不足）',
         f'q_cal的B对H AUC在{q_auc}/4窗不低于P4.2G U_R；正式U_H的平均AUC为{fmt(verdict["mean_auc"])}，'
         f'参照为{fmt(verdict["G_R_mean_auc"])}，action判定{verdict[VERDICTS[0]]}。'
         f'q_cal四窗平均AUC={fmt(q_mean_auc)}、PR={fmt(q_mean_pr)}；P4.2G U_R平均AUC='
         f'{fmt(verdict["G_R_mean_auc"])}、PR={fmt(verdict["G_R_mean_PR"])}。'
         'q是纯分类分数、U_R含幅度与非零概率，比较可定位排序差异，但不能将两者差别全部归因于分类器。'),
        ('B. q probability calibration failure（条件概率校准不足）',
         f'历史OOF Platt应用到外层后，Brier、logloss、ECE分别在{improve["Brier"]}、'
         f'{improve["logloss"]}、{improve["ECE"]}个窗口降低。'
         '应结合第4节最高概率桶的实际B率判断尾部是否可信；总体均值接近实际率不能证明准入尾部可靠。'
         '外层诊断截距/斜率只描述失真，不回流正式模型。'),
        ('C. magnitude mismatch（有益/有害幅度失配）',
         f'G_BH相对只用q_cal的AUC有{mag_auc}/4窗不退化，PR有{mag_pr}/4窗不退化。'
         '幅度头完全复用P4.2G，条件MAE/RMSE见第5节；分类区分与AP价值排序目标不同，'
         '不能仅凭某个AUC下降认定幅度模型必错，也没有无幅度正式MAP对照。'),
        ('D. non-neutral weighting r mismatch（非零概率权重失配）',
         f'U_H相对G_BH的B对H AUC有{r_auc}/4窗不退化；同用户top1重合均值：'+
         _window_values(windows, lambda x: x['diagnostics']['D3_no_r_priority']['top_overlap']['1']['mean_user_overlap'])+
         '。r确实可能改变优先级，但不改变准入集合；本轮未生成不乘r的正式推荐，不能声称已识别r对最终MAP的因果作用。'),
        ('E. conditional formulation still too conservative（条件分解仍过于保守）',
         f'主冷正例共{main}对，raw-q存活{raw_survival}对，正式校准后存活{verdict["surviving"]}对，'
         f'严格冷存活{verdict["strict_surviving"]}对。P4.2G为1对；本轮survival判定{verdict[VERDICTS[1]]}。'
         f'{separated_windows}/4窗真实B边上的最大q_cal仍低于真实B边上的最小内生阈值，详见第5节极值分离表。'
         '未存活的主正例是在匹配前没有任何真实有益槽位通过门槛，不是被匹配竞争挤掉。'),
        ('F. conditional formulation becomes unsafe（条件分解造成不安全准入）',
         f'累计插入{verdict["inserted"]}个冷侧正例、移除{verdict["removed"]}个原推荐正例；'
         f'整体平均MAP差{fmt(verdict["mean_delta"])}，暖组平均差{fmt(verdict["mean_warm_delta"])}，'
         f'最差窗{fmt(verdict["worst_delta"])}，安全判定{verdict[VERDICTS[3]]}。'
         f'恰好替换1件的用户桶MAP差：{one}。'
         '单件桶若已为负，说明并非只有多件替换才会伤害，但各桶用户不同，不能作随机对照解释。')]
    blocked = []
    if verdict[VERDICTS[0]] == 'rejected':
        blocked.append('A/C/D操作排序链未达到参照；仅凭本轮不能进一步唯一归因')
    if verdict[VERDICTS[1]] == 'rejected':
        blocked.append('E：正例在准入门槛前仍未充分存活')
    if verdict[VERDICTS[2]] == 'rejected':
        blocked.append('B/C/D形成的选择尾部精度不足；概率、幅度和权重仍需区分')
    if verdict[VERDICTS[3]] != 'supported':
        blocked.append('F：最终整体或暖侧安全门槛未通过')
    primary = '；'.join(blocked) if blocked else '没有触发禁止扩量的已注册条件；这仍不是模型晋级或扩量执行授权'
    return dict(notes=notes, calibration_improved_windows=improve,
                q_auc_nondegrade=q_auc, q_mean_auc=q_mean_auc, q_mean_pr=q_mean_pr,
                B_probability_below_threshold_windows=separated_windows,
                magnitude_auc_nondegrade=mag_auc,
                magnitude_pr_nondegrade=mag_pr, r_auc_nondegrade=r_auc,
                raw_survival=raw_survival, main_positives=main,
                novelty_Q4_better_than_Q1_windows=novelty_better,
                largest_bucket_mean_additivity_gap=additive, primary=primary)


def _exports(dest, metrics, contract):
    windows = metrics['windows']
    exports = {
        'p4_2h_q_bh_classifier': {w: dict(
            training={role: metrics['training'][w][role] for role in ('q_full', 'q_fold0', 'q_fold1')},
            raw=x['q_calibration']['q_raw_discrimination'],
            calibrated=x['q_calibration']['q_cal_discrimination']) for w, x in windows.items()},
        'p4_2h_q_bh_calibration': {w: dict(
            historical_OOF=metrics['training'][w]['calibrator'],
            outer=x['q_calibration']) for w, x in windows.items()},
        'p4_2h_nonzero_head': {w: dict(training=metrics['training'][w]['r'],
            outer=x['nonzero_head'], priority=x['diagnostics']['D3_no_r_priority']) for w, x in windows.items()},
        'p4_2h_benefit_magnitude': {w: dict(
            exact_reuse=metrics['training'][w]['magnitude_reuse']['benefit'],
            outer=x['magnitude']['benefit']) for w, x in windows.items()},
        'p4_2h_harm_magnitude': {w: dict(
            exact_reuse=metrics['training'][w]['magnitude_reuse']['harm'],
            outer=x['magnitude']['harm']) for w, x in windows.items()},
        'p4_2h_action_metrics': {w: dict(scores=x['action'], references=x['references'],
            diagnostics=x['diagnostics']) for w, x in windows.items()},
        'p4_2h_gate_tail': {w: dict(gates=x['gate'],
            distributions=x['conditional_distributions']) for w, x in windows.items()},
        'p4_2h_positive_survival': {w: x['survival'] for w, x in windows.items()},
        'p4_2h_admission_risk': {w: dict(admission=x[VARIANT]['admission'],
            buckets=x[VARIANT]['buckets']) for w, x in windows.items()},
        'p4_2h_segment_metrics': {w: dict(W0=x['W0']['segments'],
            H_conditional_BH=x[VARIANT]['segments'], mechanisms=x[VARIANT]['mechanisms'],
            cutpoints=x['cutpoints']) for w, x in windows.items()}}
    for name, data in exports.items():
        write_json(dest/(name+'.json'), dict(stage='P4.2H', run_id=contract['run_id'],
            status='completed', verification='pass', data=data, final_week='not_run',
            Warm_v2_integrated=False, full_history_run=False))


def report(repo):
    repo = Path(repo).resolve()
    dest, root = repo/'reports/phase4', repo/'artifacts/phase4'/RUN_ID
    m = read_json(dest/'P4_2H_metrics.json')
    v = read_json(dest/'P4_2H_VERIFICATION.json')
    assert m['status'] == 'completed' and m['verification'] == v['status'] == 'pass'
    lp_verification = read_json(dest/'P4_2H_LP_NUMERICAL_VERIFICATION.json')
    assert lp_verification['status'] == 'pass'
    assert lp_verification['comparison_absolute_tolerance'] == 1e-9
    assert lp_verification['fit_count'] == lp_verification['policy_changes'] == 0
    extra_audit_path = dest/'P4_2H_AUDIT_VERIFICATION.json'
    extra_audit = read_json(extra_audit_path) if extra_audit_path.exists() else None
    c = read_json(dest/'P4_2H_EXPERIMENT_CONTRACT.json')
    parity = read_json(dest/'p4_2h_data_parity.json')
    preflight = read_json(root/'PREFLIGHT.json')
    assert parity['status'] == preflight['status'] == 'pass'
    assert m['final_week'] == c['final_week'] == 'not_run'
    assert not m['full_history_run'] and not m['Warm_v2_integrated']
    ws, vd = m['windows'], m['verdicts']
    diag = _diagnostic_summary(ws, vd)
    _exports(dest, m, c)
    counts = {key: sum(x['gate']['H'][key] for x in ws.values())
              for key in ('eligible_edges', 'eligible_B', 'eligible_N', 'eligible_H')}
    sparse_survival = sum(x['survival']['H']['sparse_surviving'] for x in ws.values())
    total_seconds = preflight['seconds'] + m['seconds'] + v['seconds']
    cal = {w: m['training'][w]['calibrator'] for w in ws}
    text = [
        '# P4.2H：条件有益—有害分解（10%机制试验）',
        '状态：completed（已完成），独立复核pass（通过）。本轮保留W0；没有自动扩大100%或晋级新基线。',
        '## 1. 结论',
        f'四窗等权平均MAP@12 **{fmt(vd["mean_map"])}**，相对W0平均差 **{fmt(vd["mean_delta"])}**。'
        f'主冷正例从P4.2G的1个存活变为 **{vd["surviving"]}** 个；实际插入 **{vd["inserted"]}** 个冷侧正例，'
        f'移除 **{vd["removed"]}** 个原推荐正例。插入/移除比 **{fmt(vd["ratio"])}**，状态{vd["ratio_status"]}。',
        table(['预注册判断（本项目命名）', '含义', '结果'], [
            [VERDICTS[0], '正式U_H有益对有害排序信号', vd[VERDICTS[0]]],
            [VERDICTS[1], '主冷正例在匹配前的存活', vd[VERDICTS[1]]],
            [VERDICTS[2], '插入与移除正例的操作精度', vd[VERDICTS[2]]],
            [VERDICTS[3], '最终整体与暖侧MAP损伤边界', vd[VERDICTS[3]]],
            ['full_history_scaleup_allowed', '是否具备人工讨论100%扩量的条件', vd['full_history_scaleup_allowed']]]),
        'supported/mixed/rejected分别指完整满足预注册规则、满足部分改善规则、未获支持；不是统计显著性结论。'
        '所有判断使用未四舍五入值。无论扩量条件是否成立，本轮都停止，等待人工决策。',
        (f'本轮MAP安全门槛为{vd[VERDICTS[3]]}，但冷侧插入正例仍为0、主冷正例存活为0。'
         '接近W0主要意味着保守地减少修改，没有兑现冷侧机会，不能称作冷通道成功。'
         if vd['inserted'] == 0 and vd['surviving'] == 0 else
         'MAP安全、候选存活和最终冷侧正例增益是不同条件，分别按本轮证据判断，不能相互代替。'),
        '## 2. 术语、分母与正式决策',
        'W0（项目自定义）是冻结Warm-v1的原Top12及顺序；Warm指原推荐通道，并不表示其中每件商品必然属于暖≥21分组。'
        'Cold50（项目自定义）是冻结M4内容表示召回200件、经B0冷侧专家排序取50件后排除W0重合的候选。'
        'H_conditional_BH是本轮唯一新正式策略，文中简称H。P4.2F G是直接回归单次AP变化的旧策略；'
        'P4.2G U_R是三分类有益/中性/有害风险分离策略；旧U是P4.2R3冻结冷暖log-odds差。',
        '操作边（图模型通用概念）计量单位是“截止日—用户—Cold商品—W0位置”，每用户最多50×12条。'
        'B/N/H（项目标签）对应exact单次ΔAP@12大于0、等于0、小于0，不用容差归零。'
        '主冷正例按排除W0重合后的真实购买用户—商品对计数，不因12个替换槽位重复。'
        '跨窗累计是用户—商品—窗口观察，不是跨窗去重人数；未购买只代表未观察到隐式正反馈，不是已知曝光后的拒绝。',
        'q=P(B|B或H)是“如果替换会改变AP，其成为有益操作的条件概率”；r=P(B或H)是非零AP影响概率。'
        'q_raw是原二分类器输出，q_cal是历史用户折外校准后的正式q。m_B/m_H是分别条件于有益/有害的AP绝对幅度预测，'
        '完全复用P4.2G模型并裁剪到[0,1]。所有这些概率的对象是候选操作边，不能解释成用户购买率或线上曝光转化率。',
        '`G_BH=q_cal*m_B-(1-q_cal)*m_H`（项目条件净收益）决定是否准入：严格大于0通过、等于0拒绝。'
        '`implied_q_threshold=m_H/(m_B+m_H)`（项目内生条件概率阈值）等价要求q_cal严格大于该值；'
        '两个幅度都为0时阈值=1、G_BH=0、拒绝。`U_H=r*G_BH`仅作为匹配权重，r不改变准入集合。'
        '逐边验证两种gate布尔判断完全一致，合格边r和U_H必须为正；不一致或数值下溢均停止，不人工加epsilon挽救。',
        'exact matching（行业最大权一对一匹配）在通过G_BH门槛的边上最大化U_H总和；每件Cold和每个原槽位最多使用一次。'
        '零权虚拟节点允许完全拒绝，直接在原槽位替换，其他Warm次序不变；不加tau（额外准入阈值）、'
        'lambda（损失幅度乘数）搜索、max1（最多替换一件）、K_admit（预设准入件数）或用户配额；这些均为项目配置名。'
        '同权时使用冻结规范行列顺序及既有求解器规则，不用标签打破平局。',
        'AP@12（行业离线排名指标）为单用户前12件平均精度，分母min(下一周不同真值商品数,12)；MAP为用户AP均值。'
        '整体分母始终是该窗所有W0评测用户，无Cold候选者保留原推荐并计入。'
        'strict_cold、sparse1_5、warm_21_plus、all_cold_sparse（项目分组）分别按截止日前全站交易事件数0、1–5、≥21、0–5分商品。'
        '分组MAP以该组真值重算AP，只对有该组真值的用户平均；这是商品冷暖而非用户冷暖。',
        'ROC-AUC（行业二类区分指标）和PR-AUC（本报告使用非插值平均精度）主比较均限B/H操作边。'
        'Pearson为线性相关、Spearman为同分平均名次秩相关；正式分数与ΔAP相关性使用完整B/N/H操作空间。'
        'Brier为二分类概率平方误差均值，logloss为二分类负对数似然均值，越低越好；'
        'ECE（预期校准误差，行业指标）以10个等宽概率箱内预测均值与实际B率的绝对差按边数加权，越低越好。'
        'MAE/RMSE（行业误差指标）分别是符号对应行上的幅度平均绝对误差/均方根误差。',
        '准入覆盖率=至少发生一次替换的用户数÷该组全部用户数；插入率/移除率的分母为替换次数。'
        '插入/移除比则是两类正例观察数相除：finite为分母非零，infinite为移除0且插入>0，undefined为两者均0。'
        '不把拒绝所有候选视为精度成功。topK重合比例是同用户两排序前K边交集÷min(K,该用户合格边数)，无合格边用户不参与均值。',
        '## 3. 冻结、训练数据与成本',
        f'合同登记时间{c["created_at_utc"]}，起始Git提交{c["git_sha"]}；run_id（本阶段唯一运行标识）={c["run_id"]}。'
        '直接复用P4.2F/G同一约10%历史用户、外层名单、操作行和70列特征矩阵，数据、列序、取值、标签及SHA一致性均已核验。'
        '完整特征中文语义沿用[P4.2F报告第8节](P4_2F_FINAL.md#8-特征字段注释)，当前合同嵌入完整feature_contract。'
        '没有添加绝对qC/qW概率、原logit、ID或未来售卖代理；qC/qW是旧冷暖倾向模型，此处只保留原相对名次特征。',
        'q只训练全部B/H行且不加样本/类别权重；Neutral完全不进入q或q校准器。r使用全部B/H及原确定性2%中性样本，'
        'B/H权重1、中性权重50，不进行额外概率校准。全站聚合统计仍沿用完整合法历史，10%不表示把抽样热度当全局。'
        '所有历史标签结束日label_end严格早于外层截止日；不读取2020-09-16最终周。',
        'OOF（out-of-fold，行业折外预测）按用户ID稳定哈希分两折，同用户所有历史日期同折。q_fold0用折0训练预测折1，'
        'q_fold1反向；每个历史B/H行恰有一次未用该用户训练的原始q。'
        '这属于外层之前整个合法历史池内的用户隔离，不是每个历史日期前向训练，不声称历史逐日可部署。',
        'Platt（行业逻辑校准）用历史OOF输入logit(clip(q_raw,1e-6,1-1e-6))拟合a+b*x；'
        'logit是概率赔率的自然对数，clip表示截断到给定数值区间；无正则、无类别或样本权重，'
        'LBFGS（行业拟牛顿优化算法）求解，tol（收敛容差）=1e-8、max_iter（最大迭代数）=1000。正式要求b>0且收敛；'
        '全历史q模型预测外层后套用历史a/b。裁剪会在极端概率制造同分，不能声称raw/cal的排序指标必然完全一致。',
        '校准迁移局限：历史OOF分数来自仅半池用户训练的q模型，而外层分数来自全合法历史用户池训练的q模型。'
        '两者训练规模不同，原始概率分布可能变化；用户隔离避免该用户训练内自评，却不能保证半池到全池的概率尺度可完全迁移。'
        '本轮外层指标用于审计该迁移，不据此再拟合、平移截距或自动扩大样本。',
        table(['模型', '轮数', '学习率', '叶数', '最大深度', '叶最少样本', 'L2正则', '新训练'],
            [[name, z['n_estimators'], z['learning_rate'], z['num_leaves'], z['max_depth'],
              z['min_child_samples'], z['reg_lambda'], fresh]
             for name, z, fresh in [('q', c['q_params'], True), ('r', c['r_params'], True),
                                    ('m_B', c['benefit_params'], False), ('m_H', c['harm_params'], False)]]),
        table(['窗口', '合法历史日期数', 'q全部B/H边数', 'B数', 'H数', 'q折0训练行', 'q折1训练行', 'r训练行', 'r中性样本行'],
            [[NAMES[w], len(c['historical_pools'][w]), m['training'][w]['q_full']['rows'],
              m['training'][w]['q_full']['B'], m['training'][w]['q_full']['H'],
              m['training'][w]['q_fold0']['rows'], m['training'][w]['q_fold1']['rows'],
              m['training'][w]['r']['rows'], m['training'][w]['r']['N']] for w in ws]),
        f'预估计算与核验20–40分钟。实际数据预检{fmt(preflight["seconds"])}秒、正式训练/预测/诊断/评测'
        f'{fmt(m["seconds"])}秒、成功独立复核{fmt(v["seconds"])}秒；上述成功记录小计{total_seconds/60:.2f}分钟，'
        '不含首次失败复核（退出前未保存精确计时）、只读定位及额外审计，也不含阅读、开发和报告编写，不能当作端到端总成本。'
        f'正式峰值进程工作集{fmt(m["peak_gib"])} GiB。CPU4线程，不使用GPU。'
        '新拟合16个LightGBM（每窗全q、两折q、r）+4个历史Platt；另外8个外层诊断逻辑拟合仅用于审计。'
        'mB/mH新增拟合0次，qC/qW/上游表示新增拟合0次。',
        '报告生成器是结果汇总层，在主计算实现之后单独完成并作格式核对，不位于正式训练/诊断执行路径；'
        '它只读取已核验的metrics、合同和预检/复核记录，未修改模型、标签或诊断指标。最终输出清单另包含其源码身份。',
        '## 4. q条件概率：历史OOF与外层校准',
        table(['窗口', '历史B/H边数', '历史B数', '历史B率', 'a截距', 'b斜率', '迭代次数', '收敛'],
            [[NAMES[w], z['rows'], z['positives'], z['observed_B_rate'], z['a'], z['b'], z['n_iter'], z['converged']]
             for w, z in cal.items()]),
        table(['窗口', '历史OOF状态', '平均q', 'Brier', 'logloss', 'ECE'],
            [[NAMES[w], name, z[key]['predicted_mean'], z[key]['Brier'], z[key]['logloss'], z[key]['ECE']]
             for w, z in cal.items() for name, key in [('raw', 'raw'), ('Platt后', 'calibrated')]]),
        '下表外层诊断a/b再次拟合于外层标签，只用于测量校准偏差；这些参数没有进入正式预测。'
        '理想诊断截距/斜率约为0/1，不能将接近它们直接当作尾部安全保证。',
        table(['窗口', '状态', 'B/H边数', '实际B率', '平均q', 'Brier', 'logloss', 'ECE', '诊断截距', '诊断斜率', '诊断收敛'],
            [[NAMES[w], name, z['rows'], z['observed_rate'], z['predicted_mean'], z['Brier'], z['logloss'], z['ECE'],
              z['diagnostic_calibration']['a'], z['diagnostic_calibration']['b'], z['diagnostic_calibration']['converged']]
             for w, x in ws.items() for name, z in [('raw', x['q_calibration']['q_raw']), ('Platt后', x['q_calibration']['q_cal'])]]),
        table(['窗口', 'q原始AUC', 'q原始PR', 'q校准AUC', 'q校准PR'],
            [[NAMES[w], x['q_calibration']['q_raw_discrimination']['roc_auc'],
              x['q_calibration']['q_raw_discrimination']['pr_auc'],
              x['q_calibration']['q_cal_discrimination']['roc_auc'],
              x['q_calibration']['q_cal_discrimination']['pr_auc']] for w, x in ws.items()]),
        '以下分位桶仅在该窗完整B/H操作边上以raw-q值计算50/80/90/95/99分位边界，等值进低桶；'
        'raw/cal使用相同边，不因校准结果换桶。实际占比可因同分偏离名义百分位；空桶保留。'
        '每桶B率分母为该桶B+H行，不能解释成主正例商品存活率。']
    for w, x in ws.items():
        text += [f'### {NAMES[w]}原始q分位桶',
                 'raw-q分位值边界：'+', '.join(fmt(z) for z in x['q_calibration']['raw_q_cutpoints'])+'。',
                 table(['分位下界', '分位上界', '边数', 'B数', 'H数', '实际B率', '平均raw-q', '平均cal-q'],
                    [[z['quantile_low'], z['quantile_high'], z['rows'], z['B'], z['H'], z['observed_B_rate'],
                      z['mean_q_raw'], z['mean_q_cal']] for z in x['q_calibration']['quantile_bins']]),
                 f'### {NAMES[w]}等宽概率可靠性箱',
                 table(['q状态', '概率下界', '概率上界', '边数', '实际B数', '实际B率', '平均q'],
                    [[name, b['low'], b['high'], b['rows'], b['positives'], b['observed_rate'], b['predicted_mean']]
                     for name in ('q_raw', 'q_cal') for b in x['q_calibration'][name]['reliability']])]
    text += [
        '## 5. 幅度与内生阈值',
        'm_B/m_H模型与P4.2G逐批raw及裁剪预测完全一致。下表误差只在真实对应符号边计算；'
        '裁剪计数则覆盖全操作空间，两者分母不同。',
        table(['窗口', '幅度头', '对应符号边数', '实际幅度均值', '预测幅度均值', '条件MAE', '条件RMSE', '全边负值裁剪数', '全边超1裁剪数'],
            [[NAMES[w], name, z['edges'], z['observed_mean'], z['predicted_mean'], z['conditional_MAE'],
              z['conditional_RMSE'], z['clipped_low_edges'], z['clipped_high_edges']]
             for w, x in ws.items() for name, z in x['magnitude'].items()]),
        '以下分布的统计单位是操作边；all/B/H/N分别表示完整空间、真实有益、有害、中性子集。'
        'q_minus_threshold（项目概率安全差）为q_cal减内生阈值，严格正值才准入，不能被当成新增人工margin。',
        table(['窗口', '真实B边最大q_cal', '真实B边最小阈值', '最大q仍低于最小阈值', '真实B边q中位数', '全部边阈值中位数'],
            [[NAMES[w], x['conditional_distributions']['B']['q_cal']['max'],
              x['conditional_distributions']['B']['implied_q_threshold']['min'],
              x['conditional_distributions']['B']['q_cal']['max'] < x['conditional_distributions']['B']['implied_q_threshold']['min'],
              x['conditional_distributions']['B']['q_cal']['median'],
              x['conditional_distributions']['all']['implied_q_threshold']['median']] for w, x in ws.items()]),
        (f'四窗中有{diag["B_probability_below_threshold_windows"]}窗满足上述严格极值分离。'
         '全部四窗都满足时，这是比逐边检查更强的直接证据：任何真实有益边都不可能通过q>阈值。'
         '因此本轮零正例存活发生在G_BH门槛，而不是r或匹配删掉了已经存在的有益机会。'
         if diag['B_probability_below_threshold_windows'] == 4 else
         '极值分离仅用于描述对应窗口的门槛阻塞；未满足该条件不等于所有有益边都能通过，仍以逐边存活审计为准。'),
        'B边q的中位数与全操作空间阈值中位数分母不同，表内分别标明，不能当作同一批边的配对差。'
        '真正配对差见后续q_minus_threshold分布。即使出现极值分离，也不能据此认定幅度头必错或未验证的扩量必然无效；'
        '它说明现有条件区分与幅度尺度组合尚未产生可以执行的有益优势。']
    for w, x in ws.items():
        text += [f'### {NAMES[w]}幅度、q与阈值分布',
                 table(['真实子集', '变量', '边数', '均值', '中位数', 'p90', 'p99', '最小', '最大'],
                    [[group, key, d[key]['rows'], d[key]['mean'], d[key]['median'], d[key]['p90'],
                      d[key]['p99'], d[key]['min'], d[key]['max']]
                     for group, d in x['conditional_distributions'].items()
                     for key in ('m_B', 'm_H', 'q_raw', 'q_cal', 'r', 'implied_q_threshold', 'q_minus_threshold')]),
                 table(['真实子集', '边数', 'q高于阈值', 'q等于阈值', 'q低于阈值', 'mB/mH有限均值', '有限中位数', '有限p90', '有限p99', '比值无限边数', '0/0边数'],
                    [[group, d['edges'], d['q_above_threshold'], d['q_equal_threshold'], d['q_below_threshold'],
                      d['benefit_to_harm_ratio']['finite_positive_denominator']['mean'],
                      d['benefit_to_harm_ratio']['finite_positive_denominator']['median'],
                      d['benefit_to_harm_ratio']['finite_positive_denominator']['p90'],
                      d['benefit_to_harm_ratio']['finite_positive_denominator']['p99'],
                      d['benefit_to_harm_ratio']['infinite_edges'], d['benefit_to_harm_ratio']['undefined_edges']]
                     for group, d in x['conditional_distributions'].items()])]
    text += [
        '## 6. 操作排序与r的优先级作用',
        '正式主判据固定使用U_H；G_BH同时报告，不在两者之间事后挑选判定。D1=raw-q条件净收益G_raw，只报区分/门槛/存活；'
        'D2=q_cal纯条件分类排序；D3=是否乘r的排序比较。D1/D2/D3（项目固定诊断编号）均不生成第二套正式MAP。',
        table(['窗口', '分数', 'B/H边数', 'B对H AUC', 'B对H PR', '全边Pearson', '全边Spearman'], list(_action_rows(ws))),
        'G_raw与q_cal诊断按合同不计算全边相关性，表内记不可用。旧U引用P4.2D同一操作空间，不重训或重新生成旧推荐。',
        table(['窗口', 'r审计边数', '实际非零率', '平均r', 'Brier', 'logloss', 'ECE'],
            [[NAMES[w], x['nonzero_head']['rows'], x['nonzero_head']['observed_rate'],
              x['nonzero_head']['predicted_mean'], x['nonzero_head']['Brier'], x['nonzero_head']['logloss'],
              x['nonzero_head']['ECE']] for w, x in ws.items()]),
        'r表分母为完整B/N/H操作边，实际正类是B或H，不是有益类B。没有额外r校准。'
        '下表相关性比较G_BH与U_H两种预测分数，不是分数与真实ΔAP。',
        table(['窗口', '全部操作边Spearman', '合格边Spearman', '有Cold用户', '有合格边用户', '无合格边用户', 'top1均值', 'top5均值'],
            [[NAMES[w], z['all_edges']['spearman'], z['eligible_edges']['spearman'], z['action_users'],
              z['users_with_eligible_edges'], z['action_users_without_eligible_edges'],
              z['top_overlap']['1']['mean_user_overlap'], z['top_overlap']['5']['mean_user_overlap']]
             for w, x in ws.items() for z in [x['diagnostics']['D3_no_r_priority']]]),
        'topK同分按原Cold行再原Warm槽位稳定排序；两种分数共用合格边集合。这里只检查优先级，不运行不乘r的匹配或MAP。',
        '## 7. 合格边与主冷正例存活',
        table(['窗口', '策略', '合格边数', 'B边', 'N边', 'H边', 'B占比', 'N占比', 'H占比', 'B/H边数比'],
            [[NAMES[w], name, z['eligible_edges'], z['eligible_B'], z['eligible_N'], z['eligible_H'],
              z['B_share'], z['N_share'], z['H_share'], z['B_H_ratio']]
             for w, x in ws.items() for name, z in [*x['gate'].items(), ('raw诊断', x['diagnostics']['D1_raw_q']['gate'])]]),
        'H门槛为G_BH>0，历史G_R/F_G分别沿其原始效用>0规则。B/N/H占比分母是对应方案合格边数，'
        'q虽然不以Neutral训练，正式门槛仍作用于所有B/N/H，因此中性审计不能删除。',
        table(['窗口', '策略', '主正例总数', '存活正例', '存活率', '严格冷存活', '稀疏存活'],
            [[NAMES[w], name, z['main_positive_candidates'], z['surviving_positive_candidates'],
              z['survival_rate'], z['strict_surviving'], z['sparse_surviving']]
             for w, x in ws.items() for name, z in x['survival'].items() if name != 'old_U']),
        table(['窗口', '旧U主正例', '旧U存活', '旧U严格冷存活', '旧U稀疏存活'],
            [[NAMES[w], x['survival']['old_U']['all']['positive_candidates'],
              x['survival']['old_U']['all']['any_beneficial_edge'],
              x['survival']['old_U']['strict']['any_beneficial_edge'],
              x['survival']['old_U']['sparse']['any_beneficial_edge']] for w, x in ws.items()]),
        '存活=一个主Cold正例至少有一个真实有益Warm槽位通过门槛；分母是同一候选池内主正例用户—商品对，'
        '不是全目录冷真值，也不是最终插入数。',
        '## 8. 最终MAP与准入风险',
        table(['窗口', '正式策略', '整体MAP', '暖≥21 MAP', '严格冷MAP', '稀疏1–5 MAP', '冷稀疏0–5 MAP'],
            [[NAMES[w], name, z['map12'], *[z['segments'][s]['map12'] for s in SEGMENTS]]
             for w, x in ws.items() for name, z in [('W0', x['W0']), ('H', x[VARIANT])]]),
        table(['窗口', '整体MAP差', '暖组MAP差', '严格冷MAP差', '稀疏MAP差', '冷稀疏MAP差'],
            [[NAMES[w], x[VARIANT]['delta_vs_w0'], *[x[VARIANT]['segments'][s]['delta_vs_w0'] for s in SEGMENTS]]
             for w, x in ws.items()]),
        table(['窗口', '全部用户', '准入用户', '覆盖率', '替换总数', '每准入用户替换数', '每全部用户替换数', '最大替换数', '插入正例', '移除正例', '严格冷插入', '稀疏插入'],
            [[NAMES[w], *[x[VARIANT]['admission'][key] for key in ('users', 'users_with_admission', 'coverage',
                'replacements', 'mean_per_admitted', 'mean_per_all_users', 'max_replacements',
                'inserted_cold_positives', 'removed_warm_positives', 'strict_positive_inserted', 'sparse_positive_inserted')]]
             for w, x in ws.items()]),
        table(['窗口', '替换件数桶', '用户数', 'MAP差', '插入正例', '移除正例', '净正例', '实际AP差减单边差和'],
            [[NAMES[w], name, *[b[key] for key in ('users', 'map_delta', 'inserted_cold_positives',
                'removed_warm_positives', 'net_positives', 'mean_exact_minus_single_sum')]]
             for w, x in ws.items() for name, b in x[VARIANT]['buckets'].items()]),
        '最后一列为该桶用户实际最终AP增量减已执行各边单独AP增量之和的平均。单边标签是固定已观察真值下的列表重算差，'
        '并非线上因果效应；多边AP不一定可加。本轮对最终列表精确重算，不把分数总和当最终MAP。'
        '各替换数桶用户不同，不把桶间差异解释成限制替换件数的因果收益。',
        '## 9. 新鲜偏好及历史丰富度',
        'novelty（项目新鲜商品接受度）是用户历史购买中购买日前商品全站交易≤5的事件占比。'
        'Q1–Q4沿P4.2F历史用户—窗口四分位切点，低到高且同分进低桶。richness（项目历史丰富度）'
        '按过去交易事件数沿原三分位切点切low/medium/high。只复用切点，不对外层重切桶、训练router或调整阈值。',
        table(['窗口', '分组轴', '分组', '用户数', '覆盖率', '替换次数', '插入正例率', '移除正例率', 'MAP差'],
            [[NAMES[w], axis, name, *[b[key] for key in ('users', 'coverage', 'replacements',
                'cold_positive_insertion_rate', 'warm_positive_removal_rate', 'map_delta')]]
             for w, x in ws.items() for axis, groups in x[VARIANT]['mechanisms'].items() for name, b in groups.items()]),
        f'最高新鲜偏好Q4相对Q1的MAP差有{diag["novelty_Q4_better_than_Q1_windows"]}/4窗更好；'
        '这是固定组的描述性结果，不自动证明可以据此安全准入。',
        '## 10. 六层机制诊断',
        '以下只解释已固定实验，不反向修改模型、概率、权重或门槛。数学上，已有P4.2G的共同非零因子本就不改变符号；'
        '本轮检验的是分开训练条件分类、用户OOF校准及非零概率后整体组合是否改善，'
        '不能将差异简单解释成“删掉Neutral概率就必然救回正例”。']
    for title, content in diag['notes']:
        text += ['### '+title, content]
    text += [
        f'各替换桶实际AP与单边和的平均偏差绝对值最大为{fmt(diag["largest_bucket_mean_additivity_gap"])}。'
        '此数只衡量已执行动作的加和误差，不等于证明多件策略本身安全；结合第8节单件桶及总体风险判断。',
        '扩量条件不成立时的主要阻碍：'+diag['primary']+'。',
        '## 11. 预注册判断、核验和停止边界',
        'Action supported：正式U_H四窗平均B/H AUC≥P4.2G U_R、至少3窗AUC不退化、平均PR≥参照；'
        '否则只要平均AUC或PR一项严格提高为mixed，其余rejected。Survival supported：累计≥8、至少3窗严格多于P4.2G、'
        '至少1个严格冷存活；未通过但累计≥2为mixed（相对参照1至少翻倍，只是描述性门槛）。',
        'Policy supported：累计插入/移除比≥0.25、插入≥3、移除≤20；未通过但比≥0.15且高于1/15为mixed。'
        'Safety supported：整体平均MAP差≥−0.0005、最差窗≥−0.001、暖组平均差≥−0.0005；'
        '否则三项均严格优于P4.2F G为mixed，其余rejected。',
        '仅当action非rejected、survival及precision均为supported/mixed、安全为supported，扩量准备度才true。'
        '即使true也不授权本轮扩量；无新基线晋级。所有mixed定义在训练前冻结。',
        f'独立核验{len(v["checks"])}项用户不变量通过；全部历史OOF、全外层输出、幅度复用、门槛及最终列表重放；'
        f'对{v["LP_graphs"]}个固定用户图另用独立线性规划检查匹配目标，不夸大为全图第二求解器证明。'
        '完整明细见[P4_2H_VERIFICATION.json](P4_2H_VERIFICATION.json)。',
        '### 独立核验器的数值尺度问题',
        '首次复核在春季备用线性规划对照中失败，并非首次即通过。原目标系数很小，备用求解器给出的目标'
        f'{fmt(lp_verification["start"]["original_lp_objective"])}低于已执行合法匹配目标'
        f'{fmt(lp_verification["start"]["executed_objective"])}，差'
        f'{fmt(lp_verification["start"]["absolute_gap"])}，超过原绝对比较容差1e-9。'
        '已存在的合法匹配目标反而更高，指向备用线性规划求解器的绝对数值尺度敏感性。',
        '后置只读适配器仅将每个独立LP（linear programming，行业线性规划）目标系数除以该图最大绝对系数，'
        '求解后乘回原尺度；正比例缩放不改变可行约束和数学最优匹配。保持原1e-9比较容差不变，'
        '没有改正式匹配器、冻结核验源码、模型、分数、准入集合、推荐或MAP，也没有放宽验收误差。'
        f'按该方式重新完整复核四窗，{lp_verification["LP_graphs"]}个固定备用LP图通过，新增训练0次、策略改动0次。'
        '首次失败记录保留在[P4_2H_VERIFICATION_ENGINEERING_NOTE.md](P4_2H_VERIFICATION_ENGINEERING_NOTE.md)，'
        '成功数值复核见[P4_2H_LP_NUMERICAL_VERIFICATION.json](P4_2H_LP_NUMERICAL_VERIFICATION.json)。',
        (f'附加只读审计通过{len(extra_audit["windows"])}个窗口的准入边计数、唯一冷正例存活、概率误差及分箱、'
         '同用户前1/5边重合和关键分布核验，并独立重算预注册判定；'
         '明细见[P4_2H_AUDIT_VERIFICATION.json](P4_2H_AUDIT_VERIFICATION.json)。额外审计耗时不包含在前述成功记录小计中。'
         if extra_audit and extra_audit.get('status') == 'pass' else
         '附加只读审计未提供已通过的最终记录；本报告不把尚未完成的附加检查写成通过。'),
        '严格冻结最终周2020-09-16为not_run；没有100%训练、没有Warm-v2整合、没有P4.3、没有新召回/特征、'
        '没有超参数扫描或门槛搜索。行级用户/商品、OOF、模型、推荐及概率数组留在Git忽略的artifacts。'
        '输出清单按用户显式证据合同记录SHA，用于后续复用时对照；不是例行重复哈希检查。',
        '## 12. 用户33项问题逐项答复']
    answers = [
        (1, 'q历史B/H正例率', '第4节历史表；'+_window_values(cal, lambda z: z['observed_B_rate'])),
        (2, 'raw q外层校准', '第4节完整表；平均q：'+_window_values(ws, lambda x: x['q_calibration']['q_raw']['predicted_mean'])+
         '；实际B率：'+_window_values(ws, lambda x: x['q_calibration']['q_raw']['observed_rate'])),
        (3, 'OOF Platt是否改善', f'Brier/logloss/ECE分别在{diag["calibration_improved_windows"]["Brier"]}/4、'
         f'{diag["calibration_improved_windows"]["logloss"]}/4、{diag["calibration_improved_windows"]["ECE"]}/4窗降低；尾部见第4节分位桶，不以均值替代。'),
        (4, '四窗a/b', '第4节；'+_window_values(cal, lambda z: f'a={fmt(z["a"])}，b={fmt(z["b"])}')),
        (5, 'b是否全为正', str(all(z['b'] > 0 for z in cal.values()))+'；独立复核通过。'),
        (6, 'q AUC/PR是否优于G', f'q_cal平均AUC={fmt(diag["q_mean_auc"])}、PR={fmt(diag["q_mean_pr"])}；'
         f'G的U_R平均AUC={fmt(vd["G_R_mean_auc"])}、PR={fmt(vd["G_R_mean_PR"])}。'
         '两者分别是条件分类与含幅度/非零概率的操作价值分数；第4/6节逐窗，正式action按U_H判断。'),
        (7, '幅度头是否合理', f'第5/10节：exact复用；G_BH相对q的AUC/PR分别{diag["magnitude_auc_nondegrade"]}/4、'
         f'{diag["magnitude_pr_nondegrade"]}/4窗不退化，未运行无幅度正式MAP，不能作单因素结论。'),
        (8, '内生q阈值median/p90/p99', '第5节全边分布；'+_window_values(ws, lambda x:
         ', '.join(fmt(x['conditional_distributions']['all']['implied_q_threshold'][k]) for k in ('median', 'p90', 'p99')))),
        (9, 'B边q与阈值', '第5节；q−阈值中位数：'+_window_values(ws, lambda x: x['conditional_distributions']['B']['q_minus_threshold']['median'])),
        (10, 'H边q与阈值', '第5节；q−阈值中位数：'+_window_values(ws, lambda x: x['conditional_distributions']['H']['q_minus_threshold']['median'])),
        (11, '正式合格边数', counts['eligible_edges']),
        (12, '合格B/N/H', f'B={counts["eligible_B"]}，N={counts["eligible_N"]}，H={counts["eligible_H"]}；第7节逐窗。'),
        (13, '主Cold正例存活', f'P4.2G的1→{vd["surviving"]}，分母{diag["main_positives"]}。'),
        (14, '严格冷/稀疏存活', f'严格冷{vd["strict_surviving"]}，稀疏{sparse_survival}。'),
        (15, 'raw与cal存活差', f'raw={diag["raw_survival"]}，cal={vd["surviving"]}，cal−raw={vd["surviving"]-diag["raw_survival"]}。'),
        (16, 'r是否改变优先级', '第6节；top1重合：'+_window_values(ws, lambda x: x['diagnostics']['D3_no_r_priority']['top_overlap']['1']['mean_user_overlap'])+'；无第二正式MAP。'),
        (17, '插入Cold正例数', vd['inserted']),
        (18, '移除原推荐正例数', vd['removed']),
        (19, '插入/移除比', f'{fmt(vd["ratio"])}，{vd["ratio_status"]}。'),
        (20, '恰好替换1件是否安全', '第8节同桶MAP差：'+_window_values(ws, lambda x: x[VARIANT]['buckets']['1']['map_delta'])+'；不能用不同用户桶作因果结论。'),
        (21, '多件替换是否仍次要', f'第8/10节：桶平均非加和偏差最大绝对值{fmt(diag["largest_bucket_mean_additivity_gap"])}；'
         '需要结合单件桶，不能仅凭替换件数或加和误差断言。'),
        (22, '整体MAP差', f'四窗等权均值{fmt(vd["mean_delta"])}；最差窗{fmt(vd["worst_delta"])}。'),
        (23, '暖≥21 MAP差', f'四窗等权均值{fmt(vd["mean_warm_delta"])}。'),
        (24, '严格冷/稀疏MAP差', '第8节逐窗；均值严格冷='+fmt(float(np.mean([x[VARIANT]['segments']['strict_cold']['delta_vs_w0'] for x in ws.values()])))+
         '，稀疏='+fmt(float(np.mean([x[VARIANT]['segments']['sparse1_5']['delta_vs_w0'] for x in ws.values()])))),
        (25, '高novelty是否更安全', f'Q4相对Q1的MAP差有{diag["novelty_Q4_better_than_Q1_windows"]}/4窗更好；第9节完整值，无router。'),
        (26, 'action verdict', vd[VERDICTS[0]]),
        (27, 'survival verdict', vd[VERDICTS[1]]),
        (28, 'policy precision verdict', vd[VERDICTS[2]]),
        (29, 'MAP safety verdict', vd[VERDICTS[3]]),
        (30, '是否允许讨论100%扩量', str(vd['full_history_scaleup_allowed'])+'；本轮不执行扩量。'),
        (31, '若false，失败在哪层', diag['primary']+'；六层分析见第10节。'),
        (32, '最终周是否not_run', m['final_week']+'；2020-09-16未运行。'),
        (33, 'Warm-v2是否未整合', str(not m['Warm_v2_integrated'])+'；Warm-v2未整合，P4.3未开始。')]
    assert len(answers) == 33
    text.append(table(['序号', '问题', '本轮回答'], answers))
    (dest/'P4_2H_FINAL.md').write_text('\n\n'.join(text)+'\n', encoding='utf8')
    manifest(repo)


def manifest(repo):
    """Explicit evidence-package identity contract, scoped strictly to P4.2H."""
    repo = Path(repo).resolve()
    dest, root = repo/'reports/phase4', repo/'artifacts/phase4'/RUN_ID
    metrics = read_json(dest/'P4_2H_metrics.json')
    verification = read_json(dest/'P4_2H_VERIFICATION.json')
    assert metrics['status'] == 'completed' and metrics['verification'] == verification['status'] == 'pass'
    assert read_json(dest/'P4_2H_LP_NUMERICAL_VERIFICATION.json')['status'] == 'pass'
    paths = sorted(p for p in dest.iterdir() if p.is_file() and
                   p.name.lower().startswith('p4_2h_') and p.name != 'P4_2H_OUTPUT_MANIFEST.json')
    write_json(dest/'P4_2H_OUTPUT_MANIFEST.json', dict(
        stage='P4.2H', run_id=RUN_ID, status='completed', verification='pass',
        reports=[identity(p) for p in paths],
        source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42h*.py'))],
        tests=[identity(p) for p in sorted((repo/'tests').glob('test_p42h*.py'))],
        artifacts=[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
        final_week='not_run', Warm_v2_integrated=False, full_history_run=False,
        selected='W0', hash_purpose='explicit user evidence-package requirement; bind stage-only outputs for future reuse'))


if __name__ == '__main__':
    report(Path('.'))
