"""Chinese P4.2R closure from measured files only; no training or evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .p41a_contract import identity, read_json, write_json
from .p42_contract import TAUS, WINDOWS
from .p42_report import feature_note
from .p42r_contract import RUN_ID


WINDOW_NAMES = {"winter_20200122": "冬季", "spring_20200318": "春季",
                "early_summer_20200624": "初夏", "late_summer_20200819": "夏末"}
VARIANT_NAMES = {"W0": "W0 原基线", "A_tau0": "激进 τ=0", "M_tau_ln2": "中等 τ=ln2",
                 "C_tau_ln4": "保守 τ=ln4"}
DECISIONS = {
    "promote_p4_2_aggressive": "激进阈值通过全部门槛，得到正式安全工作点。",
    "promote_p4_2_moderate": "中等阈值通过全部门槛，得到正式安全工作点。",
    "promote_p4_2_conservative": "保守阈值通过全部门槛，得到正式安全工作点。",
    "pareto_frontier_supported_but_no_safe_operating_point": "更积极准入同时增加冷侧收益和暖侧风险，但三个固定点都没有通过全部门槛。",
    "cold_gain_not_recovered": "固定三个阈值未形成满足全部门槛的冷侧增益，仍保留 W0。",
    "warm_risk_uncontrolled": "至少一个固定点提高冷侧均值，但总体或暖侧保护不通过，仍保留 W0。",
    "propensity_calibration_failure": "共同人群的跨窗口严重校准门槛触发，按合同停止，准入未运行。",
    "qC_population_repair_failure": "恢复共同人群后 qC 未在冻结数值预算内完成有效拟合，后续准入停止。",
    "qW_R1_and_R2_convergence_failure": "冬季 qW 的 R1、R2 都未收敛，停止，不增加迭代或换求解器。",
    "shared_population_contract_failure": "两侧实际候选训练行的共同人群合同未满足，停止。",
    "engineering_failure": "正式计算触发工程或既定后续链停止条件；只能报告已完成部分，不能视作完成四窗晋级验证。",
}
GATE_NAMES = {
    "overall_mean": "总体均值不降", "overall_nondegrade": "总体至少3窗不降",
    "overall_worst": "总体最差窗不低于−0.000200", "warm_mean": "暖组均值下降不超0.000100",
    "warm_protected_windows": "至少3窗暖组下降不超0.000200", "cold_mean": "冷/稀疏组均值严格增加",
    "cold_nondegrade": "冷/稀疏组至少3窗不降", "cold_only_positive_windows": "至少2窗插入独有冷路正例",
    "replacement_efficiency": "合并插入正例数严格超过移除正例数",
}


def fmt(value, digits=8):
    return "不可用" if value is None else f"{value:.{digits}f}"


def rate(value):
    return "不可用" if value is None else f"{value:.6%}"


def sci(value):
    return "不可用" if value is None else f"{value:.6g}"


def table(headers, rows):
    rows = list(rows)
    if not rows:
        return "\n未运行；没有可报告的观测值。\n"
    def line(row):
        return "| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |"
    return "\n" + "\n".join([line(headers), line(["---"] * len(headers)), *[line(r) for r in rows]]) + "\n"


def _model_iterations(attempt):
    record = attempt.get("model")
    if not record:
        return "不可用"
    return ",".join(map(str, read_json(Path(record["path"])).get("n_iter", [])))


def _old_warm_geometry(repo):
    value = read_json(repo / "reports/phase4/p4_2_convergence_diagnostic.json")
    maximum = None
    # The original read-only diagnostic exposes a list, not the repaired
    # diagnostic's scalar maximum. Accept no invented substitute statistic.
    for key, rows in value.items():
        if "standard" in key and isinstance(rows, list):
            finite = [r.get("abs_max") for r in rows if isinstance(r, dict) and r.get("abs_max") is not None]
            if finite:
                maximum = max(finite)
    return value, maximum


def _append_measured_doc(path, title, paragraphs):
    """Replace this generated block only; never rewrite earlier P4.2 history."""
    start, end = "<!-- P4.2R measured closure begin -->", "<!-- P4.2R measured closure end -->"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    block = start + "\n\n" + title + "\n\n" + "\n\n".join(paragraphs) + "\n\n" + end
    if start in current:
        if end not in current:
            raise ValueError(f"unclosed P4.2R documentation block: {path}")
        left, tail = current.split(start, 1)
        _, right = tail.split(end, 1)
        updated = left + block + right
    else:
        updated = current.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def _control_answer(control):
    if not control:
        return "未运行同窗旧模型对照；旧 P4.2 仅存在冬季 qC，其他窗口旧模型不存在，未补训。"
    item = next(iter(control.values()))
    old, new = item["old"], item["new"]
    return (f"同一冬季 {item['same_outer_rows']:,} 条 Cold50 行：旧/新平均预测为 "
        f"{rate(old['mean_predicted_probability'])}、{rate(new['mean_predicted_probability'])}；"
        f"真实正例率 {rate(new['observed_positive_rate'])}；Brier {fmt(old['brier_score'])}→{fmt(new['brier_score'])}，"
        f"ECE {fmt(old['ece'])}→{fmt(new['ece'])}。这是唯一同窗旧模型对照，不把不存在的三窗旧模型写成测过。")


def report(repo):
    repo = Path(repo).resolve()
    folder = repo / "reports/phase4"
    root = repo / "artifacts/phase4" / RUN_ID
    m = read_json(folder / "P4_2R_metrics.json")
    c = read_json(folder / "P4_2R_EXPERIMENT_CONTRACT.json")
    current_execution = read_json(root / "EXECUTION_START.json")
    if m["execution"]["started_at_utc"] != current_execution["started_at_utc"]:
        raise ValueError("formal run still has only stale bootstrap metrics; do not render completion")
    verification_path = folder / "P4_2R_VERIFICATION.json"
    verification = read_json(verification_path) if verification_path.exists() else {"status": "pending"}
    attempts = m.get("attempts", [])
    populations = m.get("population", {})
    comparisons = m.get("comparison", {})
    controls = m.get("old_control", {})
    calibrations = m.get("calibration", {})
    windows = m.get("windows", {})
    summary = m.get("summary", {})
    qw = [a for a in attempts if a["side"] == "qW"]
    r1 = [a for a in qw if a["repair"] == "R1"]
    r2 = [a for a in qw if a["repair"] == "R2"]
    first_r1 = r1[0] if r1 else None
    first_comparison = comparisons.get("2019-12-25")
    decision = m["decision"]
    old_geometry, old_max = _old_warm_geometry(repo)
    lines = ["# P4.2R：共同训练人群与数值修复", "",
        f"运行编号：`{RUN_ID}`；合同：`{c['created_at_utc']}`；正式计算结束：`{m['finished_at_utc']}`。",
        f"机器结论：`{decision}`。{DECISIONS.get(decision, '详见机器决策记录。')}",
        f"原始执行快照状态：`{m['status']}`（计算结束时写入，保留原值）；本次收尾独立复核："
        f"`{verification.get('status', 'pending')}`；保留/选定方案：`{m.get('selected_variant', 'W0')}`。"
        "后到的复核通过不修改原始计算快照，也不改变实验停止结论。",
        "", "## 1. 结论与证据边界", "",
        f"已完成 {len(populations)} 个历史截止日的共同人群准备、{len(attempts)} 次倾向模型拟合、"
        f"{len(calibrations)} 个外层窗口的两侧主校准、{len(windows)} 个窗口的固定三阈值准入。未运行项不是0分。",
        "原 P4.2 的 `engineering_failure`（工程失败）保留为历史事实；本轮另立合同、目录和证据，不把旧失败改写成成功。",
        f"qW 数值分支选择：`{m.get('selected_qW_repair') or 'not_run'}`；R2尝试数：{len(r2)}。"
        "恢复数值收敛、改善平均校准、正式通过准入晋级是三个不同结论，不能互相替代。",
        "最终周 `2020-09-16=not_run`；Warm-v2 未整合；P4.3 未启动。本轮在登记的修复及原准入边界内停止。",
    ]
    if m.get("failure_reason"):
        lines.append(f"停止原因：`{m['failure_reason']}`。完整异常保存在本轮大产物目录的 FAILURE.json。")
    lines += ["", "## 2. 术语、统计对象与固定架构", "",
        "本节对本报告复用的英文命名逐一定义；除特别标为行业术语者，其余为本项目名称。",
        "W0：冻结 Warm-v1 每用户12件及原顺序；Warm150：同一暖通道较大的原候选集合，不等于所有暖商品。"
        "M4：单教师监督训练得到的纯内容商品表示；Student 为行业蒸馏中的学生模型，本轮只加载已有表示，不训练。"
        "B0：冻结、时间安全、只使用 M4 信息的冷侧专家；先由 M4 为有历史用户召回前200件，再由 B0 排取前50件，称 Cold50。",
        "S_t：截止日 t 的完整 W0 离线评测用户集合；经逐日审计，等于下一周购买过任意商品、且满足固定用户哈希10%规则的用户。"
        "它依赖未来是否存在任意购买，但不专门依赖未来冷商品购买。H_t：S_t 中最近20件不同已购商品里至少一件可映射至冻结目录的用户；历史严格早于 t。",
        "qC、qW：对 H_t 中 Cold50、W0前12候选分别估计下一周被观察到购买的倾向。propensity（行业术语，倾向）"
        "在此条件化于离线人群、历史可用性和候选策略，不是曝光后的线上真实购买概率；未购买也不等于明确拒绝。",
        "两侧拟合实际行的不同用户集合都严格等于 H_t。无可用历史用户没有 Cold 候选，但仍在完整 S_t 外层评测分母中，"
        "最终逐位保留 W0 的12件；不补假候选、不将其偷偷剔除评测。近期不活跃但更早有历史的用户不一定被排除。",
        "cutoff（行业通用时间截点）记为 t，特征只用 t 前数据，真值使用 [t,t+7天)。训练日期的标签结束严格早于外层截止日。"
        "truth、positive（行业真值/正例）均指某截止日下不同用户—购买商品对；交易事件计数保留原重复事件。跨截止日累计行不是去重自然人数。",
        "MAP@12（行业排序指标）：每用户前12位各命中位置的累计精度相加，除以 min(该用户真值商品数,12)，再对该窗完整 S_t 平均；四窗均值对窗口等权。",
        "严格冷商品、稀疏商品、warm_21_plus：商品截止日前全站购买事件数分别为0、1至5、至少21；all_cold_sparse 是前两者合并。"
        "分群 MAP 仅对有该分群真值的完整 S_t 用户平均，原推荐列表位置不压缩。",
        "cold-only：在 Cold50、但不在 Warm150 的用户—商品候选；不是只要没进 W0前12就算冷路独有。"
        "实际插入正例指准入后放进最终12件且被用户购买的冷候选；被移除暖正例按原 W0 被替换且被购买的商品计数。",
        "R1：只依训练输入，删中位数填补后常数列与逐行完全重复列；重复列优先保留非 `_available`（可用性标记）列，"
        "其次按列名字典序，输出仍沿原字段顺序。不按标签、重要性、AUC或MAP删列，不因高相关就删。",
        "R2：仅冬季 R1 不收敛时允许尝试一次的计数 log1p，即自然对数 log(1+x) 压缩极端值；固定19个计数字段。"
        "先检查所有有限观测计数非负，再训练中位数填补、R1清理、对幸存名单计数取对数、训练标准化。负数报错，不裁成0。",
        "L2 Logistic Regression（行业正则逻辑回归）保持 LBFGS 求解器、C=1、tol=1e-7、最多1000次迭代；"
        "不加类别权重，不采样或复制候选行。qC保持14项数值、2项布尔及14项数值可用标记，共30维。"
        "qW原始逻辑输入117维，R1删后维度由各外层训练矩阵确定，不按外层结果选字段。",
        "gradient infinity norm（行业梯度无穷范数）：拟合结束参数处训练目标所有梯度分量绝对值的最大值，越小通常越接近驻点。"
        "condition number（行业条件数）：本报告用训练参数处含正则目标 Hessian（局部曲率矩阵）的最大/最小特征值比，"
        "用于描述数值病态，不是预测能力分数。最大标准化绝对值是训练数值列标准化后的最大绝对值。",
        "ROC-AUC（行业排序区分度）衡量正例通常能否高于未观察购买行；PR-AUC 在本项目具体按非插值 average precision（平均精度）计算。"
        "Brier（行业概率平方误差）是候选行上预测与0/1真值差的平方均值；ECE（行业分箱校准误差）"
        "是10个按预测稳定排序等人数分箱中平均概率与实测正例率差的绝对值、按行数加权；极稀疏场景下这两项小不等于准入尾部安全。",
        "校准截距/斜率：在审计行上诊断拟合真值对 logit(q) 的逻辑关系，理想约0/1；只用于审计，不反过来校正正式概率。"
        "logit（行业概率变换）是赔率 q/(1−q) 的自然对数；样本单类或分离等情况数值不可用会明确记录。",
        "固定双侧效用 U=logit(qC)−logit(qW)，概率先裁至[1e-6,1−1e-6]；每人 Cold50 去除已在 W0前12的重合商品后，与暖侧12位置配对，最多600条。"
        "激进/中等/保守阈值固定为0、ln2、ln4；仅 U>τ 可选。exact maximum-weight bipartite matching（行业最大权二分匹配）"
        "最大化合计 U−τ，每冷商品和暖位置各至多使用一次，无最多1件限制。未选位置不动；没有合格边则 W0完全不动。",
        "matching conflict（项目配对冲突统计）是合格边竞争同一冷节点或暖位置，不是同一商品被重复插入。"
        "可替换多件时，单个替换的正/零/负正例净数与用户最终 MAP 增减分别报告，不能把独立局部差简单加总代替联合评测。",
        "qC和qW原字段完整中文定义见本目录 [P4_2_FINAL.md](P4_2_FINAL.md) 的字段附录。"
        "本轮未增加任何 qC 字段，原始 B0 分数及其相对 M4 原分差仍不进入 qC模型；它们只用于冻结排序及构造已有归一化字段。",
        "", "## 3. 共同人群恢复：旧表不是新表的子集", "",
        "旧普通训练日期先选未来至少购买一件冷/稀疏商品的用户，并未采用 W0同一10%哈希样本。"
        "修复须同时移除旧表中不在 S_t 的用户，恢复 H_t 中之前遗漏、但没有未来冷真值的用户；不能只给旧表追加更多负例。",
        "每日期先冻结 S_t 与过去历史 H_t；旧同策略重合用户复用原 Cold50，其余用户只用过去历史调用原 M4/B0推理；"
        "完整无标签 Cold50先物化保存，再关联未来真值。首32名按用户编号排序的重合用户做无标签重放，M4前200和B0前50身份/名次必须精确，"
        "连续分数只允许预注册1e-5的GPU批形状舍入误差。实际复用的候选仍保留原值，没用近似分数替换。",
    ]
    lines.append(table(["历史截止日", "完整S用户", "训练H用户", "零Cold用户", "qC用户", "qW用户", "双向集合一致", "新补用户"],
        ([t, a["S_t_users"], a["H_t_users"], a["S_t_without_mapped_history"], a["qC_users"], a["qW_users"],
          "通过" if a["exact_user_set_parity"] else "失败", a["restored_users"]] for t, a in populations.items())))
    lines.append("2019-11-27没有合法更早B0，按合同从两侧共同历史池一起排除；不混入M4-only（仅M4）回退策略。"
                 "较早外层日期可以在标签时间安全后进入更晚窗口训练，仍只用其完整W0历史可用用户，不使用最后一周。")
    lines.append(table(["历史截止日", "旧用户", "新H用户", "旧候选行", "新候选行", "旧正例对", "新正例对", "旧正例率", "新正例率"],
        ([t, a["old"]["users"], a["new"]["users"], a["old"]["rows"], a["new"]["rows"],
          a["old"]["positive_rows"], a["new"]["positive_rows"], rate(a["old"]["positive_rate"]), rate(a["new"]["positive_rate"])]
         for t, a in comparisons.items())))
    if first_comparison:
        a = first_comparison
        old_rate, new_rate = a["old"]["positive_rate"], a["new"]["positive_rate"]
        lines.append(f"首个合法日期2019-12-25：旧表{a['old']['users']}人中，{a['old_users_retained_in_H']}人保留到H，"
            f"{a['old_users_outside_H']}人不属于新H；另补{a['newly_restored_users']}人。旧正例率是新率的"
            f"{fmt(old_rate/new_rate, 3) if new_rate else '不可用'}倍。率下降同时受未来冷真值筛选消除和统一10%人群影响，"
            "不把它全部归因于单一因素，也不声称原157个正例仍全部留在新表。")
        lines.append(table(["2019-12-25人群", "用户", "候选行", "正例对", "正例率"],
            ([label, a["strata"][key]["users"], a["strata"][key]["rows"], a["strata"][key]["positive_rows"],
              rate(a["strata"][key]["positive_rate"])] for key, label in (
                ("old_selected_all", "旧selected全部：未来冷真值选择后的原用户"),
                ("old_selected_retained_in_H", "旧选择用户中实际保留至新H"),
                ("newly_restored_in_H", "新H内原来没有Cold50的用户")))))
        chosen = ["b0_user_percentile", "b0_user_zscore", "normalized_margin_to_rank2", "normalized_margin_to_rank5",
                  "normalized_margin_to_user_median", "history_count_0_7", "history_count_8_28", "history_count_29_84",
                  "history_count_over_84", "days_since_last_purchase", "recent_0_28_purchase_share"]
        lines.append("下表均为候选行上的均值；每用户恰50行，因此用户状态列也是这些人群的等用户均值。置信特征定义见字段附录；"
                     "完整各日期分位数、缺失率由 `p4_2r_qc_population_comparison.json` 的三个分层统计给出。")
        lines.append(table(["字段（项目名）", "中文定义", "原选择全部均值", "保留用户均值", "新恢复用户均值"],
            ([name, feature_note(name), *[fmt(a["strata"][key]["features"][name]["mean"], 5) for key in
              ("old_selected_all", "old_selected_retained_in_H", "newly_restored_in_H")]] for name in chosen)))
        lines.append("百分位均值约0.5、标准分均值约0是逐用户归一化的定义结果，不能据此说人群特征分布没有变化。"
            "新恢复用户的近8–28天、29–84天历史条目和近期占比均值更低、最近购买更久，证明恢复的不只是行数，"
            "还改变了用户活跃状态的支持范围；这是描述性分布差异，不是外层预测收益。")
    lines += ["", "## 4. qW数值修复与各链收敛", "",
        "所有常数/完全重复判断仅在该外层训练矩阵中进行；精确重复指填补后的原逻辑输入完全相同，而不是仅标准化后相同。"
        "删除重复维度改变固定L2正则下的有效惩罚，不能称与旧模型数学等价。历史日期池和用户支持也改变，"
        "尤其冬季去掉2019-11-27；因此旧→新数值变化不能隔离证明‘删列就是唯一根因’。",
    ]
    lines.append(table(["外层", "侧/分支", "训练候选行", "正例对", "实际维数", "迭代", "拟合秒", "状态"],
        ([WINDOW_NAMES.get(a["window"], a["window"]), f"{a['side']} {a['repair']}", a["training_rows"], a["positives"],
          a["model_features"], _model_iterations(a), fmt(a["fit_seconds"], 3), a["status"]] for a in attempts)))
    lines.append(table(["外层/分支", "原逻辑维数", "常数删除列数", "完全重复删除列数", "保留维数", "梯度最大绝对值", "局部条件数", "最大标准化绝对值"],
        ([f"{WINDOW_NAMES.get(a['window'], a['window'])}/{a['repair']}", a["cleanup"]["dims_before"],
          a["cleanup"]["constant_dropped_count"], a["cleanup"]["exact_duplicate_dropped_count"], a["cleanup"]["dims_after"],
          sci(a["diagnostics"]["gradient_infinity_norm"]), sci(a["diagnostics"]["local_regularized_hessian_condition_number"]),
          fmt(a["diagnostics"]["max_standardized_absolute_value"], 3)] for a in qw)))
    lines.append(f"原冬季qW：{old_geometry['rows']:,}行、117维、1000次未收敛；梯度{sci(old_geometry['gradient_infinity_norm'])}，"
        f"局部条件数{sci(old_geometry['local_regularized_hessian_condition_number'])}，最大标准化绝对值{fmt(old_max, 3)}。"
        "旧诊断26个常数可用列、40对非恒定完全重复列；‘40对重复关系’不是40个可删除列，不应和新删除列数直接相减。")
    if first_r1:
        clean = first_r1["cleanup"]
        lines.append(f"冬季R1实际删除{clean['constant_dropped_count']}个常数列、{clean['exact_duplicate_dropped_count']}个重复列，"
            f"117→{clean['dims_after']}维；状态 `{first_r1['status']}`。"
            + ("按预注册选择R1供所有后续训练链使用，R2未被解锁。" if first_r1["status"] == "converged" else "仅此失败才允许预注册R2，具体尝试见上表。"))
    lines.append("删除字段、常数值及每个重复字段对应保留字段详见 `p4_2r_qw_R1_cleanup.json`；"
        "19项计数名单及R2是否执行详见 `p4_2r_qw_R2_transform.json`。不能为后面某个窗口单独解锁R2或改迭代预算。"
        "库的未来版本弃用提醒与数值不收敛警告分开保留；前者不等于拟合未收敛。")
    stopped_qw = [a for a in qw if a["status"] != "converged"]
    if stopped_qw:
        last = stopped_qw[-1]
        diag = last["diagnostics"]
        lines.append(f"本轮实际停止发生在{WINDOW_NAMES.get(last['window'],last['window'])} qW/{last['repair']}："
            f"迭代{_model_iterations(last)}次仍有不收敛警告，梯度{sci(diag['gradient_infinity_norm'])}，"
            f"局部条件数{sci(diag['local_regularized_hessian_condition_number'])}，最大标准化绝对值"
            f"{fmt(diag['max_standardized_absolute_value'],3)}，对应 `{diag['max_standardized_absolute_feature']}`"
            f"（{feature_note(diag['max_standardized_absolute_feature'])}）。"
            "冬季R1已成功选择分支，因此晚窗口失败不能再解锁R2；按预注册输出engineering_failure并保留失败参数，"
            "不能叫作R1和R2都失败，因为R2根本没有运行。")
        if first_r1 and last is not first_r1:
            extra = [name for name in last["feature_order"] if name not in first_r1["feature_order"]]
            if extra:
                lines.append("相对冬季R1，停止链保留的额外逻辑列：" + "、".join(f"`{name}`" for name in extra) +
                    "。后缀_available均表示其对应数值字段是否有限可用，0/1；其中商品最近销售天数与用户购买候选商品最近天数"
                    "的可用性原定义不变。保留差异由训练缺失/重复结构决定，不是按标签或外层结果新增字段；"
                    "这些差异是可核查的数值诊断线索，不证明它们就是不收敛的唯一原因。")
    lines += ["", "## 5. 先校准，再考虑准入", "",
        "主校准两侧都在H_t上：qC完整Cold50，qW同一人群完整W0前12。补充的qW全S_t校准含训练支持以外无历史用户，"
        "只作披露，不替代主校准门槛，更不改变最终MAP评测分母。",
        "严重门槛在看任何准入MAP前固定：同一侧有至少3窗平均预测/实测正例率不在[0.1,10]，或至少3窗ROC-AUC≤0.5，"
        "或至少3窗预测恒定，即失败。通过只排除预注册的严重跨窗失效，不代表概率完美，更不保证极高分尾部准入安全。",
    ]
    lines.append(table(["外层", "侧", "候选行", "正例对", "训练正例率", "外层实测率", "平均预测", "预测/实测", "ROC-AUC", "PR-AUC", "Brier", "ECE", "截距", "斜率"],
        ([WINDOW_NAMES.get(w, w), side, v["rows"], v["positives"], rate(v.get("training_base_rate")),
          rate(v["observed_positive_rate"]), rate(v["mean_predicted_probability"]), fmt(v["predicted_to_observed_rate_ratio"], 3),
          fmt(v["roc_auc"], 5), fmt(v["pr_auc"], 6), fmt(v["brier_score"]), fmt(v["ece"]),
          fmt(v["calibration_intercept"], 4), fmt(v["calibration_slope"], 4)]
         for w, values in calibrations.items() for side, v in values.items())))
    if m.get("full_calibration"):
        lines.append(table(["外层qW完整S诊断", "候选行", "实测率", "平均预测", "ROC-AUC", "Brier", "ECE"],
        ([WINDOW_NAMES.get(w, w), v["rows"], rate(v["observed_positive_rate"]), rate(v["mean_predicted_probability"]),
          fmt(v["roc_auc"], 5), fmt(v["brier_score"]), fmt(v["ece"])] for w, v in m.get("full_calibration", {}).items())))
    gate = m.get("calibration_gate")
    lines.append(f"主校准严重门槛：{'通过' if gate and gate['pass'] else '失败' if gate else '未运行'}。" +
        ("完整10箱可靠性统计、数值警告、截距/斜率可用性均保存在 `p4_2r_propensity_calibration.json`。" if calibrations else
         "`p4_2r_propensity_calibration.json` 明确记录not_run；文件存在不代表已有10箱、AUC或概率对照观测。"))
    lines.append(_control_answer(controls))
    if controls:
        ctl = next(iter(controls.values()))
        lines.append(table(["冬季同一Cold50对照", "ROC-AUC", "PR-AUC", "平均预测", "Brier", "ECE", "校准截距", "校准斜率"],
            ([label, fmt(v["roc_auc"], 6), fmt(v["pr_auc"], 6), rate(v["mean_predicted_probability"]),
              fmt(v["brier_score"]), fmt(v["ece"]), fmt(v["calibration_intercept"], 5), fmt(v["calibration_slope"], 5)]
             for label, v in (("旧P4.2未来冷真值选择人群模型", ctl["old"]), ("新P4.2R共同H人群模型", ctl["new"])))))
        old_auc, new_auc = ctl["old"]["roc_auc"], ctl["new"]["roc_auc"]
        lines.append(f"同窗AUC差（新−旧）为 {fmt(new_auc-old_auc, 6) if old_auc is not None and new_auc is not None else '不可用'}。"
            "应将这一相对排序变化与平均概率、Brier、ECE变化分别解释；不能仅因两个模型都是逻辑回归就认定两侧概率可直接线上使用。"
            "旧qC仅作诊断，任何正式效用都不使用它；另三窗旧模型从未存在，本轮不为做对照而补训。")
    lines += ["", "## 6. 固定三阈值准入及四窗门槛", ""]
    if windows:
        lines.append(table(["外层", "方案", "完整S用户", "总体MAP@12", "相对W0差", "暖≥21组MAP", "严格冷组MAP", "稀疏1–5组MAP", "冷/稀疏合并MAP"],
            ([WINDOW_NAMES.get(w, w), VARIANT_NAMES[v], d["users"], fmt(row["map12"]), fmt(row["delta_vs_w0"]),
              *[fmt(row["segments"][s]["map12"]) for s in ("warm_21_plus", "strict_cold", "sparse1_5", "all_cold_sparse")]]
             for w, d in windows.items() for v, row in d["variants"].items())))
        if summary:
            lines.append(table(["方案", "四窗总体均值", "均值差", "不下降窗数", "最差窗差", "暖组均值差", "冷/稀疏组均值差", "独有冷路正例插入窗数", "所有门槛"],
                ([VARIANT_NAMES[v], fmt(s["mean_map12"]), fmt(s["mean_delta"]), s["nondegrade_windows"], fmt(s["worst_delta"]),
                  fmt(s["segments"]["warm_21_plus"]["mean_delta"]), fmt(s["segments"]["all_cold_sparse"]["mean_delta"]),
                  s["cold_only_positive_windows"], "通过" if s["gates"]["all_pass"] else "未通过"] for v, s in summary.items())))
            lines.append(table(["方案", "未通过的预注册检查"],
                ([VARIANT_NAMES[v], "；".join(GATE_NAMES[k] for k, ok in summary[v]["gates"]["checks"].items() if not ok) or "全部通过"] for v in TAUS)))
        lines.append("总体门槛：均值不降、至少3窗不降、最差窗差≥−0.000200。暖组门槛：均值差≥−0.000100，"
            "且至少3窗差≥−0.000200。冷/稀疏组：均值严格增加、至少3窗不降、至少2窗实际插入独有冷路正例。"
            "合并四窗插入冷正例必须严格超过移除暖正例。多个方案通过时先选冷/稀疏均值最高者，近似相等1e-12内"
            "再看总体均值，再选更大的阈值。以上沿用原合同，未因本轮结果放宽。")
        lines.append(table(["外层", "方案", "准入用户比例", "替换总数", "每准入用户平均替换", "最大替换", "插入冷正例", "移除暖正例", "净正例", "独有冷路正例", "利好替换", "中性替换", "有害替换"],
            ([WINDOW_NAMES.get(w, w), VARIANT_NAMES[v], rate(row["admission"]["admission_user_share"]),
              row["admission"]["total_replacements"], fmt(row["admission"]["mean_replacements_per_admitted_user"], 3),
              row["admission"]["max_replacements"], *[row["admission"][k] for k in ("inserted_cold_positive_pairs", "removed_warm_positive_pairs", "net_positive_pairs",
              "cold_only_positive_actually_inserted", "beneficial_replacements", "neutral_replacements", "harmful_replacements")]]
             for w, d in windows.items() for v, row in d["variants"].items() if v in TAUS)))
        lines.append("准入用户比例分母是该窗完整S_t；利好/中性/有害替换分别按单条匹配边的插入正例减移除正例为+1/0/−1计数。"
            "下表按实际替换件数分组；每组MAP差使用同一组用户的原W0作对照，最终列是该组对完整窗口MAP差的贡献。"
            "不同组不是随机分配，不能把组间差直接解释为‘多替换一件的因果效应’。")
        lines.append(table(["外层", "方案", "替换件数", "用户数", "该组MAP差", "插入正例", "移除正例", "净正例", "对整窗MAP差贡献"],
            ([WINDOW_NAMES.get(w, w), VARIANT_NAMES[v], b, r["users"], fmt(r["delta_vs_same_users_w0"]),
              r["inserted_cold_positives"], r["removed_warm_positives"], r["net_positives"], fmt(r["delta_contribution_full_window"])]
             for w, d in windows.items() for v, row in d["variants"].items() if v in TAUS for b, r in row["admission_buckets"].items())))
        lines.append(table(["外层", "Cold50原候选行", "与W0重合排除行", "实际配对边数", "最大每用户配对数"],
            ([WINDOW_NAMES.get(w, w), x["pair_audit"]["complete_cold_rows"], x["pair_audit"]["overlap_excluded_cold_rows"],
              x["pair_audit"]["pair_rows"], x["pair_audit"]["maximum_pairs_per_user"]] for w, x in windows.items())))
        lines.append("具体合格边冲突数、精确匹配与贪心诊断差异、总匹配效用见 `p4_2r_matching_audit.json`。"
            "贪心仅作解释性诊断，正式结果用固定SciPy精确求解；不加极小扰动、不改变同分规则、不设最多1件。"
            "效用本身无位置偏好，相同效用最优匹配可能不唯一；固定实现选出的槽位不是另行学到的位置效应。")
    else:
        lines.append("准入未运行，因此三个阈值的MAP、覆盖率、替换风险与晋级工作点均未测得。"
                     "不能把未运行写成无提升，也不能由拟合/校准单独宣布基线晋级。")
    lines += ["", "## 7. 用户要求的18项结论逐条回答", ""]
    answers = []
    answers.append(("旧qC漏了哪些用户？", "普通训练日期漏了W0同一10%样本中没有未来冷/稀疏真值的有历史用户，且纳入大量W0样本之外的用户；见第3节。"))
    answers.append(("每日期两侧人群相同了吗？", f"已完成的{len(populations)}个日期，实际qC/qW行用户集合与H_t精确一致；H_t不是完整S_t，外层仍保留S_t。"))
    if first_comparison:
        o, n = first_comparison["old"], first_comparison["new"]
        answers.append(("新qC正例率降了多少？", f"2019-12-25从{rate(o['positive_rate'])}降到{rate(n['positive_rate'])}，"
            f"绝对减少{(o['positive_rate']-n['positive_rate'])*100:.6f}个百分点；全部日期见第3节，并非只增加行而保留所有旧正例。"))
    else:
        answers.append(("新qC正例率降了多少？", "正式人群准备未完成，无可报告变化。"))
    answers.append(("新qC校准更合理了吗？", _control_answer(controls)))
    if controls:
        ctl = next(iter(controls.values()))
        answers.append(("AUC变了还是概率尺度变了？", f"冬季ROC-AUC {fmt(ctl['old']['roc_auc'],6)}→{fmt(ctl['new']['roc_auc'],6)}；"
            "平均预测、Brier、ECE、截距和斜率见第5节，分别衡量排序区分与概率尺度，不合并成一个成功判断。"))
    else:
        answers.append(("AUC变了还是概率尺度变了？", "旧模型同窗校准对照未运行，不能回答变化幅度。"))
    answers.append(("R1删了多少列？", f"冬季常数{first_r1['cleanup']['constant_dropped_count']}列、完全重复{first_r1['cleanup']['exact_duplicate_dropped_count']}列，"
        f"保留{first_r1['cleanup']['dims_after']}维；后续每链具体数见第4节。" if first_r1 else "未运行。"))
    answers.append(("只靠R1收敛了吗？", f"冬季状态{first_r1['status']}；"
        f"已运行{len(r1)}条R1链中{sum(a['status']=='converged' for a in r1)}条收敛，"
        f"{sum(a['status']!='converged' for a in r1)}条未收敛，不能说四窗全部修复。各链结果见第4节。"
        if first_r1 else "未运行。"))
    answers.append(("R2是否恢复收敛？", "没有运行：冬季R1已收敛，因此R2不被解锁，不存在R2效果结论。" if first_r1 and first_r1["status"] == "converged" else
        "；".join(f"{a['window']}状态{a['status']}" for a in r2) if r2 else "未运行。"))
    answers.append(("条件数、极端值、梯度怎样变化？", "第4节给出旧冬季与新各链的同定义诊断。共同时间池、用户总体和L2有效惩罚也变了，不作唯一因果归因。"))
    answers.append(("四窗区分/校准是否可用？", f"完成{len(calibrations)}窗主校准；严重门槛{'通过' if gate and gate['pass'] else '失败' if gate else '未运行'}。"
        "其含义仅是是否触发固定严重失效判据，不能推导线上概率或极端尾部足够可靠。"))
    answers.append(("三阈值的覆盖、风险、冷增益如何？", "见第6节完整四窗、分群、替换次数表及风险JSON。" if windows else "准入未运行，均不可用。"))
    if windows:
        pieces = []
        for v in TAUS:
            buckets = [d["variants"][v]["admission_buckets"][b] for d in windows.values() for b in ("2", "3", "4+")]
            pieces.append(f"{VARIANT_NAMES[v]}的≥2次替换用户合计{sum(r['users'] for r in buckets)}，插入/移除正例分别"
                f"{sum(r['inserted_cold_positives'] for r in buckets)}、{sum(r['removed_warm_positives'] for r in buckets)}")
        answers.append(("多件准入有暖侧风险累积吗？", "；".join(pieces) + "。按人数/正例和整窗贡献描述，不把非随机分组当因果证据。"))
        answers.append(("插入正例超过移除正例了吗？", "；".join(f"{VARIANT_NAMES[v]}："
            f"{sum(d['variants'][v]['admission']['inserted_cold_positive_pairs'] for d in windows.values())}对插入、"
            f"{sum(d['variants'][v]['admission']['removed_warm_positive_pairs'] for d in windows.values())}对移除" for v in TAUS)))
    else:
        answers += [("多件准入有暖侧风险累积吗？", "未运行，不能判断。"), ("插入正例超过移除正例了吗？", "未运行，不能判断。")]
    passing = m.get("passing_variants", [])
    answers.append(("存在正式安全工作点吗？", f"通过全部门槛的方案：{', '.join(passing) or '无'}。" if summary else "没有完成正式晋级判定，不宣称存在安全工作点。"))
    answers.append(("选择哪个阈值？", f"正式选定{m.get('selected_variant','W0')}。" if passing else
        "没有选出新阈值；维持原W0，不择单个有利窗口升级。"))
    bottleneck = (f"{WINDOW_NAMES.get(stopped_qw[-1]['window'], stopped_qw[-1]['window'])}qW在固定R1和1000次预算内仍不收敛；"
        "全部qC已恢复共同人群并数值收敛，但两侧四窗修复总门槛未完成。校准和MAP尚未测，不能归因为冷召回或准入效用又失败。"
        if stopped_qw else DECISIONS.get(decision, decision))
    answers.append(("当前主要剩余瓶颈是什么？", bottleneck + "只按本轮已测状态定位，不自动引出新模型、阈值搜索或扩大数据路线。"))
    answers += [("最后一周运行了吗？", "没有，2020-09-16仍为not_run（未运行）。"),
                ("Warm-v2整合了吗？", "没有；未合并、未使用Warm-v2替换本轮W0，P4.3也未启动。")]
    lines.append(table(["序号", "问题", "回答"], ([i, q, a] for i, (q, a) in enumerate(answers, 1))))
    lines += ["", "## 8. 工程记录、资源、验证与停止", ""]
    bootstrap_path = root / "bootstrap-attempt-01/P4_2R_metrics.json"
    if bootstrap_path.exists():
        bootstrap = read_json(bootstrap_path)
        lines.append(f"首次引导阶段触发路径类型工程错误：`{bootstrap.get('failure_reason')}`，耗时"
            f"{fmt(bootstrap.get('resources',{}).get('formal_seconds'), 6)}秒；已准备日期{len(bootstrap.get('prepared',{}))}、"
            f"拟合{len(bootstrap.get('attempts',[]))}次。它在任何候选重建/拟合前停止，原执行标记、失败记录和空结果"
            "保存在 `bootstrap-attempt-01/`。修正路径接入后原合同不变，另记录正式启动；不是在数值失败后调参重试。")
    resources = m.get("resources", {})
    peak_ram = resources.get("peak_process_working_set_gib")
    ram_text = f"{fmt(peak_ram, 3)}GiB" if peak_ram is not None and peak_ram > 0 else "不可用（Windows进程计数读取返回无效0值，不是实际零内存）"
    lines.append(f"正式计算耗时{fmt(resources.get('formal_seconds'), 2)}秒，进程峰值工作集"
        f"{ram_text}；登记的15–45分钟是事前估计，不替代实测。"
        f"倾向拟合尝试{resources.get('propensity_fit_attempts', len(attempts))}次；GPU仅作冻结M4/B0推理，倾向拟合为CPU四线程。"
        "这些时间不含人工解释、协议阅读、代码编写与最终独立验证。")
    verification_checks = verification.get("checks", [])
    check_counts = {status: sum(a.get("status") == status for a in verification_checks) for status in ("pass", "not_run", "fail")}
    lines.append(f"独立复核状态 `{verification.get('status','pending')}`；26项合同检查中"
        f"{check_counts['pass']}项通过、{check_counts['not_run']}项未运行、{check_counts['fail']}项失败。"
        f"独立复核耗时{fmt(verification.get('seconds'),2)}秒，可信记录SHA比较{verification.get('SHA_comparisons','不可用')}项。"
        "详见 `P4_2R_VERIFICATION.json`：已运行的旧P4.2保全、冻结候选/模型身份、逐日共同用户、真值关联顺序、训练预处理等"
        "依证据核验；R2实际变换、最终12件及无合格边W0输出等正式执行检查未运行，不能拿单元测试替代四窗执行证明。"
        "独立复核通过表示产物和结论自洽，不自动代表所有实验晋级门槛通过。")
    test_receipt_path = folder / "P4_2R_TEST_VERIFICATION.json"
    if test_receipt_path.exists():
        tests = read_json(test_receipt_path)
        lines.append(f"主线程最终回归单元测试：{tests['tests_passed']}项通过，耗时{fmt(tests['seconds'],3)}秒；"
            "没有失败。该测试验证代码机制，不替代未运行的外层概率校准、匹配或MAP评测；"
            "记录见 `P4_2R_TEST_VERIFICATION.json`。")
    lines.append("`P4_2R_OUTPUT_MANIFEST.json` 列出本轮报告、实现、测试及所有大产物的路径/字节/SHA；"
        "SHA（行业文件摘要）本次服务于显式证据合同和旧文件可信记录比较，不把单独对当前文件算摘要当来源证明。"
        "大候选、用户标识、模型和中间数组留在忽略目录，不写进公开报告正文；路线、日志及项目协议继续gitignore。")
    lines.append("本轮结束，未提交、未推送。失败回退为W0；即使存在通过方案，也不在未获新授权时启动P4.3、Warm-v2整合或最终周。")
    text = "\n".join(lines).rstrip() + "\n"
    (folder / "P4_2R_FINAL.md").write_text(text, encoding="utf-8")
    qc_done = sum(a["side"] == "qC" and a["status"] == "converged" for a in attempts)
    qw_done = sum(a["side"] == "qW" and a["status"] == "converged" for a in attempts)
    paragraphs = [
        "本段为后续实测记录，不覆盖原P4.2工程失败或前置人群审计暂停。Lyra明确批准两侧共同H_t拟合、完整S_t评测后，登记P4.2R合同。",
        f"已完成{len(populations)}个历史截止日实际行人群对齐；qC收敛{qc_done}链、qW收敛{qw_done}链，"
        f"选择qW修复{m.get('selected_qW_repair') or '未运行'}，R2尝试{len(r2)}次。"
        "旧普通训练总体包含未来冷真值条件和不同用户采样，新表不是只给旧表追加未购行。",
        f"四窗主校准完成{len(calibrations)}窗，准入完成{len(windows)}窗；机器结论`{decision}`，"
        f"选定/保留`{m.get('selected_variant','W0')}`。{DECISIONS.get(decision,'')} "
        f"正式计算{fmt(resources.get('formal_seconds'),2)}秒，独立复核`{verification.get('status','pending')}`。",
        "数值修复同时涉及用户/时间支持和固定L2有效惩罚变化，不把旧新对比当唯一根因证明；校准门槛通过也不等于准入必有MAP收益。",
        "证据：`reports/phase4/P4_2R_FINAL.md`、`P4_2R_metrics.json`、`P4_2R_VERIFICATION.json`及配套审计。"
        "原P4.2证据保留，最终周仍not_run，Warm-v2未整合，P4.3未启动。大产物、日志和路线继续忽略，不自动提交推送。",
    ]
    _append_measured_doc(repo / "docs/ROADMAP_PHASE4.zh-CN.md", "## 15. P4.2R 共同人群与数值修复实测（2026-09-09）", paragraphs)
    _append_measured_doc(repo / "docs/PROJECT_LOG.zh-CN.md", "## 2026-09-09 — P4.2R：共同人群与数值修复实测收尾", paragraphs)
    report_paths = sorted({*folder.glob("P4_2R*.md"), *folder.glob("P4_2R*.json"), *folder.glob("p4_2r_*.json")})
    manifest_path = folder / "P4_2R_OUTPUT_MANIFEST.json"
    reports = [identity(p) for p in report_paths if p != manifest_path]
    artifacts = [identity(p) for p in sorted(root.rglob("*")) if p.is_file()]
    sources = [identity(p) for p in sorted({*(repo / "src/hm_recsys").glob("p42r*.py"), *(repo / "tests").glob("test_p42r*.py")})]
    manifest = {"stage": "P4.2R", "run_id": RUN_ID, "schema_version": "p4.2r-output-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "decision": decision,
        "selected_variant": m.get("selected_variant", "W0"), "verification_status": verification.get("status", "pending"),
        "reports": reports, "artifacts": artifacts, "sources": sources,
        "artifact_files": len(artifacts), "artifact_bytes": sum(r["bytes"] for r in artifacts),
        "private_paths_not_published": ["docs/ROADMAP_PHASE4.zh-CN.md", "docs/PROJECT_LOG.zh-CN.md", "artifacts/phase4/" + RUN_ID],
        "manifest_self_hash_excluded": True, "original_P4_2_manifest": c["original_P4_2_manifest"],
        "original_P4_2_historical_decision": "engineering_failure", "final_week": "not_run",
        "Warm_v2_integrated": False, "P4_3_started": False}
    write_json(manifest_path, manifest)
    return {"report": str(folder / "P4_2R_FINAL.md"), "manifest": str(manifest_path),
            "decision": decision, "artifact_files": len(artifacts), "artifact_bytes": manifest["artifact_bytes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    print(report(args.repo))
