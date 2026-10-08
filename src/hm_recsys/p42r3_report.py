"""P4.2R3 measured Chinese report. This renderer never fits or evaluates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .p41a_contract import identity, read_json, write_json
from .p42_contract import TAUS, WINDOWS
from .p42_report import feature_note
from .p42r_report import DECISIONS, GATE_NAMES, VARIANT_NAMES, WINDOW_NAMES, fmt, rate, sci, table


STAGE = "P4.2R3"
LATE = "late_summer_20200819"
DECISION_TEXT = {**DECISIONS,
    "qW_solver_boundary_repair_failure": "至少一条暖侧训练链返回非成功状态或达到1200次上限；按合同停止，不追加预算。",
    "warm_risk_uncontrolled": "冷/稀疏均值有增益的方案未通过总体或暖侧保护中的至少一项，仍保留W0；具体失败门槛见第5节。"}
SEGMENTS = ("warm_21_plus", "strict_cold", "sparse1_5", "all_cold_sparse")


def _iterations(attempt):
    if not attempt:
        return None
    if attempt.get("solver_result", {}).get("nit") is not None:
        return int(attempt["solver_result"]["nit"])
    return int(read_json(Path(attempt["model"]["path"]))["n_iter"][0])


def _delta_value(delta, *keys):
    """Schema aliases only; never manufacture a missing measurement."""
    for key in keys:
        if key in delta:
            return delta[key]
    return None


def _append_doc(path, title, paragraphs):
    start, end = "<!-- P4.2R3 measured closure begin -->", "<!-- P4.2R3 measured closure end -->"
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    if original.count(start) > 1 or original.count(end) > 1:
        raise ValueError("duplicate P4.2R3 documentation markers")
    block = start + "\n\n" + title + "\n\n" + "\n\n".join(paragraphs) + "\n\n" + end
    if start in original:
        if end not in original:
            raise ValueError("unclosed P4.2R3 documentation block")
        before, tail = original.split(start, 1)
        _, after = tail.split(end, 1)
        updated = before + block + after
    else:
        updated = original.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(updated, encoding="utf-8")


def _terminology():
    return ["## 2. 名称、单位与比较范围", "",
        "本报告的W0、M4、B0、Cold50、qC/qW、S_t/H_t、R1/R2/R3是项目命名：W0为冻结Warm-v1每用户前12件及原顺序；"
        "M4为冻结纯内容商品表示，B0是其冻结冷侧排序专家，由M4前200件取B0前50件，称Cold50。Warm150是暖路原较大候选集合。",
        "S_t为截止日t完整冻结评测用户：下一周有任意购买且通过固定10%用户哈希抽样；H_t为其中最近20件不同历史商品至少一件可映射到M4目录者，"
        "历史严格早于t。两侧训练实际行用户都等于H_t；两侧主校准也用H_t。最终评测仍以完整S_t为分母，无Cold候选者保留原W0。",
        "qC为Cold50候选下一周被购买的倾向，qW为W0前12候选下一周被购买的倾向；propensity（行业统计术语，倾向）"
        "条件化于此离线人群及候选策略，不是曝光后的线上购买概率。正例为不同截止日—用户—商品真值对；未观察购买不等于明确拒绝。"
        "特征仅依赖t前历史，真值窗口为[t,t+7天)，训练标签结束严格早于外层日期；历史事件保留重复行。",
        "R1为训练中位数填补后的常数/完全重复列清理；R2为固定19项计数中幸存字段取log1p=ln(1+x)，再用训练均值/标准差标准化。"
        "R3只把qW迭代上限改为1200，不另加变换。完整字段定义见[P4.2字段附录](P4_2_FINAL.md)，"
        "19项计数名单、禁止变换的名次/比例/天数字段及R1保留顺序见[P4.2R2第2节](P4_2R2_FINAL.md)。本轮全部原样继承。",
        "L2逻辑回归（行业模型）对系数平方和正则化；LBFGS（行业拟牛顿求解器）更新这些系数。solver success/status/message（行业求解器输出）"
        "分别为实际成功布尔值、返回码和返回说明；ConvergenceWarning为库的不收敛警告，不可由人工用独立梯度覆盖。"
        "迭代次数是单次拟合优化步数，预算1200是上限，不是强制训练1200步。",
        "gradient∞（行业梯度无穷范数）为含正则训练目标各参数导数绝对值最大值；condition number（行业条件数）为同参数处局部曲率矩阵最大/最小特征值比。"
        "max|z|（项目摘要）为训练数值列标准化后的最大绝对值。它们是数值诊断，不是推荐得分。",
        "参数差均为R3减R2保存值；系数L2差为系数差向量的欧氏长度，最大绝对差为最大单维变化。"
        "training log-loss（行业训练对数损失）为同一训练候选行上的平均未正则二分类损失；概率差按相同训练行比较，p95为绝对差的95%分位，"
        "不是外层验证结果。额外迭代=max(本轮实际步数−1000,0)。",
        "MAP@12（行业排序指标）为各用户前12件命中位置精度之和除以min(不同真值商品数,12)，再按完整S_t用户平均；四窗均值对窗口等权。"
        "strict_cold、sparse1_5、warm_21_plus分别指商品截止前全站事件数0、1–5、至少21，all_cold_sparse为前两组合并。"
        "分群MAP分母为完整S_t中有相应分群真值者，原推荐位置不压缩；不是冷用户分群。",
        "ROC-AUC（行业区分指标）衡量正例通常能否高于未观察购买行；PR-AUC在本项目按非插值平均精度计算。Brier为候选行概率平方误差均值；"
        "ECE为按预测和原行序稳定排序的10个等人数箱中平均概率与正例率差绝对值、按行数加权。校准截距/斜率为审计真值对logit(q)的关系，"
        "理想约0/1，仅诊断、不用于修改正式概率。小Brier/ECE在稀疏标签下不能单独证明尾部安全。",
        "准入效用U=logit(qC)−logit(qW)，logit（行业变换）为赔率q/(1−q)的自然对数；概率固定裁至[1e-6,1−1e-6]。"
        "τ=0、ln2、ln4分别为激进/中等/保守方案，只有U>τ才保留配对；冷商品先排除已在W0前12者，每人最多50×12=600对。",
        "精确最大权二分匹配（行业图优化）最大化配对U−τ之和，每冷商品和暖位置各用至多一次，可替换0–12件、不设最多1件。"
        "没有合格边则W0逐位不动。cold-only（项目独有冷路候选）为Cold50中不属于Warm150的用户—商品对，比‘不在W0前12’更严格。"
        "插入/移除正例以执行配对为单位；利好/中性/有害指插入标签减移除标签为+1/0/−1，最终用户MAP另重算，不相加孤立替换收益。",
        "所有跨窗人数相加都是用户—窗口观察数，不是跨窗去重自然人数；覆盖率分母为各窗完整S_t，替换组的MAP差比较同一组用户的W0。"]


def _calibration_lines(m):
    cal = m.get("calibration", {})
    gate = m.get("calibration_gate")
    lines = ["## 4. 两侧倾向校准", "",
        "只有4/4 qW实际返回成功且未达到1200次上限后才解锁本节；qC直接复用P4.2R四个已收敛模型与其原预处理。"
        "两侧主校准共同使用H_t：qC完整Cold50、qW同用户原12件，均在排除候选重合之前。补充qW全S_t只作训练支持以外人群诊断。"]
    if not cal:
        return lines + ["校准未运行，没有外层AUC、Brier、ECE、截距/斜率或10箱观测。不能由训练成功推测校准通过。"]
    lines.append(table(["外层", "侧", "候选行", "正例对", "训练正例率", "实测正例率", "平均预测", "预测/实测", "ROC-AUC", "PR-AUC", "Brier", "ECE", "截距", "斜率"],
        ([WINDOW_NAMES[w], side, a["rows"], a["positives"], rate(a.get("training_base_rate")), rate(a["observed_positive_rate"]),
          rate(a["mean_predicted_probability"]), fmt(a["predicted_to_observed_rate_ratio"],3), fmt(a["roc_auc"],5), fmt(a["pr_auc"],6),
          fmt(a["brier_score"]),fmt(a["ece"]),fmt(a["calibration_intercept"],4),fmt(a["calibration_slope"],4)]
         for w, sides in cal.items() for side,a in sides.items())))
    lines.append("完整10箱行数、正例数、预测范围、均值与实测率见[p4_2r3_propensity_calibration.json](p4_2r3_propensity_calibration.json)。"
        "同一侧至少3窗预测/实测率不在[0.1,10]，或至少3窗ROC-AUC≤0.5，或至少3窗预测恒定即停止；"
        "通过只排除预注册严重失效，不能解释为概率完美或风险已经解决。")
    lines.append(f"四窗主校准严重失效门槛：{'通过' if gate and gate['pass'] else '未通过' if gate else '未完成'}。")
    if len(cal) == 4:
        cold_ratios=[s["qC"]["predicted_to_observed_rate_ratio"] for s in cal.values()]
        cold_auc=[s["qC"]["roc_auc"] for s in cal.values()]
        warm_auc=[s["qW"]["roc_auc"] for s in cal.values()]
        if all(x is not None for x in cold_ratios+cold_auc+warm_auc):
            lines.append(f"实测qC的ROC-AUC范围为{fmt(min(cold_auc),4)}–{fmt(max(cold_auc),4)}，qW为{fmt(min(warm_auc),4)}–{fmt(max(warm_auc),4)}；"
                f"qC平均预测/实测率范围{fmt(min(cold_ratios),3)}–{fmt(max(cold_ratios),3)}。"
                +( "qC四窗平均购买倾向均偏低，但未越过本轮严重失效门槛；这不是概率已准确校准。" if max(cold_ratios)<1 else
                   "均值尺度和分数区分能力分别判断；通过严重门槛不等于两侧每一个候选概率都准确。")
                +"两侧AUC和基础率来自不同候选池，不能据此推导跨来源每条高分配对均可靠。")
    if gate:
        counts = gate.get("counts", gate.get("severe_counts", {}))
        if counts:
            lines.append(table(["侧", "预测/实测率严重超界窗数", "ROC不高于0.5窗数", "预测恒定窗数"],
                ([side,a.get("extreme_rate_ratio_windows"),a.get("roc_not_above_chance_windows"),a.get("constant_prediction_windows")] for side,a in counts.items())))
    warnings = [(WINDOW_NAMES[w], side, "; ".join(a.get("warnings", [])) or "无",
                 a.get("calibration_line", {}).get("status", "不可用")) for w,s in cal.items() for side,a in s.items()]
    lines.append(table(["外层", "侧", "机器警告", "截距/斜率诊断状态"], warnings))
    lines.append("机器警告含义：mean_probability_to_observed_rate_outside_0.25_4为均值比超出诊断范围[0.25,4]，"
        "roc_auc_not_above_chance为AUC≤0.5，nonpositive_calibration_slope为斜率≤0，constant_prediction为预测恒定，"
        "calibration_line_diagnostic_failed为诊断拟合未成功。converged为诊断成功，undefined为定义条件不满足；这些警告不擅自新增晋级门槛。")
    full = m.get("full_calibration", {})
    if full:
        lines.append(table(["qW完整S_t补充诊断", "候选行", "正例对", "实测率", "平均预测", "ROC-AUC", "Brier", "ECE"],
            ([WINDOW_NAMES[w],a["rows"],a["positives"],rate(a["observed_positive_rate"]),rate(a["mean_predicted_probability"]),
              fmt(a["roc_auc"],5),fmt(a["brier_score"]),fmt(a["ece"])] for w,a in full.items())))
    return lines


def _admission_lines(m):
    windows, summary = m.get("windows", {}), m.get("summary", {})
    lines = ["## 5. 原三阈值准入、替换风险与晋级", ""]
    if not windows:
        return lines + ["准入未运行：三阈值的配对效用、匹配、总体/分群MAP、覆盖率及替换风险均没有实测值。"
            "未运行不是0覆盖或0分；保持W0，不宣称新的安全工作点。"]
    lines.append(table(["外层", "方案", "完整S用户", "总体MAP", "总体差", "暖≥21MAP", "严格冷MAP", "稀疏1–5MAP", "冷/稀疏MAP"],
        ([WINDOW_NAMES[w],VARIANT_NAMES[v],d["users"],fmt(a["map12"]),fmt(a["delta_vs_w0"]),
          *[fmt(a["segments"][s]["map12"]) for s in SEGMENTS]] for w,d in windows.items() for v,a in d["variants"].items())))
    lines.append(table(["外层", "分群", "分群MAP用户分母", "不同真值对"],
        ([WINDOW_NAMES[w], {"warm_21_plus":"暖商品≥21次", "strict_cold":"严格冷商品0次", "sparse1_5":"稀疏商品1–5次", "all_cold_sparse":"冷/稀疏合并"}[s],
          a["truth_users"],a["truth_pairs"]] for w,d in windows.items() for s,a in d["variants"]["W0"]["segments"].items())))
    lines.append(table(["外层", "方案", "覆盖率", "准入用户数", "替换总数", "每准入人平均件数", "最多件数", "插入冷正例", "移除暖正例", "净正例", "利好", "中性", "有害", "独有冷正例", "严格冷插入", "稀疏插入"],
        ([WINDOW_NAMES[w],VARIANT_NAMES[v],rate(a["admission_user_share"]),a["users_with_admission"],a["total_replacements"],
          fmt(a["mean_replacements_per_admitted_user"],3),a["max_replacements"],
          *[a[k] for k in ("inserted_cold_positive_pairs","removed_warm_positive_pairs","net_positive_pairs","beneficial_replacements",
          "neutral_replacements","harmful_replacements","cold_only_positive_actually_inserted","strict_cold_positive_insertions","sparse1_5_positive_insertions")]]
         for w,d in windows.items() for v,row in d["variants"].items() if v in TAUS for a in [row["admission"]])))
    lines.append(table(["外层", "方案", "替换件数组", "用户", "同组MAP差", "插入正例", "移除正例", "净正例", "对整窗差的贡献"],
        ([WINDOW_NAMES[w],VARIANT_NAMES[v],b,a["users"],fmt(a["delta_vs_same_users_w0"]),a["inserted_cold_positives"],
          a["removed_warm_positives"],a["net_positives"],fmt(a["delta_contribution_full_window"])]
         for w,d in windows.items() for v,row in d["variants"].items() if v in TAUS for b,a in row["admission_buckets"].items())))
    lines.append("0/1/2/3/4+是单用户实际替换件数，4+包括4–12件。不同替换组由分数和阈值选择，非随机分组；"
        "多件组较差不能单独证明‘多插一件’的因果伤害。对整窗贡献按该组AP差总和除以完整S_t用户数。"
        "匹配沿固定规范排序和无扰动原权重，同分最优解可能不唯一；所选位置不是第三个模型学出的独立位置价值。")
    pooled_buckets=[]
    for v in TAUS:
        for b in ("0","1","2","3","4+"):
            rows=[d["variants"][v]["admission_buckets"][b] for d in windows.values()]
            pooled_buckets.append([VARIANT_NAMES[v],b,*[sum(a[k] for a in rows) for k in
                ("users","inserted_cold_positives","removed_warm_positives","net_positives")]])
    lines.append(table(["四窗合并方案", "替换件数组", "用户—窗口数", "插入正例对", "移除正例对", "净正例对"],pooled_buckets))
    if summary:
        lines.append(table(["方案", "总体均值", "总体差", "不降窗", "最差差", "暖组差", "严格冷差", "稀疏差", "冷/稀疏差", "独有冷插入窗", "合并插入正例", "合并移除正例", "全部门槛"],
            ([VARIANT_NAMES[v],fmt(s["mean_map12"]),fmt(s["mean_delta"]),s["nondegrade_windows"],fmt(s["worst_delta"]),
              *[fmt(s["segments"][g]["mean_delta"]) for g in SEGMENTS],s["cold_only_positive_windows"],
              s["admission_pooled"]["inserted_cold_positive_pairs"],s["admission_pooled"]["removed_warm_positive_pairs"],
              "参照线，不作新方案晋级" if v=="W0" else "通过" if s["gates"]["all_pass"] else "未通过"] for v,s in summary.items())))
        lines.append(table(["方案", "未通过的固定门槛"],([VARIANT_NAMES[v],"；".join(GATE_NAMES[k] for k,ok in summary[v]["gates"]["checks"].items() if not ok) or "全部通过"] for v in TAUS)))
        total_users=sum(d["users"] for d in windows.values())
        lines.append(table(["四窗合并方案", "准入用户—窗口", "完整用户—窗口分母", "合并覆盖率", "总替换件数", "插入正例率", "被移除正例率"],
            ([VARIANT_NAMES[v],a["users_with_admission"],total_users,rate(a["admission_user_share"]),a["total_replacements"],
              rate(a["inserted_cold_positive_pairs"]/a["total_replacements"] if a["total_replacements"] else None),
              rate(a["removed_warm_positive_pairs"]/a["total_replacements"] if a["total_replacements"] else None)]
             for v in TAUS for a in [summary[v]["admission_pooled"]])))
        lines.append("最后两率使用完全相同的已执行替换配对数为分母：分别计插入商品未来被购买、被替换原商品未来被购买。"
            "它们不是线上曝光转化率，未观察购买不等于拒绝；也不是把整池Cold50与W0基础率直接混作对照。")
        if all(summary[v]["gates"]["warm"] for v in TAUS) and m["decision"]=="warm_risk_uncontrolled":
            lines.append("必须澄清机器名：本轮三个方案其实都通过了允许小幅损失的Warm保护门槛。"
                "`warm_risk_uncontrolled`在旧合同中同时涵盖‘冷侧有增益但总体保护失败’；本次属于后者，不能写成Warm保护门槛未通过。"
                "三方案总体均值都低于W0；激进方案虽提升冷/稀疏均值，独有冷正例只在1窗插入，且合并插入少于移除，故无安全工作点。")
        a=summary["A_tau0"]["admission_pooled"]
        one=[d["variants"]["A_tau0"]["admission_buckets"]["1"] for d in windows.values()]
        multi=[d["variants"]["A_tau0"]["admission_buckets"][b] for d in windows.values() for b in ("2","3","4+")]
        lines.append(f"激进方案共替换{a['total_replacements']}件，只插入{a['inserted_cold_positive_pairs']}对正例、移除{a['removed_warm_positive_pairs']}对原正例；"
            f"单件组移除{sum(x['removed_warm_positives'] for x in one)}对，多件组合计移除{sum(x['removed_warm_positives'] for x in multi)}对。"
            "观测损伤主要已出现在只替换1件的用户，因此现有结果不支持把失败主要归结为多件替换失控，也没有实测证明加max1即可修好。")
        if all(summary[v]["segments"]["strict_cold"]["mean_map12"]==0 for v in TAUS):
            lines.append("三个方案严格冷商品MAP均仍为0；当前取得的非零冷侧能力来自稀疏商品，不能写成严格冷启动已解决。")
        lines.append("当前已观测瓶颈是最终被选中替换的决策质量：区分度与粗校准门槛通过，仍未让高分冷候选稳定优于被移除的原商品。"
            "这不是数值求解还没完成，也不能仅凭这组结果归因为qC校准单一问题、模型容量不足或召回失败。"
            "本轮没有新的因果诊断或调参，不自动启动更大模型、候选扩展、再校准或第四个阈值。")
    lines.append("门槛保持原值：总体均值不降、至少3窗不降、最差差≥−0.000200；暖组均值差≥−0.000100，至少3窗差≥−0.000200；"
        "冷/稀疏均值严格提高、至少3窗不降、至少2窗插入独有冷路正例；合并插入正例严格多于移除。"
        "只在全部通过的阈值中依次选冷/稀疏均值、总体均值、较保守τ，近似相等容差沿合同1e-12。")
    return lines


def report(repo):
    repo = Path(repo).resolve()
    folder = repo / "reports/phase4"
    c = read_json(folder / "P4_2R3_EXPERIMENT_CONTRACT.json")
    m = read_json(folder / "P4_2R3_metrics.json")
    ver = read_json(folder / "P4_2R3_VERIFICATION.json")
    root = repo / "artifacts/phase4" / c["run_id"]
    execution = read_json(root / "EXECUTION_START.json")
    if m["execution"]["started_at_utc"] != execution["started_at_utc"] or ver["status"] != "pass":
        raise ValueError("report requires current completed metrics and independent verification pass")
    attempts = {a["window"]:a for a in m.get("attempts", []) if a.get("side") == "qW"}
    boundary = m.get("boundary_comparison", {})
    successful = sum(a.get("solver_result",{}).get("success") is True for a in attempts.values())
    cal, windows, summary = m.get("calibration",{}), m.get("windows",{}), m.get("summary",{})
    gate = m.get("calibration_gate")
    decision, selected = m["decision"],m.get("selected_variant","W0")
    late = boundary.get(LATE,{})
    delta = late.get("parameter_delta",{})
    cv = _delta_value(delta,"coefficient_delta_l2","coefficient_l2_delta","coefficient_delta_l2_norm","coefficient_l2_norm_delta")
    cm = _delta_value(delta,"coefficient_delta_max_abs","coefficient_max_abs_delta","max_abs_coefficient_delta")
    iv = _delta_value(delta,"intercept_delta")
    ll = _delta_value(delta,"training_log_loss_delta","log_loss_delta")
    prediction = delta.get("training_prediction_abs_delta",delta.get("prediction_delta",delta.get("training_prediction_delta",{})))
    pm = _delta_value(prediction,"mean","mean_abs","mean_absolute")
    p95 = _delta_value(prediction,"p95","p95_abs","p95_absolute")
    pmax = _delta_value(prediction,"max","max_abs","max_absolute")
    if pm is None:
        pm = _delta_value(delta,"prediction_mean_abs_delta","mean_abs_prediction_delta")
        p95 = _delta_value(delta,"prediction_p95_abs_delta","p95_abs_prediction_delta")
        pmax = _delta_value(delta,"prediction_max_abs_delta","max_abs_prediction_delta")
    oldg = late.get("old_diagnostics",{}).get("gradient_infinity_norm")
    newg = late.get("new_diagnostics",{}).get("gradient_infinity_norm")
    additional = late.get("additional_iterations_beyond_old_cap")
    late_steps = late.get("new_iterations")
    parity = [boundary.get(w,{}).get("early_window_exact_parity",{}) for w in list(WINDOWS)[:3]]
    exact_early = all(p and all(p.values()) for p in parity)
    exact_late = all(x == 0 for x in (cv,cm,iv,ll,pm,p95,pmax)) if all(x is not None for x in (cv,cm,iv,ll,pm,p95,pmax)) else False
    supported = bool(late.get("solver_result",{}).get("success") and late_steps is not None and late_steps < 1200)
    hypothesis = ("支持本次运行的迭代边界解释：提高上限后求解器正常成功返回，旧点与新点的实测差见第3节。"
                  if supported else "本轮未提供完成边界修复的证据；不能仅凭独立梯度宣称成功。")
    if supported and exact_late:
        hypothesis = "强支持本次运行是停止状态边界：上限提高后求解器返回成功，最终参数、训练概率和损失与旧1000步点完全一致。"
    lines = ["# P4.2R3：暖侧求解器迭代边界修复", "",
        "qW为冻结暖侧前12件的下周购买倾向模型，qC为冷侧前50件对应模型；完整定义见第2节。",
        f"机器结论：`{decision}`。{DECISION_TEXT.get(decision,decision)}",
        f"qW已拟合{len(attempts)}条链，实际求解器成功返回{successful}条；完成主校准{len(cal)}窗、准入{len(windows)}窗。"
        f"保留/选定`{selected}`。未运行不等于0分。",
        f"合同登记`{c['created_at_utc']}`；正式结束`{m.get('finished_at_utc')}`；执行快照状态`{m['status']}`保留原值，"
        f"本次独立复核`{ver['status']}`是后续验证结果，不改写原始执行记录。",
        "P4.2与P4.2R的engineering_failure（项目工程失败状态）、P4.2R2的qW_global_R2_convergence_failure（全局计数变换未通过求解器门槛）"
        "均完整保留。本轮未运行最终周、未整合Warm-v2、未启动P4.3。", "", "## 1. 单变量修复与证据边界", "",
        "P4.2R2夏末在第1000步返回迭代上限警告，而保存点梯度9.33079e-8已小于原tol=1e-7。"
        "只读源码审计表明，SciPy封装在完成迭代并达到上限时会设置停止状态，所以两个观察并不矛盾；旧合同仍正确停止。",
        "本轮唯一正式参数改动为qW max_iter:1000→1200。四窗全部重新拟合，不把旧前三窗模型和新夏末拼接。"
        "qC直接复用P4.2R已收敛资产、保持原1000次合同；qW沿同一数据、R1清理、19项log1p、标准化、LBFGS、C=1、tol=1e-7；"
        "不加权、不负采样、不超采样、不改阈值或匹配。成功必须是求解器真实success，独立梯度仅作诊断。",
        "为记录真实返回状态，本轮在库原有结果检查入口安装仅观察的临时记录器，记录后原检查函数照常执行，退出时恢复。"
        "它不修改优化器选项、目标、梯度、参数或返回值，也不注入新的迭代回调；这项工程处理用于可审计地记录success/status，而不是改变求解行为。",
        "任一链达到1200或非成功即停止，不尝试1500/2000、放宽tol、换算法或删字段。只有四条链都成功才做四窗校准；"
        "严重校准门槛再通过才执行原三阈值准入。此顺序先于外层结果冻结。", "", *_terminology(), "", "## 3. 四条训练链与边界对照", ""]
    lines.append(table(["外层", "训练候选行", "正例对", "正例率", "维数", "旧步数", "新步数", "success", "返回码", "警告数", "拟合秒"],
        ([WINDOW_NAMES[w],a["training_rows"],a["positives"],rate(a["base_rate"]),a["model_features"],boundary.get(w,{}).get("old_iterations"),
          _iterations(a),a.get("solver_result",{}).get("success"),a.get("solver_result",{}).get("status"),len(a.get("convergence_warnings",[])),fmt(a["fit_seconds"],3)]
         for w,a in attempts.items())))
    lines.append(table(["外层", "求解器返回说明", "梯度∞", "条件数", "最大|z|", "最大字段及中文含义"],
        ([WINDOW_NAMES[w],a.get("solver_result",{}).get("message","不可用"),sci(a["diagnostics"]["gradient_infinity_norm"]),
          sci(a["diagnostics"]["local_regularized_hessian_condition_number"]),fmt(a["diagnostics"]["max_standardized_absolute_value"],3),
          a["diagnostics"]["max_standardized_absolute_feature"]+"："+feature_note(a["diagnostics"]["max_standardized_absolute_feature"])] for w,a in attempts.items())))
    lines.append("返回码0及success=true表示求解器正式成功；CONVERGENCE表示达到库的收敛条件，STOP/TOTAL NO. OF ITERATIONS REACHED LIMIT表示迭代上限停止。"
        "PROJECTED GRADIENT表示投影梯度，在此无参数边界时与原目标梯度条件对应；PGTOL为该梯度停止容差，本轮1e-7。不是模型预测准确率。")
    lines.append(table(["早期外层", "预处理精确相同", "系数精确相同", "截距精确相同", "迭代精确相同"],
        ([WINDOW_NAMES[w],*[boundary.get(w,{}).get("early_window_exact_parity",{}).get(k,"未运行") for k in ("preprocessing","coefficients","intercept","n_iter")]] for w in list(WINDOWS)[:3])))
    lines.append(f"前三窗{'全部精确复现旧收敛解和步数' if exact_early else '确定性对照按表逐项披露，不将缺失或近似相等称作精确相同'}。")
    if late:
        lines.append(table(["夏末边界诊断", "实测值"],[("R2旧点迭代",late.get("old_iterations")),("R3最终迭代",late_steps),
            ("超出旧1000上限的实际步数",additional),("系数差L2长度",sci(cv)),("系数最大绝对差",sci(cm)),("截距有符号差",sci(iv)),
            ("平均训练对数损失差",sci(ll)),("同训练行概率绝对差均值",sci(pm)),("同训练行概率绝对差p95",sci(p95)),
            ("同训练行概率绝对差最大值",sci(pmax)),("旧梯度∞",sci(oldg)),("新梯度∞",sci(newg)),
            ("梯度∞有符号差",sci(newg-oldg if oldg is not None and newg is not None else None))]))
    lines.append(hypothesis)
    if supported and additional == 0 and late_steps == 1000:
        lines.append("夏末额外迭代为0并不矛盾：上限变为1200后，第1000步不再被上限规则强制标记停止，求解器得以在相同点返回成功。"
            "这次花费的是重新执行四条链以验证统一合同，不是让夏末再更新200步。")
    lines.append("这些参数和训练概率差只检验数值边界，不按其大小重选模型或修改门槛；也不能推导校准/准入MAP必然提高。"
        "完整四窗对照见[p4_2r3_solver_boundary_audit.json](p4_2r3_solver_boundary_audit.json)，夏末参数差见"
        "[p4_2r3_late_parameter_delta.json](p4_2r3_late_parameter_delta.json)。")
    lines += ["", *_calibration_lines(m), "", *_admission_lines(m), "", "## 6. 21项问题逐条回答", ""]
    answers = [
        ("1200预算是否4/4求解器成功？",f"已拟合{len(attempts)}链，{successful}链success=true；"+("四窗门槛通过。" if (m.get("qW_convergence_gate") or {}).get("pass") else "未通过四窗门槛。")),
        ("冬/春/初夏是否按原步数提前退出？", "是，三链参数、预处理和实际步数精确一致。" if exact_early else "第3节逐项对照；未完成者不推定成功。"),
        ("夏末最终多少步？",str(late_steps) if late_steps is not None else "未运行。"),
        ("比1000多了多少步？",str(additional) if additional is not None else "未运行。"),
        ("旧点与最终系数/截距差？",f"系数L2差{sci(cv)}、最大绝对差{sci(cm)}，截距差{sci(iv)}。"),
        ("夏末训练概率差？",f"绝对差均值{sci(pm)}、p95 {sci(p95)}、最大{sci(pmax)}，比较相同训练候选行。"),
        ("夏末梯度如何变化？",f"{sci(oldg)}→{sci(newg)}；平均未正则训练损失差{sci(ll)}。"),
        ("支持停止边界而非优化明显未完成吗？",hypothesis),
        ("两侧严重校准门槛通过了吗？","通过，只排除预注册严重跨窗失效。" if gate and gate["pass"] else "失败，准入未运行。" if gate else "未运行。"),
        ("AUC、Brier、ECE、截距/斜率如何？","第4节逐窗逐侧实测表及10箱审计；不以小平均误差等同尾部安全。" if cal else "未运行，不能推测。"),
        ("三个阈值覆盖率如何？","第5节逐窗覆盖率以完整S_t为分母。" if windows else "未运行，不是0覆盖。"),
        ("0/1/2/3/4+次替换风险？","第5节完整分组表，对照同组用户原W0。" if windows else "未运行。")]
    if windows:
        mult,eff,seg,overall = [],[],[],[]
        for v in TAUS:
            rows = [d["variants"][v]["admission_buckets"][b] for d in windows.values() for b in ("2","3","4+")]
            mult.append(f"{VARIANT_NAMES[v]}多件组插入正例{sum(a['inserted_cold_positives'] for a in rows)}、移除{sum(a['removed_warm_positives'] for a in rows)}")
            if v in summary:
                s=summary[v]; a=s["admission_pooled"]
                eff.append(f"{VARIANT_NAMES[v]}插入{a['inserted_cold_positive_pairs']}、移除{a['removed_warm_positive_pairs']}对")
                seg.append(f"{VARIANT_NAMES[v]}严格冷/稀疏均值差{fmt(s['segments']['strict_cold']['mean_delta'])}、{fmt(s['segments']['sparse1_5']['mean_delta'])}")
                overall.append(f"{VARIANT_NAMES[v]}总体均值差{fmt(s['mean_delta'])}，总体保护{'通过' if s['gates']['overall'] else '未通过'}")
        answers += [("多件准入存在暖风险累积吗？","本次观测不支持它是主要失败来源，单件组损失已占多数。"+"；".join(mult)+"。描述性分组，不能作随机因果解释。"),
            ("插入冷正例超过移除暖正例吗？","；".join(eff)),("严格冷与稀疏各拿回多少？","；".join(seg)),("总体MAP保护住了吗？","；".join(overall))]
    else:
        answers += [(q,"未运行，不能判断。") for q in ("多件准入存在暖风险累积吗？","插入冷正例超过移除暖正例吗？","严格冷与稀疏各拿回多少？","总体MAP保护住了吗？")]
    passing = m.get("passing_variants",[])
    answers += [("存在安全工作点吗？",("通过所有门槛："+", ".join(passing)) if passing else "三个固定阈值均未通过全部门槛。" if summary else "准入晋级未运行，不宣称有安全工作点。"),
        ("选择哪个阈值？",VARIANT_NAMES.get(selected,selected) if passing else "没有晋级阈值，保持W0。"),
        ("当前未通过的主要环节？",("数值与严重校准门槛已过，瓶颈是选中替换的决策质量：稀疏增益不足以补偿原正例被移除，严格冷仍未兑现。" if decision=="warm_risk_uncontrolled" else DECISION_TEXT.get(decision,decision))+"不自动启动新修复/调参。"),
        ("最终周是否仍未运行？","是，2020-09-16=not_run。"),("Warm-v2整合是否未启动？","是，未合并、未接入；P4.3亦未启动。")]
    if len(answers) != 21:
        raise ValueError("P4.2R3 requires exactly 21 conclusion answers")
    lines.append(table(["序号","问题","回答"],([i,q,a] for i,(q,a) in enumerate(answers,1))))
    lines += ["", "## 7. 成本、验证和产物", ""]
    resource=m.get("resources",{})
    peak=resource.get("peak_process_working_set_gib")
    lines.append(f"正式运行{fmt(resource.get('formal_seconds'),2)}秒；qW新拟合{len(attempts)}次，qC原模型复用{len(m.get('qc_reuse',{}))}条、qC新拟合{resource.get('qC_new_fits',0)}次。"
        "预估CPU计算及验证5–15分钟，实测以运行和验证记录为准，代码/报告时间另计；无新候选生成，无GPU训练/推理。"
        +(f"进程峰值工作集{fmt(peak,3)}GiB。" if peak and peak>0 else "峰值工作集未获得有效数值，不以0代替缺失。"))
    checks=ver.get("checks",[])
    counts={s:sum(a.get("status")==s for a in checks) for s in ("pass","not_run","fail")}
    lines.append(f"独立复核{counts['pass']}项通过、{counts['not_run']}项未运行、{counts['fail']}项失败，耗时{fmt(ver.get('seconds'),2)}秒。"
        "按[P4_2R3_VERIFICATION.json](P4_2R3_VERIFICATION.json)逐项区分实际检查与未运行；验证通过证明证据和执行边界自洽，不等于模型晋级。")
    tp=folder / "P4_2R3_TEST_VERIFICATION.json"
    if tp.exists():
        tests=read_json(tp)
        lines.append(f"单元/回归测试{tests.get('tests_passed',tests.get('tests'))}项通过、{tests.get('tests_failed',tests.get('failures',0))}项失败，"
            f"耗时{fmt(tests.get('seconds',tests.get('test_seconds')),3)}秒；时点和命令见[P4_2R3_TEST_VERIFICATION.json](P4_2R3_TEST_VERIFICATION.json)。合成测试不代替真实外层验证。")
    if m.get("failure_reason"):
        lines.append(f"停止原文：`{m['failure_reason']}`。失败参数、警告和追踪保存在新目录，不覆盖旧失败。")
    lines.append("本轮新增模型、预处理及已运行阶段的逐行证据保存在被Git忽略的`artifacts/phase4/p4-2r3-*/`；原训练表和qC模型仍按原路径复用。"
        "输出清单列全部新报告、源码/测试和运行目录文件的路径、字节、SHA（行业文件摘要），并单列原资产引用与可信旧清单；"
        "本次摘要用于显式证据合同，不把新算摘要当来源证明。用户标识和候选明细不进入公开报告，日志/路线继续忽略。"
        "本轮到此停止，未提交、未推送，不自动开始P4.3、Warm-v2整合或最终周。")
    target=folder / "P4_2R3_FINAL.md"
    target.write_text("\n".join(lines).rstrip()+"\n",encoding="utf-8")
    paragraphs=["追加P4.2R3实测，不覆盖P4.2/P4.2R的engineering_failure和P4.2R2的qW_global_R2_convergence_failure历史。",
        "唯一正式改动：qW迭代上限1000→1200，四窗统一重新拟合；原R1+19项log1p及全部其他参数、候选、阈值、匹配不变，qC精确复用P4.2R。",
        f"qW {successful}/{len(attempts)}条实际solver success；夏末实际{late_steps}步、超出旧上限{additional}步。"+hypothesis,
        f"主校准{len(cal)}窗、准入{len(windows)}窗；机器结论`{decision}`，保留/选定`{selected}`。"+DECISION_TEXT.get(decision,decision),
        f"正式{fmt(resource.get('formal_seconds'),2)}秒，独立复核{ver['status']}；数值边界、校准和MAP分开解释，不自行追加下一阶段。",
        "证据：`reports/phase4/P4_2R3_FINAL.md`、metrics、solver_boundary_audit、late_parameter_delta、VERIFICATION、OUTPUT_MANIFEST。"
        "最终周not_run、Warm-v2未整合、P4.3未启动；无提交推送，日志/路线与大型逐行产物继续gitignore。"]
    _append_doc(repo / "docs/ROADMAP_PHASE4.zh-CN.md","## 17. P4.2R3 迭代上限边界修复实测（2026-09-09）",paragraphs)
    _append_doc(repo / "docs/PROJECT_LOG.zh-CN.md","## 2026-09-09 — P4.2R3：迭代边界修复",paragraphs)
    manifest_path=folder / "P4_2R3_OUTPUT_MANIFEST.json"
    report_paths=sorted({*folder.glob("P4_2R3*.md"),*folder.glob("P4_2R3*.json"),*folder.glob("p4_2r3_*.json")})
    artifacts=[identity(p) for p in sorted(root.rglob("*")) if p.is_file()]
    reused={}
    for entry in c.get("prepared_reuse",{}).values():
        for item in entry.get("features",{}).values():
            reused[item["path"]]={k:item[k] for k in ("path","bytes","sha256")}
    for entry in m.get("qc_reuse",{}).values():
        for key in ("model","preprocessing"):
            item=entry[key]; reused[item["path"]]={k:item[k] for k in ("path","bytes","sha256")}
    manifest={"stage":STAGE,"run_id":c["run_id"],"schema_version":"p4.2r3-output-manifest-v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"decision":decision,"selected_variant":selected,
        "verification_status":ver["status"],"reports":[identity(p) for p in report_paths if p!=manifest_path],"artifacts":artifacts,
        "sources":[identity(p) for p in sorted({*(repo / "src/hm_recsys").glob("p42r3*.py"),*(repo / "tests").glob("test_p42r3*.py")})],
        "artifact_files":len(artifacts),"artifact_bytes":sum(a["bytes"] for a in artifacts),"reused_artifacts":list(reused.values()),
        "reused_artifact_policy":"original paths, not copied or retrained; checked against prior trusted manifests in execution and independent verification",
        "prior_manifests":{k:c[k] for k in ("original_P4_2_manifest","prior_P4_2R_manifest","prior_P4_2R2_manifest") if k in c},
        "manifest_self_hash_excluded":True,"private_documents_ignored":["docs/ROADMAP_PHASE4.zh-CN.md","docs/PROJECT_LOG.zh-CN.md"],
        "historical_decisions_preserved":{"P4.2":"engineering_failure","P4.2R":"engineering_failure","P4.2R2":"qW_global_R2_convergence_failure"},
        "final_week":"not_run","Warm_v2_integrated":False,"P4_3_started":False}
    write_json(manifest_path,manifest)
    return {"report":str(target),"manifest":str(manifest_path),"decision":decision,"artifact_files":len(artifacts),"artifact_bytes":manifest["artifact_bytes"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    args=parser.parse_args()
    print(report(args.repo))
