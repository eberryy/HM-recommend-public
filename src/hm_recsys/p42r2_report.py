"""P4.2R2 Chinese report renderer: measured evidence only, no fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .p41a_contract import identity, read_json, write_json
from .p42_contract import TAUS, WINDOWS
from .p42_report import feature_note
from .p42r_report import DECISIONS, GATE_NAMES, VARIANT_NAMES, WINDOW_NAMES, fmt, rate, sci, table


STAGE = "P4.2R2"
DECISION_TEXT = {**DECISIONS,
    "qW_global_R2_convergence_failure": "至少一条qW训练链在统一R1+log1p与冻结求解预算下未收敛，立即停止，未完成四窗修复总门槛。"}


def _iterations(a):
    if not a:
        return "未运行"
    model = a.get("model", {})
    values = read_json(Path(model["path"])).get("n_iter", []) if model else a.get("n_iter", [])
    return ",".join(map(str, values))


def _rel(new, old):
    return "不可用" if new is None or old is None or old == 0 else f"{new / old - 1:+.2%}"


def _geometry(a, field):
    return (a.get("diagnostics") or {}).get(field) if a else None


def _patch_doc(path, title, paragraphs):
    begin, end = "<!-- P4.2R2 measured closure begin -->", "<!-- P4.2R2 measured closure end -->"
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    block = begin + "\n\n" + title + "\n\n" + "\n\n".join(paragraphs) + "\n\n" + end
    if begin in content:
        if end not in content:
            raise ValueError("incomplete generated P4.2R2 documentation block")
        left, tail = content.split(begin, 1)
        _, right = tail.split(end, 1)
        result = left + block + right
    else:
        result = content.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result, encoding="utf-8")


def report(repo):
    repo = Path(repo).resolve()
    folder = repo / "reports/phase4"
    c = read_json(folder / "P4_2R2_EXPERIMENT_CONTRACT.json")
    m = read_json(folder / "P4_2R2_metrics.json")
    root = repo / "artifacts/phase4" / c["run_id"]
    execution = read_json(root / "EXECUTION_START.json")
    if m["execution"]["started_at_utc"] != execution["started_at_utc"]:
        raise ValueError("refuse stale metrics while current formal run is incomplete")
    vp = folder / "P4_2R2_VERIFICATION.json"
    ver = read_json(vp) if vp.exists() else {"status": "pending", "checks": []}
    old = read_json(folder / "P4_2R_metrics.json")
    old_qw = {a["window"]: a for a in old["attempts"] if a["side"] == "qW" and a["repair"] == "R1"}
    attempts = m.get("attempts", [])
    qw = {a["window"]: a for a in attempts if a.get("side", "qW") == "qW"}
    successful = sum(a["status"] == "converged" for a in qw.values())
    cal = m.get("calibration", {})
    full_cal = m.get("full_calibration", {})
    windows = m.get("windows", {})
    summary = m.get("summary", {})
    gate = m.get("calibration_gate")
    selected, decision = m.get("selected_variant", "W0"), m["decision"]
    numerics = m.get("numerical_comparison", {})
    lines = ["# P4.2R2：四窗统一计数长尾数值修复", "",
        f"机器结论：`{decision}`。{DECISION_TEXT.get(decision, decision)}",
        f"保留/选定方案：`{selected}`；qW已尝试{len(qw)}条链，其中{successful}条收敛；"
        f"主校准完成{len(cal)}窗，准入完成{len(windows)}窗。未运行不等于0分。",
        f"合同登记：`{c['created_at_utc']}`；正式结束：`{m.get('finished_at_utc',m.get('created_at_utc'))}`；"
        f"原执行快照状态：`{m['status']}`，保留计算结束时原值；本次收尾独立复核：`{ver['status']}`。",
        "原P4.2、P4.2R均保留engineering_failure（本项目工程失败状态）的历史结果。"
        "这次授权另立全局变换合同，不修改旧报告、旧合同、失败参数或旧R2未运行事实。",
        "", "## 1. 本轮只改变什么，为什么", "",
        "P4.2R已经恢复8个历史日期的共同训练用户，qC四条链均收敛；qW清理常数和完全重复列后前三链收敛、"
        "夏末链仍在1000次达到迭代上限。旧合同只有冬季清理失败才能解锁log1p，故无法在晚窗口单独补救。",
        "P4.2R2在看本轮外层结果之前，统一规定所有qW训练链使用相同的‘R1清理+固定19项计数log1p’。"
        "它不是只对夏末挑一种更强处理，也不是把旧合同事后改成成功；qC直接复用已收敛模型与原预处理器。",
        "本次R1对照与统一log1p使用相同候选行、用户、标签和常数/重复列规则，因此更直接对比计数变换的作用；"
        "不同于最初P4.2→P4.2R还混有训练时间池与用户总体变化。仍同时变换19项计数，不能证明某一字段是唯一根因。",
        "", "## 2. 术语与统计口径", "",
        "以下项目名均沿用历史定义。W0是冻结Warm-v1每用户12件及原顺序；M4是已训练好的纯内容商品表示；"
        "B0在M4召回前200件中重排取50件，称Cold50；Warm150是暖路较大候选集合，不等于‘全是暖商品’。本轮不重训这些上游。",
        "S_t是截止日t完整W0离线评测用户；其固定规则是下一周有任意购买且符合用户哈希10%样本，不专门要求未来冷商品购买。"
        "H_t是S_t中截止日前最近20件不同购买商品至少一件可映射到M4目录的用户。两侧训练的实际候选行用户集合都等于H_t；"
        "外层仍使用完整S_t，无Cold候选者原样保留W0并进入评测分母。",
        "qC和qW分别表示Cold50和W0前12候选在上述共同离线人群条件下、下一周被购买的倾向。"
        "propensity（行业统计术语，倾向）不是线上无条件购买概率，也不是曝光后点击/转化率。未观察购买不是用户明确拒绝。",
        "cutoff（行业时间截点）为t；特征和购买计数严格来自t前，真值来自[t,t+7天)。训练标签结束严格早于外层截止日。"
        "正例按不同的截止日—用户—商品对计数；历史交易事件计数保留重复行。跨窗口累计人数是用户—窗口观察数，不是去重自然人数。",
        "R1（项目数值清理名）：训练列中位数填补后删除常数列和逐行完全相同的重复列；优先保留非_available列，"
        "再按字段名字典序选保留列，最终沿原字段顺序输出。_available是对应数值是否有限可用的0/1标记。"
        "high-correlation（行业高相关）本身不触发删列，也不用标签、系数、AUC或MAP决定删列。",
        "Global R2（项目本轮全局变换名）：所有链都对R1后幸存的预注册计数做log1p，即自然对数log(1+x)，"
        "压缩heavy-tail（行业长尾：极少数特别大的计数）尺度。先检查所有名单内有限观测值非负；发现负值即停止，不能裁0或取绝对值。",
        "固定处理顺序为：原数值→仅训练行中位数填补→R1常数/重复清理→名单内幸存计数log1p→"
        "用变换后的训练行拟合StandardScaler（行业均值/标准差标准化器）→用冻结预处理转换训练/外层。布尔/可用标记不标准化、不取对数。",
        "L2 Logistic Regression（行业正则逻辑回归）仍为LBFGS（拟牛顿数值求解器）、C=1、tol=1e-7、max_iter=1000；"
        "不调参数、不加权、不负采样、不复制正例。qW原逻辑输入117维，qC原输入30维；四窗规则相同不意味着训练数据导出的幸存维数必须相同。",
        "converged/non_converged是求解器已收敛/未收敛状态，不是MAP提升/退化。迭代次数是一次拟合中的优化步骤数，不是超参数搜索次数。",
        "condition number（行业条件数）是训练结束参数处含正则Hessian（局部曲率矩阵）最大/最小特征值比；"
        "gradient∞（行业梯度无穷范数）是目标各参数梯度绝对值最大值；max|z|（项目数值摘要）是训练数值列标准化后最大绝对值。"
        "它们描述数值几何，不直接衡量推荐效果。最大的标准化字段只能称剩余极值，不自动等于新瓶颈。",
        "MAP@12（行业排序指标）对单用户各命中位置累计精度求和，除以min(用户不同真值数,12)，再对完整S_t平均；四窗均值对窗口等权。"
        "严格冷、稀疏1–5、warm_21_plus指商品截止日前全站事件数为0、1至5、至少21；all_cold_sparse合并前两组。"
        "分群MAP的用户分母为完整S_t中有该分群真值者，原推荐位置不压缩。",
        "ROC-AUC（行业排序区分指标）比较正例是否高于未观察购买行；PR-AUC在本项目用非插值平均精度；"
        "Brier为候选行概率平方误差均值，ECE为10个按预测稳定排序等人数分箱的概率率差绝对值、按行数加权。"
        "校准截距/斜率是对真值与logit(q)关系的诊断，理想约0/1；仅诊断，不反向改变正式预测。",
        "U=logit(qC)−logit(qW)；logit（行业概率变换）是赔率q/(1−q)的自然对数，先将概率裁到[1e-6,1−1e-6]。"
        "阈值τ固定为0、ln2、ln4；只有U>τ的冷商品—暖位置配对可用。每人Cold50去除已在W0前12的重合商品后与12个位置配对，最多600条边。",
        "最大权二分匹配（行业图优化）最大化合计U−τ，每冷商品和暖位置各至多使用一次；允许0至12次替换，不设最多1件。"
        "未匹配位置保持原样，没有合格边则整个W0不动。cold-only（项目独有冷路候选）指Cold50中不属于Warm150，"
        "不是只要不在W0前12就算独有。",
        "原字段具体中文定义见[P4_2_FINAL.md](P4_2_FINAL.md)字段附录；本报告新增的全局变换名单逐项说明如下。",
    ]
    requested = c.get("qW_repair", {}).get("R2", {}).get("log1p_columns")
    if requested is None:
        requested = c.get("global_R2", {}).get("log1p_columns", [])
    if not requested:
        from .p42r_contract import LOG1P_COUNTS
        requested = LOG1P_COUNTS
    lines.append(table(["预注册计数字段", "含义/统计单位"], ([name, feature_note(name)] for name in requested)))
    lines.append("rank、rank_pct、score、cosine、share、ratio、trend、days_since、zscore、percentile分别为名次、名次比例、分数、"
        "余弦相似、份额、比例、趋势、距今天数、标准分和百分位；这些字段及布尔/可用性标记均不取log1p。"
        "名单内若被R1删掉，只记录absent_after_R1（R1后不存在故未变换），不添加替代字段。")
    lines += ["", "## 3. qW四链统一拟合与R1对照", ""]
    lines.append(table(["外层", "训练候选行", "正例对", "正例率", "原维数", "常数删除", "重复删除", "R1后维数", "申请log1p项", "实际变换项", "迭代", "拟合秒", "状态"],
        ([WINDOW_NAMES[w], a["training_rows"], a["positives"], rate(a["base_rate"]), a["cleanup"]["dims_before"],
          a["cleanup"]["constant_dropped_count"], a["cleanup"]["exact_duplicate_dropped_count"], a["cleanup"]["dims_after"],
          len(requested), len([name for name in requested if name in a["feature_order"]]), _iterations(a), fmt(a["fit_seconds"],3), a["status"]]
         for w, a in qw.items())))
    compare_rows = []
    for w in WINDOWS:
        prev, curr = old_qw.get(w), qw.get(w)
        pcond, cond = _geometry(prev, "local_regularized_hessian_condition_number"), _geometry(curr, "local_regularized_hessian_condition_number")
        pmax, maximum = _geometry(prev, "max_standardized_absolute_value"), _geometry(curr, "max_standardized_absolute_value")
        compare_rows.append([WINDOW_NAMES[w], _iterations(prev), _iterations(curr), sci(pcond), sci(cond), _rel(cond,pcond),
                             fmt(pmax,3), fmt(maximum,3), _rel(maximum,pmax),
                             sci(_geometry(prev,"gradient_infinity_norm")), sci(_geometry(curr,"gradient_infinity_norm"))])
    lines.append(table(["外层", "R1迭代", "统一R2迭代", "R1条件数", "统一R2条件数", "条件数相对变化", "R1最大|z|", "统一R2最大|z|", "极值相对变化", "R1梯度∞", "统一R2梯度∞"], compare_rows))
    lines.append("相对变化=(统一R2数值/R1数值)−1；负数表示下降。R1对照是上轮已保存的相同训练链，不重跑旧方案或挑选最好的一次。"
        "当前任一链不收敛就立即停止，后续未运行链不能填0或称作失败；只有4/4全部收敛才允许校准。")
    lines.append(table(["外层", "统一R2最大|z|字段", "具体含义", "最大|z|"],
        ([WINDOW_NAMES[w], a["diagnostics"]["max_standardized_absolute_feature"],
          feature_note(a["diagnostics"]["max_standardized_absolute_feature"]), fmt(a["diagnostics"]["max_standardized_absolute_value"],3)] for w,a in qw.items())))
    failed = [a for a in qw.values() if a["status"] != "converged"]
    if failed:
        a = failed[-1]
        gradient = a["diagnostics"]["gradient_infinity_norm"]
        tolerance = a["diagnostics"]["frozen_gtol"]
        lines.append(f"实际停止链为{WINDOW_NAMES[a['window']]}：求解器在{_iterations(a)}次返回未收敛警告，"
            f"本轮仍按警告/返回状态门槛停止；保存参数处独立重算的梯度∞为{sci(gradient)}，"
            f"{'已经小于' if gradient < tolerance else '仍大于或等于'}tol={sci(tolerance)}。"
            "求解器终止状态与保存点梯度诊断是不同证据；特别是达到迭代上限时，不得把后者自动当成前者已成功。"
            "本轮不覆盖警告、不追加迭代、不事后改用另一种成功判据，也不能写成‘残余梯度仍必然超过阈值’。")
        solver_audit = folder / "p4_2r2_solver_stop_audit.json"
        if solver_audit.exists():
            lines.append("只读检查本机SciPy 1.17.1的L-BFGS-B封装进一步解释了这一边界：每完成新迭代先增加计数，"
                "达到max_iter就设置迭代上限停止状态；这个状态随后返回警告，而成功收敛状态才返回成功。"
                "所以第1000步保存点梯度已低于阈值与返回上限警告并不矛盾。详见"
                "[求解器停止机制审计](p4_2r2_solver_stop_audit.json)。本轮没有新增优化器调用，"
                "也不能断言第1001步一定成功。当前未通过的是预注册求解器返回状态门槛，不是证明log1p无效。")
        lines.append(table(["外层", "收敛警告条数", "首条求解器警告摘要"],
            ([WINDOW_NAMES[w],len(a.get("convergence_warnings",[])),
              a["convergence_warnings"][0].split("\n\n")[0] if a.get("convergence_warnings") else "无收敛警告"] for w,a in qw.items())))
    lines.append("`user_item_events_28d`是用户在截止前28天购买候选商品的事件数，保留原重复事件；不是商品的全站销量。"
        "其R1夏末标准化极值约93.165，是本轮预注册关注字段。该字段的逐窗对照见机器数值审计；若不在新top15极值列表中，"
        "只能说明它不是最大项之一，不能把未列出当成0。")
    # A separate named-feature receipt is required even if it leaves the top15.
    focus_rows = []
    for w, entry in numerics.items():
        if w not in WINDOWS or not isinstance(entry, dict):
            continue
        focus = entry.get("user_item_events_28d", {})
        if focus:
            prev = focus.get("R1", {})
            curr = focus.get("R2", {})
            pv, cv = prev.get("max_standardized_absolute_value"), curr.get("max_standardized_absolute_value")
            focus_rows.append([WINDOW_NAMES[w], sci(pv), sci(cv), _rel(cv,pv),
                str(curr.get("raw_finite_max", "不可用")), "已log1p" if curr.get("log1p_applied") else curr.get("status", "不可用")])
    if focus_rows:
        lines.append(table(["用户—商品28天计数字段", "R1最大|z|", "统一R2最大|z|", "相对变化", "训练原计数最大值", "变换状态"], focus_rows))
    if len(qw) == 4:
        late = qw["late_summer_20200819"]["diagnostics"]
        lines.append("本轮支持‘计数长尾被压缩’，但不支持‘四窗统一收敛已经解决’。"
            "冬季、春季的迭代反而增加，初夏减少，夏末仍碰到上限；条件数改善也没有转化为所有链都获得成功返回状态。"
            f"四窗最大标准化字段均为`item_days_since_last_sale`（商品最近销售距截止日天数），夏末仍有{fmt(late['max_standardized_absolute_value'],3)}的极值。"
            "这个天数字段不是本轮新增，且按合同禁止log1p；不能因其现在排第一就认定为唯一新根因或擅自扩大变换名单。")
    lines += ["", "## 4. qC复用、时间安全与校准", "",
        "qC继续使用P4.2R修复后的共同人群候选、模型和预处理；本轮不更改qC字段、用户总体或训练参数，也不再生成一套候选。"
        "模型复用须与上轮输出清单中的可信文件记录核对；SHA为文件摘要，服务于跨次运行冻结资产完整性，不把当前文件自算摘要当来源证明。",
        f"本轮实际复用qC模型{len(m.get('qc_reuse',{}))}条，qC新拟合0次；具体源模型、预处理与训练日期记录在本轮收敛审计的qC_reuse（qC原资产复用记录）中。",
        "主校准同在H_t候选条件下：qC完整Cold50，qW同用户原12件；补充全S_t的qW结果仅作训练支持外人群诊断。"
        "这不改变最终MAP的完整S_t分母。",
        "严重校准门槛沿旧合同：同一侧至少3窗预测均值/实测率不在[0.1,10]，或至少3窗ROC-AUC≤0.5，或至少3窗预测恒定即停止。"
        "通过只表示未触发严重跨窗失效，不能推导概率完美、尾部风险解决或线上真实购买率。",
    ]
    if cal:
        lines.append(table(["外层", "侧", "候选行", "正例对", "训练正例率", "实测率", "平均预测", "预测/实测", "ROC-AUC", "PR-AUC", "Brier", "ECE", "截距", "斜率"],
            ([WINDOW_NAMES[w], side, a["rows"], a["positives"], rate(a.get("training_base_rate")), rate(a["observed_positive_rate"]),
              rate(a["mean_predicted_probability"]), fmt(a["predicted_to_observed_rate_ratio"],3), fmt(a["roc_auc"],5),
              fmt(a["pr_auc"],6), fmt(a["brier_score"]), fmt(a["ece"]), fmt(a["calibration_intercept"],4), fmt(a["calibration_slope"],4)]
             for w, rows in cal.items() for side,a in rows.items())))
        lines.append("完整10箱可靠性、可用性与警告见`p4_2r2_propensity_calibration.json`。分箱仅按预测及原行序决定，"
            "截距/斜率审计不回流正式效用；在极稀疏标签下，小Brier/ECE仍可能掩盖少量高分候选的错误。")
    else:
        lines.append("校准未运行；对应JSON为not_run（未运行）。没有AUC、Brier、ECE或十箱观测，不凭训练收敛宣称泛化已通过。")
    if full_cal:
        lines.append(table(["qW全S补充诊断", "候选行", "实测率", "平均预测", "ROC-AUC", "Brier", "ECE"],
            ([WINDOW_NAMES[w],a["rows"],rate(a["observed_positive_rate"]),rate(a["mean_predicted_probability"]),
              fmt(a["roc_auc"],5),fmt(a["brier_score"]),fmt(a["ece"])] for w,a in full_cal.items())))
    lines.append(f"四窗主校准严重门槛：{'通过' if gate and gate['pass'] else '失败' if gate else '未运行'}。")
    lines += ["", "## 5. 原三阈值准入与风险", ""]
    if windows:
        lines.append(table(["外层", "方案", "完整S用户", "总体MAP", "总体差", "暖≥21组MAP", "严格冷MAP", "稀疏1–5MAP", "冷/稀疏合并MAP"],
            ([WINDOW_NAMES[w],VARIANT_NAMES[v],d["users"],fmt(a["map12"]),fmt(a["delta_vs_w0"]),
              *[fmt(a["segments"][s]["map12"]) for s in ("warm_21_plus","strict_cold","sparse1_5","all_cold_sparse")]]
             for w,d in windows.items() for v,a in d["variants"].items())))
        lines.append(table(["外层", "方案", "准入用户比例", "准入用户数", "替换总数", "平均每准入人替换", "最多替换", "插入冷正例", "移除暖正例", "净正例", "利好", "中性", "有害"],
            ([WINDOW_NAMES[w],VARIANT_NAMES[v],rate(a["admission"]["admission_user_share"]),a["admission"]["users_with_admission"],
              a["admission"]["total_replacements"],fmt(a["admission"]["mean_replacements_per_admitted_user"],3),a["admission"]["max_replacements"],
              *[a["admission"][s] for s in ("inserted_cold_positive_pairs","removed_warm_positive_pairs","net_positive_pairs",
                  "beneficial_replacements","neutral_replacements","harmful_replacements")]]
             for w,d in windows.items() for v,a in d["variants"].items() if v in TAUS)))
        lines.append("准入比例分母为完整S_t。插入/移除以正例用户—商品对为单位；利好、中性、有害分别按每条匹配边净正例数为+1、0、−1计数。"
            "每个用户的联合MAP另按最终列表计算，不能简单累加孤立替换的AP差。下表各组与其同一用户的W0比较；"
            "不同替换组不是随机分配，因此组间差不是多插一件的因果效果。")
        lines.append(table(["外层", "方案", "替换件数组", "用户", "同组MAP差", "插入正例", "移除正例", "净正例", "对整窗MAP差贡献"],
            ([WINDOW_NAMES[w],VARIANT_NAMES[v],b,a["users"],fmt(a["delta_vs_same_users_w0"]),a["inserted_cold_positives"],
              a["removed_warm_positives"],a["net_positives"],fmt(a["delta_contribution_full_window"])]
             for w,d in windows.items() for v,row in d["variants"].items() if v in TAUS for b,a in row["admission_buckets"].items())))
        lines.append("精确匹配冲突、边数、总效用及完整12件检查见配套pair_utility（配对效用）/matching（匹配）审计。"
            "固定规范序和SciPy精确求解，不加微扰、不加位置系数、不改同分规则。存在多个同分最优解时，"
            "求解器选的槽位不是模型另行学出的槽位价值。")
    else:
        lines.append("固定三阈值的准入、配对效用、匹配和MAP均未运行。没有新的总体/冷组MAP、覆盖率或替换风险观测，维持W0。")
    if summary:
        lines.append(table(["方案", "总体均值", "总体差", "不降窗", "最差窗差", "暖组差", "严格冷差", "稀疏差", "冷/稀疏差", "独有冷正例插入窗", "全部门槛"],
            ([VARIANT_NAMES[v],fmt(s["mean_map12"]),fmt(s["mean_delta"]),s["nondegrade_windows"],fmt(s["worst_delta"]),
              *[fmt(s["segments"][g]["mean_delta"]) for g in ("warm_21_plus","strict_cold","sparse1_5","all_cold_sparse")],
              s["cold_only_positive_windows"],"通过" if s["gates"]["all_pass"] else "未通过"] for v,s in summary.items())))
        lines.append(table(["方案", "未通过检查"],([VARIANT_NAMES[v],"；".join(GATE_NAMES[k] for k,ok in summary[v]["gates"]["checks"].items() if not ok) or "全部通过"] for v in TAUS)))
    lines.append("晋级门槛不变：总体均值不降、至少3窗不降、最差差≥−0.000200；暖组均值差≥−0.000100且至少3窗差≥−0.000200；"
        "冷/稀疏均值严格升、至少3窗不降、至少2窗插入独有冷路正例；四窗合并插入正例严格多于移除。"
        "只在全部通过者中先选冷/稀疏均值最高，再总体均值，再更保守τ（数值近似相等容差沿合同1e-12）。")
    lines += ["", "## 6. 19项结论逐条回答", ""]
    def brief(key, formatter=sci):
        return "；".join(f"{WINDOW_NAMES[w]} {formatter(_geometry(old_qw.get(w),key))}→{formatter(_geometry(qw.get(w),key))}" for w in WINDOWS)
    focus_late = numerics.get("late_summer_20200819",{}).get("user_item_events_28d",{})
    focus_previous = focus_late.get("R1",{}).get("max_standardized_absolute_value")
    focus_current = focus_late.get("R2",{}).get("max_standardized_absolute_value")
    focus_answer = (f"相同夏末训练行中，从{fmt(focus_previous,3)}降至{fmt(focus_current,3)}，相对变化{_rel(focus_current,focus_previous)}；"
        "这是该列的标准化极值变化，不是正例率或MAP变化。" if focus_previous is not None and focus_current is not None else
        "夏末该字段未完成统一R2数值审计，不能推测其效果。")
    answers = [("qW统一R2是否4/4收敛？", f"已尝试{len(qw)}链、{successful}链收敛；" + ("四窗收敛门槛通过。" if successful == 4 else "未通过4/4门槛，校准和准入未运行。")),
        ("迭代怎样变化？", "；".join(f"{WINDOW_NAMES[w]} {_iterations(old_qw.get(w))}→{_iterations(qw.get(w))}" for w in WINDOWS)),
        ("条件数怎样变化？", brief("local_regularized_hessian_condition_number")),
        ("最大标准化绝对值怎样变化？", brief("max_standardized_absolute_value",lambda x:fmt(x,3))),
        ("夏末28天用户—商品计数极端值被压缩了吗？", focus_answer),
        ("有新的最大字段成为瓶颈吗？", "剩余最大字段为商品最近销售距截止日天数；它不是新特征，且本轮不允许取对数。"
         "第3节给出极值。最大者不自动构成瓶颈，不能凭名次认定唯一原因。" if len(qw)==4 else
         "第3节列出已运行链剩余最大字段；最大者不自动构成瓶颈，不能凭名次认定唯一原因。"),
        ("校准严重门槛通过了吗？", "通过，只排除了预注册严重跨窗失效。" if gate and gate["pass"] else "失败，按合同不进入准入。" if gate else "未运行。"),
        ("两侧AUC/Brier/ECE/截距斜率如何？", "第4节逐窗表及十箱机器审计。" if cal else "全部未运行，不能从收敛推测这些值。"),
        ("三个阈值覆盖如何？", "第5节逐窗覆盖、人数和替换总数。" if windows else "未运行。"),
        ("0/1/2/3/4+替换风险如何？", "第5节按同组用户原W0计算的MAP差与正例净数。" if windows else "未运行。")]
    if windows:
        multi, efficiency, subgroup, overall = [], [], [], []
        for v in TAUS:
            records = [d["variants"][v]["admission_buckets"][b] for d in windows.values() for b in ("2","3","4+")]
            multi.append(f"{VARIANT_NAMES[v]} ≥2件组用户{sum(r['users'] for r in records)}，插入/移除正例分别"
                         f"{sum(r['inserted_cold_positives'] for r in records)}、{sum(r['removed_warm_positives'] for r in records)}")
            s = summary.get(v)
            if s:
                a = s["admission_pooled"]
                efficiency.append(f"{VARIANT_NAMES[v]}插入{a['inserted_cold_positive_pairs']}对、移除{a['removed_warm_positive_pairs']}对")
                subgroup.append(f"{VARIANT_NAMES[v]}严格冷/稀疏MAP差分别{fmt(s['segments']['strict_cold']['mean_delta'])}、{fmt(s['segments']['sparse1_5']['mean_delta'])}")
                overall.append(f"{VARIANT_NAMES[v]}总体均值差{fmt(s['mean_delta'])}，总体门槛{'通过' if s['gates']['overall'] else '未通过'}")
        answers += [("多件替换累积暖风险吗？", "；".join(multi)+"。这是观测分组，不作单一因果解释。"),
            ("插入正例是否超过移除？", "；".join(efficiency) or "未完成四窗合并。"),
            ("严格冷与稀疏各拿回多少？", "；".join(subgroup) or "未完成四窗合并。"),
            ("总体MAP保护住了吗？", "；".join(overall) or "未完成四窗合并。")]
    else:
        answers += [(q,"未运行，不能判断。") for q in ("多件替换累积暖风险吗？","插入正例是否超过移除？","严格冷与稀疏各拿回多少？","总体MAP保护住了吗？")]
    passing = m.get("passing_variants", [])
    answers += [("有正式安全工作点吗？", "通过全部门槛："+(", ".join(passing) or "无") if summary else "尚未完成正式晋级判定，不宣称有安全点。"),
        ("选哪个阈值？", VARIANT_NAMES.get(selected,selected) if passing else "没有选出新阈值，保留W0。"),
        ("没有通过时的新主要瓶颈？", DECISION_TEXT.get(decision,decision)+"不自动增加迭代、调参、换模型或启动后续阶段。"),
        ("最终周仍未运行吗？", "是，2020-09-16=not_run。"),
        ("Warm-v2仍未整合吗？", "是，未合并也未拿它替换本轮W0，P4.3也未启动。")]
    lines.append(table(["序号","问题","回答"],([i,q,a] for i,(q,a) in enumerate(answers,1))))
    lines += ["", "## 7. 成本、验证、产物与停止", ""]
    r = m.get("resources", {})
    peak = r.get("peak_process_working_set_gib")
    lines.append(f"正式运行{fmt(r.get('formal_seconds'),2)}秒；qW拟合{len(qw)}次，qC复用、不重训；"
        "无新候选生成、无GPU训练/推理。CPU计算与验证事前估计为5–15分钟，实际用时以上述运行记录为准；代码与文档时间另计。"
        + (f"进程峰值工作集{fmt(peak,3)}GiB。" if peak and peak > 0 else "进程峰值工作集未获得有效读数，不把缺失/0值当零内存。"))
    checks = ver.get("checks", [])
    counts = {s:sum(a.get("status")==s for a in checks) for s in ("pass","not_run","fail")}
    lines.append(f"独立复核`{ver['status']}`，{counts['pass']}项通过、{counts['not_run']}项未运行、{counts['fail']}项失败；"
        f"耗时{fmt(ver.get('seconds'),2)}秒。实际执行边界按`P4_2R2_VERIFICATION.json`逐项区分；"
        "复核通过表示证据及停止自洽，不等于推荐方案晋级。合成机制测试不能替代尚未运行的外层匹配/最终12件证明。")
    test_path = folder / "P4_2R2_TEST_VERIFICATION.json"
    if test_path.exists():
        t = read_json(test_path)
        lines.append(f"单元/回归测试{t.get('tests_passed',t.get('tests'))}项通过，{t.get('tests_failed',t.get('failures',0))}项失败，"
            f"耗时{fmt(t.get('seconds',t.get('test_seconds')),3)}秒；"
            "执行时间和命令以`P4_2R2_TEST_VERIFICATION.json`为准，此机制测试不替代未运行的真实外层评测。")
    if m.get("failure_reason"):
        lines.append(f"实际停止原因：`{m['failure_reason']}`。完整失败参数、警告和追踪保留在新运行目录，未覆盖旧失败。")
    lines.append("本轮新qW模型、预处理和数值诊断位于被Git忽略的`artifacts/phase4/p4-2r2-*/`；"
        "训练矩阵、qC模型和qC预处理仍在原忽略目录按清单引用，不复制、不重训。本轮未生成外层预测。"
        "输出清单包含所有本轮大产物、公开报告和实现/测试文件的路径、字节和SHA；不把私有候选行内容写进报告正文。"
        "日志/路线继续忽略。未提交、未推送；本轮结束，不自动启动P4.3、Warm-v2整合或最终周。")
    (folder / "P4_2R2_FINAL.md").write_text("\n".join(lines).rstrip()+"\n",encoding="utf-8")
    paragraphs = ["此处追加P4.2R2实测，原P4.2和P4.2R的engineering_failure历史及未运行项目均保留。",
        "本轮单独授权所有qW链统一使用R1+固定19项计数log1p，仍同一LBFGS/C/tol/1000次预算，qC直接复用；"
        "不按窗口单独救援，不改用户/候选/效用/阈值/匹配或晋级门槛。",
        f"qW已尝试{len(qw)}链、{successful}链收敛；校准{len(cal)}窗、准入{len(windows)}窗。"
        f"机器结论`{decision}`，保留/选定`{selected}`。{DECISION_TEXT.get(decision,decision)}",
        f"正式运行{fmt(r.get('formal_seconds'),2)}秒；独立复核`{ver['status']}`。"
        "数值变换有效与概率校准、总体/冷组MAP兑现分开解释，最大标准化字段不自动构成唯一原因。",
        "证据：`reports/phase4/P4_2R2_FINAL.md`、metrics、数值对照、校准和准入审计、VERIFICATION、OUTPUT_MANIFEST。"
        "最终周仍not_run、Warm-v2未整合、P4.3未启动、无提交推送；路线/日志及大产物继续gitignore。"]
    _patch_doc(repo / "docs/ROADMAP_PHASE4.zh-CN.md","## 16. P4.2R2 四窗统一计数长尾修复实测（2026-09-09）",paragraphs)
    _patch_doc(repo / "docs/PROJECT_LOG.zh-CN.md","## 2026-09-09 — P4.2R2：全局计数log1p数值修复",paragraphs)
    manifest_path = folder / "P4_2R2_OUTPUT_MANIFEST.json"
    report_files = sorted({*folder.glob("P4_2R2*.md"),*folder.glob("P4_2R2*.json"),*folder.glob("p4_2r2_*.json")})
    artifacts = [identity(p) for p in sorted(root.rglob("*")) if p.is_file()]
    # Reused input files remain at their original ignored paths. Their trusted
    # records were checked by execution/verification; do not pretend copies were
    # created in this run or repeatedly hash identical prior assets for display.
    reused = {}
    for entry in c.get("prepared_reuse", {}).values():
        for record in entry.get("features", {}).values():
            reused[record["path"]] = {k:record[k] for k in ("path","bytes","sha256")}
    for entry in m.get("qc_reuse", {}).values():
        for name in ("model","preprocessing"):
            record = entry[name]
            reused[record["path"]] = {k:record[k] for k in ("path","bytes","sha256")}
    manifest = {"stage":STAGE,"run_id":c["run_id"],"schema_version":"p4.2r2-output-manifest-v1",
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"decision":decision,"selected_variant":selected,
        "verification_status":ver["status"],"reports":[identity(p) for p in report_files if p != manifest_path],
        "artifacts":artifacts,"sources":[identity(p) for p in sorted({*(repo / 'src/hm_recsys').glob('p42r2*.py'),*(repo / 'tests').glob('test_p42r2*.py')})],
        "artifact_files":len(artifacts),"artifact_bytes":sum(a['bytes'] for a in artifacts),
        "reused_artifacts":list(reused.values()),
        "reused_artifact_policy":"refer to original paths; checked against prior trusted manifests by run and independent verification; not copied or retrained",
        "prior_manifests":{k:c[k] for k in ("original_P4_2_manifest","prior_P4_2R_manifest") if k in c},
        "manifest_self_hash_excluded":True,"private_documents_ignored":["docs/ROADMAP_PHASE4.zh-CN.md","docs/PROJECT_LOG.zh-CN.md"],
        "historical_decisions_preserved":{"P4.2":"engineering_failure","P4.2R":"engineering_failure"},
        "final_week":"not_run","Warm_v2_integrated":False,"P4_3_started":False}
    write_json(manifest_path,manifest)
    return {"report":str(folder / "P4_2R2_FINAL.md"),"manifest":str(manifest_path),"decision":decision,
            "artifact_files":len(artifacts),"artifact_bytes":manifest['artifact_bytes']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    args = parser.parse_args()
    print(report(args.repo))
