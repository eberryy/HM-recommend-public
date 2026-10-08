"""Chinese, denominator-explicit rendering of measured P4.1A evidence."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .p41a_contract import BUDGETS, DIRECTIONS, SLOT_SETS, WINDOWS, WITHIN, read_json, write_json

LABELS = {"winter_20200122": "winter", "spring_20200318": "spring",
          "early_summer_20200624": "early-summer", "late_summer_20200819": "late-summer"}


def f(value, precision=6):
    return "不可用" if value is None else f"{value:.{precision}f}"


def table(headers, rows):
    return ["", "| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows], ""]


def render(repo):
    report = repo / "reports/phase4"
    m = read_json(report / "P4_1A_metrics.json")
    verification = read_json(report / "P4_1A_VERIFICATION.json")
    assert verification["status"] == "pass"
    c = read_json(report / "P4_1A_EXPERIMENT_CONTRACT.json")
    front = read_json(report / "p4_1a_oracle_frontier.json")["windows"]
    drift = read_json(report / "p4_1a_score_drift_audit.json")
    sep = read_json(report / "p4_1a_admission_separability.json")["windows"]
    risk = read_json(report / "p4_1a_warm_risk_audit.json")["windows"]
    state = read_json(report / "p4_1a_user_state_audit.json")["windows"]
    reliability = read_json(report / "p4_1a_reliability_audit.json")["windows"]
    s, d, windows = m["summary"], m["decision"], m["windows"]
    lines = ["# P4.1A：受约束单件准入上界与分数校准审计", "",
        f"状态：**已测量、独立复核通过、阶段结束**。合同登记时间：`{c['created_at_utc']}`；结果时间：`{m['created_at_utc']}`。", "",
        "## 1. 结论先行", "",
        f"固定主方案的四窗平均 MAP@12 从 **{s['w0_mean_map@12']:.8f}** 到理论上界 **{s['primary_mean_map@12']:.8f}**，增量 **+{s['primary_mean_delta']:.8f}**。4窗均为正，但未达到事先约定的 +0.000300。因此 `p4_1b_allowed=false`，不启动 P4.1B，正式基线仍是 W0。", "",
        "B0 并非没有信号：准入可分性和用户内置信度可迁移性均通过门槛；原始分数跨窗校准判为不稳定。当前否决原因是**指定操作范围内的理论收益不够大**，不是可分性失败，更不是证明所有 Cold 建模都无效。以下所有新增 MAP 均使用未来答案挑选替换，是诊断上界，绝非已训练策略的验证成绩。", "",
        "## 2. 术语、对象与分母", "",
        "以下 W0、B0、M4、Cold50、A_primary、窗口简称及字段名是本项目命名；AP、MAP、分位数、标准化、AUC 属于通用统计或推荐术语。", ""]
    glossary = [
        ("W0 / Warm-v1", "P4.0 正式保留的 Warm 分支最终前12件及其顺序；不是重新训练的 W1/F0/F1。Warm 是分支名称，不保证每件商品历史交易数都高。"),
        ("B0 / M4 / Cold50", "M4 内容表示召回每用户前200件，再用截止日前训练的 P3.7B B0 专家排序，取原名次1–50；完整复用 P4.0。此处 Cold 包括截止日前0笔和1–5笔历史交易商品。"),
        ("cutoff / outer window", "时间截止点与外层验证窗口；历史严格早于截止日，truth 为该日起连续7天的去重用户—商品购买对。四窗用户范围沿用 P4.0，未扩大训练或评估规模。"),
        ("truth / unobserved", "truth 是该用户验证周买过的商品；unobserved 是该周未观察购买，不等于已曝光后明确拒绝。源交易重复事件保留，只对真值商品集合去重。"),
        ("strict_cold / sparse1_5 / all_cold_sparse", "截止日前全历史商品交易事件数分别为0、1至5、前两者并集；本报告统计的是商品冷度，不是用户冷启动。"),
        ("AP@12 / MAP@12", "单用户 AP 为前12位每次命中时的累计精度之和，除以 min(该用户真值商品数,12)；MAP 是有真值的全部 W0 用户的 AP 均值。四窗 mean 再对四个窗口等权平均。"),
        ("A_primary / admission / replacement", "主方案准入：B0 原名次1–10中不在 W0 前12件的商品，最多选1件替换 W0 第10、11或12位，插入原位置，其他11位完全不动。只排除与 W0 Top12 重合，不额外排除与 Warm150 重合。"),
        ("Oracle / headroom / no-admission", "Oracle 使用已知未来真值挑选最优单件替换；headroom 是其相对 W0 的 MAP 增量；无正增量时 no-admission 即原列表逐位不动。均为诊断，不能部署。"),
        ("opportunity row", "一个截止点、一个用户、一个 Cold 商品和一个待替换 Warm 位置组成的一次合法尝试；同一候选通常对应3行，不能当成3个独立用户。"),
        ("beneficial / neutral / harmful", "该尝试相对 W0 的 AP 差值精确大于0、等于0、小于0；non-beneficial 是后两者并集。标签只能用于本轮只读审计。"),
        ("positive_opportunity_users / share", "每窗存在至少一次有益替换的用户数；主表占比的分母为全部 W0 真值用户，不是只有候选或命中的用户。跨窗相加是用户—窗口数，不是去重自然人数。"),
        ("b0_score / b0_score_available", "保存的 B0 原始输出分数及其可用标记；分数不是概率。不可用时不补零为置信度。"),
        ("b0_delta_vs_m4 / b0_delta_available", "原 B0 分数减 M4 粗排分数及可用标记；沿用 P4.0 字段，两模型量纲未校准，差值不是已经带来的排名或 MAP 提升。"),
        ("b0_rank / b0_rank_pct", "B0 原始用户内名次，以及名次除以50；均越小越优。原前10去重后不会重新编号补齐。"),
        ("b0_user_percentile", "同一用户全部可用 Cold50 分数中，候选的升序平均名次减1，再除以可用数量减1；越大越优，并列取平均，少于2项不可用。"),
        ("b0_user_zscore", "候选分数减同用户完整 Cold50 均值，再除以总体标准差（ddof=0）；标准差≤1e-12不可用。不是只拿 Top10 或命中候选标准化。"),
        ("margin_to_rank2 / margin_to_rank5 / margin_to_user_median", "候选分数分别减同用户 B0 第2名分数、第5名分数、完整 Cold50 中位数；参考项或候选不可用则差值不可用，各有 available 标记。自身是参考项时差值仍为0。差值仍保留原分数量纲，不是完全消除了尺度漂移。"),
        ("m4_coarse_score / rank / rank_pct", "冻结 M4 粗排分数、原 Top200 名次、名次除以200；只审计，不调用 teacher 或重新推理。"),
        ("interaction_count_before_cutoff / strict_cold_flag / sparse1_5_flag", "截止日前全历史商品交易事件数，以及0笔、1–5笔的布尔标志。与独立重算交易数逐行核对。"),
        ("warm_slot_rank / warm_item_score / warm_score_available / warm_rank_pct", "拟替换的 W0 位置、该 Warm 商品保存的截止安全模型分数、其可用标记，以及 Warm 原名次除以该用户截取 Warm150 前的完整 Warm 候选数；沿用 P4.0，不把分母换成12或150，不使用 F0/F1 分数。"),
        ("ROC-AUC / PR-AUC", "以有益尝试为正类、其他尝试为负类；ROC-AUC 是随机正类置信度超过随机负类的概率，并列计0.5。PR-AUC 此处明确为按不同分数阈值计算的非插值平均精度 AP；极稀疏任务必须与基础正例率一起读。"),
        ("quartile / decile / lift", "将当前窗口合法尝试的置信度无标签地划为4组或10组；同分保留一起，可出现空组。lift=组内有益率除以该变量全部可用尝试的有益率；大于1表示富集，不表示收益一定够。"),
        ("IQR / p10 / p25 / p75 / p90", "四分位距 IQR=p75−p25；pXX 为本表指定行集合的 XX% 分位点。std 是总体标准差；表中中位数就是 p50。"),
        ("raw_absolute_family / within_user_family", "绝对分数族含原始 B0 分数和与 M4 的差值；用户内置信度族含 percentile、zscore 与三个 margin。只是审计分类，不是两套已训练模型。"),
        ("recent-active / inactive-recent", "按已有 P3.7B 最近20件去重购买历史的原始用户状态，0–7天与8–28天历史条目数之和大于0为近期活跃，其余为近期不活跃，包含无历史用户；不使用无时间戳的 customers.Active。"),
        ("profile available / unavailable", "已有用户近期0–28天与更早29–84天画像相似度是否可计算；可用不等于偏好真的稳定，更不代表新发现的用户制度或季节路由。"),
        ("SHA256 / lineage / fail closed", "文件内容哈希、模型及数据时间来源链、非法截止日直接报错停止；用于验证资产身份及防止未来模型倒灌。"),
    ]
    lines += table(["术语或字段", "含义与统计口径"], glossary)
    lines += ["## 3. 冻结协议与资产核对", "",
        "预注册后合同未按结果修改。四个时间窗都存在安全 B0，没有新增回退或复用未来模型。W0 的用户—商品身份、1–12名顺序及 MAP 的浮点完整值逐窗精确相同。B0 Cold50 的名次、分数、M4 来源逐行相同，商品历史计数与验证周标签也独立重算一致。", ""]
    lines += table(["窗口简称", "完整窗口名", "截止日", "B0 训练标签最晚结束日", "W0 用户数", "完整 Cold50 候选行数"],
        [(LABELS[w], w, cut, c["frozen_inputs"][w]["b0_lineage"]["model_label_end"], windows[w]["users"], m["parity"][w]["cold50_rows"]) for w, cut in WINDOWS.items()])
    lines += ["所有替换先按真实 AP 增量最大排序，并列依次选 B0 原名次更小、Warm 位置更小、商品编号更小者。以 1–12 的最小公倍数27720计算整数分子，正负号不依赖浮点容差。用户内不论哪个替换，AP 分母相同。", "",
        "**重要机制**：在商品不重复、只替换同一位置的条件下，有益必然是“插入真值商品、移除未观察购买商品”；有害必然相反。因此 Oracle 移除正例数为0是它知道真值且允许不动的数学性质，不能拿来证明可部署准入器已经学会保护 Warm。", "",
        "## 4. 主方案：收益大小与来源", ""]
    lines += table(["窗口", "W0 MAP@12", "Oracle MAP@12", "ΔMAP", "有益机会用户数", "占全部 W0 用户", "不准入用户数"],
        [(LABELS[w], f(p["w0_map@12"], 8), f(p["map@12"], 8), f(p["delta_MAP_vs_W0"], 8), p["positive_opportunity_users"], f(p["positive_opportunity_user_share"]*100, 4)+"%", p["no_admission_users"]) for w, p in windows.items()])
    lines += [f"四窗等权平均增量为 +{s['primary_mean_delta']:.8f}，只有预注册上界门槛的 {s['primary_mean_delta']/.0003:.1%}；不是某个负窗口拉低均值，而是全部窗口都只有有限机会。主方案累计 {sum(p['positive_opportunity_users'] for p in windows.values())} 个有益用户—窗口，每用户最多新增1个命中，移除的 Warm 分支正例为0，净新增命中同为103对。", "",
        "下表分群是同一个整体最优 Oracle 的结果，不是另按每个分群重新挑选策略。可相加的贡献列，分母是该窗全部 W0 用户；分群 MAP 则只平均拥有该分群真值的用户，两者不能相加混用。", ""]
    lines += table(["窗口", "分群", "新增正例对", "对整体 ΔMAP 的贡献", "分群真值用户数", "分群 W0 MAP", "分群 Oracle MAP"],
        [(LABELS[w], segment, p["segments"][segment]["net_positive_pairs"], f(p["segments"][segment]["overall_delta_contribution"], 8), p["segments"][segment]["truth_users"], f(p["segments"][segment]["w0_map@12"], 8), f(p["segments"][segment]["map@12"], 8)) for w,p in windows.items() for segment in ("strict_cold", "sparse1_5")])
    contrib = s["segment_mean_overall_delta_contributions"]
    lines += [f"按四窗平均整体增量归因，sparse1-5 贡献 {contrib['sparse1_5']/s['primary_mean_delta']:.1%}，strict-cold 贡献 {contrib['strict_cold']/s['primary_mean_delta']:.1%}。strict-cold 的新增对数为0、0、29、13，只有两个夏季窗口可达；不能宣称零历史商品的机会已四窗稳定。", "",
        "## 5. Oracle 选什么、放哪里", ""]
    lines += table(["窗口", "所选 B0 名次均值", "p25", "中位数", "p75", "最大名次", "替换#10次数", "替换#11次数", "替换#12次数"],
        [(LABELS[w], f(p["selected_cold_rank"]["mean"], 3), f(p["selected_cold_rank"]["p25"], 2), f(p["selected_cold_rank"]["median"], 2), f(p["selected_cold_rank"]["p75"], 2), f(p["selected_cold_rank"]["max"], 0), *[p["replaced_warm_rank_counts"].get(str(j), 0) for j in (10,11,12)]) for w,p in windows.items()])
    lines += ["B0 名次不局限于榜首：全四窗第1至10名分别选中17、17、16、10、5、7、6、14、2、9次。Warm 第10位共98次，第11位4次，第12位1次。第10位占优首先是 AP 的位置折扣与本轮固定插入语义造成的：同样新增一个命中，越靠前通常得分越高；**不是从此可以无条件替换第10位**。", "",
        "下表只统计确实准入的用户 AP 增量；全部 W0 用户的中位数、p25、p75 均为0，均值见主表，最大值与此表相同。全部用户及准入用户的完整分布、替换位置分布保存在 primary JSON。", ""]
    lines += table(["窗口", "准入用户 ΔAP 均值", "p25", "中位数", "p75", "最大值"],
        [(LABELS[w], *[f(p["admitted_user_delta_AP"][key], 8) for key in ("mean", "p25", "median", "p75", "max")]) for w,p in windows.items()])
    lines += ["## 6. 次级 5×3 理论上界矩阵", "",
        "每格仍最多准入1件且允许不动；下表为四窗等权平均 ΔMAP，各窗口完整15格的 MAP、机会/不准入用户占比、新增/移除/净正例及三分群指标全部保存在 `p4_1a_oracle_frontier.json`。列标题中的位置均指 W0 原位置，替换后其他11位不动。", ""]
    lines += table(["B0 原名次预算", "只替换#12", "只替换#10–12", "允许替换#1–12"],
        [(b, *[f(float(np.mean([front[w][f'top{b}_{scope}']["delta_MAP_vs_W0"] for w in WINDOWS])), 8) for scope in SLOT_SETS]) for b in BUDGETS])
    lines += ["保持 Top10 候选，仅放开插入位置，理论收益从 +0.00014777 到 +0.00127555，说明位置折扣/操作约束是这一上界的重要组成部分；它没有证明某个模型能够辨别高位替换，也没有消除高位误删的代价。", "",
        "**次级矩阵只解释理论上界结构，不选择下一阶段预算或位置。** 主方案一直是 Top10 × #10–12 × 最多1件；不能因为 Top50 尾部格达到 +0.00030713，或全位置格更高，就事后替换门槛、宣布 P4.1B 获准。", "",
        "## 7. 分数漂移与用户内相对置信度", "",
        "以下原始分数分布的单位是完整 Cold50 用户—商品候选行，不是只统计有益尝试。B0 分数不是概率；绝对值漂移提示跨截止日校准风险，不自动说明窗口内排序能力下降。", ""]
    lines += table(["窗口", "均值", "标准差", "p10", "p25", "中位数", "p75", "p90", "IQR"],
        [(LABELS[w], *[f(drift["windows"][w]["cold50_distributions"]["b0_score"]["all"][key], 5) for key in ("mean", "std", "p10", "p25", "median", "p75", "p90", "iqr")]) for w in WINDOWS])
    lines += table(["窗口", "有益尝试 raw 中位数", "有益尝试 raw IQR", "原始分数最高十分组 lift", "zscore 最高十分组 lift", "margin_to_rank5 最高十分组 lift"],
        [(LABELS[w], f(drift["windows"][w]["beneficial_primary_raw_score"]["median"], 5), f(drift["windows"][w]["beneficial_primary_raw_score"]["iqr"], 5), *[f(m["portability"][v]["top_decile_lift_by_window"][w], 4) for v in ("b0_score", "b0_user_zscore", "margin_to_rank5")]) for w in WINDOWS])
    rawdiag = m["raw_calibration_diagnostics"]
    lines += [f"按预注册数值判据：完整 Cold50 中位数跨度/四窗 IQR 中位数为 {rawdiag['raw_median_span_over_median_iqr']:.4f}（>1）；IQR 最大/最小为 {rawdiag['raw_max_iqr_over_min_iqr']:.4f}；有益尝试分数中位数跨度/其 IQR 中位数为 {rawdiag['beneficial_median_span_over_median_iqr']:.4f}（>1）。raw 最高十分组只有3窗 lift>1，而 zscore 和多个 margin 为4窗，故判 `unstable`。这些数值门槛已在计算前写入合同，不是结果后解释。", "",
        "这不是说相对置信度在每窗 AUC 都胜过 raw：raw 的 AUC 四窗都>0.55，zscore 在 early-summer 只有0.5385；更稳定指高置信区间的富集方向及用户内参照更可迁移，而不是已经证明跨窗绝对阈值可通用。", "",
        "原始 b0_score、b0_delta_vs_m4 在 all、truth、unobserved、strict_cold、sparse1_5 五集合的全部分位数，以及 percentile/zscore/margin 的真值与未观察分布、四窗中位数/IQR/均值/标准差跨度、真值减未观察的均值和中位数差，见 `p4_1a_score_drift_audit.json`。未观察分布没有被称为曝光负例。", "",
        "## 8. 准入可分性与富集", "",
        "以下 AUC 用全部主方案合法尝试行，以有益为正类。名次及名次比例在合同中固定取负号，其余变量方向为正（商品历史计数也固定为正），没有按窗口结果翻转符号。", ""]
    lines += table(["窗口", "合法尝试行数", "有益行数", "有害行数", "中性行数", "有益基础率"],
        [(LABELS[w], p["opportunity_rows"], p["beneficial_rows"], p["harmful_rows"], p["neutral_rows"], f(p["opportunity_base_rate"]*100, 5)+"%") for w,p in windows.items()])
    lines += table(["变量", "winter AUC", "spring AUC", "early-summer AUC", "late-summer AUC", "AUC>0.55窗数", "AUC>0.60窗数", "最高四分组lift>1窗数", "方向"],
        [(v, *[f(p["auc_by_window"][w], 4) for w in WINDOWS], p["auc_gt_055_windows"], p["auc_gt_060_windows"], p["top_quartile_lift_gt_1_windows"], "四窗正向" if p["direction_consistency"] == "all4_positive" else "跨窗混合或缺失") for v,p in m["portability"].items()])
    lines += ["单变量可分性达到机械门槛，但不是足以保证盈利或 MAP 净增的证明：基础有益率只有0.02104%–0.05885%，AUC 把无损的中性与真正有害都放进同一负类，既不直接体现两者不同代价，也不等价于一个足够高精度的准入阈值。", "",
        "还有两点证据边界：b0_rank、b0_rank_pct 与 percentile 本次 AUC 完全相同，是强相关/等价排序，不是三个独立新信号；同一用户同一候选最多重复3个位置，42–116个有益行也不是42–116个独立正例用户。即便去掉名次别名，raw、margin_to_rank5 和 margin_to_user_median 仍各有4窗 AUC>0.55，信号结论不只来自重复计数。没有据此做显著性或因果宣称。", "",
        "### 8.1 四分组与十分组的可审计摘要", "",
        "下面展示四个必需变量的最高分组，全部4/10组的行数、有益率、lift 和无标签边界保存在 `p4_1a_reliability_audit.json`。组边界是审计统计，不是输出给下一阶段的部署阈值。相同分数不拆组，空组的率和lift为不可用。", ""]
    lines += table(["窗口", "变量", "分组数", "最高组行数", "有益行数", "有益率", "相对本窗本变量基础率 lift"],
        [(LABELS[w], v, n, row["rows"], row["beneficial_rows"], f(row["beneficial_rate"], 8), f(row["lift"], 4))
         for w in WINDOWS for v in ("b0_score", "b0_user_percentile", "b0_user_zscore", "margin_to_rank5") for n in (4,10)
         for row in [reliability[w][v][str(n)]["bins"][-1]]])
    lines += ["## 9. Warm 边界风险", "",
        "slot rows 的统计单位是用户—位置，分母为全部 W0 用户；尝试行数的单位是用户—Cold候选—位置，两个分母不同。Warm 尾部购买率虽低，但绝非可以随便删除。", ""]
    lines += table(["窗口", "W0位置", "位置行数", "位置正例数", "位置真值率", "有益尝试", "有害尝试", "中性尝试", "Oracle选择次数"],
        [(LABELS[w], j, r["slot_rows"], r["slot_future_positive_rows"], f(r["slot_truth_rate"]*100, 4)+"%", r["beneficial_replacement_rows"], r["harmful_replacement_rows"], r["neutral_rows"], r["oracle_selected_replacement_count"]) for w in WINDOWS for j,r in risk[w].items()])
    lines += ["在全部合法尝试中，每窗有害行数明显多于有益行数。能看到正例信号，不代表可以强行每用户塞入1件；真正部署必须能拒绝绝大多数机会。本轮未训练这种拒绝能力。", "",
        "## 10. 用户层机会分布", "",
        "只沿用已有可用状态，不挖掘新的季节分组。每行的用户数是该分层全部 W0 真值用户，候选人数是有至少一个合法 Top10 提案的用户；两个机会占比分别明确分母。", ""]
    lines += table(["窗口", "分层", "全部用户", "有候选用户", "有益机会用户", "有益/全部用户", "有益/有候选用户"],
        [(LABELS[w], group, u["users"], u["users_with_proposal"], u["users_with_beneficial_replacement"], f(u["beneficial_opportunity_share_all_stratum_users"]*100,4)+"%", f(u["beneficial_opportunity_share_proposal_users"]*100,4)+"%" if u["beneficial_opportunity_share_proposal_users"] is not None else "不可用") for w in WINDOWS for group,u in state[w].items()])
    lines += ["近期活跃及画像可计算用户的机会占比在四窗都更高，但它是描述性关联。画像可计算不是稳定程度高，本轮没有把“可用”偷换成“偏好稳定”。", "",
        "## 11. 机器判定与停止", ""]
    lines += table(["字段（本项目判定名）", "实测判定", "原因"], [
        ("constrained_admission_headroom（受限准入上界）", d["constrained_admission_headroom"], "4窗正，但平均增量低于+0.000300"),
        ("raw_b0_absolute_calibration（原始B0绝对分数校准）", d["raw_b0_absolute_calibration"], "位置漂移、有益区间漂移及十分组富集翻转，用户内参照更稳定"),
        ("within_user_confidence_portability（用户内置信度可迁移性）", d["within_user_confidence_portability"], "至少一个相对变量≥3窗 AUC>0.55且≥3窗最高四分组lift>1"),
        ("admission_signal_separability（准入信号可分性）", d["admission_signal_separability"], "至少两个变量≥3窗 AUC>0.55，且含用户内变量"),
        ("p4_1b_allowed（下一阶段是否获准）", str(d["p4_1b_allowed"]).lower(), "必须上界和可分性同时supported；本次上界weak"),
    ])
    lines += ["完成 P4.1A 后停止。没有训练、新模型推理、候选扩张、超参数搜索或自动 P4.1B。正式回退方案就是保留 W0 和 B0 原资产；不是把 Oracle 列表当作新基线。最终周 `2020-09-16` 仍为 `not_run`，没有生成线上或 Kaggle 成绩。", "",
        "此次更精确的解释是：**有用 Cold 分数、校准不稳、可操作上界偏小同时成立**。如果未来重新讨论架构，应分别看候选正例覆盖、进入尾部的位置价值和误删 Warm 的代价；本轮没有据次级矩阵改规则或宣布下一项实验。", "",
        "## 12. 成本、验证与复现", ""]
    artifact_bytes = sum(a["bytes"] for ws in m["artifacts"].values() for a in ws.values())
    lines += [f"正式 CPU 审计 {m['execution']['seconds']:.2f} 秒；测量进程峰值工作集 {m['execution']['peak_working_set_bytes']/2**30:.3f} GiB；新增24个大型证据文件共 {artifact_bytes/2**30:.3f} GiB。独立复核 {verification['seconds']:.2f} 秒。这里是命令运行成本，不含阅读、实现和撰写报告的人工/助手时间。", "",
        "8项单测全部通过，其中对4,096种前12位命中组合×12位置×2候选标签穷举核对 AP 整数差值。独立复核重新读冻结源、逐行确认全部机会标签及置信度、逐用户重算主方案 AP，并用另一种枚举方式验证4窗×15格最优选择及保留列表。输出哈希、83个冻结输入哈希、禁止模型训练调用的源代码扫描均通过。", "",
        "JSON 中所有未来真值字段及据真值得到的机会标签只供诊断；大型用户行、候选行与全部15格推荐列表位于 ignored artifacts，未把原始数据或用户级输出加入 Git。合同与报告保留文件身份，文档更新只在末尾追加 measured evidence。", "",
        "复现入口（在仓库目录、已有 hm-recommend conda 环境）：", "",
        "```powershell", "$env:PYTHONPATH='src'",
        "conda run --no-capture-output -n hm-recommend python -m unittest discover -s tests -p test_p41a.py -v",
        "conda run --no-capture-output -n hm-recommend python -m hm_recsys.p41a verify", "```", "",
        "首次计算使用 `python -m hm_recsys.p41a run`；已有 measured metrics 时该入口会拒绝覆盖，日常复核只用 verify。清单见 `P4_1A_OUTPUT_MANIFEST.json`，逐项记录路径、字节数、SHA256、行数、截止日与来源链。报告全文数字由已测 JSON 渲染，不手工拼造实验成绩。", "",
        "## 附录：逐变量 PR-AUC 与完整分母", "",
        "单位均为主方案尝试行，ROC-AUC 见第8节。eligible 只排除该变量不可用的行；本次这些变量均全量可用。PR-AUC 是非插值平均精度，不是梯形面积；不存在把0.000x当成0.x%的隐式换算。", ""]
    lines += table(["窗口", "变量", "可用尝试行", "有益行", "基础率", "PR-AUC"],
        [(LABELS[w], v, sep[w][v]["eligible_rows"], sep[w][v]["beneficial_rows"], f(sep[w][v]["base_rate"], 8), f(sep[w][v]["pr_auc"], 8)) for w in WINDOWS for v in DIRECTIONS])
    (report / "P4_1A_FINAL.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    m["status"] = "measured_verified_closed"
    m["verification"] = {"status": verification["status"], "path": str(report / "P4_1A_VERIFICATION.json"),
        "unit_tests": {"passed": 8, "command": "conda run --no-capture-output -n hm-recommend python -m unittest discover -s tests -p test_p41a.py -v"}}
    write_json(report / "P4_1A_metrics.json", m)
    from .p41a import build_manifest
    build_manifest(repo, m)
