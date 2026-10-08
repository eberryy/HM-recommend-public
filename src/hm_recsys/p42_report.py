"""Render P4.2 measured evidence in Chinese; no fitting or evaluation."""
from __future__ import annotations

from pathlib import Path

from .p41a_contract import identity, read_json, write_json
from .p42_contract import RUN_ID, TAUS, WINDOWS


def fmt(x, digits=8):
    return "不可用" if x is None else f"{x:.{digits}f}"


def table(header, rows):
    def row(values):
        return "| " + " | ".join(str(x) for x in values) + " |"
    return "\n" + row(header) + "\n" + row(["---"]*len(header)) + "\n" + "\n".join(row(r) for r in rows) + "\n"


def feature_note(name):
    notes={
        "b0_rank_pct":"B0 原名次除以50，越小越靠前。",
        "b0_user_percentile":"同用户完整 Cold50 可用分数的升序平均名次减1，再除以可用数减1；越大越优。",
        "b0_user_zscore":"候选 B0 分数减同用户完整 Cold50 均值，再除以总体标准差。",
        "normalized_margin_to_rank2":"候选 B0 分数减同用户原第2名分数，再除以完整 Cold50 总体标准差。",
        "normalized_margin_to_rank5":"候选 B0 分数减同用户原第5名分数，再除以完整 Cold50 总体标准差。",
        "normalized_margin_to_user_median":"候选 B0 分数减完整 Cold50 中位数，再除以该集合总体标准差。",
        "m4_coarse_rank_pct":"M4 原粗排名次除以200。",
        "interaction_count_before_cutoff":"该商品截止日前全部历史购买事件数；不是用户自己的购买数，重复事件保留。",
        "strict_cold_flag":"上述商品历史事件数为0的标记。",
        "sparse1_5_flag":"上述商品历史事件数在1至5之间的标记。",
        "warm_rank":"冻结 W0 原位置1至12，不会重新排序暖侧。",
        "warm_rank_pct":"原暖侧名次除以截取 Warm150 前该用户完整候选数，通常100或300；不是除以12/150。",
        "warm_user_percentile":"同用户 W0 完整12件可用模型分数的升序平均名次百分位。",
        "warm_user_zscore":"同用户 W0 完整12件可用分数的总体标准分。",
        "warm_model_score_available":"该截止日存在时间安全的暖侧模型分数为1，否则0。",
        "fused_score":"已有多路加权倒数名次融合分数，沿用冻结值。",
        "source_count":"该用户—商品获得的已有召回路支持数，表示来源一致性证据。",
        "days_since_last_purchase":"P3.7B 最近20件不同已购商品中，最近一次购买距截止日天数。",
        "recent_0_28_purchase_share":"P3.7B 最近20件不同已购商品中，购买年龄0至28天的条目占有效历史条目数的比例。",
        "user_history_events_12w":"该用户截止前12周全部购买事件数。",
        "user_unique_items_12w":"该用户截止前12周不同已购商品数。",
        "user_days_since_last_purchase":"已有暖侧12周历史中最近购买距截止日天数。",
        "user_online_share_12w":"用户截止前12周线上购买事件占全部购买事件的比例。",
        "item_days_since_last_sale":"已有商品历史窗口中最近一次销售距截止日天数。",
        "item_trend_7d_vs_28d":"已有商品近7天与28天活动量对比，沿用暖侧特征定义，不新建趋势路由。",
        "item_unique_customers_28d":"商品截止前28天不同购买用户数。",
        "item2vec_is_new":"已有 Item2Vec 路是否首次补入此用户—商品候选的标记。",
        "item2vec_cosine":"已有 Item2Vec 协同表示召回记录的相似度。",
        "item2vec_best_seed_rank":"已有协同召回最佳历史种子在用户种子列表中的名次。",
        "item2vec_best_neighbor_rank":"已有协同召回中该商品在最佳种子近邻列表里的名次。",
        "item2vec_seed_support":"已有协同召回中支持该候选的历史种子数量。",
        "item2vec_vocab_count":"已有协同表示训练词表中该商品计数。",
        "repurchase_score":"复购召回路已有原始得分，不重算新复购规则。",
        "user_item_days_since_last_purchase":"已有用户—候选商品交叉中，最近一次购买该商品距截止日天数。",
    }
    if name in notes:
        return notes[name]
    sources={"repurchase":"复购", "recent_popularity":"近期热门", "product_family":"商品家族",
        "user_day_covisit":"用户同日共购", "age_popularity":"用户年龄分组热门", "attribute_content":"属性内容相似", "item2vec":"Item2Vec 协同表示"}
    for source,label in sources.items():
        if name==source+"_present":
            return f"已有{label}召回路是否支持该用户—商品，0/1。"
        if name==source+"_rank":
            return f"商品在已有{label}召回路中的原名次，越小越靠前；不是类别编号。"
        if name==source+"_rrf_contribution":
            return f"已有{label}召回路对加权倒数名次融合的分数贡献。"
    if name.startswith("history_count_"):
        return "P3.7B 最近20件不同已购商品中，购买年龄落入"+name.removeprefix("history_count_").replace("_","至").replace("over至","超过")+"天的条目数；不是交易事件总数或用户年龄。"
    family={"product_code":"商品家族", "product_type":"商品类型", "department":"部门", "garment":"服装组"}
    for key,label in family.items():
        prefix="user_"+key
        if name.startswith(prefix):
            suffix=name[len(prefix):]
            return {"_events_28d":f"用户截止前28天购买候选所属{label}的事件数。",
                "_events_12w":f"用户截止前12周购买候选所属{label}的事件数。",
                "_days_since":f"已有交叉中用户最近购买候选所属{label}距截止日天数。",
                "_share_12w":f"用户截止前12周购买候选所属{label}的事件占用户12周全部购买事件的比例。"}[suffix]
    if name.startswith("user_item_events_"):
        return "用户在截止前"+name.removeprefix("user_item_events_")+"购买候选商品的事件数。"
    if name.startswith("item_events_"):
        return "该商品在截止前"+name.removeprefix("item_events_")+"的购买事件数。"
    raise ValueError("unexplained feature: "+name)


def report(repo):
    repo=Path(repo).resolve(); folder=repo / "reports/phase4"
    m=read_json(folder / "P4_2_metrics.json")
    c=read_json(folder / "P4_2_EXPERIMENT_CONTRACT.json")
    training=read_json(folder / "p4_2_training_audit.json")
    cal=read_json(folder / "p4_2_propensity_calibration.json")["windows"]
    verification=read_json(folder / "P4_2_VERIFICATION.json") if (folder / "P4_2_VERIFICATION.json").exists() else {"status":"pending"}
    s=m["summary"]
    decision_notes={"promote_p4_2_aggressive":"激进阈值通过所有风险门槛，允许在 W0 上晋级。", "promote_p4_2_moderate":"中等阈值通过所有风险门槛，允许在 W0 上晋级。", "promote_p4_2_conservative":"保守阈值通过所有风险门槛，允许在 W0 上晋级。", "pareto_frontier_supported_but_no_safe_operating_point":"更激进的阈值同时带来更多冷侧收益与更多暖侧损伤，三个固定点均不安全。", "cold_gain_not_recovered":"三个固定点未形成满足完整要求的冷侧兑现，不可凭局部收益晋级。", "warm_risk_uncontrolled":"至少一个点提高冷侧均值，但总体或暖侧保护失败。", "propensity_calibration_failure":"依预注册判据出现跨至少三窗的严重概率率比、方向或恒定预测问题。", "engineering_failure":"工程或数据证据不足，不能视为已验证的模型结论。"}
    lines=["# P4.2：双侧购买倾向与风险阈值控制的局部准入", "",
        f"状态：`{m['status']}`；独立复核：`{verification['status']}`。合同登记于 `{c['created_at_utc']}`，结果生成于 `{m['created_at_utc']}`。",
        "", "## 1. 结论", "",
        f"机器结论：`{m['decision']}`；最终保留：`{m['selected_variant']}`。三个预注册阈值中通过全部门槛的方案为：`{m['passing_variants']}`。",
        decision_notes[m['decision']],
        "本轮独立获得授权，不改写 P4.0 的暖侧损伤结论；`p4_1b_allowed=false`、`P4_1B_started=false` 保持不变。Warm-v2 未整合，P4.3 未启动，最终周 `2020-09-16=not_run`。",
        "", "## 2. 术语、架构与统计口径", "",
        "W0（本项目基线名）是冻结 Warm-v1 的每用户前12件及原顺序。B0（本项目冷侧专家名）只在冻结 M4 内容表示的前200件内重排；Cold50 指其原前50件，含零历史和1至5笔历史交易商品。本轮不重训这两个上游。",
        "qC/qW（本项目双侧倾向名）分别预测在 B0 Cold50 / W0 Top12 候选条件下，该用户未来7天是否被观察到购买该商品；不是曝光后的线上真实购买概率。未观察购买不等于明确拒绝。",
        "两个 L2 正则逻辑回归（行业通用二分类模型）独立训练和标准化，全部历史行保留；不做负采样、正例复制或类别加权。数值缺失由训练集本列中位数填补，整列缺失填0，并增加可用标记。只标准化连续数值，布尔和可用标记保持0/1。",
        "效用 U=logit(qC)−logit(qW)，logit 是购买倾向赔率的自然对数；概率先裁剪至 [1e-6,1−1e-6]。阈值 τ 只取0、ln2、ln4，对应冷侧赔率必须严格高于暖侧的1、2、4倍。等于阈值仍拒绝，不乘位置系数。",
        "最大权二分匹配（行业通用图优化）只在 U>τ 的边上最大化总和 U−τ，冷商品和暖位置各最多使用一次；允许每人0至12次替换，没有最多1件限制。被选冷商品放到匹配的原暖位置，其余位置完全不动；没有合格边时逐位还原 W0。",
        "同分最大效用的配对可能不唯一。本轮按原 B0 名次、商品编号和原暖位置规范排序，固定 SciPy1.17.1 求解器处理并列，不加微小扰动。效用不含位置项，所以不能把同分配对的任一位置安排说成另学到的位置收益。",
        "MAP@12（行业通用离线指标）：单用户前12位每次命中时累计精度之和除以 min(该用户真值商品数,12)，再对该窗全部 W0 真值用户取平均；四窗均值对窗口等权。truth/positive pair 在此指不同的截止日—用户—购买商品对，不是交易事件数。",
        "strict-cold、sparse1-5、warm_21_plus（本项目商品冷度组）分别指截止日前全历史交易事件数为0、1至5、至少21；all_cold_sparse 为前两者合并。分群 MAP 的分母是所有拥有该分群真值的 W0 用户，推荐列表不删掉其他分群商品腾位置。",
        "cold-only（本项目来源组）指在 Cold50 但不在 Warm150，不能与‘只是不在 W0 Top12’混为一谈。准入可考虑 Warm150 第13位以后也出现的冷候选，但 cold-only 门槛采用更严格的 Warm150 参照。",
        "beneficial/neutral/harmful（本项目单次替换审计标签）按每条实际替换单独相对 W0 的 AP 正/零/负增量计数；多次同时替换的实际 MAP 由完整最终列表另算，不能把单次增量强行相加。",
        "", "## 3. 四窗总体与分群结果"]
    lines.append(table(["方案","四窗mean MAP@12","相对W0均值差","不退化窗数/4","最差窗差","全部门槛"],
        [[v,fmt(x["mean_map12"]),fmt(x["mean_delta"]),x["nondegrade_windows"],fmt(x["worst_delta"]),x["gates"]["all_pass"] if v!="W0" else "基线"] for v,x in s.items()]))
    lines.append(table(["窗口（仅标签，不是模型输入）","W0","τ=0","τ=ln2","τ=ln4"],
        [[w,*[fmt(x["variants"][v]["map12"]) for v in ("W0",*TAUS)]] for w,x in m["windows"].items()]))
    lines.append(table(["方案","Warm21+ mean","strict-cold mean","sparse1-5 mean","cold/sparse mean"],
        [[v,*[fmt(x["segments"][seg]["mean_map12"]) for seg in ("warm_21_plus","strict_cold","sparse1_5","all_cold_sparse")]] for v,x in s.items()]))
    lines.append("各窗完整分群 MAP 和真值分母如下；分母不随准入方案变化。")
    lines.append(table(["窗口","商品组","全部真值用户数","真值对数","W0","τ=0","τ=ln2","τ=ln4"],
        [[w,seg,x["variants"]["W0"]["segments"][seg]["truth_users"],x["variants"]["W0"]["segments"][seg]["truth_pairs"],
          *[fmt(x["variants"][v]["segments"][seg]["map12"]) for v in ("W0",*TAUS)]] for w,x in m["windows"].items() for seg in ("warm_21_plus","strict_cold","sparse1_5","all_cold_sparse")]))
    lines += ["", "## 4. 准入覆盖、正例交易与多次替换风险", "",
        "覆盖率分母为全部 W0 真值用户。下表跨窗累计的是用户—窗口数，不是跨窗去重人数；移除 Warm 正例按来源分支计数，并非仅历史21笔以上商品。"]
    lines.append(table(["阈值方案","准入用户—窗口数","准入覆盖率","替换次数","插入正例","移除正例","净正例","strict插入","sparse插入","cold-only命中窗数/4"],
        [[v,x["admission_pooled"]["users_with_admission"],fmt(x["admission_pooled"]["admission_user_share"],6),
          *[x["admission_pooled"][k] for k in ("total_replacements","inserted_cold_positive_pairs","removed_warm_positive_pairs","net_positive_pairs","strict_cold_positive_insertions","sparse1_5_positive_insertions")],x["cold_only_positive_windows"]] for v,x in s.items() if v!="W0"]))
    lines.append(table(["窗口","方案","准入用户","每准入用户平均替换","最大替换","有益次数","中性次数","有害次数","插入/移除正例","cold-only正例"],
        [[w,v,a["users_with_admission"],fmt(a["mean_replacements_per_admitted_user"],3),a["max_replacements"],a["beneficial_replacements"],a["neutral_replacements"],a["harmful_replacements"],f"{a['inserted_cold_positive_pairs']} / {a['removed_warm_positive_pairs']}",a["cold_only_positive_top12"]]
         for w,x in m["windows"].items() for v in TAUS for a in [x["variants"][v]["admission"]]]))
    lines.append("下表0/1/2/3/4+表示同一用户执行的替换次数。桶内 ΔMAP 与同一批用户的 W0 比较；‘全窗贡献’把该桶 AP 差值总和除以全部窗用户数。桶随模型决定，不能据此宣称次数的因果效应。")
    lines.append(table(["窗口","方案","替换次数桶","用户数","桶内ΔMAP","全窗ΔMAP贡献","插入正例","移除正例","净正例"],
        [[w,v,b,z["users"],fmt(z["delta_vs_same_users_w0"]),fmt(z["delta_contribution_full_window"]),z["inserted_cold_positives"],z["removed_warm_positives"],z["net_positives"]]
         for w,x in m["windows"].items() for v in TAUS for b,z in x["variants"][v]["admission_buckets"].items()]))
    lines += ["", "## 5. 倾向模型的可分性与校准", "",
        "ROC-AUC 是随机正例得分高于未观察候选的概率（并列计一半）；PR-AUC 此处为非插值平均精度，需要与极低正例率一起看。Brier 是概率与0/1标签的均方误差；ECE 是10个等行数概率分组里预测均值与实测率之差绝对值的加权平均，极稀疏时数值小不自动意味着校准好。",
        "校准截距/斜率是只用于事后诊断的 y~截距+斜率×logit(q) 无正则拟合，理想为0/1；不反过来修正预测。相同预测可按固定原始行顺序分到相邻等频组，分组不看标签。"]
    lines.append(table(["窗口","侧","候选行","正例行","实测率","预测均值","预测/实测","ROC-AUC","PR-AUC","Brier","ECE","截距","斜率"],
        [[w,side,x["rows"],x["positives"],fmt(x["observed_positive_rate"]),fmt(x["mean_predicted_probability"]),fmt(x["predicted_to_observed_rate_ratio"],3),fmt(x["roc_auc"],6),fmt(x["pr_auc"]),fmt(x["brier_score"]),fmt(x["ece"]),fmt(x["calibration_intercept"],4),fmt(x["calibration_slope"],4)] for w,ys in cal.items() for side,x in ys.items()]))
    lines.append("原始 B0 数值尺度不进入 qC 主模型；但删除原始尺度只是一项控制，不保证学到的概率自动跨窗校准。上表预测/实测比率与校准斜率才是本轮实际证据。完整10组表、概率裁剪数量与警告在 `p4_2_propensity_calibration.json`。")
    lines.append(table(["窗口","侧","校准警告（本项目诊断标签）"],[[w,side,", ".join(x["warnings"]) or "无"] for w,ys in cal.items() for side,x in ys.items()]))
    lines.append("警告标签说明：`mean_probability_to_observed_rate_outside_0.25_4`=预测均值/实测率超出0.25至4；`roc_auc_not_above_chance`=ROC-AUC≤0.5；`nonpositive_calibration_slope`=校准斜率非正；`constant_prediction`=裁剪后概率几乎恒定；`single_class_evaluation`=本侧只有一种标签；`calibration_line_diagnostic_failed`=诊断拟合未收敛；`calibration_line_no_finite_mle_due_to_separation`=诊断数据完全/准完全可分，无有限最大似然斜率。这些本项目警告不自动等于晋级失败，正式失败分类另按合同的严重判据。")
    lines += ["", "## 6. 历史训练链与实现检查", "",
        "所有八次正式拟合完成后才评测三个准入阈值。训练标签结束日必须严格早于目标外层截止日；上游 B0/Warm 模型还分别检查自身标签结束日不晚于所服务截止日。最早2019-11-27没有 B0 分数，不进入 qC；qW 保留该日并标记分数缺失。冷暖历史用户不同，各自保留完整总体。"]
    lines.append(table(["目标外层","侧","训练截止日","最晚标签结束","训练行","正例行","基础率","迭代数","拟合秒数"],
        [[w,side,", ".join(a["training_cutoffs"]),a["latest_training_label_end"],a["training_rows"],a["positives"],fmt(a["base_rate"]),a.get("n_iter",read_json(Path(a["model"]["path"]))["n_iter"]),fmt(a["fit_seconds"],2)] for w,ys in training["windows"].items() for side,a in ys.items()]))
    lines.append(table(["窗口","Cold原行数","排除Top12重合行","合法Cold行","全部Warm位置","合法配对行数","每用户最大配对"],
        [[w,*[x["pair_audit"][k] for k in ("complete_cold_rows","overlap_excluded_cold_rows","cold_nodes","warm_nodes","pair_rows","maximum_pairs_per_user")]] for w,x in m["windows"].items()]))
    lines.append("匹配效率定义为实际匹配边数/超过阈值的边数；冲突数是同一侧拥有两条以上合格边的节点数，不是重复插入次数。贪心差异仅用于诊断，可能只是同分最优配对不同，未用贪心列表计算另一组 MAP。")
    lines.append(table(["窗口","方案","阈值后边数","实际匹配","匹配效率","Warm冲突节点","Cold冲突节点","与贪心不同用户数"],
        [[w,v,x["edges_above_tau"],x["matched_edges"],fmt(x["matching_efficiency"],6),x["same_warm_conflict_count"],x["same_cold_conflict_count"],x["greedy_would_differ_count"]] for w,ys in m["windows"].items() for v,x in ys["matching"].items()]))
    lines += ["", "## 7. 固定晋级门槛与停止", "",
        "总体：四窗均值不低于 W0、至少3窗不退化、最差差值≥−0.0002；暖侧21+：均值差≥−0.0001，至少3窗差≥−0.0002；冷/稀疏：均值严格提高、至少3窗不退化，cold-only正例实际插入至少2窗；且累计插入正例严格多于移除正例。全部使用未四舍五入数值。"]
    lines.append(table(["方案","总体保护","暖侧保护","冷侧兑现","正例替换效率","全部通过"],
        [[v,*[s[v]["gates"][k] for k in ("overall","warm","cold","replacement_efficiency","all_pass")]] for v in TAUS]))
    lines.append("多方案通过时，先取冷/稀疏 mean MAP最高，1e-12内近似同值再取总体最高，仍同值取更保守阈值。非晋级结论依合同预注册顺序区分严重校准失败、风险—收益变化、暖侧失控或冷收益未兑现，不事后加阈值/每人硬上限。")
    lines.append(f"本轮选择 `{m['selected_variant']}`。即使存在局部分群收益，也不越过整体与暖侧保护门槛。最终周未读、未产生其 MAP、提交文件或排行榜成绩；不自动开始 P4.3。")
    lines += ["", "## 8. 成本、证据与复核", "",
        f"正式运行 {fmt(m['resources']['formal_seconds'],2)} 秒；进程峰值工作集 {fmt(m['resources']['peak_process_working_set_gib'],3)} GiB；CPU only，共8次固定正式拟合。此耗时不含方案阅读、编码、环境安装、合成测试、独立复核与报告撰写。",
        "大型逐行特征、模型、标准化器、预测、配对效用矩阵与列表都保存在项目 ignored artifacts/phase4/p4-2-v1-two-sided-risk-controlled-admission，不公开用户级数据。文件来源清单记录路径、大小、SHA、行数与时间链；哈希用于与既有可信资产和本轮证据合同比较，不把‘只对当前文件算一个哈希’当成真实性证明。",
        "复核入口：在 hm-recommend conda 环境设置 PYTHONPATH=src，运行 `python -m hm_recsys.p42 verify`；合成测试为 `python -m unittest discover -s tests -p 'test_p42*.py'`。已有结果的 run 入口拒绝覆盖。",
        "", "## 附录：确切输入特征及中文解释", "",
        "下面英文列名均为本项目既有/派生字段名，单位为一条截止日—用户—候选商品观察；除显式占比与名次外，计数单位是事件或已说明的历史条目。12w=截止前12周，28d/7d=截止前28/7天。按来源原有定义的缺失值保留缺失，由训练期处理器处理。"]
    for side,spec in c["feature_spec"].items():
        lines += ["",f"### {side}","",f"{len(spec['numeric'])} 个数值字段、{len(spec['binary'])} 个布尔字段；另为每个数值字段增加一个同名 `_available` 标记（有有限值为1，否则0），共 {2*len(spec['numeric'])+len(spec['binary'])} 个模型输入。"]
        lines.append(table(["精确字段名","类型","含义/分母"],[[name,kind,feature_note(name)] for kind,cols in (("数值",spec["numeric"]),("布尔",spec["binary"])) for name in cols]))
    (folder / "P4_2_FINAL.md").write_text("\n\n".join(part for part in lines if part)+"\n",encoding="utf-8")
    make_manifest(repo)


def make_manifest(repo):
    folder=repo / "reports/phase4"; root=repo / "artifacts/phase4" / RUN_ID
    m=read_json(folder / "P4_2_metrics.json")
    training=read_json(folder / "p4_2_training_audit.json")
    artifacts={a["path"]:a for a in m["artifacts"]}
    for ys in training["windows"].values():
        for a in ys.values():
            for key in ("model","preprocessing"):
                artifacts[a[key]["path"]]=a[key]
    for prepared in training["prepared_cutoffs"].values():
        for entry in prepared["features"].values():
            artifacts[entry["path"]]=entry
    # Small run receipts and immutable authority snapshots are evidence too.
    receipt=read_json(root / "input-verification.json")
    for entry in receipt["authority_snapshots"].values():
        p=entry["snapshot_path"]
        artifacts[p]={"path":p,"bytes":entry["bytes"],"sha256":entry["sha256"],"cutoff":None,"source_lineage":{"snapshot_of":entry["path"]}}
    for name in ("EXECUTION_START.json","input-verification.json","PREPARED_MANIFEST.json"):
        p=root / name
        artifacts[str(p)]={**identity(p),"cutoff":None,"source_lineage":{"run_id":RUN_ID}}
    paths=sorted(set(folder.glob("P4_2*")) | set(folder.glob("p4_2*")))
    paths=[p for p in paths if p.name!="P4_2_OUTPUT_MANIFEST.json" and p.is_file()]
    sources=sorted((repo / "src/hm_recsys").glob("p42*.py"))+sorted((repo / "tests").glob("test_p42*.py"))
    write_json(folder / "P4_2_OUTPUT_MANIFEST.json",{"stage":"P4.2","run_id":RUN_ID,"reports":[identity(p) for p in paths],
        "sources":[identity(p) for p in sources],"artifacts":list(artifacts.values()),"decision":m["decision"],"selected_variant":m["selected_variant"],"final_week":"not_run"})
