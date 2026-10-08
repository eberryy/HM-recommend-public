from __future__ import annotations

from pathlib import Path
from typing import Any


WINDOW_LABELS = {
    "winter_20200122": "winter",
    "spring_20200318": "spring",
    "early_summer_20200624": "early-summer",
    "late_summer_20200819": "late-summer",
}


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def write_p33(report_dir: Path, result: dict[str, Any]) -> Path:
    comparison = result["comparison"]
    representation = comparison["representation_gate"]
    end_to_end = comparison["end_to_end_gate"]
    lines = [
        "# P3.3：多视角图教师蒸馏到纯内容 Student",
        "",
        "## 结论",
        "",
        f"- 运行状态：`{result['status']}`；最终周：`{result['final_week']}`。",
        f"- 表示层门槛：`{representation['passed']}`；端到端门槛：`{end_to_end['passed']}`；正式晋级：`{comparison['promotion_passed']}`。",
        f"- 决策：`{result['decision']}`。该英文值是本项目机器可读决策码，具体含义见下文。",
        "",
        "## 术语、统计对象与边界",
        "",
        "- **多视角图教师（multi-view graph teacher，项目自定义方案）**：训练期同时提供三组商品—商品关系的监督源，分别是 Item2Vec、直接共购和 DeepWalk；三组分数不直接相加，而是分别计算损失后等权平均。",
        "- **Item2Vec 教师（行业常用方法）**：把用户—日期去重商品序列当作词序列，以 Word2Vec 的 skip-gram（用中心商品预测邻近商品的训练目标）学习局部行为邻域；本轮复用 M4 的同一时点安全资产。",
        "- **直接共购教师（direct co-vis teacher，项目自定义口径）**：若两件商品被同一用户在同一自然日购买，则记一次用户日共现；分数为共现用户日数除以两件商品各自用户日支持度的几何均值。这里没有订单或曝光日志，不能称为真实购物篮或负反馈。",
        "- **DeepWalk 教师（行业通用图表示方法）**：在共购图上做均匀随机游走，再用 skip-gram 学习向量；它用于表达多跳图邻域，不在推理期输入 Student。",
        "- **纯内容 Student（content-only Student，项目自定义角色）**：和 M4 完全相同的商品编码器，只读取 FashionCLIP 图片向量和 11 个静态商品属性，输出 128 维归一化向量。strict-cold 商品推理时不需要购买历史或教师向量。",
        "- **关系损失（relation loss，本项目对度量学习监督的称呼）**：对一条“锚点商品、教师正邻居、困难负例”训练记录，要求 Student 让锚点更接近正邻居并远离负例。三路关系各自计算后等权平均。",
        "- **`teacher_cosine` 兼容字段（项目内部字段名）**：沿用 M4 资产接口保存教师关系强度；Item2Vec/DeepWalk 中是向量余弦，直接共购中实际保存归一化共现分数，并不冒充余弦。",
        "- **strict-cold Recall@200**：cutoff 前购买事件为 0 的未来真实用户—商品对，被每用户 Top200 候选覆盖的用户平均召回率；分母是各用户对应的 strict-cold truth。",
        "- **sparse Recall@200**：同上，但商品在 cutoff 前有 1–5 次购买事件。",
        "- **positive density@20（Top20 正例密度）**：排序后每个用户前 20 个候选中，属于未来 7 天真实购买的用户—商品行数 / 全部 Top20 候选行数。",
        "- **MRR（Mean Reciprocal Rank，平均倒数排名，行业通用指标）**：每个含 cold/sparse truth 用户的首个命中名次倒数，再以所有此类用户为分母平均；未命中记 0。",
        "- **Top200→Top20 conversion（项目自定义漏斗指标）**：Top200 中已存在的真实用户—商品对，有多少比例仍位于排序后的 Top20；分母不是全部未来 truth，而是当前候选池内的 truth。",
        "",
        "## 冻结实验配置",
        "",
        "- 教师时域：每个 cutoff 前 12 周；最终验证周未读取。",
        "- 直接共购：单用户日 2–20 件、共现至少 2 个用户日、每锚点 Top100。",
        "- DeepWalk：每节点 2 条随机游走、长度 20、64 维、窗口 10、3 轮、单 worker；没有超参数搜索。",
        "- 每个教师取 Top20 正关系。困难负例优先同 product type，再同 garment group，并排除任一教师对该锚点的 Top100。",
        "- 三个教师损失固定等权：每个优化 step 从每路各取 512 条关系，关系较少的教师按固定顺序循环；Student 架构、P3.1 Top200 候选预算、P3.2 网络和训练策略均未调整。",
        "",
        "## 单教师与多教师四窗结果",
        "",
        "下表每格为 `单教师 → 多教师（差值）`。单教师来自已冻结的 P3.1/P3.2；多教师经过同一 P3.1 + P3.2 流程。",
        "",
        "| window | strict Recall@200 | sparse Recall@200 | density@20 | MRR | Top200→Top20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window, row in comparison["windows"].items():
        cells = []
        for metric in ("strict_recall200", "sparse_recall200", "density20", "mrr", "top200_to_top20"):
            value = row[metric]
            cells.append(
                f"{_fmt(value['single_teacher'])} → {_fmt(value['multiview_teacher'])} ({value['delta']:+.6f})"
            )
        lines.append(f"| {WINDOW_LABELS[window]} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## 四窗均值",
        "",
        "| metric | 单教师 | 多教师 | 绝对差值 | 相对差值 |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric, row in comparison["means"].items():
        relative = "n/a" if row["relative_delta"] is None else f"{row['relative_delta']:+.2%}"
        lines.append(
            f"| {metric} | {_fmt(row['single_teacher'])} | {_fmt(row['multiview_teacher'])} | "
            f"{row['delta']:+.6f} | {relative} |"
        )
    lines.extend([
        "",
        "## 教师关系与互补性审计",
        "",
        "`positive_pairs` 的单位是一个 cutoff 下的一条锚点商品—正邻居关系；同一商品对跨 cutoff 出现时仍是独立时点观察。`direct graph nodes` 是该 cutoff 直接共购图中至少连接一条保留边的不同商品数。",
        "",
        "| cutoff | Item2Vec pairs | direct co-vis pairs | DeepWalk pairs | direct graph nodes |",
        "|---|---:|---:|---:|---:|",
    ])
    for cutoff, row in result["teachers"]["cutoffs"].items():
        relation = row["audits"]["relations"]["views"]
        lines.append(
            f"| {cutoff} | {relation['item2vec']['positive_pairs']:,} | "
            f"{relation['direct_covisit']['positive_pairs']:,} | {relation['deepwalk']['positive_pairs']:,} | "
            f"{row['audits']['deepwalk']['nodes']:,} |"
        )
    lines.extend([
        "",
        "下表是教师正关系的成对重合。`shared pairs` 是同一 cutoff 下两个教师都列入 Top20 的锚点—正邻居对数；百分比以表头中前一个教师的正关系数为分母。较低重合只证明监督关系不同，不自动等于对未来购买有用。",
        "",
        "| cutoff | Item2Vec∩direct shared pairs / Item2Vec | Item2Vec∩DeepWalk shared pairs / Item2Vec | direct∩DeepWalk shared pairs / direct |",
        "|---|---:|---:|---:|",
    ])
    for cutoff, row in result["teachers"]["cutoffs"].items():
        overlap = row["audits"]["relations"]["cross_view_positive_overlap"]
        i2v_direct = overlap["item2vec__direct_covisit"]
        i2v_deepwalk = overlap["item2vec__deepwalk"]
        direct_deepwalk = overlap["direct_covisit__deepwalk"]
        lines.append(
            f"| {cutoff} | {i2v_direct['shared_pairs']:,} / {i2v_direct['left_fraction']:.2%} | "
            f"{i2v_deepwalk['shared_pairs']:,} / {i2v_deepwalk['left_fraction']:.2%} | "
            f"{direct_deepwalk['shared_pairs']:,} / {direct_deepwalk['left_fraction']:.2%} |"
        )
    lines.extend([
        "",
        "## 晋级门槛与停止条件",
        "",
        "表示层要求 strict/sparse Recall@200 四窗均值都提高，且两者都至少 3/4 窗不退化。端到端要求 density@20、MRR、Top200→Top20 三项均值都提高，并且至少两项达到 3/4 窗不退化。两个门槛必须同时通过才晋级。",
        "",
    ])
    for name, passed in representation["checks"].items():
        lines.append(f"- representation `{name}`：{passed}")
    for name, passed in end_to_end["mean_checks"].items():
        lines.append(f"- end-to-end mean `{name}`：{passed}")
    for name, passed in end_to_end["window_checks"].items():
        lines.append(f"- end-to-end windows `{name}`：{passed}")
    lines.extend([
        "",
        "决策码解释：`promote_multiview_student_as_phase3_baseline` 表示两个门槛都通过；`retain_multiview_as_retrieval_only_evidence` 表示只通过表示层；`stop_multiview_teacher_after_representation_failure` 表示表示层门槛未通过，按预注册停止条件不得再调教师权重或 P3.2 排序器救场。",
        "",
        "## 解释边界",
        "",
        "- 本轮隔离了唯一创新：多教师关系监督。没有改变候选预算、排序器结构、负采样比例或最终融合。",
        "- optimistic all-articles catalog 仍缺少上架、库存和曝光信息；离线命中不能直接解释为线上可售新品转化。",
        "- 教师关系复现率只用于机制诊断；正式结论以真实未来购买的 Recall、密度、MRR 和漏斗转换为准。",
        f"- 总运行时间：{result['resources']['elapsed_seconds'] / 60:.1f} 分钟；设备：{result['resources']['device']}。",
    ])
    path = report_dir / "P3_3_FINAL.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
