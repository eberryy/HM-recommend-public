from __future__ import annotations

from pathlib import Path
from typing import Any


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _f(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_p30(report_dir: Path, result: dict[str, Any]) -> None:
    checks = "\n".join(
        f"| `{name}` | {'通过' if passed else '失败'} |"
        for name, passed in result["checks"].items()
    )
    students = "\n".join(
        f"| {row['cutoff']} | `{row['role']}` | `{row['source_kind']}` | {row['embedding']['bytes'] / 2**20:.2f} MiB | `{row['embedding']['sha256'][:12]}` |"
        for row in result["student_assets"].values()
    )
    _write(
        report_dir / "P3_0_FINAL.md",
        f"""# P3.0：冻结资产与复用门禁

## 结论

- P3.0 复用门禁：**{'通过' if result['gate_passed'] else '失败'}**。
- 本轮冻结 `T_cold=5`、用户历史最多20件不同商品、粗召回预算200；没有重训 M4 Student。
- 最终验证周 `2020-09-16`：`not_run`（未运行，也未用于决策）。

## 术语与口径

- **Student（学生表示模型，知识蒸馏行业通用术语）**：M4 用商品图片与静态属性预测协同购买结构的内容模型；本轮复用其128维、L2归一化商品向量。
- **point-in-time（时间点安全，行业通用术语）**：某个截止日的特征与表示只使用该日之前可见的数据。本轮八个训练截止日使用同截止日生成的 Student 资产，外层四窗使用各自已验收的 M4 Student。
- **复用门禁（本项目自定义）**：对文件字节数、SHA256、形状、数据类型、截止日和最终周状态进行失败即停止的身份检查。
- **SHA256（行业通用文件哈希）**：文件内容身份摘要；前12位只用于报告展示，机器可读证据保存完整值。

## 门禁检查

| check | result |
|---|---|
{checks}

## Student 表示资产

| cutoff | 资产角色 | 来源 | 大小 | SHA256前12位 |
|---|---|---|---:|---|
{students}

FashionCLIP 静态资产、M4 静态目录映射和12个截止日 Student 表示均通过身份校验。训练截止日没有反向使用外层未来表示，这是本轮为满足时间点安全采用的特殊处理。
""",
    )


def write_p31(report_dir: Path, result: dict[str, Any]) -> None:
    rows = []
    concentration = []
    resources = []
    for window, row in result["windows"].items():
        metrics = row["coarse_metrics"]
        at20 = metrics["at_k"]["20"]
        at50 = metrics["at_k"]["50"]
        at100 = metrics["at_k"]["100"]
        at200 = metrics["at_k"]["200"]
        rows.append(
            f"| {window} | {_f(at20['segments']['strict_cold']['recall'])} | {_f(at50['segments']['strict_cold']['recall'])} | {_f(at100['segments']['strict_cold']['recall'])} | {_f(at200['segments']['strict_cold']['recall'])} | {_f(at20['segments']['sparse_1_5']['recall'])} | {_f(at50['segments']['sparse_1_5']['recall'])} | {_f(at100['segments']['sparse_1_5']['recall'])} | {_f(at200['segments']['sparse_1_5']['recall'])} |"
        )
        concentration.append(
            f"| {window} | {_f(at20['segments']['cold_universe']['hit_rate'])} | {_f(at50['segments']['cold_universe']['hit_rate'])} | {_f(at100['segments']['cold_universe']['hit_rate'])} | {_f(at200['segments']['cold_universe']['hit_rate'])} | {_f(metrics['ranking']['mrr'])} | {_f(metrics['ranking']['first_positive_rank_mean_hit_users'], 2)} | {_f(metrics['ranking']['first_positive_rank_median_hit_users'], 2)} |"
        )
        density = " | ".join(
            _f(metrics["at_k"][str(k)]["positive_density"], 8) for k in (20, 50, 100, 200)
        )
        incremental = row["incremental_efficiency"]
        resources.append(
            f"| {window} | {density} | {_f(at200['truth_pairs_per_1k_candidates'], 4)} | {incremental['incremental_truth_pairs']} / {incremental['incremental_candidate_rows']} | {_f(incremental['incremental_truth_pairs_per_1k_incremental_candidates'], 4)} | {row['retrieval']['active_users']} / {row['retrieval']['inactive_users']} | {row['retrieval']['elapsed_seconds']:.1f}s |"
        )
    _write(
        report_dir / "P3_1_FINAL.md",
        f"""# P3.1：Version A 粗召回 Top200

## 结论

P3.1 已建立可审计的固定 Top200 候选资产。它只作为 P3.2 的输入基线，不进行效果晋级，也不替代 M4 已有部署候选。

## 术语与统计口径

- **粗召回（行业通用术语）**：先用低成本相似度从目录中为每个用户选出最多200件候选，后续模型只在这200件内重排。
- **cold/sparse universe（冷/稀疏候选目录，本项目自定义）**：截止日前全站购买事件数不超过5的目录商品；其中 strict-cold 为0次，sparse1-5 为1至5次。
- **用户历史**：截止日前最近20件不同的已购商品；不足20件时使用全部。排序使用 Student 余弦相似度乘28天半衰期时间权重后的最大值。
- **Recall@K（行业通用召回率）**：对该分群每个有真实购买的用户，计算前K覆盖的真实商品比例，再对用户平均；未命中用户计0。
- **HitRate@K（行业通用命中率）**：该分群真实用户中，前K至少命中一件的用户比例。本报告表中分群为购买了冷/稀疏目录商品的用户。
- **MRR（Mean Reciprocal Rank，行业通用平均倒数排名）**：每个冷/稀疏真实用户的首个命中排名取倒数后平均；Top200 未命中用户贡献0。
- **positive density@K（前K正例密度，本项目核心指标）**：所有活跃用户前K候选行中，未来7天实际购买行数除以候选总行数；零正例用户仍在分母中。
- **truth pairs per 1k candidates（每千候选真实对，本项目展示单位）**：正例密度乘1000。
- **增量效率**：从 Cold Top200 去掉同用户冻结 Warm-v1 Top300 已有商品后，新增真实购买对数除以新增候选行数；`a / b` 分别为真实对/候选行。
- **active/inactive（本项目自定义）**：截止日前是否至少有一件可映射的历史商品；无历史用户不生成 Cold 候选，但仍保留在外层评价用户分母中。

## 分群 Recall

| window | strict R@20 | R@50 | R@100 | R@200 | sparse R@20 | R@50 | R@100 | R@200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 命中与排名集中度

| window | Hit@20 | Hit@50 | Hit@100 | Hit@200 | MRR | 首命中均值* | 首命中中位数* |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(concentration)}

`*` 首命中排名均值/中位数只以 Top200 内至少命中的用户为分母；MRR 则包含 Top200 未命中用户。

## 密度、增量效率与资源

| window | density@20 | @50 | @100 | @200 | Top200每千候选真实对 | Warm增量真实对/候选 | 增量每千候选真实对 | active/inactive用户 | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(resources)}

训练侧只对“未来一周至少存在一件当时属于冷/稀疏目录的真实商品”的全部用户计算候选。这是一项等价计算裁剪：其他训练用户不可能形成候选正例组。外层四窗没有采用该裁剪，保留完整用户与零正例组。

所有候选均通过用户—商品唯一、活动用户预算200、组内名次1至200连续、截止日安全及 Student SHA 身份检查。最终周：`not_run`。
""",
    )


def write_p32(report_dir: Path, result: dict[str, Any]) -> None:
    main_rows = []
    density_rows = []
    recall_hit_rows = []
    rank_rows = []
    segment_rows = []
    separation_rows = []
    attention_rows = []
    scale_rows = []
    for window, row in result["windows"].items():
        coarse = row["metrics"]["coarse_student_order"]
        candidate = row["metrics"]["candidate_aware_order"]
        main_rows.append(
            f"| {window} | {_f(coarse['at_k']['20']['positive_density'], 8)} | {_f(candidate['at_k']['20']['positive_density'], 8)} | {_f(coarse['at_k']['50']['positive_density'], 8)} | {_f(candidate['at_k']['50']['positive_density'], 8)} | {_f(coarse['ranking']['mrr'])} | {_f(candidate['ranking']['mrr'])} | {_f(coarse['ranking']['conversion']['top200_to_top20'])} | {_f(candidate['ranking']['conversion']['top200_to_top20'])} |"
        )
        for variant_name, variant_label in (
            ("coarse_student_order", "coarse"), ("candidate_aware_order", "candidate-aware")
        ):
            metrics = row["metrics"][variant_name]
            density_rows.append(
                f"| {window} | {variant_label} | "
                + " | ".join(_f(metrics['at_k'][str(k)]['positive_density'], 8) for k in (5, 10, 20, 50, 100, 200))
                + " |"
            )
            recall_hit_rows.append(
                f"| {window} | {variant_label} | "
                + " | ".join(
                    _f(metrics['at_k'][str(k)]['segments'][segment]['recall'])
                    for segment in ('strict_cold', 'sparse_1_5') for k in (5, 10, 20, 50)
                )
                + " | "
                + " | ".join(
                    _f(metrics['at_k'][str(k)]['segments']['cold_universe']['hit_rate'])
                    for k in (5, 10, 20, 50)
                )
                + " |"
            )
            ranking = metrics["ranking"]
            rank_rows.append(
                f"| {window} | {variant_label} | {_f(ranking['first_positive_rank_mean_hit_users'], 2)} | {_f(ranking['first_positive_rank_median_hit_users'], 2)} | {ranking['positive_rank_best']} | {_f(ranking['positive_rank_p25'], 2)} | {_f(ranking['positive_rank_p50'], 2)} | {_f(ranking['positive_rank_p75'], 2)} | "
                + " | ".join(_f(ranking['conversion'][f'top200_to_top{k}']) for k in (5, 10, 20, 50))
                + " |"
            )
        segment_rows.append(
            f"| {window} | {_f(coarse['at_k']['20']['segments']['strict_cold']['recall'])} | {_f(candidate['at_k']['20']['segments']['strict_cold']['recall'])} | {_f(coarse['at_k']['20']['segments']['sparse_1_5']['recall'])} | {_f(candidate['at_k']['20']['segments']['sparse_1_5']['recall'])} | {candidate['at_k']['200']['positive_rows']} |"
        )
        separation = row["score_separation"]
        separation_rows.append(
            f"| {window} | {_f(separation['positive_vs_unobserved_auc'])} | {_f(separation['same_user_positive_percentile']['median'])} | {_f(separation['positive_scores']['median'])} | {_f(separation['unobserved_scores']['median'])} |"
        )
        attention = row["attention"]
        attention_rows.append(
            f"| {window} | {_f(attention['all_candidates']['entropy_mean'])} | {_f(attention['all_candidates']['top1_weight_mean'])} | {_f(attention['all_candidates']['top3_weight_sum_mean'])} | {_f(attention['candidate_attention_l1_difference']['mean'])} | {_f(attention['truth_candidates']['entropy_mean'])} | {_f(attention['unobserved_candidates']['entropy_mean'])} |"
        )
        pair_rows = sum(value["pair_rows"] for value in row["training_pair_audits"].values())
        positives = sum(value["positive_candidate_rows"] for value in row["training_pair_audits"].values())
        groups = sum(value["positive_groups"] for value in row["training_pair_audits"].values())
        scale_rows.append(
            f"| {window} | {positives} | {groups} | {pair_rows} | {row['inner_training']['selected_epoch']} | {row['outer_training']['elapsed_seconds']:.1f}s | {row['scoring']['elapsed_seconds']:.1f}s |"
        )
    gate = result["mechanism_gate"]
    gate_rows = "\n".join(
        f"| `{name}` | {'通过' if passed else '失败'} |"
        for name, passed in gate["checks"].items()
    )
    _write(
        report_dir / "P3_2_FINAL.md",
        f"""# P3.2：候选感知的用户历史重排器

## 结论

- 机制门禁：**{'通过' if gate['gate_passed'] else '失败'}**。
- 决策状态：`{gate['decision']}`。
- 本轮只训练一个固定结构的候选感知重排器；没有搜索网络宽度、历史长度、候选预算、损失或负采样比例。
- 最终验证周 `2020-09-16`：`not_run`。

## 模型与训练口径

- **candidate-aware（候选感知，本项目核心结构）**：面对每一件候选商品，模型重新判断用户最近20件历史商品各自的重要性，因此同一用户对不同候选可以得到不同的用户表示。
- **attention（注意力，行业通用术语）**：候选与每件历史商品先形成 `[候选向量、历史向量、逐维乘积、绝对差、log1p距今天数]`，小型多层感知机输出权重；有效历史上的权重和为1，填充位置权重为0。
- **same-user pairwise logistic/BPR（同用户成对逻辑损失，行业通用排序目标）**：要求同一用户未来购买的候选分数高于未观察候选。未观察不代表用户曝光后拒绝。
- **hard/medium/easy negatives（难/中/易未观察样本，本项目自定义）**：按粗召回名次1–67、68–134、135–200分桶；每个正例最多配50个同用户样本，固定哈希抽样且三桶均有覆盖。
- **full-user formal（全用户正式口径，本项目自定义）**：训练使用全部可能形成冷候选正例的用户，不做1%/10%哈希抽样；外层验证使用全部冻结评价用户，零正例组也保留。1%与10%只验证机械正确性。
- **Top200 truth→Top20 conversion（本项目排名集中率）**：Top200 内全部真实用户—商品对中，被重排到前20的比例；分母固定为同一候选集合内的正例数。
- **positive density@K（前K正例密度，本项目核心指标）**：全部活动用户前K候选中未来7天真实购买行数除以候选行数；完整外层零正例用户仍在分母中。
- **Recall@K（行业通用召回率）**：对指定冷度分群的每位真实用户计算前K覆盖比例后取均值；未命中用户计0。
- **HitRate@K（行业通用命中率）**：存在冷/稀疏真实购买的用户中，前K至少命中一件的用户比例。
- **MRR（Mean Reciprocal Rank，行业通用平均倒数排名）**：每位冷/稀疏真实用户的首个命中名次取倒数后平均；Top200未命中用户贡献0。
- **AUC（行业通用可分性诊断）**：候选感知分数区分未来购买与未观察候选的概率，只作机制诊断，不替代 Recall/MRR。

## 核心密度、MRR 与正例集中率

| window | coarse density@20 | rerank density@20 | coarse density@50 | rerank density@50 | coarse MRR | rerank MRR | coarse 200→20 | rerank 200→20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_rows)}

## 完整前K正例密度

| window | ordering | density@5 | @10 | @20 | @50 | @100 | @200 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(density_rows)}

每项乘1000即 `truth_pairs_per_1k_candidates`（每千候选真实用户—商品对数）；完整数值也保存在 `candidate_density_audit.json`。

## 完整分群 Recall 与 HitRate

| window | ordering | strict R@5 | R@10 | R@20 | R@50 | sparse R@5 | R@10 | R@20 | R@50 | cold-universe Hit@5 | @10 | @20 | @50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(recall_hit_rows)}

`cold-universe`（本项目自定义分群）是 strict-cold 与 sparse1-5 的并集，即截止日前购买事件数不超过5的真实商品。

## 分群 Recall@20 与 Top200 守恒

| window | coarse strict R@20 | rerank strict R@20 | coarse sparse R@20 | rerank sparse R@20 | Top200正例对（两排序相同） |
|---|---:|---:|---:|---:|---:|
{chr(10).join(segment_rows)}

同一窗口两种排序逐行复用完全相同的 Top200 用户—商品集合；Recall@200、正例对数和候选身份精确守恒。

## 完整排名漏斗

| window | ordering | 首命中均值* | 首命中中位数* | 正例最佳名次 | 正例名次p25 | p50 | p75 | 200→5 | 200→10 | 200→20 | 200→50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rank_rows)}

`*` 首命中均值/中位数只以 Top200 已命中用户为分母；MRR 包含未命中用户。正例名次分位数的统计单位是 Top200 内所有真实用户—商品对。

## 分数可分性

| window | 正例vs未观察 AUC | 同用户正例百分位中位数 | 正例分数中位数 | 未观察分数中位数 |
|---|---:|---:|---:|---:|
{chr(10).join(separation_rows)}

“同用户正例百分位”以同一用户的未观察候选为参照，1表示正例高于全部未观察候选，0.5表示位于中间。
正例与未观察候选各自的均值、中位数、p90和p95完整保存在 `score_separation_audit.json`。

## Attention 机制审计

| window | 熵均值 | Top1权重均值 | Top3权重和均值 | 同用户不同候选注意力L1差均值 | 正例熵 | 未观察熵 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(attention_rows)}

注意力 L1 差以每个活动用户的粗排第1与第200候选为一对，比较两条有效历史权重向量的绝对差之和；它用于检查模型是否退化成固定用户向量，不参与晋级门禁。匿名示例及完整分布见 `attention_audit.json`。

## 正式监督规模与耗时

| window | Top200内训练正例 | 正例组 | 成对训练行 | inner选定epoch | outer训练 | 全量打分 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(scale_rows)}

## 机制门禁

| gate | result |
|---|---|
{gate_rows}

四窗均值：density@20 `{_f(gate['means']['coarse_density20'], 8)} → {_f(gate['means']['candidate_aware_density20'], 8)}`，相对变化 `{_f(gate['means']['density20_relative_improvement'], 4)}`；MRR `{_f(gate['means']['coarse_mrr'])} → {_f(gate['means']['candidate_aware_mrr'])}`；Top200→Top20 `{_f(gate['means']['coarse_top200_to_top20_conversion'])} → {_f(gate['means']['candidate_aware_top200_to_top20_conversion'])}`。

门禁结论只决定是否允许后续单独规划 P3.3 或最终融合。本轮没有启动任何后续阶段，也没有读取最终周。
""",
    )
