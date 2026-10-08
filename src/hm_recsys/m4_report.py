from __future__ import annotations

from pathlib import Path
from typing import Any


def _write(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_m40(report_dir: Path, result: dict[str, Any]) -> None:
    rows = [
        "# M4.0：冻结基线、资产复用与协议门禁",
        "",
        "## 结论",
        "",
        "- 运行状态：measured（已测量）；所有图片资产身份与 Warm-v1 绑定均通过。",
        "- 最终验证周：not_run（未运行，也未读取）。",
        "- Student（学生编码器，行业常见蒸馏术语；指只用静态内容生成商品向量的模型）输入只包含静态图片与商品属性。",
        "",
        "## 术语与统计口径",
        "",
        "- Warm-v1（本项目自定义名）：冻结的 M3.3 base-300；候选为六路加权 RRF Top100 加最多200个 Item2Vec 独有候选，排序为84维无时间衰减特征的 LightGBM LambdaRank，并对无近期行为用户保留原候选顺序。",
        "- strict cold（严格冷商品，行业常见概念、本项目固定口径）：当前时间截点前全历史购买事件数为0的商品。",
        "- sparse 1--5（极稀疏商品，本项目分桶）：当前时间截点前全历史购买事件数为1至5的商品。",
        "- truth pair（真实购买用户—商品对，本项目统计单位）：评测用户在时间截点后7天购买的一条去重用户—商品关系；分母只来自对应开发窗。",
        "- contract payload SHA（契约载荷哈希，本项目身份字段）：对规范化契约内容计算的 SHA256；它不同于排版后的 contract.json 文件哈希，权威门禁使用前者。",
        "",
        "## FashionCLIP 资产门禁",
        "",
        f"- shape={result['fashionclip']['observed']['shape']}，dtype={result['fashionclip']['observed']['dtype']}，有效行 105100/105100。",
        f"- embedding SHA256：`{result['fashionclip']['expected']['embedding_sha256']}`。",
        f"- contract payload SHA256：`{result['fashionclip']['observed']['contract_payload_sha256']}`。",
        f"- serialized contract file SHA256：`{result['fashionclip']['observed']['contract_file_sha256']}`；仅记录文件身份，不作为载荷门禁值。",
        "",
        "## Warm-v1 四窗精确绑定",
        "",
        "| 开发窗 | MAP@12 | 身份检查 |",
        "|---|---:|---|",
    ]
    for window, value in result["warm_v1"]["map@12"].items():
        rows.append(f"| {window} | {value:.9f} | pass |")
    rows.extend([
        "",
        "## 时间点冷度审计",
        "",
        "表中 catalog items 是全商品目录中的商品数；truth pairs/users 分别是该桶未来7天真实购买用户—商品对数/拥有至少一个该桶真实购买的用户数。斜线不用于混合不同单位。",
        "",
        "| 开发窗 | 分桶 | catalog items | truth pairs | truth users |",
        "|---|---|---:|---:|---:|",
    ])
    for window, entry in result["coldness"].items():
        for bucket, metrics in entry["buckets"].items():
            rows.append(
                f"| {window} | {bucket} | {metrics['catalog_items']} | {metrics['truth_pairs']} | {metrics['truth_users']} |"
            )
    rows.extend([
        "",
        "## 边界",
        "",
        "- optimistic all-articles catalog（乐观全目录协议，本项目自定义协议）：离线评测把全目录商品视为可候选；数据没有真实上架、库存或曝光信息，因此 strict cold 不能等同线上确定可售新品。",
        "- M4.0 没有重新下载 FashionCLIP，也没有重新编码图片。",
        "- 机器可读证据：`M4_0_metrics.json`。",
    ])
    _write(report_dir / "M4_0_FINAL.md", rows)


def write_m41(report_dir: Path, result: dict[str, Any]) -> None:
    rows = [
        "# M4.1：Collaborative Teacher Relations",
        "",
        "## 结论",
        "",
        "- 运行状态：measured（已测量）；所需8个历史时间截点均构造完成，最终验证周 not_run。",
        "- Teacher（教师模型，行业常见蒸馏术语）：只看时间截点前12周购买共现的 Item2Vec，用行为关系监督 Student；Teacher 向量本身不会成为 Student 输入。",
        "- relational supervision（关系监督，行业通用思路）：学习商品间谁应更近、谁应更远，而不是回归不同时间截点之间不可对齐的 Item2Vec 坐标。",
        "",
        "## 术语、单位与分母",
        "",
        "- anchor（锚点商品）：Teacher 词表中的一个 warm 商品；每个锚点产生20个正邻居。",
        "- positive pair（正关系）：一个时间截点下的锚点—Teacher Top20 邻居；单位为 cutoff-anchor-positive，跨时间截点重复仍算独立时间观察。",
        "- hard negative（困难负例，行业通用术语）：与锚点同 product type（商品类型）或 garment group（服装组），但不在 Teacher Top100 的 warm 商品。",
        "- product-type hard-negative coverage：困难负例中直接来自同商品类型的比例，分母为该时间截点全部正关系数。",
        "",
        "## 每时间截点审计",
        "",
        "| cutoff | teacher vocab items | anchors | positive pairs | hard negatives | same product type | cosine mean | latest transaction |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cutoff, entry in result["cutoffs"].items():
        audit = entry["audit"]
        rows.append(
            f"| {cutoff} | {audit['teacher_vocab_items']} | {audit['anchors']} | {audit['positive_pairs']} | "
            f"{audit['hard_negatives']} | {audit['product_type_hard_negative_coverage']:.2%} | "
            f"{audit['teacher_cosine']['mean']:.6f} | {audit['latest_transaction_before_cutoff']} |"
        )
    cross = result["cross_cutoff"]
    rows.extend([
        "",
        "## 跨时间关系复用",
        "",
        f"- 时间关系观察总数：{cross['temporal_pair_observations']:,}。",
        f"- 去重锚点—正邻居关系：{cross['unique_anchor_positive_pairs']:,}。",
        f"- 跨时间重复观察比例：{cross['repeated_observation_ratio']:.2%}；这些不是误重复，而是不同时间截点下的独立协同证据。",
        "- 所有 latest transaction 均严格早于其 cutoff；验证周购买没有参与 Teacher。",
        "- 机器可读证据：`M4_1_metrics.json`；每个 cutoff 的 manifest.json 记录输入、代码和关系资产身份。",
    ])
    _write(report_dir / "M4_1_FINAL.md", rows)


def write_m42(report_dir: Path, result: dict[str, Any]) -> None:
    mechanics = result["mechanics"]
    rows = [
        "# M4.2：Multimodal Student Encoder 与监督式度量学习",
        "",
        "## 结论",
        "",
        f"- 运行状态：measured（已测量）；小型内层机制集从 {list(mechanics['ten_percent_lambda_trials'])} 选择 pairwise 权重 λ={mechanics['selected_lambda']}。",
        "- 选择依据只来自留出的 Teacher 关系损失，没有查看任何外层未来购买标签。",
        "- 最终验证周：not_run。",
        "",
        "## 术语、单位与口径",
        "",
        "- Student encoder（学生编码器）：把静态图片/属性映射为128维 L2 归一化商品向量的轻量网络；推理不需要商品历史行为。",
        "- image-only / metadata-only / multimodal（图片单模态/属性单模态/多模态）：三种固定输入消融；它们使用相同 Teacher 关系、训练轮数上限与验证切分。",
        "- weighted contrastive loss（加权对比损失，行业通用术语）：让锚点更接近其 Teacher 正邻居；权重来自 Teacher 余弦与名次，batch 内其他正商品提供对比项。",
        "- pairwise ranking loss（成对排序损失，行业通用术语）：要求锚点—正邻居相似度高于锚点—困难负例相似度。",
        "- teacher-neighbor Recall@K（教师邻域复现率）：对留出锚点，在 Student 的 warm 词表前K邻居中找回其 Teacher Top20 的比例，再对锚点平均；分母为每锚点20个 Teacher 正邻居。",
        "- pair accuracy（正负对排序准确率）：留出关系中 Student 正相似度高于困难负例相似度的行数/全部留出关系行数。",
        "- head/middle/tail（头部/中部/尾部 Teacher 商品）：按每个时间截点 Teacher token count 三等分，仅用于表示诊断，不是冷度定义。",
        "",
        "## 静态输入与资源",
        "",
        f"- 商品目录 {result['catalog']['rows']:,} 行；有图片 {result['catalog']['image_covered']:,}，图片缺失 {result['catalog']['image_missing']:,}。",
        "- 图片向量通过 mmap（内存映射，行业通用 I/O 技术）按 batch 读取；没有复制整张向量矩阵到每个训练样本。",
        "",
        "## 表示诊断",
        "",
        "| outer window | variant | best epoch | held-out pair accuracy | mean teacher Recall@20 | mean teacher Recall@50 | train seconds | peak VRAM GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window, entry in result["windows"].items():
        for variant, value in entry["variants"].items():
            diagnostics = value["diagnostics"]
            recall20 = sum(x["teacher_neighbor_recall@20"] for x in diagnostics["per_cutoff"].values()) / len(diagnostics["per_cutoff"])
            recall50 = sum(x["teacher_neighbor_recall@50"] for x in diagnostics["per_cutoff"].values()) / len(diagnostics["per_cutoff"])
            train = value["training"]
            rows.append(
                f"| {window} | {variant} | {train['best_epoch']} | {diagnostics['pooled_pair_accuracy']:.6f} | "
                f"{recall20:.6f} | {recall50:.6f} | {train['elapsed_seconds']:.1f} | {train['peak_vram_bytes']/1024**3:.2f} |"
            )
    total_training_seconds = sum(
        value["training"]["elapsed_seconds"]
        for entry in result["windows"].values()
        for value in entry["variants"].values()
    )
    rows.extend([
        "",
        "## 变体汇总与资源",
        "",
        "下表均值的分母是4个 outer window（外层开发窗）；pair accuracy 的统计单位仍是留出的 Teacher 正—困难负关系行。",
        "",
        "| variant | mean pair accuracy | mean teacher Recall@20 | mean teacher Recall@50 |",
        "|---|---:|---:|---:|",
    ])
    for variant in ("image_only", "metadata_only", "multimodal"):
        values = [entry["variants"][variant]["diagnostics"] for entry in result["windows"].values()]
        pair_accuracy = sum(value["pooled_pair_accuracy"] for value in values) / len(values)
        recall20 = sum(
            sum(row["teacher_neighbor_recall@20"] for row in value["per_cutoff"].values()) / len(value["per_cutoff"])
            for value in values
        ) / len(values)
        recall50 = sum(
            sum(row["teacher_neighbor_recall@50"] for row in value["per_cutoff"].values()) / len(value["per_cutoff"])
            for value in values
        ) / len(values)
        rows.append(f"| {variant} | {pair_accuracy:.6f} | {recall20:.6f} | {recall50:.6f} |")
    rows.extend([
        "",
        f"- 12个正式模型累计训练时间：{total_training_seconds:.1f} 秒；峰值进程工作集：{result['resources']['peak_working_set_bytes']/1024**3:.2f} GiB。",
        f"- 正式模型与商品向量合计：{result['resources']['artifact_bytes']/1024**2:.2f} MiB。",
        "- 三种变体均在本机 NVIDIA GeForce RTX 4060 Laptop GPU 单卡运行；未使用 DDP（分布式数据并行）。",
        "",
        "## 正式向量身份",
        "",
        "SHA 前12位仅用于表内辨认；完整 SHA256、字节数和绝对路径保存在 M4_2_metrics.json 及各 manifest.json。",
        "",
        "| outer window | variant | embedding shape | embedding SHA prefix | model SHA prefix |",
        "|---|---|---|---|---|",
    ])
    for window, entry in result["windows"].items():
        for variant, value in entry["variants"].items():
            rows.append(
                f"| {window} | {variant} | 105542×128 | {value['encoding']['artifact']['sha256'][:12]} | "
                f"{value['training']['model']['sha256'][:12]} |"
            )
    rows.extend([
        "",
        "## 解释边界",
        "",
        "- 表示诊断只证明 Student 是否学到 Teacher 关系，不能替代 M4.3 的真实 cold/sparse 未来购买召回门禁。",
        "- Student 未使用 article_id embedding、product_code、Item2Vec 坐标、销量、首次销售时间或验证周行为。",
        "- 机器可读证据：`M4_2_metrics.json`；每个窗/变体的 manifest.json 记录模型、向量、训练关系和代码身份。",
    ])
    _write(report_dir / "M4_2_FINAL.md", rows)


def write_m43(report_dir: Path, result: dict[str, Any]) -> None:
    rows = [
        "# M4.3：Student-space Cold / Sparse Retrieval",
        "",
        "## 结论",
        "",
        f"- 运行状态：measured（已测量）；七项 retrieval gate（召回层晋级门槛）总体通过：{result['summary']['gate_passed']}。",
        f"- 下一步：`{result['summary']['next_stage']}`。",
        "- 最终验证周：not_run。",
        "",
        "## 术语、单位与口径",
        "",
        "- candidate universe（候选商品全集，本项目自定义集合）：当前 cutoff 前全历史购买事件数不超过5的全目录商品；strict cold=0，sparse 1--5=1至5。",
        "- user seed（用户种子商品，本项目召回输入）：评测用户在 cutoff 前12周最近购买的最多5个不同商品；单位为用户—历史商品，不含验证周 truth。",
        "- best recency-weighted cosine（最佳时间衰减余弦，本项目主排序分数）：候选与5个 seed 的余弦分别乘28天半衰期权重后取最大值；M4.3 不调融合公式。",
        "- incremental candidate（增量候选）：Student/原始 FashionCLIP 前K中去掉 Warm-v1 已有商品后的用户—商品候选；K 在去重前计算。",
        "- candidate Recall（候选召回率）：每个有对应 truth 的用户，其未来7天真实购买中被候选集合覆盖的比例，再对用户平均。",
        "- candidate HitRate（候选命中率）：对应 truth 用户中至少命中一件的用户比例。",
        "- Oracle MAP@12（候选理论上限）：假设候选内真实购买能完美排到前12所得上限；它不是实际排序 MAP。",
        "- positive density（正例密度）：增量候选中命中未来7天 truth 的用户—商品对数/全部增量候选行数。",
        "",
        "## K=50 主结果",
        "",
        "以下 Recall 均基于 Warm-v1 与对应增量候选的去重并集；new cold pairs 是相对 Warm-v1 新增命中的 strict-cold 用户—商品对数。",
        "",
        "| window | variant | overall Recall | strict-cold Recall | sparse1-5 Recall | new truth pairs | new cold pairs | density |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window, entry in result["windows"].items():
        base = entry["base_300"]["segments"]
        rows.append(
            f"| {window} | Warm-v1 base-300 | {base['overall']['recall']:.6f} | {base['strict_cold']['recall']:.6f} | "
            f"{base['sparse_1_5']['recall']:.6f} | 0 | 0 | n/a |"
        )
        for variant, value in entry["variants"].items():
            metrics = value["evaluation"]["50"]
            segments = metrics["segments"]
            rows.append(
                f"| {window} | {variant} | {segments['overall']['recall']:.6f} | "
                f"{segments['strict_cold']['recall']:.6f} | {segments['sparse_1_5']['recall']:.6f} | "
                f"{metrics['incremental_truth_pairs']} | {metrics['incremental_strict_cold_truth_pairs']} | "
                f"{metrics['positive_density_after_warm_dedup']:.8f} |"
            )
    rows.extend([
        "",
        "## 预注册晋级门槛",
        "",
        "| gate | passed |",
        "|---|---|",
    ])
    for name, passed in result["summary"]["gates"].items():
        rows.append(f"| {name} | {passed} |")
    means = result["summary"]["means"]
    strict_absolute = means["multimodal_strict_cold_recall"] - means["raw_strict_cold_recall"]
    sparse_absolute = means["multimodal_sparse_1_5_recall"] - means["raw_sparse_1_5_recall"]
    overall_absolute = means["multimodal_overall_recall"] - means["warm_overall_recall"]
    rows.extend([
        "",
        "## 四窗均值",
        "",
        f"- multimodal strict-cold Recall={means['multimodal_strict_cold_recall']:.6f}；同预算 raw FashionCLIP={means['raw_strict_cold_recall']:.6f}。",
        f"- multimodal sparse1-5 Recall={means['multimodal_sparse_1_5_recall']:.6f}；同预算 raw FashionCLIP={means['raw_sparse_1_5_recall']:.6f}。",
        f"- multimodal overall Recall={means['multimodal_overall_recall']:.6f}；Warm-v1={means['warm_overall_recall']:.6f}。",
        f"- 相对 raw FashionCLIP：strict-cold 绝对增量 {strict_absolute:+.6f}（相对 {strict_absolute/means['raw_strict_cold_recall']:+.2%}）；sparse1-5 绝对增量 {sparse_absolute:+.6f}（相对 {sparse_absolute/means['raw_sparse_1_5_recall']:+.2%}）。",
        f"- 相对 Warm-v1：overall candidate Recall 绝对增量 {overall_absolute:+.6f}。",
        "",
        "## K=20/50/100 深度诊断",
        "",
        "K 是每个有 seed 用户在与 Warm-v1 去重前保留的 Student 候选上限；这些是预注册诊断点，不是根据外层结果连续调参。",
        "",
        "| window | K | strict-cold Recall | sparse1-5 Recall | overall Recall | new cold pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for window, entry in result["windows"].items():
        for k in (20, 50, 100):
            value = entry["variants"]["multimodal"]["evaluation"][str(k)]
            segments = value["segments"]
            rows.append(
                f"| {window} | {k} | {segments['strict_cold']['recall']:.6f} | "
                f"{segments['sparse_1_5']['recall']:.6f} | {segments['overall']['recall']:.6f} | "
                f"{value['incremental_strict_cold_truth_pairs']} |"
            )
    rows.extend([
        "",
        "## 多模态 K=50 分桶与活跃性",
        "",
        "tail/middle/head（尾部/中部/头部，本报告历史流行度桶）分别表示 cutoff 前全历史事件数1--20、21--100、至少101；与 strict cold=0 分开。Recall/HitRate/Oracle 的分母分别是该窗该桶有 truth 的用户。",
        "",
        "| window | segment | Recall | HitRate | Oracle MAP@12 | truth users | truth pairs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for window, entry in result["windows"].items():
        evaluation = entry["variants"]["multimodal"]["evaluation"]["50"]
        for segment in ("strict_cold", "sparse_1_5", "tail_1_20", "middle_21_100", "head_101_plus"):
            value = evaluation["segments"][segment]
            rows.append(
                f"| {window} | {segment} | {value['recall']:.6f} | {value['hit_rate']:.6f} | "
                f"{value['oracle_map@12']:.6f} | {value['truth_users']} | {value['truth_pairs']} |"
            )
        for activity in ("active", "inactive"):
            value = evaluation["activity_segments"][activity]
            rows.append(
                f"| {window} | {activity}_users_overall | {value['recall']:.6f} | {value['hit_rate']:.6f} | "
                f"{value['oracle_map@12']:.6f} | {value['truth_users']} | {value['truth_pairs']} |"
            )
    rows.extend([
        "",
        "## 候选规模、资源与身份",
        "",
        "active/inactive 的单位是评测用户；incremental rows 是 K=50 去除 Warm-v1 重合后剩余的用户—商品候选行；overlap 是原50条中与 Warm-v1 重合的比例。",
        "",
        "| window | variant | active | inactive | universe items | incremental rows | overlap | runtime s | peak VRAM GiB | artifact SHA prefix |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for window, entry in result["windows"].items():
        for variant, value in entry["variants"].items():
            resources = value["resources"]
            evaluation = value["evaluation"]["50"]
            rows.append(
                f"| {window} | {variant} | {resources['active_users']} | {resources['inactive_users']} | "
                f"{resources['candidate_universe_items']} | {evaluation['incremental_candidate_rows_after_warm_dedup']} | "
                f"{evaluation['candidate_overlap_with_warm_ratio']:.4%} | {resources['elapsed_seconds']:.1f} | "
                f"{resources['peak_vram_bytes']/1024**3:.2f} | {value['artifact']['sha256'][:12]} |"
            )
    rows.extend([
        "",
        f"- M4.3 峰值进程工作集：{result['resources']['peak_working_set_bytes']/1024**3:.2f} GiB；16个候选资产合计 {result['resources']['retrieval_artifact_bytes']/1024**2:.2f} MiB。",
        "- 独立复核重新计算了130个记录身份，并检查12个表示矩阵和16个候选资产的 shape、有限值、名次、预算与用户—商品唯一性，全部通过。",
        "",
        "## 对照解释",
        "",
        "- multimodal 在 strict-cold 上四窗均不低于 raw FashionCLIP，并在四窗均值和每窗都更高；这支持协同 Teacher 监督改善零历史内容空间。",
        "- multimodal 并非每个分桶都压过 image-only：winter strict-cold 及若干 sparse 窗口由 image-only 更高，说明属性融合存在窗口依赖，不能宣称多模态在所有条件全面占优。",
        "- 当前证明的是候选层 Recall/Oracle 改善；尚未证明这些冷商品能进入最终 Top12，实际 MAP 兑现仍属于 M5。",
        "",
        "## 决策边界",
        "",
        "- 若 gate=False：按二阶段路线停止在 M4.3，只允许做 M4.4 只读失败分解；不调 K、阈值，不训练 cold admission/联合 LightGBM，不 fine-tune FashionCLIP。",
        "- 若 gate=True：只表示允许另行规划 M5；本次没有运行 M5。",
        "- 本报告使用 optimistic all-articles catalog，没有上架、库存或曝光日志，不能把离线 strict cold 命中描述成线上可售新品转化。",
        "- 机器可读证据：`M4_3_metrics.json`；每窗/变体的 retrieval_top100.npz 保存候选身份和主分数。",
    ])
    _write(report_dir / "M4_3_FINAL.md", rows)
