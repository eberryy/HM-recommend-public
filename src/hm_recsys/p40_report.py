from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .m4_contract import atomic_json, file_identity
from .p40_contract import RUN_ID


def render(result: dict[str, Any]) -> str:
    selected = result["decision"]["selected_variant"]
    variants = ("W0", "W1", "F0", "F1")
    window_rows = list(result["windows"].values())
    mean_summary: dict[str, dict[str, float | int]] = {}
    w0_values = [row["evaluations"]["W0"]["segments"]["overall"]["map@12"] for row in window_rows]
    w1_values = [row["evaluations"]["W1"]["segments"]["overall"]["map@12"] for row in window_rows]
    for variant in variants:
        values = [row["evaluations"][variant]["segments"]["overall"]["map@12"] for row in window_rows]
        deltas_w0 = [value - base for value, base in zip(values, w0_values)]
        deltas_w1 = [value - base for value, base in zip(values, w1_values)]
        mean_summary[variant] = {
            "mean": sum(values) / len(values),
            "mean_delta_w0": sum(deltas_w0) / len(deltas_w0),
            "mean_delta_w1": sum(deltas_w1) / len(deltas_w1),
            "non_degrade_w0": sum(value >= 0 for value in deltas_w0),
            "worst_delta_w0": min(deltas_w0),
        }
    lines = [
        "# P4.0：B0 Cold Expert + Warm-v1 跨来源融合 v2",
        "",
        "## 结论",
        "",
        f"- 机器结论：`{result['decision']['machine_decision']}`；最终选择 `{selected}`；P4.0 晋级：{result['decision']['P4_0_promoted']}。",
        "- final week（最终周 `2020-09-16`）：`not_run`。本报告只使用四个冻结开发窗口。",
        "- P4.1 仍为 deferred（延后），没有自动启动补救实验。",
        "",
        "## 术语与统计口径",
        "",
        "- **W0**：冻结 Warm-v1 Top12，不训练；候选上界按这12个最终推荐商品计算。",
        "- **W1**：只保留 Warm Top150、使用与融合方案相同时间链重新训练的控制组，用于隔离截断和重训影响。",
        "- **F0**：Warm150 与 B0 Cold50 去重并集，直接复用 M5.5 历史来源/名次分层负采样。",
        "- **F1**：候选、特征、模型参数均与 F0 相同，只把训练采样改为跨来源边界感知采样；它让 Cold 正例与 Warm 候选边界直接比较，也让 Warm 正例与 Cold 头部候选直接比较。",
        "- **cold-only truth**：某截止日—用户—真实购买商品对只出现在 Cold50、不在 Warm150；漏斗数字的单位都是用户—商品正例对。",
        "- **candidate Recall**：逐用户计算候选集合覆盖的未来一周真实购买比例后，在完整 truth 用户分母上取平均；**Oracle MAP@12** 是假设候选能被理想排序时的 MAP@12 上界。",
        "- **strict-cold / sparse1-5 / warm_21_plus**：商品在截止日前全历史交易事件数分别为0、1至5、至少21；分群 MAP 的分母是拥有该分群 truth 的全部用户，而不是命中用户。",
        "- **boundary-aware sampling**：本项目自定义的跨来源边界感知采样；未购买候选只是未观察候选，因为数据没有曝光或库存日志。",
        "- **feature gain**：LightGBM 使用某特征分裂时累计的目标增益；它只能说明模型使用情况，不代表该特征的因果贡献。",
        "",
        "## 四窗 MAP@12",
        "",
        "| window | W0 | W1 | F0 | F1 | F1-W0 | F1-F0 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        values = {name: row["evaluations"][name]["segments"]["overall"]["map@12"] for name in variants}
        lines.append(
            f"| {window} | {values['W0']:.8f} | {values['W1']:.8f} | {values['F0']:.8f} | "
            f"{values['F1']:.8f} | {values['F1']-values['W0']:+.8f} | {values['F1']-values['F0']:+.8f} |"
        )
    lines.extend([
        "", "### 四窗汇总", "",
        "`non-degrade vs W0` 表示相对冻结基线不退化的窗口数，分母固定为4；`worst delta` 是四窗中最差的 MAP@12 差值。",
        "", "| variant | mean MAP@12 | mean delta vs W0 | mean delta vs W1 | non-degrade vs W0 | worst delta vs W0 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in variants:
        row = mean_summary[variant]
        lines.append(
            f"| {variant} | {row['mean']:.8f} | {row['mean_delta_w0']:+.8f} | "
            f"{row['mean_delta_w1']:+.8f} | {row['non_degrade_w0']}/4 | {row['worst_delta_w0']:+.8f} |"
        )
    lines.extend([
        "", "## 分群 MAP@12", "",
        "| window | variant | warm_21_plus | warm truth users | strict-cold | strict truth users | sparse1-5 | sparse truth users |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("W0", "W1", "F0", "F1"):
            seg = row["evaluations"][variant]["segments"]
            lines.append(
                f"| {window} | {variant} | {seg['warm_21_plus']['map@12']:.8f} | {seg['warm_21_plus']['truth_users']} | "
                f"{seg['strict_cold']['map@12']:.8f} | {seg['strict_cold']['truth_users']} | "
                f"{seg['sparse_1_5']['map@12']:.8f} | {seg['sparse_1_5']['truth_users']} |"
            )
    lines.extend([
        "", "## 候选上界", "",
        "| window | variant | candidate Recall | Oracle MAP@12 |",
        "|---|---|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("W0", "W1", "F0", "F1"):
            ev = row["evaluations"][variant]
            lines.append(f"| {window} | {variant} | {ev['candidate_recall']:.8f} | {ev['oracle_map@12']:.8f} |")
    lines.extend([
        "", "## Cold-only truth 漏斗", "",
        "下表展示 F0/F1 的 cold-only truth；`union` 是进入候选并集的用户—商品正例对数，TopK 是统一排序后名次不超过K的正例对数。W0 不含 Cold50，因此其 cold-only truth 结构性为0。`all_cold_sparse` 是 strict-cold 与 sparse1-5 的合计口径。",
        "", "| window | variant | segment | union | Top100 | Top50 | Top20 | Top12 | best rank | p25 | median | p75 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, variants in result["audits"]["cold_truth_funnel"].items():
        for variant in ("F0", "F1"):
            for segment in ("all_cold_sparse", "strict_cold", "sparse1_5"):
                row = variants[variant][segment]
                lines.append(
                    f"| {window} | {variant} | {segment} | {row['union']} | {row['Top100']} | "
                    f"{row['Top50']} | {row['Top20']} | {row['Top12']} | {row['best']} | "
                    f"{row['p25']} | {row['median']} | {row['p75']} |"
                )
    lines.extend([
        "", "## Top12 来源与替换", "",
        "插入/移除均相对同一用户的 W0 Top12；positive pair 是未来一周真实购买的用户—商品对。Cold/sparse insertion efficiency 的分母是所有新插入且截止日前事件数不超过5的推荐对。",
        "", "| window | variant | warm-only slots | cold-only slots | both-source slots | cold-only positive slots | both-source positive slots | inserted positives | removed positives | net positives | inserted cold/sparse positives | removed warm positives | efficiency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("W0", "W1", "F0", "F1"):
            ev = row["evaluations"][variant]
            comp = ev["top12_source_composition"]
            repl = ev["replacement_accounting"]
            efficiency = "N/A" if repl["cold_sparse_insertion_efficiency"] is None else f"{repl['cold_sparse_insertion_efficiency']:.6f}"
            warm = comp["all_slots"]["warm_only"]
            cold = comp["all_slots"]["cold_only"]
            both = comp["all_slots"]["warm_and_cold"]
            lines.append(
                f"| {window} | {variant} | {warm['count']} ({warm['share']:.2%}) | "
                f"{cold['count']} ({cold['share']:.2%}) | {both['count']} ({both['share']:.2%}) | "
                f"{comp['positive_slots']['cold_only']} | {comp['positive_slots']['warm_and_cold']} | "
                f"{repl['inserted_all_positive_pairs']} | {repl['removed_all_positive_pairs']} | "
                f"{repl['net_positive_pairs']} | {repl['inserted_cold_sparse_positive_pairs']} | "
                f"{repl['removed_warm_positive_pairs']} | {efficiency} |"
            )
    lines.extend([
        "", "## 正式门禁", "",
        "| gate | F0 | F1 |",
    ])
    lines.append("|---|---:|---:|")
    for key in ("overall_gate", "sparse_gate", "cold_top12_gate", "warm_protection_gate"):
        lines.append(f"| {key} | {result['gates']['F0'][key]} | {result['gates']['F1'][key]} |")
    lines.extend([
        f"| boundary_sampling_gate | N/A | {result['gates']['boundary_sampling']['passed']} |",
        "", "## 与旧 M5 Cold branch 比较", "",
        "这里比较同样采用历史采样的 F0 与旧 M5.5 Dual Fusion；差异主要来自 B0 Cold50 及其最小上游摘要，但候选并集也随 B0 Top50 改变，因此不能解释为单一特征的因果增益。",
        "", "| window | old M5.5 | P4 F0 | delta |", "|---|---:|---:|---:|",
    ])
    for window, row in result["historical_M5_cold_branch_comparison"].items():
        lines.append(f"| {window} | {row['old_M5_5_dual_fusion_map@12']:.8f} | {row['P4_F0_B0_fusion_map@12']:.8f} | {row['delta_P4_F0_minus_old_M5_5']:+.8f} |")
    historical = list(result["historical_M5_cold_branch_comparison"].values())
    old_mean = sum(row["old_M5_5_dual_fusion_map@12"] for row in historical) / len(historical)
    f0_mean = sum(row["P4_F0_B0_fusion_map@12"] for row in historical) / len(historical)
    lines.append(f"| **四窗 mean** | **{old_mean:.8f}** | **{f0_mean:.8f}** | **{f0_mean-old_mean:+.8f}** |")
    lines.extend([
        "", "## 跨来源校准与特征使用", "",
        "F0/F1 的训练样本与外层验证均报告两类成对比较：Cold 可用正例对 Warm rank 76–150 未观察候选，以及 Warm-only 正例对 B0 rank 1–20 未观察候选。pair accuracy 是正例分数更高的候选对比例；score gap 是正例分数减未观察候选分数。它们只作机制诊断，不参与晋级门禁。",
        "", "| window | variant | sample | pair type | pair rows | pair accuracy | gap mean | gap median |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    pair_labels = {
        "cold_positive_vs_warm_boundary": "Cold正例 vs Warm 76–150",
        "warm_positive_vs_cold_hard": "Warm正例 vs B0 1–20",
    }
    for window, variant_rows in result["audits"]["cross_source_calibration"].items():
        for variant in ("F0", "F1"):
            for sample_key, sample_label in (("outer_training_sample", "外层训练样本"), ("outer_validation", "外层验证")):
                for pair_key, pair_label in pair_labels.items():
                    row = variant_rows[variant][sample_key][pair_key]
                    lines.append(
                        f"| {window} | {variant} | {sample_label} | {pair_label} | {row['pair_rows']} | "
                        f"{row['pair_accuracy']:.6f} | {row['score_gap_mean']:.6f} | {row['score_gap_median']:.6f} |"
                    )
    lines.extend([
        "", "### 新增来源特征使用", "",
        "`windows with gain` 是该特征 gain 大于0的外层窗口数，分母固定为4。gain 与 split 只证明树模型曾使用该字段，不是因果消融结果。",
        "", "| variant | feature | gain | split | windows with gain |",
        "|---|---|---:|---:|---:|",
    ])
    audited_features = (
        "b0_score", "b0_rank", "b0_rank_pct", "b0_delta_vs_m4", "cold_present",
        "source_branch", "strict_cold_flag", "sparse1_5_flag",
    )
    for variant in ("F0", "F1"):
        for feature in audited_features:
            row = result["audits"]["feature_usage"][variant][feature]
            lines.append(
                f"| {variant} | {feature} | {row['gain']:.6f} | {row['split']} | "
                f"{row['windows_with_nonzero_gain']}/4 |"
            )
    f0_recall_gain = [
        row["evaluations"]["F0"]["candidate_recall"] - row["evaluations"]["W1"]["candidate_recall"]
        for row in window_rows
    ]
    f0_cold_top12 = [row["cold_truth_funnel"]["F0"]["all_cold_sparse"]["Top12"] for row in window_rows]
    f1_cold_top12 = [row["cold_truth_funnel"]["F1"]["all_cold_sparse"]["Top12"] for row in window_rows]
    w0_warm_mean = sum(
        row["evaluations"]["W0"]["segments"]["warm_21_plus"]["map@12"] for row in window_rows
    ) / len(window_rows)
    lines.extend([
        "", "## 失败分解", "",
        "1. **Cold50 可达性并非零，但增量有限。** F0 相对同样使用 Warm150 的 W1，四窗 candidate Recall 增量为 "
        + "/".join(f"{value:+.6f}" for value in f0_recall_gain) + "；因此 B0 Cold50 确实补入了额外 truth，但不是大幅扩展候选上界。",
        f"2. **Fusion 压入 Top12 的行为跨窗不稳定。** F0 的 cold-only truth Top12 为 {f0_cold_top12}，F1 为 {f1_cold_top12}；winter/early-summer 能进入，spring/late-summer 仍为0。",
        f"3. **正式失败的直接原因是 Warm 损伤。** F0/F1 的四窗 mean warm_21_plus MAP@12 分别为 {result['gates']['F0']['mean_warm_21_plus_map@12']:.8f}/{result['gates']['F1']['mean_warm_21_plus_map@12']:.8f}，均低于 W0 的 {w0_warm_mean:.8f}；overall gate 与 warm protection gate 均失败。",
        f"4. **边界采样没有形成稳定校准优势。** F1 相对 F0 的 mean MAP@12 为 {result['gates']['boundary_sampling']['mean_delta']:+.8f}，仅 {result['gates']['boundary_sampling']['non_degrade_count']}/4 窗不退化；它提高 sparse1-5 均值，但更严重地牺牲 Warm/overall。",
        "5. **B0 数值摘要被使用，但显式来源标签基本未被使用。** b0_score、b0_rank 或 b0_delta_vs_m4 在多数窗口有非零 gain；cold_present、source_branch 与 strict_cold_flag 在 F0/F1 四窗累计 gain 都为0。当前证据更符合跨窗 admission/calibration（准入与校准）不稳，而非模型完全忽略 B0 分数。",
        "", "## 防泄漏、工程与能力边界", "",
        "- 所有 B0 分数均记录评分截止日、模型训练截止日、标签结束日、模型与预处理器 SHA256；要求 `label_end <= scoring cutoff`。最早截止日只用 M4 Top50 回退，没有同截止日自拟合。",
        "- Warm Top150 逐截止日与 M5.4 文件身份完全一致；Cold50 每一行都来自同截止日冻结 M4 Top200。共享 PIT 特征仅使用截止日前历史。",
        "- 增量物化只对旧 M5.4 并集中不存在的新用户—商品行重算；重合行复用已审计 PIT 值，最终要求一对一全覆盖和用户—商品唯一。",
        "- 当前是离线验证；没有曝光、库存和在线转化证据。即使晋级，也只能说部分 cold/sparse truth 被安全引入 Top12，不能说 strict cold 或冷启动已解决。",
        "", "## 下一步", "",
        f"- 当前选择 `{selected}`。Phase 4 下一步：" + ("冻结 P4.0 候选方案，P4.1 仍等待人工决定。" if result["decision"]["P4_0_promoted"] else "停止自动 Fusion 扩展，依据失败分解等待人工决定。"),
        "- 最终周仍为 `not_run`；本任务未生成 Kaggle submission 或 leaderboard score。",
        "",
    ])
    return "\n".join(lines)


def write_report(report_dir: Path, result: dict[str, Any]) -> None:
    (report_dir / "P4_0_FINAL.md").write_text(render(result), encoding="utf-8")


def write_manifest(repo_root: Path, artifact_dir: Path, report_dir: Path) -> dict[str, Any]:
    report_names = [
        "P4_0_EXPERIMENT_CONTRACT.json", "P4_0_FINAL.md", "P4_0_metrics.json",
        "p4_0_candidate_union_audit.json", "p4_0_upstream_lineage_audit.json",
        "p4_0_sampling_audit.json", "p4_0_feature_contract.json", "p4_0_feature_usage.json",
        "p4_0_cross_source_calibration.json", "p4_0_cold_truth_funnel.json",
        "p4_0_replacement_accounting.json",
    ]
    sources = [
        repo_root / "src" / "hm_recsys" / name
        for name in ("p40.py", "p40_cli.py", "p40_contract.py", "p40_report.py", "p40_verify.py")
    ] + [repo_root / "scripts" / "run_p40.ps1", repo_root / "tests" / "test_p40.py"]
    large = [file_identity(path) for path in artifact_dir.rglob("*") if path.is_file()]
    manifest = {
        "schema_version": "phase4-p4.0-output-manifest-v1", "stage": "P4.0",
        "run_id": RUN_ID, "status": "completed",
        "reports": {name: file_identity(report_dir / name) for name in report_names},
        "sources": {
            str(path.relative_to(repo_root)).replace("\\", "/"): file_identity(path)
            for path in sources
        },
        "large_artifacts": large, "large_artifact_count": len(large),
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P4_0_OUTPUT_MANIFEST.json", manifest)
    return manifest
