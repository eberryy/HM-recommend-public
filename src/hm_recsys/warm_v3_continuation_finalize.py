"""Close the second Warm-v3 continuation with the corrected valid champion."""
from __future__ import annotations

import json
from pathlib import Path

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write


FINAL_JSON = common.REPORT / "WARM_V3_FINAL.json"
FINAL_MD = common.REPORT / "WARM_V3_FINAL.md"
MANIFEST = common.REPORT / "WARM_V3_OUTPUT_MANIFEST.json"
ARCHITECTURE = common.REPORT / "WARM_V3_PROMOTED_ARCHITECTURE.mmd"
RUN_STATE = common.ROOT / "reports/private/warm_v3/AUTONOMOUS_RUN_STATE_CONTINUE_20260910.json"
DECISION_LOG = common.ROOT / "reports/private/warm_v3/AUTONOMOUS_DECISION_LOG.md"
ROADMAP = common.ROOT / "docs/ROADMAP_WARM_V3.zh-CN.md"
PROJECT_LOG = common.ROOT / "docs/PROJECT_LOG.zh-CN.md"


def continuation_rows(registry: dict) -> list[dict]:
    wanted = {
        "WV3-720", "WV3-721", "WV3-730", "WV3-731", "WV3-740", "WV3-741",
        "WV3-750", "WV3-751", "WV3-760", "WV3-761", "WV3-770", "WV3-780",
        "WV3-790", "WV3-791", "WV3-800", "WV3-801", "WV3-810",
    }
    return [row for row in registry["trials"] if row["experiment_id"] in wanted]


def architecture_text() -> str:
    return """flowchart TD
    A[六路启发式召回 Top100\n复购 / 近期热门 / 商品家族 / 同日共现 / 年龄热门 / 属性内容] --> C[冻结候选池]
    B[Item2Vec 扩展候选\n总预算最多约300] --> C
    C --> D0[E0: 84维 LightGBM LambdaRank]
    C --> D1[E1: 加入冻结 BPR 表示的 LightGBM LambdaRank]
    D0 --> E[等权倒数名次融合\nWV2-601 基础顺序]
    D1 --> E
    E --> F[活跃 Warm 用户 Top50\n第1-7名受保护]
    F --> G0[WV3-661: 99维点式购买倾向]
    F --> G1[WV3-680: 99维组内 LambdaRank]
    G0 --> H[等权 RRF60 模型名次融合]
    G1 --> H
    H --> I[无标签最多两次尾部换位\n第13-50名挑战第8-12名]
    I --> J[Warm Top12]
    K[无近期行为用户] --> L[WV2-601 inactive fallback 原样保留]
    L --> J
"""


def family_summary() -> list[dict]:
    return [
        {"experiments": ["WV3-720", "WV3-721"], "family": "决策阶段标签泄漏审计与无标签修复", "result": "确认WV3-691无效；冻结模型不变的合法重放为4/4正增，WV3-721成为有效开发冠军。", "decision": "阶段晋级，后被WV3-741替代"},
        {"experiments": ["WV3-730", "WV3-731"], "family": "BPR新增候选丰富特征准入", "result": "特征重建与缓存通过；四窗内层均退化，未读外层。", "decision": "未晋级"},
        {"experiments": ["WV3-740", "WV3-741"], "family": "历史点式模型与LambdaRank互补融合", "result": "固定等权RRF60内层4/4增益；一次外层确认继续4/4增益，成为最终有效冠军。", "decision": "最终晋级"},
        {"experiments": ["WV3-750", "WV3-751", "WV3-790", "WV3-791"], "family": "直接替换关系与有害动作否决", "result": "三分类接管排序明显退化；只作有害否决仍3/4轻微退化，关系模型复用路线关闭。", "decision": "未晋级"},
        {"experiments": ["WV3-760", "WV3-761"], "family": "最近热度Top600两阶段级联", "result": "宽池Oracle足以越过0.03，但Top600压至Top130仅保留38.38%真值且Oracle低于门槛，未训练末级排序器。", "decision": "未晋级"},
        {"experiments": ["WV3-770"], "family": "全部历史购买回补", "result": "当前池遗漏部分主要是老商品，密度低于Top600，平均Oracle仅+0.000174，关闭。", "decision": "未晋级"},
        {"experiments": ["WV3-780"], "family": "深层同日共现扩展", "result": "边际真值和Oracle存在，但前三窗密度提升未达1.5倍门槛，关闭。", "decision": "未晋级"},
        {"experiments": ["WV3-800", "WV3-801"], "family": "含正例用户组点式模型", "result": "历史监督充足，但加入三路融合后四窗全部退化，说明全负组稀释不是缺失互补信号。", "decision": "未晋级"},
        {"experiments": ["WV3-810"], "family": "近期热门与深层共现一致性候选", "result": "三窗达到2倍密度，但spring仅1.76倍，未过全窗门槛。", "decision": "未晋级"},
    ]


def build_final(registry: dict) -> dict:
    baseline = read(common.REPORT / "WV3-741_OUTER.json")["windows"]
    champion = read(common.REPORT / "WV3-741_OUTER.json")
    corrected = read(common.REPORT / "WV3-721_CORRECTED_OUTER_REPLAY.json")
    popular = read(common.REPORT / "WV3-760_POPULAR_CASCADE_AUDIT.json")
    stage1 = read(common.REPORT / "WV3-761_SCREEN.json")
    deep = read(common.REPORT / "WV3-780_DEEP_COVISIT_AUDIT.json")
    consensus = read(common.REPORT / "WV3-810_POP_COVISIT_CONSENSUS_AUDIT.json")
    top100 = read(common.REPORT / "WV3-640_TOP100_ORACLE_AUDIT.json")
    rows = continuation_rows(registry)
    runtime_sum = float(sum((row.get("runtime") or 0.0) for row in rows))
    generated = now()
    return {
        "generated_at": generated,
        "status": "complete_below_target",
        "branch": "warm-v3-architecture-lab",
        "timer": {
            "started_at": "2026-09-10T13:19:04.0890781+00:00",
            "deadline": "2026-09-10T18:19:04.0890781+00:00",
            "search_stopped_at": generated,
            "stop_reason": "Evidence converged before deadline: three candidate expansions failed density or filtering gates and three ranking/supervision variants failed inner incremental gates; remaining time reserved for mandatory closure.",
        },
        "validity_correction": {
            "invalidated_experiment": "WV3-691",
            "reason": "validation-label-derived unit_gain participated in action ordering",
            "repair": "WV3-721 projected decisions to identifiers, ranks and frozen scores and passed decision-invariance checks",
            "repair_evidence_status": "post_audit_corrected_development_replay_not_blind",
        },
        "promotion": {
            "target_mean_MAP@12": 0.03,
            "target_achieved": False,
            "promoted_architecture": "WV3-741",
            "mean_MAP@12": champion["mean_MAP"],
            "gap_to_target": 0.03 - champion["mean_MAP"],
            "delta_vs_WV2_601": champion["delta_vs_WV2_601"],
            "nondegrade_windows_vs_WV2_601": champion["nondegrade_windows"],
            "worst_delta_vs_WV2_601": champion["worst_delta"],
            "incremental_mean_vs_WV3_721": champion["incremental_vs_WV3_721"]["mean_delta_vs_WV3_721"],
            "incremental_nondegrade_windows_vs_WV3_721": champion["incremental_vs_WV3_721"]["nondegrade_windows_vs_WV3_721"],
            "incremental_worst_delta_vs_WV3_721": champion["incremental_vs_WV3_721"]["worst_delta_vs_WV3_721"],
        },
        "baseline_2020": {
            "MAP@12_by_window": {name: row["baseline_MAP@12"] for name, row in baseline.items()},
            "mean_MAP@12": float(np_mean([row["baseline_MAP@12"] for row in baseline.values()])),
        },
        "champion_2020": {
            "MAP@12_by_window": champion["per_window_MAP"],
            "delta_vs_WV2_601_by_window": {name: row["MAP@12"] - row["baseline_MAP@12"] for name, row in baseline.items()},
            "incremental_vs_WV3_721_by_window": champion["incremental_vs_WV3_721"]["per_window_delta_vs_WV3_721"],
        },
        "promoted_architecture": {
            "retrieval": "frozen WV2-601 six heuristic sources plus Item2Vec expansion",
            "base_ranking": "two LightGBM LambdaRank views, one with frozen BPR representation, combined by equal reciprocal-rank fusion",
            "local_action_population": "active Warm users; frozen base ranks1-50",
            "candidate_representation": "99 cutoff-safe candidate features spanning ranks, source evidence, trends and user-attribute affinities",
            "local_models": "frozen WV3-661 pointwise purchase propensity and WV3-680 within-user LambdaRank",
            "local_fusion": "equal RRF60 over the two local model ranks",
            "action": "label-free greedy maximum two disjoint swaps from ranks13-50 into ranks8-12; ranks1-7 protected",
            "fallback": "exact WV2-601 ordering for inactive users or no selected action",
            "output": "Top12",
        },
        "families": family_summary(),
        "bottleneck_evidence": {
            "Top100_same_pool_one_swap_oracle_mean": np_mean([row["top100_oracle_delta"] for row in top100["windows"].values()]),
            "recent_popularity_Top600_rank12_oracle_mean": popular["mean_rank12_constrained_oracle_population_delta"],
            "recent_popularity_Top130_truth_retention_mean": stage1["mean_truth_retention"],
            "recent_popularity_Top130_rank12_oracle_mean": stage1["mean_selected_rank12_oracle_population_delta"],
            "deep_covisit_rank12_oracle_mean": deep["mean_rank12_constrained_oracle_population_delta"],
            "popularity_covisit_consensus_rank12_oracle_mean": consensus["mean_rank12_constrained_oracle_population_delta"],
            "conclusion": "Candidate headroom exists, but label-free conversion is limited by very sparse actionable positives and unstable discrimination; widening pools without a genuinely stronger cascade ranker increases dilution.",
        },
        "cost": {
            "second_continuation_registered_runtime_seconds_sum": runtime_sum,
            "new_outer_exposures": 2,
            "new_outer_exposure_ids": ["WV3-721", "WV3-741"],
            "total_registry_outer_exposure_count": registry["outer_exposure_count"],
            "tests_run": 148,
            "test_failures": 0,
            "test_errors": 0,
            "test_runner_seconds": 11.076121400110424,
        },
        "next_research_boundary": {
            "recommended": "If research resumes, build a truly stacked two-stage ranker with out-of-fold first-stage predictions and source-balanced supervision before reconsidering Top600 or deep co-visitation candidates.",
            "prerequisite": "historical out-of-fold first-stage candidate scores, exact same-pool control, candidate-source density audit, and a budget that covers feature reconstruction plus four-window confirmation",
            "not_recommended": ["threshold rescue on WV3-751 or WV3-791", "Top130/K/negative-ratio tuning on WV3-761", "direct Top600 mixing", "all-history purchase append", "final-week selection"],
        },
        "outer_exposure": {"count": registry["outer_exposure_count"], "second_continuation_ids": ["WV3-721", "WV3-741"]},
        "tests": {"status": "pass", "framework": "Python unittest", "scope": "tests/test_warm_v3*.py", "tests_run": 148, "failures": 0, "errors": 0, "runner_seconds": 11.076121400110424},
        "final_week": {"cutoff": "2020-09-16", "status": "not_run", "labels_accessed": False, "submission_created": False},
        "corrected_replay": {"WV3_721_mean_MAP@12": corrected["mean_MAP"], "evidence_status": "post_audit_corrected_development_replay_not_blind"},
    }


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def render(final: dict) -> str:
    p = final["promotion"]
    base = final["baseline_2020"]["MAP@12_by_window"]
    champ = final["champion_2020"]["MAP@12_by_window"]
    delta = final["champion_2020"]["delta_vs_WV2_601_by_window"]
    inner = final["champion_2020"]["incremental_vs_WV3_721_by_window"]
    lines = [
        "# Warm-v3 自主架构研究最终报告（有效性修复后续跑）",
        "",
        "## 结论",
        "",
        f"最终有效开发冠军是 **WV3-741**：四窗平均 MAP@12 `{p['mean_MAP@12']:.12f}`，相对 WV2-601 增加 `{p['delta_vs_WV2_601']:+.12f}`，4/4 窗口不退化；距离预设目标0.03仍差 `{p['gap_to_target']:.12f}`。",
        "",
        "WV3-691 已由 WV3-720 确认存在决策阶段标签泄漏，永久作废。WV3-721 用无标签动作合同修复后成为合法开发结果；WV3-741 在相同候选池上融合严格历史训练的点式模型与 LambdaRank，并通过一次冻结外层确认。WV3-721 是泄漏发现后的更正重放，不能称为独立盲测；WV3-741 是随后预注册的新机制外层确认。",
        "",
        "## 术语与评测口径",
        "",
        "- MAP@12（推荐评测通用）：逐用户计算前12个推荐的平均准确率，再对该窗口全部有真值用户取平均；候选未覆盖用户仍在分母中。",
        "- 点式购买倾向（推荐排序常见）：对每个用户—候选商品独立预测下一周购买倾向；WV3-661 使用99个截止日前特征。",
        "- LambdaRank（行业通用学习排序）：在同一用户候选组内学习正例应排在负例前；WV3-680 使用同一99维表示。",
        "- RRF60 名次融合（推荐系统常见方法，本项目固定常数）：两个模型各自转为用户内名次，分数相加 `1/(60+名次)`；只比较次序，不混合未校准原始分数。",
        "- 无标签尾部换位（本项目自定义）：动作阶段只读取用户/商品标识、冻结名次和模型分数；每用户最多两次把第13–50名商品换入第8–12名，第1–7名不动。",
        "- 外层窗口（本项目开发协议）：与内层窗口相邻但独立的四个2020验证切片；每个正式方案最多读取一次，不是比赛最终周。",
        "",
        "## 四窗结果",
        "",
        "| 外层窗口 | WV2-601 | WV3-741 | 相对基线 | 相对 WV3-721 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in champ:
        lines.append(f"| {name} | {base[name]:.9f} | {champ[name]:.9f} | {delta[name]:+.9f} | {inner[name]:+.9f} |")
    lines += [
        "",
        f"相对 WV3-721 的平均增量 `{p['incremental_mean_vs_WV3_721']:+.9f}`，4/4 不退化，最差窗口 `{p['incremental_worst_delta_vs_WV3_721']:+.9f}`。",
        "",
        "## 最终架构",
        "",
        "```mermaid",
        architecture_text().rstrip(),
        "```",
        "",
        "同一 Mermaid 源文件见 `WARM_V3_PROMOTED_ARCHITECTURE.mmd`。BPR 在最终冠军中作为冻结协同表示进入基础排序器，不是本轮新扩展候选源；WV3-710 的 BPR 全目录候选扩展已被拒绝。",
        "",
        "## 本次续跑的实验闭环",
        "",
        "| 实验 | 动机与结果 | 决定 |",
        "|---|---|---|",
    ]
    for row in final["families"]:
        lines.append(f"| {', '.join(row['experiments'])} | {row['result']} | {row['decision']} |")
    lines += [
        "",
        "## 为什么仍未达到 0.03",
        "",
        "现有 Top100 同池一换一 Oracle 约为 `+0.00705`，而最近热度 Top600 的第12名受限 Oracle 平均为 `+0.001966`，说明真值不是完全不存在。真正失败发生在可部署的无标签选择：热门 Top600 压到 Top130 后平均只保留 `38.38%` 边际真值；深层共现虽有 `+0.000709` Oracle，但前三窗候选密度不足；关系三分类、伤害否决和含正例组模型也都未跨窗改善 WV3-741。",
        "",
        "因此当前结论不是‘Recall 提升无意义’，而是：候选上界存在，但可学习的有益换位极稀疏，现有单阶段/局部模型兑现率太低；继续直接扩池只会放大负例稀释。若以后重开，合理的新项目单元是带历史窗口外预测的真正两阶段级联：第一阶段宽召回提纯，第二阶段用更丰富特征联合排序，并保持来源平衡和同池对照。它需要单独预算，不能作为本轮失败参数的临时救援。",
        "",
        "## 成本、边界与可复现性",
        "",
        f"本次续跑登记实验运行时间合计约 `{final['cost']['second_continuation_registered_runtime_seconds_sum']:.1f}` 秒；新增外层暴露仅 WV3-721 与 WV3-741 两次，注册表累计外层暴露 `{final['cost']['total_registry_outer_exposure_count']}` 次。完整 Warm-v3 测试 `148` 个，0失败、0错误。",
        "",
        "最终周 `2020-09-16` 保持 `not_run`：未读取标签、未评分、未生成提交。没有推送、合并或修改 main/Cold/Admission。输出清单只验证文件存在、非空和 JSON 可解析；没有在缺少可信对照值时用单文件哈希冒充完整性证明。",
        "",
    ]
    return "\n".join(lines)


def append_once(path: Path, marker: str, content: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")


def build_manifest() -> dict:
    files = []
    all_json = True
    for path in sorted(common.REPORT.glob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        row = {"path": path.relative_to(common.ROOT).as_posix(), "bytes": path.stat().st_size, "nonempty": path.stat().st_size > 0}
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
                row["json_parse_ok"] = True
            except Exception:
                row["json_parse_ok"] = False
                all_json = False
        files.append(row)
    return {"generated_at": now(), "status": "current_corrected_closure_inventory", "verification": "structural existence, nonempty size and JSON parsing; no standalone hashes without a trusted comparison artifact", "file_count": len(files), "all_nonempty": all(row["nonempty"] for row in files), "all_json_parse": all_json, "files": files, "final_week": "not_run"}


def run() -> dict:
    common.setup()
    registry = read(common.REGISTRY)
    final = build_final(registry)
    ARCHITECTURE.write_text(architecture_text(), encoding="utf-8")
    write(FINAL_JSON, final)
    FINAL_MD.write_text(render(final), encoding="utf-8")
    registry = read(common.REGISTRY)
    registry["current_champion"] = "WV3-741"
    registry["current_valid_champion"] = "WV3-741"
    registry["status"] = "complete_below_target"
    registry["search_stop_reason"] = final["timer"]["stop_reason"]
    registry["finalized_at"] = final["generated_at"]
    registry["best_stable"] = {"experiment_id": "WV3-741", "mean_MAP": final["promotion"]["mean_MAP@12"], "delta_vs_WV2_601": final["promotion"]["delta_vs_WV2_601"], "nondegrade_windows": final["promotion"]["nondegrade_windows_vs_WV2_601"], "worst_delta": final["promotion"]["worst_delta_vs_WV2_601"]}
    registry["validity_correction"] = final["validity_correction"]
    registry["final_week"] = "not_run"
    write(common.REGISTRY, registry)
    state = read(RUN_STATE)
    state.update({"status": "complete_below_target", "current_valid_champion": "WV3-741", "search_stopped_at_utc": final["generated_at"], "stop_reason": final["timer"]["stop_reason"], "target_achieved": False, "final_week": "not_run"})
    write(RUN_STATE, state)
    append_once(DECISION_LOG, "WV3 second continuation closure", f"""## {final['generated_at']} — WV3 second continuation closure

### Observation
WV3-741 is the only new label-free architecture with a passing frozen outer confirmation. WV3-761, WV3-770, WV3-780 and WV3-810 found candidate headroom but failed preregistered density/filter gates; WV3-751, WV3-791 and WV3-801 failed inner ranking gates.

### Decision
Stop architecture search before the deadline and reserve time for complete evidence closure. Promote WV3-741 as the valid development champion; target 0.03 remains unmet.

### Next action
Human review only. If research resumes, require a separately budgeted out-of-fold two-stage cascade rather than parameter rescue. Final week remains not_run.""")
    append_once(ROADMAP, "## 续跑收口（2026-09-11）", """## 续跑收口（2026-09-11）

- 最终有效开发冠军：WV3-741，四窗 mean MAP@12 = 0.028172096764，4/4 相对 WV2-601 不退化。
- WV3-691 因决策阶段标签泄漏作废；WV3-721 为无标签更正重放，WV3-741 为后续一次冻结外层确认。
- 热门 Top600 级联、历史购买回补、深层共现、跨来源一致性、关系否决和含正例组模型均已按预注册门槛关闭。
- 后续若重开，只考虑另立预算的窗口外预测两阶段级联；不做当前失败方案的阈值、TopK、负采样或树参数救援。
- final week 2020-09-16：not_run。""")
    append_once(PROJECT_LOG, "Warm-v3 有效性修复与第二次续跑收口", f"""## {final['generated_at']} — Warm-v3 有效性修复与第二次续跑收口

- 问题：WV3-720 发现 WV3-691 的验证标签派生 `unit_gain` 进入动作排序，原冠军无效。
- 修复：WV3-721 将动作输入限制为标识、名次和冻结模型分数，并通过决策不变性检查；更正后 mean MAP@12 0.028098753698。
- 最终改进：WV3-741 等权融合99维点式模型与 LambdaRank 的用户内名次，保持候选池和无标签两次尾部换位；外层 mean 0.028172096764，相对 WV2-601 +0.000292043460，4/4不退化。
- 后续诊断：Top600 有足够 Oracle，但轻量 Top130 过滤保留率只有38.38%；历史回补边际太旧，深层共现和热门×共现一致性没有全窗通过密度门槛；三分类关系、伤害否决与含正例组模型均未改善当前冠军。
- 状态：已解决评测合同错误并形成有效冠军；0.03目标未解决。最终周 not_run；未推送、未合并。""")
    write(MANIFEST, build_manifest())
    return final


if __name__ == "__main__":
    run()
