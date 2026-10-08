from __future__ import annotations

from typing import Any


def render_development_report(result: dict[str, Any]) -> str:
    train = result["datasets"]["train"]
    valid = result["datasets"]["validation"]
    inputs = result["candidate_inputs"]
    orderings = result["evaluation"]["orderings"]
    lines = [
        "# M2 temporal development run",
        "",
        "## 角色与边界",
        "",
        "- 本运行是 development window，不是最终周结果。",
        f"- 训练目标周：`{train['cutoff']}`，hash {inputs['train']['sample_rate']:.0%} 用户。",
        f"- 开发验证周：`{valid['cutoff']}`，hash {inputs['validation']['sample_rate']:.0%} 用户。",
        "- 开发周只用于比较预声明模型形式；没有进入 fitting 或 early stopping。",
        "- target=0 仅表示未观察到购买，不是曝光后的真实负反馈。",
        "",
        "## 指标",
        "",
        "| ordering | MAP@12 | Recall@12 | HitRate@12 | Candidate Recall@100 | Oracle MAP@12 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("rrf", "lightgbm_retrieval", "lightgbm_full"):
        metric = orderings[name]["segments"]["overall"]
        lines.append(
            f"| {name} | {metric['map@12']:.6f} | {metric['recall@12']:.6f} | "
            f"{metric['hit_rate@12']:.6f} | {metric['candidate_recall@100']:.6f} | "
            f"{metric['oracle_map@12']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Full model top feature gain",
            "",
            "| feature | gain | splits |",
            "|---|---:|---:|",
        ]
    )
    for row in result["models"]["full"]["top_feature_importance"][:15]:
        lines.append(f"| {row['feature']} | {row['gain']:.2f} | {row['split']} |")
    lines.extend(
        [
            "",
            "## 后续门禁",
            "",
            "- 只允许依据本 development 结果冻结模型形式。",
            "- 在再次运行 final 前，不再根据 2020-09-16 的既有结果修改该形式。",
            "- 图片和文本 embedding 未启动。",
            "",
        ]
    )
    return "\n".join(lines)
