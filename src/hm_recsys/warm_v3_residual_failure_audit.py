"""WV3-602: read-only decomposition of WV3-601 inner admission decisions."""
from __future__ import annotations

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write


TRIAL = "WV3-602"
CONTRACT = common.REPORT / "WV3-602_RESIDUAL_FAILURE_AUDIT_CONTRACT.json"
OUTPUT = common.REPORT / "WV3-602_RESIDUAL_FAILURE_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-602_RESIDUAL_FAILURE_AUDIT.md"


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "read_only_failure_decomposition",
        "hypothesis": "WV3-601 captures little oracle headroom because a classifier trained only on discordant pairs cannot distinguish actionable pairs from neutral pairs at inference.",
        "inputs": ["WV3-600_RESIDUAL_ADMISSION_AUDIT.json", "WV3-601_SCREEN.json"],
        "statistics": [
            "selected neutral share",
            "beneficial-user capture relative to the one-swap oracle",
            "realized MAP delta divided by one-swap oracle delta",
            "harmful-to-beneficial selected-user ratio",
        ],
        "diagnostic_gate": {
            "mean_selected_neutral_share_min": 0.9,
            "mean_beneficial_user_capture_max": 0.1,
            "mean_realized_oracle_ratio_max": 0.1,
        },
        "training": "none",
        "new_outer_exposure": 0,
        "fallback": "If the gate fails, do not add an actionability head; investigate another bottleneck.",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(5)
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "residual_admission_failure_diagnostic",
            experiment_contract()["hypothesis"],
            candidate_protocol="unchanged; reads only completed WV3-600/601 inner summaries",
            training_protocol="no training and no new label access",
            params={"diagnostic_gate": experiment_contract()["diagnostic_gate"], "final_week": "not_run"},
            expected_minutes=5,
        )


def run() -> dict:
    common.setup()
    common.budget(5)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    oracle = read(common.REPORT / "WV3-600_RESIDUAL_ADMISSION_AUDIT.json")
    screen = read(common.REPORT / "WV3-601_SCREEN.json")
    oracle_inner = {row["window"]: row for row in oracle["windows"] if row["role"] == "inner_screen"}
    rows = {}
    for window, result in screen["windows"].items():
        reference = oracle_inner[window]
        selected = result["selected_users"]
        rows[window] = {
            "selected_users": selected,
            "selected_neutral_share": result["neutral_selected_users"] / selected,
            "beneficial_selected_users": result["beneficial_selected_users"],
            "oracle_beneficial_users": reference["beneficial_users"],
            "beneficial_user_capture": result["beneficial_selected_users"] / reference["beneficial_users"],
            "harmful_to_beneficial_ratio": result["harmful_selected_users"] / result["beneficial_selected_users"],
            "realized_MAP_delta": result["population_delta"],
            "one_swap_oracle_delta": reference["one_swap_oracle_delta"],
            "realized_oracle_ratio": result["population_delta"] / reference["one_swap_oracle_delta"],
        }
    aggregate = {
        "mean_selected_neutral_share": float(np.mean([row["selected_neutral_share"] for row in rows.values()])),
        "mean_beneficial_user_capture": float(np.mean([row["beneficial_user_capture"] for row in rows.values()])),
        "mean_realized_oracle_ratio": float(np.mean([row["realized_oracle_ratio"] for row in rows.values()])),
        "all_inner_deltas_positive": all(row["realized_MAP_delta"] > 0 for row in rows.values()),
    }
    gate = experiment_contract()["diagnostic_gate"]
    aggregate["actionability_head_warranted"] = (
        aggregate["mean_selected_neutral_share"] >= gate["mean_selected_neutral_share_min"]
        and aggregate["mean_beneficial_user_capture"] <= gate["mean_beneficial_user_capture_max"]
        and aggregate["mean_realized_oracle_ratio"] <= gate["mean_realized_oracle_ratio_max"]
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "read_only_completed_inner_failure_decomposition",
        "windows": rows,
        "aggregate": aggregate,
        "new_training": False,
        "new_outer_exposure": 0,
        "final_week": "not_run",
    }
    write(OUTPUT, result)
    lines = [
        "# WV3-602 局部准入失败分解",
        "",
        "## 结论",
        "",
        f"actionability head（本项目称可行动性预测头）机制门槛：**{'pass' if aggregate['actionability_head_warranted'] else 'fail'}**。可行动性指一对商品在监督周恰好一件被购买；它不同于 WV3-601 在已知异标签条件下判断哪件更好的偏好预测头。",
        "",
        "| 内层窗口 | 入选中性占比 | 可受益用户捕获率 | MAP上限兑现率 | 有害/有益人数比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for window, row in rows.items():
        lines.append(
            f"| {window} | {row['selected_neutral_share']:.2%} | {row['beneficial_user_capture']:.2%} | "
            f"{row['realized_oracle_ratio']:.2%} | {row['harmful_to_beneficial_ratio']:.3f} |"
        )
    lines += [
        "",
        f"四窗均值：中性占比 {aggregate['mean_selected_neutral_share']:.2%}，可受益用户捕获率 {aggregate['mean_beneficial_user_capture']:.2%}，MAP上限兑现率 {aggregate['mean_realized_oracle_ratio']:.2%}。",
        "",
        "本审计只复用已完成的内层汇总，没有重训、没有重新读取外层。若门槛通过，只授权预注册一个独立的双头期望收益架构：历史偏好头乘以历史可行动性头，再按位置AP权重选择至多一次交换；不修改 WV3-601 的阈值或轮数。最终周保持 not_run。",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    common.update(
        TRIAL,
        decision="diagnostic_supports_actionability_head" if aggregate["actionability_head_warranted"] else "diagnostic_rejects_actionability_head",
        inner_evidence=aggregate,
        runtime=0.0,
        artifact_paths=[str(OUTPUT), str(MARKDOWN)],
    )
    common.log(
        TRIAL,
        "WV3-601 was stable but converted only a small fraction of its tail-only oracle and selected mostly neutral pairs.",
        experiment_contract()["hypothesis"],
        str(OUTPUT),
        f"mean neutral share={aggregate['mean_selected_neutral_share']:.2%}; beneficial capture={aggregate['mean_beneficial_user_capture']:.2%}; oracle realization={aggregate['mean_realized_oracle_ratio']:.2%}; gate={aggregate['actionability_head_warranted']}.",
        "Preregister one two-head expected-gain architecture." if aggregate["actionability_head_warranted"] else "Do not pursue actionability modeling.",
        "Use all historical pair outcomes for a separate actionability head; keep the frozen WV3-601 preference head and one-swap action space.",
        alternatives="Do not tune the already outer-exposed WV3-601 threshold or swap count. A new head changes the supervised factorization rather than rescuing its parameters.",
        experiment="Read-only arithmetic over completed WV3-600/601 inner summaries; no model and no new labels.",
        reflection="A high neutral share is operational inefficiency, while low beneficial capture and oracle realization show that pair prioritization is also the main remaining opportunity.",
    )
    print({"trial": TRIAL, "aggregate": aggregate}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
