"""WV3-600: read-only one-swap tail-admission oracle audit."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal


TRIAL = "WV3-600"
VICTIM_RANKS = (8, 9, 10, 11, 12)
CHALLENGER_LOW = 13
CHALLENGER_HIGH = 50
CONTRACT = common.REPORT / "WV3-600_RESIDUAL_ADMISSION_AUDIT_CONTRACT.json"
REPORT_JSON = common.REPORT / "WV3-600_RESIDUAL_ADMISSION_AUDIT.json"
REPORT_MD = common.REPORT / "WV3-600_RESIDUAL_ADMISSION_AUDIT.md"

SOURCES = (
    ("history_20190220", "2019_outer_2019-02-20", "historical_supervision"),
    ("history_20190522", "2019_outer_2019-05-22", "historical_supervision"),
    ("history_20190821", "2019_outer_2019-08-21", "historical_supervision"),
    ("history_20191120", "2019_outer_2019-11-20", "historical_supervision"),
    ("winter_20200122", "2020_inner_2019-12-25", "inner_screen"),
    ("spring_20200318", "2020_inner_2020-02-19", "inner_screen"),
    ("early_summer_20200624", "2020_inner_2020-05-27", "inner_screen"),
    ("late_summer_20200819", "2020_inner_2020-07-22", "inner_screen"),
)


def apk_from_targets(targets: np.ndarray, truth_count: int, k: int = 12) -> float:
    """Exact AP@12 for one ranked binary target vector and the full truth count."""
    values = np.asarray(targets, dtype=np.uint8)[:k]
    denominator = min(int(truth_count), k)
    if denominator <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    positions = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(np.sum((cumulative / positions) * values) / denominator)


def best_one_swap(targets: np.ndarray, truth_count: int) -> dict:
    """Return the exact best no-op-or-one-swap result for ranks 8-12 versus 13-50."""
    values = np.asarray(targets, dtype=np.uint8)
    if len(values) < CHALLENGER_HIGH:
        raise ValueError("Expected at least 50 ranked candidates")
    baseline = apk_from_targets(values, truth_count)
    challenger_indices = np.flatnonzero(values[CHALLENGER_LOW - 1 : CHALLENGER_HIGH] == 1)
    victim_indices = np.flatnonzero(values[np.asarray(VICTIM_RANKS) - 1] == 0)
    best = baseline
    selected_victim = None
    selected_challenger = None
    for victim_offset in victim_indices:
        victim_rank = VICTIM_RANKS[int(victim_offset)]
        for challenger_offset in challenger_indices:
            challenger_rank = CHALLENGER_LOW + int(challenger_offset)
            changed = values.copy()
            changed[victim_rank - 1], changed[challenger_rank - 1] = (
                changed[challenger_rank - 1],
                changed[victim_rank - 1],
            )
            score = apk_from_targets(changed, truth_count)
            if score > best + 1e-15:
                best = score
                selected_victim = victim_rank
                selected_challenger = challenger_rank
    return {
        "baseline_ap": baseline,
        "oracle_ap": best,
        "delta_ap": best - baseline,
        "victim_rank": selected_victim,
        "challenger_rank": selected_challenger,
        "beneficial_pairs": int(len(challenger_indices) * len(victim_indices)),
    }


def source_meta(folder: str) -> tuple[Path, dict]:
    root = common.ART / "gate_data" / folder
    meta = read(root / "DATA.json")
    assert meta["final_week"] == "not_run"
    path = Path(meta["ranks_path"])
    assert path.is_file()
    return path, meta


def audit_one(name: str, folder: str, role: str) -> dict:
    path, meta = source_meta(folder)
    started = time.perf_counter()
    with connection() as con:
        frame = con.execute(
            f"""SELECT customer_id,article_id,ap_rf baseline_rank,target,truth_count,
                user_history_events_12w,candidate_rank,r0,r1,score_base,score_bpr,
                latent,missing,rrf_score
                FROM read_parquet({literal(path)})
                WHERE ap_rf<=50
                ORDER BY customer_id,ap_rf,article_id"""
        ).fetchdf()
    sizes = frame.groupby("customer_id", sort=False).size()
    assert len(sizes) == meta["included_users"]
    assert (sizes >= CHALLENGER_HIGH).all()
    assert not frame.duplicated(["customer_id", "article_id"]).any()
    assert not frame.duplicated(["customer_id", "baseline_rank"]).any()

    baseline_sum = 0.0
    oracle_sum = 0.0
    beneficial_users = 0
    beneficial_pairs = 0
    challenger_truth_pairs = 0
    protected_truth_pairs = 0
    victim_counts: Counter[int] = Counter()
    challenger_counts: Counter[int] = Counter()
    active_users = 0
    user_rows = []
    for customer, group in frame.groupby("customer_id", sort=False):
        ordered = group.sort_values("baseline_rank", kind="mergesort")
        targets = ordered["target"].to_numpy(np.uint8)
        truth_count = int(ordered["truth_count"].iloc[0])
        is_active = int(ordered["user_history_events_12w"].iloc[0]) > 0
        if is_active:
            active_users += 1
            result = best_one_swap(targets, truth_count)
        else:
            baseline = apk_from_targets(targets, truth_count)
            result = {
                "baseline_ap": baseline,
                "oracle_ap": baseline,
                "delta_ap": 0.0,
                "victim_rank": None,
                "challenger_rank": None,
                "beneficial_pairs": 0,
            }
        baseline_sum += result["baseline_ap"]
        oracle_sum += result["oracle_ap"]
        beneficial_pairs += result["beneficial_pairs"]
        challenger_truth_pairs += int(targets[CHALLENGER_LOW - 1 : CHALLENGER_HIGH].sum())
        protected_truth_pairs += int(targets[:7].sum())
        if result["delta_ap"] > 1e-15:
            beneficial_users += 1
            victim_counts[int(result["victim_rank"])] += 1
            challenger_counts[int(result["challenger_rank"])] += 1
        user_rows.append(result["delta_ap"])

    baseline_map = baseline_sum / meta["total_users"]
    oracle_map = oracle_sum / meta["total_users"]
    parity_error = baseline_map - meta["baseline_map_population_component"]
    assert abs(parity_error) < 1e-12
    deltas = np.asarray(user_rows, dtype=np.float64)
    return {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": role,
        "source": str(path),
        "total_users_denominator": meta["total_users"],
        "included_active_users": meta["included_users"],
        "users_with_positive_history": active_users,
        "candidate_rows_top50": len(frame),
        "baseline_MAP@12": baseline_map,
        "one_swap_oracle_MAP@12": oracle_map,
        "one_swap_oracle_delta": oracle_map - baseline_map,
        "beneficial_users": beneficial_users,
        "beneficial_user_share_of_full_denominator": beneficial_users / meta["total_users"],
        "beneficial_challenger_victim_pairs": beneficial_pairs,
        "truth_pairs_at_ranks13_50": challenger_truth_pairs,
        "truth_pairs_protected_at_ranks1_7": protected_truth_pairs,
        "selected_victim_rank_counts": {str(k): victim_counts[k] for k in VICTIM_RANKS},
        "selected_challenger_rank_counts": {str(k): challenger_counts[k] for k in sorted(challenger_counts)},
        "positive_user_delta_p50": float(np.median(deltas[deltas > 0])) if beneficial_users else 0.0,
        "baseline_parity_error": parity_error,
        "head_ranks1_7_changed": 0,
        "maximum_swaps_per_user": 1,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture_family": "bounded_residual_admission_audit",
        "hypothesis": "Freezing baseline ranks1-7 and allowing at most one exact swap from ranks13-50 into ranks8-12 leaves material AP@12 headroom without requiring broad reranking.",
        "population": "exact cached covered-active rank rows; all MAP values divide by the original full truth-user count for that window",
        "baseline": "WV2-601 ap_rf ranking",
        "protected_ranks": [1, 7],
        "victim_ranks": [8, 12],
        "challenger_ranks": [13, 50],
        "maximum_swaps_per_user": 1,
        "inputs": [folder for _, folder, _ in SOURCES],
        "inner_gate": {
            "mean_one_swap_oracle_delta_min": 0.002,
            "each_window_one_swap_oracle_delta_min": 0.001,
            "each_window_beneficial_users_min": 100,
            "baseline_parity_error_abs_max": 1e-12,
        },
        "resource_budget_minutes": 10,
        "training": "none",
        "outer_labels": "not accessed",
        "fallback": "If the gate fails, close residual admission before any learned model. If it passes, preregister a separate temporally trained admission experiment.",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(10)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        registry["status"] = "resumed_research"
        registry["resume_started_at"] = "2026-09-10T07:14:34.2631816+00:00"
        registry["resume_deadline"] = "2026-09-10T12:14:34.2631816+00:00"
        write(common.REGISTRY, registry)
        common.register(
            TRIAL,
            "bounded_residual_admission_audit",
            contract()["hypothesis"],
            candidate_protocol="unchanged WV2-601 pool; read-only ranks8-12 versus ranks13-50 opportunities",
            training_protocol="no training; exact no-op-or-one-swap AP@12 oracle on four historical and four inner windows",
            params={
                "protected_ranks": [1, 7],
                "victim_ranks": [8, 12],
                "challenger_ranks": [13, 50],
                "maximum_swaps_per_user": 1,
                "inner_gate": contract()["inner_gate"],
                "final_week": "not_run",
            },
            expected_minutes=10,
        )


def render_markdown(result: dict) -> str:
    lines = [
        "# WV3-600 局部残差准入上限审计",
        "",
        "## 结论",
        "",
        f"审计门槛：**{'pass' if result['gate']['passed'] else 'fail'}**。本阶段只计算理想上限，没有训练推荐模型、没有读取新外层标签。",
        "",
        "## 术语与口径",
        "",
        "- 局部残差准入（本项目自定义）：保留 WV2-601 主排序，只判断一个较低名次候选是否值得替换 Top12 尾部的一件商品。",
        "- 一换一 Oracle（本项目自定义理想上限）：知道验证周真值后，每个用户可选择不操作，或从基线第13–50名取一件商品，与第8–12名中的一件交换；第1–7名永不改变。",
        "- beneficial user（本报告称可受益用户）：至少存在一次能提高该用户 AP@12 的允许交换；人数除以完整真值用户分母计算占比。",
        "- MAP@12：全部真值用户的 AP@12 均值；缓存只含候选覆盖活跃用户，但未覆盖用户仍通过完整分母贡献 0。",
        "",
        "## 结果",
        "",
        "| 角色/窗口 | 完整用户分母 | 可受益用户 | 基线 MAP@12 | 一换一 Oracle MAP@12 | 增益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["windows"]:
        lines.append(
            f"| {row['role']} / {row['window']} | {row['total_users_denominator']:,} | {row['beneficial_users']:,} | "
            f"{row['baseline_MAP@12']:.9f} | {row['one_swap_oracle_MAP@12']:.9f} | {row['one_swap_oracle_delta']:+.9f} |"
        )
    lines += [
        "",
        "## 门槛与边界",
        "",
        f"四个内层窗口平均上限增益为 {result['gate']['mean_inner_delta']:+.9f}，最小为 {result['gate']['minimum_inner_delta']:+.9f}，每窗最少可受益用户 {result['gate']['minimum_inner_beneficial_users']:,}。",
        "",
        "通过只表示该受限动作空间有足够上限，不能证明上限可学习。下一阶段必须用严格早于 2020 内层窗口的历史监督训练，并在四个 2020 内层窗口筛选；若跨时间信号不稳定，则不读取外层。",
        "",
        "第1–7名在所有审计中变更数为 0；基线重算误差绝对值均小于 1e-12。最终周 2020-09-16 保持 not_run。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    rows = [audit_one(*spec) for spec in SOURCES]
    inner = [row for row in rows if row["role"] == "inner_screen"]
    gate = {
        "mean_inner_delta": float(np.mean([row["one_swap_oracle_delta"] for row in inner])),
        "minimum_inner_delta": min(row["one_swap_oracle_delta"] for row in inner),
        "minimum_inner_beneficial_users": min(row["beneficial_users"] for row in inner),
        "baseline_parity": all(abs(row["baseline_parity_error"]) < 1e-12 for row in rows),
    }
    gate["passed"] = (
        gate["mean_inner_delta"] >= 0.002
        and gate["minimum_inner_delta"] >= 0.001
        and gate["minimum_inner_beneficial_users"] >= 100
        and gate["baseline_parity"]
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "read_only_bounded_oracle_audit",
        "contract": str(CONTRACT),
        "windows": rows,
        "gate": gate,
        "new_training": False,
        "new_outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "2020-09-16 not_run",
    }
    write(REPORT_JSON, result)
    REPORT_MD.write_text(render_markdown(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="audit_pass" if gate["passed"] else "reject_mechanism",
        inner_evidence=gate,
        runtime=result["runtime_seconds"],
        artifact_paths=[str(REPORT_JSON), str(REPORT_MD)],
    )
    common.log(
        TRIAL,
        "The prior run repeatedly lost shared Top12 positives; the remaining question is whether a strictly tail-only action space has material headroom.",
        contract()["hypothesis"],
        str(REPORT_JSON),
        f"mean inner oracle delta={gate['mean_inner_delta']:+.9f}; minimum={gate['minimum_inner_delta']:+.9f}; minimum beneficial users={gate['minimum_inner_beneficial_users']}; gate={'pass' if gate['passed'] else 'fail'}.",
        "Proceed to a separately preregistered temporal learnability experiment." if gate["passed"] else "Close residual admission before training.",
        "If passed, train only from strictly earlier historical windows and keep Top1-7 immutable; otherwise return to WV2-601.",
        alternatives="A broad reranker and a fixed unanimous single swap already failed; expanding retrieval first would worsen density without testing protection.",
        experiment="Read-only exact AP@12 no-op-or-one-swap oracle over ranks8-12 versus13-50; no model fit and no outer access.",
        reflection="A passing oracle establishes action-space capacity only. Temporal learnability remains a separate falsifiable condition.",
    )
    print({"trial": TRIAL, "gate": gate, "seconds": result["runtime_seconds"]}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
