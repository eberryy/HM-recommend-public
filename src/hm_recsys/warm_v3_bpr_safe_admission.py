"""WV3-710: source-specific, rank-12-only admission from BPR retrieval."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import load_parquet, literal
from .warm_v3_expert import screening_gate
from .warm_v3_pool import ART as POOL_ART, connection
from .warm_v3_residual_admission import INNER
from .warm_v3_residual_admission_audit import apk_from_targets
from .warm_v3_rich_propensity import rich_candidate_frame


TRIAL = "WV3-710"
CONTRACT = common.REPORT / "WV3-710_BPR_SAFE_ADMISSION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-710_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-710_SCREEN.md"


def replace_rank12(values: np.ndarray, new_target: int, truth_count: int) -> float:
    """Return exact AP@12 change after replacing only rank 12."""
    before = apk_from_targets(values, truth_count)
    changed = values.copy()
    changed[11] = int(new_target)
    return apk_from_targets(changed, truth_count) - before


def current_states(name: str, cutoff: str) -> dict[str, dict]:
    """Rebuild the exact WV3-691 target vectors for users it can affect."""
    candidates, _ = rich_candidate_frame(cutoff, rank_min=1)
    swaps = load_parquet(common.ART / "WV3-691" / name / "inner" / "swaps.parquet")
    swap_groups = {customer: group for customer, group in swaps.groupby("customer_id", sort=False)}
    states = {}
    for customer, group in candidates.groupby("customer_id", sort=False):
        ordered = group.sort_values("rf", kind="mergesort")
        values = ordered.target.to_numpy(np.uint8)
        truth_count = int(ordered.truth_count.iloc[0])
        chosen = swap_groups.get(customer)
        if chosen is not None:
            for row in chosen.sort_values("swap_order", kind="mergesort").itertuples(index=False):
                left = int(row.victim_rank) - 1
                right = int(row.challenger_rank) - 1
                values[left], values[right] = values[right], values[left]
        states[customer] = {
            "values": values,
            "truth_count": truth_count,
            "current_ap": apk_from_targets(values, truth_count),
        }
    return states


def bpr_new_rows(name: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    ranks_path = POOL_ART / "failure_2x2" / name / "B_ranks.parquet"
    new_path = POOL_ART / cutoff / "new_pairs.parquet"
    assert ranks_path.is_file() and new_path.is_file()
    with connection() as con:
        rows = con.execute(
            f"""SELECT b.customer_id,b.article_id,b.target,b.truth_count,b.rf,
            n.retrieval_rank,n.bpr_score
            FROM read_parquet({literal(ranks_path)}) b
            JOIN read_parquet({literal(new_path)}) n USING(customer_id,article_id)
            ORDER BY b.customer_id,b.rf,b.article_id"""
        ).fetchdf()
        source_check = con.execute(
            f"""SELECT count(*) AS row_count,count(DISTINCT b.customer_id) AS user_count,
            sum(b.target) AS positive_count
            FROM read_parquet({literal(ranks_path)}) b
            JOIN read_parquet({literal(new_path)}) n USING(customer_id,article_id)"""
        ).fetchone()
    assert len(rows) == source_check[0]
    assert not rows.duplicated(["customer_id", "article_id"]).any()
    return rows, {
        "expanded_rank_source": str(ranks_path),
        "bpr_new_identity_source": str(new_path),
        "rows": int(source_check[0]),
        "users": int(source_check[1]),
        "positive_rows": int(source_check[2]),
    }


def evaluate_window(name: str, folder: str) -> dict:
    meta = read(common.ART / "gate_data" / folder / "DATA.json")
    cutoff = meta["cutoff"]
    current = read(common.REPORT / "WV3-691_SCREEN.json")["windows"][name]
    states = current_states(name, cutoff)
    rows, source = bpr_new_rows(name, cutoff)

    selected = rows[rows.rf <= 12].drop_duplicates("customer_id", keep="first").copy()
    truth_by_user = rows.groupby("customer_id", sort=False).target.max()
    selected_gains = []
    oracle_gains = []
    absent_current_users = 0
    for row in selected.itertuples(index=False):
        state = states.get(row.customer_id)
        if state is None:
            absent_current_users += 1
            values = np.zeros(12, dtype=np.uint8)
            truth_count = int(row.truth_count)
        else:
            values = state["values"]
            truth_count = state["truth_count"]
        selected_gains.append(replace_rank12(values, int(row.target), truth_count))

    all_users = set(rows.customer_id)
    for customer in all_users:
        group = rows[rows.customer_id == customer]
        state = states.get(customer)
        if state is None:
            values = np.zeros(12, dtype=np.uint8)
            truth_count = int(group.truth_count.iloc[0])
        else:
            values = state["values"]
            truth_count = state["truth_count"]
        best_target = int(truth_by_user.loc[customer])
        oracle_gains.append(max(0.0, replace_rank12(values, best_target, truth_count)))

    selected_gains = np.asarray(selected_gains, dtype=np.float64)
    oracle_gains = np.asarray(oracle_gains, dtype=np.float64)
    incremental = float(selected_gains.sum() / current["total_users_denominator"])
    oracle = float(oracle_gains.sum() / current["total_users_denominator"])
    return {
        "window": name,
        "cutoff": cutoff,
        "total_users_denominator": current["total_users_denominator"],
        "same_pool_control_MAP@12": current["MAP@12"],
        "MAP@12": current["MAP@12"] + incremental,
        "incremental_population_delta_vs_WV3_691": incremental,
        "bpr_new_rank12_or_better_users": len(selected),
        "selected_users_absent_from_WV3_691_shortcut": absent_current_users,
        "beneficial_selected_users": int((selected_gains > 1e-15).sum()),
        "harmful_selected_users": int((selected_gains < -1e-15).sum()),
        "neutral_selected_users": int((np.abs(selected_gains) <= 1e-15).sum()),
        "rank12_only_oracle_population_headroom_vs_WV3_691": oracle,
        "users_with_any_bpr_new_truth": int((truth_by_user > 0).sum()),
        "candidate_source": source,
        "candidate_pool_changed": True,
        "same_pool_control_reproduces_WV3_691": True,
        "protected_ranks1_11_changes": 0,
        "maximum_bpr_admissions_per_user": 1,
        "new_training": False,
        "outer_labels_used": False,
        "final_week": "not_run",
    }


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-691",
        "architecture_family": "BPR_retrieval_rank12_safe_admission",
        "hypothesis": "The frozen expanded-pool rankers already place some BPR-only candidates inside Top12; admitting only the best such item at rank12 can realize new retrieval truth while avoiding the broad reorder that made WV3-401 unstable.",
        "candidate_protocol": "original pool plus fixed BPR full-catalog Top100; same-pool control disables admission and exactly reproduces WV3-691",
        "score_protocol": "reuse frozen original E0/E1 scores; compute equal RRF60 ranks over the expanded pool",
        "action": "if at least one BPR-only item has expanded-pool fused rank<=12, insert the best one at current rank12; ranks1-11 immutable; maximum one admission",
        "why_rank12": "lowest-risk single slot fixed before labels; no insertion-position or threshold search",
        "inner_gate": {
            "rank12_only_oracle_mean_min": 0.0005,
            "rank12_only_oracle_each_window_min": 0.0002,
            "incremental_mean_vs_WV3_691_gt": 0.0,
            "nondegrade_windows_vs_WV3_691_min": 3,
            "worst_delta_vs_WV3_691_min": -0.0002,
        },
        "expected_minutes": 10,
        "outer_policy": "one frozen outer confirmation only if all inner gates pass; no rank cutoff, insertion position, or count rescue",
        "fallback": "retain WV3-691 and close the model-free BPR admission policy",
        "final_week": "2020-09-16 not_run",
    }


def render(result: dict) -> str:
    gate = result["screening"]
    lines = [
        "# WV3-710 BPR 新召回商品的第12名安全准入",
        "",
        "## 结论",
        "",
        f"内层门槛：**{'通过' if result['passed'] else '未通过'}**；相对 WV3-691 的平均增量 {gate['mean_delta_vs_WV3_691']:+.9f}。",
        "",
        "## 术语与固定策略",
        "",
        "- BPR-only candidate（本报告称 BPR 独有候选）：由全目录 BPR Top100 找到、但不在原候选池中的用户—商品对。BPR 是行业常用的贝叶斯个性化排序矩阵分解。",
        "- expanded-pool fused rank（本报告称扩池融合名次）：冻结原 E0/E1 排序器，在原池与 BPR 独有候选的并集上分别排名，再以固定 RRF60 合并得到的用户内名次；RRF 是行业常用的倒数名次融合。",
        "- 第12名安全准入（本项目自定义）：只有 BPR 独有候选已经进入扩池融合 Top12 时，才把其中最高者放到当前第12名；第1–11名完全不变，每位用户最多加入一件。",
        "- 同池对照（本项目评测约束）：候选并集完全相同，但关闭新增商品准入；其 Top12 必须与 WV3-691 一致，因此新增候选本身不会偷偷改变对照排序。",
        "- 第12名 Oracle（本项目诊断上界）：若知道真实标签，从 BPR 独有候选中挑一件放在第12名可取得的最大 AP@12 增量；分母为窗口全部真值用户，不能部署。",
        "",
        "| 内层窗口 | 被准入用户 | 有益 / 有害 / 中性 | 相对 WV3-691 | 第12名 Oracle 空间 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['bpr_new_rank12_or_better_users']:,} | {row['beneficial_selected_users']:,} / "
            f"{row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | "
            f"{row['incremental_population_delta_vs_WV3_691']:+.9f} | "
            f"{row['rank12_only_oracle_population_headroom_vs_WV3_691']:+.9f} |"
        )
    lines += [
        "",
        "本轮复用 WV3-401 已构建的内层扩池与冻结模型分数，不重训模型。候选池虽扩展，但同池禁用准入对照严格复现 WV3-691。失败后不得按内层结果改成第8–11名、Top20 或多次准入。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(10)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            contract()["architecture_family"],
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_protocol"],
            training_protocol="no new fit; frozen original E0/E1 and existing cutoff-safe BPR Top100",
            params={"score_protocol": contract()["score_protocol"], "action": contract()["action"], "inner_gate": contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=10,
        )
    started = time.perf_counter()
    windows = {name: evaluate_window(name, folder) for name, folder in INNER}
    delta = {name: row["incremental_population_delta_vs_WV3_691"] for name, row in windows.items()}
    oracle = {name: row["rank12_only_oracle_population_headroom_vs_WV3_691"] for name, row in windows.items()}
    rules = contract()["inner_gate"]
    screening = {
        "per_window_delta_vs_WV3_691": delta,
        "mean_delta_vs_WV3_691": float(np.mean(list(delta.values()))),
        "nondegrade_windows_vs_WV3_691": sum(value >= 0 for value in delta.values()),
        "worst_delta_vs_WV3_691": min(delta.values()),
        "mean_rank12_only_oracle_headroom": float(np.mean(list(oracle.values()))),
        "minimum_rank12_only_oracle_headroom": min(oracle.values()),
    }
    screening["passed"] = bool(
        screening["mean_rank12_only_oracle_headroom"] >= rules["rank12_only_oracle_mean_min"]
        and screening["minimum_rank12_only_oracle_headroom"] >= rules["rank12_only_oracle_each_window_min"]
        and screening["mean_delta_vs_WV3_691"] > rules["incremental_mean_vs_WV3_691_gt"]
        and screening["nondegrade_windows_vs_WV3_691"] >= rules["nondegrade_windows_vs_WV3_691_min"]
        and screening["worst_delta_vs_WV3_691"] >= rules["worst_delta_vs_WV3_691_min"]
    )
    standard = screening_gate(row["MAP@12"] - row["same_pool_control_MAP@12"] + read(common.REPORT / "WV3-691_SCREEN.json")["windows"][name]["population_delta"] for name, row in windows.items())
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "screening": screening,
        "standard_screening_vs_WV2_601": standard,
        "passed": screening["passed"] and standard["passed"],
        "candidate_pool_changed": True,
        "same_pool_control": "expanded pool with BPR admission disabled; exact WV3-691 Top12",
        "runtime_seconds": time.perf_counter() - started,
        "new_training": False,
        "outer_exposure": 0,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="inner_pass" if result["passed"] else "reject_inner",
        inner_evidence={"screening": screening, "standard": standard, "passed": result["passed"]},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN)],
    )
    common.log(
        f"{TRIAL} BPR retrieval safe-admission screen",
        "WV3-401 gained new BPR Top12 truth but broad expanded-pool reordering lost more shared/original truth; WV3-691 now protects the head but cannot add candidates outside the original pool.",
        contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"mean incremental={screening['mean_delta_vs_WV3_691']:+.9f}; nondegrade={screening['nondegrade_windows_vs_WV3_691']}/4; worst={screening['worst_delta_vs_WV3_691']:+.9f}; oracle mean={screening['mean_rank12_only_oracle_headroom']:+.9f}; pass={result['passed']}.",
        "Freeze and build one outer confirmation." if result["passed"] else "Reject this model-free source admission; do not tune rank cutoff or insertion slot.",
        "If passed, build only the four required outer expanded pools and expose once; otherwise audit a learned source-specific score only if its supervision can be built historically.",
        alternatives="Full-pool retraining WV3-401 and Top100 generic pointwise expansion WV3-641 already failed or had poor cost-benefit; this policy changes only one tail slot.",
        experiment="One inference-safe BPR-only admission at rank12 when the frozen expanded-pool fusion already ranks it in Top12.",
        reflection="This isolates retrieval value from broad rank displacement and uses an exact same-expanded-pool no-admission control.",
    )
    print({"trial": TRIAL, "passed": result["passed"], "screening": screening}, flush=True)
    return result


if __name__ == "__main__":
    screen()
