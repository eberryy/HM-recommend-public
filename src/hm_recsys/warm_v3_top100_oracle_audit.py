"""WV3-640: read-only marginal one-swap oracle audit for challengers at ranks 51-100."""
from __future__ import annotations

import time

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal
from .warm_v3_residual_admission import INNER, source
from .warm_v3_residual_admission_audit import apk_from_targets


TRIAL = "WV3-640"
CONTRACT = common.REPORT / "WV3-640_TOP100_ORACLE_AUDIT_CONTRACT.json"
OUTPUT = common.REPORT / "WV3-640_TOP100_ORACLE_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-640_TOP100_ORACLE_AUDIT.md"


def best_delta(targets: np.ndarray, truth_count: int, challenger_high: int) -> float:
    values = np.asarray(targets, dtype=np.uint8)
    baseline = apk_from_targets(values, truth_count)
    if not values[12:challenger_high].any() or values[7:12].all():
        return 0.0
    best = baseline
    challenger = int(np.flatnonzero(values[12:challenger_high] == 1)[0] + 12)
    for victim in np.flatnonzero(values[7:12] == 0) + 7:
        changed = values.copy()
        changed[victim], changed[challenger] = changed[challenger], changed[victim]
        best = max(best, apk_from_targets(changed, truth_count))
    return best - baseline


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "read_only_top100_marginal_oracle_audit",
        "hypothesis": "Ranks51-100 contain enough users whose only admissible tail truth lies below rank50 to justify doubling propensity training and inference support.",
        "inputs": "four completed 2020 inner rank files only",
        "comparison": "exact no-op-or-one-swap oracle with challengers13-50 versus challengers13-100; victims8-12 and protected ranks1-7 unchanged",
        "gate": {
            "mean_incremental_oracle_delta_min": 0.001,
            "each_window_incremental_oracle_delta_min": 0.0003,
            "each_window_newly_beneficial_users_min": 100,
            "rationale": "WV3-631 realizes about four percent of Top50 inner oracle headroom; 0.001 extra oracle projects only about 0.00004 MAP and is the minimum worth doubling support",
        },
        "training": "none",
        "new_outer_exposure": 0,
        "expected_minutes": 5,
        "fallback": "If the gate fails, keep Top50 and do not pay for Top100 propensity training.",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(5)
    champion = read(common.REPORT / "WV3-631_OUTER.json")
    assert champion["better_than_highest_mean_candidate"] and champion["final_week"] == "not_run"
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "top100_marginal_one_swap_oracle_audit",
            experiment_contract()["hypothesis"],
            candidate_protocol="read-only inner comparison; ranks1-7 fixed, victims8-12, challengers13-50 versus13-100",
            training_protocol="none; no outer labels",
            params={"gate": experiment_contract()["gate"], "final_week": "not_run"},
            expected_minutes=5,
        )


def run() -> dict:
    common.setup()
    common.budget(5)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        path, meta = source(folder)
        with connection() as con:
            frame = con.execute(
                f"""SELECT customer_id,ap_rf baseline_rank,target,truth_count,user_history_events_12w
                FROM read_parquet({literal(path)}) WHERE ap_rf<=100
                ORDER BY customer_id,ap_rf,article_id"""
            ).fetchdf()
        delta50 = 0.0
        delta100 = 0.0
        beneficial50 = 0
        beneficial100 = 0
        newly_beneficial = 0
        marginal_truth = 0
        for _, group in frame.groupby("customer_id", sort=False):
            values = group.target.to_numpy(np.uint8)
            if len(values) < 100 or int(group.user_history_events_12w.iloc[0]) <= 0:
                continue
            truth_count = int(group.truth_count.iloc[0])
            d50 = best_delta(values, truth_count, 50)
            d100 = best_delta(values, truth_count, 100)
            delta50 += d50
            delta100 += d100
            beneficial50 += d50 > 1e-15
            beneficial100 += d100 > 1e-15
            newly_beneficial += d50 <= 1e-15 and d100 > 1e-15
            marginal_truth += int(values[50:100].sum())
        row = {
            "window": name,
            "cutoff": meta["cutoff"],
            "total_users_denominator": meta["total_users"],
            "top50_oracle_delta": delta50 / meta["total_users"],
            "top100_oracle_delta": delta100 / meta["total_users"],
            "incremental_oracle_delta": (delta100 - delta50) / meta["total_users"],
            "top50_beneficial_users": beneficial50,
            "top100_beneficial_users": beneficial100,
            "newly_beneficial_users_from_ranks51_100": newly_beneficial,
            "truth_user_item_pairs_at_ranks51_100": marginal_truth,
            "baseline_ranks1_7_changes": 0,
            "maximum_swaps_per_user": 1,
            "final_week": "not_run",
        }
        windows[name] = row
        print({"top100_oracle": name, "incremental": row["incremental_oracle_delta"], "new_users": newly_beneficial}, flush=True)
    values = [row["incremental_oracle_delta"] for row in windows.values()]
    gate = experiment_contract()["gate"]
    passed = bool(
        np.mean(values) >= gate["mean_incremental_oracle_delta_min"]
        and all(value >= gate["each_window_incremental_oracle_delta_min"] for value in values)
        and all(row["newly_beneficial_users_from_ranks51_100"] >= gate["each_window_newly_beneficial_users_min"] for row in windows.values())
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "mean_incremental_oracle_delta": float(np.mean(values)),
        "minimum_incremental_oracle_delta": min(values),
        "top100_propensity_authorized": passed,
        "runtime_seconds": time.perf_counter() - started,
        "new_training": False,
        "new_outer_exposure": 0,
        "final_week": "not_run",
    }
    write(OUTPUT, result)
    lines = [
        "# WV3-640 Top100 尾部边际上限审计",
        "",
        "## 结论",
        "",
        f"Top100 购买倾向实验授权门槛：**{'pass' if passed else 'fail'}**。第51–100名带来的平均额外一换一 Oracle MAP@12 上限为 {result['mean_incremental_oracle_delta']:+.9f}。",
        "",
        "## 术语与口径",
        "",
        "- Top100（本项目自定义候选预算）：基线融合排序的前100件商品；现稳定方案 WV3-631 只让第13–50名挑战第8–12名，本审计把挑战商品末位扩到100。",
        "- 边际一换一 Oracle（本项目自定义理想上限）：Top100 一换一 Oracle MAP 增益减 Top50 一换一 Oracle MAP 增益；两者都允许每位活跃用户不操作或最多交换一次，且第1–7名不变。",
        "- 新增可受益用户（本项目自定义）：第13–50名没有任何有益挑战商品、但第51–100名至少有一件能提高 AP@12 的用户。",
        "- 真值用户—商品对：该用户在监督周实际购买且落在指定名次范围的候选；表中按每个窗口独立计数。",
        "",
        "| 内层窗口 | Top50 上限 | Top100 上限 | 额外上限 | 新增可受益用户 | 第51–100名真值对 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        lines.append(f"| {name} | {row['top50_oracle_delta']:+.9f} | {row['top100_oracle_delta']:+.9f} | {row['incremental_oracle_delta']:+.9f} | {row['newly_beneficial_users_from_ranks51_100']:,} | {row['truth_user_item_pairs_at_ranks51_100']:,} |")
    lines += [
        "",
        "门槛要求平均额外上限至少0.001、每窗至少0.0003且每窗至少100名新增可受益用户。原因是 WV3-631 只兑现约4%的 Top50 内层上限；低于该量级时，即使维持当前兑现率，双倍候选成本也只能换来不足约0.00004 MAP。",
        "",
        "本审计没有训练、没有读取外层。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    common.update(TRIAL, decision="diagnostic_supports_top100" if passed else "diagnostic_rejects_top100", inner_evidence={"mean_incremental_oracle_delta": result["mean_incremental_oracle_delta"], "minimum_incremental_oracle_delta": result["minimum_incremental_oracle_delta"], "authorized": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(OUTPUT), str(MARKDOWN)])
    common.log(
        f"{TRIAL} Top100 marginal oracle audit",
        "WV3-631 is stable in all outer windows but still below the 0.03 target; expanding candidates is only justified if ranks51-100 add material one-swap headroom.",
        experiment_contract()["hypothesis"],
        str(OUTPUT),
        f"mean incremental oracle={result['mean_incremental_oracle_delta']:+.9f}; minimum={result['minimum_incremental_oracle_delta']:+.9f}; authorized={passed}.",
        "Preregister one Top100 propensity variant." if passed else "Keep Top50; do not pay for Top100 training.",
        "Only expand the already stable pointwise architecture if the fixed cost-benefit gate passes.",
        alternatives="Multi-swap and broad reranking were not attempted because their AP effects interact and previous broad rerankers were unstable.",
        experiment="Read-only exact Top50 versus Top100 one-swap oracle on four inner windows.",
        reflection="The gate converts candidate expansion into a measurable expected-return decision rather than assuming a larger budget helps.",
    )
    print({"trial": TRIAL, "mean_incremental": result["mean_incremental_oracle_delta"], "authorized": passed}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
