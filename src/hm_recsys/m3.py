from __future__ import annotations

import ctypes
import gc
import json
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctypes import wintypes

import numpy as np

from .m2 import M2Config, _sha256, _write_json, build_category_maps
from .m21 import CandidateWindow
from .m26 import _evaluate_models, _save_category_maps
from .m29 import (
    M29Window,
    _attach_popularity_segments,
    _read_json,
    build_expanded_candidate_cache,
    build_feature_cache,
    build_source_cache,
)
from .m210 import (
    _file_identity,
    _target_feature_paths,
    build_target_aware_cache,
    score_models,
)
from .m211 import SAMPLING_SEED, _load_category_maps, load_distribution_sample, train_lambdarank
from .m212 import feature_sets, load_inner_validation, train_inner_ranker


SCHEMA_VERSION = "m3.0-three-window-robustness-v1"
ALL_CUTOFFS = (
    "2020-04-29",
    "2020-05-27",
    "2020-06-24",
    "2020-07-22",
    "2020-08-19",
)
ROLLING_PROTOCOL = {
    "roll_20200624": {
        "inner_train": ["2020-04-29"],
        "inner_validation": "2020-05-27",
        "outer_train": ["2020-04-29", "2020-05-27"],
        "outer_validation": "2020-06-24",
    },
    "roll_20200722": {
        "inner_train": ["2020-05-27"],
        "inner_validation": "2020-06-24",
        "outer_train": ["2020-05-27", "2020-06-24"],
        "outer_validation": "2020-07-22",
    },
    "roll_20200819": {
        "inner_train": ["2020-06-24"],
        "inner_validation": "2020-07-22",
        "outer_train": ["2020-06-24", "2020-07-22"],
        "outer_validation": "2020-08-19",
    },
}
ANCHOR_NAME = "no_decay_anchor"
FALLBACK_NAME = f"{ANCHOR_NAME}__inactive_rrf"


def validate_rolling_protocol() -> None:
    if len(ROLLING_PROTOCOL) != 3:
        raise RuntimeError("M3.0 requires exactly three rolling outer windows")
    validations: list[str] = []
    for name, row in ROLLING_PROTOCOL.items():
        sequence = [*row["inner_train"], row["inner_validation"], row["outer_validation"]]
        if sequence != sorted(sequence) or len(sequence) != len(set(sequence)):
            raise RuntimeError(f"{name} is not strictly temporal")
        if row["outer_train"] != [*row["inner_train"], row["inner_validation"]]:
            raise RuntimeError(f"{name} outer refit does not extend inner training")
        validations.append(row["outer_validation"])
    if validations != sorted(validations) or len(set(validations)) != 3:
        raise RuntimeError("M3.0 outer validation cutoffs must be unique and ordered")
    if any(cutoff >= "2020-09-16" for cutoff in ALL_CUTOFFS):
        raise RuntimeError("M3.0 must not read the final validation week")


def _peak_working_set_bytes() -> int | None:
    if os.name != "nt":
        try:
            import resource

            value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return value * (1 if os.uname().sysname == "Darwin" else 1024)
        except (ImportError, AttributeError, OSError):
            return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    function = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    function.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    function.restype = wintypes.BOOL
    ok = function(
        process, ctypes.byref(counters), counters.cb
    )
    return int(counters.PeakWorkingSetSize) if ok else None


def _stage_call(timers: dict[str, float], name: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timers[name] = time.perf_counter() - started


def _model_artifacts(development: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for window in development.values():
        for section in ("inner_model", "outer_model"):
            model = window[section]
            artifacts.append(_file_identity(Path(model["model_path"])))
        scoring = window["scoring"]
        artifacts.append(_file_identity(Path(scoring["prediction_path"])))
        artifacts.append(_file_identity(Path(scoring["evaluation_db"])))
        for section in ("inner_category_encoding", "outer_category_encoding"):
            artifacts.append(_file_identity(Path(window[section]["path"])))
    return artifacts


def summarize_m3(
    development: dict[str, Any],
    m212_metrics: dict[str, Any],
    metric_k: int,
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    orderings: dict[str, Any] = {}
    for ordering in ("expanded_append_order", ANCHOR_NAME, FALLBACK_NAME):
        values = {
            name: float(row["evaluation"]["orderings"][ordering]["segments"]["overall"][metric])
            for name, row in development.items()
        }
        numeric = np.asarray(list(values.values()), dtype=np.float64)
        orderings[ordering] = {
            "window_map@12": values,
            "mean_map@12": float(numeric.mean()),
            "std_map@12": float(numeric.std(ddof=0)),
            "min_map@12": float(numeric.min()),
            "max_map@12": float(numeric.max()),
        }

    anchor_values = orderings[FALLBACK_NAME]["window_map@12"]
    retrieval_values = orderings["expanded_append_order"]["window_map@12"]
    deltas = {name: anchor_values[name] - retrieval_values[name] for name in development}
    segment_map: dict[str, dict[str, float]] = {}
    for segment in ("overall", "warm", "cold"):
        segment_map[segment] = {
            name: float(row["evaluation"]["orderings"][FALLBACK_NAME]["segments"][segment][metric])
            for name, row in development.items()
        }

    activity: dict[str, dict[str, float]] = {}
    popularity: dict[str, dict[str, float]] = {}
    ceilings: dict[str, dict[str, float]] = {}
    for name, row in development.items():
        ordering = row["evaluation"]["orderings"][FALLBACK_NAME]
        for segment in ordering["activity_segments"]:
            activity.setdefault(segment["activity_segment"], {})[name] = float(segment[metric])
        for segment_name, segment in ordering["popularity_segments"].items():
            popularity.setdefault(segment_name, {})[name] = float(segment[metric])
        overall = ordering["segments"]["overall"]
        ceilings[name] = {
            "candidate_recall@expanded_pool": float(overall["candidate_recall@expanded_pool"]),
            "candidate_hit_rate@expanded_pool": float(overall["candidate_hit_rate@expanded_pool"]),
            "oracle_map@12": float(overall[f"oracle_map@{metric_k}"]),
            "map@12": float(overall[metric]),
            "oracle_realization_fraction": (
                float(overall[metric]) / float(overall[f"oracle_map@{metric_k}"])
                if float(overall[f"oracle_map@{metric_k}"]) > 0
                else None
            ),
        }

    parity = {}
    prior_mapping = {
        "roll_20200722": "dev_a",
        "roll_20200819": "dev_b",
    }
    for current, prior in prior_mapping.items():
        prior_row = m212_metrics["development"][prior]
        expected = float(
            prior_row["evaluation"]["orderings"][FALLBACK_NAME]["segments"]["overall"][metric]
        )
        actual = anchor_values[current]
        expected_round = int(prior_row["outer_models"][ANCHOR_NAME]["selected_rounds_from_inner"])
        actual_round = int(development[current]["outer_model"]["selected_rounds_from_inner"])
        parity[current] = {
            "prior_m2_12_window": prior,
            "expected_map@12": expected,
            "actual_map@12": actual,
            "absolute_difference": abs(actual - expected),
            "expected_round": expected_round,
            "actual_round": actual_round,
            "passed": abs(actual - expected) <= 1e-12 and expected_round == actual_round,
        }

    return {
        "orderings": orderings,
        "anchor_delta_vs_expanded_append_order_map@12": deltas,
        "stability_gate": {
            "accepted": all(value > 0 for value in deltas.values()),
            "rule": "fixed no-decay anchor with exact inactive RRF fallback beats expanded append order in all three rolling windows",
        },
        "segment_map@12": segment_map,
        "activity_segment_map@12": activity,
        "popularity_segment_map@12": popularity,
        "candidate_ceiling": ceilings,
        "m2_12_reproduction": {
            "windows": parity,
            "all_passed": all(row["passed"] for row in parity.values()),
        },
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    windows = list(result["development"])
    rows = [
        "# M3.0：三窗口滚动稳定性验证",
        "",
        "## 结论",
        "",
        (
            "固定 no-decay LambdaRank + exact inactive RRF fallback "
            + ("通过" if summary["stability_gate"]["accepted"] else "未通过")
            + "预注册的三窗口稳定性门槛。M3.0 只验证固定主体，不据此重新搜索召回源、特征或参数。"
        ),
        "",
        "## 协议",
        "",
        "- 候选池：six-source Top100 + up to 200 Item2Vec-only；10% hash development users。",
        "- 排序器：固定 no-decay 21 个 target-aware 特征、distribution-aware unweighted 30:1、LambdaRank。",
        "- 每个 outer window 仅用更早 inner validation 选择最多 200 轮内的 boosting round。",
        "- inactive_12w 严格按 candidate_rank（原 RRF/append 顺序）兜底；active 用户按模型分数。",
        "- optimistic all-articles catalog；行为聚合严格 t_dat < cutoff；final week 未运行。",
        "",
        "## Overall MAP@12",
        "",
        "| ordering | " + " | ".join(windows) + " | mean | std | min |",
        "|---|" + "---:|" * (len(windows) + 3),
    ]
    for name in ("expanded_append_order", ANCHOR_NAME, FALLBACK_NAME):
        row = summary["orderings"][name]
        rows.append(
            "| " + name + " | "
            + " | ".join(f"{row['window_map@12'][window]:.6f}" for window in windows)
            + f" | {row['mean_map@12']:.6f} | {row['std_map@12']:.6f} | {row['min_map@12']:.6f} |"
        )
    rows.extend([
        "",
        "相对 expanded append order 的逐窗增量："
        + "/".join(f"{summary['anchor_delta_vs_expanded_append_order_map@12'][window]:+.6f}" for window in windows)
        + "。",
        "",
        "## 候选上限与兑现率",
        "",
        "| window | candidate Recall | HitRate | Oracle MAP@12 | MAP@12 | MAP/Oracle |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for window in windows:
        row = summary["candidate_ceiling"][window]
        rows.append(
            f"| {window} | {row['candidate_recall@expanded_pool']:.6f} | "
            f"{row['candidate_hit_rate@expanded_pool']:.6f} | {row['oracle_map@12']:.6f} | "
            f"{row['map@12']:.6f} | {row['oracle_realization_fraction']:.2%} |"
        )
    rows.extend(["", "## 分群 MAP@12", ""])
    for title, key in (
        ("Warm / cold", "segment_map@12"),
        ("活跃度", "activity_segment_map@12"),
        ("商品流行度", "popularity_segment_map@12"),
    ):
        rows.extend([
            f"### {title}",
            "",
            "| segment | " + " | ".join(windows) + " |",
            "|---|" + "---:|" * len(windows),
        ])
        for segment, values in summary[key].items():
            rows.append(
                "| " + segment + " | "
                + " | ".join(f"{values[window]:.6f}" for window in windows)
                + " |"
            )
        rows.append("")
    resources = result["resources"]
    rows.extend([
        "## 复现与资源",
        "",
        f"- M2.12 两个重叠窗口精确复现：`{summary['m2_12_reproduction']['all_passed']}`。",
        f"- wall time：{result['elapsed_seconds']:.2f} 秒；进程 peak working set："
        + (f"{resources['peak_working_set_bytes'] / 1024**3:.2f} GiB。" if resources["peak_working_set_bytes"] else "未能读取。"),
        f"- 模型/预测/评测库/category-map 产物：{resources['artifact_count']} 个，"
        f"合计 {resources['artifact_bytes'] / 1024**2:.2f} MiB，均记录 path/bytes/SHA256。",
        "",
        "## 证据边界与下一步",
        "",
        "- 06-24 曾作为 M2.12 inner validation，07-22/08-19 也被反复观察；这是更严格的滚动开发证据，不是 pristine blind test。",
        "- M3.0 没有做 source/feature ablation，因此不能把它写成完整 M3 收尾。下一步 M3.1 在同三窗协议下做有界召回源和特征组消融。",
        "- 当前 MAP 与 Oracle 仍有明显差距；稳定性通过只说明这条 anchor 值得继续作为比较基座，不代表效果已足够。",
        "- final week 未运行。",
        "",
        "机器可读证据：`metrics.json`。",
        "",
    ])
    return "\n".join(rows)


def run_m3(
    *,
    raw_dir: Path,
    work_dir: Path,
    transactions_path: Path,
    m21_metrics_path: Path,
    m28_metrics_path: Path,
    m212_metrics_path: Path,
    windows: list[CandidateWindow],
    source_cache_dir: Path,
    m29_cache_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    validate_rolling_protocol()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M3.0 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    by_cutoff = {window.cutoff: window for window in windows}
    if tuple(sorted(by_cutoff)) != ALL_CUTOFFS:
        raise ValueError(f"M3.0 requires baseline candidate cutoffs {ALL_CUTOFFS}")
    m212 = _read_json(m212_metrics_path.resolve())
    if m212.get("schema_version") != "m2.12-inner-temporal-selection-v1" or m212.get("status") != "measured":
        raise ValueError("M3.0 requires measured M2.12 evidence")
    if m212["contract"].get("final_week") != "not_run":
        raise ValueError("M2.12 final-week boundary is not intact")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    timers: dict[str, float] = {}
    try:
        m29_windows = [
            M29Window(window.cutoff, window.candidate_path, window.manifest_path)
            for window in windows
        ]
        print("M3.0: validating/building five cutoff-safe Item2Vec source caches", flush=True)
        sources = _stage_call(
            timers,
            "source_cache",
            build_source_cache,
            windows=m29_windows,
            transactions_path=transactions_path,
            m28_metrics_path=m28_metrics_path,
            source_cache_dir=source_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        print("M3.0: validating/building expanded candidate caches", flush=True)
        candidates = _stage_call(
            timers,
            "expanded_candidate_cache",
            build_expanded_candidate_cache,
            windows=m29_windows,
            sources=sources,
            cache_dir=m29_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        base_features = _stage_call(
            timers,
            "base_feature_cache",
            build_feature_cache,
            raw_dir=raw_dir,
            work_dir=work_dir,
            cache_dir=m29_cache_dir,
            candidates=candidates,
            config=config,
            cutoffs=ALL_CUTOFFS,
        )
        print("M3.0: validating/building target-aware no-decay caches", flush=True)
        target_features = _stage_call(
            timers,
            "target_feature_cache",
            build_target_aware_cache,
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            m29_cache_dir=m29_cache_dir,
            cache_dir=feature_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        features = feature_sets()[ANCHOR_NAME]
        development: dict[str, Any] = {}
        for name, protocol in ROLLING_PROTOCOL.items():
            print(f"M3.0 {name}: inner selection", flush=True)
            window_timers: dict[str, float] = {}
            window_dir = artifact_dir / name
            window_dir.mkdir()
            inner_frame, inner_sizes, inner_sampling = _stage_call(
                window_timers,
                "inner_sample_load",
                load_distribution_sample,
                [_target_feature_paths(feature_cache_dir, cutoff)[0] for cutoff in protocol["inner_train"]],
                features,
                seed=SAMPLING_SEED,
            )
            inner_maps = build_category_maps(inner_frame)
            inner_map_evidence = _save_category_maps(window_dir / "inner-category-maps.json", inner_maps)
            inner_validation, inner_validation_sizes, truth_counts, validation_evidence = _stage_call(
                window_timers,
                "inner_validation_load",
                load_inner_validation,
                dataset_path=_target_feature_paths(feature_cache_dir, protocol["inner_validation"])[0],
                transactions_path=transactions_path,
                cutoff=protocol["inner_validation"],
                features=features,
            )
            inner_model, inner_evidence = _stage_call(
                window_timers,
                "inner_fit",
                train_inner_ranker,
                train_frame=inner_frame,
                train_group_sizes=inner_sizes,
                train_evidence=inner_sampling,
                validation_frame=inner_validation,
                validation_group_sizes=inner_validation_sizes,
                validation_truth_counts=truth_counts,
                validation_evidence=validation_evidence,
                features=features,
                name=ANCHOR_NAME,
                artifact_dir=window_dir,
                config=config,
                category_maps=inner_maps,
            )
            del inner_model, inner_frame, inner_sizes, inner_validation, inner_validation_sizes
            gc.collect()

            print(f"M3.0 {name}: outer refit and complete scoring", flush=True)
            outer_frame, outer_sizes, outer_sampling = _stage_call(
                window_timers,
                "outer_sample_load",
                load_distribution_sample,
                [_target_feature_paths(feature_cache_dir, cutoff)[0] for cutoff in protocol["outer_train"]],
                features,
                seed=SAMPLING_SEED,
            )
            outer_maps = build_category_maps(outer_frame)
            outer_map_evidence = _save_category_maps(window_dir / "outer-category-maps.json", outer_maps)
            rounds = int(inner_evidence["best_iteration"])
            model, outer_evidence = _stage_call(
                window_timers,
                "outer_fit",
                train_lambdarank,
                frame=outer_frame,
                group_sizes=outer_sizes,
                group_evidence=outer_sampling,
                features=features,
                name=ANCHOR_NAME,
                use_ipw=False,
                artifact_dir=window_dir,
                config=replace(config, num_boost_round=rounds),
                category_maps=outer_maps,
            )
            outer_evidence["selected_rounds_from_inner"] = rounds
            outer_evidence["inner_selection_score"] = inner_evidence["best_score"]
            del outer_frame, outer_sizes
            gc.collect()
            scoring = _stage_call(
                window_timers,
                "outer_scoring",
                score_models,
                dataset_path=_target_feature_paths(feature_cache_dir, protocol["outer_validation"])[0],
                models={ANCHOR_NAME: (model, features, outer_maps)},
                evaluation_db=window_dir / "evaluation.duckdb",
                prediction_path=window_dir / "outer-validation-predictions.parquet",
                config=config,
            )
            del model
            gc.collect()
            evaluation = _stage_call(
                window_timers,
                "outer_evaluation",
                _evaluate_models,
                evaluation_db=window_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                metric_k=config.metric_k,
                variant_names=[ANCHOR_NAME],
            )
            evaluation = _stage_call(
                window_timers,
                "popularity_evaluation",
                _attach_popularity_segments,
                evaluation=evaluation,
                evaluation_db=window_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                variant_names=[ANCHOR_NAME],
                budget=config.candidate_k,
            )
            development[name] = {
                "protocol": protocol,
                "inner_model": inner_evidence,
                "inner_category_encoding": inner_map_evidence,
                "outer_sampling": outer_sampling,
                "outer_model": outer_evidence,
                "outer_category_encoding": outer_map_evidence,
                "scoring": scoring,
                "evaluation": evaluation,
                "timings_seconds": window_timers,
            }

        summary = summarize_m3(development, m212, config.metric_k)
        if not summary["m2_12_reproduction"]["all_passed"]:
            raise RuntimeError("M3.0 failed exact M2.12 overlap-window reproduction")
        artifacts = _model_artifacts(development)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.0",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "three_rolling_development_windows_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "six-source Top100 plus up to 200 Item2Vec-only",
                "catalog_protocol": "optimistic_all_articles",
                "sampling": "M2.11 unweighted distribution-aware source-rank-bucket hash 30x",
                "features": features,
                "removed_decay_features": True,
                "model": "LightGBM LambdaRank",
                "fallback": "user_history_events_12w = 0 uses exact candidate_rank/RRF order; otherwise model score",
                "rolling_protocol": ROLLING_PROTOCOL,
                "selection_scope": "boosting round only on earlier inner validation",
                "source_and_feature_ablation": "deferred_to_M3.1_not_run",
                "final_week": "not_run",
            },
            "inputs": {
                "m21_metrics": _file_identity(m21_metrics_path),
                "m28_metrics": _file_identity(m28_metrics_path),
                "m212_metrics": _file_identity(m212_metrics_path),
                "transactions": _file_identity(transactions_path),
                "baseline_candidates": {
                    cutoff: {
                        "candidate": _file_identity(window.candidate_path),
                        "manifest": _file_identity(window.manifest_path),
                    }
                    for cutoff, window in by_cutoff.items()
                },
            },
            "cache": {
                "item2vec_sources": sources,
                "expanded_candidates": candidates,
                "base_features": base_features,
                "target_features": target_features,
            },
            "development": development,
            "summary": summary,
            "resources": {
                "stage_timings_seconds": timers,
                "peak_working_set_bytes": _peak_working_set_bytes(),
                "artifact_count": len(artifacts),
                "artifact_bytes": int(sum(item["bytes"] for item in artifacts)),
                "artifacts": artifacts,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "artifacts": {},
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        failure = output_dir / f"failure-{time.time_ns()}.json"
        _write_json(
            failure,
            {
                "schema_version": "m3.0-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
