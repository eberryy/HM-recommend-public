from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .m4_contract import FINAL_CUTOFF, file_identity
from .p37b import (
    RUN_ID,
    WINDOWS,
    _b1_vs_b0_selection,
    _complexity_gate,
    _final_decision,
    _formal_gate,
)
from .p37b_report import REQUIRED_OUTPUTS


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_VARIANTS = ("C0", "C1", "B0", "B1")
_HYBRIDS = ("B0", "B1")
_METRIC_TOLERANCE = 1e-12


def _close(left: Any, right: Any, *, tolerance: float = _METRIC_TOLERANCE) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and math.isclose(left_value, right_value, rel_tol=tolerance, abs_tol=tolerance)
    )


def _candidate_identity_valid(
    metrics: dict[str, Any], candidate_audit: dict[str, Any]
) -> bool:
    """Validate persisted Top200 identity, conservation, and cutoff-safety evidence."""

    audit_windows = candidate_audit.get("windows", {})
    if (
        candidate_audit.get("status") != "measured"
        or candidate_audit.get("final_week") != "not_run"
        or candidate_audit.get("all_windows_exact") is not True
        or set(audit_windows) != set(WINDOWS)
    ):
        return False
    for window in WINDOWS:
        outer = metrics.get("windows", {}).get(window, {})
        persisted = outer.get("candidate_identity", {})
        audit = audit_windows.get(window, {})
        if persisted != audit:
            return False
        if not all(
            audit.get(field) is True
            for field in (
                "all_variants_exact",
                "candidate_identity_exact",
                "candidate_rows_exact",
                "recall200_exact",
                "top200_truth_pairs_exact",
                "m4_cutoff_safe",
                "p33_cutoff_safe",
            )
        ):
            return False
        variants = audit.get("variants", {})
        metric_variants = outer.get("variants", {})
        if set(variants) != set(_VARIANTS) or set(metric_variants) != set(_VARIANTS):
            return False
        reference: tuple[Any, ...] | None = None
        for variant in _VARIANTS:
            row = variants[variant]
            measured = metric_variants[variant]
            at_200 = measured.get("at_k", {}).get("200", {})
            ranking = measured.get("ranking", {})
            signature = (
                row.get("candidate_identity_sha256"),
                row.get("candidate_rows"),
                row.get("top200_truth_pairs"),
                row.get("strict_recall200"),
                row.get("sparse_recall200"),
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                return False
            if (
                not isinstance(row.get("candidate_identity_sha256"), str)
                or len(row["candidate_identity_sha256"]) != 64
                or row.get("per_user_rank_permutation_1_200") is not True
                or row.get("candidate_rows") != at_200.get("candidate_rows")
                or row.get("top200_truth_pairs") != at_200.get("positive_rows")
                or row.get("top200_truth_pairs")
                != ranking.get("coarse_positive_pairs_top200")
                or not _close(
                    row.get("strict_recall200"),
                    at_200.get("segments", {}).get("strict_cold", {}).get("recall"),
                )
                or not _close(
                    row.get("sparse_recall200"),
                    at_200.get("segments", {}).get("sparse_1_5", {}).get("recall"),
                )
            ):
                return False
    return True


def _initial_residual_valid(row: dict[str, Any]) -> bool:
    return (
        isinstance(row.get("rows"), int)
        and row["rows"] > 0
        and row.get("delta_max_abs") == 0.0
        and row.get("score_vs_raw_m4_max_abs") == 0.0
        and row.get("last_layer_weight_zero") is True
        and row.get("last_layer_bias_zero") is True
    )


def _initialization_pair_valid(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> bool:
    """Validate paired B0/B1 initialization when the optional audit is persisted."""

    if left is None and right is None:
        return True
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    common_fields = (
        "scheme",
        "seed",
        "common_relation_input_sha256",
        "shared_downstream_sha256",
    )
    if any(left.get(field) != right.get(field) for field in common_fields):
        return False
    return (
        left.get("scheme") == "paired-layerwise-v1"
        and isinstance(left.get("seed"), int)
        and all(
            isinstance(left.get(field), str) and len(left[field]) == 64
            for field in ("common_relation_input_sha256", "shared_downstream_sha256")
        )
        and left.get("auxiliary_relation_input_sha256") is None
        and isinstance(right.get("auxiliary_relation_input_sha256"), str)
        and len(right["auxiliary_relation_input_sha256"]) == 64
    )


def _pair_audit_valid(row: dict[str, Any]) -> bool:
    buckets = row.get("bucket_counts", {})
    return (
        isinstance(row.get("pair_array_sha256"), str)
        and len(row["pair_array_sha256"]) == 64
        and row.get("same_user_only") is True
        and row.get("all_buckets_covered") is True
        and row.get("maximum_negatives_per_positive") == 50
        and set(buckets) == {"hard", "medium", "easy"}
        and all(isinstance(value, int) and value > 0 for value in buckets.values())
        and sum(buckets.values()) == row.get("pair_rows")
    )


def _inner_trace_valid(inner: dict[str, Any]) -> bool:
    trace = inner.get("trace", [])
    selected_epoch = inner.get("selected_epoch")
    max_epochs = inner.get("max_epochs")
    patience = inner.get("patience_consecutive_nonimprovements")
    if (
        not trace
        or not isinstance(selected_epoch, int)
        or not isinstance(max_epochs, int)
        or not isinstance(patience, int)
        or selected_epoch < 1
        or len(trace) > max_epochs
        or [row.get("epoch") for row in trace] != list(range(1, len(trace) + 1))
    ):
        return False
    for row in trace:
        numeric = (
            row.get("bpr_loss"),
            row.get("pair_accuracy"),
            row.get("inner_density20"),
            row.get("inner_mrr"),
            row.get("inner_top200_to_top20"),
            row.get("scoring_seconds"),
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
            return False
        if not 0.0 <= float(row["pair_accuracy"]) <= 1.0 or row.get("pair_rows", 0) <= 0:
            return False
    best = max(trace, key=lambda row: (row["inner_density20"], row["inner_mrr"]))
    selected_key = inner.get("selected_key", {})
    if (
        best["epoch"] != selected_epoch
        or not _close(best["inner_density20"], selected_key.get("density20"))
        or not _close(best["inner_mrr"], selected_key.get("mrr"))
    ):
        return False
    if len(trace) < max_epochs and len(trace) - selected_epoch != patience:
        return False
    return _pair_audit_valid(inner.get("pair_audit", {}))


def _training_audit_valid(training: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Return (trace validity, pair sampling parity, paired-init validity)."""

    windows = training.get("windows", {})
    if (
        training.get("status") != "measured"
        or training.get("final_week") != "not_run"
        or set(windows) != set(WINDOWS)
    ):
        return False, False, False
    traces_valid = True
    pairs_valid = True
    initialization_valid = True
    initialization_presence: list[bool] = []
    for window in WINDOWS:
        variants = windows.get(window, {})
        if set(variants) != set(_HYBRIDS):
            return False, False, False
        for stage in ("inner", "outer"):
            left_init = variants["B0"].get(stage, {}).get("initialization")
            right_init = variants["B1"].get(stage, {}).get("initialization")
            initialization_presence.extend([left_init is not None, right_init is not None])
            initialization_valid &= _initialization_pair_valid(left_init, right_init)
            initialization_valid &= _initial_residual_valid(
                variants["B0"].get(stage, {}).get("initial_residual", {})
            )
            initialization_valid &= _initial_residual_valid(
                variants["B1"].get(stage, {}).get("initial_residual", {})
            )
        for variant in _HYBRIDS:
            inner = variants[variant].get("inner", {})
            outer = variants[variant].get("outer", {})
            traces_valid &= _inner_trace_valid(inner)
            outer_trace = outer.get("trace", [])
            selected_epoch = inner.get("selected_epoch")
            traces_valid &= (
                outer.get("epochs_from_inner_early_stopping") == selected_epoch
                and len(outer_trace) == selected_epoch
                and [row.get("epoch") for row in outer_trace]
                == list(range(1, len(outer_trace) + 1))
            )
            outer_audits = outer.get("pair_audits", {})
            pairs_valid &= (
                bool(outer_audits)
                and all(_pair_audit_valid(row) for row in outer_audits.values())
                and inner.get("training_cutoff") in outer_audits
                and inner.get("pair_audit") == outer_audits.get(inner.get("training_cutoff"))
            )
            expected_pairs = sum(row.get("pair_rows", 0) for row in outer_audits.values())
            pairs_valid &= all(row.get("pair_rows") == expected_pairs for row in outer_trace)
        left_inner = variants["B0"]["inner"].get("pair_audit", {})
        right_inner = variants["B1"]["inner"].get("pair_audit", {})
        pairs_valid &= left_inner == right_inner
        pairs_valid &= (
            variants["B0"]["outer"].get("pair_audits")
            == variants["B1"]["outer"].get("pair_audits")
        )
    if any(initialization_presence) and not all(initialization_presence):
        initialization_valid = False
    return bool(traces_valid), bool(pairs_valid), bool(initialization_valid)


def _p33_auxiliary_audit_valid(metrics: dict[str, Any], audit: dict[str, Any]) -> bool:
    audit_windows = audit.get("windows", {})
    if (
        audit.get("status") != "measured"
        or audit.get("promotion_metric") is not False
        or audit.get("final_week") != "not_run"
        or set(audit_windows) != set(WINDOWS)
    ):
        return False
    for window in WINDOWS:
        row = audit_windows.get(window, {})
        full = row.get("B1_full", {})
        masked = row.get("B1_p33_auxiliary_masked", {})
        difference = row.get("full_minus_masked", {})
        metric_b1 = metrics.get("windows", {}).get(window, {}).get("variants", {}).get("B1", {})
        at_20 = metric_b1.get("at_k", {}).get("20", {})
        ranking = metric_b1.get("ranking", {})
        fields = ("density20", "mrr", "top200_to_top20")
        if (
            not isinstance(row.get("definition"), str)
            or not row["definition"].strip()
            or row.get("promotion_metric") is not False
            or any(field not in full or field not in masked or field not in difference for field in fields)
            or not all(
                _close(difference[field], full[field] - masked[field]) for field in fields
            )
            or not _close(full.get("density20"), at_20.get("positive_density"))
            or not _close(full.get("mrr"), ranking.get("mrr"))
            or not _close(
                full.get("top200_to_top20"),
                ranking.get("conversion", {}).get("top200_to_top20"),
            )
        ):
            return False
        scoring = row.get("scoring", {})
        if (
            scoring.get("mask_p33_auxiliary") is not True
            or scoring.get("rows")
            != metric_b1.get("at_k", {}).get("200", {}).get("candidate_rows")
            or not isinstance(scoring.get("elapsed_seconds"), (int, float))
            or not math.isfinite(float(scoring["elapsed_seconds"]))
            or float(scoring["elapsed_seconds"]) < 0.0
        ):
            return False
    return True


def _feature_lineage_manifests_valid(
    manifest_rows: list[dict[str, Any]],
) -> bool:
    """Validate the 8 training + 4 outer feature manifests bound by the output manifest."""

    feature_rows = [
        row
        for row in manifest_rows
        if Path(str(row.get("path", ""))).name == "manifest.json"
        and "features-v1" in Path(str(row.get("path", ""))).parts
    ]
    if len(feature_rows) != 12:
        return False
    all_rows = {str(row.get("path")): row for row in manifest_rows}
    observed_keys: set[tuple[str, str]] = set()
    for row in feature_rows:
        path = Path(str(row["path"]))
        payload = _json(path)
        cutoff = payload.get("cutoff")
        lineage = payload.get("lineage_audit", {})
        sources = payload.get("source_artifacts", {})
        artifacts = payload.get("artifacts", {})
        cutoff_audit = lineage.get("cutoff", {})
        history = lineage.get("history_reconstruction", {})
        p33 = lineage.get("p33_embedding", {})
        if (
            payload.get("schema_version") != "phase3-p3.7b-raw-features-v1"
            or payload.get("status") != "completed"
            or payload.get("final_week") != "not_run"
            or not isinstance(cutoff, str)
            or cutoff >= FINAL_CUTOFF
            or row.get("cutoff_identity") != cutoff
            or cutoff_audit.get("cutoff") != cutoff
            or cutoff_audit.get("cutoff_safe") is not True
            or not isinstance(cutoff_audit.get("latest_behavior_before_cutoff"), str)
            or cutoff_audit["latest_behavior_before_cutoff"] >= cutoff
            or history.get("catalog_row_exact") is not True
            or history.get("days_exact") is not True
            or history.get("mask_exact") is not True
            or history.get("latest_behavior_before_cutoff") is not True
            or p33.get("content_only_inference") is not True
            or p33.get("passed") is not True
            or any(value > cutoff for value in p33.get("declared_training_cutoffs", []))
        ):
            return False
        for key in ("candidates", "histories", "users", "m4_embedding"):
            if lineage.get(key, {}).get("passed") is not True:
                return False
        source_lineage_pairs = (
            ("candidates", "candidates", "observed"),
            ("histories", "histories", "observed"),
            ("users", "users", "observed"),
            ("m4_embeddings", "m4_embedding", "observed"),
            ("p33_embeddings", "p33_embedding", "observed"),
            ("p33_manifest", "p33_embedding", "manifest"),
        )
        for source_key, lineage_key, identity_key in source_lineage_pairs:
            if sources.get(source_key) != lineage.get(lineage_key, {}).get(identity_key):
                return False
        if set(artifacts) != {
            "m4_relation",
            "p33_relation",
            "p33_global",
            "user_state",
            "candidate_state",
        }:
            return False
        for identity in artifacts.values():
            bound = all_rows.get(str(identity.get("path")))
            if bound is None or any(
                bound.get(field) != identity.get(field) for field in ("path", "bytes", "sha256")
            ):
                return False
            if bound.get("cutoff_identity") != cutoff:
                return False
        observed_keys.add((str(path.parent.parent.name), cutoff))
    return len(observed_keys) == 12


def verify_p37b(*, report_dir: Path) -> dict[str, Any]:
    report_dir = report_dir.resolve()
    metrics_path = report_dir / "P3_7B_metrics.json"
    manifest_path = report_dir / "P3_7B_OUTPUT_MANIFEST.json"
    metrics = _json(metrics_path)
    manifest = _json(manifest_path)
    candidate_audit = _json(report_dir / "p3_7b_candidate_identity_audit.json")
    training_audit = _json(report_dir / "p3_7b_training_audit.json")
    p33_auxiliary_audit = _json(report_dir / "p3_7b_p33_auxiliary_audit.json")
    checks: dict[str, Any] = {
        "run_id": metrics.get("run_id") == RUN_ID,
        "status_measured": metrics.get("status") == "measured",
        "final_week_not_run": metrics.get("final_week") == "not_run"
        and manifest.get("final_week") == "not_run",
        "all_required_outputs_exist": all((report_dir / name).is_file() for name in REQUIRED_OUTPUTS),
    }
    output_identity_checks: dict[str, bool] = {}
    for name, expected in manifest["outputs"].items():
        observed = file_identity(report_dir / name)
        output_identity_checks[name] = (
            observed["bytes"] == expected["bytes"]
            and observed["sha256"] == expected["sha256"]
        )
    checks["output_manifest_identities"] = all(output_identity_checks.values())
    large_checks: dict[str, bool] = {}
    for row in manifest.get("large_ignored_artifacts", []):
        observed = file_identity(Path(row["path"]))
        large_checks[row["path"]] = (
            observed["bytes"] == row["bytes"] and observed["sha256"] == row["sha256"]
        )
    checks["large_artifact_identities"] = all(large_checks.values())
    checks["feature_lineage_manifests"] = _feature_lineage_manifests_valid(
        manifest.get("large_ignored_artifacts", [])
    )
    checks["all_four_windows_present"] = set(metrics.get("windows", {})) == set(WINDOWS)
    checks["all_four_variants_present"] = all(
        set(metrics["windows"][window].get("variants", {})) == {"C0", "C1", "B0", "B1"}
        for window in WINDOWS
    )
    checks["candidate_audit_embedded_exact"] = (
        metrics.get("candidate_identity_audit") == candidate_audit
    )
    checks["training_audit_embedded_exact"] = metrics.get("training_audit") == training_audit
    checks["p33_auxiliary_audit_embedded_exact"] = (
        metrics.get("p33_auxiliary_audit") == p33_auxiliary_audit
    )
    checks["candidate_identity_and_recall200_exact"] = _candidate_identity_valid(
        metrics, candidate_audit
    )
    checks["p37a_controls_exact"] = all(
        metrics["windows"][window]["p3_7a_parity"]["C0_A00_metrics_exact"]
        and metrics["windows"][window]["p3_7a_parity"]["C1_A01_metrics_exact"]
        for window in WINDOWS
    )
    checks["preprocessing_attribution_exact"] = all(
        all(metrics["windows"][window]["preprocessing_parity"].values())
        for window in WINDOWS
    )
    checks["alpha_normalized"] = all(
        metrics["time_gate_audit"]["windows"][window][variant]["alpha_sum_max_abs_error"]
        <= 2e-3
        for window in WINDOWS
        for variant in ("B0", "B1")
    )
    traces_valid, pairs_valid, paired_initialization_valid = _training_audit_valid(
        training_audit
    )
    checks["inner_early_stopping_trace_valid"] = traces_valid
    checks["same_pair_sampling_identities"] = pairs_valid
    checks["paired_initialization_if_present"] = paired_initialization_valid
    checks["p33_masked_audit_complete_and_exact"] = _p33_auxiliary_audit_valid(
        metrics, p33_auxiliary_audit
    )
    recomputed_b0 = _formal_gate(metrics["windows"], "B0")
    recomputed_b1 = _formal_gate(metrics["windows"], "B1")
    recomputed_complexity = _complexity_gate(metrics["windows"])
    recomputed_selection = _b1_vs_b0_selection(metrics["windows"])
    recomputed_decision = _final_decision(
        recomputed_b0, recomputed_b1, recomputed_complexity, recomputed_selection
    )
    stored = metrics["decision"]
    checks["B0_gate_recomputed"] = recomputed_b0 == metrics["gates"]["B0"]
    checks["B1_gate_recomputed"] = recomputed_b1 == metrics["gates"]["B1"]
    checks["complexity_gate_recomputed"] = recomputed_complexity == metrics["gates"]["B1_vs_C1"]
    checks["selection_gate_recomputed"] = recomputed_selection == metrics["gates"]["B1_vs_B0_selection_if_both_supported"]
    checks["decision_recomputed"] = all(
        stored.get(key) == recomputed_decision.get(key)
        for key in ("status", "B0_gate", "B1_gate", "B1_vs_C1_complexity_gate", "selected_variant")
    )
    checks["fusion_policy_consistent"] = bool(stored.get("warm_cold_fusion_allowed")) == (
        stored.get("selected_variant") in {"B0", "B1"}
        and (
            recomputed_b0["passed"]
            if stored.get("selected_variant") == "B0"
            else recomputed_b1["passed"]
        )
    )
    passed = all(bool(value) for value in checks.values())
    return {
        "schema_version": "phase3-p3.7b-verification-v1",
        "status": "passed" if passed else "failed",
        "run_id": RUN_ID,
        "checks": checks,
        "output_identity_checks": output_identity_checks,
        "large_artifact_checks": large_checks,
        "recomputed_decision": recomputed_decision,
        "final_week": "not_run",
    }
