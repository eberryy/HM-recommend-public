from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m4_contract import file_identity
from .p40 import _gate_for_variant
from .p40_contract import FUSION_FEATURES, K_COLD, K_FINAL, K_WARM, RUN_ID


REQUIRED = (
    "P4_0_EXPERIMENT_CONTRACT.json", "P4_0_FINAL.md", "P4_0_metrics.json",
    "p4_0_candidate_union_audit.json", "p4_0_upstream_lineage_audit.json",
    "p4_0_sampling_audit.json", "p4_0_feature_contract.json", "p4_0_feature_usage.json",
    "p4_0_cross_source_calibration.json", "p4_0_cold_truth_funnel.json",
    "p4_0_replacement_accounting.json", "P4_0_OUTPUT_MANIFEST.json",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cold_selection_exact(report_dir: Path, metrics: dict[str, Any]) -> bool:
    """Independently reconstruct every frozen M4-Top200 -> Cold50 identity and rank."""
    try:
        repo_root = report_dir.parent.parent
        p31 = _json(repo_root / "reports" / "phase3" / "P3_1_metrics.json")
        catalog_path = (
            repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation"
            / "student-v1" / "static_catalog" / "catalog_items.csv"
        )
        catalog = pd.read_csv(catalog_path, dtype={"article_id": str}).sort_values("catalog_row")
        catalog_items = catalog["article_id"].astype(str).to_numpy()
        catalog_index = pd.Index(catalog_items)
        for cutoff, audit in metrics["cutoffs"].items():
            asset_key = (
                f"outer_validation:{audit['p31_label']}"
                if audit["role"] == "outer_validation"
                else f"training:{audit['p31_label']}"
            )
            declared = p31["all_assets"][asset_key]
            users = pd.read_csv(
                declared["artifacts"]["users"]["path"], dtype={"customer_id": str}
            )["customer_id"].astype(str).to_numpy()
            with np.load(declared["artifacts"]["candidates"]["path"], allow_pickle=False) as source:
                candidate_user = np.asarray(source["user_index"], dtype=np.int64)
                candidate_catalog = np.asarray(source["catalog_row"], dtype=np.int64)
                coarse_rank = np.asarray(source["rank"], dtype=np.int64)
            cold_path = Path(audit["cold"]["artifact"]["path"])
            observed = duckdb.connect().execute(
                "SELECT customer_id,article_id,cold_rank,b0_rank,b0_score_available,m4_coarse_rank "
                "FROM read_parquet(?)",
                [str(cold_path)],
            ).fetchdf()
            observed_user = pd.Index(users).get_indexer(observed["customer_id"].astype(str))
            observed_catalog = catalog_index.get_indexer(observed["article_id"].astype(str))
            if np.any(observed_user < 0) or np.any(observed_catalog < 0):
                return False
            if audit["cold"]["selection"] == "M4 coarse Top50 fallback":
                selection_rank = coarse_rank
                if not (observed["b0_score_available"].eq(0).all() and observed["b0_rank"].isna().all()):
                    return False
            else:
                selection_rank = np.load(cold_path.parent / "b0_ranks.float32.npy").astype(np.int64)
                if not observed["b0_score_available"].eq(1).all():
                    return False
            keep = selection_rank <= K_COLD
            expected_key = (
                candidate_user[keep].astype(np.uint64) << np.uint64(32)
            ) | candidate_catalog[keep].astype(np.uint64)
            observed_key = (
                observed_user.astype(np.uint64) << np.uint64(32)
            ) | observed_catalog.astype(np.uint64)
            expected_order = np.argsort(expected_key, kind="stable")
            observed_order = np.argsort(observed_key, kind="stable")
            if not np.array_equal(expected_key[expected_order], observed_key[observed_order]):
                return False
            if not np.array_equal(
                selection_rank[keep][expected_order],
                observed["cold_rank"].to_numpy(dtype=np.int64)[observed_order],
            ):
                return False
            if not np.array_equal(
                coarse_rank[keep][expected_order],
                observed["m4_coarse_rank"].to_numpy(dtype=np.int64)[observed_order],
            ):
                return False
        return True
    except Exception:
        return False


def verify(report_dir: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    checks["required_outputs_exist"] = all((report_dir / name).is_file() for name in REQUIRED)
    if not checks["required_outputs_exist"]:
        missing = [name for name in REQUIRED if not (report_dir / name).is_file()]
        raise RuntimeError(f"P4.0 required outputs missing: {missing}")
    contract = _json(report_dir / "P4_0_EXPERIMENT_CONTRACT.json")
    metrics = _json(report_dir / "P4_0_metrics.json")
    manifest = _json(report_dir / "P4_0_OUTPUT_MANIFEST.json")
    feature = _json(report_dir / "p4_0_feature_contract.json")
    lineage = _json(report_dir / "p4_0_upstream_lineage_audit.json")
    checks["run_id_exact"] = contract["run_id"] == metrics["run_id"] == manifest["run_id"] == RUN_ID
    checks["contract_preregistered"] = contract["status"] == "preregistered_before_formal_computation"
    checks["final_week_not_run"] = all(x.get("final_week") == "not_run" for x in (contract, metrics, manifest, feature, lineage))
    checks["four_outer_windows"] = len(metrics["windows"]) == 4
    checks["lineage_safe"] = bool(lineage["all_safe"]) and all(
        row["lineage_safe"] and (not row["available"] or row["model_label_end"] <= cutoff)
        for cutoff, row in lineage["cutoffs"].items()
    )
    checks["earliest_fallback_exact"] = not lineage["cutoffs"]["2019-11-27"]["available"]
    checks["warm_identity_exact"] = all(row["warm"]["exact_frozen_identity"] for row in metrics["cutoffs"].values())
    checks["cold_from_m4_top200"] = _cold_selection_exact(report_dir, metrics)
    checks["union_unique"] = all(row["union"]["invalid_groups"] == 0 for row in metrics["cutoffs"].values())
    checks["source_branch_exclusive"] = all(
        set(row["union"]["branch_rows"]) == {"warm_only", "cold_only", "warm_and_cold"}
        and sum(row["union"]["branch_rows"].values()) == row["union"]["rows"]
        for row in metrics["cutoffs"].values()
    )
    checks["candidate_budgets_frozen"] = (
        (K_WARM, K_COLD, K_FINAL) == (150, 50, 12)
        and contract["candidate_budgets"] == {"warm": 150, "cold": 50, "final": 12}
        and all(row["union"]["max_rows"] <= K_WARM + K_COLD for row in metrics["cutoffs"].values())
    )
    checks["feature_row_conservation"] = all(row["union"]["rows"] == row["features"]["rows"] for row in metrics["cutoffs"].values())
    checks["cold_threshold"] = all(row["features"]["cold_pool_t_le_5"] for row in metrics["cutoffs"].values())
    checks["feature_list_exact"] = feature["features"] == FUSION_FEATURES
    checks["no_b0_internal_features"] = not feature["forbidden_b0_internal_present"]
    checks["W0_exact"] = all(
        abs(row["evaluations"]["W0"]["segments"]["overall"]["map@12"] - row["evaluations"]["W0"]["segments"]["overall"]["warm_baseline_map@12"]) <= 1e-12
        for row in metrics["windows"].values()
    )
    checks["outer_denominator_preserved"] = all(
        len({row["evaluations"][variant]["segments"]["overall"]["truth_users"] for variant in ("W0", "W1", "F0", "F1")}) == 1
        for row in metrics["windows"].values()
    )
    m54 = _json(report_dir.parent / "m5_4" / "M5_4_metrics.json")
    checks["warm_upstream_lineage_safe"] = all(
        cutoff in m54["cutoffs"]
        and m54["cutoffs"][cutoff]["lineage"]["warm_score"]["safe"]
        and (
            not m54["cutoffs"][cutoff]["lineage"]["warm_score"]["available"]
            or m54["cutoffs"][cutoff]["lineage"]["warm_score"]["latest_training_label_end"] <= cutoff
        )
        for cutoff in metrics["cutoffs"]
    )
    recomputed_f0 = _gate_for_variant(metrics["windows"], "F0")
    recomputed_f1 = _gate_for_variant(metrics["windows"], "F1")
    checks["F0_gate_recomputed"] = all(metrics["gates"]["F0"][key] == recomputed_f0[key] for key in recomputed_f0)
    checks["F1_gate_recomputed"] = all(metrics["gates"]["F1"][key] == recomputed_f1[key] for key in recomputed_f1)
    checks["artifact_manifest_hashes"] = True
    for group in (manifest["reports"].values(), manifest["sources"].values(), manifest["large_artifacts"]):
        for expected in group:
            observed = file_identity(Path(expected["path"]))
            if observed["bytes"] != expected["bytes"] or observed["sha256"] != expected["sha256"]:
                checks["artifact_manifest_hashes"] = False
                break
    status = "passed" if all(checks.values()) else "failed"
    result = {"schema_version": "phase4-p4.0-verification-v1", "status": status,
              "checks": checks, "check_count": len(checks),
              "machine_decision": metrics["decision"]["machine_decision"], "final_week": "not_run"}
    return result
