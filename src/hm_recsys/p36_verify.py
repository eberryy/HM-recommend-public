from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

from .m4_contract import atomic_json, file_identity
from .p36_audit import _derive_verdicts


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_identity(identity: dict[str, Any]) -> None:
    observed = file_identity(Path(identity["path"]))
    if observed["bytes"] != identity["bytes"] or observed["sha256"] != identity["sha256"]:
        raise RuntimeError(f"identity mismatch: {identity['path']}")


def verify(repo_root: Path) -> dict[str, Any]:
    report_dir = repo_root / "reports" / "phase3"
    master_path = report_dir / "P3_6_SIMILARITY_COMPLEMENTARITY_AUDIT.json"
    output_manifest_path = report_dir / "P3_6_OUTPUT_MANIFEST.json"
    master = _json(master_path)
    output_manifest = _json(output_manifest_path)
    if master["status"] != "measured" or master["final_week"] != "not_run":
        raise RuntimeError("P3.6 measured/final-week boundary failed")
    if master["runtime"]["model_training"] or master["runtime"]["new_recommendation_candidate_generation"]:
        raise RuntimeError("P3.6 scope boundary failed")
    for identity in output_manifest["artifacts"].values():
        _verify_identity(identity)

    row_checks: dict[str, Any] = {}
    connection = duckdb.connect()
    try:
        for window, data in master["candidate_relation_features"]["windows"].items():
            proxy = data["warm_proxy_mapping"]
            candidate = data["candidate_relation_features"]
            _verify_identity(proxy)
            _verify_identity(candidate)
            proxy_rows = connection.execute(
                f"SELECT count(*) FROM read_parquet('{Path(proxy['path']).as_posix()}')"
            ).fetchone()[0]
            candidate_rows = connection.execute(
                f"SELECT count(*) FROM read_parquet('{Path(candidate['path']).as_posix()}')"
            ).fetchone()[0]
            pair_rows = 0
            for part in data["pair_relation_features"]["parts"]:
                _verify_identity(part)
                observed_rows = connection.execute(
                    f"SELECT count(*) FROM read_parquet('{Path(part['path']).as_posix()}')"
                ).fetchone()[0]
                if observed_rows != part["rows"]:
                    raise RuntimeError(f"pair parquet row mismatch: {part['path']}")
                pair_rows += observed_rows
            expected_candidate = master["proxy_quality"]["windows"][window]["candidate_rows"]
            expected_pair = master["relation_matrix"]["windows"][window]["valid_history_observations"]
            expected_unique = master["proxy_quality"]["windows"][window]["unique_candidate_items"]
            checks = {
                "candidate_rows_match": candidate_rows == candidate["rows"] == expected_candidate,
                "pair_rows_match": pair_rows == data["pair_relation_features"]["rows"] == expected_pair,
                "proxy_rows_match_unique_items_times_5": proxy_rows == proxy["rows"] == expected_unique * 5,
            }
            if not all(checks.values()):
                raise RuntimeError(f"large artifact row conservation failed at {window}: {checks}")
            row_checks[window] = {
                **checks,
                "candidate_rows": candidate_rows,
                "pair_rows": pair_rows,
                "proxy_rows": proxy_rows,
            }
    finally:
        connection.close()

    recomputed_verdicts, recomputed_evidence = _derive_verdicts(
        proxy_quality=master["proxy_quality"],
        p32=master["p32_relation_attribution"],
        p33=master["p33_relation_attribution"],
        higher_order=master["deepwalk_higher_order_audit"],
        history_age=master["history_age_relation_audit"],
        teacher_composition=master["teacher_relation_composition"],
    )
    checks = {
        "core_output_identities_verified": True,
        "large_artifact_identities_and_rows_verified": all(
            all(value[key] for key in ("candidate_rows_match", "pair_rows_match", "proxy_rows_match_unique_items_times_5"))
            for value in row_checks.values()
        ),
        "verdicts_recomputed_exactly": recomputed_verdicts == master["verdicts"],
        "verdict_evidence_recomputed_exactly": recomputed_evidence == master["verdict_evidence"],
        "all_runtime_audit_checks_passed": master["audit_checks"]["all_passed"],
        "final_week_not_run": master["final_week"] == "not_run" and output_manifest["final_week"] == "not_run",
        "no_training": not master["runtime"]["model_training"],
        "no_new_recommendation_candidates": not master["runtime"]["new_recommendation_candidate_generation"],
    }
    checks["all_passed"] = all(checks.values())
    if not checks["all_passed"]:
        raise RuntimeError(f"P3.6 verification failed: {checks}")
    return {
        "schema_version": "phase3-p3.6-verification-v1",
        "stage": "P3.6",
        "status": "verified",
        "master": file_identity(master_path),
        "row_checks": row_checks,
        "recomputed_verdicts": recomputed_verdicts,
        "checks": checks,
        "final_week": "not_run",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify P3.6 outputs independently")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    result = verify(repo_root)
    report_dir = repo_root / "reports" / "phase3"
    verification_path = report_dir / "P3_6_VERIFICATION.json"
    atomic_json(verification_path, result)
    output_manifest_path = report_dir / "P3_6_OUTPUT_MANIFEST.json"
    output_manifest = _json(output_manifest_path)
    output_manifest["verification"] = file_identity(verification_path)
    atomic_json(output_manifest_path, output_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
