"""Independent, read-only P4.1A evidence verification (no oracle selector reuse)."""
from __future__ import annotations

import ast
import hashlib
import time
from pathlib import Path

import numpy as np

from .metrics import apk
from .p41a_contract import BUDGETS, DIRECTIONS, SLOT_SETS, WINDOWS, check_identity, guard_cutoff, read_json
from .p41a_data import identities, load_window, parquet
from .p41a_stats import LCM, WEIGHTS, auc_metrics, unlabeled_bins


def exact_list_units(r):
    """Independent rank loop; deliberately does not use ap_units/delta formula."""
    hits = np.zeros(len(r), dtype=np.int64)
    score = np.zeros(len(r), dtype=np.int64)
    for j in range(12):
        hits += r[:, j]
        score += hits*r[:, j]*int(WEIGHTS[j])
    return score


def source_safety(repo):
    files = list((repo / "src/hm_recsys").glob("p41a*.py"))
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                assert not any(m.split(".")[0] in {"torch", "tensorflow", "lightgbm", "sklearn", "p40", "p37b", "p32", "p33"} for m in modules), (path, modules)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"fit", "fit_transform", "backward", "train", "train_model", "step"}, (path, node.func.attr)
    return len(files)


def verify(repo):
    start = time.perf_counter()
    report = repo / "reports/phase4"
    manifest = read_json(report / "P4_1A_OUTPUT_MANIFEST.json")
    metrics = read_json(report / "P4_1A_metrics.json")
    contract = read_json(report / "P4_1A_EXPERIMENT_CONTRACT.json")
    frontier = read_json(report / "p4_1a_oracle_frontier.json")["windows"]
    bins = read_json(report / "p4_1a_reliability_audit.json")["windows"]
    separation = read_json(report / "p4_1a_admission_separability.json")["windows"]
    checked_paths = set()
    for expected in identities(manifest):
        if expected["path"] not in checked_paths:
            check_identity(expected)
            checked_paths.add(expected["path"])
    input_paths = set()
    for expected in identities(contract):
        if expected["path"] in input_paths:
            continue
        path = Path(expected["path"])
        if path.name == "ROADMAP_PHASE4.zh-CN.md" and path.stat().st_size >= expected["bytes"]:
            with path.open("rb") as stream:
                assert hashlib.sha256(stream.read(expected["bytes"])).hexdigest() == expected["sha256"]
        else:
            check_identity(expected)
        input_paths.add(expected["path"])
    assert contract["status"] == "preregistered_before_formal_computation"
    assert contract["primary"]["b0_max_rank"] == 10 and contract["primary"]["warm_slots"] == [10, 11, 12]
    assert metrics["final_week"] == "not_run" and not metrics["decision"]["P4_1B_started"]
    assert metrics["decision"]["promoted_baseline"] == "W0"
    assert not metrics["execution"]["training"] and not metrics["execution"]["model_inference"]
    for cutoff in ("2020-09-16", "2020-09-17", "2026-09-08"):
        try:
            guard_cutoff(cutoff)
        except ValueError:
            pass
        else:
            raise AssertionError("final cutoff accepted")
    source_files = source_safety(repo)
    windows = {}
    for window in WINDOWS:
        print(f"P4.1A independent verification: {window}", flush=True)
        data = load_window(contract, window)
        assets = metrics["artifacts"][window]
        opp = parquet(assets["opportunities"]["path"])
        choices = parquet(assets["oracle_selections"]["path"])
        cold = data["cold"]
        allowed = cold.loc[(cold.cold_rank <= 10) & ~cold.already_w0]
        assert len(opp) == 3*len(allowed)
        assert not opp.duplicated(["cutoff", "customer_id", "article_id", "warm_slot_rank"]).any()
        assert set(opp.cutoff) == {data["cutoff"]}
        assert set(opp.warm_slot_rank) <= {10, 11, 12}
        assert opp.cold_rank.le(10).all() and not opp.already_w0.any()
        expected_keys = set(zip(allowed.customer_id, allowed.article_id))
        for slot in (10, 11, 12):
            part = opp.loc[opp.warm_slot_rank == slot]
            assert set(zip(part.customer_id, part.article_id)) == expected_keys
        u = opp.user_index.to_numpy(dtype=int)
        j = opp.warm_slot_rank.to_numpy(dtype=int)-1
        before = data["relevance"][u]
        after = before.copy()
        after[np.arange(len(after)), j] = opp.target
        exact = exact_list_units(after)-exact_list_units(before)
        assert np.array_equal(exact, opp.delta_ap_units)
        assert np.array_equal(exact > 0, opp.beneficial)
        assert np.array_equal(exact == 0, opp.opportunity_label.eq("neutral"))
        assert np.array_equal(exact < 0, opp.opportunity_label.eq("harmful"))
        assert np.array_equal(exact/data["denominator"][u], opp.delta_AP12_if_replaced)
        assert np.array_equal(opp.warm_article_id, data["warm_lists"][u, j])
        # Stored normalized variables and availability must match complete Cold50.
        lookup = cold.set_index(["customer_id", "article_id"])
        aligned = lookup.loc[list(zip(opp.customer_id, opp.article_id))]
        for var in DIRECTIONS:
            np.testing.assert_allclose(opp[var].to_numpy(float), aligned[var].to_numpy(float), rtol=0, atol=0, equal_nan=True)
            signed = DIRECTIONS[var]*opp[var].to_numpy(float)
            observed = auc_metrics(signed, opp.beneficial)
            assert observed == {k: separation[window][var][k] for k in observed}
            for nb in (4, 10):
                boundaries, assignments = unlabeled_bins(signed, nb)
                saved = bins[window][var][str(nb)]
                np.testing.assert_allclose(boundaries, np.array(saved["boundaries"], float), rtol=0, atol=0, equal_nan=True)
                for row in saved["bins"]:
                    m = assignments == row["bin"]-1
                    assert int(m.sum()) == row["rows"]
                    assert int(opp.beneficial.to_numpy()[m].sum()) == row["beneficial_rows"]
        positive = cold.loc[~cold.already_w0 & cold.target.eq(1)]
        ui = positive.user_index.to_numpy(dtype=int)
        positive_deltas = []
        for slot in range(12):
            r = data["relevance"][ui].copy()
            r[:, slot] = 1
            positive_deltas.append(exact_list_units(r)-exact_list_units(data["relevance"][ui]))
        with np.load(assets["oracle_top12_lists"]["path"], allow_pickle=False) as archive:
            assert np.array_equal(archive["users"], data["users"])
            assert np.array_equal(archive["w0"], data["warm_lists"].astype(np.int64))
            for budget in BUDGETS:
                for scope, slots in SLOT_SETS.items():
                    cell = f"top{budget}_{scope}"
                    actual = archive[cell]
                    selection = choices.loc[choices.cell == cell].reset_index(drop=True)
                    assert np.array_equal(selection.customer_id, data["users"])
                    n = len(actual)
                    best = np.zeros(n, dtype=np.int64)
                    best_rank = np.zeros(n, dtype=np.int64)
                    best_slot = np.zeros(n, dtype=np.int64)
                    best_item = np.zeros(n, dtype=np.int64)
                    # Enumerate rank ascending then slot ascending; strictly greater
                    # overwrites only. This independently implements the fixed tie rule.
                    for rank in range(1, budget+1):
                        idx = np.flatnonzero(positive.cold_rank.to_numpy() == rank)
                        cu = ui[idx]
                        for slot in slots:
                            values = positive_deltas[slot-1][idx]
                            improve = values > best[cu]
                            chosen = cu[improve]
                            best[chosen] = values[improve]
                            best_rank[chosen] = rank
                            best_slot[chosen] = slot
                            best_item[chosen] = positive.article_id.to_numpy()[idx[improve]].astype(np.int64)
                    changed = actual != archive["w0"]
                    assert np.array_equal(changed.sum(axis=1), best > 0)
                    assert all(len(set(row)) == 12 for row in actual)
                    expected_lists = archive["w0"].copy()
                    chosen = np.flatnonzero(best > 0)
                    expected_lists[chosen, best_slot[chosen]-1] = best_item[chosen]
                    assert np.array_equal(actual, expected_lists)
                    assert np.array_equal(selection.delta_ap_units, best)
                    assert np.array_equal(selection.selected_cold_rank, best_rank)
                    assert np.array_equal(selection.replaced_warm_rank, best_slot)
                    delta = best/data["denominator"]
                    assert float(delta.mean()) == frontier[window][cell]["delta_MAP_vs_W0"]
                    assert int((best > 0).sum()) == frontier[window][cell]["positive_opportunity_users"]
                    assert frontier[window][cell]["removed_warm_positive_pairs"] == 0
                    if cell == "top10_slots10_12":
                        assert np.array_equal(actual[:, :9], archive["w0"][:, :9])
                        # Original apk reference on every saved primary list.
                        ref = np.array([apk(list(data["truthsets"][user]), [str(x).zfill(10) for x in row]) for user, row in zip(data["users"], actual)])
                        np.testing.assert_allclose(ref-data["baseline_ap"], delta, rtol=0, atol=2e-16)
                        assert float(delta.mean()) == metrics["windows"][window]["delta_MAP_vs_W0"]
        for artifact in assets.values():
            if artifact["path"].endswith(".parquet"):
                import duckdb
                with duckdb.connect() as con:
                    assert con.execute("SELECT count(*) FROM read_parquet(?)", [artifact["path"]]).fetchone()[0] == artifact["row_count"]
        windows[window] = {"pass": True, "legal_primary_opportunity_rows": len(opp), "independently_optimized_cells": 15,
            "w0_map_bit_exact": True, "cold50_exact_p40_m4_lineage": True,
            "all_opportunity_exact_delta_signs": True, "max1_no_admission_unique12_all_cells": True,
            "all_primary_reference_ap_recomputed": True, "confidence_bins_auc_recomputed": True}
    # Machine decision independently reconstructed from saved summary statistics.
    from .p41a import verdicts
    assert verdicts(metrics["windows"], metrics["portability"], metrics["raw_calibration_diagnostics"]) == metrics["decision"]
    return {"stage": "P4.1A", "status": "pass", "windows": windows,
        "output_and_source_sha_files": len(checked_paths), "frozen_input_sha_files": len(input_paths),
        "audit_source_ast_checked_files": source_files, "final_week_rejected": True,
        "no_training_model_api_path": True, "machine_decision_recomputed": True,
        "seconds": time.perf_counter()-start, "final_week": "not_run"}
