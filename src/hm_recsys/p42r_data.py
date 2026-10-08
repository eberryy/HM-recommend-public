"""P4.2R shared-history population repair, with frozen M4/B0 inference only.

S is the complete frozen W0 offline roster; H is its pre-cutoff mapped-history
subset. Both propensity training tables cover exactly H. Full outer scoring
continues to use p42_data.load_cutoff and therefore retains all S users.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from .m4_contract import atomic_json, file_identity
from .p3 import _retrieve_top200
from .p37b import _rank_hybrid, _reload_outer_model
from .p37b_features import (
    USER_STATE_COLUMNS, _bucket_cosine_summaries, _cosine_tensor,
    _same_attribute_counts, assign_age_buckets, compute_candidate_state,
    compute_user_state_batch,
)
from .p37b_model import FeaturePreprocessor
from .p41a_contract import guard_cutoff
from .p41a_stats import distribution
from .p42_data import (
    STATE_COLUMNS, _path, _warm_source_order, normalize_cold_confidence,
    validate_lineages, validate_warm_identity,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


_VERIFIED_EXTRA = {}


def _verify_declared(declared):
    """Check restarted mutable frozen assets against prior trusted records."""
    path = Path(_path(declared)).resolve()
    key = (str(path), declared["sha256"], declared["bytes"])
    if key not in _VERIFIED_EXTRA:
        observed = file_identity(path)
        if observed["sha256"] != declared["sha256"] or observed["bytes"] != declared["bytes"]:
            raise ValueError(f"frozen reconstruction input integrity mismatch: {path}")
        _VERIFIED_EXTRA[key] = {**observed, "trusted_sha256": declared["sha256"], "matched": True}
    return _VERIFIED_EXTRA[key]


def _find_declared(tree, name):
    if isinstance(tree, dict):
        if "path" in tree and "sha256" in tree and Path(tree["path"]).name == name:
            return tree
        for value in tree.values():
            found = _find_declared(value, name)
            if found:
                return found
    elif isinstance(tree, list):
        for value in tree:
            found = _find_declared(value, name)
            if found:
                return found
    return None


def _parquet(path):
    with duckdb.connect() as con:
        return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()


def _save_frame(frame, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite P4.2R evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.register("output_rows", frame)
        con.execute("COPY output_rows TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    return {"path": str(path), "bytes": path.stat().st_size, "rows": len(frame)}


def shared_users(s_users, cold, warm):
    """Actual row-support parity; a nominal list cannot substitute for rows."""
    expected = set(map(str, s_users))
    cold_users = set(cold.customer_id.astype(str))
    warm_users = set(warm.customer_id.astype(str))
    if expected != cold_users or expected != warm_users:
        raise ValueError("shared_population_contract_failure: actual candidate user sets differ")
    if cold.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("Cold50 has duplicate user-item rows")
    if not cold.groupby("customer_id").size().eq(50).all():
        raise ValueError("Cold50 budget is not exactly 50")
    validate_warm_identity(warm)
    return {"H_t_users": len(expected), "qC_users": len(cold_users),
            "qW_users": len(warm_users), "exact_user_set_parity": True,
            "qC_rows": len(cold), "qW_rows": len(warm)}


def history_context(transactions, cutoff, users, catalog_items):
    """Frozen P3.1 most-recent-20 distinct history, with NO future-label SQL.

Rank before mapping, as in P3.1. All project transaction items are catalog
mapped; retaining this order also protects exact old-history parity.
"""
    guard_cutoff(cutoff)
    roster = pd.DataFrame({"customer_id": users})
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='4GB'")
        con.register("roster", roster)
        counts_frame = con.execute("SELECT article_id,count(*) AS events FROM read_parquet(?) "
            "WHERE t_dat<CAST(? AS DATE) GROUP BY article_id", [str(transactions), cutoff]).fetchdf()
        history = con.execute("""WITH latest AS (
            SELECT t.customer_id,t.article_id,max(t.t_dat) latest_date
            FROM read_parquet(?) t JOIN roster USING(customer_id)
            WHERE t.t_dat<CAST(? AS DATE) GROUP BY t.customer_id,t.article_id
          ), ranked AS (
            SELECT *,row_number() OVER(PARTITION BY customer_id
                ORDER BY latest_date DESC,article_id) AS history_rank FROM latest
          ) SELECT customer_id,article_id,history_rank,
              date_diff('day',latest_date,CAST(? AS DATE)) AS history_days
          FROM ranked WHERE history_rank<=20 ORDER BY customer_id,history_rank""",
            [str(transactions), cutoff, cutoff]).fetchdf()
    item_index = {str(i): j for j, i in enumerate(catalog_items)}
    user_index = {str(u): j for j, u in enumerate(users)}
    counts = np.zeros(len(catalog_items), np.int32)
    for row in counts_frame.itertuples(index=False):
        if str(row.article_id) in item_index:
            counts[item_index[str(row.article_id)]] = row.events
    rows = np.full((len(users), 20), -1, np.int32)
    days = np.zeros((len(users), 20), np.float32)
    for row in history.itertuples(index=False):
        if str(row.article_id) in item_index:
            u, h = user_index[str(row.customer_id)], int(row.history_rank) - 1
            rows[u, h] = item_index[str(row.article_id)]
            days[u, h] = row.history_days
    if np.any(days[rows >= 0] <= 0):
        raise ValueError("history contains a cutoff-day or future purchase")
    return counts, rows, days


def m4_relation_only(items, history_rows, history_days, embedding, product, garment, device):
    """Exact six-field P3.7B M4 relation using its original pure helpers.

No P3.3 embedding is loaded: B0 never consumes P3.3 fields. This avoids
materializing irrelevant auxiliary arrays, not a change in B0 input values.
"""
    mask = history_rows >= 0
    bucket = assign_age_buckets(history_days, mask)
    similarity = _cosine_tensor(candidate_rows=items, history_rows=history_rows,
        history_mask=mask, embeddings=embedding, device=device)
    maxima, top3, count = _bucket_cosine_summaries(similarity, bucket)
    result = np.empty((*items.shape, 4, 6), np.float32)
    result[..., 0] = (count > 0)[:, None, :]
    result[..., 1] = count[:, None, :]
    result[..., 2], result[..., 3] = maxima, top3
    for column, attribute in ((4, product), (5, garment)):
        result[..., column] = _same_attribute_counts(candidate_rows=items,
            history_rows=history_rows, bucket_index=bucket, catalog_attribute=attribute)
    return result


def score_b0(arrays, history_rows, history_days, state, counts, embedding,
             product, garment, lineage, device):
    """Bounded, deterministic frozen B0 inference, no estimator fitting."""
    model = _reload_outer_model(variant="B0", model_path=Path(_path(lineage["model"])), device=device)
    preprocessing = FeaturePreprocessor.from_manifest(_read(_path(lineage["preprocessing"])))
    if preprocessing.variant != "B0":
        raise ValueError("population repair can only score the frozen B0")
    n = len(arrays["rank"])
    candidate_state = compute_candidate_state(candidates=arrays, interaction_counts=counts)
    score = np.empty(n, np.float32)
    items = arrays["catalog_row"].reshape(-1, 200)
    group_users = arrays["user_index"].reshape(-1, 200)[:, 0]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(items), 32):
            end = min(start + 32, len(items))
            users = group_users[start:end]
            lo, hi = start * 200, end * 200
            relation = m4_relation_only(items[start:end], history_rows[users], history_days[users],
                                        embedding, product, garment, device).reshape(-1, 4, 6)
            def tensor(value):
                return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(device)
            output = model(
                tensor(arrays["coarse_score"][lo:hi]),
                tensor(preprocessing.relation.transform(relation)),
                tensor(preprocessing.user_state.transform(state[arrays["user_index"][lo:hi]])),
                tensor(preprocessing.candidate_state.transform(candidate_state[lo:hi])),
            )
            score[lo:hi] = output.score.cpu().numpy().astype(np.float32)
    if not np.isfinite(score).all():
        raise ValueError("non-finite frozen B0 prediction")
    ranks = _rank_hybrid(arrays, score)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score, ranks


def shortlist(arrays, scores, ranks, users, items):
    """Unlabelled original P4.0 Cold50 fields, same dtypes and normalization."""
    keep = np.flatnonzero(ranks <= 50)
    br = ranks[keep].astype(np.float32)
    frame = pd.DataFrame({
        "customer_id": np.asarray(users)[arrays["user_index"][keep]],
        "article_id": np.asarray(items)[arrays["catalog_row"][keep]],
        "cold_rank": ranks[keep].astype(np.int16), "cold_rank_pct": br / 50,
        "b0_score": scores[keep].astype(np.float32), "b0_score_available": np.uint8(1),
        "b0_rank": br, "b0_rank_pct": br / 50,
        "m4_coarse_score": arrays["coarse_score"][keep].astype(np.float32),
        "m4_coarse_rank": arrays["rank"][keep].astype(np.int16),
        "m4_coarse_rank_pct": arrays["rank"][keep].astype(np.float32) / 200,
        "b0_delta_available": np.uint8(1),
    })
    frame["b0_delta_vs_m4"] = frame.b0_score - frame.m4_coarse_score
    return frame.sort_values(["customer_id", "cold_rank", "article_id"], ignore_index=True)


def join_labels_last(cold, transactions, cutoff, users):
    """Only call after the entire Cold50 selection has been persisted."""
    guard_cutoff(cutoff)
    end = (date.fromisoformat(cutoff) + timedelta(days=7)).isoformat()
    roster = pd.DataFrame({"customer_id": users})
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.register("roster", roster)
        truth = con.execute("""SELECT DISTINCT customer_id,article_id
            FROM read_parquet(?) JOIN roster USING(customer_id)
            WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)""",
            [str(transactions), cutoff, end]).fetchdf()
    truth["target"] = np.uint8(1)
    result = cold.merge(truth, on=["customer_id", "article_id"], how="left", validate="one_to_one")
    result["target"] = result.target.fillna(0).astype(np.uint8)
    return result, truth


def verify_offline_roster(transactions, cutoff, users):
    """Audit formal any-purchase hash10% roster, not a cold-truth selector."""
    guard_cutoff(cutoff)
    with duckdb.connect() as con:
        observed = con.execute("""SELECT DISTINCT customer_id FROM read_parquet(?)
          WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY
          AND hash(customer_id)%1000000<100000 ORDER BY customer_id""",
            [str(transactions), cutoff, cutoff]).fetchdf().customer_id.to_numpy()
    np.testing.assert_array_equal(np.sort(np.asarray(users)), observed)
    return {"any_next_week_truth_hash10pct_exact_S_t": True,
            "requires_pre_cutoff_history": False, "requires_next_week_any_purchase": True,
            "requires_next_week_cold_truth": False, "verified_users": len(observed)}


def population_comparison(old, new):
    old_users, new_users = set(old.customer_id), set(new.customer_id)
    retained = old_users & new_users
    restored = new_users - old_users
    def summary(frame):
        return {"users": int(frame.customer_id.nunique()), "rows": len(frame),
                "positive_rows": int(frame.target.sum()),
                "positive_rate": float(frame.target.mean()) if len(frame) else None}
    columns = ["b0_user_percentile", "b0_user_zscore", "normalized_margin_to_rank2",
               "normalized_margin_to_rank5", "normalized_margin_to_user_median", *STATE_COLUMNS]
    strata = {
        "old_selected_all": old,
        "old_selected_retained_in_H": new.loc[new.customer_id.isin(retained)],
        "newly_restored_in_H": new.loc[new.customer_id.isin(restored)],
    }
    return {"unit": "candidate rows; users deduplicated within cutoff",
        "old": summary(old), "new": summary(new),
        "old_users_outside_H": len(old_users - new_users),
        "old_users_retained_in_H": len(retained), "newly_restored_users": len(restored),
        "strata": {name: {**summary(frame), "features": {
            column: distribution(frame[column]) for column in columns}}
            for name, frame in strata.items()}}


def _replay_probe(arrays, scores, ranks, users, source, original_users, probe_users):
    original_index = {u: i for i, u in enumerate(original_users)}
    wanted = np.array([original_index[u] for u in probe_users])
    original = np.load(_path(source["p31_assets"]["candidates"]), allow_pickle=False)
    keep = np.isin(original["user_index"], wanted)
    observed_keep = arrays["user_index"] < len(probe_users)
    expected_pairs = pd.DataFrame({"customer_id": original_users[original["user_index"][keep]],
        "catalog_row": original["catalog_row"][keep], "rank": original["rank"][keep],
        "coarse_score": original["coarse_score"][keep],
        "b0_score": np.load(_path(source["b0_lineage"]["score_artifact"]))[keep],
        "b0_rank": np.load(_path(source["b0_lineage"]["rank_artifact"]))[keep],
    }).sort_values(["customer_id", "rank"], ignore_index=True)
    observed_pairs = pd.DataFrame({"customer_id": np.asarray(users)[arrays["user_index"][observed_keep]],
        "catalog_row": arrays["catalog_row"][observed_keep], "rank": arrays["rank"][observed_keep],
        "coarse_score": arrays["coarse_score"][observed_keep],
        "b0_score": scores[observed_keep], "b0_rank": ranks[observed_keep],
    }).sort_values(["customer_id", "rank"], ignore_index=True)
    for name in ("customer_id", "catalog_row", "rank"):
        np.testing.assert_array_equal(expected_pairs[name], observed_pairs[name], err_msg=f"frozen probe {name}")
    expected_cold50 = expected_pairs.loc[expected_pairs.b0_rank <= 50].sort_values(["customer_id", "b0_rank"])
    observed_cold50 = observed_pairs.loc[observed_pairs.b0_rank <= 50].sort_values(["customer_id", "b0_rank"])
    for name in ("customer_id", "catalog_row", "b0_rank"):
        np.testing.assert_array_equal(expected_cold50[name], observed_cold50[name], err_msg=f"frozen Cold50 probe {name}")
    errors = {}
    for name in ("coarse_score", "b0_score"):
        errors[name] = float(np.max(np.abs(expected_pairs[name].astype(float)-observed_pairs[name].astype(float))))
        np.testing.assert_allclose(expected_pairs[name], observed_pairs[name], rtol=0, atol=1e-5)
    return {"users": len(probe_users), "candidate_rows": len(expected_pairs),
        "m4_top200_identity_rank_exact": True,
        "b0_all200_rank_exact": bool(np.array_equal(expected_pairs.b0_rank, observed_pairs.b0_rank)),
        "b0_cold50_identity_rank_exact": True, "score_absolute_tolerance": 1e-5,
        "max_absolute_errors": errors, "label_blind_probe": True}


def prepare_cutoff(repo, contract, cutoff, output_root, device="cuda"):
    """Generate both H-only training tables after preregistration.

All original rows remain immutable. Existing overlap users retain bit-exact
frozen candidates; only the missing H users require new M4/B0 inference.
The first 32 sorted retained users are replayed as label-free mechanics probes.
"""
    cutoff = guard_cutoff(cutoff)
    repo, output_root = Path(repo).resolve(), Path(output_root).resolve()
    if contract.get("status") != "preregistered_before_formal_computation":
        raise ValueError("candidate reconstruction requires preregistration")
    disk_contract = repo / "reports/phase4/P4_2R_EXPERIMENT_CONTRACT.json"
    if not disk_contract.is_file() or _read(disk_contract).get("run_id") != contract.get("run_id"):
        raise ValueError("persisted P4.2R preregistration is required")
    if not output_root.is_relative_to(repo / "artifacts/phase4") or not output_root.name.startswith("p4-2r-"):
        raise ValueError("P4.2R outputs must stay in its new ignored artifact root")
    source = contract["inputs"][cutoff]
    validate_lineages(source, cutoff)
    if not source["b0_lineage"]["available"]:
        raise ValueError("no_safe_B0: exclude cutoff from BOTH propensity pools")
    folder = output_root / "prepared" / cutoff
    if folder.exists():
        raise FileExistsError(f"P4.2R cutoff already started: {folder}")
    folder.mkdir(parents=True)
    started = time.perf_counter()
    timeline = [{"event": "begin_shared_population_repair", "at_utc": _now()}]
    frozen_w0 = _warm_source_order(source)
    if frozen_w0 is None:
        raise ValueError("safe upstream B0 but no original W0 database")
    validate_warm_identity(frozen_w0)
    users = frozen_w0.customer_id.drop_duplicates().to_numpy()
    catalog = pd.read_csv(_path(contract["catalog"]), dtype={"article_id": str}).sort_values("catalog_row")
    items = catalog.article_id.to_numpy()
    transactions = _path(contract["transactions"])
    counts, all_rows, all_days = history_context(transactions, cutoff, users, items)
    active = (all_rows >= 0).any(axis=1)
    history_users = users[active]
    rows, days = all_rows[active], all_days[active]
    timeline.append({"event": "S_and_H_fixed_from_W0_and_past_history", "at_utc": _now()})
    old_shortlist = _parquet(_path(source["cold50"]))
    if "target" in old_shortlist:
        raise ValueError("expected the original unlabelled Cold50 selection artifact")
    existing_users = set(old_shortlist.customer_id)
    retained = sorted(set(history_users) & existing_users)
    missing = sorted(set(history_users) - existing_users)
    reused = old_shortlist.loc[old_shortlist.customer_id.isin(history_users)].copy()
    p31_metrics_declared = contract["authoritative_inputs"]["reports/phase3/P3_1_metrics.json"]
    extra_integrity = [_verify_declared(p31_metrics_declared)]
    p31 = _read(repo / "reports/phase3/P3_1_metrics.json")
    role_key = f"{source['role']}:{source['p31_label']}"
    declared_p31 = p31["all_assets"][role_key]
    for name in ("candidates", "users"):
        if declared_p31["artifacts"][name]["sha256"] != source["p31_assets"][name]["sha256"]:
            raise ValueError(f"P3.1 and P4.2 frozen source alignment mismatch: {name}")
    extra_integrity.append(_verify_declared(declared_p31["artifacts"]["manifest"]))
    p31_manifest = _read(_path(declared_p31["artifacts"]["manifest"]))
    if p31_manifest["cutoff"] != cutoff:
        raise ValueError("M4 manifest cutoff mismatch")
    extra_integrity.append(_verify_declared(p31_manifest["student_embedding"]))
    extra_integrity.append(_verify_declared(p31_manifest["artifacts"]["histories"]))
    embedding_path = Path(_path(p31_manifest["student_embedding"]))
    embedding = np.load(embedding_path, mmap_mode="r")
    # The P3.7B output manifest binds its metrics, which bind the raw attribute
    # snapshot. Do not substitute a current-file hash for that older evidence.
    p37b_manifest_declared = contract["authoritative_inputs"]["reports/phase3/P3_7B_OUTPUT_MANIFEST.json"]
    extra_integrity.append(_verify_declared(p37b_manifest_declared))
    p37b_metric_declared = _find_declared(_read(_path(p37b_manifest_declared)), "P3_7B_metrics.json")
    if p37b_metric_declared is None:
        raise ValueError("missing trusted P3.7B attribute provenance")
    extra_integrity.append(_verify_declared(p37b_metric_declared))
    article_declared = _read(_path(p37b_metric_declared))["authoritative_inputs"]["articles"]
    extra_integrity.append(_verify_declared(article_declared))
    attributes = pd.read_csv(repo / "data/raw/articles.csv", dtype={"article_id": str},
        usecols=["article_id", "product_type_no", "garment_group_no"])
    aligned = catalog[["article_id", "catalog_row"]].merge(attributes, on="article_id",
        how="left", validate="one_to_one").sort_values("catalog_row")
    product = aligned.product_type_no.fillna(-1).to_numpy(np.int32)
    garment = aligned.garment_group_no.fillna(-1).to_numpy(np.int32)
    state = np.empty((len(history_users), len(USER_STATE_COLUMNS)), np.float32)
    for start in range(0, len(state), 2048):
        stop = min(start + 2048, len(state))
        state[start:stop] = compute_user_state_batch(history_rows=rows[start:stop],
            history_days=days[start:stop], history_mask=rows[start:stop] >= 0,
            m4_embeddings=embedding, catalog_product_type=product)
    original_users = pd.read_csv(_path(source["p31_assets"]["users"]), dtype={"customer_id": str}).customer_id.to_numpy()
    old_user_index, h_index = {u: i for i, u in enumerate(original_users)}, {u: i for i, u in enumerate(history_users)}
    old_state = np.load(_path(source["user_state"]), mmap_mode="r")
    shared_state_users = sorted(set(history_users) & set(original_users))
    with np.load(_path(p31_manifest["artifacts"]["histories"]), allow_pickle=False) as old_history:
        new_idx = [h_index[u] for u in shared_state_users]
        old_idx = [old_user_index[u] for u in shared_state_users]
        np.testing.assert_array_equal(rows[new_idx], old_history["catalog_row"][old_idx])
        np.testing.assert_array_equal(days[new_idx], old_history["days_since_purchase"][old_idx])
    # All six qC history fields are bit-exact on every reused source user.
    np.testing.assert_array_equal(state[[h_index[u] for u in shared_state_users], :6],
                                  old_state[[old_user_index[u] for u in shared_state_users], :6])
    probe = {"status": "not_needed_no_new_users", "users": 0}
    resources, generated_artifacts = {}, {}
    if missing:
        probe_users = retained[:32]
        generation_users = probe_users + missing
        idx = np.array([h_index[u] for u in generation_users])
        gpu = torch.device(device)
        if gpu.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for fixed M4 inference but unavailable")
        arrays, resources = _retrieve_top200(embeddings_path=embedding_path,
            candidate_rows=np.flatnonzero(counts <= 5).astype(np.int32),
            history_rows=rows[idx], history_days=days[idx], truth={}, device=gpu)
        if arrays["target"].any():
            raise ValueError("label-blind retrieval unexpectedly has targets")
        timeline.append({"event": "new_M4_top200_fixed_without_truth", "at_utc": _now()})
        score_started = time.perf_counter()
        scores, ranks = score_b0(arrays, rows[idx], days[idx], state[idx], counts,
            embedding, product, garment, source["b0_lineage"], gpu)
        resources["b0_inference_seconds"] = time.perf_counter() - score_started
        if probe_users:
            probe = _replay_probe(arrays, scores, ranks, generation_users, source, original_users, probe_users)
        generated = shortlist(arrays, scores, ranks, generation_users, items)
        generated = generated.loc[generated.customer_id.isin(missing)].copy()
        selected = pd.concat([reused, generated], ignore_index=True)
        # Store generated Top200 with no target field, including probe users.
        np.savez_compressed(folder / "new-m4-top200-unlabelled.npz",
            **{key: value for key, value in arrays.items() if key != "target"},
            b0_score=scores, b0_rank=ranks)
        generated_artifacts["new_m4_top200_unlabelled"] = str(folder / "new-m4-top200-unlabelled.npz")
        generated_artifacts["generation_users"] = _save_frame(pd.DataFrame({
            "customer_id": generation_users, "is_probe": [u in set(probe_users) for u in generation_users]}),
            folder / "generation-users.parquet")
        del arrays, generated, scores, ranks
    else:
        selected = reused
    selected = selected.sort_values(["customer_id", "cold_rank", "article_id"], ignore_index=True)
    # This file is the temporal fence: no future item labels have been read in
    # this function before the source-specific shortlist is fixed and saved.
    selection_artifact = _save_frame(selected, folder / "cold50-selection-before-labels.parquet")
    timeline.append({"event": "complete_Cold50_persisted_before_label_join", "at_utc": _now()})
    atomic_json(folder / "SELECTION_BEFORE_LABEL_JOIN.json", {
        "stage": "P4.2R", "cutoff": cutoff, "timeline": timeline,
        "S_t_users": len(users), "H_t_users": len(history_users),
        "existing_candidate_users_reused": len(retained), "new_users_generated": len(missing),
        "selection": selection_artifact, "future_item_labels_read": False,
    })
    label_started = _now()
    cold, truth = join_labels_last(selected, transactions, cutoff, history_users)
    roster_audit = verify_offline_roster(transactions, cutoff, users)
    timeline.append({"event": "future_truth_join_completed", "started_at_utc": label_started, "at_utc": _now()})
    state_frame = pd.DataFrame(state[:, :6], columns=STATE_COLUMNS)
    state_frame.insert(0, "customer_id", history_users)
    cold = cold.merge(state_frame, on="customer_id", how="left", validate="many_to_one")
    count_map = dict(zip(items, counts))
    cold["interaction_count_before_cutoff"] = cold.article_id.map(count_map).astype(np.int64)
    cold["strict_cold_flag"] = (cold.interaction_count_before_cutoff == 0).astype(np.uint8)
    cold["sparse1_5_flag"] = cold.interaction_count_before_cutoff.between(1, 5).astype(np.uint8)
    cold = normalize_cold_confidence(cold)
    cold["target_cutoff"] = cutoff
    old_prepared = Path(contract.get("old_prepared_root", repo / "artifacts/phase4/p4-2-v1-two-sided-risk-controlled-admission/prepared")) / cutoff
    old_cold = _parquet(old_prepared / "qC-features.parquet")
    old_warm = _parquet(old_prepared / "qW-features.parquet")
    pd.testing.assert_frame_equal(old_warm[list(frozen_w0.columns)].reset_index(drop=True), frozen_w0,
                                  check_dtype=False, check_exact=True)
    warm = old_warm.loc[old_warm.customer_id.isin(history_users)].reset_index(drop=True)
    # Recompute W0 labels in the same shared population after generation.
    truth_pairs = set(zip(truth.customer_id, truth.article_id))
    observed = np.array([int((u, i) in truth_pairs) for u, i in zip(warm.customer_id, warm.article_id)])
    np.testing.assert_array_equal(warm.target, observed)
    parity = shared_users(history_users, cold, warm)
    spec = contract["feature_spec"]["qC"]
    feature_columns = [*spec["numeric"], *spec["binary"]]
    original_overlap = old_cold.loc[old_cold.customer_id.isin(retained)].sort_values(["customer_id", "article_id"])
    current_overlap = cold.loc[cold.customer_id.isin(retained)].sort_values(["customer_id", "article_id"])
    for key in ("customer_id", "article_id", "target"):
        np.testing.assert_array_equal(original_overlap[key], current_overlap[key])
    np.testing.assert_allclose(original_overlap[feature_columns], current_overlap[feature_columns],
                               rtol=0, atol=1e-12, equal_nan=True)
    comparison = population_comparison(old_cold, cold)
    artifacts = {
        "qC": _save_frame(cold, folder / "qC-features.parquet"),
        "qW": _save_frame(warm, folder / "qW-features.parquet"),
        "selection": selection_artifact,
        "roster": _save_frame(pd.DataFrame({"customer_id": users, "has_mapped_history": active}),
                              folder / "shared-population-roster.parquet"),
        **generated_artifacts,
    }
    audit = {"cutoff": cutoff, "status": "completed", "S_t_users": len(users),
        "S_t_without_mapped_history": int((~active).sum()), **parity,
        "qW_full_S_t_rows_before_training_filter": len(old_warm),
        "shared_support_definition": "H_t = exact frozen W0 S_t intersect pre-cutoff M4-history-available users",
        "no_cold_truth_conditioned_eligibility": True,
        "future_any_purchase_condition": "inherited frozen W0 offline evaluation roster",
        "generation_before_truth_join": True, "timeline": timeline,
        "formal_offline_roster_audit": roster_audit,
        "extra_reconstruction_input_integrity": extra_integrity,
        "M4_B0_frozen": True, "B0_lineage_safe": True,
        "B0_model_label_end": source["b0_lineage"]["model_label_end"],
        "reused_users": len(retained), "restored_users": len(missing),
        "reused_Cold50_identity_score_rank_exact": True,
        "reused_qC_features_max_abs_tolerance": 1e-12,
        "all_shared_original_six_user_state_fields_bit_exact": True,
        "all_shared_original_history_rows_and_days_exact": True,
        "reconstruction_probe": probe, "inference_resources": resources,
        "full_outer_denominator_unchanged": True,
        "no_Cold_users_action": "retain every original W0 item and slot; remain in full S_t evaluation",
        "elapsed_seconds": time.perf_counter() - started,
        "final_week": "not_run", "student_or_B0_training": "not_run"}
    atomic_json(folder / "PREPARED_MANIFEST.json", {"audit": audit, "comparison": comparison, "artifacts": artifacts})
    return {"cold": cold, "warm": warm, "users": users, "history_users": history_users,
            "audit": audit, "comparison": comparison, "artifacts": artifacts}
