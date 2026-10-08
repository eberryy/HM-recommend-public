"""Bounded, resumable P4.3A execution. A deadline pause is not a model failure."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from lightgbm import Booster, LGBMClassifier, LGBMRanker
from threadpoolctl import threadpool_limits

from .p41a_contract import check_identity, identity, read_json, write_json
from .p42_contract import branch_guard
from .p42e_resources import memory, rss
from .p42f_contract import GLOBAL, WINDOWS, earlier, now
from .p42f_core import batches


class BudgetPause(TimeoutError):
    """An approved wall-clock limit was reached; keep all partial evidence."""


def guard(repo, deadline_epoch=None):
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise BudgetPause("Approved execution-session deadline reached; no new work authorized")
    if shutil.disk_usage(repo).free < 15 * 2**30:
        raise MemoryError("D: reserve below 15 GiB; preserve progress, do not downsample")
    if memory().available < 1.5 * 2**30:
        raise MemoryError("System free RAM below 1.5 GiB; preserve other tasks")


def relevance(y):
    y = np.asarray(y)
    return np.where(y >= .10, 4, np.where(y > 0, 3, np.where(y == 0, 1, 0))).astype(np.int32)


def model_order(configs):
    order = {"A": 0, "B": 1, "E": 2, "D": 3, "C": 4, "F": 5}
    return sorted(configs, key=lambda c: (order[c["arm"]], c["id"]))


def pool(repo, root, window, sampling, deadline_epoch):
    """Disk-backed concatenation; never merge one user's groups across dates."""
    from .p43a_data import prepare_date, ensure_meta, META_COLUMNS
    folder = root / "pools" / window / sampling
    receipt = folder / "POOL.json"
    if receipt.exists():
        result = read_json(receipt)
        if result["status"] != "completed":
            raise RuntimeError("Incomplete pool requires explicit recovery")
        return result
    guard(repo, deadline_epoch)
    sources = []
    cutoffs = earlier(WINDOWS[window])
    for t in cutoffs:
        a = prepare_date(repo, root, t, deadline_epoch=deadline_epoch, sampling='A1' if sampling == 'D' else sampling)
        variant = dict(a, **a.get("paths", {}))
        if sampling == 'D':
            variant['meta'] = ensure_meta(repo, root, t, deadline_epoch=deadline_epoch)
        sources.append((t, variant))
    folder.mkdir(parents=True, exist_ok=True)
    rows = sum(int(a["rows"]) for _, a in sources)
    write_json(folder / "POOL_START.json", dict(at=now(), cutoffs=cutoffs, rows=rows, sampling=sampling))
    names = GLOBAL + META_COLUMNS if sampling == 'D' else GLOBAL
    x = np.lib.format.open_memmap(folder / "X.npy", mode="w+", dtype=np.float32, shape=(rows, len(names)))
    y = np.lib.format.open_memmap(folder / "y.npy", mode="w+", dtype=np.float64, shape=(rows,))
    ui_out = np.lib.format.open_memmap(folder / "user_index.npy", mode="w+", dtype=np.int64, shape=(rows,))
    groups = []
    provenance = []
    cursor = 0
    counts = dict(B=0, N=0, H=0)
    for t, a in sources:
        xx = np.load(a["X"], mmap_mode="r")
        yy = np.load(a["y"], mmap_mode="r")
        ui = np.load(a["user_index"], mmap_mode="r")
        hh = np.load(a["hard_neutral"], mmap_mode="r") if sampling == "E" else None
        mx = np.load(a['meta']['paths']['X'], mmap_mode='r') if sampling == 'D' else None
        ei = np.load(a['edge_index'], mmap_mode='r') if sampling == 'D' else None
        local_ui = []
        begin = cursor
        for lo in range(0, len(yy), 100000):
            guard(repo, deadline_epoch)
            hi = min(lo + 100000, len(yy))
            mask = ((yy[lo:hi] != 0) | hh[lo:hi]) if hh is not None else np.ones(hi-lo, bool)
            target = yy[lo:hi][mask]
            ids = ui[lo:hi][mask]
            n = len(target)
            if mx is None:
                x[cursor:cursor+n] = xx[lo:hi][mask]
            else:
                x[cursor:cursor+n, :len(GLOBAL)] = xx[lo:hi][mask]
                x[cursor:cursor+n, len(GLOBAL):] = mx[ei[lo:hi][mask]]
            y[cursor:cursor+n] = target
            ui_out[cursor:cursor+n] = ids
            local_ui.append(ids)
            for key, flag in [("B", target > 0), ("N", target == 0), ("H", target < 0)]:
                counts[key] += int(flag.sum())
            cursor += n
        ids = np.concatenate(local_ui)
        if len(ids) and np.any(np.diff(ids) < 0):
            raise ValueError("Rows must be contiguous by user inside each date")
        sizes = np.diff(np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)])
        assert np.all(sizes > 0) and int(sizes.sum()) == cursor-begin
        groups.extend(sizes.tolist())
        provenance.append(dict(cutoff=t, row_start=begin, row_end=cursor, groups=len(sizes), source=a))
        del xx, yy, ui, hh, mx, ei, local_ui, ids
        gc.collect()
    assert cursor == rows
    x.flush(); y.flush(); ui_out.flush()
    del x, y, ui_out
    np.save(folder / "group.npy", np.asarray(groups, np.int32))
    result = dict(status="completed", at=now(), window=window, cutoff=WINDOWS[window], cutoffs=cutoffs,
                  sampling=sampling, rows=rows, groups=len(groups), counts=counts, group_unit="user-window",
                  X=str(folder/"X.npy"), y=str(folder/"y.npy"), group=str(folder/"group.npy"),
                  user_index=str(folder/"user_index.npy"), sources=provenance, features=names)
    write_json(receipt, result)
    return result


def fit(repo, root, window, config, deadline_epoch, source_receipt):
    from .p43a_resources import snapshot
    folder = root / "models" / config["id"] / window
    result_path = folder / "FIT_RESULT.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["status"] != "completed":
            raise RuntimeError("A non-completed fit cannot be treated as a trained model")
        return Booster(model_file=str(folder/"model.txt")), result
    if (folder / "FIT_START.json").exists():
        raise RuntimeError(f"Prior incomplete fit preserved at {folder}; no silent restart")
    if config["arm"] not in {"A", "B", "D", "E"}:
        raise NotImplementedError("This entry point only fits A/B/D/E edge feature families")
    sample = config["sampling"]
    sample = {"B1": "A1", "B2": "A2", "hard": "E", "hard_neutral": "E", "hard_only": "E"}.get(sample, sample)
    if config['arm'] == 'D':
        sample = 'D'
    p = pool(repo, root, window, sample, deadline_epoch)
    guard(repo, deadline_epoch)
    x = np.load(p["X"], mmap_mode="r")
    yy = np.load(p["y"], mmap_mode="r")
    group = np.load(p["group"])
    names = p['features']
    ranking = config['arm'] in ('A','D')
    y = relevance(yy) if ranking else (yy > 0).astype(np.int32)
    if len(np.unique(y)) < 2:
        raise ValueError("Insufficient training classes, not a valid fit")
    weight = None
    if config["arm"] == "B":
        positive_weight = config.get("positive_weight", config.get("pos_weight"))
        if positive_weight is None:
            raise ValueError("The preregistered positive weight is missing")
        weight = np.where(y == 1, float(positive_weight), 1.)
    params = dict(config["params"])
    if ranking:
        params.update(objective="lambdarank", label_gain=[0, 1, 4, 10, 20])
    else:
        params.update(objective="binary")
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(status="running", at=now(), config=config, params=params, window=window,
                  cutoff=WINDOWS[window], training_cutoffs=p["cutoffs"], rows=len(y), groups=len(group),
                  counts=p["counts"], sample=sample, features=names, row_weight="positive class only" if weight is not None else "all ones",
                  negative_weight=1, source_receipt=source_receipt, pool=str(Path(p["X"]).parent/"POOL.json"))
    record['resources_before_fit']=snapshot()
    write_json(folder/"FIT_START.json", record)
    start = time.perf_counter()

    def bounded_callback(env):
        try:
            guard(repo, deadline_epoch)
        except (TimeoutError, MemoryError):
            env.model.save_model(str(folder/"partial-model.txt"))
            write_json(folder/"FIT_PAUSED.json", dict(at=now(), completed_iterations=env.iteration+1,
                       planned_iterations=params["n_estimators"], valid_for_tournament=False,resources=snapshot()))
            raise
        if (env.iteration+1)%25==0:
            write_json(folder/'FIT_RESOURCE_PROGRESS.json',dict(at=now(),iteration=env.iteration+1,resources=snapshot()))
    bounded_callback.order = 50
    bounded_callback.before_iteration = False
    estimator = LGBMRanker(**params) if ranking else LGBMClassifier(**params)
    try:
        with threadpool_limits(limits=4):
            kwargs = dict(feature_name=names, callbacks=[bounded_callback])
            if ranking:
                kwargs["group"] = group
            if weight is not None:
                kwargs["sample_weight"] = weight
            estimator.fit(x, y, **kwargs)
        booster = estimator.booster_
        assert booster.feature_name() == names
        test = booster.predict(x[:min(256, len(x))], num_threads=4)
        if not np.isfinite(test).all():
            raise ValueError("Non-finite trained scores")
        booster.save_model(str(folder/"model.txt"))
        record.update(status="completed", finished_at=now(), seconds=time.perf_counter()-start,
                      iterations=booster.current_iteration(), trees=booster.num_trees(), peak_gib=rss()/2**30)
        record['resources_after_fit']=snapshot()
        write_json(result_path, record)
        print(f"P4.3A fitted {config['id']} {window}: {len(y):,} rows, {record['seconds']:.1f}s", flush=True)
        return booster, record
    except Exception:
        write_json(folder/"FIT_EXCEPTION.json", dict(at=now(), error=traceback.format_exc(), seconds=time.perf_counter()-start))
        raise
    finally:
        del x, yy, group, y, weight, estimator
        gc.collect()


def predict(repo, root, froot, window, config, booster, deadline_epoch):
    folder = root/"evaluation"/config["id"]/window
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder/"PREDICTION.json"
    data = joblib.load(froot/"prepared"/WINDOWS[window]/"data.joblib")
    if receipt.exists():
        saved = read_json(receipt)
        score = np.load(saved["path"], mmap_mode="r")
        assert tuple(saved["shape"]) == score.shape == (len(data["cold"]), 12)
        return data, score, folder
    guard(repo, deadline_epoch)
    write_json(folder/"PREDICTION_START.json", dict(at=now(), model=str(root/"models"/config["id"]/window/"model.txt")))
    score = np.lib.format.open_memmap(folder/"scores.npy", mode="w+", dtype=np.float64, shape=(len(data["cold"]), 12))
    start = time.perf_counter()
    mx = None
    if config['arm'] == 'D':
        from .p43a_data import ensure_meta, META_COLUMNS
        meta = ensure_meta(repo, root, data['cutoff'], deadline_epoch=deadline_epoch)
        mx = np.load(meta['paths']['X'], mmap_mode='r')
        assert booster.feature_name() == GLOBAL + META_COLUMNS
    for lo, cc, x, _ in batches(data, size=4096, labels=False):
        guard(repo, deadline_epoch)
        if mx is not None:
            x = np.concatenate((x, mx[lo*12:(lo+len(cc))*12]), axis=1)
        pred = booster.predict(x, num_threads=4).reshape(-1, 12)
        if not np.isfinite(pred).all():
            raise ValueError("Non-finite development score")
        score[lo:lo+len(cc)] = pred
    score.flush()
    write_json(receipt, dict(status="completed", at=now(), path=str(folder/"scores.npy"),
                            shape=list(score.shape), seconds=time.perf_counter()-start,
                            scoring_cutoff=data["cutoff"], training_cutoffs=earlier(data["cutoff"]), final_week="not_run"))
    return data, score, folder


def _ledger_state(root, configs, policy_list):
    path = root/"TOURNAMENT_STATE.json"
    if path.exists():
        return read_json(path)
    return dict(stage="P4.3A", status="running", configs={c["id"]: dict(config=c, windows={}, policies={}, status="not_run")
                 for c in configs}, final_week="not_run", started_at=now(), policy_count=len(policy_list),
                 selected_baseline="W0", full_history_run=False)


def run_f(repo, root, contract, deadline_epoch, max_new_windows=0):
    """Complete the frozen 128 heuristics after all trainable families."""
    from .p43a_special import evaluate_f_config
    state = read_json(root/'TOURNAMENT_STATE.json')
    new_windows=0
    for config in contract['F']['configs']:
        guard(repo, deadline_epoch)
        record = state['configs'].setdefault(config['id'], dict(
            config=dict(config, arm='F'), windows={}, policies={}, status='not_run'))
        if record['status'] == 'all_policies_pruned':
            continue
        for wi, window in enumerate(WINDOWS):
            if window in record['windows']:
                continue
            guard(repo, deadline_epoch)
            rows = evaluate_f_config(repo, root, window, config, deadline_epoch)
            if len(rows) != 1 or rows[0]['policy_id'] != config['id']:
                raise ValueError('F receipt does not match its registered config')
            folder = root/'evaluation'/config['id']/window/'policy'
            record['windows'][window] = dict(evaluated_policies=1,
                receipt=str(folder/'EVALUATION.json'), trainable=False)
            record['status'] = 'partial' if wi < 3 else 'completed'
            if wi == 1 and contract['pruning']['enabled']:
                first = record['windows'][next(iter(WINDOWS))]['receipt']
                deltas = [read_json(Path(first))['rows'][0]['delta_map'], rows[0]['delta_map']]
                if all(delta < 0 for delta in deltas) and np.mean(deltas) < -.0015:
                    record['policies'][config['id']] = dict(status='pruned_bad_config')
                    record['status'] = 'all_policies_pruned'
            state['updated_at'] = now()
            write_json(root/'TOURNAMENT_STATE.json', state)
            print(f"P4.3A evaluated {config['id']} {window}, heuristic policy", flush=True)
            new_windows+=1
            if max_new_windows and new_windows>=max_new_windows:
                state['status']='yielded_window_boundary'
                write_json(root/'TOURNAMENT_STATE.json',state)
                return state
            if record['status'] == 'all_policies_pruned':
                break
    return state


def tournament(repo, root, contract, deadline_epoch, source_receipt, max_new_windows=0):
    from .p43a_contract import model_configs
    from .p43a_policy import evaluate_grid, policies
    from .p43a_session import pause_on_gate
    configs = model_order(model_configs())
    policy_list = policies()
    state = _ledger_state(root, configs, policy_list)
    new_windows=0
    froot = repo/"artifacts/phase4/p4-2f-v1-relative-utility-hash10-user-oof"
    try:
        for config in configs:
            if config['arm'] == 'C':
                # C's three heads form one family and must be evaluated jointly.
                continue
            if config["arm"] not in {"A", "B", "D", "E"}:
                raise ValueError('Unexpected registered trainable arm')
            if config["arm"] == "E":
                # Its different challenger policy has a separate implementation,
                # not the universal A/B/D thresholds.
                from .p43a_policy import evaluate_residual
            record = state["configs"][config["id"]]
            if pause_on_gate(root, state, record):
                return state
            for wi, (window, cutoff) in enumerate(WINDOWS.items()):
                if window in record["windows"]:
                    continue
                guard(repo, deadline_epoch)
                active = [p for p in policy_list if record["policies"].get(p["id"], {}).get("status") != "pruned_bad_config"]
                if config["arm"] != "E" and not active:
                    record["status"] = "all_policies_pruned"
                    break
                booster, fitted = fit(repo, root, window, config, deadline_epoch, source_receipt)
                data, score, folder = predict(repo, root, froot, window, config, booster, deadline_epoch)
                if config["arm"] == "E":
                    rows = evaluate_residual(data, score, folder/"policy", deadline_epoch=deadline_epoch)
                else:
                    rows = evaluate_grid(data, score, active, folder/"policy", deadline_epoch=deadline_epoch)
                record["windows"][window] = dict(fit=fitted, evaluated_policies=len(rows), receipt=str(folder/"policy"/"EVALUATION.json"))
                # Full per-policy evidence stays in window receipts. Keeping all
                # window dictionaries in process RAM would duplicate gigabytes.
                if wi == 1 and config["arm"] != "E":
                    first = record["windows"][next(iter(WINDOWS))]["receipt"]
                    first_rows = {r["policy_id"]: r for r in read_json(Path(first))["rows"]}
                    for row in rows:
                        ds = [first_rows[row["policy_id"]]["delta_map"], row["delta_map"]]
                        if np.mean(ds) < -.0015 and all(d < 0 for d in ds):
                            record["policies"][row["policy_id"]] = dict(status="pruned_bad_config")
                record["status"] = "partial" if wi < 3 else "completed"
                write_json(root/"TOURNAMENT_STATE.json", state)
                print(f"P4.3A evaluated {config['id']} {window}, {len(rows)} policies", flush=True)
                del booster, data, score
                gc.collect()
                if pause_on_gate(root, state, record):
                    return state
                new_windows+=1
                if max_new_windows and new_windows>=max_new_windows:
                    state['status']='yielded_window_boundary'
                    return state
            write_json(root/"TOURNAMENT_STATE.json", state)
        else:
            state["status"] = "trainable_edge_arms_completed"
            write_json(root/'TOURNAMENT_STATE.json', state)
            from .p43a_c import run_c
            guard(repo, deadline_epoch)
            state = (run_c(repo, root, deadline_epoch, source_receipt,max_new_windows=max_new_windows)
                     if max_new_windows else run_c(repo, root, deadline_epoch, source_receipt))
            if state['status'] not in ('paused_time_budget', 'paused_resources', 'paused_candidate_numeric_gate','yielded_window_boundary'):
                state = (run_f(repo, root, contract, deadline_epoch,max_new_windows=max_new_windows)
                         if max_new_windows else run_f(repo, root, contract, deadline_epoch))
                if state['status']!='yielded_window_boundary':
                    state['status'] = 'tournament_completed'
    except TimeoutError:
        state.update(status="paused_time_budget", paused_at=now(), reason="Approved execution-session deadline")
    except MemoryError:
        state.update(status="paused_resources", paused_at=now(), error=traceback.format_exc())
    except Exception:
        state.update(status="engineering_failure", paused_at=now(), error=traceback.format_exc())
        write_json(root/f"FAILURE_{int(time.time())}.json", dict(at=now(), error=state["error"]))
        raise
    finally:
        # C/F persist their own finer-grained progress. Never overwrite it with
        # the parent snapshot if a deadline or exception interrupted a call.
        if (root/'TOURNAMENT_STATE.json').exists():
            saved = read_json(root/'TOURNAMENT_STATE.json')
            for key, value in saved['configs'].items():
                if key == 'C-oracle' or value.get('config', {}).get('arm') == 'F':
                    state['configs'][key] = value
        state["updated_at"] = now()
        state["peak_gib"] = rss()/2**30
        write_json(root/"TOURNAMENT_STATE.json", state)
    return state


def execute(repo, command, session_path=None, max_new_windows=0):
    from .p43a_contract import RUN_ID, model_configs, f_configs
    from .p43a_policy import policies
    repo = Path(repo).resolve()
    branch_guard(repo)
    if __import__("subprocess").check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "main":
        raise RuntimeError("P4.3A only runs on the Cold main workspace")
    cpath = repo/"reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json"
    c = read_json(cpath)
    if c["model_configs"] != model_configs() or c["policy_grid"] != policies() or c["F"]["configs"] != f_configs():
        raise ValueError("Runtime search definitions differ from the immutable registered contract")
    root = repo/"artifacts/phase4"/RUN_ID
    root.mkdir(parents=True, exist_ok=True)
    session = None
    if session_path is not None:
        from .p43a_session import load_session
        if command != 'run':
            raise ValueError('Continuation reuses completed oracle, not a new oracle run')
        session = load_session(session_path, root)
    deadline = (session['budget'] if session else c['budget'])['deadline_epoch']
    guard(repo, deadline)
    start = root/"EXECUTION_START.json"
    if not start.exists():
        # These are reused mutable caches, compared with the earlier manifest
        # identities required by the experiment, not routine standalone hashes.
        for record in c["trusted_F_prepared"] + c["authority"]:
            guard(repo, deadline)
            check_identity(record)
        write_json(start, dict(at=now(), contract=identity(cpath), source=[identity(p) for p in sorted((repo/"src/hm_recsys").glob("p43a*.py"))],
                   shared_source=[identity(repo/"src/hm_recsys"/p) for p in ["p42f_core.py", "p42d_stats.py", "metrics.py", "p42_evaluate.py"]],
                   deadline_epoch=deadline, final_week="not_run"))
    else:
        check_identity(read_json(start)["contract"])
    if command == "oracle":
        from .p43a_oracle import run_oracle
        return run_oracle(repo, root, list(WINDOWS.values()), deadline_epoch=deadline)
    oracle_path = repo/"reports/phase4/p4_3a_oracle_headroom.json"
    if not oracle_path.exists() or read_json(oracle_path)["status"] != "completed":
        raise RuntimeError("Run and verify oracle before tournament fits")
    if session:
        old_state=read_json(root/'TOURNAMENT_STATE.json')
        old_state.update(status='running', active_session=session, resumed_at=now(), paused_at=None,reason=None)
        old_state.pop('error',None)
        write_json(root/'TOURNAMENT_STATE.json',old_state)
    source = root/f"SOURCE_RUN_{time.time_ns()}.json"
    archive = root/'source-archive'/source.stem
    archive.mkdir(parents=True, exist_ok=False)
    bound = []
    for p in sorted((repo/'src/hm_recsys').glob('p43a*.py')):
        # Preserve the exact process-start implementation as well as its name.
        # Later predeclared arms may be implemented without erasing this version.
        copied = archive/p.name
        shutil.copy2(p, copied)
        bound.append(dict(identity(copied), original_path=str(p)))
    write_json(source, dict(at=now(), contract=read_json(start)["contract"], source=bound,
               session=identity(Path(session_path)) if session else None,
               reason="Append-only source binding before this process's fits; search contract unchanged"))
    return tournament(repo, root, c, deadline, str(source),max_new_windows=max_new_windows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["oracle", "run"])
    parser.add_argument("--repo", default=".")
    parser.add_argument('--session', default=None)
    parser.add_argument('--max-new-windows',type=int,default=0,choices=[0,1])
    args = parser.parse_args()
    execute(args.repo, args.command, args.session,args.max_new_windows)
