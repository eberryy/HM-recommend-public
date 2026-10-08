"""Independent P018 policy reconstruction and complete Top12 MAP replay."""
from __future__ import annotations

import time
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from . import mind_warm_interest_gate as prior
from . import mind_warm_rank50_gate as trial
from . import mind_warm_relation as relation
from . import mind_warm_relation_audit as independent
from . import mind_warm_transfer as transfer


TOL = 1e-12


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def run():
    started = time.perf_counter()
    metrics_path = relation.REPORT / "MIND_WARM_RANK018_METRICS.json"
    metrics = relation.read(metrics_path)
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK018_CONTRACT.json")
    require(metrics["status"] == "completed_development_pass_pending_independent_replay" and metrics["gate"]["passed"], "provisional P018 pass required")
    result = {"schema": "mind-warm-rank018-independent-replay-v1", "status": "running", "passed": False,
              "new_training": False, "actions_changed": False, "selection_used_labels": False,
              "final_week": "not_run", "checks": {}, "windows": {},
              "definitions_zh": {
                  "独立策略重建": "不调用P018筛选函数，只用P017已冻结动作与截止日前MIND候选字段重新执行“原始MIND名次不大于50”的固定规则，并逐字段对比保存动作。",
                  "完整Top12重放": "从WV3-741原始排序及冻结换位重新构造每个评测用户的12个商品，应用P018动作后按真实标签重新计算每用户AP@12；分母包含未行动用户。",
                  "数值容忍度": "保存指标与独立重算结果的绝对误差必须不超过1e-12。"
              }}
    deltas = []
    for cutoff in contract["schedule"]:
        require(date.fromisoformat(cutoff) < date(2020, 9, 16), "final week prohibited")
        source = transfer.source(cutoff)
        prior_path = prior.ART / cutoff / "decisions.parquet"
        saved_path = trial.ART / cutoff / "decisions.parquet"
        pair_meta = relation.read(relation.ART / cutoff / "PAIRS.json")
        with duckdb.connect() as db:
            db.execute("SET memory_limit='512MiB'"); db.execute("SET threads=2")
            rebuilt = db.execute("""SELECT a.* FROM read_parquet(?) a JOIN read_parquet(?) s
                ON a.customer_id=s.customer_id AND a.challenger_article_id=s.article_id
                WHERE s.mind_is_new=1 AND s.mind_rank<=50 ORDER BY a.customer_id""", [str(prior_path), str(source)]).fetchdf()
            saved = db.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id", [str(saved_path)]).fetchdf()
            pd.testing.assert_frame_equal(rebuilt, saved, check_dtype=False, check_exact=True)
            top, arrays = independent.frozen_baseline(db, pair_meta["baseline"])
            db.execute(f"CREATE OR REPLACE TEMP VIEW labels AS SELECT customer_id,article_id,target FROM read_parquet({independent.parquet_literal(relation.ART / cutoff / 'labels.parquet')})")
            db.register("saved_actions", saved)
            labeled = db.execute("""SELECT a.customer_id,a.challenger_article_id,a.victim_article_id,a.champion_rank,
                    c.target challenger_target,v.target victim_target
                FROM saved_actions a JOIN labels c ON a.customer_id=c.customer_id AND a.challenger_article_id=c.article_id
                JOIN labels v ON a.customer_id=v.customer_id AND a.victim_article_id=v.article_id ORDER BY a.customer_id""").fetchdf()
        require(not saved.customer_id.duplicated().any() and saved.champion_rank.between(8, 12).all(), "action invariant failed")
        users = pd.Index(arrays["users"]); row = users.get_indexer(labeled.customer_id); pos = labeled.champion_rank.to_numpy(int) - 1
        require((row >= 0).all(), "action user outside denominator")
        items = top.article_id.to_numpy().reshape(-1, 12)
        require(np.array_equal(items[row, pos], labeled.victim_article_id.to_numpy()), "victim identity mismatch")
        require(np.array_equal(arrays["labels"][row, pos], labeled.victim_target.to_numpy(np.int8)), "victim label mismatch")
        changed = arrays["labels"].copy(); changed[row, pos] = labeled.challenger_target.to_numpy(np.int8)
        delta = float((independent.ap_matrix(changed, arrays["truth"]) - arrays["aps"]).mean())
        expected = metrics["windows"][cutoff]
        require(abs(delta - expected["delta_vs_baseline"]) <= TOL, "independent MAP differs")
        require(len(saved) == expected["selected_users"] and expected["selection_used_labels"] is False, "saved action count/provenance drift")
        result["checks"].update({cutoff + "_policy_exact": True, cutoff + "_one_action_tail_only": True,
                                  cutoff + "_full_MAP_replay": True, cutoff + "_no_current_label_selection": True})
        result["windows"][cutoff] = {"total_users": len(users), "actions": len(saved), "independent_delta_vs_WV3_741": delta,
                                      "expected_delta": expected["delta_vs_baseline"], "absolute_error": abs(delta - expected["delta_vs_baseline"])}
        deltas.append(delta)
    gate = {"passed": bool(np.mean(deltas) > 0 and all(v >= 0 for v in deltas) and min(deltas) >= -.0002),
            "mean_delta": float(np.mean(deltas)), "nondegrade_windows": sum(v >= 0 for v in deltas), "worst_delta": min(deltas)}
    require(gate == metrics["gate"] and gate["passed"], "gate replay failed")
    result["checks"].update(gate_exact=True, final_week_not_run=metrics["final_week"] == contract["final_week"] == "not_run")
    result.update(status="completed_verified", passed=all(result["checks"].values()), replayed_gate=gate,
                  elapsed_seconds=time.perf_counter() - started)
    require(result["passed"], "P018 independent audit failed")
    audit_path = relation.REPORT / "MIND_WARM_RANK018_VERIFICATION.json"
    relation.save(audit_path, result)
    metrics["status"] = "completed_verified_development_pass"
    metrics["independent_confirmation"] = str(audit_path)
    metrics["independent_confirmation_passed"] = True
    relation.save(metrics_path, metrics)
    return result


if __name__ == "__main__": run()
