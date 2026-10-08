"""Read-only fixed candidate-rank consensus audit across completed OOF models."""
from __future__ import annotations

import time

import duckdb
import numpy as np
import pandas as pd

from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer


SOURCES = {
    "rank003": transfer.ART.parent / "MIND-WARM-RANK-003",
    "rank005": transfer.ART / "MIND-WARM-RANK-005",
    "rank011": transfer.ART / "MIND-WARM-RANK-011",
    "rank012": transfer.ART / "MIND-WARM-RANK-012",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def run() -> dict:
    started = time.perf_counter()
    result = {
        "schema": "mind-warm-candidate-consensus-audit-v1", "status": "running", "new_training": False,
        "policy_or_parameter_search": False, "final_week": "not_run",
        "fixed_before_label_read": {
            "sources": list(SOURCES),
            "ordering": "先按四个训练外排序器中进入前5的票数降序，再按四者名次中位数升序，最后article_id升序；所有来源等权。",
        },
        "definitions_zh": {
            "训练外排序器": "行业OOF思想的时间版本；每个候选名次由只使用更早截止日标签训练的模型产生。本审计使用003、005、011、012四个已完成模型。",
            "前5票数": "本项目自定义一致性统计：一个用户—商品候选分别在四个排序器中进入前5的次数，取值0至4；分母固定为四个排序器。",
            "名次中位数": "行业通用中位数统计；同一用户—商品在四个排序器中的四个名次排序后取中间两项均值，用于在票数相同时选择更稳定靠前的候选。",
            "条件Hit@K": "候选池至少存在一个真实正例的用户中，一致性排序前K至少含一个正例的用户比例；分母不是全部评测用户。"
        },
        "windows": {},
    }
    for cutoff in ("2020-03-18", "2020-06-24", "2020-08-19"):
        paths = {name: root / cutoff / ("candidate_ranker-order.parquet" if name == "rank003" else "order.parquet") for name, root in SOURCES.items()}
        require(all(path.exists() for path in paths.values()), f"missing completed order for {cutoff}")
        with duckdb.connect() as db:
            db.execute("SET memory_limit='1GiB'"); db.execute("SET threads=2")
            query = "SELECT a.customer_id,a.article_id," + ",".join(f"{alias}.rank {name}" for name, alias in zip(paths, ["a", "b", "c", "d"])) + \
                    " FROM read_parquet(?) a JOIN read_parquet(?) b USING(customer_id,article_id) JOIN read_parquet(?) c USING(customer_id,article_id) JOIN read_parquet(?) d USING(customer_id,article_id)"
            frame = db.execute(query, [str(path) for path in paths.values()]).fetchdf()
            labels = db.execute("SELECT customer_id,article_id,target FROM read_parquet(?)", [str(transfer.ART.parent / "MIND-WARM-RANK-003" / cutoff / "candidates.parquet")]).fetchdf()
        rank_cols = list(paths)
        require(not frame.duplicated(candidate.KEYS).any() and len(frame) == len(labels), "candidate identity mismatch")
        ranks = frame[rank_cols].to_numpy(int)
        frame["top5_votes"] = (ranks <= 5).sum(1)
        frame["median_rank"] = np.median(ranks, axis=1)
        frame = frame.sort_values(["customer_id", "top5_votes", "median_rank", "article_id"], ascending=[True, False, True, True], kind="mergesort")
        frame["rank"] = frame.groupby("customer_id", sort=False).cumcount() + 1
        # Convert the fixed lexicographic order to an arbitrary finite score;
        # the saved rank, not this score's magnitude, is the object of audit.
        frame["score"] = frame.top5_votes * 1000.0 - frame.median_rank
        order = frame[candidate.KEYS + ["score", "rank"]]
        metrics = candidate.candidate_metrics(order, labels, relation.read(transfer.ART.parent / "MIND-WARM-RANK-003" / cutoff / "INPUT.json")["total_users"])
        base = relation.read(relation.REPORT / "MIND_WARM_RANK005_METRICS.json")["windows"][cutoff]["internal"]
        result["windows"][cutoff] = {"candidate_rows": len(frame), "identity_conserved": True, "consensus": metrics,
                                      "delta_vs_005_conditional_Hit1": metrics["conditional_hit_at_1"] - base["conditional_hit_at_1"],
                                      "delta_vs_005_conditional_Hit5": metrics["conditional_hit_at_5"] - base["conditional_hit_at_5"]}
    d1 = [v["delta_vs_005_conditional_Hit1"] for v in result["windows"].values()]
    d5 = [v["delta_vs_005_conditional_Hit5"] for v in result["windows"].values()]
    result["summary"] = {"Hit1_mean_delta": float(np.mean(d1)), "Hit1_nondegrade_windows": sum(v >= 0 for v in d1),
                         "Hit5_mean_delta": float(np.mean(d5)), "Hit5_nondegrade_windows": sum(v >= 0 for v in d5)}
    result.update(status="completed_read_only", elapsed_seconds=time.perf_counter() - started)
    relation.save(relation.REPORT / "MIND_WARM_CANDIDATE_CONSENSUS_AUDIT.json", result)
    return result


if __name__ == "__main__": run()
