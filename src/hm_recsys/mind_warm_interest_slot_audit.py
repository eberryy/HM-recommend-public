"""Read-only MIND routing-slot and score-distribution audit."""
from __future__ import annotations

import time

import duckdb

from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer


def run():
    started = time.perf_counter()
    result = {
        "schema": "mind-warm-interest-slot-audit-v1", "status": "running", "new_training": False,
        "known_development_labels_posthoc": True, "final_week": "not_run",
        "definitions_zh": {
            "最佳兴趣槽": "MIND模型为用户产生最多三个兴趣向量；对候选商品余弦相似度最高的向量编号称最佳兴趣槽。编号0至2是模型内部槽位，不预先对应固定服装类别，也不天然表示主次兴趣。",
            "正例密度": "某兴趣槽候选中验证周实际购买的用户—商品行数除以该槽全部候选行数；未购买不等于曝光后拒绝。",
            "槽内前5": "先按用户和最佳兴趣槽分组，再按原始MIND相似度取每槽前5；此处只审计候选覆盖，不作为部署策略。"
        },
        "windows": {},
    }
    for cutoff in ("2020-01-22", "2020-03-18", "2020-06-24", "2020-08-19"):
        source = transfer.source(cutoff)
        with duckdb.connect() as db:
            db.execute("SET memory_limit='512MiB'"); db.execute("SET threads=2")
            slots = db.execute("""SELECT cast(mind_best_interest AS INTEGER) slot,
                    count(*) candidate_rows,sum(target) positive_rows,
                    sum(target)::DOUBLE/count(*) positive_density,
                    count(DISTINCT customer_id) users,
                    count(DISTINCT customer_id) FILTER(WHERE target=1) positive_users
                FROM read_parquet(?) WHERE mind_is_new=1 GROUP BY 1 ORDER BY 1""", [str(source)]).fetchdf().to_dict("records")
            top = db.execute("""WITH ranked AS(
                    SELECT *,row_number() OVER(PARTITION BY customer_id,mind_best_interest ORDER BY mind_score DESC,article_id) slot_rank
                    FROM read_parquet(?) WHERE mind_is_new=1)
                SELECT count(*) FILTER(WHERE slot_rank<=5) candidate_rows,
                       sum(target) FILTER(WHERE slot_rank<=5) positive_rows,
                       count(DISTINCT customer_id) FILTER(WHERE slot_rank<=5 AND target=1) positive_users,
                       sum(target) total_positive_rows,
                       count(DISTINCT customer_id) FILTER(WHERE target=1) total_positive_users
                FROM ranked""", [str(source)]).fetchdf().to_dict("records")[0]
            by_count = db.execute("""SELECT cast(mind_user_interest_count AS INTEGER) active_interests,
                    count(*) candidate_rows,sum(target) positive_rows,sum(target)::DOUBLE/count(*) positive_density,
                    count(DISTINCT customer_id) users,count(DISTINCT customer_id) FILTER(WHERE target=1) positive_users
                FROM read_parquet(?) WHERE mind_is_new=1 GROUP BY 1 ORDER BY 1""", [str(source)]).fetchdf().to_dict("records")
        result["windows"][cutoff] = {"by_best_interest_slot": slots, "per_slot_top5_coverage": top, "by_active_interest_count": by_count}
    result.update(status="completed_read_only", elapsed_seconds=time.perf_counter() - started)
    relation.save(relation.REPORT / "MIND_WARM_INTEREST_SLOT_AUDIT.json", result)
    return result


if __name__ == "__main__": run()
