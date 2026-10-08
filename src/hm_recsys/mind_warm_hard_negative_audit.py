"""Label-bearing audit of MIND positives versus their within-user competitors."""
import time
import duckdb

from . import mind_warm_transfer as transfer
from . import mind_warm_relation as relation
from . import mind_warm_candidate_ranker as candidate


SCORES = {
    "005_purchase_score": "score005",
    "raw_MIND_score": "mind_score",
    "raw_MIND_user_z": "mind_score_user_z",
    "inverse_MIND_rank": "-mind_rank",
    "product_code_events_12w": "user_product_code_events_12w",
    "negative_product_code_days_since": "-user_product_code_days_since",
    "product_type_events_12w": "user_product_type_events_12w",
    "negative_product_type_days_since": "-user_product_type_days_since",
    "department_events_12w": "user_department_events_12w",
    "garment_events_12w": "user_garment_events_12w",
    "colour_events_12w": "user_colour_events_12w",
    "recent_item_events": "item_events_7d",
    "MIND_interest_count": "mind_candidate_interest_count",
}


def run():
    started = time.perf_counter()
    transfer.budget()
    result = {"schema": "mind-hard-negative-audit-v1", "windows": {}, "new_training": False,
              "labels": "known chronological development labels; post-hoc diagnostic only",
              "pairwise_unit": "ordered positive-negative candidate pair within the same user MIND-only pool",
              "hard_negative": "the highest005-scored non-purchased MIND candidate for a user who has at least one purchased MIND candidate",
              "final_week": "not_run"}
    with candidate.connection() as db:
        db.execute("SET memory_limit='3GiB'")
        for cutoff in ("2020-03-18", "2020-06-24", "2020-08-19"):
            transfer.budget()
            source = relation.ART / "aggregate_adapter" / cutoff / "features.parquet"
            cand = candidate.ART / cutoff / "candidates.parquet"
            sp = relation.old._sql_path(source)
            cp = relation.old._sql_path(cand)
            op = relation.old._sql_path(transfer.ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet")
            db.execute(f"CREATE OR REPLACE TEMP VIEW enriched AS SELECT s.*,o.score score005,o.rank rank005 FROM read_parquet('{sp}') s JOIN read_parquet('{cp}') c USING(customer_id,article_id) LEFT JOIN read_parquet('{op}') o USING(customer_id,article_id) WHERE s.mind_is_new=1")
            counts = db.execute("SELECT count(*) candidate_rows,sum(target) positive_rows,count(DISTINCT customer_id) users,count(DISTINCT customer_id) FILTER(WHERE target=1) positive_users FROM enriched").fetchdf().to_dict("records")[0]
            hard_fields = []
            for name, expression in SCORES.items():
                sign, field = ("-", expression[1:]) if expression.startswith("-") else ("", expression)
                hard_fields.extend([f"{sign}p.{field} p_{name}", f"{sign}n.{field} n_{name}"])
            db.execute("CREATE OR REPLACE TEMP VIEW hard AS WITH pos AS (SELECT * EXCLUDE(rn) FROM (SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score005 DESC,article_id) rn FROM enriched WHERE target=1) WHERE rn=1), neg AS (SELECT * EXCLUDE(rn) FROM (SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score005 DESC,article_id) rn FROM enriched WHERE target=0) WHERE rn=1) SELECT p.customer_id," + ",".join(hard_fields) + " FROM pos p JOIN neg n USING(customer_id)")
            diagnostics = {}
            for name, expression in SCORES.items():
                def qualified(alias):
                    return ("-" if expression.startswith("-") else "") + alias + "." + expression.lstrip("-")
                positive, negative = qualified("p"), qualified("n")
                top = db.execute(f"WITH r AS (SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY {expression} DESC NULLS LAST,article_id) r FROM enriched) SELECT count(DISTINCT customer_id) FILTER(WHERE target=1 AND r=1),count(DISTINCT customer_id) FILTER(WHERE target=1 AND r<=5) FROM r").fetchone()
                pair = db.execute(f"SELECT count(*) pair_count,avg(CASE WHEN ({positive})>({negative}) THEN 1 WHEN ({positive})=({negative}) THEN .5 ELSE 0 END) pairwise_accuracy FROM enriched p JOIN enriched n ON p.customer_id=n.customer_id AND p.target=1 AND n.target=0 WHERE ({positive}) IS NOT NULL AND ({negative}) IS NOT NULL").fetchdf().to_dict("records")[0]
                hard = db.execute(f"SELECT count(*) compared_users,avg(CASE WHEN p_{name}>n_{name} THEN 1 WHEN p_{name}=n_{name} THEN .5 ELSE 0 END) positive_beats_005_hard_negative,median(p_{name}-n_{name}) median_score_difference FROM hard WHERE p_{name} IS NOT NULL AND n_{name} IS NOT NULL").fetchdf().to_dict("records")[0]
                diagnostics[name] = {"hit_users_at_1": top[0], "hit_users_at_5": top[1], **pair, **hard}
            result["windows"][cutoff] = {"counts": counts, "scores": diagnostics}
    result["elapsed_seconds"] = time.perf_counter() - started
    relation.save(relation.REPORT / "MIND_WARM_HARD_NEGATIVE_AUDIT.json", result)
    print({c: {k: {"h1": v["hit_users_at_1"], "h5": v["hit_users_at_5"], "auc": v["pairwise_accuracy"], "hard": v["positive_beats_005_hard_negative"]} for k, v in w["scores"].items()} for c, w in result["windows"].items()}, flush=True)
    return result


if __name__ == "__main__":
    run()
