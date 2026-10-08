"""Post-label error audit for the frozen P017 action set; no policy changes."""
from __future__ import annotations

import time

import duckdb
import numpy as np

from . import mind_warm_interest_gate as trial
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer


FIELDS = [
    "mind_rank", "mind_score", "mind_score_user_z", "mind_candidate_interest_count",
    "user_history_events_12w", "user_days_since_last_purchase", "item_events_7d",
    "item_events_28d", "item_events_12w", "item_trend_7d_vs_28d",
    "dense_mind_max_cosine", "dense_mind_top2_gap", "dense_history_max_cosine_0_7",
    "dense_history_max_cosine_8_28", "dense_history_max_cosine_29_84",
]
DENSE = [field for field in FIELDS if field.startswith("dense_")]
SOURCE = [field for field in FIELDS if field not in DENSE]


def summary(frame, field):
    out = {}
    for name, mask in {"beneficial": frame.actual_delta > 1e-15, "harmful": frame.actual_delta < -1e-15, "neutral": frame.actual_delta.abs() <= 1e-15}.items():
        values = frame.loc[mask, field].dropna().to_numpy(float)
        out[name] = {"rows": len(values), "median": float(np.median(values)) if len(values) else None,
                     "minimum": float(values.min()) if len(values) else None, "maximum": float(values.max()) if len(values) else None}
    return out


def run():
    started = time.perf_counter()
    result = {"schema": "mind-warm-rank017-error-audit-v1", "status": "running", "new_training": False,
              "policy_selection": False, "known_development_labels_posthoc": True, "final_week": "not_run",
              "definitions_zh": {
                  "非零动作": "执行后用户AP@12相对WV3-741确实上升或下降的P017动作；未来未买挑战商品且被替换商品也未买时为中性动作。",
                  "候选兴趣支持数": "同一MIND候选同时出现在几个有效兴趣向量各自检索结果中的次数，取值1至3；不同于用户有效兴趣向量总数，本审计中的P017用户后者固定为3。",
                  "结果分层": "每个字段分别在有益、有害和中性动作中的非缺失行数、中位数、最小值、最大值；样本极少，只用于判断是否存在跨窗重复差异。"
              }, "windows": {}}
    for cutoff in ("2020-03-18", "2020-06-24", "2020-08-19"):
        evaluated = trial.ART / cutoff / "evaluated.parquet"
        source = transfer.source(cutoff)
        candidates = candidate.ART / cutoff / "candidates.parquet"
        with duckdb.connect() as db:
            db.execute("SET memory_limit='512MiB'"); db.execute("SET threads=2")
            frame = db.execute(f"SELECT e.actual_delta,{','.join('s.' + f for f in SOURCE)},{','.join('c.' + f for f in DENSE)} FROM read_parquet(?) e JOIN read_parquet(?) s ON e.customer_id=s.customer_id AND e.challenger_article_id=s.article_id JOIN read_parquet(?) c ON e.customer_id=c.customer_id AND e.challenger_article_id=c.article_id", [str(evaluated), str(source), str(candidates)]).fetchdf()
        bins = frame.groupby("mind_candidate_interest_count").actual_delta.agg(
            action_rows="size", beneficial=lambda x: int((x > 1e-15).sum()),
            harmful=lambda x: int((x < -1e-15).sum()), neutral=lambda x: int((x.abs() <= 1e-15).sum())
        ).reset_index().to_dict("records")
        result["windows"][cutoff] = {"actions": len(frame), "nonzero_actions": int((frame.actual_delta.abs() > 1e-15).sum()),
                                      "by_candidate_interest_support_count": bins,
                                      "feature_outcome_summaries": {field: summary(frame, field) for field in FIELDS}}
    result.update(status="completed_read_only", elapsed_seconds=time.perf_counter() - started)
    relation.save(relation.REPORT / "MIND_WARM_RANK017_ERROR_AUDIT.json", result)
    return result


if __name__ == "__main__": run()
