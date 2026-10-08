from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .m3 import _peak_working_set_bytes
from .m4_contract import FINAL_CUTOFF, atomic_json, file_identity, validate_warm_baseline


RUN_ID = "m5-3-v1-warm-shortlist-retention"
K_VALUES = (50, 100, 150, 200, 300)
RETENTION_GATE = 0.95


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def choose_k(window_metrics: dict[str, dict[str, Any]]) -> tuple[int, str]:
    for k in K_VALUES[:-1]:
        if all(
            float(rows[str(k)]["warm_positive_retention@k"]) >= RETENTION_GATE
            for rows in window_metrics.values()
        ):
            return k, "smallest_k_passing_0.95_in_all_four_outer_windows"
    return 300, "k_le_200_failed_in_at_least_one_window_fallback_to_300"


def _audit_window(
    *, connection: duckdb.DuckDBPyConnection, evaluation_db: Path,
    transactions_path: Path, cutoff: str,
) -> dict[str, Any]:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError("M5.3 must not read the final week")
    connection.execute("DETACH warm_db") if "warm_db" in {
        row[0] for row in connection.execute("SHOW DATABASES").fetchall()
    } else None
    connection.execute(f"ATTACH {_literal(evaluation_db)} AS warm_db (READ_ONLY)")
    tx = _literal(transactions_path)
    latest = connection.execute(
        f"SELECT max(t_dat) FROM read_parquet({tx}) WHERE t_dat < DATE '{cutoff}'"
    ).fetchone()[0]
    if latest is not None and str(latest) >= cutoff:
        raise RuntimeError(f"M5.3 cutoff leakage: {cutoff}")
    base = """
        WITH item_counts AS (
            SELECT article_id,count(*)::BIGINT AS events
            FROM read_parquet({tx}) WHERE t_dat < DATE '{cutoff}' GROUP BY article_id
        ), ordered AS (
            SELECT p.*,
                   row_number() OVER (
                       PARTITION BY customer_id
                       ORDER BY
                         CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
                         CASE WHEN user_history_events_12w>0 THEN score_anchor END DESC NULLS LAST,
                         candidate_rank,article_id
                   )::INTEGER AS warm_rank,
                   coalesce(c.events,0)::BIGINT AS item_events_before_cutoff,
                   CASE WHEN coalesce(c.events,0)=0 THEN 'strict_cold'
                        WHEN c.events<=5 THEN 'sparse_1_5'
                        WHEN c.events<=20 THEN 'sparse_6_20'
                        ELSE 'warm_21_plus' END AS coldness_bucket
            FROM warm_db.predictions p LEFT JOIN item_counts c USING(article_id)
        )
    """.format(tx=tx, cutoff=cutoff)
    audit = connection.execute(
        base + """
        SELECT count(*)::BIGINT,
               count(DISTINCT (customer_id,article_id))::BIGINT,
               count(DISTINCT customer_id)::BIGINT,
               min(warm_rank)::BIGINT,max(warm_rank)::BIGINT,
               sum(target)::BIGINT,
               count(DISTINCT customer_id) FILTER(WHERE target=1)::BIGINT,
               count(*) FILTER(WHERE warm_rank<1 OR warm_rank>300)::BIGINT
        FROM ordered
        """
    ).fetchone()
    if audit is None:
        raise RuntimeError("M5.3 audit returned no rows")
    rows, unique_rows, users, min_rank, max_rank, total_positives, positive_users, rank_errors = map(int, audit)
    if rows != unique_rows or min_rank != 1 or max_rank > 300 or rank_errors:
        raise RuntimeError(f"M5.3 candidate identity/rank audit failed: {cutoff}")
    by_k: dict[str, Any] = {}
    for k in K_VALUES:
        row = connection.execute(
            base + f"""
            SELECT count(*)::BIGINT AS candidate_rows,
                   sum(target)::BIGINT AS retained_positives,
                   count(DISTINCT customer_id) FILTER(WHERE target=1)::BIGINT AS retained_positive_users
            FROM ordered WHERE warm_rank<={k}
            """
        ).fetchone()
        candidate_rows, retained_positives, retained_positive_users = map(int, row)
        buckets = connection.execute(
            base + f"""
            SELECT coldness_bucket,count(*)::BIGINT AS candidate_rows,
                   sum(target)::BIGINT AS positive_pairs
            FROM ordered WHERE warm_rank<={k}
            GROUP BY coldness_bucket ORDER BY coldness_bucket
            """
        ).fetchall()
        by_k[str(k)] = {
            "warm_top300_positive_pairs": total_positives,
            "retained_positive_pairs": retained_positives,
            "warm_positive_retention@k": retained_positives / max(total_positives, 1),
            "warm_top300_positive_users": positive_users,
            "retained_positive_users": retained_positive_users,
            "truth_user_retention@k": retained_positive_users / max(positive_users, 1),
            "candidate_rows": candidate_rows,
            "positive_density": retained_positives / max(candidate_rows, 1),
            "coldness_bucket_distribution": {
                str(bucket): {
                    "candidate_rows": int(bucket_rows),
                    "positive_pairs": int(bucket_positives),
                    "candidate_row_share": int(bucket_rows) / max(candidate_rows, 1),
                }
                for bucket, bucket_rows, bucket_positives in buckets
            },
        }
    connection.execute("DETACH warm_db")
    return {
        "cutoff": cutoff,
        "candidate_audit": {
            "rows": rows,
            "unique_user_item_rows": unique_rows,
            "users": users,
            "min_warm_rank": min_rank,
            "max_warm_rank": max_rank,
            "latest_history_date": str(latest),
            "cutoff_safe": latest is None or str(latest) < cutoff,
        },
        "k_metrics": by_k,
    }


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# M5.3：Warm 候选短名单保留率门禁",
        "",
        "## 结论",
        "",
        f"- 冻结 `K_warm={result['decision']['k_warm']}`。选择规则：{result['decision']['reason']}。",
        "- 本阶段只审计 Warm-v1 Top300 的正例保留情况，没有训练排序器，也没有使用 outer MAP 选择 K。",
        "- 最终验证周：not_run。",
        "",
        "## 术语与分母",
        "",
        "- **Warm 候选短名单**：按冻结 Warm-v1 最终顺序保留每个用户前 K 个商品；单位是用户—商品候选行。活跃用户按 `score_anchor` 降序，12周无历史用户沿用冻结 `candidate_rank` 顺序。",
        "- **Warm Top300 正例对**：Warm-v1 完整候选中命中随后7天购买 truth 的去重用户—商品对，是 `warm_positive_retention@K` 的分母。",
        "- **warm_positive_retention@K**：TopK 保留正例对数除以 Top300 正例对数。本项目门禁要求四窗均不低于0.95。",
        "- **truth-user retention@K**：TopK 至少仍有一个正例的用户数除以 Top300 至少有一个正例的用户数。它是用户级指标，与正例对保留率分母不同。",
        "- **positive density**：TopK 正例用户—商品对数除以 TopK 全部候选行数。",
        "- **coldness bucket**：按 cutoff 前全历史购买事件数分为 strict_cold=0、sparse_1_5=1至5、sparse_6_20=6至20、warm_21_plus=至少21；表中数量单位均为候选行。",
        "",
        "## 四窗保留率",
        "",
        "| window | K | Top300 positive pairs | retained positive pairs | pair retention | Top300 positive users | retained positive users | user retention | candidate rows | positive density |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        for k in K_VALUES:
            metric = row["k_metrics"][str(k)]
            lines.append(
                f"| {window} | {k} | {metric['warm_top300_positive_pairs']} | "
                f"{metric['retained_positive_pairs']} | {metric['warm_positive_retention@k']:.6f} | "
                f"{metric['warm_top300_positive_users']} | {metric['retained_positive_users']} | "
                f"{metric['truth_user_retention@k']:.6f} | {metric['candidate_rows']} | "
                f"{metric['positive_density']:.8f} |"
            )
    lines.extend(["", "## 冻结 K 的候选冷度分布", ""])
    lines.extend([
        "| window | bucket | candidate rows | row share | positive pairs |",
        "|---|---|---:|---:|---:|",
    ])
    selected = str(result["decision"]["k_warm"])
    for window, row in result["windows"].items():
        for bucket, values in row["k_metrics"][selected]["coldness_bucket_distribution"].items():
            lines.append(
                f"| {window} | {bucket} | {values['candidate_rows']} | "
                f"{values['candidate_row_share']:.6f} | {values['positive_pairs']} |"
            )
    lines.extend([
        "",
        "## 审计边界",
        "",
        "- 候选用户—商品唯一性、Warm 排名范围与 `latest history < cutoff` 均为硬检查。",
        "- 商品全集仍采用 optimistic all-articles 假设；这里的冷度不代表当时真实可售。",
        f"- 运行耗时 {result['resources']['elapsed_seconds']:.2f} 秒；峰值工作集 {result['resources']['peak_working_set_bytes'] / 2**30:.2f} GiB。",
        "- 本门禁不证明排序质量提升，只为 M5.4 固定最小且保留充分正例的 Warm 候选预算。",
        "",
    ])
    return "\n".join(lines)


def run(*, source_root: Path, report_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    metrics_path = source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json"
    warm = validate_warm_baseline(metrics_path)
    m3 = json.loads(metrics_path.read_text(encoding="utf-8"))
    transactions = Path(m3["inputs"]["transactions"]["path"])
    windows: dict[str, Any] = {}
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=8")
        connection.execute("SET memory_limit='12GB'")
        for window, row in m3["development"].items():
            cutoff = row["protocol"]["outer_validation"]
            windows[window] = _audit_window(
                connection=connection,
                evaluation_db=Path(row["scoring"]["evaluation_db"]),
                transactions_path=transactions,
                cutoff=cutoff,
            )
    finally:
        connection.close()
    k_warm, reason = choose_k({name: row["k_metrics"] for name, row in windows.items()})
    result = {
        "schema_version": "m5.3-warm-shortlist-retention-v1",
        "stage": "M5.3",
        "status": "measured",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "warm_baseline": "M3.3 anchor__inactive_rrf Top300",
            "k_values": list(K_VALUES),
            "selection": "smallest K with pair retention >=0.95 in every outer window; otherwise K=300",
            "outer_map_used_for_selection": False,
            "final_week": "not_run",
        },
        "inputs": {
            "m3_3_metrics": file_identity(metrics_path),
            "transactions": file_identity(transactions),
            "warm_evaluation_dbs": warm["evaluation_dbs"],
        },
        "windows": windows,
        "decision": {"k_warm": k_warm, "reason": reason, "gate_passed": True},
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
        },
        "final_week": "not_run",
    }
    report_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(report_dir / "M5_3_metrics.json", result)
    (report_dir / "M5_3_FINAL.md").write_text(_render_report(result), encoding="utf-8")
    return result
