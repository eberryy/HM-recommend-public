"""Independent replay of the frozen MIND-WARM-FINAL-001 action file."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-FINAL-001"
REPORT = ROOT / "reports/mind_warm_side"
TX = ROOT / "data/interim/audit/transactions.parquet"
BASE100 = ROOT / "artifacts/m1_5/m1-5-v11-full-v8-anchored/candidates.parquet"
OUTPUT = REPORT / "MIND_WARM_FINAL001_VERIFICATION.json"
KEYS = ["customer_id", "challenger_article_id", "victim_article_id", "champion_rank"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def exact_map(frame: pd.DataFrame, total_users: int) -> float:
    frame = frame.sort_values(["customer_id", "rank", "article_id"], kind="mergesort").copy()
    frame["prefix_hits"] = frame.groupby("customer_id", sort=False).target.cumsum()
    frame["ap_term"] = frame.target * frame.prefix_hits / frame["rank"]
    numerator = frame.groupby("customer_id", sort=False).ap_term.sum()
    denominator = frame.groupby("customer_id", sort=False).truth_count.first().clip(lower=1, upper=12)
    return float((numerator / denominator).sum() / total_users)


def main() -> dict:
    metrics = load_json(REPORT / "MIND_WARM_FINAL001_METRICS.json")
    manifest = load_json(ART / "ACTION_MANIFEST.json")
    if not manifest.get("actions_frozen") or manifest.get("final_week_status") != "evaluated_once":
        raise AssertionError("completed frozen-action manifest required")
    with duckdb.connect() as db:
        base = db.execute("SELECT customer_id,article_id,champion_rank AS rank FROM read_parquet(?) WHERE champion_rank<=12 ORDER BY customer_id,rank,article_id", [str(ART / "wv3-741-top50.parquet")]).fetchdf()
        actions = db.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id", [str(ART / "p018-frozen-actions.parquet")]).fetchdf()
        p005 = db.execute("SELECT * FROM read_parquet(?)", [str(ART / "p005-actions.parquet")]).fetchdf()
        p006 = db.execute("SELECT * FROM read_parquet(?)", [str(ART / "p006-actions.parquet")]).fetchdf()
        winners = db.execute("SELECT customer_id,article_id,mind_rank,interest_count FROM read_parquet(?)", [str(ART / "p005-order.parquet")]).fetchdf()
        truth = db.execute("""SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet(?) t SEMI JOIN (SELECT DISTINCT customer_id FROM read_parquet(?)) u USING(customer_id)
            WHERE t.t_dat>=DATE '2020-09-16' AND t.t_dat<DATE '2020-09-23'""", [str(TX), str(BASE100)]).fetchdf()
    action_keys = set(map(tuple, actions[KEYS].itertuples(index=False, name=None)))
    expected_keys = set(map(tuple, p005[KEYS].merge(p006[KEYS], on=KEYS, validate="one_to_one")[KEYS].itertuples(index=False, name=None)))
    winner_check = actions[["customer_id", "challenger_article_id"]].merge(
        winners.rename(columns={"article_id": "challenger_article_id"}),
        on=["customer_id", "challenger_article_id"], validate="one_to_one",
    )
    truth_key = pd.MultiIndex.from_frame(truth)
    truth_count = truth.groupby("customer_id").size().rename("truth_count")
    base["target"] = pd.MultiIndex.from_frame(base[["customer_id", "article_id"]]).isin(truth_key).astype(np.uint8)
    base = base.merge(truth_count, on="customer_id", how="left", validate="many_to_one")
    replacement = actions[["customer_id", "champion_rank", "challenger_article_id"]].rename(columns={"champion_rank": "rank"})
    final = base.merge(replacement, on=["customer_id", "rank"], how="left", validate="one_to_one")
    final["article_id"] = final.challenger_article_id.fillna(final.article_id)
    final["target"] = pd.MultiIndex.from_frame(final[["customer_id", "article_id"]]).isin(truth_key).astype(np.uint8)
    baseline_map = exact_map(base, 68_984)
    p018_map = exact_map(final, 68_984)
    forbidden = {"target", "truth_count", "challenger_target", "victim_target", "actual_delta"}
    checks = {
        "complete_user_denominator": base.customer_id.nunique() == 68_984,
        "twelve_rows_per_user": bool((base.groupby("customer_id").size() == 12).all() and (final.groupby("customer_id").size() == 12).all()),
        "final_item_identity_unique": not final.duplicated(["customer_id", "article_id"]).any(),
        "protected_ranks_1_7_unchanged": base[base["rank"] <= 7].article_id.tolist() == final[final["rank"] <= 7].article_id.tolist(),
        "maximum_one_action_per_user": not actions.duplicated("customer_id").any(),
        "victim_positions_8_12_only": bool(actions.champion_rank.between(8, 12).all()),
        "p009_exact_action_intersection": action_keys == expected_keys,
        "p017_exact_three_interest_gate": bool((winner_check.interest_count == 3).all()),
        "p018_exact_raw_mind_top50_gate": bool((winner_check.mind_rank <= 50).all()),
        "forbidden_action_fields_absent": not bool(forbidden & set(actions.columns)),
        "baseline_map_matches": abs(baseline_map - metrics["baseline_MAP@12"]) <= 1e-12,
        "p018_map_matches": abs(p018_map - metrics["P018_MAP@12"]) <= 1e-12,
        "delta_matches": abs((p018_map - baseline_map) - metrics["delta_MAP@12"]) <= 1e-12,
    }
    result = {
        "schema": "mind-warm-final001-independent-verification-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "independent_of_primary_ap_helpers": True,
        "checks": checks, "passed": all(checks.values()),
        "replayed": {"baseline_MAP@12": baseline_map, "P018_MAP@12": p018_map,
                     "delta_MAP@12": p018_map - baseline_map, "actions": len(actions),
                     "truth_pairs": len(truth), "users": base.customer_id.nunique()},
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise AssertionError({name: value for name, value in checks.items() if not value})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
