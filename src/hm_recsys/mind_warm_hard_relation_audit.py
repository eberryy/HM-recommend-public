"""Independent P013 replay and post-label challenger/victim factor audit.

This module never trains a model and never changes an action.  Labels are joined
only after the saved score and decision artifacts have been frozen.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_hard_relation as hard
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer
from .mind_warm_baseline import reconstruct_full_baseline


TOL = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def action_delta_tables(baseline: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-user exact AP changes for 0->1 and 1->0 at each Top12 rank."""
    top = baseline[baseline.champion_rank <= 12].sort_values(["customer_id", "champion_rank"])
    users = pd.Index(top.customer_id.to_numpy()[::12])
    require(len(top) == 12 * len(users), "baseline is not a complete Top12 matrix")
    labels = top.target.to_numpy(np.int8).reshape(-1, 12)
    truth = top.truth_count.to_numpy(np.int64).reshape(-1, 12)
    require(np.all(truth == truth[:, :1]), "truth denominator differs within user")
    denom = np.minimum(truth[:, 0], 12).astype(float)
    before = relation.old._mean_ap(top, len(users))
    up = np.zeros_like(labels, dtype=float)
    down = np.zeros_like(labels, dtype=float)
    for position in range(7, 12):
        changed = labels.copy()
        changed[:, position] = 1
        up[:, position] = relation.old._apk_matrix(changed, truth[:, 0]) - relation.old._apk_matrix(labels, truth[:, 0]) if hasattr(relation.old, "_apk_matrix") else np.nan
        changed[:, position] = 0
        down[:, position] = relation.old._apk_matrix(changed, truth[:, 0]) - relation.old._apk_matrix(labels, truth[:, 0]) if hasattr(relation.old, "_apk_matrix") else np.nan
    # Most historical implementations expose scalar AP only.  Replace the
    # optional fast path with a self-contained vectorized replay when absent.
    if np.isnan(up).any() or np.isnan(down).any():
        rank = np.arange(1, 13, dtype=float)
        base_ap = (np.cumsum(labels, axis=1) / rank * labels).sum(1) / denom
        for position in range(7, 12):
            changed = labels.copy(); changed[:, position] = 1
            up[:, position] = (np.cumsum(changed, axis=1) / rank * changed).sum(1) / denom - base_ap
            changed = labels.copy(); changed[:, position] = 0
            down[:, position] = (np.cumsum(changed, axis=1) / rank * changed).sum(1) / denom - base_ap
    require(np.isfinite(up[:, 7:]).all() and np.isfinite(down[:, 7:]).all(), "nonfinite AP delta table")
    return users, labels, truth[:, 0], up, down


def within_user_pairwise_auc(frame: pd.DataFrame) -> dict:
    """Compare saved utility for every beneficial-vs-harmful row from one user."""
    wins = ties = pairs = users = 0
    for _, group in frame.groupby("customer_id", sort=False):
        good = group.loc[group.actual_delta > TOL, "replacement_score"].to_numpy(float)
        bad = group.loc[group.actual_delta < -TOL, "replacement_score"].to_numpy(float)
        if not len(good) or not len(bad):
            continue
        users += 1
        difference = good[:, None] - bad[None, :]
        wins += int((difference > 0).sum())
        ties += int((difference == 0).sum())
        pairs += difference.size
    return {"comparable_users": users, "benefit_harm_pairs": pairs,
            "pairwise_accuracy": (wins + .5 * ties) / pairs if pairs else None}


def factor(frame: pd.DataFrame, total_users: int) -> dict:
    favorite = frame.sort_values(
        ["customer_id", "replacement_score", "challenger_article_id", "victim_article_id"],
        ascending=[True, False, True, True], kind="mergesort",
    ).drop_duplicates("customer_id")
    favorite_keys = favorite[["customer_id", "challenger_article_id", "victim_article_id", "champion_rank"]]
    fixed_c = frame.merge(favorite_keys[["customer_id", "challenger_article_id"]], on=["customer_id", "challenger_article_id"], how="inner")
    fixed_v = frame.merge(favorite_keys[["customer_id", "victim_article_id", "champion_rank"]], on=["customer_id", "victim_article_id", "champion_rank"], how="inner")

    def oracle(values: pd.DataFrame) -> pd.Series:
        return values.groupby("customer_id", sort=False).actual_delta.max().clip(lower=0)

    full, a, b = oracle(frame), oracle(fixed_c), oracle(fixed_v)
    favorite_delta = favorite.set_index("customer_id").actual_delta
    accepted = favorite.replacement_score.to_numpy(float) > 0
    actual = favorite.actual_delta.to_numpy(float)
    full_sum = float(full.sum())
    return {
        "users_with_shortlist": int(frame.customer_id.nunique()),
        "users_with_any_beneficial_pair": int((full > TOL).sum()),
        "full_shortlist_oracle_MAP_delta": full_sum / total_users,
        "A_fixed_model_favorite_challenger_oracle_victim_MAP_delta": float(a.sum() / total_users),
        "B_fixed_model_favorite_victim_oracle_challenger_MAP_delta": float(b.sum() / total_users),
        "A_share_of_full_oracle": float(a.sum() / full_sum) if full_sum else None,
        "B_share_of_full_oracle": float(b.sum() / full_sum) if full_sum else None,
        "favorite_pair_without_rejection": {
            "beneficial_users": int((actual > TOL).sum()), "harmful_users": int((actual < -TOL).sum()),
            "neutral_users": int((np.abs(actual) <= TOL).sum()), "signed_MAP_delta": float(actual.sum() / total_users),
        },
        "saved_positive_gate": {
            "accepted_users": int(accepted.sum()),
            "beneficial_users": int((accepted & (actual > TOL)).sum()),
            "harmful_users": int((accepted & (actual < -TOL)).sum()),
            "signed_MAP_delta": float(actual[accepted].sum() / total_users),
            "rejected_beneficial_favorite_pairs": int(((~accepted) & (actual > TOL)).sum()),
            "rejected_positive_MAP_potential": float(actual[(~accepted) & (actual > TOL)].sum() / total_users),
        },
        "model_favorite_challenger_is_in_some_beneficial_pair_users": int((a > TOL).sum()),
        "model_favorite_victim_has_some_beneficial_challenger_users": int((b > TOL).sum()),
        "within_user_score_separation": within_user_pairwise_auc(frame),
    }


def run() -> dict:
    started = time.perf_counter()
    metrics = relation.read(relation.REPORT / "MIND_WARM_RANK013_METRICS.json")
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK013_CONTRACT.json")
    require(metrics["status"] == "completed_rejected", "completed P013 result required")
    result = {
        "schema": "mind-warm-rank013-independent-factor-audit-v1",
        "status": "running", "new_training": False, "actions_changed": False,
        "selection_used_labels": False, "posthoc_labels_used_only_for_audit": True,
        "final_week": "not_run", "passed": False,
        "definitions_zh": {
            "模型首选商品对": "每个用户在P013已保存的全部“挑战商品×原第8至12名商品”中，按保存的替换分数取最高的一对；这一步不读取当前窗口购买标签。",
            "完整短名单事后上限": "固定005排序器给出的前5个MIND挑战商品，事后用当前窗口真实标签从每用户最多25个替换动作中选AP@12增量最大的动作，并允许不替换；数值为所有评测用户增量均值，仅用于诊断。",
            "A固定挑战商品": "固定P013模型首选的挑战商品，事后只在原第8至12名中选择被替换商品并允许拒绝；用于衡量挑战商品选择质量。",
            "B固定被替换商品": "固定P013模型首选的原商品及位置，事后只在5个挑战商品中选择并允许拒绝；用于衡量被替换商品选择质量。",
            "窗内成对准确率": "只比较同一用户中真实有益动作与真实有害动作的P013替换分数；分母是所有这种有益—有害动作对，0.5表示随机排序。",
        },
        "windows": {},
    }
    for cutoff, training in contract["schedule"].items():
        require(all(date.fromisoformat(c) + timedelta(days=7) <= date.fromisoformat(cutoff) for c in training), "temporal overlap")
        baseline, baseline_meta = reconstruct_full_baseline(cutoff)
        users, labels, truth, up, down = action_delta_tables(baseline)
        scores_path = hard.ART / cutoff / "scores.parquet"
        decisions_path = hard.ART / cutoff / "decisions.parquet"
        pair_path = relation.ART / cutoff / "pairs.parquet"
        with duckdb.connect() as db:
            frame = db.execute("""SELECT s.*,p.challenger_target,p.victim_target
                FROM read_parquet(?) s JOIN read_parquet(?) p
                USING(customer_id,challenger_article_id,victim_article_id,champion_rank)
                ORDER BY customer_id,challenger_article_id,champion_rank""", [str(scores_path), str(pair_path)]).fetchdf()
            saved = db.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id", [str(decisions_path)]).fetchdf()
        require(set(scores := [*relation.KEYS, *relation.PROBS]) == set(frame.columns) - {"challenger_target", "victim_target"}, "score schema drift")
        p = frame[relation.PROBS].to_numpy(float)
        require(np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-6), "invalid probabilities")
        frame["replacement_score"] = (frame.p_benefit - frame.p_harm) / frame.champion_rank
        replay = relation.select_actions(frame[scores])
        pd.testing.assert_frame_equal(replay.sort_values(relation.KEYS).reset_index(drop=True), saved.sort_values(relation.KEYS).reset_index(drop=True), check_dtype=False)
        row = users.get_indexer(frame.customer_id)
        pos = frame.champion_rank.to_numpy(int) - 1
        require((row >= 0).all() and np.array_equal(labels[row, pos], frame.victim_target.to_numpy(np.int8)), "baseline victim mismatch")
        cy = frame.challenger_target.to_numpy(np.int8); vy = frame.victim_target.to_numpy(np.int8)
        frame["actual_delta"] = np.where((cy == 1) & (vy == 0), up[row, pos], np.where((cy == 0) & (vy == 1), down[row, pos], 0.))
        stats = factor(frame, baseline_meta["total_users"])
        expected = metrics["windows"][cutoff]["delta_vs_baseline"]
        require(abs(stats["saved_positive_gate"]["signed_MAP_delta"] - expected) <= TOL, "saved MAP delta does not replay")
        result["windows"][cutoff] = {"training_cutoffs": training, "score_rows": len(frame), "saved_decisions_exactly_reproduced": True,
                                      "baseline_MAP@12": baseline_meta["MAP@12"], **stats}
    result["status"] = "completed_independent_replay"
    result["passed"] = True
    result["elapsed_seconds"] = time.perf_counter() - started
    relation.save(relation.REPORT / "MIND_WARM_RANK013_AUDIT.json", result)
    return result


if __name__ == "__main__":
    run()
