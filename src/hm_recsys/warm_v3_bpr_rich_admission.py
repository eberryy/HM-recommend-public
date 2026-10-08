"""WV3-730/731: audit and test rich-propensity admission of BPR-only candidates."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal, load_parquet, save_parquet
from .warm_v3_bpr_safe_admission import replace_rank12
from .warm_v3_expert import screening_gate
from .warm_v3_pool import ART as POOL_ART
from .warm_v3_residual_admission import INNER
from .warm_v3_rich_feature_audit import EXTRA_FEATURES
from .warm_v3_rich_propensity import FEATURES, rich_candidate_frame


AUDIT_TRIAL = "WV3-730"
TRIAL = "WV3-731"
AUDIT_CONTRACT = common.REPORT / "WV3-730_BPR_RICH_FEASIBILITY_CONTRACT.json"
AUDIT_REPORT = common.REPORT / "WV3-730_BPR_RICH_FEASIBILITY_AUDIT.json"
AUDIT_MARKDOWN = common.REPORT / "WV3-730_BPR_RICH_FEASIBILITY_AUDIT.md"
CONTRACT = common.REPORT / "WV3-731_BPR_RICH_ADMISSION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-731_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-731_SCREEN.md"
MODEL_PATH = common.ART / "WV3-661" / "MODEL.txt"
AUDIT_ART = common.ART / AUDIT_TRIAL

PAIR_FEATURES = [
    "repurchase_present",
    "item2vec_is_new",
    "user_item_events_12w",
    "user_item_days_since_last_purchase",
    "user_product_type_share_12w",
    "user_department_share_12w",
    "item_days_since_last_sale",
]
RAW_COLUMNS = list(
    dict.fromkeys(
        [
            "user_history_events_12w",
            "user_unique_items_12w",
            "user_days_since_last_purchase",
            *PAIR_FEATURES,
            *EXTRA_FEATURES,
        ]
    )
)
DECISION_COLUMNS = [
    "customer_id",
    "article_id",
    "expanded_rank",
    "challenger_score",
    "victim_article_id",
    "victim_score",
]


def feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the exact 99-column WV3-661 input transform from ranked raw rows."""
    frame = raw.copy()
    n = frame.candidate_count.to_numpy(np.float64)
    r0 = frame.r0.to_numpy(np.float64)
    r1 = frame.r1.to_numpy(np.float64)
    rf = frame.rf.to_numpy(np.float64)
    std = frame.latent_std.fillna(0.0).to_numpy(np.float64)
    latent_z = (frame.latent.to_numpy(np.float64) - frame.latent_mean.to_numpy(np.float64)) / (std + 1e-6)
    frame["r0_reciprocal"] = 60.0 / (60.0 + r0)
    frame["r1_reciprocal"] = 60.0 / (60.0 + r1)
    frame["r0_fraction"] = r0 / n
    frame["r1_fraction"] = r1 / n
    frame["rank_gap"] = (r0 - r1) / n
    frame["r0_top12"] = (r0 <= 12).astype(np.float32)
    frame["r1_top12"] = (r1 <= 12).astype(np.float32)
    frame["r0_top50"] = (r0 <= 50).astype(np.float32)
    frame["r1_top50"] = (r1 <= 50).astype(np.float32)
    frame["latent_within_user_z"] = np.clip(np.nan_to_num(latent_z, nan=0.0), -10.0, 10.0)
    frame["bpr_missing"] = frame.missing
    frame["log_history"] = np.log1p(frame.user_history_events_12w.clip(0, 10000))
    frame["log_unique"] = np.log1p(frame.user_unique_items_12w.clip(0, 10000))
    frame["log_recency"] = np.log1p(frame.user_days_since_last_purchase.clip(0, 10000))
    for column in PAIR_FEATURES:
        if "events" in column or "days" in column:
            frame[column] = np.log1p(frame[column].clip(0, 10000))
    frame["rf_reciprocal"] = 1.0 / (60.0 + rf)
    frame["rf_fraction"] = rf / 50.0
    frame["rf_victim_band"] = ((rf >= 8) & (rf <= 12)).astype(np.float32)
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    frame[FEATURES] = values
    if not np.isfinite(frame[FEATURES].to_numpy(np.float64)).all():
        raise AssertionError("Reconstructed rich feature matrix is not finite")
    return frame


def raw_select(prefix: str) -> str:
    return ",".join(f"f.{name}" for name in RAW_COLUMNS)


def expanded_raw(name: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    ranks_path = POOL_ART / "failure_2x2" / name / "B_ranks.parquet"
    target_path = POOL_ART / "target" / cutoff / "features.parquet"
    bpr_path = POOL_ART / cutoff / "bpr_features.parquet"
    for path in (ranks_path, target_path, bpr_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    select_raw = raw_select("f")
    with connection() as con:
        coverage = con.execute(
            f"""SELECT count(*) total_rows,
                count(p.customer_id) bpr_feature_rows,
                count(f.customer_id) target_feature_rows
            FROM read_parquet({literal(ranks_path)}) b
            LEFT JOIN read_parquet({literal(bpr_path)}) p USING(customer_id,article_id)
            LEFT JOIN read_parquet({literal(target_path)}) f USING(customer_id,article_id)"""
        ).fetchone()
        raw = con.execute(
            f"""WITH joined AS (
                SELECT b.customer_id,b.article_id,b.candidate_rank,b.target,b.truth_count,b.rf stored_rf,
                    b.score_e0,b.score_e1,p.wv2_bpr_user_item_score latent,
                    p.wv2_bpr_unavailable missing,f.bpr_is_new,{select_raw},
                    row_number() OVER(PARTITION BY b.customer_id ORDER BY b.score_e0 DESC,b.candidate_rank,b.article_id) r0,
                    row_number() OVER(PARTITION BY b.customer_id ORDER BY b.score_e1 DESC,b.candidate_rank,b.article_id) r1,
                    count(*) OVER(PARTITION BY b.customer_id) candidate_count,
                    avg(p.wv2_bpr_user_item_score) OVER(PARTITION BY b.customer_id) latent_mean,
                    stddev_samp(p.wv2_bpr_user_item_score) OVER(PARTITION BY b.customer_id) latent_std
                FROM read_parquet({literal(ranks_path)}) b
                JOIN read_parquet({literal(bpr_path)}) p USING(customer_id,article_id)
                JOIN read_parquet({literal(target_path)}) f USING(customer_id,article_id)
            ), fused AS (
                SELECT *,1.0/(60+r0)+1.0/(60+r1) frozen_rrf_score FROM joined
            ), ranked AS (
                SELECT *,row_number() OVER(PARTITION BY customer_id
                    ORDER BY frozen_rrf_score DESC,candidate_rank,article_id) learned_rf FROM fused
            ), final AS (
                SELECT *,CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE learned_rf END rf
                FROM ranked
            )
            SELECT * FROM final WHERE bpr_is_new=1 AND rf<=50
            ORDER BY customer_id,rf,article_id"""
        ).fetchdf()
        rank_errors = con.execute(
            f"""WITH joined AS (
                SELECT b.*,f.user_history_events_12w,
                    row_number() OVER(PARTITION BY b.customer_id ORDER BY b.score_e0 DESC,b.candidate_rank,b.article_id) r0,
                    row_number() OVER(PARTITION BY b.customer_id ORDER BY b.score_e1 DESC,b.candidate_rank,b.article_id) r1
                FROM read_parquet({literal(ranks_path)}) b
                JOIN read_parquet({literal(target_path)}) f USING(customer_id,article_id)
            ), fused AS (
                SELECT *,1.0/(60+r0)+1.0/(60+r1) frozen_rrf_score FROM joined
            ), ranked AS (
                SELECT *,row_number() OVER(PARTITION BY customer_id
                    ORDER BY frozen_rrf_score DESC,candidate_rank,article_id) learned_rf FROM fused
            )
            SELECT count(*) FROM ranked
            WHERE rf<>CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE learned_rf END"""
        ).fetchone()[0]
    audit = {
        "ranks_path": str(ranks_path),
        "target_feature_path": str(target_path),
        "bpr_feature_path": str(bpr_path),
        "total_expanded_rows": int(coverage[0]),
        "bpr_feature_rows": int(coverage[1]),
        "target_feature_rows": int(coverage[2]),
        "join_coverage_complete": coverage[0] == coverage[1] == coverage[2],
        "stored_expanded_rank_replay_errors": int(rank_errors),
        "bpr_only_top50_rows": len(raw),
        "bpr_only_top50_users": raw.customer_id.nunique(),
    }
    return raw, audit


def original_raw(cutoff: str) -> pd.DataFrame:
    meta = read(common.ART / "candidate_gate_data" / cutoff / "DATA.json")
    keys_path = Path(meta["keys"])
    target_path = POOL_ART / "target" / cutoff / "features.parquet"
    bpr_path = POOL_ART / cutoff / "bpr_features.parquet"
    select_raw = raw_select("f")
    with connection() as con:
        return con.execute(
            f"""WITH joined AS (
                SELECT k.customer_id,k.article_id,k.candidate_rank,k.target,k.truth_count,k.r0,k.r1,k.rf,
                    p.wv2_bpr_user_item_score latent,p.wv2_bpr_unavailable missing,{select_raw},
                    count(*) OVER(PARTITION BY k.customer_id) candidate_count,
                    avg(p.wv2_bpr_user_item_score) OVER(PARTITION BY k.customer_id) latent_mean,
                    stddev_samp(p.wv2_bpr_user_item_score) OVER(PARTITION BY k.customer_id) latent_std
                FROM read_parquet({literal(keys_path)}) k
                JOIN read_parquet({literal(bpr_path)}) p USING(customer_id,article_id)
                JOIN read_parquet({literal(target_path)}) f USING(customer_id,article_id)
            ) SELECT * FROM joined WHERE rf<=50 ORDER BY customer_id,rf,article_id"""
        ).fetchdf()


def parity_audit(cutoff: str) -> dict:
    rebuilt = feature_frame(original_raw(cutoff)).sort_values(["customer_id", "article_id"]).reset_index(drop=True)
    reference, _ = rich_candidate_frame(cutoff, rank_min=1)
    reference = reference.sort_values(["customer_id", "article_id"]).reset_index(drop=True)
    if not rebuilt[["customer_id", "article_id"]].equals(reference[["customer_id", "article_id"]]):
        raise AssertionError("Original-pool item identities changed during expanded-cache reconstruction")
    differences = np.abs(
        rebuilt[FEATURES].to_numpy(np.float32) - reference[FEATURES].to_numpy(np.float32)
    )
    max_abs = float(differences.max(initial=0.0))
    return {
        "rows": len(rebuilt),
        "features": len(FEATURES),
        "max_absolute_feature_difference": max_abs,
        "exact_within_float32_tolerance": max_abs <= 1e-6,
    }


def current_rank12(name: str, cutoff: str, model: lgb.Booster) -> pd.DataFrame:
    candidates, _ = rich_candidate_frame(cutoff, rank_min=1)
    candidates = candidates.sort_values(["customer_id", "rf", "article_id"], kind="mergesort")
    candidates["victim_score"] = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    swaps = load_parquet(common.ART / "WV3-721" / name / "inner" / "swaps.parquet")
    swap_groups = {customer: group for customer, group in swaps.groupby("customer_id", sort=False)}
    rows = []
    for customer, group in candidates.groupby("customer_id", sort=False):
        ordered = group.sort_values("rf", kind="mergesort").reset_index(drop=True)
        if len(ordered) < 12:
            continue
        item_ids = ordered.article_id.astype(str).tolist()
        selected = swap_groups.get(customer)
        if selected is not None:
            for row in selected.sort_values("swap_order", kind="mergesort").itertuples(index=False):
                left = int(row.victim_rank) - 1
                right = int(row.challenger_rank) - 1
                item_ids[left], item_ids[right] = item_ids[right], item_ids[left]
        victim_id = item_ids[11]
        victim = ordered[ordered.article_id.astype(str) == victim_id].iloc[0]
        rows.append(
            {
                "customer_id": customer,
                "victim_article_id": victim_id,
                "victim_score": float(victim.victim_score),
            }
        )
    return pd.DataFrame(rows)


def select_admissions(candidate_scores: pd.DataFrame, victims: pd.DataFrame) -> pd.DataFrame:
    """Freeze one BPR-only challenger per user using scores and identities only."""
    if list(candidate_scores.columns) != ["customer_id", "article_id", "expanded_rank", "challenger_score"]:
        raise ValueError("Candidate decision input contains unregistered fields")
    if list(victims.columns) != ["customer_id", "victim_article_id", "victim_score"]:
        raise ValueError("Victim decision input contains unregistered fields")
    proposals = candidate_scores.sort_values(
        ["customer_id", "challenger_score", "expanded_rank", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).drop_duplicates("customer_id", keep="first")
    proposals = proposals.merge(victims, on="customer_id", how="inner", validate="one_to_one")
    proposals = proposals[proposals.challenger_score > proposals.victim_score].copy()
    proposals["score_difference"] = proposals.challenger_score - proposals.victim_score
    return proposals.sort_values("customer_id", kind="mergesort").reset_index(drop=True)


def audit_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "role": "read_only_BPR_only_rich_feature_reconstruction_and_decision_feasibility_audit",
        "hypothesis": (
            "The existing expanded-pool caches can reconstruct all 99 cutoff-safe WV3-661 inputs for BPR-only "
            "candidates, and the frozen model scores enough of them above WV3-721's current rank-12 item to justify one bounded screen."
        ),
        "gates": {
            "full_join_coverage_each_window": True,
            "stored_expanded_rank_replay_errors_each_window": 0,
            "original_feature_parity_max_abs": 1e-6,
            "BPR_only_candidates_in_expanded_top50_each_window_min": 500,
            "label_free_admission_proposals_each_window_min": 50,
        },
        "training": "none",
        "labels": "not used by the feasibility decision; target columns are stored separately for a later preregistered screen",
        "expected_minutes": 20,
        "failure_action": "do not register WV3-731; retain WV3-721 and close this reuse route",
        "final_week": "2020-09-16 not_run",
    }


def register_audit() -> dict:
    common.setup()
    common.budget(20)
    if not AUDIT_CONTRACT.exists():
        write(AUDIT_CONTRACT, audit_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == AUDIT_TRIAL for row in state["trials"]):
        common.register(
            AUDIT_TRIAL,
            "BPR_only_rich_feature_reconstruction_audit",
            audit_contract()["hypothesis"],
            candidate_protocol="inspect fixed BPR Top100 union; no ranking action",
            training_protocol="none; exact frozen WV3-661 model",
            features=FEATURES,
            params={"gates": audit_contract()["gates"], "final_week": "not_run"},
            expected_minutes=20,
        )
    return read(AUDIT_CONTRACT)


def render_audit(result: dict) -> str:
    lines = [
        "# WV3-730：BPR 独有候选的丰富特征可行性审计",
        "",
        "## 结论",
        "",
        f"后续 WV3-731 内层实验授权：**{'通过' if result['screen_authorized'] else '未通过'}**。本审计不训练模型，也不计算候选准入后的 MAP。",
        "",
        "## 术语",
        "",
        "- BPR 独有候选（本项目自定义）：由全目录 BPR Top100 召回、但不在原候选池中的用户—商品对；统计单位为去重后的用户—商品对。",
        "- 丰富特征重建（本项目自定义）：从扩池缓存恢复 WV3-661 训练时使用的 99 个数值输入，包括排序名次、召回来源、用户状态、热度及多层属性偏好。",
        "- 原池特征一致性（工程审计）：同一原候选使用扩池缓存重建后，与既有 WV3-661 输入逐列比较；表中为所有行、所有特征的最大绝对差。",
        "- 无标签准入提案（本项目自定义）：BPR 独有候选的冻结模型分数高于 WV3-721 当前第12名商品；计数分母是存在原 Top12 且有扩池 Top50 BPR 独有候选的用户，不读取未来购买标签。",
        "- 扩池 Top50（本项目自定义）：在原候选与 BPR 独有候选的并集上，用冻结 E0/E1 排名和固定 RRF60 得到的前50名；用于避免将 WV3-661 外推到未见的长尾名次范围。",
        "",
        "| 内层窗口 | 扩池总行 | BPR 独有 Top50 行 / 用户 | 无标签提案用户 | 名次重放错误 | 原池特征最大差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['expanded']['total_expanded_rows']:,} | "
            f"{row['expanded']['bpr_only_top50_rows']:,} / {row['expanded']['bpr_only_top50_users']:,} | "
            f"{row['label_free_admission_proposals']:,} | {row['expanded']['stored_expanded_rank_replay_errors']:,} | "
            f"{row['original_feature_parity']['max_absolute_feature_difference']:.3g} |"
        )
    lines += [
        "",
        "未来购买标签不在模型输入或准入选择中；审计产物将候选分数与标签分文件保存。"
        "只有全部窗口同时满足覆盖、名次、特征一致性和提案数量门槛，才允许注册固定的第12名准入实验。最终周保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def audit() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == AUDIT_TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-730 must start from the preregistered audit state")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        cutoff = read(common.ART / "gate_data" / folder / "DATA.json")["cutoff"]
        raw, expanded = expanded_raw(name, cutoff)
        candidates = feature_frame(raw)
        candidates["challenger_score"] = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
        victims = current_rank12(name, cutoff, model)
        score_input = candidates[["customer_id", "article_id", "rf", "challenger_score"]].rename(columns={"rf": "expanded_rank"})
        proposals = select_admissions(score_input, victims)
        parity = parity_audit(cutoff)
        root = AUDIT_ART / name
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(score_input, root / "candidate_scores.parquet")
        save_parquet(victims, root / "victim_scores.parquet")
        save_parquet(candidates[["customer_id", "article_id", "target", "truth_count"]], root / "candidate_labels.parquet")
        save_parquet(proposals, root / "admission_proposals.parquet")
        windows[name] = {
            "cutoff": cutoff,
            "expanded": expanded,
            "original_feature_parity": parity,
            "label_free_admission_proposals": len(proposals),
            "decision_fields": DECISION_COLUMNS,
            "target_excluded_from_features": "target" not in FEATURES,
            "target_excluded_from_decision": "target" not in DECISION_COLUMNS,
            "artifacts": {
                "candidate_scores": str(root / "candidate_scores.parquet"),
                "victim_scores": str(root / "victim_scores.parquet"),
                "candidate_labels": str(root / "candidate_labels.parquet"),
                "admission_proposals": str(root / "admission_proposals.parquet"),
            },
        }
        print({"bpr_rich_audit": name, "candidates": len(candidates), "proposals": len(proposals), "parity": parity["max_absolute_feature_difference"]}, flush=True)
    gates = audit_contract()["gates"]
    passed = all(
        row["expanded"]["join_coverage_complete"]
        and row["expanded"]["stored_expanded_rank_replay_errors"] == gates["stored_expanded_rank_replay_errors_each_window"]
        and row["original_feature_parity"]["max_absolute_feature_difference"] <= gates["original_feature_parity_max_abs"]
        and row["expanded"]["bpr_only_top50_rows"] >= gates["BPR_only_candidates_in_expanded_top50_each_window_min"]
        and row["label_free_admission_proposals"] >= gates["label_free_admission_proposals_each_window_min"]
        and row["target_excluded_from_features"]
        and row["target_excluded_from_decision"]
        for row in windows.values()
    )
    result = {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "windows": windows,
        "gates": gates,
        "screen_authorized": bool(passed),
        "new_training": False,
        "new_outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(AUDIT_REPORT, result)
    AUDIT_MARKDOWN.write_text(render_audit(result), encoding="utf-8")
    common.update(
        AUDIT_TRIAL,
        decision="diagnostic_supports_BPR_rich_admission" if passed else "diagnostic_rejects_BPR_rich_admission",
        inner_evidence={"screen_authorized": bool(passed), "gates": gates},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(AUDIT_CONTRACT), str(AUDIT_REPORT), str(AUDIT_MARKDOWN)],
        validity="valid_label_free_feasibility_audit",
    )
    common.log(
        f"{AUDIT_TRIAL} BPR rich-score feasibility audit",
        "BPR-only candidates have material rank-12 oracle headroom, while generic expanded-pool RRF admission failed; the missing test was source-specific rich scoring on the new items.",
        audit_contract()["hypothesis"],
        f"{AUDIT_REPORT}; {AUDIT_MARKDOWN}",
        f"screen_authorized={passed}; proposal counts={[row['label_free_admission_proposals'] for row in windows.values()]}; feature parity max={max(row['original_feature_parity']['max_absolute_feature_difference'] for row in windows.values()):.3g}.",
        "Preregister one fixed inner admission screen." if passed else "Close frozen rich-model reuse for BPR-only candidates.",
        "If authorized, change only one rank-12 admission action and compare against exact WV3-721 same-pool control.",
        alternatives="Retraining on the expanded pool is costlier and confounds representation with candidate shift; generic expanded RRF already failed.",
        experiment="Reconstruct the frozen model's full input vector, replay original inputs, and count score-qualified actions without reading target labels.",
        reflection="This audit distinguishes missing feature availability from an actual inability of the frozen propensity model to score external candidates.",
    )
    return result


def screen_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-721",
        "architecture_family": "BPR_only_rich_propensity_rank12_admission",
        "hypothesis": (
            "Among BPR-only candidates already inside the frozen expanded Top50, WV3-661's rich historical propensity "
            "score can reject weak retrievals and safely replace WV3-721's current rank-12 item."
        ),
        "candidate_protocol": "original WV3-721 pool plus fixed BPR full-catalog Top100; same-pool control exactly WV3-721",
        "decision": (
            "per user choose the highest-scoring BPR-only expanded-Top50 item; replace current rank12 only when its "
            "frozen WV3-661 score is strictly greater; one admission maximum; ranks1-11 unchanged"
        ),
        "inner_gate": {
            "mean_incremental_vs_WV3_721_gt": 0.0,
            "nondegrade_windows_vs_WV3_721_min": 3,
            "worst_incremental_vs_WV3_721_min": -0.0002,
            "standard_gate_vs_WV2_601": "mean>=0.0001, >=3 positive, worst>=-0.0005",
        },
        "expected_minutes": 10,
        "outer_policy": "one frozen outer exposure only if inner passes; no cutoff, threshold or insertion-position rescue",
        "fallback": "retain WV3-721 and close frozen-model external-candidate admission",
        "final_week": "2020-09-16 not_run",
    }


def register_screen() -> dict:
    common.setup()
    common.budget(10)
    if not read(AUDIT_REPORT)["screen_authorized"]:
        raise AssertionError("WV3-730 did not authorize WV3-731")
    if not CONTRACT.exists():
        write(CONTRACT, screen_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            screen_contract()["architecture_family"],
            screen_contract()["hypothesis"],
            candidate_protocol=screen_contract()["candidate_protocol"],
            training_protocol="none; exact frozen WV3-661 model and WV3-730 scores",
            features=FEATURES,
            params={"decision": screen_contract()["decision"], "inner_gate": screen_contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=10,
        )
    return read(CONTRACT)


def current_state(name: str, cutoff: str) -> tuple[dict[str, dict], dict]:
    candidates, _ = rich_candidate_frame(cutoff, rank_min=1)
    swaps = load_parquet(common.ART / "WV3-721" / name / "inner" / "swaps.parquet")
    swap_groups = {customer: group for customer, group in swaps.groupby("customer_id", sort=False)}
    states = {}
    for customer, group in candidates.groupby("customer_id", sort=False):
        ordered = group.sort_values("rf", kind="mergesort").reset_index(drop=True)
        values = ordered.target.to_numpy(np.uint8)
        items = ordered.article_id.astype(str).tolist()
        selected = swap_groups.get(customer)
        if selected is not None:
            for row in selected.sort_values("swap_order", kind="mergesort").itertuples(index=False):
                left = int(row.victim_rank) - 1
                right = int(row.challenger_rank) - 1
                values[left], values[right] = values[right], values[left]
                items[left], items[right] = items[right], items[left]
        states[customer] = {
            "values": values,
            "items": items,
            "truth_count": int(ordered.truth_count.iloc[0]),
        }
    baseline = read(common.REPORT / "WV3-721_SCREEN.json")["windows"][name]
    return states, baseline


def evaluate_screen_window(name: str, folder: str) -> dict:
    cutoff = read(common.ART / "gate_data" / folder / "DATA.json")["cutoff"]
    root = AUDIT_ART / name
    scores = load_parquet(root / "candidate_scores.parquet")
    victims = load_parquet(root / "victim_scores.parquet")
    labels = load_parquet(root / "candidate_labels.parquet")
    decisions = select_admissions(scores, victims)
    if any(column in decisions for column in ("target", "truth_count", "unit_gain")):
        raise AssertionError("Future label reached WV3-731 action decisions")
    frozen_keys = decisions[["customer_id", "article_id", "victim_article_id"]].copy()
    evaluated = frozen_keys.merge(
        labels,
        on=["customer_id", "article_id"],
        how="left",
        validate="one_to_one",
    )
    states, baseline = current_state(name, cutoff)
    gains = []
    for row in evaluated.itertuples(index=False):
        state = states[row.customer_id]
        if state["items"][11] != str(row.victim_article_id):
            raise AssertionError("Recorded victim does not match WV3-721 rank12")
        gains.append(replace_rank12(state["values"], int(row.target), state["truth_count"]))
    gains = np.asarray(gains, dtype=np.float64)
    delta = float(gains.sum() / baseline["total_users_denominator"])
    return {
        "window": name,
        "cutoff": cutoff,
        "total_users_denominator": baseline["total_users_denominator"],
        "same_pool_control_MAP@12": baseline["MAP@12"],
        "MAP@12": baseline["MAP@12"] + delta,
        "incremental_delta_vs_WV3_721": delta,
        "selected_users": len(decisions),
        "beneficial_selected_users": int((gains > 1e-15).sum()),
        "harmful_selected_users": int((gains < -1e-15).sum()),
        "neutral_selected_users": int((np.abs(gains) <= 1e-15).sum()),
        "gross_positive_MAP": float(gains[gains > 0].sum() / baseline["total_users_denominator"]),
        "gross_negative_MAP": float(gains[gains < 0].sum() / baseline["total_users_denominator"]),
        "protected_ranks1_11_changes": 0,
        "maximum_BPR_admissions_per_user": 1,
        "decision_uses_future_labels": False,
        "labels_joined_after_actions_frozen": True,
        "same_pool_control_reproduces_WV3_721": True,
        "candidate_pool_changed": True,
        "final_week": "not_run",
    }


def render_screen(result: dict) -> str:
    gate = result["incremental_screening_vs_WV3_721"]
    lines = [
        "# WV3-731：BPR 独有候选的丰富分数第12名准入",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'通过' if result['passed'] else '未通过'}**；相对 WV3-721 平均变化 `{gate['mean_delta_vs_WV3_721']:+.9f}`。",
        "",
        "## 术语与固定动作",
        "",
        "- 丰富倾向分数（本项目自定义）：冻结 WV3-661 LightGBM 对单个用户—商品对输出的购买倾向排序分数；输入为截止日前的99维行为、召回、热度和属性交叉特征，不是概率校准值。",
        "- 第12名准入（本项目自定义）：每位用户最多把一件 BPR 独有候选放入第12名，仅当其丰富倾向分数严格高于 WV3-721 当前第12名商品；原第1—11名完全不动。",
        "- 同池对照（本项目评测约束）：实验和对照共享原池+BPR扩池，但对照关闭新增候选准入并精确保持 WV3-721 Top12；因此差值只来自准入动作。",
        "- 有益/有害/中性用户（本项目自定义）：准入后该用户精确 AP@12 分别升高、降低或不变；三者分母是实际执行一次准入的用户。",
        "",
        "| 内层窗口 | 准入用户 | 有益 / 有害 / 中性 | 相对 WV3-721 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / "
            f"{row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | "
            f"{row['incremental_delta_vs_WV3_721']:+.9f} |"
        )
    lines += [
        "",
        "候选分数与标签分文件保存；选择完成后才连接标签评分。失败后不改变 Top50 限制、分数阈值或插入位置。"
        "最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-731 inner screen must start preregistered and unexposed")
    started = time.perf_counter()
    windows = {name: evaluate_screen_window(name, folder) for name, folder in INNER}
    increments = {name: row["incremental_delta_vs_WV3_721"] for name, row in windows.items()}
    rules = screen_contract()["inner_gate"]
    incremental = {
        "per_window_delta_vs_WV3_721": increments,
        "mean_delta_vs_WV3_721": float(np.mean(list(increments.values()))),
        "nondegrade_windows_vs_WV3_721": sum(value >= 0 for value in increments.values()),
        "worst_delta_vs_WV3_721": min(increments.values()),
    }
    incremental["passed"] = bool(
        incremental["mean_delta_vs_WV3_721"] > rules["mean_incremental_vs_WV3_721_gt"]
        and incremental["nondegrade_windows_vs_WV3_721"] >= rules["nondegrade_windows_vs_WV3_721_min"]
        and incremental["worst_delta_vs_WV3_721"] >= rules["worst_incremental_vs_WV3_721_min"]
    )
    baseline_deltas = {
        name: row["MAP@12"] - read(common.REPORT / "WV3-721_SCREEN.json")["windows"][name]["baseline_MAP@12"]
        for name, row in windows.items()
    }
    standard = screening_gate(baseline_deltas.values())
    passed = bool(incremental["passed"] and standard["passed"])
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "incremental_screening_vs_WV3_721": incremental,
        "standard_screening_vs_WV2_601": standard,
        "passed": passed,
        "candidate_pool_changed": True,
        "same_pool_control": "expanded pool with admission disabled; exact WV3-721 Top12",
        "new_training": False,
        "outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="inner_pass" if passed else "reject_inner",
        inner_evidence={"incremental": incremental, "standard": standard, "passed": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN)],
        validity="valid_label_free_inner_screen",
    )
    common.log(
        f"{TRIAL} BPR rich-score inner screen",
        "WV3-730 proved complete 99-feature reconstruction and enough label-free score-qualified BPR-only proposals.",
        screen_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"mean incremental={incremental['mean_delta_vs_WV3_721']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_721']}/4; worst={incremental['worst_delta_vs_WV3_721']:+.9f}; pass={passed}.",
        "Authorize one frozen outer confirmation." if passed else "Reject and retain WV3-721.",
        "Expose once only if both incremental and standard gates pass; otherwise close this family.",
        alternatives="No threshold, TopK or insertion-rank sweep; full expanded-pool retraining remains a higher-cost fallback only if this focused test demonstrates signal.",
        experiment="Use one frozen rich score comparison to decide one rank-12 BPR-only admission, then join labels and recompute exact AP.",
        reflection="The test isolates external-candidate scoring from broad expanded-pool reorder effects.",
    )
    print({"trial": TRIAL, "incremental": incremental, "standard": standard, "passed": passed}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register-audit", "audit", "register-screen", "screen"])
    command = parser.parse_args().command
    {"register-audit": register_audit, "audit": audit, "register-screen": register_screen, "screen": screen}[command]()
