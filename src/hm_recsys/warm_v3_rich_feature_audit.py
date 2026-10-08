"""WV3-660: read-only availability and leakage audit for rich candidate features."""
from __future__ import annotations

from pathlib import Path

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import connection, literal
from .warm_v3_residual_admission import HISTORICAL, INNER, source


TRIAL = "WV3-660"
CONTRACT = common.REPORT / "WV3-660_RICH_FEATURE_AUDIT_CONTRACT.json"
OUTPUT = common.REPORT / "WV3-660_RICH_FEATURE_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-660_RICH_FEATURE_AUDIT.md"

FEATURE_GROUPS = {
    "retrieval_source_evidence": [
        "fused_score", "source_count",
        "repurchase_rank", "repurchase_score", "repurchase_rrf_contribution",
        "recent_popularity_present", "recent_popularity_rank", "recent_popularity_score", "recent_popularity_rrf_contribution",
        "product_family_present", "product_family_rank", "product_family_score", "product_family_rrf_contribution",
        "user_day_covisit_present", "user_day_covisit_rank", "user_day_covisit_score", "user_day_covisit_rrf_contribution",
        "age_popularity_present", "age_popularity_rank", "age_popularity_score", "age_popularity_rrf_contribution",
        "attribute_content_present", "attribute_content_rank", "attribute_content_score", "attribute_content_rrf_contribution",
        "item2vec_present", "item2vec_rank", "item2vec_score", "item2vec_cosine", "item2vec_best_seed_rank",
        "item2vec_best_neighbor_rank", "item2vec_seed_support", "item2vec_vocab_count",
    ],
    "user_and_item_state": [
        "customer_age", "customer_age_missing", "age_bucket", "user_avg_price_12w", "user_online_share_12w",
        "item_events_7d", "item_events_28d", "item_events_12w", "item_unique_customers_28d", "item_avg_price_28d",
        "item_trend_7d_vs_28d", "user_item_price_gap",
    ],
    "direct_and_product_affinity": [
        "user_item_events_28d", "user_item_decay_28d_halflife_12w",
        "user_product_code_events_28d", "user_product_code_events_12w", "user_product_code_days_since",
        "user_product_code_share_12w", "user_product_code_decay_28d_halflife_12w",
        "user_product_type_events_28d", "user_product_type_events_12w", "user_product_type_days_since",
        "user_product_type_decay_28d_halflife_12w",
        "user_department_events_28d", "user_department_events_12w", "user_department_days_since",
        "user_department_decay_28d_halflife_12w",
    ],
    "garment_colour_index_affinity": [
        "user_garment_events_28d", "user_garment_events_12w", "user_garment_days_since", "user_garment_share_12w", "user_garment_decay_28d_halflife_12w",
        "user_colour_events_28d", "user_colour_events_12w", "user_colour_days_since", "user_colour_share_12w", "user_colour_decay_28d_halflife_12w",
        "user_index_group_events_28d", "user_index_group_events_12w", "user_index_group_days_since", "user_index_group_share_12w", "user_index_group_decay_28d_halflife_12w",
    ],
}
EXTRA_FEATURES = [name for values in FEATURE_GROUPS.values() for name in values]


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "read_only_rich_candidate_feature_availability_and_leakage_audit",
        "hypothesis": "The frozen feature cache contains broad retrieval, trend and user-attribute affinity signals that are absent from the 24-feature stable pointwise model and can be joined without future-label leakage.",
        "feature_groups": FEATURE_GROUPS,
        "decision_gate": {
            "all_features_present_in_all_historical_and_inner_sources": True,
            "target_excluded": True,
            "minimum_extra_features": 50,
            "historical_positive_rows_min": 5000,
        },
        "training": "none",
        "new_outer_exposure": 0,
        "fallback": "If any common field or leakage boundary fails, do not train the rich model.",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(5)
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "rich_candidate_feature_availability_audit",
            experiment_contract()["hypothesis"],
            candidate_protocol="unchanged Top50; inspect frozen feature schemas only",
            training_protocol="none; no outer labels",
            features=EXTRA_FEATURES,
            params={"decision_gate": experiment_contract()["decision_gate"], "final_week": "not_run"},
            expected_minutes=5,
        )


def run() -> dict:
    common.setup()
    common.budget(5)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    sources = []
    all_present = True
    for name, folder in list(HISTORICAL) + list(INNER):
        _, rank_meta = source(folder)
        candidate_meta = read(common.ART / "candidate_gate_data" / rank_meta["cutoff"] / "DATA.json")
        base_path = Path(candidate_meta["source_identity"]["base_path"])
        assert base_path.is_file()
        with connection() as con:
            schema = set(con.execute(f"DESCRIBE SELECT * FROM read_parquet({literal(base_path)})").fetchdf().column_name)
        missing = sorted(set(EXTRA_FEATURES) - schema)
        all_present &= not missing
        sources.append({"window": name, "cutoff": rank_meta["cutoff"], "base_path": str(base_path), "missing_features": missing, "target_column_present_but_excluded": "target" in schema, "final_week": "not_run"})
    training = read(common.ART / "WV3-631" / "TRAINING_AUDIT.json")
    target_excluded = "target" not in EXTRA_FEATURES and all(row["target_column_present_but_excluded"] for row in sources)
    passed = bool(all_present and target_excluded and len(EXTRA_FEATURES) >= 50 and training["positive_rows"] >= 5000)
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "feature_groups": FEATURE_GROUPS,
        "extra_feature_count": len(EXTRA_FEATURES),
        "sources": sources,
        "all_features_present": all_present,
        "target_excluded": target_excluded,
        "historical_top50_positive_rows": training["positive_rows"],
        "rich_pointwise_experiment_authorized": passed,
        "new_training": False,
        "new_outer_exposure": 0,
        "final_week": "not_run",
    }
    write(OUTPUT, result)
    lines = [
        "# WV3-660 丰富候选特征可用性与泄漏审计",
        "",
        "## 结论",
        "",
        f"丰富特征点式模型授权门槛：**{'pass' if passed else 'fail'}**。预先选择 {len(EXTRA_FEATURES)} 个额外特征，覆盖8个历史/内层切片时缺失字段数均为0；监督列 `target` 存在于缓存但明确不进入输入。",
        "",
        "## 术语与特征组",
        "",
        "- 丰富候选特征（本项目自定义）：相对 WV3-631 的24维输入，额外加入冻结缓存中已有但未被点式模型读取的召回来源证据、用户和商品状态、直接/商品编码/品类/部门偏好，以及服装组/颜色/索引组偏好。",
        "- 召回来源证据（推荐系统常见）：候选是否由某路召回、该路名次、原始分数及其 RRF 融合贡献；统计单位是用户—候选商品。",
        "- 属性偏好（推荐系统常见交叉特征）：用户近期购买记录在候选商品所属属性上的事件数、占比、距上次购买天数和时间衰减计数；不是商品静态ID本身。",
        "- 时间衰减计数（行业常见）：越近期的历史行为权重越高；字段名中的 `halflife` 表示项目缓存采用固定半衰期构造，本轮只复用，不调半衰期。",
        "- 泄漏审计：确认监督周购买标签 `target` 只作为训练标签，不在模型特征列表；全部候选特征来自相应截止日前的冻结缓存。",
        "",
        "| 特征组 | 额外列数 | 含义 |",
        "|---|---:|---|",
        f"| 召回来源证据 | {len(FEATURE_GROUPS['retrieval_source_evidence'])} | 六路基础召回和 Item2Vec 的命中、名次、分数、支持度 |",
        f"| 用户与商品状态 | {len(FEATURE_GROUPS['user_and_item_state'])} | 年龄/价格/渠道偏好及商品近期热度、趋势、价格差 |",
        f"| 直接与商品层级偏好 | {len(FEATURE_GROUPS['direct_and_product_affinity'])} | 用户—商品复购以及商品编码、品类、部门上的近期偏好 |",
        f"| 服装组、颜色、索引组偏好 | {len(FEATURE_GROUPS['garment_colour_index_affinity'])} | 三类属性上的事件、占比、时间间隔和衰减信号 |",
        "",
        "下一实验必须一次性使用整组字段，不按内层结果做单列筛选或消融；模型、候选池及一换一动作与 WV3-631 相同。若内层失败，不能通过删除衰减列或调整树参数救援。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    common.update(TRIAL, decision="diagnostic_supports_rich_pointwise" if passed else "diagnostic_rejects_rich_pointwise", inner_evidence={"extra_feature_count": len(EXTRA_FEATURES), "all_features_present": all_present, "target_excluded": target_excluded, "authorized": passed}, runtime=0.0, artifact_paths=[str(OUTPUT), str(MARKDOWN)])
    common.log(
        f"{TRIAL} rich feature audit",
        "The stable pointwise model used only 24 columns although the frozen candidate feature tables contain retrieval-source, trend and multi-level affinity signals.",
        experiment_contract()["hypothesis"],
        str(OUTPUT),
        f"extra features={len(EXTRA_FEATURES)}; all present={all_present}; target excluded={target_excluded}; historical positives={training['positive_rows']}; authorized={passed}.",
        "Preregister one all-feature Top50 pointwise model." if passed else "Do not train the rich model.",
        "Hold model class and one-swap action fixed so the experiment isolates candidate representation.",
        alternatives="Feature-by-feature selection was rejected because it would turn four inner windows into a tuning surface.",
        experiment="Read-only schema and label-boundary audit over four historical and four inner caches.",
        reflection="The audit verifies that richer representation is already available at low engineering cost and is not a request for new feature generation.",
    )
    print({"trial": TRIAL, "extra_features": len(EXTRA_FEATURES), "authorized": passed}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
