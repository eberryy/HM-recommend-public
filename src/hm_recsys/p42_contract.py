"""P4.2 preregistration: two frozen propensity heads, no upstream training."""
from __future__ import annotations

import importlib.metadata
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .p41a_contract import identity, read_json, write_json

RUN_ID = "p4-2-v1-two-sided-risk-controlled-admission"
FINAL_CUTOFF = "2020-09-16"
WINDOWS = {"winter_20200122": "2020-01-22", "spring_20200318": "2020-03-18",
           "early_summer_20200624": "2020-06-24", "late_summer_20200819": "2020-08-19"}
CUTOFFS = ["2019-11-27", "2019-12-25", "2020-01-22", "2020-02-19", "2020-03-18",
           "2020-04-29", "2020-05-27", "2020-06-24", "2020-07-22", "2020-08-19"]
TAUS = {"A_tau0": 0.0, "M_tau_ln2": 0.6931471805599453, "C_tau_ln4": 1.3862943611198906}
STATE = ["history_count_0_7", "history_count_8_28", "history_count_29_84", "history_count_over_84",
         "days_since_last_purchase", "recent_0_28_purchase_share"]
SOURCES = ["repurchase", "recent_popularity", "product_family", "user_day_covisit", "age_popularity", "attribute_content"]
QC_NUMERIC = ["b0_rank_pct", "b0_user_percentile", "b0_user_zscore", "normalized_margin_to_rank2",
              "normalized_margin_to_rank5", "normalized_margin_to_user_median", "m4_coarse_rank_pct",
              "interaction_count_before_cutoff", *STATE]
QW_NUMERIC = ["warm_rank", "warm_rank_pct", "warm_user_percentile", "warm_user_zscore",
              "fused_score", "source_count",
              *[s + suffix for s in SOURCES for suffix in ("_rank", "_rrf_contribution")],
              "repurchase_score", "item2vec_rank", "item2vec_cosine", "item2vec_best_seed_rank",
              "item2vec_best_neighbor_rank", "item2vec_seed_support", "item2vec_vocab_count",
              "user_history_events_12w", "user_unique_items_12w", "user_days_since_last_purchase",
              "user_online_share_12w", "item_events_7d", "item_events_28d", "item_events_12w",
              "item_unique_customers_28d", "item_days_since_last_sale", "item_trend_7d_vs_28d",
              "user_item_events_12w", "user_item_events_28d", "user_item_days_since_last_purchase",
              *["user_" + family + suffix for family in ("product_code", "product_type", "department", "garment")
                for suffix in ("_events_28d", "_events_12w", "_days_since", "_share_12w")]]
FEATURE_SPEC = {
    "qC": {"numeric": QC_NUMERIC, "binary": ["strict_cold_flag", "sparse1_5_flag"]},
    "qW": {"numeric": QW_NUMERIC, "binary": ["warm_model_score_available", *[s+"_present" for s in SOURCES],
                                                "item2vec_present", "item2vec_is_new"]},
}


def guard_cutoff(cutoff):
    t = date.fromisoformat(cutoff)
    if t >= date.fromisoformat(FINAL_CUTOFF) or t + timedelta(days=7) > date.fromisoformat(FINAL_CUTOFF):
        raise ValueError("P4.2 rejects final week and overlapping truth intervals")
    return t.isoformat()


def history_cutoffs(outer, available=None):
    t = date.fromisoformat(guard_cutoff(outer))
    return [c for c in CUTOFFS if date.fromisoformat(c)+timedelta(days=7) < t
            and (available is None or available[c])]


def branch_guard(repo):
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "main":
        raise ValueError(f"P4.2 Cold main-only task; current branch={branch!r}; no writes permitted")
    return branch


def feature_contract():
    return {
        "stage": "P4.2", "run_id": RUN_ID, "feature_spec": FEATURE_SPEC,
        "output_order": "numeric, binary, then one <numeric>_available per numeric column",
        "qC_population": "all historical B0 Cold50 rows with b0_score_available=1; not intersected with Warm users",
        "qW_population": "all historical frozen Warm-v1 Top12 rows, including no-positive users and unavailable-score fallback",
        "source": "P4.0 PIT backbone; qC six raw user-state fields from the corresponding role-specific P3.7B manifest",
        "qC_normalization": "full same-user Cold50 finite available B0 scores before any overlap exclusion: average ascending rank percentile (rank-1)/(n-1); zscore ddof=0; margins to original B0 rank2/rank5/full50 median divided by full50 population std; n<2 or std<=1e-12 or missing reference => NaN and availability0",
        "qW_normalization": "full same-user exact W0Top12 finite available Warm-v1 scores, same percentile and population zscore; unavailable score is NaN, never treated as observed zero",
        "rank_pct": "reuse existing b0_rank/50, m4_rank/200, warm_rank/full upstream candidate count before Warm150 truncation (not /12 or /150)",
        "user_state": "counts of purchase-age buckets on last20 distinct items before cutoff; days since last such purchase; recent0..28 share of valid last20 distinct items, not all event rows",
        "preprocessing": "each outer and side separately: train-only finite median per numeric (all missing=>0), train-only StandardScaler; binary missing=>0; all availability and binaries remain0/1 unscaled; no log transform or feature selection",
        "missing_indicators": {side: [c+"_available" for c in spec["numeric"]] for side,spec in FEATURE_SPEC.items()},
        "forbidden": {"qC": ["b0_score", "b0_delta_vs_m4", "article_id", "target", "B1", "P3.3 hidden teacher"],
                      "qW": ["raw category/article IDs", "cold_present", "source_branch", "B0 features", "target", "future counts"]},
        "feature_notes_zh": {
            "source_count": "同一用户—商品获得的已有召回路支持数，沿用已有值，作为来源一致性证据。",
            "source_rank": "该商品在对应召回路里的原名次，越小越靠前；不是品类编号。",
            "rrf_contribution": "原加权倒数名次融合中该召回路贡献的分数；RRF是行业通用融合方法。",
            "user_family_counts": "已有截止安全交叉：用户在商品家族、商品类型、部门、服装组的28天/12周事件数、最近购买距今天数、12周偏好占比；不输入类别编码本身。",
            "item_trend": "已有近7天与28天交易热度对比，不新建季节或窗口路由。",
        }, "final_week": "not_run",
    }


def preregister(repo):
    repo = Path(repo).resolve()
    branch_guard(repo)
    report = repo / "reports/phase4"
    target = report / "P4_2_EXPERIMENT_CONTRACT.json"
    if target.exists():
        c = read_json(target)
        if c["run_id"] != RUN_ID or c["feature_spec"] != FEATURE_SPEC:
            raise ValueError("existing preregistration differs; never overwrite")
        return c
    if (report / "P4_2_metrics.json").exists():
        raise ValueError("results exist before preregistration")
    p40 = read_json(report / "P4_0_metrics.json")
    p41 = read_json(report / "P4_1A_metrics.json")
    prev = read_json(report / "P4_1A_EXPERIMENT_CONTRACT.json")
    p31 = read_json(repo / "reports/phase3/P3_1_metrics.json")
    m54 = read_json(repo / "reports/m5_4/M5_4_metrics.json")
    assert p40["decision"]["machine_decision"] == "fusion_improves_cold_but_hurts_warm"
    assert p41["decision"]["p4_1b_allowed"] is False and p41["decision"]["P4_1B_started"] is False
    assert sorted(p40["cutoffs"]) == CUTOFFS
    authority = [repo / "docs/ROADMAP_PHASE4.zh-CN.md"]
    authority += [report / f for f in ("P4_0_FINAL.md", "P4_0_metrics.json", "P4_0_EXPERIMENT_CONTRACT.json",
        "P4_0_OUTPUT_MANIFEST.json", "P4_1A_FINAL.md", "P4_1A_metrics.json", "P4_1A_EXPERIMENT_CONTRACT.json",
        "P4_1A_VERIFICATION.json", "P4_1A_OUTPUT_MANIFEST.json", "p4_1a_score_drift_audit.json",
        "p4_1a_admission_separability.json", "p4_1a_primary_oracle.json", "p4_1a_oracle_frontier.json")]
    authority += [repo / "reports/phase3" / f for f in ("P3_7B_FINAL.md", "P3_7B_metrics.json", "P3_7B_EXPERIMENT_CONTRACT.json", "P3_7B_OUTPUT_MANIFEST.json", "P3_1_metrics.json")]
    authority += [repo / "reports/m5_4/M5_4_metrics.json"]
    inputs = {}
    for cutoff in CUTOFFS:
        a = p40["cutoffs"][cutoff]
        statepath = repo / f"artifacts/phase3/phase3-p3.7b-v1-time-aware-hybrid/features-v1/{a['role']}/{a['p31_label']}/manifest.json"
        state = read_json(statepath)
        p31assets = p31["all_assets"][f"{a['role']}:{a['p31_label']}"]["artifacts"]
        assert state["source_artifacts"]["users"]["sha256"] == p31assets["users"]["sha256"]
        b0 = p40["audits"]["upstream_lineage"][cutoff]
        warm = m54["cutoffs"][cutoff]["lineage"]["warm_score"]
        inputs[cutoff] = {"cutoff": cutoff, "cold50": a["cold"]["artifact"], "features": a["features"]["artifact"],
            "warm150": a["warm"]["artifact"], "b0_lineage": b0, "warm_lineage": warm,
            "p31_assets": {k:p31assets[k] for k in ("candidates", "users")},
            "user_state_manifest": identity(statepath), "user_state": state["artifacts"]["user_state"],
            "user_state_columns": state["columns"]["user_state"], "role": a["role"], "p31_label": a["p31_label"]}
        if warm.get("score_database"):
            inputs[cutoff]["w0_database"] = warm["score_database"]
        if cutoff in WINDOWS.values():
            window = next(k for k,v in WINDOWS.items() if v == cutoff)
            for k in ("w0_database", "w0_map", "w0_users"):
                inputs[cutoff][k] = prev["frozen_inputs"][window][k]
        if b0["available"]:
            assert b0["lineage_safe"] and b0["model_label_end"] <= cutoff
        if warm["available"]:
            assert warm["safe"] and warm["latest_training_label_end"] <= cutoff
    available = {c:a["b0_lineage"]["available"] for c,a in inputs.items()}
    versions = {n:importlib.metadata.version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "duckdb")}
    assert versions["scipy"] == "1.17.1" and versions["scikit-learn"] == "1.8.0"
    fc = feature_contract()
    c = {"schema_version": "p4.2-contract-v1", "stage": "P4.2", "run_id": RUN_ID,
        "status": "preregistered_before_formal_computation", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=repo,text=True).strip(), "branch": "main",
        "authoritative_inputs": {str(p):identity(p) for p in authority}, "inputs": inputs,
        "transactions": prev["transactions"], "catalog": prev["catalog"], "windows": WINDOWS,
        "existing_cutoff_sequence": CUTOFFS, "feature_spec": FEATURE_SPEC, "feature_contract": fc,
        "historical_pools": {w:{"qC":history_cutoffs(t,available), "qW":history_cutoffs(t)} for w,t in WINDOWS.items()},
        "time_rule": "historical label_end=cutoff+7days STRICTLY < outer cutoff; every upstream score separately label_end<=its scoring cutoff; no outer fit",
        "model": {"family":"sklearn.linear_model.LogisticRegression", "solver":"lbfgs", "penalty":"l2", "C":1.0,
            "max_iter":1000, "tol":1e-7, "random_state":20260909, "fit_intercept":True, "class_weight":None,
            "downsample":False, "oversample":False, "keep_zero_positive_users":True, "thread_limit":4,
            "fits": "one qC and one qW per outer; fixed full-row fit; insufficient class support or nonconvergence => fail closed, no retuning"},
        "software_versions": versions, "epsilon":1e-6, "utility":"logit(clip(qC))-logit(clip(qW))", "taus":TAUS,
        "pairing": {"cold":"exact B0 Cold50 minus W0Top12 overlap", "warm":"exact all W0Top12 slots", "max_pairs_per_user":600,
            "edges":"U > tau strictly; equality rejected", "weight":"U-tau", "position_multiplier":False,
            "matching":"scipy.optimize.linear_sum_assignment; Warm rows, canonical Cold columns then12 zero dummy columns; negative weights cost; illegal edges infinity",
            "tie":"unchanged float64 weights, SciPy1.17.1 deterministic on Cold sorted(originalB0rank,article_id), Warm sorted originalslot; no jitter; equal objective assignment need not be unique",
            "max_admission_hard_cap":None, "replacement":"0..12 one-to-one pairs; replace same Warm slot, all other positions unchanged; no positive edges=>exactW0"},
        "calibration": {"population":"each full outer candidate side separately before overlap exclusion; no rebalancing", "bins":10,
            "bin_rule":"sort by(q,canonical original row index), split into10 equal row-count bins; ties may straddle bins deterministically, labels not used",
            "probability":"next-week observed truth propensity conditional on candidate source, not exposure-conditioned online purchase probability",
            "metrics":["observed_positive_rate","mean_predicted_probability","brier","roc_auc","pr_auc_average_precision","ece","calibration_intercept","calibration_slope"],
            "diagnostic_fit":"unregularized two-parameter logistic MLE y~intercept+slope*clippedlogit(q), scipy L-BFGS-B mean logistic loss, init[0,1],maxiter200,ftol1e-12,gtol1e-8; oneclass/constant/separation =>null; diagnostic only, never deployed/recalibrated",
            "warning":"predicted/observed rate outside[0.25,4], ROC<=0.5, nonpositive slope, constant probabilities or numerical failure; diagnostic warning not a new promotion gate",
            "severe_failure":"either side has ratio outside[0.1,10] in>=3windows OR ROC<=0.5 in>=3windows OR constant clipped probabilities in>=3windows; invalid output is engineering failure"},
        "metrics_contract": {"primary":"exact historical apk user mean; all W0 truth users incl noCold/noadmission; four-window means equally weighted",
            "segments":{"warm_21_plus":"item history events>=21", "strict_cold":"events=0", "sparse1_5":"1<=events<=5", "all_cold_sparse":"0<=events<=5"},
            "segment_denominator":"all W0 users with at least one segment truth, nonsegment recommendations remain nonhits at their original positions; min(segment truth count,12)",
            "cold_only":"Cold50 source_branch=cold_only, absent from Warm150 (stricter than just notinW0Top12)",
            "positive_pairs":"distinct cutoff-user-article future truth, not event rows; pooled acrosswindows are user-window observations",
            "beneficial_neutral_harmful":"each executed pair examined individually as a same-slot change against W0 (insertlabel-removelabel sign); joint actual AP recomputed from full final12, individual deltas not assumed additive",
            "removed_warm_positive":"removed truth item from Warm branch at any historical count; also report >=21 subset",
            "admission_buckets":["0","1","2","3","4+"], "bucket_map":"final vs same bucket users' W0; additionally contribution divided by all windowW0users",
            "matching_efficiency":"matched edges / above-threshold edges; report node conflicts as degree>1 counts, greedy difference diagnostic only"},
        "gates": {"overall_mean_delta_min":0.0,"overall_nondegrade_min":3,"overall_worst_delta_min":-0.0002,
            "warm_mean_delta_min":-0.0001,"warm_window_delta_min":-0.0002,"warm_protected_windows_min":3,
            "cold_mean_delta_strictly_positive":True,"cold_nondegrade_min":3,"cold_only_positive_windows_min":2,
            "pooled_inserted_cold_positive_strictly_exceeds_removed_warm_positive":True},
        "selection": {"among":"only tau passing every gate", "order":["highest mean all_cold_sparse MAP","highest mean overall MAP","larger tau"],"near_equal_abs_tolerance":1e-12},
        "decisions": {"precedence":["engineering_failure","passing_tau_promotion","propensity_calibration_failure",
                "pareto_frontier_supported_but_no_safe_operating_point","warm_risk_uncontrolled","cold_gain_not_recovered"],
            "pareto_rule":"no passing tau; cold meanMAP A>=M>=C with A>C, Warm damage -meanWarmDelta A>=M>=C with A>C, and A ColdDelta>0; use tolerance1e-12 only for descriptive monotonicity",
            "warm_risk_rule":"no preceding decision, any tau with meanColdDelta>0 that fails overall or Warm protection",
            "fallback":"W0 for every nonpromotion, no P4.1B/P4.3; model data insufficiency=>engineering_failure with explicit reason"},
        "budget": {"cpu_only":True,"estimated_formal_minutes":[20,60],"formal_soft_stop_seconds":5400,
            "no_sweeps":True,"stop":"after fixed three taus, audit, report, independent verification; budget/nonconvergence failure never prompts automatic tuning"},
        "historical_boundaries": {"P4_0_decision":p40["decision"],"P4_1A_decision":p41["decision"],"p4_1b_allowed":False,"P4_1B_started":False,
                                  "Warm_v2_integrated":False,"P4_3_started":False},
        "reject_cutoff_on_or_after":FINAL_CUTOFF,"final_week":"not_run"}
    write_json(report / "p4_2_feature_contract.json",fc)
    write_json(target,c)
    return c


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",default=".")
    args=parser.parse_args()
    contract=preregister(Path(args.repo))
    print({"status":contract["status"],"created_at_utc":contract["created_at_utc"],"historical_pools":contract["historical_pools"]})
