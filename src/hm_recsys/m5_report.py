from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from .m33 import ROLLING_PROTOCOL
from .m4_contract import atomic_json, file_identity
from .m5_model import _warm_state


def _score_frame(path: Path):
    con = duckdb.connect()
    escaped = path.resolve().as_posix().replace("'", "''")
    try:
        return con.execute(f"SELECT * FROM read_parquet('{escaped}')").fetch_df()
    finally:
        con.close()


def internal_ranking_audit(
    *, source_root: Path, artifact_dir: Path, window: str,
) -> dict[str, Any]:
    cutoff = ROLLING_PROTOCOL[window]["outer_validation"]
    score_path = artifact_dir / "models-v1" / window / "outer-cold-scores.parquet"
    warm_db = source_root / "artifacts" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / window / "evaluation.duckdb"
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    users, _top12, warm_pool, truth = _warm_state(
        evaluation_db=warm_db, transactions_path=transactions, cutoff=cutoff
    )
    frame = _score_frame(score_path)
    available_pairs = 0
    positive_users = 0
    source_hits = {1: 0, 3: 0, 12: 0}
    expert_hits = {1: 0, 3: 0, 12: 0}
    source_rr: list[float] = []
    expert_rr: list[float] = []
    eligible_users = 0
    for user, group in frame.groupby("customer_id", sort=False):
        user = str(user)
        group = group.loc[~group["article_id"].astype(str).isin(warm_pool.get(user, set()))].copy()
        if group.empty:
            continue
        eligible_users += 1
        relevant = truth.get(user, set())
        positives = set(group.loc[group["article_id"].astype(str).isin(relevant), "article_id"].astype(str))
        if not positives:
            continue
        available_pairs += len(positives)
        positive_users += 1
        source_order = group.sort_values(["student_rank", "article_id"])["article_id"].astype(str).tolist()
        expert_order = group.sort_values(
            ["score", "student_rank", "article_id"], ascending=[False, True, True]
        )["article_id"].astype(str).tolist()
        for k in source_hits:
            source_hits[k] += sum(item in positives for item in source_order[:k])
            expert_hits[k] += sum(item in positives for item in expert_order[:k])
        source_first = next(index for index, item in enumerate(source_order, start=1) if item in positives)
        expert_first = next(index for index, item in enumerate(expert_order, start=1) if item in positives)
        source_rr.append(1.0 / source_first)
        expert_rr.append(1.0 / expert_first)
    return {
        "definition": "only M4 Student Top50 candidates absent from the full frozen Warm-v1 candidate pool; positives are distinct next-week user-item purchases",
        "eligible_users": eligible_users,
        "available_positive_pairs": available_pairs,
        "positive_users": positive_users,
        "source_order_positive_pairs_at_k": {str(k): value for k, value in source_hits.items()},
        "expert_order_positive_pairs_at_k": {str(k): value for k, value in expert_hits.items()},
        "source_order_mrr": float(np.mean(source_rr)) if source_rr else 0.0,
        "expert_order_mrr": float(np.mean(expert_rr)) if expert_rr else 0.0,
        "expert_top1_positive_density": expert_hits[1] / max(eligible_users, 1),
        "expert_top3_positive_density": expert_hits[3] / max(eligible_users * 3, 1),
        "score_artifact": file_identity(score_path),
    }


def enrich_and_write_reports(
    *, source_root: Path, artifact_dir: Path, report_dir: Path,
) -> dict[str, Any]:
    m50_path = report_dir / "M5_0_metrics.json"
    m52_path = report_dir / "M5_2_metrics.json"
    m50 = json.loads(m50_path.read_text(encoding="utf-8"))
    m52 = json.loads(m52_path.read_text(encoding="utf-8"))
    diagnostics = {
        window: internal_ranking_audit(
            source_root=source_root, artifact_dir=artifact_dir, window=window
        ) for window in ROLLING_PROTOCOL
    }
    m52["contract"]["negative_sampling"] = (
        "all positives plus at most 100 rank-stratified unobserved candidates per positive; "
        "selection key is stable hash(cutoff,user_index,catalog_row,student_rank)"
    )
    m52["authoritative_revision"] = "stable-identity-hash-negative-sampler-rerun"
    m52["failure_diagnostics"] = diagnostics
    atomic_json(m52_path, m52)
    (report_dir / "M5_0_FINAL.md").write_text(render_m50(m50), encoding="utf-8")
    (report_dir / "M5_1_FINAL.md").write_text(render_m51(m52), encoding="utf-8")
    (report_dir / "M5_2_FINAL.md").write_text(render_m5(m52), encoding="utf-8")
    (report_dir / "M5_FINAL.md").write_text(render_m5(m52), encoding="utf-8")
    return m52


def render_m50(result: dict[str, Any]) -> str:
    lines = [
        "# M5.0 冷商品排序监督规模门禁",
        "",
        "## 结论",
        "",
        f"M5.0 门禁：**{'通过' if result['gate_passed'] else '失败'}**。最终训练规模选择 "
        f"`{result['decision']['selected_training_scale']}`，模型类型选择 "
        f"`{result['decision']['selected_model_type']}`。final week 未运行。",
        "",
        "## 术语与统计单位",
        "",
        "- **正例用户—商品对（positive pair）**：某个截止日生成的 Student Top50 中，用户在截止日起下一周实际购买的一个不同商品；单位是去重用户—商品对。",
        "- **正例组（positive group）**：至少含一个上述正例的“截止日—用户”候选组；单位是用户组，不是候选行。",
        "- **正例密度（positive density）**：正例用户—商品对数除以同规模下全部候选行数。",
        "- **时间点安全 Student（point-in-time Student）**：本项目自定义名称，表示该模型只使用目标截止日前的 Teacher 行为关系训练，不能用更晚关系回填历史训练周。",
        "",
        "## 四个 outer-train 的规模统计",
        "",
        "| outer 窗口 | 规模 | 候选行 | 正例对 | 正例组 | 正例密度 | 正例组内正例中位数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window, scales in result["outer_train_aggregate"].items():
        for scale in ("10pct", "30pct", "full"):
            row = scales[scale]
            lines.append(
                f"| {window} | {scale} | {row['candidate_rows']:,} | {row['positive_pairs']:,} | "
                f"{row['positive_groups']:,} | {row['positive_density']:.8f} | "
                f"{row['positive_per_positive_group_median']:.1f} |"
            )
    lines += [
        "",
        "10% 和 30% 的四窗正例数均不足预注册阈值；因此升至 full users。四窗正例组中位数均为1，按预注册判据只允许 pooled binary LightGBM，不运行 LambdaRank/MLP 搜索。",
        "",
        "## 边界",
        "",
        "- 用户分母是下一周有购买记录且截止日前12周存在可编码购买 seed 的用户；模型只在有内容查询的用户上产生 cold candidates。",
        "- `target=0` 只是下一周没有观察到购买，并非曝光后拒绝，因为数据没有 impression/exposure 日志。",
        "- 商品可售性仍采用 `optimistic_all_articles`；zero-history 不等同真实线上新品。",
    ]
    return "\n".join(lines) + "\n"


def render_m51(result: dict[str, Any]) -> str:
    lines = [
        "# M5.1 来源专属冷商品专家排序器",
        "",
        "## 结论",
        "",
        "已按 M5.0 的数据判据训练唯一一类模型：全量用户 pooled binary LightGBM。这里的 `pooled binary` 是行业常用二分类训练方式，表示把多个截止日、多个用户的候选行合并，以“下一周是否购买”为0/1标签；它不把零正例用户组丢掉，也不与 LambdaRank/MLP 做事后比较。",
        "",
        "## 冻结输入",
        "",
        "- 候选：多模态 Student Top50；每行单位是一个截止日—用户—候选商品观察。",
        "- 正例：下一周实际购买的去重用户—商品对；未观察候选不是已曝光负例。",
        "- 负采样：保留全部正例，并在 Student 原始名次的五个十分位桶中，按截止日—用户索引—商品索引—原名次的稳定哈希抽取合计不超过100个未观察候选/正例。",
        "- 特征：Student/原始 FashionCLIP 相似度与差值、商品冷度、商品静态属性、用户12周历史摘要，以及用户对五类商品属性的历史事件占比。所有动态特征只读截止日前12周。",
        "",
        "## 四窗训练证据",
        "",
        "| 窗口 | inner 训练正例行 | early-stop 轮数 | outer 训练正例行 | outer 训练候选行 |",
        "|---|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        lines.append(
            f"| {window} | {row['inner']['model']['train_positive_rows']:,} | "
            f"{row['inner']['model']['best_iteration']} | "
            f"{row['outer']['training_data']['positive_rows']:,} | "
            f"{row['outer']['training_data']['rows']:,} |"
        )
    lines += [
        "",
        "## 主要特征信号",
        "",
        "以下只列各 outer 模型按 split gain 排名前五的特征；gain 是 LightGBM 在树分裂中带来的累计损失下降量，只能说明模型使用程度，不能单独证明因果增益。",
        "",
    ]
    for window, row in result["windows"].items():
        top = row["outer"]["model"]["top_feature_importance"][:5]
        lines.append(
            f"- {window}：" + "；".join(f"`{value['feature']}`={value['gain']:.1f}" for value in top) + "。"
        )
    lines += [
        "",
        "## 边界",
        "",
        "M5.1 只回答专家能否在冷候选池内学习相对次序；是否值得挤掉 Warm Top12 必须由 M5.2 的真实替换损益判断。模型训练完成不等于 MAP 晋级。",
        "",
        "final week 未运行。",
    ]
    return "\n".join(lines) + "\n"


def render_m5(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# M5 Cold expert 与双通道 MAP 兑现",
        "",
        "## 结论",
        "",
        "M5.1/M5.2 的正式晋级门禁 **未通过**。四个 inner validation 都选择 `K_admit=0`；因此正式 outer 方案逐窗精确退回冻结 Warm-v1，mean MAP@12 没有真实提升。M6 不获准运行，final week 保持未读取。",
        "",
        "## 术语与统计单位",
        "",
        "- **Cold expert（冷商品专家排序器）**：只给多模态 Student Top50 冷/稀疏候选打分的 LightGBM 二分类模型，不改 Warm-v1 的300候选排序。",
        "- **K_admit（冷候选准入数）**：本项目自定义超参数，表示每个有候选用户最多用多少个 Cold expert 高分商品替换 Warm Top12 尾部；K=1 替换第12名，K=3替换第10--12名。",
        "- **插入/移除正例对**：准入后新进入 Top12、或从 Warm Top12 被挤出的下一周真实购买用户—商品对；单位均为去重用户—商品对。",
        "- **内部正例名次**：只在“Student Top50 且不属于 Warm-v1 完整候选池”的增量候选中计算；TopK 命中数的单位是正例用户—商品对。",
        "- **MRR（平均倒数排名）**：行业通用排序指标；对每个至少有一个可达正例的用户取首个正例名次的倒数，再对这些用户求平均。分母不是所有评测用户。",
        "",
        "## Inner temporal selection",
        "",
        "`inner temporal selection` 指只用更早训练周训练专家，在下一历史周比较 K=0/1/3 的 overall MAP@12；outer 标签没有参与 K 的选择。",
        "",
        "| 窗口 | K=0 | K=1 | K=3 | 选择 |",
        "|---|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        values = row["inner"]["admission_results"]
        lines.append(
            f"| {window} | {values['0']['segments']['overall']['map@12']:.6f} | "
            f"{values['1']['segments']['overall']['map@12']:.6f} | "
            f"{values['3']['segments']['overall']['map@12']:.6f} | {row['selected_k_from_inner']} |"
        )
    lines += [
        "",
        "四个 inner 窗口中，任何正数准入量都低于 K=0，因此不存在跨窗可部署的非零准入策略。",
        "",
        "## Outer 强制准入诊断（不参与选择）",
        "",
        "| 窗口 | K | MAP@12 | 相对 Warm | 插入正例对 | 移除正例对 | 插入 strict-cold 正例对 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        base = row["outer"]["all_k_diagnostic"]["0"]["segments"]["overall"]["map@12"]
        for k in (1, 3):
            value = row["outer"]["all_k_diagnostic"][str(k)]
            lines.append(
                f"| {window} | {k} | {value['segments']['overall']['map@12']:.6f} | "
                f"{value['segments']['overall']['map@12']-base:+.6f} | "
                f"{value['inserted_positive_pairs']} | {value['removed_positive_pairs']} | "
                f"{value['inserted_strict_cold_positive_pairs']} |"
            )
    lines += [
        "",
        "## Cold expert 内部排序审计",
        "",
        "| 窗口 | 增量可达正例对 | 正例用户 | 原 Student Top1/Top3/Top12 | Expert Top1/Top3/Top12 | Student MRR | Expert MRR | Expert Top1 正例密度 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window, value in result["failure_diagnostics"].items():
        source = value["source_order_positive_pairs_at_k"]
        expert = value["expert_order_positive_pairs_at_k"]
        lines.append(
            f"| {window} | {value['available_positive_pairs']} | {value['positive_users']} | "
            f"{source['1']}/{source['3']}/{source['12']} | {expert['1']}/{expert['3']}/{expert['12']} | "
            f"{value['source_order_mrr']:.4f} | {value['expert_order_mrr']:.4f} | "
            f"{value['expert_top1_positive_density']:.6f} |"
        )
    lines += [
        "",
        "这张表中的斜杠只用于同一统计量在 Top1、Top3、Top12 三个 K 值下的并列展示；三者单位都是正例用户—商品对，不混用用户数或比率。",
        "",
        "## 晋级门禁",
        "",
    ]
    for name, passed in summary["gates"].items():
        lines.append(f"- `{name}`：{'通过' if passed else '失败'}")
    lines += [
        "",
        "## 失败原因与可复用结论",
        "",
        "1. M4 已经解决“候选不可达”的一部分问题，但没有解决准入所需的极高 precision。每位活跃用户插一个商品意味着数千次替换，而四窗 Expert Top1 中真实正例只有个位数，远少于被替换的 Warm 正例。",
        "2. 全量训练提高了正例绝对数量，却没有改变每个正例组通常只有一个正例、总体密度约万分之一的事实。扩大样本解决方差，不会自动解决类别极不平衡和缺少曝光负例的问题。",
        "3. 强制 K=1/3 在四窗全部退化，说明失败不是四窗门控过严；即便事后观察 outer，也找不到一个正数 K 可以成为候选 baseline。",
        "4. 因此保留 M4 的 supervised multimodal retrieval 作为有效模块，但按预注册停止局部 K/threshold 搜索；Warm-v1 仍是最终排序 baseline。",
        "",
        "## 资源与边界",
        "",
        f"- M5.1/M5.2 wall time：{result['resources']['elapsed_seconds']/60:.2f} 分钟；峰值进程内存：{result['resources']['peak_working_set_bytes']/2**30:.2f} GiB。",
        "- M5.0 的八个训练截止日全部满足 `latest history < cutoff`，外层 Warm MAP 与冻结 M3.3 精确一致。",
        "- 没有读取 2020-09-16 final week，没有运行 M6，也没有将失败的 cold admission 宣称为新 baseline。",
    ]
    return "\n".join(lines) + "\n"
