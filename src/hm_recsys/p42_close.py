"""Truthful, no-training closure of a P4.2 preregistered engineering stop.

Added after the first execution stopped. It does not alter the original runner,
the experiment contract, fitted parameters, feature values or matching policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .p41a_contract import identity, read_json, write_json
from .p42_contract import RUN_ID, WINDOWS, TAUS, branch_guard
from .p42_report import fmt, table, feature_note


def close_failure(repo):
    repo=Path(repo).resolve(); branch_guard(repo)
    folder=repo / "reports/phase4"; root=repo / "artifacts/phase4" / RUN_ID
    c=read_json(folder / "P4_2_EXPERIMENT_CONTRACT.json")
    failures=sorted(root.glob("FAILURE-*.json"))
    if len(failures)!=1:
        raise ValueError("this closure requires exactly one unchanged formal attempt")
    failure=read_json(failures[0]); prepared=read_json(root / "PREPARED_MANIFEST.json")
    execution=read_json(root / "EXECUTION_START.json")
    training={}; artifacts=[]
    for path in sorted((root / "models").glob("*/*-training-audit.json")):
        audit=read_json(path)
        training.setdefault(path.parent.name,{})[audit["side"]]=audit
        artifacts.extend(audit[k] for k in ("model","preprocessing"))
        artifacts.append({**identity(path),"cutoff":audit["outer_cutoff"],"row_count":audit["training_rows"],"source_lineage":{"fit_status":audit["status"]}})
    for row in prepared.values():
        artifacts.extend(row["features"].values())
    assert set(training)=={"winter_20200122"} and set(training["winter_20200122"])=={"qC","qW"}
    assert training["winter_20200122"]["qC"]["status"]=="converged"
    assert training["winter_20200122"]["qW"]["status"]=="non_converged"
    assert not (root / "outer").exists(), "outer inference must not have run after this failure"
    write_json(folder / "p4_2_training_audit.json",{"stage":"P4.2","run_id":RUN_ID,"status":"stopped_after_winter_qW_nonconvergence",
        "windows":training,"prepared_cutoffs":prepared,"model_fits_planned":8,"model_fits_attempted":2,"final_week":"not_run"})
    verification_path=folder / "P4_2_VERIFICATION.json"
    verified=verification_path.exists() and read_json(verification_path).get("status")=="pass"
    m={"stage":"P4.2","run_id":RUN_ID,"status":"engineering_failure_closed" if verified else "engineering_failure_pending_verification",
        "created_at_utc":failure["failed_at_utc"],"closure_updated_at_utc":datetime.now(timezone.utc).isoformat(),
        "decision":"engineering_failure","selected_variant":"W0","failed_component":"winter_20200122/qW",
        "reason":failure["reason"],"contract":execution["contract"],"execution":execution,
        "windows":{w:{"status":"not_run","reason":"stopped_before_outer_propensity_and_admission","parity":prepared[t]["parity"]} for w,t in WINDOWS.items()},
        "summary":None,"artifacts":artifacts,
        "resources":{"formal_seconds":failure["elapsed_seconds"],"model_fits_planned":8,"model_fits_attempted":2,"converged_model_fits":1,"device":"CPU only","peak_process_working_set_gib":None},
        "historical_boundaries":c["historical_boundaries"],"final_week":"not_run",
        "evidence_boundary":"no outer propensity, calibration, pair utility, admission, new-policy MAP or promotion evaluation exists; engineering stop is not a negative model-quality result",
        "post_prereg_discovery":"P3.1 training-role cold candidates condition user selection on future cold/sparse truth; full-row reuse is not full-evaluation-population prevalence; no protocol change or reweighting performed"}
    write_json(folder / "P4_2_metrics.json",m)
    for name in ("propensity_calibration","pair_utility_audit","matching_audit","admission_risk","segment_metrics"):
        write_json(folder / f"p4_2_{name}.json",{"stage":"P4.2","run_id":RUN_ID,"status":"not_run",
            "reason":"preregistered stop: first qW fit reached max_iter=1000 without convergence; no outer inference allowed",
            "windows":{w:{"status":"not_run"} for w in WINDOWS},"variants":["W0",*TAUS],"new_metric_values":None,"final_week":"not_run"})
    render(repo,m,c,training,prepared,verified)
    manifest(repo,m,failures[0])
    return {"decision":m["decision"],"status":m["status"],"outer_evaluation":"not_run","selected_variant":"W0"}


def render(repo,m,c,training,prepared,verified):
    folder=repo / "reports/phase4"
    qc,qw=training["winter_20200122"]["qC"],training["winter_20200122"]["qW"]
    models={s:read_json(Path(a["model"]["path"])) for s,a in (("qC",qc),("qW",qw))}
    lines=["# P4.2：双侧风险控制准入——工程停止与证据收尾", "",
        "## 1. 结论", "",
        "**本轮没有完成四窗准入评测。** 首窗暖侧逻辑回归达到固定1,000次迭代仍未收敛，程序按预注册停止条件退出。机器结论为 `engineering_failure`（本项目工程失败状态），不是‘准入策略已证明无效’。最终保留 W0。",
        f"合同登记于 `{c['created_at_utc']}`；正式启动于 `{m['execution']['started_at_utc']}`；停止于 `{m['created_at_utc']}`。失败证据独立复核：{'通过；只表示停止记录与已有资产一致，不表示模型晋级' if verified else '待完成'}。",
        "没有增加迭代次数、换求解器、删特征、重加权、重采样或尝试其他阈值。没有计算 qC/qW 外层概率、校准指标、新准入 MAP，也没有生成新基线。",
        "P4.0 的‘融合改善冷侧但伤害暖侧’结论不改；P4.1A 的 `p4_1b_allowed=false`、`P4_1B_started=false` 不改。Warm-v2 未整合，P4.3 未启动，最终周 `2020-09-16=not_run`。",
        "", "## 2. 本轮原定问题与术语", "",
        "W0（本项目基线名）指冻结 Warm-v1 的每用户前12件及其原顺序；M4 是冻结的纯内容商品表示模型；B0 是在 M4 前200件内重排的冻结冷侧专家；Cold50 为 B0 原前50件。strict-cold/sparse1-5 指商品截止日前全历史交易事件数为0/1至5，研究的是商品冷度，不是新用户。",
        "qC/qW（本项目预测名）原计划分别用一套 L2 逻辑回归，估计 Cold50/W0 Top12 商品在下一周成为真值的倾向。propensity（统计通用术语，在此限定为离线候选条件下的购买倾向）不是曝光后的真实线上购买概率。用户—商品在未来7天至少购买一次为正例；未观察购买不等于明确拒绝，交易重复事件仍保留在历史统计里。",
        "原定效用 U=logit(qC)−logit(qW)，其中 logit 是概率赔率的自然对数，概率先裁剪至 [1e-6,1−1e-6]。τ=0/ln2/ln4 分别要求冷侧赔率严格高于拟替换暖商品的1/2/4倍；等于阈值仍拒绝。",
        "最多为每人形成 Cold50×Warm12=600 对，先排除已经在 W0 Top12 的冷候选。精确最大权二分匹配（通用图优化）只选择 U>τ 的边、最大化 U−τ 总和，冷商品与暖位置各最多使用一次；允许0至12次替换，不限制最多1件。没有合格边时原列表逐位保持。上述结构只完成代码和合成测试，本轮未在真实外层执行。",
        "cold-only（本项目来源名）指 Cold50 中不属于 Warm150 的用户—商品候选，比‘不属于 W0 Top12’更严格。MAP@12 是单用户前12位累计命中精度除以 min(不同真值商品数,12)，然后对该窗全部真值用户平均；四窗 mean 对窗口等权。",
        "", "## 3. 实际完成与未完成的部分", "",
        "已完成：十个截止日的冻结候选/PIT特征读取、历史标签与商品事件数独立重算、B0/M4身份核对、四窗 W0 原身份/顺序/MAP精确一致、训练集专属预处理实现、精确匹配实现及合成测试。PIT（Point-in-Time，通用时点安全要求）表示特征仅依赖截止日前历史。",
        "两次正式拟合如下；‘收敛’只是数值求解器达到停止标准，不表示泛化表现已验证。迭代不是超参数搜索；本轮只使用固定的一组设置。"]
    lines.append(table(["窗口","模型侧","训练截止日","训练行","正例行","正例率","实际迭代","拟合秒数","状态"],
        [["winter_20200122",s,", ".join(a["training_cutoffs"]),a["training_rows"],a["positives"],fmt(a["base_rate"]),models[s]["n_iter"][0],fmt(a["fit_seconds"],2),"已收敛" if a["status"]=="converged" else "未收敛"] for s,a in (("qC",qc),("qW",qw))]))
    lines.append("固定求解器为 L-BFGS（行业通用拟牛顿优化算法），L2正则（对系数平方和施加惩罚）参数 C=1，迭代上限 max_iter=1000，梯度停止容差 tol=1e-7，随机种子20260909；连续字段训练集列中位数填补后用训练集均值/总体标准差标准化，布尔与可用标记保持0/1。没有类别权重或负采样。暖侧54个数值字段、9个布尔字段，加54个可用标记，共117维；冷侧14+2+14=30维。")
    lines.append("未收敛警告：")
    lines.extend(["> "+x.replace("\n"," ") for x in qw["convergence_warnings"]])
    lines.append("这是程序中的实际停止分支，不是事后看到不利 MAP 才选择不跑。其余6次正式拟合，以及全部四窗×三个阈值的准入评测均未启动；缺失指标在 JSON 中记为 `not_run`/null，绝不填0。")
    lines += ["", "## 4. 另一个实质性发现：排序训练裁剪不等价于概率训练", "",
        "在预注册后、正式运行期间的只读追溯中，确认 P3.1 的训练角色只为‘下一周至少购买一件冷/稀疏商品’的用户生成候选；外层角色则使用全部冻结评测用户。原证据在 `src/hm_recsys/p3.py` 的 `_eligible_training_users`、`_outer_users_and_warm`，以及 `reports/phase3/P3_1_FINAL.md` 最后一段。",
        "P4.0 复用了这些角色对应的候选，所以本轮扩大历史池时混合了两种用户总体：普通训练日期经过上述未来标签条件筛选，历史外层日期使用完整评测总体。首窗 qC 仅用2019-12-25，完全属于前一种总体。",
        "对于只需要同用户正负配对的排序损失，没有任何冷真值的用户无法贡献正负训练对，旧裁剪有其适用范围。但逻辑回归要学习绝对概率时，所有未观察候选用户会影响基础正例率；删掉他们不再等价。即使保留当前文件里所有行、甚至保留候选池里零命中的用户，也恢复不了当初被省略的用户。",
        "因此本轮‘不再次负采样’只能保证保留现有资产的正例率，不能声称保留完整外层用户总体的概率率。训练角色上的 qC 更接近条件概率 P(购买 | 特征, Cold50, 已知该用户本周有冷真值)。这是训练总体选择偏差，不是把当前外层答案灌入训练，也不能直接证明所有历史排序结论失效。",
        "**本轮没有计算外层校准，所以尚未实测该偏差造成多大的概率高估或 MAP 损伤；也不能把暖侧未收敛归因于这个冷侧人群问题。** 两项发现必须分开。完整追溯及已存在候选正例率见 `P4_2_TRAINING_POPULATION_NOTE.zh-CN.md` 与对应 JSON。",
        "", "## 5. 数值停止诊断与下一步边界", "",
        "只读检查保存的模型、训练矩阵和优化器目标，不增加拟合。如果存在常数/重复字段、强相关性或极端标准化值，只能视为数值条件风险；未做受控修复对照前不声称已证明单一根因。检查结果见 `p4_2_convergence_diagnostic.json`。",
        "实测：梯度无穷范数（各参数目标导数绝对值的最大值）为2.096638e-6，约为固定1e-7标准的21倍；确实没有达到设定的梯度停止条件。117列中有26列常数可用标记、40对完全重复的非常数列，多数是召回存在标记与派生可用标记重复。共有47对列的相关系数绝对值≥0.995。40对列不等于40列可直接删除。",
        "用户—商品12周购买数标准化后最大绝对值仍达191.75；局部正则曲率矩阵的条件数（最大/最小特征值比，越大表示不同参数方向尺度越悬殊）约376万。这些是数值求解困难的证据；独立复核未发现训练中位数/标准化器计算错误，但没有用删列或换算法对照证明唯一根因。",
        "统一为每个数值列添加可用标记是本轮预注册的缺失处理选择，避免把填补值当成真实观察；本次暴露其中有常数和重复信息。这个工程选择需要反思，不能只把未收敛归咎于原始数据。训练集预测均值接近训练正例率也不是外层校准通过。",
        "本轮至此停止，回退 W0，保留冻结 B0。下一次决策应先解决‘排序用候选资产是否适合绝对概率拟合’这一已证实的数据总体差异，再单独讨论暖侧数值收敛；仅增加迭代不能修正冷侧训练总体的选择偏差。恢复历史完整用户候选或修改固定求解预算均需另行明确设计和授权，不能覆盖本次合同，也不自动转入 P4.3。",
        "", "## 6. 原任务13个问题的真实回答"]
    lines.append(table(["问题","本轮可给出的答案"],[
        ["1. qC有稳定可分性吗？","未评测；首窗训练已收敛不等于四窗泛化通过。"],
        ["2. qW有稳定可分性吗？","未评测；首窗固定求解器未收敛。"],
        ["3. 用户内相对特征解决原始分数漂移了吗？","没有外层概率/校准结果，不能确认；另发现训练人群选择偏差，单靠去掉原始尺度不解决。"],
        ["4. 两侧概率可直接比较吗？","目前不能凭此任务作出已校准的宣称。"],
        ["5. 三个阈值覆盖多少用户？","全部not_run，不是0覆盖。"],
        ["6. 每人0/1/2/3/4+次替换分布？","未运行；合成测试已证明实现没有max1硬上限。"],
        ["7. 多次准入是否累积暖侧损伤？","未评测；不能凭架构直觉定性。"],
        ["8. 收益来自strict-cold还是sparse？","没有新策略收益，不能归因。"],
        ["9. 新插正例是否多于移除正例？","未运行。"],
        ["10. 总体和Warm保护通过吗？","门槛尚未评测，不得标成通过或模型失败。"],
        ["11. 哪个安全阈值/新基线可选？","没有任何已验证的新阈值；保留W0。"],
        ["12. 失败类型和原因是什么？","工程停止：首窗qW未收敛；另有确定的冷侧历史总体选择偏差，不能据此宣布策略能力失败。"],
        ["13. 最终周和Warm-v2动了吗？","均未动；2020-09-16=not_run，Warm-v2未整合，P4.3/P4.1B均未启动。"],
    ]))
    lines += ["", "## 7. 成本与证据", "",
        f"单次正式运行到停止耗时 {fmt(m['resources']['formal_seconds'],2)} 秒，CPU only，未使用GPU；其中 qC/qW 拟合分别为 {fmt(qc['fit_seconds'],2)}/{fmt(qw['fit_seconds'],2)} 秒，其余主要为输入核验和十截止日预处理。原定20至60分钟只是事前预算，不是实测耗时。本次异常退出未保存进程峰值内存，不补造数值。",
        "上述耗时不含代码实现、环境安装、合成测试、只读追溯、独立复核和报告写作。失败模型、标准化器、逐行特征及原始失败堆栈都保存在被Git忽略的 D盘项目 artifacts/phase4/p4-2-v1-two-sided-risk-controlled-admission；没有把用户级数据公开。",
        "SHA用于与既有可信资产清单和本轮证据合同比较，验证迁移/缓存身份；完整输出清单见 P4_2_OUTPUT_MANIFEST.json。模型、预处理器与准备数据记录对应截止日、行数与来源链。",
        "独立复核只检验已运行部分和停止机制。其余外层校准、匹配和晋级项目必须列为未运行；复核状态通过不等于实验晋级通过。", "",
        "本轮41项合成测试通过；独立复核19.66秒，检查184个输入/输出文件身份，其中131个为已有冻结输入。25项正式核查中14项仅在已运行证据范围通过，11项外层准入核查未运行；数值收敛门槛保留失败状态。", "",
        "## 附录：固定输入的逐项解释", "",
        "所有下列英文列名是本项目字段名。12w表示截止前12周，28d/7d表示截止前28/7天；计数单位在对应行说明。每个数值列另带 `_available` 标记（有限值为1，缺失为0）。名次、占比、计数均不是原始商品/类别ID。"]
    for side,spec in c["feature_spec"].items():
        lines += ["",f"### {side}",""]
        lines.append(table(["字段","类型","含义/分母"],[[col,kind,feature_note(col)] for kind,names in (("数值",spec["numeric"]),("布尔",spec["binary"])) for col in names]))
    (folder / "P4_2_FINAL.md").write_text("\n\n".join(part for part in lines if part)+"\n",encoding="utf-8")


def manifest(repo,m,failure_path):
    folder=repo / "reports/phase4"; root=repo / "artifacts/phase4" / RUN_ID
    receipt=read_json(root / "input-verification.json")
    arts={a["path"]:a for a in m["artifacts"]}
    for entry in receipt["authority_snapshots"].values():
        p=entry["snapshot_path"]
        arts[p]={"path":p,"bytes":entry["bytes"],"sha256":entry["sha256"],"cutoff":None,"source_lineage":{"snapshot_of":entry["path"]}}
    for p in (root / "EXECUTION_START.json",root / "input-verification.json",root / "PREPARED_MANIFEST.json",failure_path,root / "STOPPED_MODEL_DIAGNOSTIC.json"):
        arts[str(p)]={**identity(p),"cutoff":None,"source_lineage":{"run_id":RUN_ID,"status":"engineering_failure"}}
    reports=sorted(set(folder.glob("P4_2*")) | set(folder.glob("p4_2*")))
    reports=[p for p in reports if p.name!="P4_2_OUTPUT_MANIFEST.json" and p.is_file()]
    sources=[repo / "dependency.txt"]+sorted((repo / "src/hm_recsys").glob("p42*.py"))+sorted((repo / "tests").glob("test_p42*.py"))
    write_json(folder / "P4_2_OUTPUT_MANIFEST.json",{"stage":"P4.2","run_id":RUN_ID,"status":m["status"],
        "reports":[identity(p) for p in reports],"sources":[identity(p) for p in sources],"artifacts":list(arts.values()),
        "no_outer_artifacts":True,"decision":"engineering_failure","selected_variant":"W0","final_week":"not_run"})


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",default="."); args=parser.parse_args()
    print(close_failure(Path(args.repo)))
