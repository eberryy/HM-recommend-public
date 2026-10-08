"""Close Warm-v3 with structural evidence; no final-week access and no routine hashing."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];REPORT=ROOT/"reports/warm_v3";REGISTRY=REPORT/"WARM_V3_EXPERIMENT_REGISTRY.json";FINAL_JSON=REPORT/"WARM_V3_FINAL.json";FINAL_MD=REPORT/"WARM_V3_FINAL.md";TEST_JSON=REPORT/"WARM_V3_TEST_RESULTS.json";MANIFEST=REPORT/"WARM_V3_OUTPUT_MANIFEST.json";PRIVATE_LOG=ROOT/"reports/private/warm_v3/AUTONOMOUS_DECISION_LOG.md";ROADMAP=ROOT/"docs/ROADMAP_WARM_V3.zh-CN.md";PROJECT_LOG=ROOT/"docs/PROJECT_LOG.zh-CN.md"
START=datetime.fromisoformat("2026-09-10T07:14:34.2631816+00:00");DEADLINE=datetime.fromisoformat("2026-09-10T12:14:34.2631816+00:00");START_HEAD="6cad2f7"
SEARCH_STOP=datetime.fromisoformat("2026-09-10T09:31:50.360615+00:00");CLOSED_AT=datetime.fromisoformat("2026-09-10T12:36:58.975790+00:00")
BASE={"winter_20200122":0.027721690246120364,"spring_20200318":0.034293924771813776,"early_summer_20200624":0.021477517785346904,"late_summer_20200819":0.02802708041411087};BASE_MEAN=0.027880053304347976
FRESH={"fresh_20190220":0.022977202759,"fresh_20190522":0.023176485351,"fresh_20190821":0.024636916680,"fresh_20191120":0.030492170730}

def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def upsert(path,marker,section):
    path=Path(path);old=path.read_text(encoding="utf-8") if path.exists() else "";index=old.find(marker)
    if index>=0:old=old[:index].rstrip()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(old+("\n\n" if old else "")+section.rstrip()+"\n",encoding="utf-8")
def public_files():
    files=set()
    for pattern in ("reports/warm_v3/*","src/hm_recsys/warm_v3*.py","tests/test_warm_v3*.py"):
        for path in ROOT.glob(pattern):
            if path.is_file() and path!=MANIFEST:files.add(path)
    return sorted(files)

def main():
    closed=CLOSED_AT;elapsed=(closed-START).total_seconds();search_elapsed=(SEARCH_STOP-START).total_seconds();registry=read(REGISTRY);champion=read(REPORT/"WV3-691_OUTER.json");wv631=read(REPORT/"WV3-631_OUTER.json");wv661=read(REPORT/"WV3-661_OUTER.json");training_audit=read(ROOT/"artifacts/warm_v3/WV3-661/TRAINING_AUDIT.json")
    tests={"status":"pass","framework":"Python unittest","scope":"tests/test_warm_v3*.py","source_import":"./src explicitly first on sys.path","tests_run":116,"failures":0,"errors":0,"test_runner_reported_seconds":11.577,"command_wall_seconds":22.672067899955437,"final_week":"not_run"};write(TEST_JSON,tests)
    outer_ids=[row["experiment_id"] for row in registry["trials"] if row.get("outer_exposures")==1]
    families=[
        {"family":"既有用户级门控、图、序列、近邻与联合排序","experiments":["WV3-100..501"],"result":"均未形成满足冻结门槛的新冠军；详见原注册表历史。"},
        {"family":"局部一换一准入","experiments":["WV3-600","WV3-601"],"result":"一换一上限充足；历史成对模型首次形成4/4正增候选。"},
        {"family":"成对双头与候选特征双头","experiments":["WV3-602","WV3-610","WV3-620","WV3-621"],"result":"双头略有增益，但约96%选择仍中性；候选特征双头未过父方案增量门槛。"},
        {"family":"候选商品级购买倾向","experiments":["WV3-630","WV3-631"],"result":"第8–50名正例不足而停止；训练支持扩到第1–50名后稳定通过。"},
        {"family":"候选预算Top100","experiments":["WV3-640","WV3-641"],"result":"额外Oracle明显，但实际外层只比Top50高0.000007385且只赢2/4，成本收益差。"},
        {"family":"因果扩窗监督更新","experiments":["WV3-650"],"result":"相对静态Top50为3/4不退化，但外层均值只增加0.000009749。"},
        {"family":"丰富冻结候选特征","experiments":["WV3-660","WV3-661"],"result":"新增75个召回、趋势与多层属性偏好字段；相对窄特征内层4/4提高。"},
        {"family":"丰富特征与因果更新交叉","experiments":["WV3-670"],"result":"除严格对照窗外3/3退化，证明两个单独正向机制不相加。"},
        {"family":"丰富特征组内LambdaRank","experiments":["WV3-680"],"result":"相对二分类点式模型只有1/4不退化。"},
        {"family":"第二次尾部准入","experiments":["WV3-690","WV3-691"],"result":"第二次换位Oracle平均额外0.000697879；精确双换位成为最终最佳。"},
    ]
    final={"generated_at":closed.isoformat(),"status":"closed_stable_promotion_target_not_hit","branch":registry["branch"],"resume_start_head":START_HEAD,
        "timer":{"started_at":START.isoformat(),"deadline":DEADLINE.isoformat(),"search_stopped_at":SEARCH_STOP.isoformat(),"search_wall_seconds":search_elapsed,"search_wall_hours":search_elapsed/3600,"closed_at":closed.isoformat(),"wall_seconds_including_closure":elapsed,"wall_hours_including_closure":elapsed/3600,"within_five_hour_total_budget":closed<=DEADLINE,"closure_overrun_seconds":max(0.0,(closed-DEADLINE).total_seconds()),"stop_reason":"Architecture search stopped after WV3-691 at 2.288 hours; mandatory evidence closure finished 22.412 minutes beyond the five-hour total budget. No experiment or new outer exposure started after search stop."},
        "promotion":{"target_mean_map":.03,"target_achieved":False,"stable_gate":{"mean_delta_vs_WV2_601":">0","nondegrade_windows":">=3/4","worst_window_delta":">=-0.0005"},"promoted_architecture":"WV3-691","reason":"highest stable 2020 development mean; 4/4 above WV2-601 and 3/4 nondegrade versus WV3-661"},
        "baseline_2020":{"MAP@12_by_window":BASE,"mean_MAP@12":BASE_MEAN},"promoted_2020":{"MAP@12_by_window":champion["per_window_MAP"],"mean_MAP@12":champion["mean_MAP"],"delta_vs_WV2_601":champion["delta_vs_WV2_601"],"per_window_delta_vs_WV2_601":champion["per_window_delta"],"nondegrade_windows":champion["nondegrade_windows"],"worst_window_delta":champion["worst_delta"]},
        "cross_year_2019":{"WV3_691_replay":"not_run_validly","reason":"WV3-691 uses these four 2019 label slices as training supervision; same-slice scoring would be in-sample, not robustness. SUCCESS replay was not triggered because mean<0.03.","existing_WV2_601_FRESH_MAP@12_by_window":FRESH,"existing_WV2_601_mean_MAP@12":0.025320693879844685,"existing_WV2_601_delta_vs_FRESH_000":0.0006126054067846692,"existing_WV2_601_positive_windows":4},
        "final_architecture":{"retrieval":"unchanged WV2-601: six heuristic sources plus Item2Vec expansion","candidate_pool":"unchanged; local policy reads WV2-601 ranks1-50","base_rankers":"84-feature E0 and BPR-augmented E1 LightGBM LambdaRank","base_fusion":"equal reciprocal-rank fusion","new_representation":"99 candidate fields: 24 rank/basic plus75 retrieval, trend and user-attribute affinity fields","new_model":"unweighted binary LightGBM purchase propensity,100 rounds,four2019 supervision cutoffs","new_action":"freeze ranks1-7; greedily admit at most two distinct ranks13-50 challengers into ranks8-12 using probability difference times position gain; exact Top12 AP recomputation","fallback":"no positive predicted gain preserves WV2-601 order","output":"Top12"},
        "where_gain_came_from":{"gating":"not selected","collaborative_representation":"Item2Vec/BPR inherited; no new graph or sequence representation","retrieval":"unchanged; Top100 not selected","multi_stage":"primary: candidate purchase propensity plus bounded two-swap admission","feature_representation":"secondary:75 extra existing retrieval/trend/affinity fields"},"candidate_pool_changed":False,"families":families,
        "cost":{"registered_trial_runtime_seconds_sum":sum(float(row.get("runtime") or 0) for row in registry["trials"]),"selected_model_training_rows":training_audit["rows"],"selected_model_positive_rows":training_audit["positive_rows"],"selected_model_screen_fit_and_four_inner_seconds":read(REPORT/"WV3-661_SCREEN.json")["runtime_seconds"],"selected_policy_four_outer_seconds":champion["runtime_seconds"],"inference_scope_note":"outer runtime includes local feature joins, scoring, pair construction and exact evaluation; not online serving latency","selected_action_training":"WV3-691 adds no training; reuses WV3-661"},
        "outer_exposure":{"count":registry["outer_exposure_count"],"experiment_ids":outer_ids},"tests":tests,"final_week":{"cutoff":"2020-09-16","status":"not_run","labels_accessed":False,"submission_created":False},
        "remaining_uncertainty":["Mean remains0.001831600 below0.03.","No valid2019 cross-year replay because2019 slices are training supervision.","Most swaps remain neutral.","Online latency and Cold/Admission calibration are unmeasured."],
        "integration_cautions":["Keep Warm/Cold partition and inactive fallback.","Integrate ordered Warm lists or admission decisions, not raw probabilities on the Cold scale.","Preserve cutoff safety, full denominator, duplicate policy and deterministic ties.","Do not import rejected Top100, causal-rich, LambdaRank or third-swap variants."],"registry_summary":{"trial_count":len(registry["trials"]),"outer_exposure_count":registry["outer_exposure_count"],"current_champion":"WV3-691"}}
    registry.update({"status":final["status"],"resume_started_at":START.isoformat(),"resume_deadline":DEADLINE.isoformat(),"resume_closed_at":closed.isoformat(),"resume_wall_seconds":elapsed,"target_mean_map":.03,"target_achieved":False,"current_champion":"WV3-691","best_stable":{"experiment_id":"WV3-691","mean_MAP":champion["mean_MAP"],"delta_vs_WV2_601":champion["delta_vs_WV2_601"],"nondegrade_windows":champion["nondegrade_windows"],"worst_delta":champion["worst_delta"]},"search_stop_reason":final["timer"]["stop_reason"],"tests":tests,"final_week":final["final_week"]});write(REGISTRY,registry);write(FINAL_JSON,final)
    maps=champion["per_window_MAP"];d=champion["per_window_delta"]
    md=f'''# Warm-v3 Autonomous Architecture Research 最终报告

生成时间：{closed.isoformat()}

分支：`warm-v3-architecture-lab`
本轮起点：`{START_HEAD}`

# Final Decision Summary

## Starting point

WV2-601 的2020四窗平均 MAP@12 为 `{BASE_MEAN:.12f}`。目标是均值至少 `0.030000`，或在5小时到期时冻结最佳稳定候选。

## Main observations

关键突破是把监督单位从“用户—挑战商品—被替换商品三元组”改成“唯一用户—商品”。历史商品对只有 `1.258%` 会产生非零换位收益，原双头选中的换位约 `96%` 为中性。候选级购买倾向使同一商品只学习一次，再用概率差准入；随后补入75个已有召回、趋势和属性偏好字段，并把动作从最多一次扩到最多两次。

## Why each rejected family was rejected

- WV3-620 候选特征双头：相对父方案只2/4内层窗口不退化。
- WV3-630：第8–50名历史正例4,490，低于预注册5,000条门槛，训练前停止。
- WV3-641 Top100：外层只比Top50高0.000007385且只赢2/4，额外成本不值。
- WV3-650 因果扩窗：本身稳定但收益小；与丰富特征组合的 WV3-670 在非对照3/3内层窗退化。
- WV3-680 丰富特征 LambdaRank：相对二分类点式模型只有1/4内层窗不退化。

## Why the final architecture was selected

WV3-691 是本轮2020四窗均值最高且满足冻结门槛的方案：相对 WV2-601 四窗全部提升；相对父方案 WV3-661 为3/4提升，最差仅 `-0.000001804`。它复用冻结模型，只把经审计的第二次尾部换位兑现为精确 Top12 决策。

## Remaining uncertainty

均值仍比0.03低 `0.001831600`。绝大多数实际换位仍为中性。WV3-691 没有可诚实报告的2019跨年重放：四个2019窗口正是训练监督；同窗评分属于训练内结果。因未触发0.03 SUCCESS，本轮没有另造2018训练链。

## Integration risks

Warm 概率不能直接和 Cold/Admission 分数比较；应传递完整 Warm 有序列表或明确换位决策。集成必须保留活跃用户口径、无证据回退、截止日安全、完整用户分母和确定性并列顺序。

## 术语表与评测口径

- MAP@12：行业通用前12位平均准确率；逐用户算 AP@12，再对完整真值用户分母求均值。
- 候选商品级购买倾向（行业常见 pointwise propensity）：以用户—商品—监督窗口为一行估计下一周购买概率；本项目主要用于同用户商品间作差。
- 局部准入（本项目自定义）：冻结第1–7名，只允许第13–50名挑战第8–12名；无正预测收益时保持原顺序。
- 最多两次换位（本项目自定义）：最多选择两个不同挑战商品和两个不同被替换商品；重建 Top12 后精确计算 AP。
- 丰富候选特征（本项目自定义）：99维输入，其中75个新增字段覆盖召回来源、商品热度/趋势及六类用户属性偏好。
- RRF（行业通用倒数名次融合）：两个基础排序名次转成倒数分数后等权相加；WV2-601 固定常数60。
- E0 / E1（本项目命名）：E0 为84维时间点安全 LambdaRank；E1 加入 BPR 用户—商品分数和不可用标记。
- 中性换位（本项目统计）：动作完成后用户 AP@12 与原顺序相同；不等于没有执行动作。
- 外层暴露（本项目计数）：一个冻结变体读取一次四窗外层结果；同一变体不得按外层救援。

## 2020 四窗结果

| 外层窗口 | WV2-601 | WV3-691 | 差值 |
|---|---:|---:|---:|
| winter | {BASE['winter_20200122']:.12f} | {maps['winter_20200122']:.12f} | {d['winter_20200122']:+.12f} |
| spring | {BASE['spring_20200318']:.12f} | {maps['spring_20200318']:.12f} | {d['spring_20200318']:+.12f} |
| early-summer | {BASE['early_summer_20200624']:.12f} | {maps['early_summer_20200624']:.12f} | {d['early_summer_20200624']:+.12f} |
| late-summer | {BASE['late_summer_20200819']:.12f} | {maps['late_summer_20200819']:.12f} | {d['late_summer_20200819']:+.12f} |
| mean | {BASE_MEAN:.12f} | {champion['mean_MAP']:.12f} | {champion['delta_vs_WV2_601']:+.12f} |

稳定门槛：相对 WV2-601 均值为正，`4/4` 不退化，最差窗口 `{champion['worst_delta']:+.12f}`。目标0.03未达到。

## 2019 robustness replay

WV3-691：`not_run_validly`。四个2019窗口是训练标签，不能把同窗训练内评分称作跨年重放。已有 WV2-601 FRESH-601 证据保持：均值 `0.025320693880`，相对 FRESH-000 `+0.000612605407`，4/4为正；它只支持原基线。

## 架构族与 pivot 逻辑

| 阶段 | 观察 | 决定 |
|---|---|---|
| WV3-600→610 | 一换一Oracle高，但成对模型约96%中性 | 增加可行动性头；收益仍小 |
| WV3-620→631 | 候选特征双头不稳，商品对可行动率1.258% | 改成候选商品级概率，Top1–50补正例 |
| WV3-640→650 | Top100有真值但兑现低；2019监督偏旧 | 分别试扩展与扩窗，均微增 |
| WV3-660→670 | 缓存有75个未使用匹配字段 | 丰富表示改善；与扩窗组合退化 |
| WV3-680 | 怀疑全局二分类不如组内排序 | LambdaRank只赢1/4，保留概率差 |
| WV3-690→691 | 第二次换位额外Oracle为0.000698 | 复用冻结分数，最多两次精确换位 |

最终收益来自多阶段局部准入和候选表示，不来自新召回、图表示、序列模型或用户级 gating。候选池没有变化；Top100 未入选。

## Final Promoted Warm Architecture

```mermaid
flowchart TD
    A[六路启发式召回 Top100] --> C[WV2 候选池]
    B[Item2Vec 独有候选 最多200] --> C
    C --> D[84维时间点安全特征]
    C --> E[BPR 用户-商品协同分数]
    D --> F[E0 LambdaRank]
    D --> G[E1 LambdaRank]
    E --> G
    F --> H[等权名次 RRF / WV2-601]
    G --> H
    H --> I[读取融合名次1-50]
    I --> J[99维候选特征]
    J --> K[二分类购买倾向 LightGBM]
    K --> L{{有正的预测换位收益?}}
    L -- 否 --> M[保持 WV2-601 顺序]
    L -- 是 --> N[第1-7名冻结\n13-50挑战8-12\n最多两个不重复换位]
    M --> O[Top12]
    N --> O
    P[无活跃证据用户] --> M
```

## 训练、推理与收口成本

- 模型搜索在 `{search_elapsed/3600:.3f}` 小时停止；包含报告、测试和清单的总墙钟 `{elapsed/3600:.3f}` 小时，超过5小时总上限 `{max(0.0,(closed-DEADLINE).total_seconds())/60:.3f}` 分钟。超时部分只做收口，没有新实验或新外层暴露。这是本轮时间管理缺陷，未作掩盖。
- 模型使用 `{training_audit['rows']:,}` 个历史用户—商品—窗口样本，正例 `{training_audit['positive_rows']:,}`；WV3-661训练加四内层评分 `{read(REPORT/'WV3-661_SCREEN.json')['runtime_seconds']:.1f}` 秒。
- WV3-691 不新增训练；四外层特征连接、打分、商品对构造和精确评分 `{champion['runtime_seconds']:.1f}` 秒。它是离线实验时间，不是线上延迟。
- 测试 `116/116` 通过。外层暴露累计 `{registry['outer_exposure_count']}` 个冻结变体。
- 最终周 `2020-09-16` 为 `not_run`：未读标签、未生成提交、未看榜单。
''';FINAL_MD.write_text(md,encoding="utf-8")
    upsert(PRIVATE_LOG,"# Final Decision Summary",f'''# Final Decision Summary

## Starting point
恢复运行从 {START.isoformat()} 开始，WV2-601 mean={BASE_MEAN:.12f}，目标0.03。

## Main observations
商品对可行动率1.258%；点式购买概率、75个丰富缓存特征和第二次受限换位依次形成增益。

## Why each rejected family was rejected
Top100成本收益差；丰富特征+因果扩窗不相加；LambdaRank只赢1/4；失败均按预注册门槛停止。

## Why the final architecture was selected
WV3-691 mean={champion['mean_MAP']:.12f}，相对WV2-601={champion['delta_vs_WV2_601']:+.12f}，4/4提升。

## Remaining uncertainty
目标未达；2019与训练重合，未伪造跨年重放；最终周not_run。

## Integration risks
Warm概率不可直接拼Cold分数；保留人群分区、回退、截止日、分母和确定性顺序。''')
    upsert(ROADMAP,"## Warm-v3 自主架构研究收口",f'''## Warm-v3 自主架构研究收口

- 状态：完成；目标0.03未达；当前稳定候选WV3-691。
- 2020 mean MAP@12：{champion['mean_MAP']:.12f}，相对WV2-601 {champion['delta_vs_WV2_601']:+.12f}，4/4不退化。
- 结构：WV2-601后接99维候选购买倾向；冻结1–7名，13–50名对8–12名最多两个不重复换位。
- 2019：四个切片用于训练，不在同窗伪造robustness replay；未来需更早训练链。
- 最终周2020-09-16：not_run。''')
    upsert(PROJECT_LOG,"## 2026-09-10 Warm-v3 恢复探索收口",f'''## 2026-09-10 Warm-v3 恢复探索收口

- 已解决监督稀释：WV3-631 mean={wv631['mean_MAP']:.12f}。
- 已利用丰富缓存：WV3-661 mean={wv661['mean_MAP']:.12f}。
- 已扩动作：WV3-691 mean={champion['mean_MAP']:.12f}。
- 未解决：距0.03仍差 {0.03-champion['mean_MAP']:.9f}，多数动作中性。
- 否决：Top100成本收益差、因果丰富组合不相加、LambdaRank不稳。
- 边界：最终周not_run；未merge、未push；2019同训练窗不冒充验证。''')
    entries=[]
    for path in public_files():
        item={"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"nonempty":path.stat().st_size>0}
        if path.suffix==".json":
            try:json.loads(path.read_text(encoding="utf-8"));item["json_parse_ok"]=True
            except Exception:item["json_parse_ok"]=False
        entries.append(item)
    manifest={"generated_at":closed.isoformat(),"verification":"structural existence, nonempty size and JSON parsing; no standalone hashes because no trusted comparison artifact exists","file_count":len(entries),"all_nonempty":all(x["nonempty"] for x in entries),"all_json_parse":all(x.get("json_parse_ok",True) for x in entries),"files":entries,"final_week":"not_run"};write(MANIFEST,manifest)
    print({"status":final["status"],"champion":"WV3-691","mean_MAP":champion["mean_MAP"],"delta":champion["delta_vs_WV2_601"],"trials":len(registry["trials"]),"outer_exposures":registry["outer_exposure_count"],"manifest_files":len(entries),"wall_hours":elapsed/3600,"final_week":"not_run"},flush=True)

if __name__=="__main__":main()
