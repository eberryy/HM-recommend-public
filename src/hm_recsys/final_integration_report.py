"""Close only the completed 10% integration; never start full-scale work."""
import hashlib
from datetime import datetime, timezone
import subprocess
from .final_integration import ROOT, REPORT, RUN, CONTRACT, WROOT, read, write, WINDOWS


def main():
    result=read(REPORT/'FINAL_INTEGRATION_10PCT.json')
    verification=read(REPORT/'FINAL_INTEGRATION_10PCT_VERIFICATION.json')
    assert verification['status']=='pass'
    c=read(CONTRACT)
    rows=result['windows']
    warm=dict(name='WV3-741',source_commit='1e512a4e6de309daa203abb1e962f0180f36e7b2',
        source_branch='warm-v3-architecture-lab',merge_commit='3fd79df33aaed0b6492dff58cafa62fff3982e07',
        historical_replay_contract='FINAL_HISTORY_REBUILD_CONTRACT.json',
        candidate_lineage='same M3.3 six-source Top100 + up to200 Item2Vec-only, full-history global stats and fixed hash10 users',
        first_stage='frozen WV2-000 84 features and WV2-501 86 features incl BPR; equal RRF60',
        second_stage_models={name:read(WROOT/f'artifacts/warm_v3/{name}/MODEL.json') for name in ['WV3-661','WV3-680']},
        RRF=dict(constant=60,weights=[1,1],normalization='per user score ranks'),
        swaps=dict(maximum=2,challengers=[13,50],victims=[8,12],protected=[1,7],disjoint=True),
        P018_in_final=False,later_time_evidence=read(ROOT/'reports/mind_warm_side/MIND_WARM_FINAL001_VERIFICATION.json')['replayed'],
        later_time_scope='2020-09-16 Warm confirmation only; optimistic catalog and fixed truth-user cohort; not Kaggle leaderboard',
        final_week_integration='not_run')
    write(REPORT/'FINAL_WARM_BASELINE_CONTRACT.json',warm)
    full=dict(status='not_run',reason='user requested stop and report after10pct' if result['fullscale_allowed'] else '10pct failed frozen integration gates',final_week='not_run')
    write(REPORT/'FINAL_INTEGRATION_FULL.json',full)
    (REPORT/'FINAL_INTEGRATION_FULL.md').write_text('# 100%历史训练扩量\n\n尚未运行。原因：'+full['reason']+'。\n',encoding='utf-8')
    lines=['# WV3-741 + 冻结 Cold-A：10%集成结果','',
        '## 结论','',f"四个开发窗平均 MAP@12 增量为 **{result['mean_delta']:+.12f}**。扩量门槛：**{'通过' if result['fullscale_allowed'] else '未通过'}**。本轮100%未运行，2020-09-16融合未运行。",'',
        '## 术语与分母','',
        '- MAP@12（行业通用离线指标）：每用户前12项命中的累计位置精度，除以min(该用户真实购买商品数,12)，再对本窗固定用户等权平均；四窗汇总是窗口等权均值。',
        '- Warm（暖商品）组为截止日前交易记录数至少21；strict cold（严格冷商品）为0；sparse（稀疏商品）为1–5；cold+sparse为0–5。这些是商品分组，不是冷用户分组。分组MAP只保留该组真值，分母为该组有真值的用户，不能将各组MAP直接相加。',
        '- 有益/有害/中性动作：一次替换使对应用户总体AP上升/下降/不变。插入、移除正例为四窗累计命中的用户—商品—窗口对，不是不同商品数；顺序和真值分母也影响MAP。',
        '- 覆盖率为至少执行一次Cold准入的用户数除以本窗全部固定用户数。',
        '- 冻结A为A1-L15-D6-M50：70维动作特征、15叶/深度6/叶最少50样本、300棵树、学习率0.05，全部有益/有害加确定性2%中性，权重全部1。',
        '- p43a-policy-0332（项目策略编号）：每用户先取全动作最高分1条，再同时要求全窗平均并列名次百分位>=99.9%、Cold原B0名次<=5；位置1–12，每人最多1条。不回退选择第二条动作，不要求原始分数为正。','',
        '## 完整用户结果','',
        '| 开发窗 | 用户数 | WV3-741 | 集成 | 增量 | 替换数 | 插入正例 | 移除正例 | 有益/有害/中性动作数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for w,r in rows.items():
        a=r['executed_actions']
        lines.append(f"| {WINDOWS[w]} | {r['users']} | {r['baseline_map']:.12f} | {r['overall_map']:.12f} | {r['delta_map']:+.12f} | {r['replacements']} | {r['inserted_positives']} | {r['removed_positives']} | {a['beneficial']} / {a['harmful']} / {a['neutral']} |")
    lines+=['','## 商品分组增量','',
        '| 开发窗 | 暖商品>=21 | 严格冷商品0 | 稀疏1–5 | 冷稀疏0–5 | 准入覆盖率 |',
        '|---|---:|---:|---:|---:|---:|']
    for w,r in rows.items():
        values=[r['segments'][s]['delta'] for s in ['warm_21_plus','strict_cold','sparse1_5','all_cold_sparse']]
        lines.append('| '+WINDOWS[w]+' | '+' | '.join(f'{v:+.12f}' for v in values)+f" | {r['coverage']:.6%} |")
    lines+=['',f"暖商品平均增量 {result['mean_warm_delta']:+.12f}；冷稀疏平均增量 {result['mean_cold_sparse_delta']:+.12f}。插入/移除合计 {result['inserted_positives']} / {result['removed_positives']}。",'',
        '## 逐项门槛','']
    labels={'mean_positive':'整体均值差>0','three_nondegrade':'至少3窗差>=-1e-5','worst':'最差窗差>=-5e-5','warm':'暖组均值差>=-2e-5','cold_sparse':'冷稀疏均值差>=0','inserted_positive':'实际插入冷正例至少1','efficiency':'插入正例不少于移除正例'}
    for k,v in result['gates'].items():lines.append(f"- {labels[k]}：{'通过' if v else '未通过'}。")
    lines+=['','## 重建与证据边界','',
        '历史Warm预测采用已登记时间安全旧模型；四开发窗恢复冻结WV3-741。Cold50保持原M4/B0来源，在完整50件内计算相对特征后，再根据新Warm Top12排除重合。新动作人口和无标签特征先落盘，之后才接入对应周真值计算替换标签；不沿用旧W0标签。',
        'Warm分数改用WV3-741两模型的RRF60融合分数，在新Top12内重新归一化；Warm排名比例分母仍为完整候选数。qC/qW辅助模型不重训、不重新校准，只用时间安全原模型对新输入推断；因此本次不声称解决辅助模型向新Warm分布迁移的问题。',
        '独立核验未调用主策略匹配器：直接重放最高分1条和平均并列百分位门，并用旧AP函数重算全部分组MAP；另外核对最终列表、无候选用户不变、模型参数/训练日期和序列化模型分数抽样。全部通过，不等于独立重训四个模型。',
        '冬季第一次保存汇总因NumPy整数JSON编码失败；模型、分数、决策和最终列表已经保存。只恢复汇总序列化，无重训、无新策略。错误及恢复记录保留。',
        '这是在多次暴露的开发窗上，将开发选择出的固定策略迁移到新Warm环境；不是未触碰测试集的独立确认。没有根据本次结果调参或更换方案。','',
        '## 复现与当前状态','',
        '`python -m hm_recsys.final_integration prepare --cutoff YYYY-MM-DD`；`window --window WINDOW_ID`；完成四窗后`finish`；独立核验`python -m hm_recsys.final_integration_verify`。合同已存在，不再register；失败目录不能无记录重跑。',
        '本轮按用户要求停在10%结果汇报。没有运行100%，也未宣布全项目最终文档已完成。报告全部本地gitignore。']
    (REPORT/'FINAL_INTEGRATION_10PCT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    selected='WV3-741' if not result['fullscale_allowed'] else 'WV3-741 pending permitted fullscale decision'
    write(REPORT/'FINAL_SYSTEM_DECISION.json',dict(warm_baseline='WV3-741',cold_candidate=c['model']['id'],cold_policy=c['policy']['id'],
        cold_10pct_integration_status='completed',cold_10pct_metrics=result,cold_fullscale_status='not_run',final_system=selected,
        final_week_integration_status='not_run',project_wrapup='documents_pending',decision_reason='10pct gate fail' if not result['fullscale_allowed'] else 'pause at user requested10pct report'))
    files=[CONTRACT,REPORT/'FINAL_INTEGRATION_10PCT.json',REPORT/'FINAL_INTEGRATION_10PCT_VERIFICATION.json',REPORT/'FINAL_INTEGRATION_10PCT.md']
    files+=list((ROOT/'src/hm_recsys').glob('final_*.py'))
    files+=list(RUN.glob('models/*/model.txt'))
    # Explicit reproducibility manifest; current-source hashes are not evidence
    # that a pre-recovery source file was identical. Record the limited repair.
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='10pct final integration only',
        source_repair='winter summary np.int64 encoding fixed after fit; later source retains identical model/policy params',
        files=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        final_week='not_run',fullscale='not_run')
    write(REPORT/'FINAL_INTEGRATION_10PCT_MANIFEST.json',manifest)


if __name__=='__main__':main()
