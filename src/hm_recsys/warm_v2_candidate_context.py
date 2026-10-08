"""Label-blind features relative to each user's complete frozen candidate pool."""
from __future__ import annotations

import time

import numpy as np

from .warm_v2_contract import ARTIFACT, REPORT, evidence_id, guard, now, read, write
from .warm_v2_engine import connection, literal, save_parquet


SIGNALS={
    'item_events_7d':'商品7天全局销量',
    'item_events_28d':'商品28天全局销量',
    'item2vec_cosine':'冻结Item2Vec相似度的非负部分',
    'user_item_events_28d':'用户28天同商品购买事件数',
    'user_product_code_events_28d':'用户28天同商品族购买事件数',
    'user_product_type_events_28d':'用户28天同类别购买事件数',
    'user_department_events_28d':'用户28天同部门购买事件数',
    'user_garment_events_28d':'用户28天同服装组购买事件数',
    'user_colour_events_28d':'用户28天同主色购买事件数',
}
CONTEXT_SPEC={'candidate_context':{f'context_{signal}_{suffix}':description
    for signal,label in SIGNALS.items() for suffix,description in [
        ('rank_fraction',f'{label}在该用户完整冻结候选池中的降序并列名次减1，除以候选数减1；无正信号缺失；不是采样训练组中的名次'),
        ('strength_share',f'候选的{label}非负值除以该用户完整候选池对应信号非负值总和；分母至少1e-12。是候选集合归一化，不是新的购买概率')]
}}


def register_context():
    path=REPORT/'WARM_FEATURE_SPEC_CONTEXT.json'
    if not path.exists():
        write(path,{'registered_at':now(),'families':CONTEXT_SPEC,'classification':'All names are project-defined; objects are frozen user-candidate pairs.',
            'hypothesis':'Trees see absolute strengths but not how strong an item is compared with the same user other candidates. Normalize existing signals within the full frozen pool, before negative sampling.',
            'leakage_boundary':'Read only explicit label-free base columns; NEVER compute over positive-only or sampled groups; outer candidate universe fixed; no target/read of labels in aggregation.',
            'cost_estimate':'4 historical fits, CPU 8 threads, about 3-8 minutes; no new transactions or model family.',
            'gate':'Same preregistered historical/outer gates; at most remaining original outer-confirmation budget. Do not bypass exhausted item_demand family through combinations.',
            'final_week':'not_run'})


def compute(con,source):
    fields=[]
    for signal in SIGNALS:
        value=f'greatest(coalesce({signal},0),0)'
        fields += [f'''CASE WHEN {value}>0 THEN
            (rank() OVER(PARTITION BY customer_id ORDER BY {value} DESC)-1)::DOUBLE /
            greatest(count(*) OVER(PARTITION BY customer_id)-1,1) END AS wv2_context_{signal}_rank_fraction''',
            f'{value}/greatest(sum({value}) OVER(PARTITION BY customer_id),1e-12) AS wv2_context_{signal}_strength_share']
    return con.execute(f"SELECT customer_id,article_id,{','.join(fields)} FROM {source} ORDER BY customer_id,article_id").fetchdf()


def build_context(cutoff,family,engine):
    guard(cutoff);register_context()
    path=ARTIFACT/'features-context-v1'/cutoff/'candidate_context.parquet';meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        return path,read(meta)
    start=time.perf_counter()
    with connection() as con:
        source=f"(SELECT customer_id,article_id,{','.join(SIGNALS)} FROM read_parquet({literal(engine.base_path(cutoff))}))"
        frame=compute(con,source)
    keys=['customer_id','article_id']
    assert not frame.duplicated(keys).any()
    cols=[c for c in frame.columns if c.startswith('wv2_')]
    frame[cols]=frame[cols].astype(np.float32)
    assert not np.isinf(frame[cols].to_numpy()).any()
    save_parquet(frame,path)
    result={'cutoff':cutoff,'family':family,'keys':keys,'rows':len(frame),'features':cols,
        'source':str(engine.base_path(cutoff)),'source_sha256':engine.history['feature_cache'][cutoff]['artifact']['sha256'],
        'PIT':'label-free deterministic transformation of trusted PIT base fields over complete frozen candidates',
        'pit_safe':True,'future_labels_used':False,'runtime_seconds':time.perf_counter()-start,
        'artifact':evidence_id(path,reason='explicit_registry_evidence')}
    write(meta,result)
    print(f'Warm candidate-context cache {cutoff}: {len(frame)} pairs, {result["runtime_seconds"]:.1f}s',flush=True)
    return path,result


if __name__=='__main__':
    from .warm_v2_lab import screen
    register_context()
    screen('WV2-110',['candidate_context'],'原特征只有绝对强度，树模型无法直接比较同用户其他候选；在完整冻结候选池内归一化已有销量、向量相似度与个人层级偏好。')
