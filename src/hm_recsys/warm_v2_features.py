"""Bounded Warm feature bundles. Every behavioral query is strictly pre-cutoff."""
from __future__ import annotations

from functools import lru_cache
import time

import numpy as np
import pandas as pd

from .warm_v2_contract import ARTIFACT, REPORT, evidence_id, guard, read, write
from .warm_v2_engine import connection, literal, load_parquet, save_parquet

# Each entry explicitly states object, unit/denominator, and construction.
SPEC = {
    "item_demand": {
        "item_events_1d": "商品截止前1天原始交易事件数（保留重复行）",
        "item_events_3d": "商品截止前3天交易事件数",
        "item_events_14d": "商品截止前14天交易事件数",
        "item_events_56d": "商品截止前56天交易事件数",
        "item_users_3d": "商品截止前3天去重购买用户数",
        "item_users_7d": "商品截止前7天去重购买用户数",
        "item_rate_3_vs_prior3": "(商品近3天事件数+1)/(此前3天事件数+1)，等长短期增长比",
        "item_rate_3_vs_prior25": "商品近3天平滑日销量与此前25天平滑日销量之比；各加1次事件后除各自天数",
        "item_rate_7_vs_prior49": "商品近7天平滑日销量与此前49天平滑日销量之比；各加1次事件后除各自天数",
        "item_days_sold_7d": "近7天商品有至少一笔销售的去重日期数",
        "item_days_sold_28d": "近28天商品有销售的去重日期数",
        "item_peak_age_28d": "近28天日销量最高日距截止日的天数；并列选最近一天，不称真实上市/促销日",
        "item_peak_share_28d": "近28天最高单日销量除以28天总销量；无近期销量缺失",
        "item_repeat_event_share_28d": "(28天事件数−去重买家数)/事件数；是重复买家事件代理，含原始重复记录，不称真实复购率",
        "item_observed_age": "截止前全历史首次观察到商品销售距截止日天数；不等同真实上市年龄",
        "item_last_sale_age_all": "截止前全历史最后商品销售距截止日天数；不同于原84天截断",
        "item_events_all": "截止前全历史商品交易事件数",
        "item_lifetime_daily_rate": "截止前全历史商品事件数除以首次观察销售至截止日的天数",
        "item_recent_vs_lifetime": "商品近7天日销量除以全历史平均日销量；分母加1e-6",
        "item_observed_within28": "首次观察销售在近28天以内的布尔值，不宣称库存可得",
    },
    "user_rhythm": {
        "user_events_7d": "用户近7天交易事件数",
        "user_events_28d": "用户近28天交易事件数",
        "user_items_7d": "用户近7天去重商品数",
        "user_items_28d": "用户近28天去重商品数",
        "user_purchase_days_28d": "用户近28天有购买的去重日期数；无真实订单号",
        "user_purchase_days_84d": "用户近84天有购买的去重日期数",
        "user_gap_mean_84d": "用户近84天相邻购买日期间隔的均值，单位天；不足两日缺失",
        "user_gap_median_84d": "用户近84天相邻购买日期间隔中位数，单位天",
        "user_gap_std_84d": "用户近84天相邻购买日期间隔的总体标准差，单位天",
        "user_last_gap_84d": "用户近84天最后两个购买日期的间隔，单位天",
        "user_event_share_7_84": "用户7天事件数除以84天事件数，分母至少1",
        "user_events_per_purchase_day": "用户84天事件数除以该期购买日期数；用户日篮子代理，不是真实订单大小",
        "user_online_share_28d": "用户近28天sales_channel_id=2事件数占同期全部事件数；零事件缺失",
        "user_price_median_28d": "用户近28天交易价格中位数，沿用数据价格单位",
        "user_price_std_28d": "用户近28天交易价格总体标准差",
        "item_user_price_ratio": "候选商品冻结28天均价除以用户28天价格中位数；分母加1e-6，缺失不补均值",
        "item_user_price_standard_gap": "候选28天均价减用户28天价格中位数，再除用户价格标准差+1e-6，截断到[-10,10]",
        "user_cycle_phase": "用户距最近购买天数除以其84天购买间隔中位数+1；用户状态而非未来活动标签",
    },
    "repeat_affinity": {
        "user_item_events_3d": "用户近3天购买该候选商品的事件数",
        "user_item_events_7d": "用户近7天购买该候选商品的事件数",
        "user_item_days_84d": "用户近84天购买该商品的去重日期数，区别于重复事件数",
        "user_code_events_7d": "用户近7天购买候选同product_code商品族的事件数",
        "user_type_events_7d": "用户近7天购买候选同商品类别的事件数",
        "user_department_events_7d": "用户近7天购买候选同部门的事件数",
        "user_garment_events_7d": "用户近7天购买候选同服装组的事件数",
        "user_colour_events_7d": "用户近7天购买候选同主色组的事件数",
        "same_last_code": "候选与用户最近购买商品是否同商品族；同日选编号最大者，只是确定性代理",
        "same_last_type": "候选与用户最近购买商品是否同商品类别；同日没有真实顺序信息",
        "same_last_colour": "候选与用户最近购买商品是否同主色组",
        "item_repeat_recency_strength": "用户84天购买该商品事件数除以距最后购买该商品天数+1；无匹配为0",
        "code_repeat_recency_strength": "用户84天购买同商品族事件数除以距最后同族购买天数+1；无匹配为0",
        "type_recent_share_7d": "用户7天同类别购买事件数除以用户7天全部购买事件数，分母至少1",
        "type_recent_vs_long": "用户7天同类别购买数除以84天同类别购买数+1，近期偏好集中度",
        "trend_times_type_affinity": "log1p(商品7天销量)乘用户7天该类别份额；交叉商品需求与个人偏好",
        "item2vec_times_recent_type": "冻结Item2Vec余弦相似度乘用户7天该类别份额；不重训Item2Vec",
    },
    "source_agreement": {
        "source_best_rank": "六条原始召回路中已出现候选的最小名次；缺失路不计",
        "source_second_rank": "六条路已出现候选的第二小名次；不足两路缺失",
        "source_log_rank_std": "六条已出现路的log1p名次总体标准差，只有一路为0",
        "source_top10_support": "六条已出现召回路中名次≤10的路数；是原路内名次不是合并排名",
        "source_top30_support": "六条已出现召回路中名次≤30的路数",
        "source_best_strength_share": "六路最大1/(60+名次)除以各已出现路相同强度之和；不是重新融合排序",
        "source_first_second_gap": "第二好召回名次减最好名次；不足两路缺失",
        "repurchase_and_item2vec": "复购路出现且Item2Vec出现的布尔交叉",
        "covisit_and_item2vec": "用户日共现路出现且Item2Vec出现的布尔交叉",
        "source_count_times_i2v": "原source_count乘冻结Item2Vec余弦相似度；不改变候选",
    },
}
SOURCES = ("repurchase","recent_popularity","product_family","user_day_covisit","age_popularity","attribute_content")


def columns_for(families):
    from .warm_v2_extra_features import EXTRA_SPEC
    from .warm_v2_candidate_context import CONTEXT_SPEC
    from .warm_v2_bpr import BPR_SPEC
    catalog={**SPEC,**EXTRA_SPEC,**CONTEXT_SPEC,**BPR_SPEC}
    return list(dict.fromkeys("wv2_"+key for family in families for key in catalog[family]))


def register_spec():
    path = REPORT/"WARM_FEATURE_SPEC.json"
    if not path.exists():
        write(path,{"schema":"warm-v2-feature-spec-v1","status":"frozen_before_first_feature_fit",
            "families":SPEC,"PIT":"all transaction SQL t_dat < cutoff, no final labels or outer-window router",
            "classification":"all wv2_* are project-defined names; descriptions specify objects/units/denominators",
            "missing":"counts missing->0; unavailable prices/recencies/references->NaN unless description says absent affinity->0"})


def _base_views(con,cutoff,engine):
    guard(cutoff)
    con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet({literal(engine.base_path(cutoff))})")
    con.execute("CREATE TEMP TABLE wanted_users AS SELECT DISTINCT customer_id FROM base")
    con.execute(f"CREATE TEMP VIEW history AS SELECT * FROM read_parquet({literal(engine.transactions)}) WHERE t_dat<DATE '{cutoff}'")
    con.execute(f"CREATE TEMP VIEW h84 AS SELECT * FROM history WHERE t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY")
    con.execute("CREATE TEMP VIEW uh AS SELECT h.* FROM h84 h SEMI JOIN wanted_users u USING(customer_id)")


def _item(con,c):
    aggs = [f"count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL {d} DAY)::DOUBLE AS e{d}" for d in (1,3,6,7,14,28,56)]
    aggs += [f"count(DISTINCT customer_id) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL {d} DAY)::DOUBLE AS u{d}" for d in (3,7,28)]
    aggs += [f"count(DISTINCT t_dat) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL {d} DAY)::DOUBLE AS d{d}" for d in (7,28)]
    con.execute(f"CREATE TEMP TABLE recent_item AS SELECT article_id,{','.join(aggs)} FROM h84 GROUP BY article_id")
    con.execute(f"CREATE TEMP TABLE life AS SELECT article_id,count(*)::DOUBLE AS eall,date_diff('day',min(t_dat),DATE '{c}')::DOUBLE AS age,date_diff('day',max(t_dat),DATE '{c}')::DOUBLE AS lastage FROM history GROUP BY article_id")
    con.execute(f"""CREATE TEMP TABLE peaks AS SELECT article_id,date_diff('day',t_dat,DATE '{c}')::DOUBLE AS peakage,n AS peakn
        FROM (SELECT article_id,t_dat,count(*)::DOUBLE n FROM h84 WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY GROUP BY article_id,t_dat)
        QUALIFY row_number() OVER(PARTITION BY article_id ORDER BY n DESC,t_dat DESC)=1""")
    return con.execute("""SELECT l.article_id,
        coalesce(e1,0) AS wv2_item_events_1d,coalesce(e3,0) AS wv2_item_events_3d,
        coalesce(e14,0) AS wv2_item_events_14d,coalesce(e56,0) AS wv2_item_events_56d,
        coalesce(u3,0) AS wv2_item_users_3d,coalesce(u7,0) AS wv2_item_users_7d,
        (coalesce(e3,0)+1)/(coalesce(e6-e3,0)+1) AS wv2_item_rate_3_vs_prior3,
        (coalesce(e3,0)+1)/3/((coalesce(e28-e3,0)+1)/25) AS wv2_item_rate_3_vs_prior25,
        (coalesce(e7,0)+1)/7/((coalesce(e56-e7,0)+1)/49) AS wv2_item_rate_7_vs_prior49,
        coalesce(d7,0) AS wv2_item_days_sold_7d,coalesce(d28,0) AS wv2_item_days_sold_28d,
        peakage AS wv2_item_peak_age_28d,peakn/nullif(e28,0) AS wv2_item_peak_share_28d,
        (e28-u28)/nullif(e28,0) AS wv2_item_repeat_event_share_28d,
        age AS wv2_item_observed_age,lastage AS wv2_item_last_sale_age_all,eall AS wv2_item_events_all,
        eall/greatest(age,1) AS wv2_item_lifetime_daily_rate,
        coalesce(e7,0)/7/(eall/greatest(age,1)+1e-6) AS wv2_item_recent_vs_lifetime,
        (age<=28)::DOUBLE AS wv2_item_observed_within28
        FROM life l LEFT JOIN recent_item USING(article_id) LEFT JOIN peaks USING(article_id)""").fetchdf()


def _rhythm(con,c):
    aggs = [f"count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL {d} DAY)::DOUBLE AS e{d}" for d in (7,28)]
    aggs += [f"count(DISTINCT article_id) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL {d} DAY)::DOUBLE AS i{d}" for d in (7,28)]
    con.execute(f"""CREATE TEMP TABLE user_stats AS SELECT customer_id,{','.join(aggs)},count(*)::DOUBLE AS e84,
        count(DISTINCT t_dat)::DOUBLE AS d84,count(DISTINCT t_dat) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY)::DOUBLE AS d28,
        avg((sales_channel_id=2)::DOUBLE) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY) AS online28,
        median(price) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY) AS price28,
        stddev_pop(price) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY) AS pricestd28
        FROM uh GROUP BY customer_id""")
    con.execute("""CREATE TEMP TABLE user_gaps AS SELECT customer_id,avg(gap) AS gapmean,median(gap) AS gapmedian,stddev_pop(gap) AS gapstd,arg_max(gap,t_dat) AS lastgap FROM (
        SELECT customer_id,t_dat,date_diff('day',lag(t_dat) OVER(PARTITION BY customer_id ORDER BY t_dat),t_dat)::DOUBLE AS gap
        FROM (SELECT DISTINCT customer_id,t_dat FROM uh)) GROUP BY customer_id""")
    return con.execute("""SELECT u.customer_id,e7 AS wv2_user_events_7d,e28 AS wv2_user_events_28d,
        i7 AS wv2_user_items_7d,i28 AS wv2_user_items_28d,d28 AS wv2_user_purchase_days_28d,d84 AS wv2_user_purchase_days_84d,
        gapmean AS wv2_user_gap_mean_84d,gapmedian AS wv2_user_gap_median_84d,gapstd AS wv2_user_gap_std_84d,lastgap AS wv2_user_last_gap_84d,
        e7/greatest(e84,1) AS wv2_user_event_share_7_84,e84/greatest(d84,1) AS wv2_user_events_per_purchase_day,
        online28 AS wv2_user_online_share_28d,price28 AS wv2_user_price_median_28d,pricestd28 AS wv2_user_price_std_28d
        FROM user_stats u LEFT JOIN user_gaps USING(customer_id)""").fetchdf()


def _affinity(con,c,engine):
    articles = engine.history['prerequisite_cache']['target_features'][c]['inputs']['articles']['path']
    con.execute(f"""CREATE TEMP TABLE dim AS SELECT article_id,product_code AS code,product_type_no AS type,
        department_no AS department,garment_group_no AS garment,perceived_colour_master_id AS colour
        FROM read_csv({literal(articles)},all_varchar=true)""")
    con.execute("CREATE TEMP TABLE history_dim AS SELECT h.*,d.* EXCLUDE(article_id) FROM uh h JOIN dim d USING(article_id)")
    con.execute(f"""CREATE TEMP TABLE ui AS SELECT customer_id,article_id,
        count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 3 DAY)::DOUBLE AS e3,
        count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 7 DAY)::DOUBLE AS e7,
        count(DISTINCT t_dat)::DOUBLE AS days84 FROM uh GROUP BY customer_id,article_id""")
    joins=[]
    selects=[]
    for name in ('code','type','department','garment','colour'):
        con.execute(f"CREATE TEMP TABLE a_{name} AS SELECT customer_id,{name},count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 7 DAY)::DOUBLE AS e7 FROM history_dim GROUP BY customer_id,{name}")
        joins.append(f"LEFT JOIN a_{name} a{name} ON b.customer_id=a{name}.customer_id AND d.{name}=a{name}.{name}")
        selects.append(f"coalesce(a{name}.e7,0) AS wv2_user_{name}_events_7d")
    con.execute("""CREATE TEMP TABLE lastitem AS SELECT customer_id,code,type,colour FROM history_dim
        QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY t_dat DESC,article_id DESC)=1""")
    con.execute(f"CREATE TEMP TABLE user7 AS SELECT customer_id,count(*)::DOUBLE e7 FROM uh WHERE t_dat>=DATE '{c}'-INTERVAL 7 DAY GROUP BY customer_id")
    return con.execute(f"""SELECT b.customer_id,b.article_id,coalesce(ui.e3,0) AS wv2_user_item_events_3d,
        coalesce(ui.e7,0) AS wv2_user_item_events_7d,coalesce(ui.days84,0) AS wv2_user_item_days_84d,
        {','.join(selects)},coalesce((d.code=l.code)::DOUBLE,0) AS wv2_same_last_code,
        coalesce((d.type=l.type)::DOUBLE,0) AS wv2_same_last_type,coalesce((d.colour=l.colour)::DOUBLE,0) AS wv2_same_last_colour,
        coalesce(b.user_item_events_12w/(b.user_item_days_since_last_purchase+1),0) AS wv2_item_repeat_recency_strength,
        coalesce(b.user_product_code_events_12w/(b.user_product_code_days_since+1),0) AS wv2_code_repeat_recency_strength,
        coalesce(atype.e7,0)/greatest(coalesce(u7.e7,0),1) AS wv2_type_recent_share_7d,
        coalesce(atype.e7,0)/(b.user_product_type_events_12w+1) AS wv2_type_recent_vs_long,
        ln(1+b.item_events_7d)*coalesce(atype.e7,0)/greatest(coalesce(u7.e7,0),1) AS wv2_trend_times_type_affinity,
        b.item2vec_cosine*coalesce(atype.e7,0)/greatest(coalesce(u7.e7,0),1) AS wv2_item2vec_times_recent_type
        FROM base b LEFT JOIN dim d USING(article_id) LEFT JOIN ui USING(customer_id,article_id)
        {' '.join(joins)} LEFT JOIN lastitem l ON b.customer_id=l.customer_id LEFT JOIN user7 u7 ON b.customer_id=u7.customer_id
        ORDER BY b.customer_id,b.candidate_rank,b.article_id""").fetchdf()


def build(cutoff,family,engine):
    if family=='bpr_bias':
        from .warm_v2_bpr import build_bias
        return build_bias(cutoff,engine)
    if family=='bpr_match':
        from .warm_v2_bpr import build_features
        return build_features(cutoff,engine)
    if family=='candidate_context':
        from .warm_v2_candidate_context import build_context
        return build_context(cutoff,family,engine)
    from .warm_v2_extra_features import EXTRA_SPEC, build_extra
    if family in EXTRA_SPEC:
        return build_extra(cutoff,family,engine)
    guard(cutoff)
    register_spec()
    root = ARTIFACT/'features-v1'/cutoff
    path = root/f'{family}.parquet'
    meta = root/f'{family}.json'
    if path.exists() and meta.exists():
        return path,read(meta)
    start=time.perf_counter()
    with connection() as con:
        _base_views(con,cutoff,engine)
        frame=_item(con,cutoff) if family=='item_demand' else _rhythm(con,cutoff) if family=='user_rhythm' else _affinity(con,cutoff,engine) if family=='repeat_affinity' else None
        if frame is None:
            raise ValueError(family)
        latest=str(con.execute('SELECT max(t_dat) FROM history').fetchone()[0])
        assert latest<cutoff
    keys=['article_id'] if family=='item_demand' else ['customer_id'] if family=='user_rhythm' else ['customer_id','article_id']
    assert not frame.duplicated(keys).any()
    numeric=[v for v in frame.columns if v.startswith('wv2_')]
    frame[numeric]=frame[numeric].astype(np.float32)
    assert not np.isinf(frame[numeric].to_numpy()).any()
    save_parquet(frame,path)
    evidence={'cutoff':cutoff,'family':family,'latest_history_date':latest,'keys':keys,'rows':len(frame),
        'features':numeric,'source':str(engine.base_path(cutoff)),'source_sha256':engine.history['feature_cache'][cutoff]['artifact']['sha256'],
        'pit_safe':True,'future_labels_used':False,'events_policy':'keep raw event duplicates; deduplicate purchase dates only for explicit date/gap statistics',
        'runtime_seconds':time.perf_counter()-start,'artifact':evidence_id(path,reason='explicit_registry_evidence')}
    write(meta,evidence)
    print(f'Warm feature cache {cutoff} {family}: {len(frame)} rows, {len(numeric)} columns, {evidence["runtime_seconds"]:.1f}s',flush=True)
    return path,evidence


@lru_cache(maxsize=3)
def _indexed(path,keys):
    frame=load_parquet(path)
    return frame.set_index(list(keys),verify_integrity=True)


def source_features(frame):
    ranks=np.stack([frame[s+'_rank'].to_numpy(float) for s in SOURCES],axis=1)
    present=np.stack([frame[s+'_present'].to_numpy(bool) for s in SOURCES],axis=1)
    valid=present & np.isfinite(ranks) & (ranks>0)
    safe=np.where(valid,ranks,np.nan)
    count=valid.sum(axis=1)
    order=np.sort(np.where(valid,ranks,np.inf),axis=1)
    strength=np.where(valid,1/(60+np.where(valid,ranks,0)),0)
    mean=np.nansum(np.log1p(safe),axis=1)/np.maximum(count,1)
    var=np.nansum((np.log1p(safe)-mean[:,None])**2,axis=1)/np.maximum(count,1)
    return {
        'wv2_source_best_rank':np.where(count>0,order[:,0],np.nan),
        'wv2_source_second_rank':np.where(count>1,order[:,1],np.nan),
        'wv2_source_log_rank_std':np.where(count>0,np.sqrt(var),np.nan),
        'wv2_source_top10_support':(valid & (ranks<=10)).sum(axis=1),
        'wv2_source_top30_support':(valid & (ranks<=30)).sum(axis=1),
        'wv2_source_best_strength_share':strength.max(axis=1)/np.maximum(strength.sum(axis=1),1e-12),
        'wv2_source_first_second_gap':np.subtract(order[:,1],order[:,0],out=np.full(len(frame),np.nan),where=count>1),
        'wv2_repurchase_and_item2vec':frame.repurchase_present.to_numpy()*frame.item2vec_present.to_numpy(),
        'wv2_covisit_and_item2vec':frame.user_day_covisit_present.to_numpy()*frame.item2vec_present.to_numpy(),
        'wv2_source_count_times_i2v':frame.source_count.to_numpy()*frame.item2vec_cosine.to_numpy(),
    }


def attach_features(frame,cutoff,families,engine):
    added={}
    for family in families:
        if family=='source_agreement':
            added.update(source_features(frame))
            continue
        path,meta=build(cutoff,family,engine)
        indexed=_indexed(str(path),tuple(meta['keys']))
        key=frame[meta['keys'][0]] if len(meta['keys'])==1 else pd.MultiIndex.from_frame(frame[meta['keys']])
        aligned=indexed.reindex(key)
        assert len(aligned)==len(frame)
        if family=='repeat_affinity':
            assert not aligned.wv2_user_item_events_7d.isna().any(), 'candidate join dropped identity'
        for name in aligned.columns:
            added[name]=aligned[name].to_numpy(np.float32)
        if family=='user_rhythm':
            for name in ('user_events_7d','user_events_28d','user_items_7d','user_items_28d','user_purchase_days_28d','user_purchase_days_84d','user_event_share_7_84','user_events_per_purchase_day'):
                added['wv2_'+name]=np.nan_to_num(added['wv2_'+name],nan=0)
            med=added['wv2_user_price_median_28d']; std=added['wv2_user_price_std_28d']
            price=frame.item_avg_price_28d.to_numpy(float)
            added['wv2_item_user_price_ratio']=price/(med+1e-6)
            added['wv2_item_user_price_standard_gap']=np.clip((price-med)/(std+1e-6),-10,10)
            added['wv2_user_cycle_phase']=frame.user_days_since_last_purchase.to_numpy(float)/(added['wv2_user_gap_median_84d']+1)
        if family=='item_demand':
            for name in ('item_events_1d','item_events_3d','item_events_14d','item_events_56d','item_users_3d','item_users_7d','item_days_sold_7d','item_days_sold_28d','item_events_all','item_observed_within28'):
                added['wv2_'+name]=np.nan_to_num(added['wv2_'+name],nan=0)
    assert set(added)==set(columns_for(families)),(set(added)-set(columns_for(families)),set(columns_for(families))-set(added))
    extra=pd.DataFrame({k:np.asarray(v,np.float32) for k,v in added.items()},index=frame.index)
    assert not np.isinf(extra.to_numpy()).any()
    return pd.concat([frame,extra],axis=1)
