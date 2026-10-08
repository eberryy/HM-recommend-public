"""Second registered feature batch: category competition, prices, fine hierarchy."""
from __future__ import annotations

import time

import numpy as np

from .warm_v2_contract import ARTIFACT, REPORT, evidence_id, guard, now, read, write
from .warm_v2_engine import connection, literal, save_parquet


EXTRA_SPEC={
    'category_competition':{
        **{f'{level}_{name}':description for level,label in [('type','商品类别'),('department','部门'),('garment','服装组')]
            for name,description in [
                ('item_share_7d',f'商品7天事件数占同{label}全部商品7天事件数；分母至少1'),
                ('item_share_28d',f'商品28天事件数占同{label}全部商品28天事件数；分母至少1'),
                ('item_rank_fraction_7d',f'商品7天销量在同{label}全目录中降序并列名次减1，除以该组目录商品数减1；分母至少1，越小越热门'),
                ('group_recent_growth',f'同{label}全体商品近7天平滑日事件率，除以前21天平滑日事件率；两段各加1次事件'),
                ('item_vs_group_growth',f'商品平滑7天与此前21天日事件率比，除以同{label}对应增长比；度量相对品类趋势')]
        }},
    'price_context':{
        'price_item_median_7d':'商品近7天价格中位数；无交易缺失，沿用原始价格单位',
        'price_item_median_28d':'商品近28天价格中位数',
        'price_item_std_28d':'商品近28天价格总体标准差',
        'price_item_min_28d':'商品近28天观察到的最低交易价格，不是真实标价',
        'price_item_max_28d':'商品近28天观察到的最高交易价格',
        'price_item_recent_ratio':'商品7天价格中位数除以28天中位数+1e-6；不能直接解释为促销因果效应',
        'price_item_coefficient_variation':'商品28天价格标准差除以中位数+1e-6',
        'price_user_type_events_28d':'用户近28天购买候选同类别商品的事件数；无匹配为0',
        'price_user_type_median_28d':'用户近28天候选同类别商品购买价格中位数；无匹配缺失',
        'price_user_type_std_28d':'用户近28天候选同类别购买价格总体标准差',
        'price_item_user_type_ratio':'候选商品28天价格中位数除以用户同类别28天价格中位数+1e-6',
        'price_item_user_type_standard_gap':'两种中位数之差除以用户同类别28天价格标准差+1e-6，截断[-10,10]',
    },
    'fine_hierarchy':{
        **{f'fine_{level}_{name}':description for level,label in [('section','section_no销售分区'),('colour_group','colour_group_code细颜色组'),('appearance','graphical_appearance_no图案'),('index','index_code产品系列')]
            for name,description in [
                ('events_28d',f'用户近28天购买候选同{label}商品的事件数'),
                ('events_84d',f'用户近84天购买候选同{label}商品的事件数'),
                ('share_28d',f'用户近28天同{label}事件数除以该用户28天全部事件数；分母至少1'),
                ('days_since',f'用户84天内最后购买同{label}距截止日的天数；无匹配缺失')]
        }},
}


def register_extra_spec():
    path=REPORT/'WARM_FEATURE_SPEC_BATCH2.json'
    if not path.exists():
        write(path,{'registered_at':now(),'families':EXTRA_SPEC,'classification':'All wv2 names are project-defined; descriptions in Chinese define objects and denominators.',
            'PIT':'Transactions strictly before cutoff; catalog same optimistic snapshot as frozen baseline; no categorical window router.',
            'protocol':'Each family tested independently using original four historical inner cutoffs; no new outer result required for feature construction.',
            'cost_estimate':'3 families x 4 historical fits, CPU 8 threads, approximately 10-20 minutes; new caches below 1 GiB target.',
            'gate':'Unchanged WARM_V2_EXPERIMENT_CONTRACT historical and outer gates; nonpassing families not forced into combinations.',
            'fallback':'Keep frozen champion; preserve failures; no change to candidates, sampling, objective or Cold.',
            'final_week':'not_run'})


def dimensions(con,cutoff,engine):
    path=engine.history['prerequisite_cache']['target_features'][cutoff]['inputs']['articles']['path']
    con.execute(f"""CREATE TEMP TABLE dim AS SELECT article_id,product_type_no AS type,
        department_no AS department,garment_group_no AS garment,section_no AS section,
        colour_group_code AS colour_group,graphical_appearance_no AS appearance,index_code AS index
        FROM read_csv({literal(path)},all_varchar=true)""")


def category(con,c):
    con.execute(f"""CREATE TEMP TABLE item_count AS SELECT article_id,count(*)::DOUBLE e28,
        count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 7 DAY)::DOUBLE e7
        FROM h84 WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY GROUP BY article_id""")
    con.execute('CREATE TEMP TABLE counts AS SELECT d.*,coalesce(e7,0) e7,coalesce(e28,0) e28 FROM dim d LEFT JOIN item_count USING(article_id)')
    aggs=[]
    fields=[]
    for level in ('type','department','garment'):
        aggs += [f'sum(e7) OVER(PARTITION BY {level}) s7_{level}',f'sum(e28) OVER(PARTITION BY {level}) s28_{level}',
            f'(rank() OVER(PARTITION BY {level} ORDER BY e7 DESC)-1)::DOUBLE/greatest(count(*) OVER(PARTITION BY {level})-1,1) rank_{level}']
        growth=f'((s7_{level}+1)/7/((s28_{level}-s7_{level}+1)/21))'
        fields += [f'e7/greatest(s7_{level},1) AS wv2_{level}_item_share_7d',f'e28/greatest(s28_{level},1) AS wv2_{level}_item_share_28d',
            f'rank_{level} AS wv2_{level}_item_rank_fraction_7d',f'{growth} AS wv2_{level}_group_recent_growth',
            f'((e7+1)/7/((e28-e7+1)/21))/{growth} AS wv2_{level}_item_vs_group_growth']
    return con.execute(f"WITH stats AS (SELECT *,{','.join(aggs)} FROM counts) SELECT article_id,{','.join(fields)} FROM stats").fetchdf()


def price(con,c):
    con.execute(f"""CREATE TEMP TABLE ip AS SELECT article_id,median(price) m28,stddev_pop(price) s28,min(price) low28,max(price) high28,
        median(price) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 7 DAY) m7 FROM h84
        WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY GROUP BY article_id""")
    con.execute(f"""CREATE TEMP TABLE up AS SELECT customer_id,type,count(*)::DOUBLE n,median(price) m,stddev_pop(price) s
        FROM uh JOIN dim USING(article_id) WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY GROUP BY customer_id,type""")
    return con.execute("""SELECT b.customer_id,b.article_id,m7 AS wv2_price_item_median_7d,m28 AS wv2_price_item_median_28d,
        s28 AS wv2_price_item_std_28d,low28 AS wv2_price_item_min_28d,high28 AS wv2_price_item_max_28d,
        m7/(m28+1e-6) AS wv2_price_item_recent_ratio,s28/(m28+1e-6) AS wv2_price_item_coefficient_variation,
        coalesce(up.n,0) AS wv2_price_user_type_events_28d,up.m AS wv2_price_user_type_median_28d,up.s AS wv2_price_user_type_std_28d,
        m28/(up.m+1e-6) AS wv2_price_item_user_type_ratio,
        CASE WHEN up.m IS NULL OR m28 IS NULL THEN NULL ELSE greatest(-10,least(10,(m28-up.m)/(up.s+1e-6))) END AS wv2_price_item_user_type_standard_gap
        FROM base b LEFT JOIN dim d USING(article_id) LEFT JOIN ip USING(article_id)
        LEFT JOIN up ON b.customer_id=up.customer_id AND d.type=up.type
        ORDER BY b.customer_id,b.candidate_rank,b.article_id""").fetchdf()


def hierarchy(con,c):
    con.execute('CREATE TEMP TABLE hd AS SELECT h.*,d.* EXCLUDE(article_id) FROM uh h JOIN dim d USING(article_id)')
    con.execute(f"CREATE TEMP TABLE totals AS SELECT customer_id,count(*)::DOUBLE n28 FROM uh WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY GROUP BY customer_id")
    fields=[]; joins=[]
    for level in ('section','colour_group','appearance','index'):
        con.execute(f"""CREATE TEMP TABLE a_{level} AS SELECT customer_id,{level},count(*)::DOUBLE n84,
            count(*) FILTER(WHERE t_dat>=DATE '{c}'-INTERVAL 28 DAY)::DOUBLE n28,
            date_diff('day',max(t_dat),DATE '{c}')::DOUBLE recency FROM hd GROUP BY customer_id,{level}""")
        joins.append(f'LEFT JOIN a_{level} a{level} ON b.customer_id=a{level}.customer_id AND d.{level}=a{level}.{level}')
        fields += [f'coalesce(a{level}.n28,0) AS wv2_fine_{level}_events_28d',f'coalesce(a{level}.n84,0) AS wv2_fine_{level}_events_84d',
            f'coalesce(a{level}.n28,0)/greatest(coalesce(t.n28,0),1) AS wv2_fine_{level}_share_28d',f'a{level}.recency AS wv2_fine_{level}_days_since']
    return con.execute(f"""SELECT b.customer_id,b.article_id,{','.join(fields)} FROM base b LEFT JOIN dim d USING(article_id)
        {' '.join(joins)} LEFT JOIN totals t ON b.customer_id=t.customer_id
        ORDER BY b.customer_id,b.candidate_rank,b.article_id""").fetchdf()


def build_extra(cutoff,family,engine):
    from .warm_v2_features import _base_views
    guard(cutoff)
    register_extra_spec()
    root=ARTIFACT/'features-v2'/cutoff
    path=root/f'{family}.parquet';meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        return path,read(meta)
    start=time.perf_counter()
    with connection() as con:
        _base_views(con,cutoff,engine)
        dimensions(con,cutoff,engine)
        frame={'category_competition':category,'price_context':price,'fine_hierarchy':hierarchy}[family](con,cutoff)
        latest=str(con.execute('SELECT max(t_dat) FROM history').fetchone()[0])
        assert latest<cutoff
    keys=['article_id'] if family=='category_competition' else ['customer_id','article_id']
    assert not frame.duplicated(keys).any()
    numeric=[v for v in frame.columns if v.startswith('wv2_')]
    assert set(numeric)=={'wv2_'+v for v in EXTRA_SPEC[family]}
    frame[numeric]=frame[numeric].astype(np.float32)
    assert not np.isinf(frame[numeric].to_numpy()).any()
    save_parquet(frame,path)
    result={'cutoff':cutoff,'family':family,'keys':keys,'rows':len(frame),'features':numeric,
        'latest_history_date':latest,'pit_safe':True,'future_labels_used':False,'runtime_seconds':time.perf_counter()-start,
        'artifact':evidence_id(path,reason='explicit_registry_evidence')}
    write(meta,result)
    print(f'Warm batch2 cache {cutoff} {family}: {len(frame)} rows, {result["runtime_seconds"]:.1f}s',flush=True)
    return path,result


def main():
    from .warm_v2_lab import screen
    register_extra_spec()
    for trial,family,hypothesis in [
        ('WV2-107','category_competition','绝对商品销量没有显式区分类别整体上涨与类内商品竞争；检验同类别、部门、服装组的份额和相对增长。'),
        ('WV2-108','price_context','整体用户均价混合了不同品类价格带；检验同类别个人价格匹配和商品近期实际交易价分布。'),
        ('WV2-109','fine_hierarchy','现有主色与品类偏好未覆盖细颜色、图案、销售分区和产品系列；检验新的层级偏好。')]:
        screen(trial,[family],hypothesis)


if __name__=='__main__':
    main()
