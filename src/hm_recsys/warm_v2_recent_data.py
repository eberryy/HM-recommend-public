"""Build new historical training weeks with unchanged Warm retrieval/features."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
import shutil
import time
import traceback

from .m1 import M1Config, _create_population
from .m15 import _create_candidates_long, _generate_sources, _create_collaborative_fusion, _create_wide_candidates, _wide_evidence
from .m2 import M2Config, RETRIEVAL_FEATURES, _create_static_dimensions, build_point_in_time_dataset, validate_candidate_artifact
from .m29 import M29Window, _build_new_source, build_expanded_candidate_cache, ITEM2VEC_FEATURES, FEATURE_SCHEMA_VERSION
from .m210 import build_target_aware_cache
from .warm_v2_contract import ARTIFACT, REPORT, assert_branch, guard, now, read, write
from .warm_v2_engine import Engine, connection as _connection, literal

ROOT=ARTIFACT/'recent-target-v1'
RAW=Path(__file__).resolve().parents[2] / 'data/raw'


def connection():
    # Different cutoff prefetch processes must never share DuckDB spill filenames.
    con=_connection()
    temp=ROOT/'spill'/str(os.getpid());temp.mkdir(parents=True,exist_ok=True)
    con.execute(f'SET temp_directory={literal(temp.resolve())}')
    return con


def raw_views(con, engine):
    con.execute(f'CREATE VIEW transactions AS SELECT * FROM read_parquet({literal(engine.transactions)})')
    for table in ('articles','customers'):
        con.execute(f'CREATE VIEW {table} AS SELECT * FROM read_csv_auto({literal(RAW/(table+".csv"))},header=true,all_varchar=true)')


def compact_completed(cutoff):
    """Remove only closed, reconstructible SQL scratch databases of a completed build."""
    guard(cutoff)
    result=read(ROOT/cutoff/'BUILD.json');assert result['status']=='completed'
    marker=ROOT/cutoff/'SCRATCH_CLEANUP.json'
    if marker.exists():return read(marker)
    targets=[ROOT/cutoff/'source'/'source-build.duckdb',
        ROOT/'expanded'/cutoff/'candidate-build.duckdb',
        ROOT/'target'/cutoff/'feature-build.duckdb']
    removed=[]
    for target in targets:
        resolved=target.resolve();resolved.relative_to(ROOT.resolve())
        assert resolved.suffix=='.duckdb' and not resolved.with_suffix('.duckdb.wal').exists()
        if resolved.is_file():
            removed.append({'path':str(resolved),'bytes':resolved.stat().st_size})
            resolved.unlink()
    out={'cutoff':cutoff,'created_at':now(),'removed':removed,
        'reason':'Disk budget: only completed construction SQL databases; all source/candidate/feature Parquet, vectors, models and manifests retained. Reconstructible from retained inputs/code.',
        'recovered_bytes':sum(v['bytes'] for v in removed),'final_week':'not_run'}
    write(marker,out)
    return out


def build(cutoff):
    assert_branch();guard(cutoff)
    destination=ROOT/cutoff/'BUILD.json'
    if destination.exists():return read(destination)
    assert shutil.disk_usage('.').free>5*1024**3, 'disk floor'
    start=time.perf_counter();timings={}
    cfg=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json');engine=Engine(cfg)
    old_manifest=read(engine.history['inputs']['baseline_windows']['2019-11-27']['manifest']['path'])
    m1config=M1Config(**{**old_manifest['config'],'cutoff':cutoff})
    root=ROOT/cutoff;root.mkdir(parents=True,exist_ok=True)
    cp=root/'candidate_features.parquet';mp=root/'manifest.json'
    if not mp.exists():
        assert not cp.exists(), 'preserve incomplete build'
        print(f'{cutoff}: six frozen retrieval sources',flush=True)
        with connection() as con:
            raw_views(con,engine)
            population=_create_population(con,cutoff,m1config)
            _generate_sources(con,f"DATE '{cutoff}'",m1config,timings)
            _create_candidates_long(con)
            t=time.perf_counter();_create_collaborative_fusion(con,m1config)
            _create_wide_candidates(con,m1config);timings['fusion_wide']=time.perf_counter()-t
            wide=_wide_evidence(con,m1config)
            con.execute(f'COPY m15_wide_candidates TO {literal(cp)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        write(mp,{'schema_version':'m1.5-wide-v1','run_id':'warm-recent-'+cutoff,
            'builder':'Original m1/m15 source/fusion/wide functions; omitted unused persistent source cache and evaluation reports, not retrieval semantics.',
            'cutoff':cutoff,'sample_rate':.1,'catalog_protocol':m1config.catalog_protocol,
            'sampled_user_fingerprint':population['sampled_user_fingerprint'],
            'config':asdict(m1config),'wide_evidence':wide,
            'artifacts':{'candidate_features':str(cp.resolve()),'candidate_features_bytes':cp.stat().st_size},
            'timings':timings,'final_week':'not_run'})
    else:timings.update(read(mp)['timings'])
    # Original CUDA exact-neighbor implementation and all original Item2Vec parameters.
    window=M29Window(cutoff,cp,mp);source_root=root/'source';sm=source_root/'source-manifest.json'
    if sm.exists():source=read(sm)
    else:
        print(f'{cutoff}: frozen Item2Vec fit/exact neighbors',flush=True)
        source=_build_new_source(cutoff=cutoff,window=window,
            baseline_identity=validate_candidate_artifact(cp,mp,100),
            transactions_path=Path(engine.transactions),transaction_identity=engine.history['inputs']['transactions'],source_dir=source_root)
    timings['item2vec']=source['elapsed_seconds']
    expanded=build_expanded_candidate_cache(windows=[window],sources={cutoff:source},cache_dir=ROOT/'expanded',cutoffs=(cutoff,))
    eroot=ROOT/'expanded'/cutoff;fp=eroot/'features.parquet';fm=eroot/'feature-manifest.json'
    if not fm.exists():
        print(f'{cutoff}: frozen tabular features',flush=True)
        with connection() as con:
            raw_views(con,engine);_create_static_dimensions(con)
            ev=build_point_in_time_dataset(con,eroot/'expanded-candidates.parquet',
                {'cutoff':cutoff,'declared_rows':expanded[cutoff]['audit']['rows']},fp,M2Config(**cfg['config']),
                retrieval_features=RETRIEVAL_FEATURES+ITEM2VEC_FEATURES,candidate_group_range=(100,300))
        ev.update(schema_version=FEATURE_SCHEMA_VERSION,candidate_sha256=expanded[cutoff]['artifact']['sha256'],item2vec_features=ITEM2VEC_FEATURES)
        write(fm,ev)
    t=time.perf_counter()
    target=build_target_aware_cache(raw_dir=RAW,transactions_path=Path(engine.transactions),
        m29_cache_dir=ROOT/'expanded',cache_dir=ROOT/'target',cutoffs=(cutoff,))[cutoff]
    timings['target_features']=time.perf_counter()-t
    with connection() as con:
        names={r[0] for r in con.execute(f'DESCRIBE SELECT * FROM read_parquet({literal(target["artifact"]["path"])})').fetchall()}
    assert set(engine.features)<=names
    result={'cutoff':cutoff,'status':'completed','target':target,'timings':timings,
        'wall_seconds':time.perf_counter()-start,'bytes_under_cutoff':sum(f.stat().st_size for f in root.rglob('*') if f.is_file()),
        'disk_free_gib':shutil.disk_usage('.').free/1024**3,'final_week':'not_run',
        'semantics':'all original six-source functions, original Item2Vec configuration, 100+up-to200, same84 selected downstream; no validation candidate changed'}
    write(destination,result)
    print({k:result[k] for k in ('cutoff','wall_seconds','disk_free_gib')},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cutoff');p.add_argument('--compact-completed',action='store_true');a=p.parse_args()
    try:compact_completed(a.cutoff) if a.compact_completed else build(a.cutoff)
    except Exception:
        write(REPORT/f'FAILURE_RECENT_BUILD_{time.time_ns()}.json',{'traceback':traceback.format_exc(),'cutoff':a.cutoff,'final_week':'not_run'})
        raise
