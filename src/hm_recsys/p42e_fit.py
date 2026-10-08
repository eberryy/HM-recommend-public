"""Same qC objective/features, disk-backed all-row matrices; no sampling."""
from __future__ import annotations

import gc
import time
import warnings
from pathlib import Path

import duckdb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .p42_propensity import MODEL_PARAMS, _design_matrix
from .p41a_contract import write_json
from .p42e_resources import memory, rss


def chunks(paths,fold=None):
    with duckdb.connect() as con:
        con.execute('SET threads=4'); con.execute("SET memory_limit='1GB'")
        for path in paths:
            query='SELECT * FROM read_parquet(?)'+(' WHERE fold=?' if fold is not None else '')
            yield con.execute(query,[str(path),fold] if fold is not None else [str(path)]).fetchdf()


def census(paths,fold=None):
    with duckdb.connect() as con:
        con.execute('SET threads=4'); con.execute("SET memory_limit='2GB'")
        where=' WHERE fold=?' if fold is not None else ''
        q='SELECT count(*) n,sum(target) p,count(DISTINCT customer_id) users,count(DISTINCT CASE WHEN target=1 THEN customer_id END) positive_users FROM read_parquet(?)'+where
        n,p,users,pu=con.execute(q,[[str(p) for p in paths],fold] if fold is not None else [[str(p) for p in paths]]).fetchone()
    return dict(rows=int(n),positives=int(p),users=int(users),positive_users=int(pu))


def preprocess_inplace(x,spec):
    numeric,binary=spec['numeric'],spec['binary']; p,b=len(numeric),len(binary)
    prep=dict(format_version=1,numeric=numeric,binary=binary,
              columns=numeric+binary+[n+'_available' for n in numeric],
              numeric_availability=[n+'_available' for n in numeric],training_rows=len(x),median={},
              missing_training_rows={},all_missing_numeric=[],
              numeric_missing_rule='nonfinite -> training median; all missing -> 0',
              binary_missing_rule='nonfinite -> 0; finite values must be 0 or 1',
              scaling_rule='training-only StandardScaler ddof=0 for numeric; binary and availability unchanged')
    for j,name in enumerate(numeric):
        raw=x[:,j]; valid=np.isfinite(raw)
        med=float(np.median(raw[valid])) if valid.any() else 0.
        prep['median'][name]=med; prep['missing_training_rows'][name]=int((~valid).sum())
        if not valid.any(): prep['all_missing_numeric'].append(name)
        x[:,p+b+j]=valid; x[~valid,j]=med
    for j,name in enumerate(binary):
        v=x[:,p+j]; valid=np.isfinite(v)
        if not np.isin(v[valid],[0,1]).all(): raise ValueError('nonbinary '+name)
        x[~valid,p+j]=0.
    # Keep the ORIGINAL multi-column reduction order. The attempted
    # column-separated scaler failed the 1e-12 real-data equivalence check.
    scaler=StandardScaler(copy=False)
    scaler.fit(x[:,:p]); scaler.transform(x[:,:p],copy=False)
    prep['scaler']=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                        var=scaler.var_.tolist(),n_samples_seen=int(scaler.n_samples_seen_))
    return prep


def fit(paths,spec,folder,fold=None):
    if folder.exists(): raise FileExistsError('refuse repeated fit '+str(folder))
    counts=census(paths,fold); n=counts['rows']; dims=2*len(spec['numeric'])+len(spec['binary'])
    # Physical budget includes matrix plus optimizer row vectors and headroom.
    needed=n*(dims*8+max(48,len(spec['numeric'])*9))+2*2**30
    if memory().available<needed: raise RuntimeError(f'exact full-fit memory guard: need {needed/2**30:.2f}GiB available')
    folder.mkdir(parents=True); started=time.perf_counter()
    write_json(folder/'FIT_START.json',dict(fold=fold,counts=counts,params=MODEL_PARAMS,feature_spec=spec,
                                          paths=[str(p) for p in paths],estimated_physical_bytes=needed))
    x=np.lib.format.open_memmap(folder/'design.float64.npy',mode='w+',dtype=np.float64,shape=(n,dims))
    y=np.lib.format.open_memmap(folder/'labels.int8.npy',mode='w+',dtype=np.int8,shape=(n,))
    cursor=0; fields=spec['numeric']+spec['binary']
    for f in chunks(paths,fold):
        end=cursor+len(f); x[cursor:end,:len(fields)]=f[fields].to_numpy(float); y[cursor:end]=f.target.to_numpy(np.int8); cursor=end
    assert cursor==n and int(y.sum())==counts['positives']
    with threadpool_limits(limits=4):
        prep=preprocess_inplace(x,spec); x.flush(); y.flush()
        # Array interface is contiguous float64; sklearn need not copy design.
        assert x.flags.c_contiguous and np.isfinite(x).all()
        estimator=LogisticRegression(**MODEL_PARAMS)
        fit_start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); estimator.fit(x,y)
    warns=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    model=dict(format_version=1,type='L2 LogisticRegression',params=dict(MODEL_PARAMS),
               classes=estimator.classes_.astype(int).tolist(),coefficient=estimator.coef_[0].tolist(),
               intercept=float(estimator.intercept_[0]),n_iter=estimator.n_iter_.astype(int).tolist())
    result=dict(model=model,preprocessing=prep,audit=dict(**counts,fitted_rows=n,fold=fold,
                fit_seconds=time.perf_counter()-fit_start,total_seconds=time.perf_counter()-started,
                process_peak_working_set_bytes=rss(),design_matrix_bytes=x.nbytes,
                status='non_converged' if warns else 'converged',convergence_warnings=warns,
                negative_sampling=False,class_weight=None,oversampling=False))
    write_json(folder/'model.json',model); write_json(folder/'preprocessing.json',prep); write_json(folder/'FIT_RESULT.json',result)
    del x,y,estimator; gc.collect()
    if warns or not np.isfinite(model['coefficient']).all(): raise RuntimeError('raw qC solver failed; no retry or parameter change')
    return result


def logits(fitted,frame):
    with threadpool_limits(limits=4):
        x,_=_design_matrix(frame,fitted['preprocessing'],None)
        return x@np.asarray(fitted['model']['coefficient'])+fitted['model']['intercept']
