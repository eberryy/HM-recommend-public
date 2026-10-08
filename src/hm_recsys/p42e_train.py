"""Bounded full + two user-fold qC fits, then frozen outer diagnostics."""
from __future__ import annotations

import gc
import time
import traceback
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit

from .p41a_contract import read_json,write_json,identity,check_identity
from .p42e import context
from .p42e_contract import now
from .p42e_fit import fit,chunks,census,logits
from .p42e_stats import calibrator,monotonic_parity,probability,ranking,tails,readiness,verdicts
from .p42r_data import _parquet,_save_frame


def train_all(repo,resume_memory=False):
    c,root=context(repo); reports=repo/'reports/phase4'
    m=read_json(root/('TRAIN_PROGRESS.json' if resume_memory else 'PROGRESS.json'))
    previous_seconds=0.
    if resume_memory:
        assert m['status']=='engineering_failure' and m.get('failure','').startswith('exact full-fit memory guard:')
        assert len(m['windows'])==3 and m['qC_successful_fits']==9
        assert not (root/'models/late_summer_20200819/full').exists()
        if (root/'MEMORY_RECOVERY_START.json').exists(): raise FileExistsError('only one fresh-process recovery')
        for w,r in m['windows'].items():
            check_identity(r['model']); check_identity(r['preprocessor'])
        write_json(root/'MEMORY_RECOVERY_START.json',dict(at=now(),previous_failure=identity(root/'TRAIN_FAILURE.json'),
            reason='memory released after process exit; no late-summer optimizer/preprocessor was started',
            completed_windows_reused=list(m['windows']),actual_prior_raw_fits=9,prior_prefit_guard_stops=1))
        m['qC_fits']=9; m['prefit_memory_stops']=1
        previous_seconds=m['training_seconds']; m.pop('failure',None); m.pop('traceback',None)
    else:
        if m['status']!='generation_complete_training_not_run': raise ValueError('all8 generations and parity must pass first')
        if (root/'TRAIN_START.json').exists(): raise FileExistsError('never repeat formal fits')
    check_identity(read_json(root/'EXECUTION_START.json')['contract'])
    if not resume_memory: write_json(root/'TRAIN_START.json',dict(at=now(),contract=identity(reports/'P4_2E_EXPERIMENT_CONTRACT.json')))
    started=time.perf_counter(); m['status']='training_running'
    if not resume_memory: m['qC_successful_fits']=0
    def checkpoint():
        write_json(reports/'P4_2E_metrics.json',m)
        write_json(root/'TRAIN_PROGRESS.json',m)
    def time_guard():
        elapsed=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(read_json(root/'EXECUTION_START.json')['at'])).total_seconds()
        if elapsed>7200: raise RuntimeError('formal wall-time exceeded preregistered 2h limit; stop before next fit')
    try:
        for w,t in c['outer_cutoffs'].items():
            if w in m['windows']: continue
            time_guard()
            dates=c['historical_pools'][w]
            assert all((pd.Timestamp(d)+pd.Timedelta(days=7))<pd.Timestamp(t) for d in dates)
            paths=[p for d in dates for p in sorted((root/'prepared'/d).glob('chunk-*/training.parquet'))]
            old_paths=[Path(c['old_training'][d]['features']['qC']['path']) for d in dates]
            scale=dict(old=census(old_paths),full=census(paths),historical_cutoffs=dates,
                       user_unit='distinct customer within pool; rows cutoff-user-item',
                       user_date_observations=sum(m['population'][d]['eligible_users'] for d in dates))
            with duckdb.connect() as con:
                inconsistent=con.execute('SELECT count(*) FROM (SELECT customer_id FROM read_parquet(?) GROUP BY customer_id HAVING count(DISTINCT fold)<>1)',[[str(p) for p in paths]]).fetchone()[0]
            assert inconsistent==0
            folder=root/'models'/w
            m['qC_fits']+=1; checkpoint()
            full=fit(paths,c['feature_spec'],folder/'full')
            m['qC_successful_fits']+=1; checkpoint()
            folded=[]
            for f in (0,1):
                time_guard()
                m['qC_fits']+=1; checkpoint()
                folded.append(fit(paths,c['feature_spec'],folder/f'fold{f}',f)); m['qC_successful_fits']+=1; checkpoint()
            n=scale['full']['rows']; oof_root=root/'oof'/w; oof_root.mkdir(parents=True)
            z=np.lib.format.open_memmap(oof_root/'logits.float64.npy',mode='w+',dtype=np.float64,shape=(n,))
            y=np.lib.format.open_memmap(oof_root/'labels.int8.npy',mode='w+',dtype=np.int8,shape=(n,))
            owner=np.lib.format.open_memmap(oof_root/'fold.uint8.npy',mode='w+',dtype=np.uint8,shape=(n,))
            cursor=0; layout=[]
            for path,f in zip(paths,chunks(paths)):
                end=cursor+len(f); block=np.full(len(f),np.nan); seen=np.zeros(len(f),np.uint8)
                for trainfold in (0,1):
                    mask=f.fold.ne(trainfold).to_numpy()
                    if mask.any(): block[mask]=logits(folded[trainfold],f.loc[mask]); seen[mask]+=1
                assert (seen==1).all() and np.isfinite(block).all()
                z[cursor:end]=block; y[cursor:end]=f.target.to_numpy(np.int8); owner[cursor:end]=f.fold.to_numpy(np.uint8)
                layout.append(dict(path=str(path),start=cursor,end=end)); cursor=end
            assert cursor==n
            z.flush(); y.flush(); owner.flush()
            ca=calibrator(z,y); write_json(folder/'calibrator.json',ca)
            raw=expit(z); cal=expit(ca['a']+ca['b']*z)
            op=monotonic_parity(raw,cal)
            oa=dict(calibrator=ca,parity=op,raw=probability(y,raw,False),calibrated=probability(y,cal,False),
                    fold0=folded[0]['audit'],fold1=folded[1]['audit'],user_disjoint=True,each_row_once=True,
                    no_inconsistent_user_folds=True,rows=n,positives=int(y.sum()))
            write_json(oof_root/'AUDIT.json',oa); write_json(oof_root/'LAYOUT.json',layout)
            del z,y,owner,raw,cal; gc.collect()
            source=repo/'artifacts/phase4'/c['old_run_id']/'outer'/w
            cold=_parquet(source/'qC-predictions.parquet')
            warm=_parquet(source/'qW-predictions.parquet'); eligible=_parquet(source/'eligible_cold.parquet')
            users=warm.customer_id.to_numpy().reshape(-1,12)[:,0]
            truth=_parquet(source/'truth.parquet')
            tc=truth.groupby('customer_id').article_id.nunique().reindex(users).to_numpy(int)
            np.testing.assert_array_equal(cold.iloc[eligible.cold_row_index][['customer_id','article_id']].to_numpy(),eligible[['customer_id','article_id']].to_numpy())
            assert not cold.iloc[eligible.cold_row_index].already_w0.any()
            oz=logits(full,cold); qr=expit(oz); qc=expit(ca['a']+ca['b']*oz)
            outparity=monotonic_parity(qr,qc)
            qroles={'Q_old':cold.propensity.to_numpy(),'Q_full_raw':qr,'Q_full_cal':qc}
            wout=dict(scale=scale,raw_fit=full['audit'],oof=oa,outer_parity=outparity,ranking={},tails={},probability={},readiness={})
            wout['model']=identity(folder/'full'/'model.json')
            wout['preprocessor']=identity(folder/'full'/'preprocessing.json')
            dest=root/'outer'/w; dest.mkdir(parents=True)
            preds=cold[['customer_id','article_id']].copy(); preds['raw_logit']=oz
            for role,q in qroles.items():
                rk,r=ranking(cold,q); wout['ranking'][role]=rk
                wout['tails'][role]=tails(cold,q,r)
                wout['probability'][role]=probability(cold.target.to_numpy(),q)
                ready,u=readiness(cold,warm,eligible,tc,q); wout['readiness'][role]=ready
                if role=='Q_old': np.testing.assert_array_equal(u,np.load(source/'pair-utility.float64.npy',mmap_mode='r'))
                np.save(dest/(role+'-utility.npy'),u)
                preds[role]=q
            assert wout['ranking']['Q_full_raw']==wout['ranking']['Q_full_cal']
            assert len(cold)==sum(x['rows'] for x in wout['probability']['Q_full_raw']['calibration_bins'])
            _save_frame(preds,dest/'predictions.parquet')
            wout['frozen_qW']=identity(source/'qW-predictions.parquet')
            wout['outer_frozen_users']=len(users)
            write_json(dest/'METRICS.json',wout); m['windows'][w]=wout; checkpoint()
            print(f'P4.2E {w} complete: historical positives {scale["old"]["positives"]}->{scale["full"]["positives"]}',flush=True)
            time_guard()
        m['verdicts']=verdicts(m['windows']); m['status']='completed_pending_verification'
    except Exception as exc:
        m.update(status='engineering_failure',failure=str(exc),traceback=traceback.format_exc())
        write_json(root/('TRAIN_FAILURE_resume.json' if resume_memory else 'TRAIN_FAILURE.json'),dict(at=now(),**m)); raise
    finally:
        m['training_seconds']=previous_seconds+time.perf_counter()-started; m['finished_at_utc']=now(); checkpoint()
        for name,key in [('qc_training_scale','scale'),('qc_ranking','ranking'),('qc_extreme_tail','tails'),
                         ('oof_calibrator','oof'),('probability_calibration','probability'),('cross_source_readiness','readiness')]:
            write_json(reports/f'p4_2e_{name}.json',{w:r[key] for w,r in m['windows'].items()})
