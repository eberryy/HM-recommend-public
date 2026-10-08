"""Read-only SQL verification of reconstructed Warm lists, no truth access."""
import json
from pathlib import Path
import duckdb
from .final_history_rebuild import ROOT, WROOT, OUT, CONTRACT, read, write, WINDOWS
from .warm_v2_engine import literal


def verify():
    report = {}
    db = duckdb.connect()
    db.execute('SET threads=4')
    for t, spec in read(CONTRACT)['mappings'].items():
        folder = OUT/t
        if not (folder/'RESULT.json').exists():
            raise FileNotFoundError('Incomplete history output: '+t)
        db.execute('CREATE OR REPLACE TEMP VIEW output AS SELECT * FROM read_parquet('+literal(folder/'warm-top12.parquet')+')')
        db.execute('CREATE OR REPLACE TEMP VIEW input AS SELECT customer_id,article_id FROM read_parquet('+literal(spec['base'])+')')
        missing = db.execute('SELECT count(*) FROM (SELECT DISTINCT customer_id FROM input EXCEPT SELECT DISTINCT customer_id FROM output)').fetchone()[0]
        extra = db.execute('SELECT count(*) FROM (SELECT DISTINCT customer_id FROM output EXCEPT SELECT DISTINCT customer_id FROM input)').fetchone()[0]
        bad = db.execute('''SELECT count(*) FROM (SELECT customer_id,count(*) n,count(DISTINCT article_id) k,
          count(DISTINCT ap_rf) ranks,min(ap_rf) lo,max(ap_rf) hi FROM output GROUP BY 1)
          WHERE n<>12 OR k<>12 OR ranks<>12 OR lo<>1 OR hi<>12''').fetchone()[0]
        out_of_pool = db.execute('SELECT count(*) FROM (SELECT customer_id,article_id FROM output EXCEPT SELECT * FROM input)').fetchone()[0]
        db.execute('ATTACH '+literal(folder/'stage.duckdb')+' AS stage (READ_ONLY)')
        protected_changed = db.execute('''SELECT count(*) FROM output o JOIN stage.top50 b USING(customer_id,ap_rf)
          WHERE (o.ap_rf<=7 OR b.user_history_events_12w=0) AND o.article_id<>b.article_id''').fetchone()[0]
        db.execute('DETACH stage')
        parity = None
        for window, cutoff in WINDOWS.items():
            if cutoff != t: continue
            old = WROOT/f'artifacts/warm_v3/WV3-741/{window}/outer/swaps.parquet'
            cols = 'customer_id,challenger_article_id,victim_article_id,victim_rank,challenger_rank,swap_order'
            a,b = str(folder/'swaps.parquet'),str(old)
            parity = db.execute('SELECT count(*) FROM ((SELECT '+cols+' FROM read_parquet(?) EXCEPT SELECT '+cols+' FROM read_parquet(?)) UNION ALL (SELECT '+cols+' FROM read_parquet(?) EXCEPT SELECT '+cols+' FROM read_parquet(?)))',[a,b,b,a]).fetchone()[0]
        passed = not any([missing,extra,bad,out_of_pool,protected_changed]) and parity in [None,0]
        report[t] = dict(passed=passed,missing_users=missing,extra_users=extra,invalid_top12=bad,
            out_of_candidate_pool=out_of_pool,protected_or_inactive_changes=protected_changed,
            frozen_development_swap_difference=parity,output=read(folder/'RESULT.json'))
    result = dict(status='pass' if all(v['passed'] for v in report.values()) else 'fail',windows=report,
        labels_read=False,exact_map_recomputed=False,verification_scope='identity, candidate membership, population, protected/inactive invariants and four frozen development swap sets',
        final_week='not_run',cold_integration='not_started')
    write(ROOT/'reports/final/FINAL_HISTORY_REBUILD_VERIFICATION.json',result)
    print(json.dumps({t:{k:v for k,v in r.items() if k!='output'} for t,r in report.items()}))
    if result['status']!='pass': raise AssertionError('History reconstruction parity failed')


if __name__=='__main__': verify()
