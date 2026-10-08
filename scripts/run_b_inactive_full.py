"""Resume-safe CPU B export. Original assets are read-only; no training/upload."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sql(path):
    return str(path.resolve()).replace("'", "''")


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def merge_submission(db, original, sample, articles, inactive, rankings, output,
                     total=1371980, inactive_count=882427):
    """Validate both populations, replace inactive only, and read back the CSV."""
    require(not output.exists(), 'Never overwrite an export')
    db.execute(f"CREATE TEMP TABLE original AS SELECT row_number() OVER() row_id,* FROM read_csv_auto('{sql(original)}',header=true,all_varchar=true)")
    db.execute(f"CREATE TEMP TABLE sample AS SELECT customer_id FROM read_csv_auto('{sql(sample)}',header=true,all_varchar=true)")
    db.execute(f"CREATE TEMP TABLE catalog AS SELECT article_id FROM read_csv_auto('{sql(articles)}',header=true,all_varchar=true)")
    db.execute(f"CREATE TEMP TABLE inactive AS SELECT DISTINCT customer_id FROM read_parquet('{sql(inactive)}')")
    db.execute('CREATE TEMP TABLE top AS SELECT customer_id,article_id,output_rank FROM read_parquet(?) WHERE output_rank<=12',
               [[str(p) for p in rankings]])
    for relation, expected in [('original', total), ('sample', total), ('inactive', inactive_count)]:
        stats = db.execute(f'SELECT count(*),count(DISTINCT customer_id),count(*) FILTER(WHERE customer_id IS NULL) FROM {relation}').fetchone()
        require(stats == (expected, expected, 0), f'{relation} population invalid: {stats}')
    require(db.execute('''SELECT count(*) FROM original o FULL JOIN sample s USING(customer_id)
        WHERE o.customer_id IS NULL OR s.customer_id IS NULL''').fetchone()[0] == 0, 'Original/sample identities differ')
    require(db.execute('SELECT count(*) FROM inactive ANTI JOIN original USING(customer_id)').fetchone()[0] == 0,
            'Inactive user outside submission')
    db.execute('''CREATE TEMP TABLE top_groups AS SELECT customer_id,count(*) n,
        count(DISTINCT article_id) ni,count(DISTINCT output_rank) nr,min(output_rank) lo,max(output_rank) hi
        FROM top GROUP BY customer_id''')
    require(db.execute('SELECT count(*) FROM top_groups').fetchone()[0] == inactive_count, 'Incomplete batch population')
    require(db.execute('SELECT count(*) FROM top_groups WHERE n<>12 OR ni<>12 OR nr<>12 OR lo<>1 OR hi<>12').fetchone()[0] == 0,
            'Invalid or duplicated Top12 batch rows')
    require(db.execute('''SELECT count(*) FROM top_groups t FULL JOIN inactive i USING(customer_id)
        WHERE t.customer_id IS NULL OR i.customer_id IS NULL''').fetchone()[0] == 0, 'Batch user membership differs')
    db.execute("CREATE TEMP TABLE replacement AS SELECT customer_id,string_agg(article_id,' ' ORDER BY output_rank) prediction FROM top GROUP BY customer_id")
    db.execute('''CREATE TEMP TABLE final AS SELECT o.row_id,o.customer_id,
        CASE WHEN i.customer_id IS NOT NULL THEN r.prediction ELSE o.prediction END prediction
        FROM original o LEFT JOIN inactive i USING(customer_id) LEFT JOIN replacement r USING(customer_id)''')
    db.execute(f"COPY (SELECT customer_id,prediction FROM final ORDER BY row_id) TO '{sql(output)}' (HEADER,DELIMITER ',')")
    db.execute(f"CREATE TEMP TABLE exported AS SELECT * FROM read_csv_auto('{sql(output)}',header=true,all_varchar=true)")
    require(db.execute('SELECT count(*),count(DISTINCT customer_id) FROM exported').fetchone() == (total, total), 'Export population mismatch')
    require(db.execute('''SELECT count(*) FROM exported e FULL JOIN final f USING(customer_id)
        WHERE e.prediction IS DISTINCT FROM f.prediction OR e.customer_id IS NULL OR f.customer_id IS NULL''').fetchone()[0] == 0,
            'Export readback mismatch')
    active_changes = db.execute('''SELECT count(*) FROM exported e JOIN original o USING(customer_id)
        ANTI JOIN inactive i USING(customer_id) WHERE e.prediction IS DISTINCT FROM o.prediction''').fetchone()[0]
    require(active_changes == 0, 'Active recommendations changed')
    require(db.execute("SELECT count(*) FROM exported WHERE prediction IS NULL OR array_length(string_split(prediction,' '))<>12").fetchone()[0] == 0,
            'Missing/invalid prediction list')
    db.execute("CREATE TEMP TABLE tokens AS SELECT customer_id,unnest(string_split(prediction,' ')) article_id FROM exported")
    require(db.execute('SELECT count(*) FROM (SELECT customer_id,count(DISTINCT article_id) n FROM tokens GROUP BY customer_id) WHERE n<>12').fetchone()[0] == 0,
            'Duplicate recommended article')
    require(db.execute("SELECT count(*) FROM tokens WHERE article_id IS NULL OR article_id !~ '^[0-9]{10}$'").fetchone()[0] == 0,
            'Article formatting invalid')
    require(db.execute('SELECT count(*) FROM tokens ANTI JOIN catalog USING(article_id)').fetchone()[0] == 0, 'Article outside catalog')
    changed = db.execute('''SELECT count(*) FROM exported e JOIN original o USING(customer_id)
        SEMI JOIN inactive i USING(customer_id) WHERE e.prediction<>o.prediction''').fetchone()[0]
    return {'rows': total, 'inactive_users': inactive_count, 'active_users': total-inactive_count,
            'active_changed_users': int(active_changes), 'inactive_changed_users': int(changed),
            'each_user_12_distinct_catalog_articles': True, 'export_readback_passed': True}


def check_batch(db, directory, offset, users, old_inactive):
    receipt = read(directory / 'RESULT.json')
    require(receipt['status'] == 'completed' and receipt['users'] == users
            and receipt['candidate_rows'] == users*100, 'Batch receipt incomplete')
    c = receipt['contract']
    require(c['stage'] == 'B-INACTIVE-CPU-BATCH-v1' and c['offset'] == offset
            and c['cutoff'] == '2020-09-23', 'Batch contract mismatch')
    require(receipt['original_top12_reconstruction_mismatches'] == 0
            and receipt['active_score_max_abs_error'] <= 1e-10
            and receipt['independent_rank_replay'] == 'passed', 'Batch correctness gate failed')
    path = directory / 'pilot-ranking.parquet'
    db.execute(f"CREATE OR REPLACE TEMP VIEW batch_rows AS SELECT * FROM read_parquet('{sql(path)}')")
    stats = db.execute('''SELECT count(*),sum(n),count(*) FILTER(WHERE n<>100 OR ni<>100 OR nr<>100 OR lo<>1 OR hi<>100)
        FROM (SELECT customer_id,count(*) n,count(DISTINCT article_id) ni,count(DISTINCT output_rank) nr,
        min(output_rank) lo,max(output_rank) hi FROM batch_rows GROUP BY customer_id)''').fetchone()
    require(stats == (users, users*100, 0), 'Completed batch artifact invalid')
    expected = f"SELECT DISTINCT customer_id FROM read_parquet('{sql(old_inactive)}') ORDER BY customer_id LIMIT {users} OFFSET {offset}"
    mismatch = db.execute(f'''WITH expected AS ({expected}), actual AS (SELECT DISTINCT customer_id FROM batch_rows)
        SELECT count(*) FROM expected e FULL JOIN actual a USING(customer_id)
        WHERE e.customer_id IS NULL OR a.customer_id IS NULL''').fetchone()[0]
    require(mismatch == 0, 'Batch covers the wrong users')
    return path


def run(args):
    import duckdb
    import fcntl
    import shutil
    bundle, old, raw, out = [p.resolve() for p in [args.bundle_dir, args.original_run, args.data_dir, args.output_dir]]
    for protected in [bundle, old, raw]:
        require(not out.is_relative_to(protected) and not protected.is_relative_to(out), 'Output overlaps original assets')
    require(1 <= args.batch_users <= 8192, 'Invalid batch size')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(shutil.disk_usage(out).free >= 30 * 2**30, 'Less than 30 GiB free; stop')
    worker = Path(__file__).with_name('run_b_inactive_pilot.py')
    # Resume fingerprints address mutable historical input/code identity, not new-file completion.
    inputs = [old / 'RUN_MANIFEST.json', old / 'submission_wv3_741.csv',
              old / 'submission_wv3_741.json', old / 'work/input/transactions.parquet',
              old / 'artifacts/m1-active/inactive-top12.parquet',
              old / 'artifacts/base-features.parquet', old / 'artifacts/ranks-top50.parquet']
    inputs += list((old / 'artifacts/bpr').glob('*'))
    inputs += [p for p in (old / 'artifacts/item2vec-source').rglob('*') if p.is_file()]
    inputs += [raw / name for name in ['articles.csv', 'customers.csv', 'sample_submission.csv']]
    inputs += [p for p in bundle.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    snapshot = {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(inputs) if p.is_file()}
    contract = {'stage': 'B-INACTIVE-FULL-v1', 'cutoff': '2020-09-23', 'inactive_users': 882427,
                'batch_users': args.batch_users, 'threads': 8, 'training': False, 'upload': False,
                'input_snapshot': snapshot, 'worker_sha256': digest(worker),
                'orchestrator_sha256': digest(Path(__file__)),
                'paths': {'bundle': str(bundle), 'original_run': str(old), 'raw': str(raw)}}
    cp = out / 'CONTRACT.json'
    if cp.exists():
        require(read(cp) == contract, 'Inputs/code/contract changed since run started; do not mix batch versions')
    else:
        require(not list(out.glob('batch-*')), 'Unbound batches already exist')
        write(cp, contract)
    original = old / 'submission_wv3_741.csv'
    require(digest(original) == read(old / 'submission_wv3_741.json')['sha256'], 'Original CSV differs from trusted completed receipt')
    result_path = out / 'RESULT.json'
    if result_path.exists():
        print(json.dumps(read(result_path), indent=2)); return
    began = time.perf_counter()
    old_inactive = old / 'artifacts/m1-active/inactive-top12.parquet'
    ranking_paths = []
    with duckdb.connect() as db:
        db.execute('SET threads=8'); db.execute("SET memory_limit='32GB'")
        spill = out / 'merge-spill'; spill.mkdir(exist_ok=True)
        db.execute(f"SET temp_directory='{sql(spill)}'")
        for offset in range(0, 882427, args.batch_users):
            users = min(args.batch_users, 882427-offset)
            batch = out / f'batch-{offset:06d}'; batch.mkdir(exist_ok=True)
            done = batch / 'COMPLETED.json'
            if done.exists():
                directory = batch / read(done)['attempt']
                require(directory.resolve().parent == batch.resolve(), 'Unsafe completed-attempt path')
                ranking_paths.append(check_batch(db, directory, offset, users, old_inactive))
                print(f'Reused {offset+users}/882427', flush=True)
                continue
            # Recover an already completed worker if interrupted before parent checkpoint.
            completed_attempts = sorted(p.parent for p in batch.glob('attempt-*/RESULT.json'))
            if completed_attempts:
                directory = completed_attempts[-1]
            else:
                directory = batch / ('attempt-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                command = [sys.executable, '-B', str(worker), '--batch-mode', '--offset', str(offset),
                           '--users', str(users), '--threads', '8', '--bundle-dir', str(bundle),
                           '--original-run', str(old), '--data-dir', str(raw), '--output-dir', str(directory)]
                print(f'Start {offset}:{offset+users} /882427', flush=True)
                with (batch / (directory.name + '.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            ranking_paths.append(check_batch(db, directory, offset, users, old_inactive))
            write(done, {'attempt': directory.name, 'offset': offset, 'users': users})
            receipt = read(directory / 'RESULT.json')
            print(f"Completed {offset+users}/882427; batch {receipt['total_seconds']:.1f}s", flush=True)
        # Fresh export attempt preserves any interrupted partial CSV for inspection.
        export = out / ('export-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        export.mkdir()
        csv = export / 'submission_wv3_741_B_no_fallback.csv'
        stats = merge_submission(db, original, raw/'sample_submission.csv', raw/'articles.csv',
                                 old_inactive, ranking_paths, csv)
    import zipfile
    zipped = csv.with_suffix('.zip')
    with zipfile.ZipFile(zipped, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(csv, arcname=csv.name)
    with zipfile.ZipFile(zipped) as archive:
        require(archive.testzip() is None and archive.getinfo(csv.name).file_size == csv.stat().st_size,
                'ZIP integrity mismatch')
    result = {'status': 'completed', 'stage': contract['stage'], 'cutoff': contract['cutoff'],
              **stats, 'batches': len(ranking_paths), 'current_invocation_seconds': time.perf_counter()-began,
              'csv': str(csv), 'zip': str(zipped), 'csv_bytes': csv.stat().st_size,
              'labels_read': False, 'MAP_evaluated': False, 'models_retrained': False, 'uploaded': False}
    write(result_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['bundle-dir', 'original-run', 'data-dir', 'output-dir']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batch-users', type=int, default=8192)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        print(f'STOPPED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise
