"""P4.3A report/evidence closeout only; no experiment execution or promotion.

Call only after the worker has stopped and TOURNAMENT_STATE is non-running.
Generated checksums establish a new closeout baseline, except the contract's
explicit comparison to its pre-computation identity. Large model inputs/arrays
receive structural metadata, not expensive unauthenticated rehashing.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .p41a_contract import identity, read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def _array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        # NumPy's internal reader supports v3 UTF-8 descriptors without loading
        # any array body. This is metadata inspection, not data deserialization.
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
    return dict(shape=list(shape), dtype=str(dtype), fortran_order=bool(fortran))


def structural_metadata(path):
    path = Path(path)
    result = dict(path=str(path.resolve()), exists=True, bytes=path.stat().st_size,
                  mtime_ns=path.stat().st_mtime_ns, sha256=None,
                  check='existence_size_structure_only_not_content_hash')
    if path.suffix.lower() == '.npy':
        with path.open('rb') as stream:
            result.update(_array_header(stream))
    elif path.suffix.lower() == '.npz':
        members = []
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = dict(name=info.filename, compressed_bytes=info.compress_size,
                              uncompressed_bytes=info.file_size)
                if info.filename.endswith('.npy'):
                    with archive.open(info) as stream:
                        member.update(_array_header(stream))
                members.append(member)
        result['members'] = members
    else:
        result['check'] = 'existence_size_only_not_deserialized_or_content_hashed'
    return result


def receipt_paths(root):
    """All in-scope JSON receipts, excluding archived evidence and prior closes."""
    result = []
    excluded = {'closure-history', 'source-archive', 'verification-attempts'}
    for path in Path(root).rglob('*.json'):
        relative = path.relative_to(root)
        if excluded.intersection(relative.parts):
            continue
        result.append(path)
    return sorted(result, key=lambda p: str(p).lower())


def _public_reports(repo):
    folder = Path(repo)/'reports/phase4'
    return sorted((p for p in folder.iterdir() if p.is_file() and
                   p.name.lower().startswith('p4_3a') and
                   p.name != 'P4_3A_OUTPUT_MANIFEST.json'), key=lambda p: p.name.lower())


def _source_paths(repo):
    return sorted([*Path(repo).glob('src/hm_recsys/p43a*.py'),
                   *Path(repo).glob('tests/test_p43a*.py')], key=lambda p: str(p).lower())


def _artifact_paths(root):
    suffixes = {'.npy', '.npz', '.joblib', '.txt', '.parquet'}
    excluded = {'closure-history', 'source-archive', 'verification-attempts'}
    return sorted((p for p in Path(root).rglob('*') if p.is_file() and
                   p.suffix.lower() in suffixes and
                   not excluded.intersection(p.relative_to(root).parts)),
                  key=lambda p: str(p).lower())


def _archive_existing(root, paths, close_id):
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return []
    target = Path(root)/'closure-history'/close_id
    target.mkdir(parents=True, exist_ok=False)
    result = []
    for path in existing:
        destination = target/path.name
        shutil.copy2(path, destination)
        # Continuous local copy; a second checksum would add no new evidence.
        assert destination.stat().st_size == path.stat().st_size
        result.append(dict(original=str(path), archived=str(destination),
                           bytes=destination.stat().st_size,
                           check='direct_local_copy_and_size_not_independent_hash_verification'))
    return result


def close(repo, root):
    """Refresh public reports + manifest after a real worker pause/completion.

    Does not change the tournament state, deadline, historical authorities,
    Git, selection policy, models, or training artifacts.
    """
    from .p43a_report import write_report

    repo, root = Path(repo).resolve(), Path(root).resolve()
    state_path = root/'TOURNAMENT_STATE.json'
    state_bytes = state_path.read_bytes()
    state = read_json(state_path)
    status = state.get('status')
    if not status or status in ('running', 'starting', 'in_progress'):
        raise RuntimeError('P4.3A close requires a stopped, non-running tournament state')
    if state.get('final_week') != 'not_run':
        raise ValueError('Final week must remain not_run')
    report_folder = repo/'reports/phase4'
    contract_path = report_folder/'P4_3A_EXPERIMENT_CONTRACT.json'
    contract = read_json(contract_path)
    contract_id = identity(contract_path)
    expected = read_json(root/'EXECUTION_START.json')['contract']
    if any(contract_id[key] != expected[key] for key in ('bytes', 'sha256')):
        raise ValueError('Registered contract changed; close is not authorized to repair it')
    if contract.get('final_week') != 'not_run':
        raise ValueError('Registered final-week boundary drift')
    if contract.get('sealed_confirmatory_holdout') != '2020-09-16':
        raise ValueError('Unknown sealed holdout boundary')
    from .p43a_session import execution_budget
    budget=execution_budget(contract,state)
    deadline = budget['deadline_epoch']
    verify_path = report_folder/'P4_3A_VERIFICATION.json'
    verification_bytes = verify_path.read_bytes() if verify_path.exists() else None
    inherited = read_json(verify_path) if verification_bytes is not None else None
    close_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+str(time.time_ns())
    manifest_path = report_folder/'P4_3A_OUTPUT_MANIFEST.json'
    closure_path = root/'CLOSURE.json'
    archived = _archive_existing(root, [manifest_path, closure_path], close_id)
    start = time.perf_counter()
    started_at = now()
    summary = write_report(repo, root)
    report_seconds = time.perf_counter()-start
    if state_path.read_bytes() != state_bytes:
        raise RuntimeError('Tournament state changed during close; worker was not settled')
    if verification_bytes is not None and verify_path.read_bytes() != verification_bytes:
        raise RuntimeError('Report generation changed independent verification evidence')
    closure = dict(stage='P4.3A', close_id=close_id, status=status,
                   kind='report_and_evidence_closeout_only', started_at=started_at,
                   report_finished_at=now(), report_seconds=report_seconds,
                   paused_at=state.get('paused_at'),
                   actual_pause_time_source='TOURNAMENT_STATE.paused_at; null means not recorded, not inferred',
                   immutable_deadline_epoch=deadline,
                   immutable_deadline_utc=budget.get('deadline_utc'),
                   execution_session=state.get('active_session'), original_budget=contract['budget'],
                   contract_identity=contract_id, contract_comparison='pass_against_EXECUTION_START',
                   snapshot_state_status=status, report_status=summary.get('status'),
                   fit_count=0, policy_changes=0, continued_training=False,
                   promotion_executed=False, fullscale_started=False,
                   closure_does_not_imply_tournament_completed=True,
                   independent_verification_preserved=verification_bytes is not None,
                   inherited_verification_status=inherited.get('status') if inherited else 'not_run',
                   previous_closures_archived=archived, final_week='not_run')
    write_json(closure_path, closure)
    public = [identity(p) for p in _public_reports(repo)]
    sources = [identity(p) for p in _source_paths(repo)]
    receipts = [identity(p) for p in receipt_paths(root)]
    metadata = [structural_metadata(p) for p in _artifact_paths(root)]
    if state_path.read_bytes() != state_bytes:
        raise RuntimeError('Tournament state changed during manifest indexing')
    manifest = dict(stage='P4.3A', status=status, close_id=close_id, created_at=now(),
                    final_week='not_run', complete_stage_pass=False,
                    snapshot_state_status=status, paused_at=state.get('paused_at'),
                    immutable_deadline_epoch=deadline,
                    continued_training=False, promotion_executed=False, fullscale_started=False,
                    check_semantics={
                        'new_checksum_baseline': 'Public P4.3A reports, all p43a stage sources/tests, and every selected core JSON receipt. SHA records current bytes for future comparison; creation is NOT retrospective unchanged verification.',
                        'contract_trusted_comparison': 'Current contract bytes/SHA compared to trusted pre-computation EXECUTION_START identity.',
                        'inherited_checks': 'Existing independent verification retained byte-for-byte; its exact completed-snapshot scope is inherited, not enlarged.',
                        'large_artifacts': 'Existence, size and NPY/NPZ dtype/shape/header only. No full-content hash, no claim of numerical correctness from metadata.',
                        'receipt_scope': 'All JSON below this run root except source-archive, verification-attempts, closure-history; includes current partial/failed receipts as recorded, without declaring them completed.',
                        'exclusions': 'No MIND, Warm checkout, older P4.2H outputs, raw dataset/images, Git, or roadmap modifications. Manifest excludes itself to avoid recursive hashing.',
                        'current_source_vs_trained_source': 'Current stage source identities are a new baseline; actual fitted-source identities remain in SOURCE_RUN and trusted source archives, as independently verified.'},
                    public_reports=public, stage_source_and_tests=sources,
                    core_receipts=receipts, artifact_metadata=metadata,
                    counts=dict(public_reports=len(public), stage_source_and_tests=len(sources),
                                core_receipts=len(receipts), artifacts_metadata_only=len(metadata)),
                    inherited_verification={
                        'status': inherited.get('status') if inherited else 'not_run',
                        'attempt_id': inherited.get('attempt_id') if inherited else None,
                        'scope_model_windows': inherited.get('scope_model_windows', []) if inherited else [],
                        'scope_policy_rows': inherited.get('scope_policy_rows') if inherited else None,
                        'complete_stage_pass': False,
                        'note': 'This closeout does not run an additional model/list correctness audit.'},
                    closeout_seconds=time.perf_counter()-start,
                    previous_closures_archived=archived)
    write_json(manifest_path, manifest)
    return manifest
