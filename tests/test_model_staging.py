"""Real checkpoint copies and the exact generated server-path handoff."""
# Execute only repository-generated, hash-bound launcher code in these fixtures.
# ruff: noqa: S102
from __future__ import annotations

import ast
import copy
import errno
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import model_staging
from scripts.extract_llm_launcher import extract_launcher
from scripts.llm_assets import checkpoint_report
from scripts.llm_daemon import LLMDaemon
from scripts.model_staging import stage_checkpoints


@pytest.fixture
def local_storage(monkeypatch):
    # The executor's pytest /tmp is RAM-backed. Simulate disk classification
    # for tiny fixtures only; allocations, fsyncs and copies are still real.
    original = model_staging._storage_profile
    def profile(root):
        return dict(original(root), filesystem='fixture-local-disk')
    monkeypatch.setattr(model_staging, '_storage_profile', profile)


@pytest.fixture
def checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv('ARC3_MODEL_OWNER', 'fixture-owner')
    sources = []
    for name, count in [('target', 38), ('draft', 2)]:
        root = tmp_path / 'input' / name
        root.mkdir(parents=True)
        weights = {}
        for i in range(count):
            file = f'model-{i:02}.safetensors'
            (root / file).write_bytes(f'synthetic shard {name} {i}'.encode())
            weights[f'mtp.layers.0.mlp.experts.{i}.gate_proj.weight'] = file
        (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
        config = ({'quant_method': 'auto-round', 'bits': 4} if name == 'target' else
                  {'quant_method': 'compressed-tensors', 'config_groups': {'mtp_routed_experts': {
                      'targets': ['RoutedExperts'], 'weights': {'num_bits': 4, 'group_size': 32, 'symmetric': True}}}})
        (root / 'config.json').write_text(json.dumps({'quantization_config': config}))
        (root / 'tokenizer.json').write_text('{"synthetic": true}')
        (root / 'chat_template.jinja').write_text('synthetic template')
        sources.append(root)
    return sources


def test_all_38_shards_and_draft_are_real_local_files(checkpoints, tmp_path):
    target, draft = checkpoints
    (target / 'alias.safetensors').symlink_to(target / 'model-00.safetensors')
    report = stage_checkpoints(target, draft, tmp_path / 'local', tmp_path / 'report.json')
    assert report['status'] == 'complete' and report['models']['target']['shard_count'] == 38
    assert report['models']['draft']['shard_count'] == 2
    for kind, source in [('target', target), ('draft', draft)]:
        local = Path(report['models'][kind]['path'])
        manifest = json.loads(Path(report['models'][kind]['manifest']).read_text())
        for file in manifest:
            copied = local / file['name']
            assert not copied.is_symlink()
            assert copied.read_bytes() == (source / file['name']).read_bytes()
            assert hashlib.sha256(copied.read_bytes()).hexdigest() == file['source_stream_sha256']
        original, staged = checkpoint_report(source), checkpoint_report(local)
        assert original['shard_sizes'] == staged['shard_sizes']
        assert original['index_sha256'] == staged['index_sha256']
    assert json.loads((tmp_path / 'report.json').read_text()) == report


def binding(source):
    nodes = ast.parse(source).body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.If)
                 and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                         and child.func.id == 'stage_checkpoints' for child in ast.walk(node)))
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'LOCAL_MODEL_DIR' for target in node.targets))
    return compile(ast.Module(body=nodes[start:end+1], type_ignores=[]), '<actual-staging-handoff>', 'exec')


def environment(checkpoints, tmp_path):
    return {'Path': Path, 'os': os, 'stage_checkpoints': stage_checkpoints, 'checkpoint_report': checkpoint_report,
            'asset_report': {'target': checkpoint_report(checkpoints[0]), 'draft': checkpoint_report(checkpoints[1])},
            'WORKING_DIR': tmp_path / 'working', 'MODEL_DIR': str(checkpoints[0]),
            'DRAFT_MODEL_DIR': str(checkpoints[1])}


@pytest.mark.parametrize('fail_preferred', [False, True])
def test_generated_model_and_draft_launch_paths_are_local(checkpoints, tmp_path, monkeypatch, local_storage, fail_preferred):
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '1')
    primary, fallback = tmp_path / 'local', tmp_path / 'fallback'
    if fail_preferred:
        allocate = os.posix_fallocate
        def capacity(fd, start, size):
            if Path(os.readlink(f'/proc/self/fd/{fd}')).is_relative_to(primary):
                raise OSError(errno.ENOSPC, 'preferred storage exhausted')
            return allocate(fd, start, size)
        monkeypatch.setattr(os, 'posix_fallocate', capacity)
    expected_root = fallback if fail_preferred else primary
    source, manifest = extract_launcher(local_model_root=str(primary),
        local_model_fallback_roots=(str(fallback),), local_model_headroom_bytes=4096)
    scope = environment(checkpoints, tmp_path)
    exec(binding(source), scope)
    # Execute the actual retained argument assembly and draft-view helpers,
    # with synthetic metadata and no installation, CUDA, network or live server.
    scope.update(copy=copy, hashlib=hashlib, json=json, SGLANG='/fake/sglang',
                 SERVED_MODEL_NAME='flashnext', SERVED_MODEL_PORT=8001)
    nodes = ast.parse(source).body
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    constants = [node for node in nodes if any(assignment(node, name) for name in ('CFG', 'EXPERT_TARGET', 'DENSE_IGNORE'))]
    helpers = [node for node in nodes if isinstance(node, ast.FunctionDef)
               and node.name in ('indexed_shards', 'prepare_draft_view')]
    exec(compile(ast.Module(body=constants+helpers, type_ignores=[]), '<retained-helpers>', 'exec'), scope)
    scope['CFG']['FRSPEC'] = False
    prefix = tmp_path / 'prefix'; prefix.mkdir()
    scope['DRAFT_VIEW'] = scope['prepare_draft_view'](scope['MODEL_DIR'], scope['DRAFT_MODEL_DIR'], prefix)
    arguments = [node for node in nodes if assignment(node, 'graph_bs') or assignment(node, 'args')]
    exec(compile(ast.Module(body=arguments, type_ignores=[]), '<actual-server-arguments>', 'exec'), scope)
    spec = next(node for node in nodes if isinstance(node, ast.If) and ast.unparse(node.test) == "CFG['SPEC']"
                and any(isinstance(child, ast.AugAssign) for child in ast.walk(node)))
    exec(compile(ast.Module(body=[spec], type_ignores=[]), '<actual-draft-arguments>', 'exec'), scope)
    args = scope['args']
    assert Path(args[args.index('--model-path')+1]).is_relative_to(expected_root)
    view = Path(args[args.index('--speculative-draft-model-path')+1])
    for link in view.rglob('*'):
        if link.is_symlink():
            assert link.resolve().is_relative_to(expected_root)
    assert manifest['local_model_staging']['default'] == '1'
    assert source.index('staging_report = stage_checkpoints(') < source.index('args = [SGLANG') < source.index('subprocess.Popen(args')
    assert 'precache_model_thread.start()' in source  # Explicit old-path control remains.


def test_copy_barrier_blocks_handoff_until_every_worker_finishes(checkpoints, tmp_path, monkeypatch, local_storage):
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '1')
    original = model_staging._stage_file
    entered, release, launched = threading.Event(), threading.Event(), threading.Event()
    failures = []
    def slow(item, destination):
        if item[1].name == 'model-37.safetensors':
            entered.set()
            assert release.wait(5)
        return original(item, destination)
    monkeypatch.setattr(model_staging, '_stage_file', slow)
    source, _ = extract_launcher(local_model_root=str(tmp_path / 'local'), local_model_headroom_bytes=4096)
    scope = environment(checkpoints, tmp_path)
    def boot():
        try:
            exec(binding(source), scope)
            launched.set()
        except Exception as error:  # noqa: BLE001 - Return thread failures to the main test.
            failures.append(error)
    worker = threading.Thread(target=boot)
    worker.start()
    try:
        assert entered.wait(5)
        assert not launched.is_set() and worker.is_alive()
        assert scope['MODEL_DIR'] == str(checkpoints[0])
    finally:
        release.set(); worker.join(10)
    assert not failures and launched.is_set()
    assert scope['MODEL_DIR'] != str(checkpoints[0])


def test_failed_copy_never_publishes_or_retargets_server(checkpoints, tmp_path, monkeypatch, local_storage):
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '1')
    source, _ = extract_launcher(local_model_root=str(tmp_path / 'local'), local_model_headroom_bytes=4096)
    scope = environment(checkpoints, tmp_path)
    def fail(item, destination):
        raise OSError('synthetic copy failure')
    monkeypatch.setattr(model_staging, '_stage_file', fail)
    with pytest.raises(OSError, match='copy failure'):
        exec(binding(source), scope)
    assert scope['MODEL_DIR'] == str(checkpoints[0]) and 'LOCAL_MODEL_DIR' not in scope
    assert not list((tmp_path / 'local').glob('run-*'))
    report = json.loads((tmp_path / 'working/diagnostics/model_staging.json').read_text())
    assert report['status'] == 'failed'


def test_insufficient_disk_prevents_any_copy(checkpoints, tmp_path, monkeypatch):
    monkeypatch.setattr(model_staging.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    copier = Mock(); monkeypatch.setattr(model_staging, '_stage_file', copier)
    with pytest.raises(OSError, match='insufficient local model storage'):
        stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json')
    copier.assert_not_called()
    assert json.loads((tmp_path / 'report.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('bad', ['missing_shard', 'unsafe_index', 'escaped_symlink', 'directory_symlink'])
def test_incomplete_or_unsafe_inputs_fail_before_copy(checkpoints, tmp_path, bad):
    target = checkpoints[0]
    if bad == 'missing_shard':
        (target / 'model-00.safetensors').unlink()
    elif bad == 'unsafe_index':
        (target / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': '../escape'}}))
    else:
        outside = tmp_path / 'outside'; outside.mkdir()
        if bad == 'escaped_symlink':
            (outside / 'data').write_bytes(b'outside')
            (target / 'escape').symlink_to(outside / 'data')
        else:
            (target / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, FileNotFoundError)):
        stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json')
    assert not list((tmp_path / 'local').glob('run-*'))


def test_staging_off_preserves_input_reference_path(checkpoints, tmp_path, monkeypatch):
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '0')
    source, _ = extract_launcher(local_model_root=str(tmp_path / 'local'))
    scope = environment(checkpoints, tmp_path)
    exec(binding(source), scope)
    assert scope['LOCAL_MODEL_DIR'] == checkpoints[0]
    assert not (tmp_path / 'local').exists()


def test_target_only_copy_does_not_require_draft(checkpoints, tmp_path):
    report = stage_checkpoints(checkpoints[0], None, tmp_path / 'local', tmp_path / 'report.json')
    assert set(report['models']) == {'target'} and report['status'] == 'complete'


def test_local_root_cannot_resolve_into_input_mount(checkpoints, tmp_path):
    with pytest.raises(ValueError, match='/kaggle/input'):
        stage_checkpoints(*checkpoints, Path('/kaggle/input/models'), tmp_path / 'report.json')


def test_supervisor_surfaces_staging_completion_once(tmp_path):
    event = Mock()
    daemon = LLMDaemon(tmp_path / 'launcher', tmp_path, event)
    path = tmp_path / 'diagnostics/model_staging.json'; path.parent.mkdir()
    report = {'status': 'complete', 'models': {'target': {'shard_count': 38, 'path': '/local/target'}}}
    path.write_text(json.dumps(report))
    daemon.staging_report(); daemon.staging_report()
    event.assert_called_once_with('model_staging', report=report)


def test_every_file_and_headroom_reserved_before_first_copy(checkpoints, tmp_path, monkeypatch, local_storage):
    original = model_staging._stage_file
    file_count = sum(len(model_staging._staging_files(path)[0]) for path in checkpoints)
    observations = []
    def copy_reserved(item, destination):
        transaction = destination.parent
        report = json.loads((tmp_path / 'report.json').read_text())
        attempt = report['storage_attempts'][0]
        assert report['status'] == 'copying' and attempt['status'] == 'reserved'
        assert attempt['reserved_files'] == file_count
        assert (transaction / 'headroom.reserve').stat().st_blocks * 512 >= 4096
        path = destination / item[1]
        assert path.stat().st_size == item[2] and path.stat().st_blocks * 512 >= item[2]
        observations.append(path)
        return original(item, destination)
    monkeypatch.setattr(model_staging, '_stage_file', copy_reserved)
    report = stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json',
                               reserve=True, headroom_bytes=4096)
    assert len(observations) == file_count and report['capacity_reservation']
    assert not list((tmp_path / 'local').rglob('headroom.reserve'))
    for kind, source in zip(('target', 'draft'), checkpoints, strict=True):
        local = Path(report['models'][kind]['path'])
        for path in source.iterdir():
            assert (local / path.name).read_bytes() == path.read_bytes()


@pytest.mark.parametrize('failure', ['allocate_enospc', 'fsync_enospc', 'unsupported'])
def test_reservation_failure_despite_high_free_space_selects_next_root(checkpoints, tmp_path, monkeypatch, local_storage, failure):
    bad, good = tmp_path / 'bad', tmp_path / 'good'
    bad.mkdir(); (bad / 'unrelated').write_text('preserve me')
    allocate, sync = os.posix_fallocate, os.fsync
    def allocation(fd, start, size):
        path = Path(os.readlink(f'/proc/self/fd/{fd}'))
        if path.is_relative_to(bad) and failure != 'fsync_enospc':
            number = errno.ENOSPC if failure == 'allocate_enospc' else errno.ENOTSUP
            raise OSError(number, 'synthetic destination capacity failure')
        return allocate(fd, start, size)
    def fsync(fd):
        if Path(os.readlink(f'/proc/self/fd/{fd}')).is_relative_to(bad) and failure == 'fsync_enospc':
            raise OSError(errno.ENOSPC, 'synthetic fsync capacity failure')
        return sync(fd)
    monkeypatch.setattr(os, 'posix_fallocate', allocation)
    monkeypatch.setattr(os, 'fsync', fsync)
    original = model_staging._stage_file
    destinations = []
    def copier(item, destination):
        destinations.append(destination)
        return original(item, destination)
    monkeypatch.setattr(model_staging, '_stage_file', copier)
    report = stage_checkpoints(*checkpoints, bad, tmp_path / 'report.json', reserve=True,
                               fallback_roots=(good,), headroom_bytes=4096)
    assert report['root'] == str(good) and report['status'] == 'complete'
    assert report['storage_attempts'][0]['status'] == 'failed'
    assert report['storage_attempts'][0]['disk_free_before'] > report['bytes_required']
    assert all(path.is_relative_to(good) for path in destinations)
    assert not list(bad.glob('run-*')) and (bad / 'unrelated').read_text() == 'preserve me'


def test_exhausted_candidates_never_read_weight_files_or_publish(checkpoints, tmp_path, monkeypatch, local_storage):
    def exhausted(path, size):
        raise OSError(errno.EDQUOT, 'synthetic quota exhausted')
    monkeypatch.setattr(model_staging, '_reserve_file', exhausted)
    copier = Mock(); monkeypatch.setattr(model_staging, '_stage_file', copier)
    roots = (tmp_path / 'one', tmp_path / 'two')
    with pytest.raises(OSError, match='no local storage candidate'):
        stage_checkpoints(*checkpoints, roots[0], tmp_path / 'report.json', reserve=True,
                          fallback_roots=(roots[1],), headroom_bytes=4096)
    copier.assert_not_called()
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['status'] == 'failed' and not report['models']
    assert [attempt['errno'] for attempt in report['storage_attempts']] == [errno.EDQUOT] * 2
    assert all(not list(root.glob('run-*')) for root in roots)


def test_last_allocation_blocks_all_copy_workers(checkpoints, tmp_path, monkeypatch, local_storage):
    entered, release = threading.Event(), threading.Event()
    original = model_staging._reserve_file
    def reserve(path, size):
        if path.name == 'headroom.reserve':
            entered.set()
            assert release.wait(5)
        return original(path, size)
    monkeypatch.setattr(model_staging, '_reserve_file', reserve)
    copier = Mock(wraps=model_staging._stage_file)
    monkeypatch.setattr(model_staging, '_stage_file', copier)
    failures = []
    def boot():
        try:
            stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json',
                              reserve=True, headroom_bytes=4096)
        except Exception as error:  # noqa: BLE001 - Forward thread failures.
            failures.append(error)
    thread = threading.Thread(target=boot); thread.start()
    try:
        assert entered.wait(5)
        copier.assert_not_called()
        assert not list((tmp_path / 'local').rglob('target'))
    finally:
        release.set(); thread.join(10)
    assert not thread.is_alive() and not failures
    assert json.loads((tmp_path / 'report.json').read_text())['status'] == 'complete'


def test_sparse_truncate_cannot_fake_capacity(checkpoints, tmp_path, monkeypatch, local_storage):
    monkeypatch.setattr(os, 'posix_fallocate', lambda fd, start, size: os.ftruncate(fd, size))
    copier = Mock(); monkeypatch.setattr(model_staging, '_stage_file', copier)
    with pytest.raises(OSError, match='no local storage candidate'):
        stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json',
                          reserve=True, headroom_bytes=4096)
    copier.assert_not_called()
    report = json.loads((tmp_path / 'report.json').read_text())
    assert 'reservation is sparse' in report['storage_attempts'][0]['error']


@pytest.mark.parametrize('filesystem', ['tmpfs', 'fuse.gcsfuse', 'nfs4'])
def test_memory_and_network_mounts_are_not_local_disk_candidates(checkpoints, tmp_path, monkeypatch, filesystem):
    monkeypatch.setattr(model_staging, '_storage_profile', lambda root: {
        'disk_free_before': 10**12, 'free_inodes': 1000, 'filesystem': filesystem})
    allocator = Mock(); monkeypatch.setattr(model_staging, '_reserve_file', allocator)
    with pytest.raises(OSError, match='no local storage candidate'):
        stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json',
                          reserve=True, headroom_bytes=4096)
    allocator.assert_not_called()
    assert json.loads((tmp_path / 'report.json').read_text())['storage_attempts'][0]['errno'] == errno.ENOTSUP


def test_late_copy_capacity_failure_cleans_and_tries_reserved_fallback(checkpoints, tmp_path, monkeypatch, local_storage):
    bad, good = tmp_path / 'bad', tmp_path / 'good'
    original = model_staging._stage_file
    def copy(item, destination):
        if destination.is_relative_to(bad) and item[1].name == 'model-37.safetensors':
            raise OSError(errno.ENOSPC, 'late backing-store failure')
        return original(item, destination)
    monkeypatch.setattr(model_staging, '_stage_file', copy)
    report = stage_checkpoints(*checkpoints, bad, tmp_path / 'report.json', reserve=True,
                               fallback_roots=(good,), headroom_bytes=4096)
    assert report['status'] == 'complete' and report['root'] == str(good)
    assert not list(bad.glob('run-*'))
    assert all(Path(model['path']).is_relative_to(good) for model in report['models'].values())


def test_capacity_ablation_does_not_reserve_or_try_other_roots(checkpoints, tmp_path, monkeypatch):
    allocator = Mock(); monkeypatch.setattr(model_staging, '_reserve_file', allocator)
    report = stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json',
                               reserve=False, fallback_roots=(tmp_path / 'other',))
    allocator.assert_not_called()
    assert not report['capacity_reservation'] and len(report['storage_attempts']) == 1
    assert not (tmp_path / 'other').exists()


def test_unsafe_fallback_is_rejected_before_any_allocation(checkpoints, tmp_path, monkeypatch):
    allocator = Mock(); monkeypatch.setattr(model_staging, '_reserve_file', allocator)
    with pytest.raises(ValueError, match='/kaggle/input'):
        stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json', reserve=True,
                          fallback_roots=(Path('/kaggle/input/cache'),), headroom_bytes=4096)
    allocator.assert_not_called()
    assert not (tmp_path / 'local').exists()


def test_inode_exhaustion_is_rejected_before_file_allocation(checkpoints, tmp_path, monkeypatch):
    monkeypatch.setattr(model_staging, '_storage_profile', lambda root: {
        'disk_free_before': 10**12, 'free_inodes': 0, 'total_inodes': 10000, 'filesystem': 'ext4'})
    allocator = Mock(); monkeypatch.setattr(model_staging, '_reserve_file', allocator)
    with pytest.raises(OSError, match='no local storage candidate'):
        stage_checkpoints(*checkpoints, tmp_path / 'local', tmp_path / 'report.json',
                          reserve=True, headroom_bytes=4096)
    allocator.assert_not_called()
    assert 'inodes' in json.loads((tmp_path / 'report.json').read_text())['storage_attempts'][0]['error']


def test_stopped_server_removes_only_its_owned_output_copy(checkpoints, tmp_path, local_storage):
    daemon = LLMDaemon(tmp_path / 'launcher', tmp_path, Mock())
    root = tmp_path / '.arc3-models'
    report = stage_checkpoints(*checkpoints, root, tmp_path / 'diagnostics/model_staging.json',
        reserve=True, headroom_bytes=4096, owner_token=daemon._staging_owner)
    unrelated = root / 'run-other'; unrelated.mkdir(); (unrelated / 'keep').write_text('keep')
    assert Path(report['transaction']).is_dir()
    daemon.stop()
    assert not Path(report['transaction']).exists()
    assert (unrelated / 'keep').read_text() == 'keep'


@pytest.mark.parametrize('unsafe', ['other_owner', 'outside_output_root', 'symlink'])
def test_staging_cleanup_cannot_delete_unowned_paths(tmp_path, unsafe):
    daemon = LLMDaemon(tmp_path / 'launcher', tmp_path, Mock())
    root = tmp_path / '.arc3-models'; root.mkdir()
    transaction = root / 'run-fixture'
    if unsafe == 'outside_output_root':
        transaction = tmp_path / 'run-outside'
    protected = tmp_path / 'protected'; protected.mkdir()
    if unsafe == 'symlink':
        transaction.symlink_to(protected, target_is_directory=True)
    else:
        transaction.mkdir()
        protected = transaction
    (protected / '.arc3-owner').write_text('different-owner' if unsafe == 'other_owner' else daemon._staging_owner)
    (protected / 'keep').write_text('keep')
    diagnostics = tmp_path / 'diagnostics'; diagnostics.mkdir()
    (diagnostics / 'model_staging.json').write_text(json.dumps({'transaction': str(transaction)}))
    daemon.stop()
    assert (protected / 'keep').read_text() == 'keep'
