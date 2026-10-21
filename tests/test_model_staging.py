"""Real checkpoint copies and the exact generated server-path handoff."""
# Execute only repository-generated, hash-bound launcher code in these fixtures.
# ruff: noqa: S102
from __future__ import annotations

import ast
import copy
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
def checkpoints(tmp_path):
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


def test_generated_model_and_draft_launch_paths_are_local(checkpoints, tmp_path, monkeypatch):
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '1')
    source, manifest = extract_launcher(local_model_root=str(tmp_path / 'local'))
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
    assert Path(args[args.index('--model-path')+1]).is_relative_to(tmp_path / 'local')
    view = Path(args[args.index('--speculative-draft-model-path')+1])
    for link in view.rglob('*'):
        if link.is_symlink():
            assert link.resolve().is_relative_to(tmp_path / 'local')
    assert manifest['local_model_staging']['default'] == '1'
    assert source.index('staging_report = stage_checkpoints(') < source.index('args = [SGLANG') < source.index('subprocess.Popen(args')
    assert 'precache_model_thread.start()' in source  # Explicit old-path control remains.


def test_copy_barrier_blocks_handoff_until_every_worker_finishes(checkpoints, tmp_path, monkeypatch):
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
    source, _ = extract_launcher(local_model_root=str(tmp_path / 'local'))
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


def test_failed_copy_never_publishes_or_retargets_server(checkpoints, tmp_path, monkeypatch):
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '1')
    source, _ = extract_launcher(local_model_root=str(tmp_path / 'local'))
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
