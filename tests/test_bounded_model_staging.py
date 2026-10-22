"""Real CPU safetensors, bounded disk use and pinned loader integration.

Install torch from https://download.pytorch.org/whl/cpu and safetensors to run
the tensor fixtures without CUDA; neither dependency enters the agent sandbox.
"""
# Execute only repository code and the documented pinned upstream fixture.
# ruff: noqa: S102
from __future__ import annotations

import ast
import concurrent.futures
import copy
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import bounded_model_staging as bounded
from scripts.extract_llm_launcher import extract_launcher
from scripts.llm_assets import checkpoint_report


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    safetensors = pytest.importorskip('safetensors.torch')
    monkeypatch.setenv('ARC3_MODEL_OWNER', 'synthetic-owner')
    profile = bounded._storage_profile
    monkeypatch.setattr(bounded, '_storage_profile', lambda root: dict(profile(root), filesystem='ext4'))
    assets, expected = {}, {}
    for kind, count in [('target', 38), ('draft', 2)]:
        source = tmp_path / 'input' / kind
        source.mkdir(parents=True)
        index = {}
        for i in range(count):
            name = f'mtp.layers.0.mlp.experts.{kind}-{i}.weight'
            tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4) + i
            filename = f'model-{i:02}.safetensors'
            safetensors.save_file({name: tensor}, source / filename)
            index[name] = filename
            expected[name] = tensor
        config = ({'quant_method': 'auto-round', 'bits': 4} if kind == 'target' else
            {'quant_method': 'compressed-tensors', 'config_groups': {'mtp_routed_experts': {
                'targets': ['RoutedExperts'], 'weights': {'num_bits': 4, 'group_size': 32, 'symmetric': True}}}})
        (source / 'config.json').write_text(json.dumps({'quantization_config': config}))
        (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
        (source / 'tokenizer.json').write_text('{}')
        assets[kind] = checkpoint_report(source)
    root = tmp_path / 'working' / '.arc3-models'
    report_path = tmp_path / 'working' / 'diagnostics' / 'model_staging.json'
    report = bounded.prepare_bounded_checkpoints(assets, root, report_path,
        headroom_bytes=4096, owner_token='synthetic-owner')
    manifest = Path(report['manifest'])
    monkeypatch.setenv('ARC3_BOUNDED_MANIFEST', str(manifest))
    monkeypatch.setenv('ARC3_BOUNDED_MANIFEST_SHA256', hashlib.sha256(manifest.read_bytes()).hexdigest())
    return SimpleNamespace(assets=assets, report=report, report_path=report_path, manifest=manifest,
        transaction=Path(report['transaction']), expected=expected, torch=torch)


def files(setup, kind='target'):
    return bounded.managed_weights(setup.report['models'][kind]['path'])


def test_all_38_plus_2_shards_match_and_retained_tensors_survive_eviction(setup, monkeypatch):
    import safetensors
    opened = safetensors.safe_open
    observations = []
    def local_open(path, *args, **kwargs):
        cache = setup.transaction / 'cache'
        assert Path(path).parent == cache
        assert len(list(cache.iterdir())) <= 2
        physical_bytes = sum(p.stat().st_blocks * 512 for p in cache.iterdir())
        assert physical_bytes <= setup.report['cache_bound_bytes'] + 2 * os.statvfs(cache).f_frsize
        observations.append(sum(p.stat().st_size for p in cache.iterdir()))
        return opened(path, *args, **kwargs)
    monkeypatch.setattr(safetensors, 'safe_open', local_open)
    retained = dict(bounded.weights_iterator(files(setup)))
    retained.update(bounded.weights_iterator(files(setup, 'draft')))
    assert len(observations) == 40 and max(observations) <= setup.report['cache_bound_bytes']
    assert not list((setup.transaction / 'cache').iterdir())
    for name, tensor in retained.items():
        assert setup.torch.equal(tensor, setup.expected[name])
    report = json.loads(setup.report_path.read_text())
    assert report['loaded_shards'] == 40 and report['copied_shards'] == 40
    assert len(report['shards']) == 40
    assert all(item['verified_sha256'] == item['source_stream_sha256'] for item in report['shards'])


def test_metadata_only_does_not_publish_placeholder_or_source_link(setup):
    for model in setup.report['models'].values():
        local = Path(model['path'])
        assert not list(local.glob('*.safetensors'))
        assert not any(path.is_symlink() for path in local.rglob('*'))
        assert (local / 'config.json').read_bytes() == Path(model['source'], 'config.json').read_bytes()
    assert setup.report['status'] == 'metadata_ready'


def test_loader_waits_for_complete_copy_before_first_open(setup, monkeypatch):
    original = bounded._stage_file
    entered, release = threading.Event(), threading.Event()
    def blocking(item, destination):
        entered.set()
        assert release.wait(5)
        return original(item, destination)
    monkeypatch.setattr(bounded, '_stage_file', blocking)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.submit(lambda: list(bounded.weights_iterator(files(setup)[:1])))
        assert entered.wait(5) and not result.done()
        release.set()
        assert len(result.result(timeout=5)) == 1


@pytest.mark.parametrize('failure', ['fsync', 'digest', 'size'])
def test_corrupt_or_incomplete_copy_never_reaches_safetensors_open(setup, monkeypatch, failure):
    import safetensors
    original = bounded._stage_file
    opened = []
    monkeypatch.setattr(safetensors, 'safe_open', lambda *args, **kwargs: opened.append(args))
    def fail(item, destination):
        if failure == 'fsync':
            raise OSError('synthetic ENOSPC at fsync')
        result = original(item, destination)
        (destination / item[1]).write_bytes(b'bad' if failure == 'size' else b'x'*item[2])
        return result
    monkeypatch.setattr(bounded, '_stage_file', fail)
    with pytest.raises((OSError, ValueError)):
        list(bounded.weights_iterator(files(setup)[:1]))
    assert not opened and not list((setup.transaction / 'cache').iterdir())
    assert json.loads(setup.report_path.read_text())['status'] == 'failed'


def test_next_copy_failure_cleans_cache_and_propagates(setup, monkeypatch):
    original = bounded._stage_file
    def fail_next(item, destination):
        if item[1].name == 'model-01.safetensors':
            raise OSError('synthetic next-copy failure')
        return original(item, destination)
    monkeypatch.setattr(bounded, '_stage_file', fail_next)
    iterator = bounded.weights_iterator(files(setup)[:2])
    next(iterator)
    with pytest.raises(OSError, match='next-copy'):
        next(iterator)
    assert not list((setup.transaction / 'cache').iterdir())


def test_cancellation_joins_prefetch_then_reclaims_both_slots(setup):
    iterator = bounded.weights_iterator(files(setup))
    _, tensor = next(iterator)
    iterator.close()
    assert not list((setup.transaction / 'cache').iterdir())
    assert tensor.shape == (4, 4)
    assert json.loads(setup.report_path.read_text())['error'].startswith('GeneratorExit:')


def test_target_and_draft_iterators_share_interprocess_lock(setup, monkeypatch):
    original = bounded._stage_file
    entered, release, draft_attempted = threading.Event(), threading.Event(), threading.Event()
    calls = []
    flock = bounded.fcntl.flock
    def observed_lock(fd, operation):
        if entered.is_set():
            draft_attempted.set()
        return flock(fd, operation)
    monkeypatch.setattr(bounded.fcntl, 'flock', observed_lock)
    def blocking(item, destination):
        calls.append(item[0].parent.name)
        entered.set()
        assert release.wait(5)
        return original(item, destination)
    monkeypatch.setattr(bounded, '_stage_file', blocking)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        target = pool.submit(lambda: list(bounded.weights_iterator(files(setup)[:1])))
        assert entered.wait(5)
        draft = pool.submit(lambda: list(bounded.weights_iterator(files(setup, 'draft')[:1])))
        assert draft_attempted.wait(5) and not draft.done() and calls == ['target']
        release.set()
        assert target.result(timeout=5) and draft.result(timeout=5)
    assert calls == ['target', 'draft']


@pytest.mark.parametrize('failure', ['manifest', 'owner', 'source_size', 'unmanaged', 'prefetch', 'stale_cache'])
def test_fail_closed_on_drift_and_unsupported_paths(setup, monkeypatch, failure):
    selected = files(setup)[:1]
    if failure == 'manifest':
        setup.manifest.write_text(setup.manifest.read_text()+' ')
    elif failure == 'owner':
        monkeypatch.setenv('ARC3_MODEL_OWNER', 'other-run')
    elif failure == 'source_size':
        Path(setup.assets['target']['path'], 'model-00.safetensors').write_bytes(b'short')
    elif failure == 'unmanaged':
        selected = ['/kaggle/input/unmanaged.safetensors']
    elif failure == 'stale_cache':
        (setup.transaction / 'cache' / 'leftover').write_bytes(b'partial')
    with pytest.raises(ValueError):
        list(bounded.weights_iterator(selected, prefetch=failure == 'prefetch'))


def test_capacity_rejection_reads_no_weight_bytes(setup, tmp_path, monkeypatch):
    monkeypatch.setattr(bounded, '_storage_profile', lambda root: {'disk_free_before': 1})
    monkeypatch.setattr(bounded, '_stage_file', lambda *args: pytest.fail('must reject before copying'))
    with pytest.raises(OSError, match='two shards'):
        bounded.prepare_bounded_checkpoints(setup.assets, tmp_path / 'small', tmp_path / 'report.json',
                                            headroom_bytes=4096, owner_token='owner')


def test_growing_source_cannot_exceed_manifest_disk_bound(tmp_path):
    source = tmp_path / 'source'; source.write_bytes(b'x'*32)
    destination = tmp_path / 'cache'; destination.mkdir()
    with pytest.raises(OSError, match='grew beyond'):
        bounded._stage_file((source, Path('shard'), 8), destination)
    assert (destination / 'shard').stat().st_size == 8


def test_exact_generated_handoff_and_native_loader_consume_local_cache(setup, tmp_path, monkeypatch):
    monkeypatch.setenv('ENABLE_BOUNDED_MODEL_STAGING', '1')
    monkeypatch.setenv('ENABLE_LOCAL_MODEL_STAGING', '1')
    source, manifest = extract_launcher(local_model_root=str(tmp_path / 'working' / '.arc3-models'),
                                      bounded_model_headroom_bytes=4096)
    nodes = ast.parse(source).body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.If)
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                         and call.func.id == 'prepare_bounded_checkpoints' for call in ast.walk(node)))
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'LOCAL_MODEL_DIR' for t in node.targets))
    scope = {'Path': Path, 'os': os, 'hashlib': hashlib, 'asset_report': setup.assets,
             'WORKING_DIR': tmp_path / 'working', 'DRAFT_MODEL_DIR': setup.assets['draft']['path'],
             'prepare_bounded_checkpoints': bounded.prepare_bounded_checkpoints}
    exec(compile(ast.Module(body=nodes[start:end+1], type_ignores=[]), '<generated-handoff>', 'exec'), scope)
    local = scope['LOCAL_MODEL_DIR']
    assert local.is_relative_to(tmp_path / 'working' / '.arc3-models')
    scope.update(copy=copy, json=json, bounded_indexed_shards=bounded.bounded_indexed_shards,
                 SGLANG='/fixture/sglang', SERVED_MODEL_NAME='flashnext', SERVED_MODEL_PORT=8001)
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    helpers = [node for node in nodes if (isinstance(node, ast.FunctionDef)
        and node.name in ('indexed_shards', 'prepare_draft_view')) or
        any(assignment(node, name) for name in ('CFG', 'EXPERT_TARGET', 'DENSE_IGNORE'))]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), '<generated-draft-helpers>', 'exec'), scope)
    scope['CFG']['FRSPEC'] = False
    scope['DRAFT_VIEW'] = scope['prepare_draft_view'](str(local), scope['DRAFT_MODEL_DIR'], tmp_path)
    view = scope['DRAFT_VIEW']
    assert not list(view.glob('*.safetensors'))  # No placeholders or FUSE weight links.
    bounded.register_bounded_alias(Path(scope['staging_report']['manifest']), view, Path(scope['DRAFT_MODEL_DIR']))
    arguments = [node for node in nodes if assignment(node, 'graph_bs') or assignment(node, 'args')]
    exec(compile(ast.Module(body=arguments, type_ignores=[]), '<generated-args>', 'exec'), scope)
    assert scope['args'][scope['args'].index('--model-path')+1] == str(local)
    # Actual pinned upstream methods, exercised on CPU instead of mocking the
    # loader/iterator boundary. Only heavyweight CUDA dependencies are omitted.
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'pennyroyal_loader_contract.json').read_text())
    contract = 'from __future__ import annotations\nclass DefaultModelLoader:\n' + '\n'.join(fixture['methods'].values())
    patched = bounded.patch_bounded_loader(contract, expected_sha256=hashlib.sha256(contract.encode()).hexdigest())
    monkeypatch.setitem(sys.modules, 'arc3_bounded_loader', bounded)
    formats = SimpleNamespace(SAFETENSORS='safetensors', NPCACHE='npcache', FASTSAFETENSORS='fast')
    loader_scope = {'os': os, 'LoadFormat': formats, 'get_model': lambda: SimpleNamespace(
        weight_loader_disable_mmap=False, weight_loader_prefetch_checkpoints=False,
        weight_loader_prefetch_num_threads=4, weight_loader_drop_cache_after_load=True)}
    exec(patched, loader_scope)
    loader = loader_scope['DefaultModelLoader']()
    loader._maybe_download_from_modelscope = lambda path, revision: path
    # Draft loaders may omit enable_multithread_load; the adapter still owns
    # that path instead of selecting the upstream unbounded thread loader.
    loader.load_config = SimpleNamespace(load_format='safetensors',
        model_loader_extra_config={}, draft_model_idx=None)
    loader.counter_before_loading_weights = 1
    source_record = SimpleNamespace(model_or_path=str(local), revision=None, fall_back_to_pt=False,
                                    model_config=None, prefix='')
    loaded = dict(loader._get_weights_iterator(source_record))
    assert len(loaded) == 38 and fixture['full_file_sha256'] == bounded.PINNED_LOADER_SHA256
    for name, tensor in loaded.items():
        assert setup.torch.equal(tensor, setup.expected[name])
    draft_record = SimpleNamespace(**vars(source_record))
    draft_record.model_or_path = str(view)
    draft_loaded = dict(loader._get_weights_iterator(draft_record))
    assert len(draft_loaded) == 2
    for name, tensor in draft_loaded.items():
        assert setup.torch.equal(tensor, setup.expected[name])
    assert manifest['bounded_model_staging']['cache_slots'] == 2


def test_loader_hash_change_rejected_before_patch(tmp_path):
    with pytest.raises(ValueError, match='hash differs'):
        bounded.patch_bounded_loader('class DefaultModelLoader: pass\n')


def test_generated_support_modules_and_installed_adapter_are_self_contained(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts/bounded_model_staging.py').read_text()
    staging = (root / 'scripts/model_staging.py').read_text()
    fixture = json.loads((Path(__file__).parent / 'fixtures' / 'pennyroyal_loader_contract.json').read_text())
    contract = 'from __future__ import annotations\nclass DefaultModelLoader:\n' + '\n'.join(fixture['methods'].values())
    # The production hash check has its own rejection test; exercise the exact
    # installer with the reviewed method fixture in place of a 173 KB wheel file.
    patch = bounded.patch_bounded_loader
    monkeypatch.setattr(bounded, 'patch_bounded_loader', lambda text: patch(text,
        expected_sha256=hashlib.sha256(contract.encode()).hexdigest()))
    venv = tmp_path / 'venv'
    loader_path = venv / 'lib/python3.12/site-packages/sglang/srt/model_loader/loader.py'
    loader_path.parent.mkdir(parents=True)
    loader_path.write_text(contract)
    support = tmp_path / 'clean_python'; support.mkdir()
    report = bounded.install_bounded_loader(venv, support, source, staging)
    assert hashlib.sha256(loader_path.read_bytes()).hexdigest() == report['patched_sha256']
    # Import like the actual isolated SGLang child, outside the repository
    # package namespace; this catches missing relative/support imports.
    monkeypatch.syspath_prepend(str(support))
    monkeypatch.delitem(sys.modules, 'arc3_bounded_loader', raising=False)
    monkeypatch.delitem(sys.modules, 'arc3_model_staging', raising=False)
    import arc3_bounded_loader
    assert arc3_bounded_loader.PINNED_LOADER_SHA256 == bounded.PINNED_LOADER_SHA256


def test_draft_alias_uses_same_cache_without_source_weight_links(setup, tmp_path):
    view = tmp_path / 'draft-view'
    view.mkdir()
    original = Path(setup.report['models']['draft']['path'])
    (view / 'model.safetensors.index.json').symlink_to(original / 'model.safetensors.index.json')
    bounded.register_bounded_alias(setup.manifest, view, original)
    loaded = dict(bounded.weights_iterator(bounded.managed_weights(str(view))))
    assert len(loaded) == 2 and not list(view.glob('*.safetensors'))
