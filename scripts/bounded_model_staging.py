"""Two-slot local shard pipeline for the pinned Pennyroyal safetensors loader.

Only the copier reads input weights. SGLang sees local metadata and consumes
synced, digest-checked local shards. CPU clones detach tensors from file mmaps
before eviction; target and draft share one interprocess disk-cache lock.
"""
from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from .model_staging import _stage_file, _staging_files, _storage_profile
except ImportError:
    from arc3_model_staging import _stage_file, _staging_files, _storage_profile

PINNED_LOADER_SHA256 = '1016b2c6c2d7eb50ef25e0a731e20096e4b79239bccb6fea26aa59d3f29bf8fe'


def _bounded_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare_bounded_checkpoints(assets: dict, root: Path, report_path: Path, *,
                                headroom_bytes: int = 4 * 1024**3,
                                owner_token: str) -> dict:
    """Copy metadata only; no absent shard is represented as a ready file."""
    if type(headroom_bytes) is not int or headroom_bytes < 0 or not owner_token:
        raise ValueError('bounded staging needs nonnegative headroom and an owner')
    root = root.resolve()
    models = {kind: model for kind, model in assets.items() if kind in ('target', 'draft')}
    if ('target' not in models or root.is_relative_to(Path('/kaggle/input'))
            or any(root.is_relative_to(Path(model['path']).resolve()) for model in models.values())):
        raise ValueError('bounded cache must be outside source checkpoints')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'bounded_shards', 'status': 'preparing', 'root': str(root),
              'models': {}, 'headroom_bytes': headroom_bytes, 'cache_slots': 2}
    _bounded_save(report_path, report)
    transaction = None
    try:
        inventories = {kind: _staging_files(Path(model['path']).resolve(strict=True))
                       for kind, model in models.items()}
        sizes = [size for model in models.values() for size in model['shard_sizes'].values()]
        if any(Path(name).name != name for model in models.values() for name in model['shard_sizes']):
            raise ValueError('bounded loader requires flat indexed shard paths')
        maximum = max(sizes)
        metadata_bytes = sum(size for files, _ in inventories.values() for _, rel, size in files
                             if rel.suffix != '.safetensors')
        cache_bound = 2 * maximum
        profile = _storage_profile(root)
        report.update(storage=profile, cache_bound_bytes=cache_bound, metadata_bytes=metadata_bytes)
        filesystem = profile.get('filesystem', '')
        if (profile.get('read_only') or filesystem.startswith('fuse')
                or filesystem in ('tmpfs', 'ramfs', 'nfs', 'nfs4', 'cifs', '9p')):
            raise OSError('bounded checkpoint cache needs writable local disk')
        if profile['disk_free_before'] < metadata_bytes + cache_bound + headroom_bytes:
            raise OSError('insufficient disk for two shards plus metadata and headroom')
        if profile.get('total_inodes', 0) and profile['free_inodes'] < 64 + sum(
                len(files) for files, _ in inventories.values()):
            raise OSError('insufficient cache inodes')
        transaction = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        (transaction / '.arc3-owner').write_text(owner_token)
        report['transaction'] = str(transaction)
        _bounded_save(report_path, report)
        plan = {'transaction': str(transaction), 'owner': owner_token, 'models': {},
                'aliases': {}, 'cache_bound_bytes': cache_bound, 'headroom_bytes': headroom_bytes,
                'report_path': str(report_path), 'schema': 1}
        for kind, model in models.items():
            source = Path(model['path']).resolve(strict=True)
            local = transaction / kind
            local.mkdir()
            copied = [_stage_file(item, local) for item in inventories[kind][0]
                      if item[1].suffix != '.safetensors']
            for filename, field in [('config.json', 'config_sha256'),
                                    ('model.safetensors.index.json', 'index_sha256')]:
                if hashlib.sha256((local / filename).read_bytes()).hexdigest() != model[field]:
                    raise ValueError(f'bounded metadata changed: {kind}/{filename}')
            index = json.loads((local / 'model.safetensors.index.json').read_text())
            if set(index['weight_map'].values()) != set(model['shard_sizes']):
                raise ValueError('bounded shard index changed')
            plan['models'][str(local)] = {'source': str(source), 'shards': model['shard_sizes'], 'kind': kind}
            report['models'][kind] = {'path': str(local), 'source': str(source),
                'shard_count': len(model['shard_sizes']), 'metadata_files': copied}
        (transaction / 'cache').mkdir()
        manifest = transaction / 'bounded-manifest.json'
        _bounded_save(manifest, plan)
        report.update(status='metadata_ready', manifest=str(manifest), loaded_shards=0,
                      copied_shards=0, cache_high_water_bytes=0)
        _bounded_save(report_path, report)
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        _bounded_save(report_path, report)
        if transaction is not None:
            try:
                shutil.rmtree(transaction)
            except OSError as cleanup_error:
                report['cleanup_error'] = str(cleanup_error)
                _bounded_save(report_path, report)
        raise


def _bounded_plan() -> tuple[Path, dict]:
    manifest = Path(os.environ['ARC3_BOUNDED_MANIFEST'])
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != os.environ['ARC3_BOUNDED_MANIFEST_SHA256']:
        raise ValueError('bounded manifest changed')
    plan = json.loads(raw)
    transaction = Path(plan['transaction'])
    if (manifest.parent != transaction or transaction.is_symlink()
            or (transaction / '.arc3-owner').read_text() != plan['owner']
            or plan['owner'] != os.environ['ARC3_MODEL_OWNER']):
        raise ValueError('bounded transaction ownership changed')
    return transaction, plan


def managed_weights(model_path: str) -> list[str]:
    """Resolve indexed logical paths without globbing placeholders or FUSE links."""
    _, plan = _bounded_plan()
    path = str(Path(model_path).resolve())
    original = plan['aliases'].get(path, path)
    if original not in plan['models']:
        raise ValueError(f'unmanaged bounded checkpoint: {model_path}')
    return [str(Path(path) / name) for name in sorted(plan['models'][original]['shards'])]


def bounded_indexed_shards(model_path: str) -> tuple[dict, list[str]]:
    files = managed_weights(model_path)
    index = json.loads(Path(model_path, 'model.safetensors.index.json').read_text())
    return index, [str(Path(file).relative_to(model_path)) for file in files]


def register_bounded_alias(manifest: Path, alias: Path, original: Path) -> None:
    plan = json.loads(manifest.read_text())
    if str(original.resolve()) not in plan['models']:
        raise ValueError('unmanaged draft alias')
    plan['aliases'][str(alias.resolve())] = str(original.resolve())
    _bounded_save(manifest, plan)
    os.environ['ARC3_BOUNDED_MANIFEST_SHA256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()


def patch_bounded_loader(source: str, *, expected_sha256: str = PINNED_LOADER_SHA256) -> str:
    """Fail closed if the installed wheel differs from the reviewed loader."""
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ValueError('Pennyroyal loader hash differs from pinned upstream')
    anchor = '        is_local = os.path.isdir(model_name_or_path)\n'
    replacement = '''        from arc3_bounded_loader import managed_weights
        if self.load_config.load_format != LoadFormat.SAFETENSORS:
            raise ValueError("bounded loading requires safetensors")
        return model_name_or_path, managed_weights(model_name_or_path), True

'''+anchor
    iterator = 'weights_iterator = safetensors_weights_iterator('
    changes = {'_prepare_weights': [(anchor, replacement)], '_get_weights_iterator': [(iterator,
        'weights_iterator = __import__("arc3_bounded_loader").weights_iterator('),
        ('use_multithread = extra_config.get("enable_multithread_load", True)',
         'use_multithread = False  # The bounded pipeline owns target and draft I/O.') ]}
    loader = next(node for node in ast.parse(source).body
                  if isinstance(node, ast.ClassDef) and node.name == 'DefaultModelLoader')
    lines = source.splitlines(keepends=True)
    found = set()
    for node in reversed(loader.body):
        if isinstance(node, ast.FunctionDef) and node.name in changes:
            segment = ''.join(lines[node.lineno-1:node.end_lineno])
            for before, after in changes[node.name]:
                if segment.count(before) != 1:
                    raise ValueError('Pennyroyal loader integration boundary changed')
                segment = segment.replace(before, after)
            lines[node.lineno-1:node.end_lineno] = [segment]
            found.add(node.name)
    if found != set(changes):
        raise ValueError('Pennyroyal default loader methods missing')
    patched = ''.join(lines)
    compile(patched, '<bounded-pennyroyal-loader>', 'exec')
    return patched


def install_bounded_loader(venv: Path, support_dir: Path, bounded_source: str,
                           staging_source: str) -> dict:
    loaders = list(venv.glob('lib/python*/site-packages/sglang/srt/model_loader/loader.py'))
    if len(loaders) != 1:
        raise ValueError('cannot locate unique pinned SGLang loader')
    path = loaders[0]
    patched = patch_bounded_loader(path.read_text())
    (support_dir / 'arc3_bounded_loader.py').write_text(bounded_source)
    (support_dir / 'arc3_model_staging.py').write_text(staging_source)
    path.write_text(patched)
    return {'original_sha256': PINNED_LOADER_SHA256,
            'patched_sha256': hashlib.sha256(patched.encode()).hexdigest()}


def weights_iterator(hf_weights_files: list[str], disable_mmap: bool = False,
                     prefetch: bool = False, prefetch_num_threads: int = 4,
                     drop_cache_after_load: bool = False):
    """At most current + next shard, shared across all managed model iterators."""
    # Imports occur only in the serving venv, never in the agent REPL.
    import safetensors

    if disable_mmap or prefetch:
        raise ValueError('bounded loading owns mmap and prefetch; disable competing loaders')
    transaction, plan = _bounded_plan()
    cache = transaction / 'cache'
    report_path = Path(plan['report_path'])
    entries = []
    for filename in hf_weights_files:
        path = Path(filename)
        model_path = str(path.parent.resolve())
        original = plan['aliases'].get(model_path, model_path)
        model = plan['models'].get(original)
        if model is None or path.name not in model['shards'] or path.is_symlink():
            raise ValueError(f'unmanaged bounded shard: {filename}')
        entries.append((Path(model['source']) / path.name, Path(path.name), model['shards'][path.name]))
    if not entries or len({item[0] for item in entries}) != len(entries):
        raise ValueError('empty or duplicate bounded shard list')
    # One lock covers the whole iterator: concurrent target/draft or child
    # processes cannot exceed the aggregate two-slot bound.
    with (transaction / 'cache.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(cache.iterdir()):
            raise ValueError('unconsumed cache from interrupted loader; start a fresh transaction')
        report = json.loads(report_path.read_text())

        def save(**changes):
            report.update(changes)
            _bounded_save(report_path, report)

        def copy(item):
            source, relative, expected = item
            if source.is_symlink() or source.stat().st_size != expected:
                raise ValueError('source shard changed after manifest preparation')
            if shutil.disk_usage(cache).free < expected + plan['headroom_bytes']:
                raise OSError('bounded cache headroom exhausted')
            started = time.monotonic()
            result = _stage_file(item, cache)
            destination = cache / relative
            with destination.open('rb') as stream:
                local_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if local_hash != result['source_stream_sha256']:
                raise ValueError('local shard digest differs from copied source stream')
            result['verified_sha256'] = local_hash
            result['copy_seconds'] = round(time.monotonic()-started, 3)
            return result

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(copy, entries[0])
                for index, item in enumerate(entries):
                    result = pending.result()  # Complete copy/fsync/digest barrier.
                    current = cache / item[1]
                    # Bound accounting includes the next shard's complete size
                    # even while its worker is still writing it.
                    active_bytes = item[2] + (entries[index+1][2] if index+1 < len(entries) else 0)
                    if active_bytes > plan['cache_bound_bytes']:
                        raise ValueError('bounded cache size invariant violated')
                    high = max(report['cache_high_water_bytes'], active_bytes)
                    if index+1 < len(entries):
                        pending = pool.submit(copy, entries[index+1])
                    save(status='loading', current_shard=item[1].name,
                         copied_shards=report['copied_shards']+1, cache_high_water_bytes=high)
                    with safetensors.safe_open(current, framework='pt', device='cpu') as opened:
                        for name in opened.keys():  # noqa: SIM118 -- safe_open is not iterable.
                            # Keep consumers' retained tensors independent of
                            # this file and asynchronous device copies.
                            tensor = opened.get_tensor(name).clone()
                            yield name, tensor
                            del tensor
                    current.unlink()
                    report.setdefault('shards', []).append(result)
                    save(loaded_shards=report['loaded_shards']+1)
            save(status='iterator_complete', current_shard=None)
        except BaseException as error:
            save(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            # Pool shutdown joins the sole writer before cancellation cleanup.
            for path in cache.iterdir():
                path.unlink()
