"""Bounded local shard pipeline for the pinned Pennyroyal safetensors loader.

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
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from .model_staging import _stage_file, _staging_files, _storage_profile
except ImportError:
    from arc3_model_staging import _stage_file, _staging_files, _storage_profile

PINNED_LOADER_SHA256 = '1016b2c6c2d7eb50ef25e0a731e20096e4b79239bccb6fea26aa59d3f29bf8fe'
PINNED_WHEEL_SHA256 = 'd0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225'


def stage_shard(item: tuple[Path, Path, int], destination: Path, *,
                 read_workers: int = 4, chunk_bytes: int = 8 * 1024**2) -> dict:
    """Overlap bounded FUSE range reads; hash/write bytes in original order.

    At most read_workers requests and (read_workers+1)*chunk_bytes of Python
    data are live. Disk still respects the manifest's indexed shard window.
    """
    if not 1 <= read_workers <= 8 or not 1 <= chunk_bytes <= 8 * 1024**2:
        raise ValueError('invalid bounded range-read configuration')
    if read_workers == 1:
        return dict(_stage_file(item, destination), read_workers=1)
    source, relative, expected = item
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with source.open('rb', buffering=0) as reader, target.open('xb') as writer:
        def read_range(offset: int) -> bytearray:
            block = bytearray(min(chunk_bytes, expected-offset))
            view = memoryview(block)
            filled = 0
            while filled < len(block):
                count = os.preadv(reader.fileno(), [view[filled:]], offset+filled)
                if not count:
                    raise OSError('source shard truncated during bounded range read')
                filled += count
            return block

        offsets = iter(range(0, expected, chunk_bytes))
        pending = deque()
        with ThreadPoolExecutor(max_workers=read_workers) as pool:
            try:
                for _ in range(read_workers):
                    offset = next(offsets, None)
                    if offset is not None:
                        pending.append(pool.submit(read_range, offset))
                while pending:
                    block = pending.popleft().result()
                    writer.write(block)
                    digest.update(block)
                    count += len(block)
                    del block
                    offset = next(offsets, None)
                    if offset is not None:
                        pending.append(pool.submit(read_range, offset))
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        if os.pread(reader.fileno(), 1, expected):
            raise OSError('source grew beyond bounded shard inventory')
        writer.flush()
        os.fsync(writer.fileno())
    if count != expected or target.stat().st_size != expected:
        raise OSError('incomplete bounded shard copy')
    return {'name': str(relative), 'bytes': count, 'read_workers': read_workers,
            'source_stream_sha256': digest.hexdigest()}


def _bounded_save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare_bounded_checkpoints(assets: dict, root: Path, report_path: Path, *,
                                headroom_bytes: int = 4 * 1024**3,
                                owner_token: str, cache_slots: int = 3) -> dict:
    """Copy metadata only; no absent shard is represented as a ready file."""
    if type(headroom_bytes) is not int or headroom_bytes < 0 or not owner_token:
        raise ValueError('bounded staging needs nonnegative headroom and an owner')
    if type(cache_slots) is not int or cache_slots not in (2, 3):
        raise ValueError('bounded staging needs two or three slots')
    root = root.resolve()
    models = {kind: model for kind, model in assets.items() if kind in ('target', 'draft')}
    if ('target' not in models or root.is_relative_to(Path('/kaggle/input'))
            or any(root.is_relative_to(Path(model['path']).resolve()) for model in models.values())):
        raise ValueError('bounded cache must be outside source checkpoints')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'bounded_shards', 'status': 'preparing', 'root': str(root),
              'models': {}, 'headroom_bytes': headroom_bytes, 'cache_slots': cache_slots}
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
        cache_bound = cache_slots * maximum
        profile = _storage_profile(root)
        report.update(storage=profile, cache_bound_bytes=cache_bound, metadata_bytes=metadata_bytes)
        filesystem = profile.get('filesystem', '')
        if (profile.get('read_only') or filesystem.startswith('fuse')
                or filesystem in ('tmpfs', 'ramfs', 'nfs', 'nfs4', 'cifs', '9p')):
            raise OSError('bounded checkpoint cache needs writable local disk')
        if profile['disk_free_before'] < metadata_bytes + cache_bound + headroom_bytes:
            raise OSError('insufficient disk for bounded shards plus metadata and headroom')
        if profile.get('total_inodes', 0) and profile['free_inodes'] < 64 + sum(
                len(files) for files, _ in inventories.values()):
            raise OSError('insufficient cache inodes')
        transaction = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        (transaction / '.arc3-owner').write_text(owner_token)
        report['transaction'] = str(transaction)
        _bounded_save(report_path, report)
        plan = {'transaction': str(transaction), 'owner': owner_token, 'models': {},
                'aliases': {}, 'cache_bound_bytes': cache_bound, 'headroom_bytes': headroom_bytes,
                'report_path': str(report_path), 'schema': 1, 'cache_slots': cache_slots}
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
    loader = next(node for node in ast.parse(source).body
                  if isinstance(node, ast.ClassDef) and node.name == 'DefaultModelLoader')
    method = next(node for node in loader.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_get_weights_iterator')
    calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'safetensors_weights_iterator']
    if (len(calls) != 1 or len(calls[0].args) != 1
            or not isinstance(calls[0].args[0], ast.Name) or calls[0].args[0].id != 'hf_weights_files'
            or any(keyword.arg not in {'disable_mmap', 'prefetch', 'prefetch_num_threads',
                       'prefetch_lookahead', 'drop_cache_after_load'} for keyword in calls[0].keywords)):
        raise ValueError('unreviewed Pennyroyal safetensors iterator call')
    native_call = ast.get_source_segment(source, calls[0])
    assert native_call is not None
    # Own the complete copy/mmap/prefetch policy instead of forwarding native
    # optional knobs (the packaged wheel adds prefetch_lookahead). Unknown
    # arguments still fail above; no catch-all keyword acceptance is used.
    changes = {'_prepare_weights': [(anchor, replacement)], '_get_weights_iterator': [(native_call,
        '__import__("arc3_bounded_loader").weights_iterator(hf_weights_files)'),
        ('use_multithread = extra_config.get("enable_multithread_load", True)',
         'use_multithread = False  # The bounded pipeline owns target and draft I/O.')]}
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
                           staging_source: str, wheel_path: Path, contract_path: Path) -> dict:
    loaders = list(venv.glob('lib/python*/site-packages/sglang/srt/model_loader/loader.py'))
    if len(loaders) != 1:
        raise ValueError('cannot locate unique pinned SGLang loader')
    path = loaders[0]
    with wheel_path.open('rb') as stream:
        wheel_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if wheel_digest != PINNED_WHEEL_SHA256:
        raise ValueError('Pennyroyal wheel differs from approved artifact')
    with zipfile.ZipFile(wheel_path) as archive:
        member = archive.read('sglang/srt/model_loader/loader.py')
    if path.read_bytes() != member:
        raise ValueError('installed Pennyroyal loader differs from approved wheel member')
    source = member.decode('utf-8')
    original_digest = hashlib.sha256(member).hexdigest()
    tree = ast.parse(source)
    loader = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == 'DefaultModelLoader')
    lines = source.splitlines(keepends=True)
    methods = {node.name: ''.join(lines[node.lineno-1:node.end_lineno]) for node in loader.body
               if isinstance(node, ast.FunctionDef) and node.name in ('_prepare_weights', '_get_weights_iterator')}
    contract = {'wheel_sha256': wheel_digest, 'full_file_sha256': original_digest,
                'upstream_reference_sha256': PINNED_LOADER_SHA256, 'methods': methods}
    _bounded_save(contract_path, contract)
    try:
        patched = patch_bounded_loader(source, expected_sha256=original_digest)
    except (ValueError, StopIteration):
        print('Bounded loader integration rejected: '+json.dumps(contract), flush=True)
        raise
    (support_dir / 'arc3_bounded_loader.py').write_text(bounded_source)
    (support_dir / 'arc3_model_staging.py').write_text(staging_source)
    path.write_text(patched)
    return {'original_sha256': original_digest, 'wheel_sha256': wheel_digest,
            'contract': str(contract_path),
            'method_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in methods.items()},
            'patched_sha256': hashlib.sha256(patched.encode()).hexdigest()}


def weights_iterator(hf_weights_files: list[str], disable_mmap: bool = False,
                     prefetch: bool = False, prefetch_num_threads: int = 4,
                     drop_cache_after_load: bool = False):
    """At most manifest.cache_slots files across all managed model iterators."""
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
    # processes cannot exceed the aggregate slot/byte bound.
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
            result = stage_shard(item, cache,
                read_workers=4 if os.getenv('ENABLE_PARALLEL_SHARD_READS', '0') == '1' else 1)
            destination = cache / relative
            with destination.open('rb') as stream:
                local_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
            if local_hash != result['source_stream_sha256']:
                raise ValueError('local shard digest differs from copied source stream')
            result['verified_sha256'] = local_hash
            result['copy_seconds'] = round(time.monotonic()-started, 3)
            return result

        try:
            slots = plan['cache_slots']
            if slots not in (2, 3):
                raise ValueError('invalid bounded cache slots')
            with ThreadPoolExecutor(max_workers=slots) as pool:
                remaining = iter(entries)
                pending = deque()
                for _ in range(slots):
                    item = next(remaining, None)
                    if item is not None:
                        pending.append((item, pool.submit(copy, item)))
                save(status='copying', active_shards=[item[1].name for item, _ in pending])
                while pending:
                    item, future = pending.popleft()
                    result = future.result()  # Complete copy/fsync/digest barrier.
                    current = cache / item[1]
                    # Count every in-flight file's complete size, including
                    # producers still writing, before opening the current one.
                    active_bytes = item[2] + sum(entry[2] for entry, _ in pending)
                    if active_bytes > plan['cache_bound_bytes']:
                        raise ValueError('bounded cache size invariant violated')
                    high = max(report['cache_high_water_bytes'], active_bytes)
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
                    # Unlink before replenishing: a consumed file and its
                    # replacement must never occupy an extra disk slot.
                    item = next(remaining, None)
                    if item is not None:
                        pending.append((item, pool.submit(copy, item)))
            save(status='iterator_complete', current_shard=None)
        except BaseException as error:
            save(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            # Pool shutdown joins every writer before cancellation cleanup.
            for path in cache.iterdir():
                path.unlink()
