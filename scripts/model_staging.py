"""Materialize complete model checkpoints on local disk before server startup.

This stdlib-only source is embedded in the notebook launcher. A completed copy
is published only after every worker has closed and fsynced its output. The
caller never receives a partial checkpoint or a symlink back to input storage.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _staging_files(source: Path) -> tuple[list[tuple[Path, Path, int]], int]:
    source = source.resolve(strict=True)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if not shards or not (source / 'config.json').is_file():
        raise ValueError(f'incomplete checkpoint: {source}')
    for name in shards:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'unsafe shard path: {name}')
        if not (source / relative).is_file() or (source / relative).stat().st_size < 8:
            raise FileNotFoundError(f'missing checkpoint shard: {name}')
    files = []
    for parent, directories, names in os.walk(source, followlinks=False):
        if any((Path(parent) / name).is_symlink() for name in directories):
            raise ValueError('model staging cannot follow directory symlinks')
        for name in sorted(names):
            path = Path(parent) / name
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(source) or not resolved.is_file():
                raise ValueError(f'checkpoint file escapes source: {path}')
            files.append((path, path.relative_to(source), path.stat().st_size))
    return sorted(files), len(shards)


def _stage_file(item: tuple[Path, Path, int], destination: Path) -> dict:
    source, relative, expected = item
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    # r+b preserves allocated extents; opening with wb would release them.
    reserved = target.exists()
    if reserved and (target.is_symlink() or target.stat().st_size != expected):
        raise OSError(f'invalid reserved checkpoint file: {relative}')
    with source.open('rb') as reader, target.open('r+b' if reserved else 'xb') as writer:
        while block := reader.read(8 * 1024**2):
            writer.write(block)
            digest.update(block)
            count += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if count != expected or target.stat().st_size != expected:
        raise OSError(f'incomplete local copy: {relative}')
    return {'name': str(relative), 'bytes': count, 'source_stream_sha256': digest.hexdigest()}


def _storage_profile(root: Path) -> dict:
    """Record the actual mount and inode view, not only host-wide free bytes."""
    stats = os.statvfs(root)
    profile = {'device': root.stat().st_dev, 'free_inodes': stats.f_favail, 'total_inodes': stats.f_files,
               'read_only': bool(stats.f_flag & os.ST_RDONLY),
               'block_size': stats.f_frsize, 'disk_free_before': shutil.disk_usage(root).free}
    mountinfo = Path('/proc/self/mountinfo')
    if mountinfo.is_file():
        for line in mountinfo.read_text().splitlines():
            fields = line.split()
            mount = Path(fields[4].replace(r'\040', ' ').replace(r'\134', '\\'))
            if root.is_relative_to(mount) and len(str(mount)) >= len(profile.get('mount', '')):
                profile.update(mount=str(mount), filesystem=fields[fields.index('-')+1])
    return profile


def _reserve_file(path: Path, size: int) -> int:
    """Allocate real extents and sync now, before reading a FUSE weight shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        if size:
            os.posix_fallocate(stream.fileno(), 0, size)
        os.fsync(stream.fileno())
        allocated = os.fstat(stream.fileno()).st_blocks * 512
        if allocated < size:
            raise OSError(errno.ENOSPC, f'checkpoint reservation is sparse: {path.name}')
    return allocated


def stage_checkpoints(target: Path, draft: Path | None, root: Path,
                      report_path: Path, *, workers: int = 4, reserve: bool = False,
                      fallback_roots: tuple[Path, ...] = (),
                      headroom_bytes: int = 16 * 1024**3, owner_token: str = '') -> dict:
    """Blocking copy barrier for all checkpoints; any failure prevents launch.

    Use a fresh owned transaction directory, never adopt a partial previous run.
    Hashes are observations of copied bytes, not independent upstream signatures.
    The supervisor's unchanged global startup deadline bounds the whole process.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('staging needs 1..8 workers')
    if type(headroom_bytes) is not int or headroom_bytes < 0:
        raise ValueError('staging headroom must be nonnegative bytes')
    began = time.monotonic()
    sources = {'target': target.resolve(strict=True)}
    if draft is not None:
        sources['draft'] = draft.resolve(strict=True)
    roots = list(dict.fromkeys(path.resolve() for path in (root, *fallback_roots))) if reserve else [root.resolve()]
    for candidate in roots:
        if candidate.is_relative_to(Path('/kaggle/input')):
            raise ValueError('local checkpoint staging cannot use /kaggle/input')
        if any(candidate.is_relative_to(source) for source in sources.values()):
            raise ValueError('staging destination cannot be inside a source checkpoint')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'reserving' if reserve else 'copying', 'mode': 'local_copy',
              'root': str(roots[0]), 'workers': workers, 'models': {},
              'capacity_reservation': reserve, 'storage_attempts': [],
              'headroom_bytes': headroom_bytes if reserve else 512 * 1024**2}

    def save() -> None:
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(report_path)

    def cleanup(path: Path, attempt: dict) -> None:
        try:
            shutil.rmtree(path)
        except OSError as error:
            # Preserve the copy error and allow independent mounts to be tried.
            # A read-only mount cannot be repaired by retrying deletion here.
            attempt['cleanup_error'] = {'errno': error.errno, 'error': str(error)}
            report.setdefault('orphaned_transactions', []).append(str(path))

    transaction: Path | None = None
    save()
    try:
        inventories = {kind: _staging_files(source) for kind, source in sources.items()}
        required = sum(size for files, _ in inventories.values() for _, _, size in files)
        report['bytes_required'] = required
        for candidate in roots:
            attempt: dict = {'root': str(candidate), 'status': 'checking'}
            report['storage_attempts'].append(attempt)
            report.update(root=str(candidate), models={}, status='reserving' if reserve else 'copying')
            save()
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                attempt.update(_storage_profile(candidate))
                report['disk_free_before'] = attempt['disk_free_before']
                filesystem = attempt.get('filesystem', '')
                if attempt.get('read_only'):
                    raise OSError(errno.EROFS, 'checkpoint candidate is read-only')
                if reserve and (filesystem.startswith('fuse') or filesystem in ('tmpfs', 'ramfs', 'nfs', 'nfs4', 'cifs', '9p')):
                    raise OSError(errno.ENOTSUP, f'checkpoint storage is not local disk: {filesystem}')
                if attempt['disk_free_before'] < required + report['headroom_bytes']:
                    raise OSError(errno.ENOSPC, f'insufficient local model storage: need {required} bytes plus headroom')
                if attempt.get('total_inodes', 1) > 0 and attempt['free_inodes'] < sum(len(files) for files, _ in inventories.values()) + 16:
                    raise OSError(errno.ENOSPC, 'insufficient local model inodes')
                transaction = Path(tempfile.mkdtemp(prefix='run-', dir=candidate))
                (transaction / '.arc3-owner').write_text(owner_token)
                report['transaction'] = str(transaction)
                save()
                for kind in inventories:
                    (transaction / (kind+'.copying')).mkdir()
                if reserve:
                    reserved_at = time.monotonic()
                    attempt.update(status='reserving', allocated_bytes=0, reserved_files=0)
                    save()
                    for kind, (files, _) in inventories.items():
                        for _, relative, size in files:
                            attempt['allocated_bytes'] += _reserve_file(transaction / (kind+'.copying') / relative, size)
                            attempt['reserved_files'] += 1
                    attempt['allocated_bytes'] += _reserve_file(transaction / 'headroom.reserve', headroom_bytes)
                    attempt.update(status='reserved', reservation_seconds=round(time.monotonic()-reserved_at, 3))
                    report['status'] = 'reserved'
                    save()
                report['status'] = 'copying'
                save()
                for kind, (files, shards) in inventories.items():
                    started = time.monotonic()
                    attempt.update(copying_kind=kind, copied_files=0, copied_bytes=0)
                    partial = transaction / (kind+'.copying')
                    # Resolve every future and join all workers before publishing.
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(_stage_file, item, partial) for item in files]
                        try:
                            copied = []
                            for future in as_completed(futures):
                                file = future.result()
                                copied.append(file)
                                attempt['copied_files'] += 1
                                attempt['copied_bytes'] += file['bytes']
                                if len(copied) % 4 == 0 or len(copied) == len(files):
                                    save()
                        except BaseException:
                            for future in futures:
                                future.cancel()
                            raise
                    copied.sort(key=lambda file: file['name'])
                    final = transaction / kind
                    partial.rename(final)
                    manifest = transaction / (kind+'.manifest.json')
                    manifest.write_text(json.dumps(copied, indent=2)+'\n')
                    report['models'][kind] = {'source': str(sources[kind]), 'path': str(final),
                        'file_count': len(copied), 'shard_count': shards,
                        'bytes': sum(file['bytes'] for file in copied),
                        'copy_seconds': round(time.monotonic()-started, 3), 'manifest': str(manifest)}
                    save()
                    print(f'Local checkpoint complete: {kind}: {shards} shards, '
                          f'{report["models"][kind]["bytes"]} bytes -> {final}', flush=True)
                # Release only our headroom reservation for serving installation.
                (transaction / 'headroom.reserve').unlink(missing_ok=True)
                attempt['status'] = 'complete'
                report.update(status='complete', elapsed_seconds=round(time.monotonic()-began, 3))
                save()
                return report
            except OSError as error:
                attempt.update(status='failed', error_type=type(error).__name__, error=str(error), errno=error.errno)
                if transaction is not None:
                    cleanup(transaction, attempt)
                    transaction = None
                report.pop('transaction', None)
                report['models'] = {}
                save()
                if not reserve or error.errno not in (errno.ENOSPC, errno.EDQUOT, errno.ENOTSUP, errno.EACCES, errno.EROFS):
                    raise
        raise OSError(errno.ENOSPC, 'no local storage candidate could stage the complete checkpoint set')
    except BaseException as error:
        if transaction is not None:
            cleanup(transaction, attempt)
        report.pop('transaction', None)
        report.update(status='failed', models={}, error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=round(time.monotonic()-began, 3))
        save()
        raise
