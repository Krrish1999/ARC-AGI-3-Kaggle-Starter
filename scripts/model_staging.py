"""Materialize complete model checkpoints on local disk before server startup.

This stdlib-only source is embedded in the notebook launcher. A completed copy
is published only after every worker has closed and fsynced its output. The
caller never receives a partial checkpoint or a symlink back to input storage.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
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
    with source.open('rb') as reader, target.open('xb') as writer:
        while block := reader.read(8 * 1024**2):
            writer.write(block)
            digest.update(block)
            count += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    if count != expected or target.stat().st_size != expected:
        raise OSError(f'incomplete local copy: {relative}')
    return {'name': str(relative), 'bytes': count, 'source_stream_sha256': digest.hexdigest()}


def stage_checkpoints(target: Path, draft: Path | None, root: Path,
                      report_path: Path, *, workers: int = 4) -> dict:
    """Blocking copy barrier for all checkpoints; any failure prevents launch.

    Use a fresh owned transaction directory, never adopt a partial previous run.
    Hashes are observations of copied bytes, not independent upstream signatures.
    The supervisor's unchanged global startup deadline bounds the whole process.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError('staging needs 1..8 workers')
    began = time.monotonic()
    root = root.resolve()
    if root.is_relative_to(Path('/kaggle/input')):
        raise ValueError('local checkpoint staging cannot use /kaggle/input')
    sources = {'target': target.resolve(strict=True)}
    if draft is not None:
        sources['draft'] = draft.resolve(strict=True)
    if any(root.is_relative_to(source) for source in sources.values()):
        raise ValueError('staging destination cannot be inside a source checkpoint')
    root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'copying', 'mode': 'local_copy', 'root': str(root),
              'workers': workers, 'models': {}}

    def save() -> None:
        temporary = report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(report_path)

    transaction: Path | None = None
    save()
    try:
        inventories = {kind: _staging_files(source) for kind, source in sources.items()}
        required = sum(size for files, _ in inventories.values() for _, _, size in files)
        free = shutil.disk_usage(root).free
        report.update(bytes_required=required, disk_free_before=free)
        save()
        if free < required + 512 * 1024**2:
            raise OSError(f'insufficient local model storage: need {required} bytes plus 512 MiB; free={free}')
        transaction = Path(tempfile.mkdtemp(prefix='run-', dir=root))
        for kind, (files, shards) in inventories.items():
            started = time.monotonic()
            partial = transaction / (kind+'.copying')
            partial.mkdir()
            # Resolve every future and join all workers before publishing.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_stage_file, item, partial) for item in files]
                try:
                    copied = [future.result() for future in futures]
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
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
        report.update(status='complete', elapsed_seconds=round(time.monotonic()-began, 3))
        save()
        return report
    except BaseException as error:
        if transaction is not None:
            shutil.rmtree(transaction, ignore_errors=True)
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=round(time.monotonic()-began, 3))
        save()
        raise
