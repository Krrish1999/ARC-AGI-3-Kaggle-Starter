"""Inspect versioned offline inputs before the extracted launcher installs them."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

SGLANG_WHEEL = 'sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl'


def asset_digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_report(path: Path) -> dict:
    index_path = path / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    names = sorted(set(index['weight_map'].values()))
    if not names:
        raise ValueError('empty model shard index')
    shards = {}
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe shard path')
        shard = path / relative
        if not shard.is_file() or shard.stat().st_size < 8:
            raise FileNotFoundError(f'missing or empty weight shard: {shard}')
        shards[name] = shard.stat().st_size
    return {'path': str(path), 'config_sha256': asset_digest(path / 'config.json'),
            'index_sha256': asset_digest(index_path), 'shard_sizes': shards,
            'shard_content_hashes': 'not independently supplied; Kaggle input provenance is recorded separately'}


def verify_offline_assets(wheelhouse: Path, target: Path, draft: Path | None) -> dict:
    wheel_dirs = [wheelhouse / 'wheels'] if (wheelhouse / 'wheels').is_dir() else list(wheelhouse.rglob('wheels'))
    if len(wheel_dirs) != 1:
        raise FileNotFoundError('expected one offline wheels directory')
    wheels = wheel_dirs[0]
    locks = list(wheelhouse.rglob('requirements.lock'))
    if len(locks) != 1 or not (wheels / SGLANG_WHEEL).is_file():
        raise FileNotFoundError('missing lock or pinned Pennyroyal wheel')
    files = sorted(wheels.glob('*.whl'))
    if len(list(wheels.glob('uv-*.whl'))) != 1:
        raise FileNotFoundError('missing or ambiguous bundled uv wheel')
    hashes = {}
    for path in files:
        if not zipfile.is_zipfile(path):
            raise ValueError(f'invalid wheel archive: {path.name}')
        hashes[path.name] = asset_digest(path)
    config = json.loads((target / 'config.json').read_text())
    quantization = config.get('quantization_config', {})
    if quantization.get('quant_method') != 'auto-round' or quantization.get('bits') != 4:
        raise ValueError('target is not the pinned AutoRound INT4 format')
    report = {'wheelhouse': str(wheelhouse), 'wheel_sha256_observed': hashes,
              'requirements_sha256': asset_digest(locks[0]), 'target': checkpoint_report(target),
              'identity': 'versioned Kaggle inputs; observed hashes are recorded, not invented upstream signatures'}
    if draft is not None:
        candidates = []
        for path in draft.rglob('config.json'):
            quant = json.loads(path.read_text()).get('quantization_config', {})
            if quant.get('quant_method') == 'compressed-tensors' and 'mtp_routed_experts' in quant.get('config_groups', {}):
                candidates.append(path.parent)
        if len(candidates) != 1:
            raise ValueError('expected one compatible compressed-tensors draft checkpoint')
        report['draft'] = checkpoint_report(candidates[0])
    return report
