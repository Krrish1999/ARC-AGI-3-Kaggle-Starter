"""Decode notebook patches as data; extract complete new modules without exec.

Modified upstream files need the external source bundle. Never concatenate hunk
fragments into a purported implementation. Preserve both patches and all expected
Dhia hashes so the full upstream application can later be verified.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import lzma
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "research_lab/top_programs/baseline_28_score.ipynb"
DEST = ROOT / "agent/baseline"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract() -> dict:
    notebook_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    original = ''.join(notebook['cells'][3]['source']).split('\n', 1)[1]
    cell = ''.join(notebook['cells'][5]['source'])
    literals = {}
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {'_DHIA_PATCH_B64', '_DHIA_SRC_HASHES'}:
                    literals[name.id] = ast.literal_eval(node.value)
    overlay = lzma.decompress(base64.b64decode(literals['_DHIA_PATCH_B64'], validate=True)).decode()
    if (DEST / 'manifest.json').exists():
        manifest = json.loads((DEST / 'manifest.json').read_text())
        if manifest['notebook_sha256'] != digest(notebook_bytes):
            raise ValueError("parent notebook hash changed")
        for label, data in [('cell3', original), ('dhia_0_1_21', overlay)]:
            if (DEST / (label + '.patch')).read_text() != data:
                raise ValueError(f"patch extraction changed: {label}")
        for name, entry in manifest['modules'].items():
            expected = entry.get('ported_sha256', entry['extracted_sha256'])
            if digest((DEST / name).read_bytes()) != expected:
                raise ValueError(f"module hash changed: {name}")
        return manifest
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / '__init__.py').write_text('"""Inspectable baseline mechanisms; see manifest.json for provenance and gaps."""\n')
    manifest: dict = {'notebook_sha256': digest(notebook_bytes), 'patches': {},
                      'modules': {}, 'requires_upstream_base': [],
                      'expected_dhia_source_hashes': literals['_DHIA_SRC_HASHES']}
    for label, patch in [('cell3', original), ('dhia_0_1_21', overlay)]:
        (DEST / (label + '.patch')).write_text(patch)
        manifest['patches'][label] = digest(patch.encode())
        for block in re.split(r'(?m)^diff ', patch)[1:]:
            header = next(line for line in block.splitlines() if line.startswith('+++ '))
            path = header.split('\t')[0][4:].split('/inference/', 1)[1]
            if not path.endswith('.py'):
                continue
            if not re.search(r'(?m)^@@ -0,0 \+1,\d+ @@', block):
                manifest['requires_upstream_base'].append({'patch': label, 'path': path})
                continue
            source = ''.join(line[1:] for line in block.splitlines(keepends=True)
                             if line.startswith('+') and not line.startswith('+++'))
            ast.parse(source)
            name = Path(path).name
            (DEST / name).write_text(source)
            manifest['modules'][name] = {'patch': label, 'upstream': path,
                                        'original_sha256': digest(source.encode())}
    # The archived animation utility imports grid_utils, which is not present in
    # either patch. Provide its two tiny display dependencies in an explicit port.
    animation = DEST / 'animation.py'
    src = animation.read_text().replace(
        'from inference.utils.grid_utils import ARC_COLOR_CHARS, format_grid_ascii',
        'from .grid_display import ARC_COLOR_CHARS, format_grid_ascii')
    animation.write_text(src)
    for name, entry in manifest['modules'].items():
        entry['extracted_sha256'] = digest((DEST / name).read_bytes())
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    result = extract()
    print(f"Extracted {len(result['modules'])} complete modules; upstream gaps recorded in manifest.json")
