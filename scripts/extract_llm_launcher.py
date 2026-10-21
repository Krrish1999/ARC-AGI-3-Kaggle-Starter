"""Extract the retained baseline's serving code without executing the notebook."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract_launcher() -> tuple[str, dict]:
    path = ROOT / 'research_lab/top_programs/baseline_28_score.ipynb'
    raw = path.read_bytes()
    expected = json.loads((ROOT / 'agent/baseline/manifest.json').read_text())['notebook_sha256']
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Retained baseline notebook hash changed')
    notebook = json.loads(raw)
    setup = ''.join(notebook['cells'][5]['source'])
    serving = ''.join(notebook['cells'][13]['source'])
    # Reuse mount resolution and identities, not the other harness's pickle,
    # patches, policy settings or early-game startup bypass.
    wanted = {'WHEELHOUSE_DIR', 'MODEL_DIR', 'DRAFT_MODEL_DIR', 'SERVED_MODEL_NAME',
              'SERVED_MODEL_PORT'}
    extracted = []
    for node in ast.parse(setup).body:
        if ((isinstance(node, ast.FunctionDef) and node.name == 'resolve_path')
                or (isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets))):
            extracted.append(ast.get_source_segment(setup, node))
    header = ('import os, time\nfrom pathlib import Path\n'
              'WORKING_DIR = Path(os.environ["ARC3_LLM_WORKING"])\n'
              'SERVED_MODEL_PORT = 8001\n' + '\n\n'.join(extracted) + '\n')
    marker = '# ---- launch detached and wait for health ----'
    if serving.count(marker) != 1:
        raise ValueError('Baseline serving boundary changed')
    serving = serving.split(marker)[0]
    edits = {
        'PREFIX = "/tmp/sgl-intel"': 'PREFIX = os.environ["ARC3_LLM_PREFIX"]',
        'CTX=(116+12+8)*1024': 'CTX=65536',
        'MEMFRAC=0.96': 'MEMFRAC=0.88',
        'MAXREQ=10': 'MAXREQ=4',
        'CUDAGRAPH_MAXBS=10': 'CUDAGRAPH_MAXBS=4',
        'MAMBA_CACHE=60': 'MAMBA_CACHE=16',
        '    SPEC=True,': '    SPEC=os.getenv("ENABLE_SPECULATIVE_DRAFT", "1") == "1",',
        '"--no-index", "--find-links", str(WHEELS)':
            '"--no-index", "--only-binary", ":all:", "--find-links", str(WHEELS)',
    }
    for before, after in edits.items():
        if serving.count(before) != 1:
            raise ValueError(f'Baseline launcher fragment changed: {before}')
        serving = serving.replace(before, after)
    # Keep install + server in the supervisor's process group. The parent owns
    # the single bounded readiness deadline, termination and reaping.
    tail = ('\nprint("SGLang launch:", args, flush=True)\n'
            'with open(LOG, "ab", buffering=0) as logf:\n'
            '    proc = subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT)\n'
            'raise SystemExit(proc.wait())\n')
    assets_source = (ROOT / 'scripts/llm_assets.py').read_text().replace('from __future__ import annotations\n', '')
    inspection = (
        '\nasset_report = verify_offline_assets(Path(WHEELHOUSE_DIR), Path(MODEL_DIR),\n'
        '    Path(DRAFT_MODEL_DIR) if os.getenv("ENABLE_SPECULATIVE_DRAFT", "1") == "1" else None)\n'
        'asset_report_path = WORKING_DIR / "diagnostics/llm_assets.json"\n'
        'asset_report_path.parent.mkdir(parents=True, exist_ok=True)\n'
        'asset_report_path.write_text(json.dumps(asset_report, indent=2) + "\\n")\n')
    source = header + assets_source + inspection + serving + tail
    compile(source, '<extracted-pennyroyal-launcher>', 'exec')
    provenance = {
        'baseline_notebook_sha256': expected,
        'baseline_cells': {str(i): hashlib.sha256(''.join(notebook['cells'][i]['source']).encode()).hexdigest()
                           for i in (5, 13)},
        'launcher_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'wheelhouse': 'dfranzen/pennyroyal-v253', 'dataset_version_observed': 1,
        'pennyroyal_release': 'v2.5.3',
        'upstream_commit': 'd00d88efc8d6281b12be4f4073126aec95038c55',
        'sglang_wheel': 'sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl',
        'served_model': 'flashnext', 'startup_seconds': 900,
        'radix_cache': 'enabled by server default; schedule-policy=lpm; cache reporting enabled',
        'configuration_edits': edits,
    }
    return source, provenance
