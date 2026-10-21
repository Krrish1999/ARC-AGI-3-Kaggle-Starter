"""Extract the retained baseline's serving code without executing the notebook."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DATASETS = {
    'MODEL_DIR': 'mirzamilanfarabi/intel-qwen3-8-flash-next-w4a16-autoround',
    'DRAFT_MODEL_DIR': 'mirzamilanfarabi/albucino-qwen3-8-flash-next-drafter',
}


def extract_launcher(*, local_model_root: str = '/tmp/models/arc3',
                     local_model_fallback_roots: tuple[str, ...] = (),
                     local_model_headroom_bytes: int = 16 * 1024**3) -> tuple[str, dict]:
    path = ROOT / 'research_lab/top_programs/baseline_28_score.ipynb'
    raw = path.read_bytes()
    expected = json.loads((ROOT / 'agent/baseline/manifest.json').read_text())['notebook_sha256']
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Retained baseline notebook hash changed')
    notebook = json.loads(raw)
    setup = ''.join(notebook['cells'][5]['source'])
    cache_setup = ''.join(notebook['cells'][7]['source'])
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
            name = node.targets[0].id if isinstance(node, ast.Assign) else None
            if name in WEIGHT_DATASETS:
                owner, slug = WEIGHT_DATASETS[name].split('/')
                candidates = [f'/kaggle/input/{slug}', f'/kaggle/input/datasets/{owner}/{slug}',
                              f'/kaggle/input/{owner}/{slug}']
                extracted.append(f'{name} = resolve_path({candidates!r})')
            else:
                extracted.append(ast.get_source_segment(setup, node))
    header = ('import os, time\nfrom pathlib import Path\n'
              'WORKING_DIR = Path(os.environ["ARC3_LLM_WORKING"])\n'
              'SERVED_MODEL_PORT = 8001\n' + '\n\n'.join(extracted) + '\n')
    cache_function = next(node for node in ast.parse(cache_setup).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'precache')
    cache_source = ast.get_source_segment(cache_setup, cache_function)
    assert cache_source is not None
    if cache_source.count('except: pass') != 2:
        raise ValueError('Baseline precache error boundaries changed')
    cache_source = cache_source.replace('except: pass',
        'except OSError as error: print(f"Prefetch skipped: {error}", flush=True)')
    prefetch = ('\nimport threading\nfrom concurrent.futures import ThreadPoolExecutor\n'
                + cache_source + '\n'
                'precache_model_thread = None\n'
                'if os.getenv("ENABLE_BASELINE_PREFETCH", "1") == "1":\n'
                '    precache(WHEELHOUSE_DIR)\n'
                '    if os.getenv("ENABLE_LOCAL_MODEL_STAGING", "1") != "1":\n'
                '        precache_model_thread = threading.Thread(target=precache,\n'
                '            args=(MODEL_DIR, DRAFT_MODEL_DIR) if os.getenv("ENABLE_SPECULATIVE_DRAFT", "1") == "1" else (MODEL_DIR,),\n'
                '            kwargs={"delay": 60, "threads": 3}, daemon=True)\n')
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
        '"--model-path", MODEL_DIR,': '"--model-path", str(LOCAL_MODEL_DIR),',
    }
    for before, after in edits.items():
        if serving.count(before) != 1:
            raise ValueError(f'Baseline launcher fragment changed: {before}')
        serving = serving.replace(before, after)
    # Keep install + server in the supervisor's process group. The parent owns
    # the single bounded readiness deadline, termination and reaping.
    tail = ('\nif precache_model_thread is not None:\n    precache_model_thread.start()\n'
            'print("SGLang launch:", args, flush=True)\n'
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
    staging_source = (ROOT / 'scripts/model_staging.py').read_text().replace('from __future__ import annotations\n', '')
    staging = (
        '\n# Blocking checkpoint-copy barrier: publish no launch arguments before completion.\n'
        'if os.getenv("ENABLE_LOCAL_MODEL_STAGING", "1") == "1":\n'
        '    staging_report = stage_checkpoints(Path(asset_report["target"]["path"]),\n'
        '        Path(asset_report["draft"]["path"]) if "draft" in asset_report else None,\n'
        f'        Path({local_model_root!r}), WORKING_DIR / "diagnostics/model_staging.json",\n'
        '        reserve=os.getenv("ENABLE_STAGING_CAPACITY_RESERVATION", "1") == "1",\n'
        f'        fallback_roots=tuple(Path(path) for path in {local_model_fallback_roots!r}),\n'
        f'        headroom_bytes={local_model_headroom_bytes!r},\n'
        '        owner_token=os.environ["ARC3_MODEL_OWNER"])\n'
        '    for kind, model in staging_report["models"].items():\n'
        '        local_report = checkpoint_report(Path(model["path"]))\n'
        '        for field in ("config_sha256", "index_sha256", "shard_sizes"):\n'
        '            if local_report[field] != asset_report[kind][field]:\n'
        '                raise ValueError("Staged checkpoint verification failed: " + kind + ":" + field)\n'
        '    MODEL_DIR = staging_report["models"]["target"]["path"]\n'
        '    DRAFT_MODEL_DIR = staging_report["models"].get("draft", {}).get("path", DRAFT_MODEL_DIR)\n'
        'LOCAL_MODEL_DIR = Path(MODEL_DIR)\n')
    source = header + prefetch + assets_source + inspection + staging_source + staging + serving + tail
    compile(source, '<extracted-pennyroyal-launcher>', 'exec')
    provenance = {
        'baseline_notebook_sha256': expected,
        'baseline_cells': {str(i): hashlib.sha256(''.join(notebook['cells'][i]['source']).encode()).hexdigest()
                           for i in (5, 7, 13)},
        'launcher_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'wheelhouse': 'dfranzen/pennyroyal-v253', 'dataset_version_observed': 1,
        'weight_datasets': WEIGHT_DATASETS,
        'pennyroyal_release': 'v2.5.3',
        'upstream_commit': 'd00d88efc8d6281b12be4f4073126aec95038c55',
        'sglang_wheel': 'sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl',
        'served_model': 'flashnext', 'startup_seconds': 2400,
        'prefetch': 'Cell 7 wheel prewarm (16 readers); models copied with a blocking 4-worker barrier before installation/launch. Original model page-cache prefetch is an explicit staging-off control only.',
        'local_model_staging': {'root': local_model_root, 'toggle': 'ENABLE_LOCAL_MODEL_STAGING',
                                'default': '1', 'workers': 4, 'copy_barrier': 'all files closed/fsynced; target and draft validated before args/Popen',
                                'capacity_toggle': 'ENABLE_STAGING_CAPACITY_RESERVATION',
                                'capacity_default': '1', 'fallback_roots': list(local_model_fallback_roots),
                                'headroom_bytes': local_model_headroom_bytes,
                                'reservation': 'posix_fallocate and fsync every destination file plus headroom before FUSE reads; retain extents during copy'},
        'radix_cache': 'enabled by server default; schedule-policy=lpm; cache reporting enabled',
        'configuration_edits': edits,
    }
    return source, provenance
