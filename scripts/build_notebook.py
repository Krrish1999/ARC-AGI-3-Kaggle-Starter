"""Bundle deployed modules into an offline RTX 6000 submission notebook.

The payload contains both agent entry modules and reusable support modules,
verifies their SHA256 hashes, runs behavioral containment before evaluation,
and bounds the framework lifetime against a shared 29,700-second run target
and 30,600-second hard ceiling, with an early fallback submission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent, indent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "rtx6000"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    sources = sorted((ROOT / "agent").rglob("*.py"))
    for required in ("my_agent.py", "arclog.py", "runtime.py", "sandbox.py"):
        if not (ROOT / "agent" / required).is_file():
            raise SystemExit(f"Missing bundled module: {required}")
    write_cells = [code_cell(
        "%%writefile /tmp/agent_bundle/" + path.relative_to(ROOT / "agent").as_posix()
        + "\n" + path.read_text()) for path in sources]

    source_hashes = {path.relative_to(ROOT / "agent").as_posix():
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    verification_cell = code_cell(
        "import hashlib\nfrom pathlib import Path\n"
        + "expected = " + repr(source_hashes) + "\n"
        + "for name, digest in expected.items():\n"
        + "    assert hashlib.sha256((Path('/tmp/agent_bundle') / name).read_bytes()).hexdigest() == digest, name\n"
    )
    audit_cell = code_cell("%%writefile /tmp/submission_audit.py\n" +
                           (ROOT / "scripts/audit_containment.py").read_text().replace(
                               'ROOT = Path(__file__).resolve().parents[1]',
                               "ROOT = Path('/tmp/agent_preflight')"))

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    setup_cell = code_cell(dedent("""\
        import os
        import time
        from pathlib import Path
        # Capture a fresh notebook-wide start before installation or evaluation.
        start_time = time.time()
        os.environ['ARC3_RUN_STARTED'] = str(start_time)
        import pandas as pd
        submission_path = Path('/kaggle/working/submission.parquet')
        if not submission_path.exists():
            # Also runs on the hidden competition rerun, before any agent code.
            # Never overwrite an existing gateway result on cell re-execution.
            placeholder = pd.DataFrame(
                data=[['1_0', '1', True, 0]],
                columns=['row_id', 'game_id', 'end_of_game', 'score'])
            temporary = submission_path.with_suffix('.parquet.tmp')
            placeholder.to_parquet(temporary, index=False)
            temporary.replace(submission_path)
        for name in ('ENABLE_SANDBOX_LIMITS', 'ENABLE_PREFIX_LOCK', 'ENABLE_LEVEL_BUDGET',
                     'ENABLE_ACTION_FALLBACK'):
            os.environ.setdefault(name, '1')
        Path('/tmp/agent_bundle/baseline').mkdir(parents=True, exist_ok=True)
    """))

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp -r /tmp/agent_bundle/. \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/

            # Engine-free planner interpreter; seccomp support is mandatory at
            # worker startup. No system-site-packages or online installation.
            import subprocess
            import sys
            subprocess.run([sys.executable, '-m', 'venv',
                            '/kaggle/working/.venv_agent_sandbox'], check=True)
            os.environ['ARC3_SANDBOX_PYTHON'] = '/kaggle/working/.venv_agent_sandbox/bin/python'
            from pathlib import Path
            preflight = Path('/tmp/agent_preflight')
            preflight.mkdir(exist_ok=True)
            for name, target in [('agent', '/tmp/agent_bundle'),
                                 ('.venv_agent_sandbox', '/kaggle/working/.venv_agent_sandbox')]:
                link = preflight / name
                if not link.is_symlink():
                    link.symlink_to(target, target_is_directory=True)
            # Behavioral Gate 1 must pass before live dispatch.
            subprocess.run([sys.executable, '/tmp/submission_audit.py'], check=True)

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(
                    'from typing import Type\\n'
                    'from dotenv import load_dotenv\\n'
                    'from .agent import Agent, Playback\\n'
                    'from .swarm import Swarm\\n'
                    'from .templates.random_agent import Random\\n'
                    'from .templates.my_agent import MyAgent\\n'
                    '\\n'
                    'load_dotenv()\\n'
                    '\\n'
                    'AVAILABLE_AGENTS: dict[str, Type[Agent]] = {\\n'
                    "    'random': Random,\\n"
                    "    'myagent': MyAgent,\\n"
                    '}\\n'
                )

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(
                    'SCHEME=http\\n'
                    'HOST=gateway\\n'
                    'PORT=8001\\n'
                    'ARC_API_KEY=test-key-123\\n'
                    'ARC_BASE_URL=http://gateway:8001/\\n'
                    'OPERATION_MODE=online\\n'
                    'ENVIRONMENTS_DIR=\\n'
                    'RECORDINGS_DIR=/kaggle/working/server_recording\\n'
                )

            # Include all setup in the shared clock. SIGINT invokes the
            # framework's cleanup(), which closes the gateway scorecard and
            # lets it replace the early parquet with the evaluated submission.
            import signal
            import time
            start_time = float(os.environ['ARC3_RUN_STARTED'])
            target_seconds = 29_700
            hard_limit_seconds = 30_600
            if time.time() - start_time < target_seconds:
                run_env = dict(os.environ, MPLBACKEND='agg')
                process = subprocess.Popen(
                    [sys.executable, 'main.py', '--agent', 'myagent'],
                    cwd='/kaggle/working/ARC-AGI-3-Agents', env=run_env,
                    start_new_session=True,
                )
                try:
                    while process.poll() is None:
                        elapsed = time.time() - start_time
                        if elapsed >= hard_limit_seconds or elapsed >= target_seconds:
                            print('Global runtime limit reached; finalizing the gateway session.')
                            break
                        try:
                            process.wait(timeout=min(1.0, target_seconds - elapsed))
                        except subprocess.TimeoutExpired:
                            continue
                finally:
                    try:
                        if process.poll() is None:
                            # The runner's SIGINT handler closes the scorecard.
                            # Leave worker threads alone during finalization.
                            try:
                                process.send_signal(signal.SIGINT)
                            except ProcessLookupError:
                                pass
                            remaining = max(0.0, hard_limit_seconds - (time.time() - start_time))
                            try:
                                process.wait(timeout=min(60.0, remaining))
                            except subprocess.TimeoutExpired:
                                print('Framework cleanup timed out; retaining the submission on disk.')
                    finally:
                        # Reap descendants even if signalling/cleanup fails.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                if process.returncode:
                    print(f'Framework exited with {process.returncode}; retaining submission.parquet.')
            else:
                print('Run budget exhausted during setup; retaining the early submission.')

        """
    )
    run_cell = code_cell(
        "try:\n" + indent(run_cell_source, "    ")
        + "except Exception:\n"
        + "    import traceback\n"
        + "    from pathlib import Path\n"
        + "    detail = traceback.format_exc()\n"
        + "    traceback.print_exc()\n"
        + "    try:\n"
        + "        Path('scratch').mkdir(exist_ok=True)\n"
        + "        Path('scratch/runner_errors.log').write_text(detail)\n"
        + "    except OSError:\n"
        + "        print('Unable to save runner traceback to scratch/.')\n"
        + "    print('Runner failed; retaining the early submission.parquet.')\n"
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py`, `agent/arclog.py` and support modules via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            setup_cell,
            install_cell,
            *write_cells,
            verification_cell,
            audit_cell,
            run_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
