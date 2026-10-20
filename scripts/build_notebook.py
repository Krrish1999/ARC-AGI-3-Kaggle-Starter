"""Bundle deployed modules into an offline RTX 6000 submission notebook.

The payload contains both agent entry modules and reusable support modules,
verifies their SHA256 hashes, runs behavioral containment before evaluation,
and bounds the framework lifetime against a shared 29,700-second run target
and 30,600-second hard ceiling. Only the gateway may generate evaluated submission rows;
persistent diagnostics distinguish ordinary saves from actual evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

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
    runtime_source = (ROOT / "scripts/notebook_runtime.py").read_text()
    build_id = hashlib.sha256((json.dumps(source_hashes, sort_keys=True)
                              + runtime_source + Path(__file__).read_text()).encode()).hexdigest()[:16]
    setup_cell = code_cell(runtime_source + f"\nrunner = NotebookRunner(build_id={build_id!r})\nrunner.start()\n")
    install_cell = code_cell("runner.install_dependencies()\n")
    verification_cell = code_cell("runner.verify_sources(" + repr(source_hashes) + ")\n")
    audit_cell = code_cell("%%writefile /tmp/submission_audit.py\n" +
                           (ROOT / "scripts/audit_containment.py").read_text().replace(
                               'ROOT = Path(__file__).resolve().parents[1]',
                               "ROOT = Path('/tmp/agent_preflight')"))
    run_cell = code_cell("runner.run()\n")
    preflight_cell = code_cell("if not runner.rerun:\n    runner.preflight_framework()\n")

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "arc3": {"build_id": build_id, "submission_policy": "gateway-only"},
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
                "`make submit`.\n\n"
                "Only the competition gateway writes evaluated rows. Ordinary saves are "
                "logged as NOT_EVALUATED and emit a zero-row parquet for code submission. Diagnostics are saved under "
                "`/kaggle/working/diagnostics/`."
            ),
            setup_cell,
            install_cell,
            *write_cells,
            verification_cell,
            audit_cell,
            preflight_cell,
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
