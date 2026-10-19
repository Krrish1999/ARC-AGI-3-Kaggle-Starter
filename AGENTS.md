# AGENTS.md — ARC-AGI-3 Autonomous PhD Research Laboratory

## 1. Mission & Mathematical Objective
We are competing in ARC Prize 2026 (ARC-AGI-3). Our baseline score is 28.00 RHAE, which sits inside the stochastic noise band of Daniel Franzen's Milestone 2 Duck-harness baseline (Mean: 25.77, SD: 3.93, Range: 24.95–32.60). Our objective is to engineer a verifiable, offline World-Model Harness that exceeds 65.00+ RHAE (Tufa Labs / Yi-Chia Chen tier) before the November 2, 2026 deadline.

All engineering decisions must optimize Relative Human Action Efficiency (RHAE):
- **Level Efficiency Score:** `S_{l,e} = min(1.15, (h_{l,e} / a_{l,e})^2)` where `h_{l,e}` is the human baseline and `a_{l,e}` is agent actions taken on level `l` of environment `e` (hard cutoff at `5 * h_{l,e}`).
- **Environment Score with Completion Cap:** `E_e = min(sum_{l=1..k}(w_l) / sum_{l=1..n}(w_l), sum_{l=1..n}(w_l * S_{l,e}) / sum_{l=1..n}(w_l))` where level weight is linear (`w_l = l`).
- **Zero-Action Arbitrage Law:** Only commands that advance the live environment state increment `a_{l,e}`. Internal Python REPL execution, historical `log.txt` retrodiction, and offline search inside a synthesized `step(state, action)` simulator cost ZERO environment actions.

## 2. Mandatory Preflight & Containment Rules
Before modifying any code, execute this preflight sequence:
1. Activate the primary environment (`.venv/bin/python`) and verify the isolated agent REPL sandbox (`.venv_agent_sandbox/bin/python`).
2. **STRICT CONTAINMENT LAW (The Kepler Audit):** You are strictly prohibited from reading files inside `environment_files/` or importing `arc_agi` engine internals inside the agent's REPL sandbox. Every candidate run must execute `python scripts/audit_containment.py` and produce `"containment_passed": true` inside `containment.json`.
3. **Offline Deployment Target:** All runtime code must compile cleanly into `agent/my_agent.py` and `agent/arclog.py`, packaged via `scripts/build_notebook.py` for Kaggle's offline NVIDIA RTX 6000 (`g4-standard-48`, `ACCELERATOR = "rtx6000"`) within a 9-hour (32,400-second) wall-clock limit.

## 3. Evolutionary Mutation Protocol (`mle-solver`)
1. Always branch from the highest-scoring verified parent in `research_lab/top_programs/`.
2. Declare your mutation operator explicitly in `research_lab/ledger.md` and your PR title:
   - `[REPLACE]`: Complete architectural rewrite of a target subsystem.
   - `[EDIT]`: Targeted line-range modification protected by a Reki-style environment-variable ablation toggle (e.g., `os.getenv("ENABLE_RETRODICTION", "1") == "1"`).
3. Perform up to `N_rep = 3` automated repair cycles if `ruff`, `pyright`, or `make play-local GAME=ls20` raises an exception. If the 3rd repair fails, revert cleanly and log the failure mode in `research_lab/ledger.md`.

## 4. Standard Verification Commands
- **Lint & Type Check:** `ruff check agent/ scripts/ && pyright agent/`
- **Containment Audit:** `python scripts/audit_containment.py`
- **Single-Game Fast Smoke Test:** `make play-local GAME=ls20` (followed by `GAME=ft09` and `GAME=vc33`)
- **Full 25-Game Benchmark:** `make play-local`
- **Scorecard Compression:** `python scripts/summarize_diagnostics.py --log-dir logs/ --out research_lab/ledger.md`