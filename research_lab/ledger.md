# ARC-AGI-3 PhD Research Laboratory Ledger

## Sprint 1 Task Packet — PI issue; critic hold

Issued 2026-10-20 (Asia/Calcutta). Audited checkout: `4eb5cd0` (remote main, including the newly added baseline notebook). The earlier source audit at `7a39d05` remains valid for unchanged Python files; the notebook findings below supersede assumptions that no LLM baseline artifact exists.

**Selected architectural parent:** `research_lab/top_programs/baseline_28_score.ipynb`, newly supplied on remote main. Preserve it unchanged and port its relevant mechanisms into the deployed Python interface with ablations. It is the available LLM-harness reference, but its filename is not score verification: saved output records one ft09 rehearsal at 47.62, not a multi-game 28.00 leaderboard result. No highest-scoring verified champion can yet be selected from supplied evidence. **Control/reference source:** `research_lab/top_programs/v0_franzen_m2_baseline.py`, byte-identical to the active random starter `agent/my_agent.py`. These are distinct architectures, and the local runner/builder do not use the archived LLM notebook. No code mutation or submission is approved by this documentation audit.

Execution order: systems containment prerequisite first, perception transition interface next, world-model integration after those contracts. At most these three orthogonal implementation tickets may proceed in parallel. Shared file edits require coordination. No prompt-wording-only tickets or game-specific policies.

### S1-P: phd-perception | [EDIT] observation and action interception

- Targets: empty `agent/arclog.py`, observation adapter and `MyAgent.choose_action`. Port/reuse the archived FrameView segmentation, frame diff, animation and transition-evidence contracts before adding parallel implementations. Do not discard the notebook's existing no-op/stale-state guards; test their semantics against the desired per-object dead signatures.
- Independent default-1 ablations: `ENABLE_HUD_MASK`, `ENABLE_ANIMATION_DIFF`, `ENABLE_DEAD_SIGNATURE`, `ENABLE_DOUBLE_RESET_GUARD`. Each toggle must restore its reference path when 0.
- Implement generic connected components and object signatures; record immutable transitions with action/coordinates, level/episode identity, pre-state, ordered intermediate grids, settled post-state, reset/terminal flags, HUD mask and causal diffs. Separate action-independent monotonic border evidence from causal changes; retain transient changes that revert before settling.
- Do not confuse outer `frames` history with a single action's animation sequence. Verify temporal ordering of the grids within `FrameData.frame` against the installed wrapper, without reading private game source.
- Handoff API to S1-W: `get_board_objects`, `get_step_diffs`, `get_hud_mask`, `is_dead_signature`. Block signatures after two consecutive zero settled AND transient causal effects on the current level; clear scoped counters at boundaries. Block consecutive reset proposals before dispatch, replan using available actions, and log blocked proposals separately from live actions.
- Required validation: synthetic fixtures for reverting flash, object translation, border timer, inert object and consecutive reset; real targets `ls20`, `vc33`, plus `ft09` when available. Report before/after zero-causal-diff live-action counts, reset losses, denominators and paired seeds. Game IDs may appear in test selection, never deployed policy.

### S1-W: phd-world-model | [REPLACE] evidence-based memory and planning

- Target: reasoning/state pipeline in `agent/my_agent.py`, consuming S1-P transitions through the S1-S sandbox. There is no FIFO implementation in the active Python agent; the notebook does have oldest-block eviction and persistent section memory. Port that evidence interface, then replace its eviction policy under ablation; do not describe all memory as missing from the archived parent.
- Default-1 ablations: `ENABLE_PLAYBOOK_RESET`, `ENABLE_RETRODICTION`, `ENABLE_EXPECT_QUEUE`, `ENABLE_TIER1_SIMULATOR`.
- Distill verified mechanics, counterexamples and unresolved hypotheses into versioned `playbook.md` at level boundaries/token ceilings. Start a fresh context with an immutable snapshot; append turns within each context. Preserve full replay evidence outside the model context.
- Reuse the archived `replay_transitions` and `execute_plan` designs, but remove their optional-use loophole in the new gated path. The archived replay defaults to 16 cases, clamps to 64 and scans only the latest 128 records; implement explicit chunking/coverage accounting over the full relevant history, and distinguish truncated/empty/limited results from a complete pass. Replay every candidate mechanic over all relevant historical transitions before permitting one disambiguating live probe. Reject counterexamples at zero live-action cost; log crashes/timeouts as failures, not successful reasoning.
- Queue concrete framework action enums/coordinates with non-empty expected settled non-HUD cells and level-up predictions. Validate after each action; mismatch/reset/level transition invalidates remaining queue before another dispatch. After 40 wasted moves on level >=3, Tier-1 synthesized `step(state, action)` must compile and retrodict before bounded search; expose every fallback.
- Required validation: fixtures reject an incorrect mechanic at zero live-action cost, accept a consistent mechanic, interrupt queue on mismatch, and preserve learned facts across level boundaries. Targets `ls20`, `ft09`, `vc33`, `re86` if available, with paired seeds and late-level reporting. Targets: >=60% of rejected hypotheses falsified by zero-action replay; >=75% non-probing prediction coverage; >=85% match rate. Report numerators/denominators and simulator compile/replay success separately; absent coverage is N/A, not zero errors.

### S1-S: phd-systems | [EDIT] fail-closed containment and offline deployment

- First prerequisite: verify primary Python and separate engine-free sandbox. Harden `scripts/audit_containment.py`: missing sandbox, unexpected subprocess failures, engine import success and filesystem access into resolved `environment_files/` paths must fail closed. Test traversal, symlinks, writes and subprocess escape using synthetic canaries; do not inspect real game files. A venv alone does not enforce filesystem containment. Remove the starter's game-specific branch in the implementation change, then rerun Gate 1.
- Targets thereafter: sandbox launcher, `scripts/build_notebook.py`, runtime integration. Bundle BOTH `my_agent.py` and `arclog.py`; select `rtx6000` and verify generated metadata. Resolve actual offline model weights, wheels and supported inference API before choosing flags; the archived notebook configures Pennyroyal/Qwen inference, but this is not wired into the active starter workflow. Inventory the required source bundle, offline wheels, target and draft weights before claiming the port is runnable; preserve package and source hash verification.
- Default-1 ablations for new mechanisms: `ENABLE_SANDBOX_LIMITS`, `ENABLE_PREFIX_LOCK`, `ENABLE_LEVEL_BUDGET`. Resource-limit-off runs only on controlled synthetic fixtures; security containment remains mandatory for every evaluation. Sandbox calls require 15-second timeout, memory/process limits, explicit error telemetry and child cleanup.
- Enforce immutable prefixes within S1-W context IDs, with whole-context resets only. Implement level-weighted budgets and rollover with 29,700-second target under 32,400-second hard cap. Measure cache hit rate/throughput only with an actual backend; >95% cache reuse is a performance target, not static-code evidence.
- Required validation: missing sandbox cannot pass; engine import and forbidden file access denied; infinite-loop canary terminated within timeout tolerance; planner failures visible. Offline notebook payload includes both modules and RTX metadata. Public smoke targets `ls20`, `ft09`, `vc33` after data/network prerequisites; every within-context prefix must remain unchanged.

### Shared critic acceptance and promotion gates

Containment must pass with behavioral evidence before candidate runs. Run `ruff check agent/ scripts/`, `pyright agent/`, representative smoke checks and then the full available suite; allow at most three logged repair cycles per implementation failure. Missing games are distinct from failed games. Preserve parent and candidate diffs.

Use paired fixed seeds, identical dependency/data versions and action budgets for each toggle 1 vs 0. Prespecify seed set/sample size; report paired confidence intervals, late-level completion and wasted-action counts. Promotion requires Gate 1, coverage/match thresholds, positive causal lift, and multi-seed mean gain >=5 RHAE or >=20% reduction in Level 3–5 actions. Do not infer statistical confidence from a single-run delta. PI's stronger +8 two-sided target must be relative to a measured parent: 28+8=36, not the previous ledger's 33.80. Project-provided variance figures are not measurements from this checkout. Pivot after two stalled idea iterations. No champion promotion or Kaggle submission authorized by these findings.

## Exact current architecture audit

### agent/my_agent.py

- Lines 22–30 import random/time, engine data types and framework `Agent`. No LLM client, prompt, tokenizer, conversation list, REPL, subprocess, simulator or search exists.
- Lines 37–44 set `MAX_ACTIONS=80` and seed process-global randomness with wall-clock microseconds plus salted Python `hash(game_id)`. Despite the comment, this does not reproduce runs per game; no fixed-seed control exists.
- Lines 47–52 append the action cap to the name and stop only on `GameState.WIN`.
- Lines 54–59 accept `frames` and `latest_frame`, but read only `latest_frame.state`. `NOT_PLAYED` or `GAME_OVER` immediately returns RESET without checking previous action or attaching reset-specific reasoning.
- Lines 71–76 choose every enum action except RESET, ignoring `latest_frame.available_actions`. The `ls20` game-prefix branch doubles ACTION4 probability; others sample uniformly. This concrete game-specific policy violates containment. Enum membership alone does not establish legal action availability.
- Lines 79–87 attach uniformly random complex coordinates in [0,63] plus a reasoning dictionary; simple actions get a reasoning string. Selection does not inspect grid cells, object identities, HUD, animation, completed levels or prior action effects.
- **Context eviction:** none. Both overridden methods ignore `frames`; there is no FIFO, token window, context reset, playbook or learned memory. Context amnesia is a proposed future risk, not an observed failure of an existing LLM subsystem.
- **Observation history:** installed dependency `ARC-AGI-3-Agents` commit `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`, `agents/agent.py`, initializes placeholder `FrameData`; `main` supplies accumulated `self.frames` and converted `arc_env.observation_space`. `append_frame` appends returned frames without eviction. `_convert_raw_frame_data` preserves `raw.frame` as a list of grids. Outer history entries and within-response animation grids are distinct; this starter inspects neither.
- **Dispatch/action accounting:** framework `do_action_request` sends `arc_env.step(action, data=..., reasoning=...)` and normalizes string reasoning to a dictionary. Its main loop increments the counter per iteration and checks `<= MAX_ACTIONS`; absent an earlier win, cap 80 permits 81 loop iterations (cap 50 permits 51). This counter is not a measured per-level RHAE live-action total. Local runner sets `record=False`, so this path does not automatically retain trajectory evidence.
- `agent/arclog.py` is empty. None of the eight required ablation toggles is implemented. No observation logging, prediction checks, sandbox isolation, model-memory retention or retrodiction exists in deployed code.

### scripts/build_notebook.py

- `ACCELERATOR="t4"` maps to `nvidiaTeslaT4`; requested RTX 6000 target is not active. No SGLang service, model assets, prefix cache, sandbox or budget scheduler is configured.
- `build()` reads only `agent/my_agent.py`, emitting a writefile cell targeting `/tmp/my_agent.py`. `arclog.py` is not packaged.
- Competition rerun installs from competition offline wheels, waits for gateway, copies the supplied framework, registers MyAgent, writes gateway configuration, and invokes framework `main.py`. No isolated agent REPL is started.
- Non-rerun branch writes dummy parquet via pandas; that path cannot validate real gameplay or leaderboard score. `main()` writes the ignored notebook and can synchronize tracked kernel metadata when accelerator changes.

### Archived LLM notebook: baseline_28_score.ipynb (static audit)

Cell numbers below are zero-based JSON cell indices. Code and saved outputs were inspected as data; no notebook cell, embedded patch, model server, pickle or game source was executed. The compressed Dhia patch was decoded as a literal into temporary text solely for inspection.

- **Identity and integration:** cell 3 writes a roughly 499 KB unified diff against Duck harness `ARC3-Inference`. Cell 5 copies an external Kaggle source bundle, applies that diff, then applies an embedded base64/LZMA Dhia 0.1.21 patch and compares 49 source files with embedded SHA256 values. The external base source is not in this checkout; patch context is evidence of intended changes, not a locally reconstructed, verified installation. The active builder never reads this notebook, and its agent implementation is not `agent/my_agent.py`.
- **Model/system path:** cell 5 configures a localhost OpenAI-compatible `flashnext` endpoint on port 8001. Cell 13 launches an Intel AutoRound Qwen3.8-Flash-Next target plus speculative draft via Pennyroyal/SGLang: context `(116+12+8)*1024`, memory fraction 0.96, max requests 10, FP8 KV cache, speculative steps 3. External source bundle, wheelhouse, model and draft mounts are prerequisites. Notebook metadata names `nvidiaRtxPro6000` but also has `isGpuEnabled=false`; saved census says GPU unavailable. Do not copy that contradictory metadata as a verified hardware configuration.
- **Context settings (cell 5):** `LOCAL_ANALYZER_CONTEXT_WINDOW=(116+12)*1024`, output reserve `12*1024`, `ARC3_CONTEXT_DRAIN_TOKENS=58*1024`, `ARC3_HISTORY_ASSISTANT_TURNS=150`, `ARC3_HISTORY_TURN_DRAIN=30`, coalescing enabled. The markdown's “58K / 30 turns” is not a 30-turn cap. These are configured token/drain values, not measured exact retained-history lengths.
- **Eviction implementation (cell 3, tool_agent.py diff):** `_trim_messages_for_context` holds the system message separately, estimates token use and removes oldest history blocks when budget is exceeded; drain size is capped at half the usable budget. `_keep_recent_history_turns` counts assistant messages, triggers above the 150 cap and drains toward 120, or coalesces with a token eviction. `_drop_oldest_history_block` preserves a recent tail and the last user query. `_note_history_evicted` marks prefix invalidation and scheduler handover. Optional control/image pruning and rolling-summary replacement are implemented, but their mere presence does not establish activation in cell 5. This path mutates prefixes during eviction; it is not append-only throughout a game.
- **Persistent memory:** `ARC3_MEMORY_SECTIONS=on` enables all configured section labels, rather than `slim` or `medium`. `_update_summarized_knowledge_from_assistant` parses scientist-note sections into `_summarized_knowledge`, keeps prior values on empty/degenerate updates, and renders carried knowledge in prompts. The diff includes world model, action/goal model, plan and cross-level notes. This is existing structured memory, not zero retention and not the requested versioned playbook reset. `ARC3_PERSISTENT_FUNCTIONS=1`, scope `game`, imports enabled; the Dhia overlay adds transactional helper retention. Mutable REPL variables are not universally persisted.
- **Observations:** FrameView exposes ASCII, raw grid, lazy segmentation and object diffs. Host history produces chronological TransitionViews with before/after frame, actual action and per-action result, separately from the latest action-call batch summary. Refused calls do not fabricate transitions. Animation flags/diff images are enabled. Dhia adds `grid_view`, `grid_candidates`, `bbox_rc` and `transition_evidence`; evidence is assembled from at most 64 recent frames. These are candidate perception/evidence tools, not proof of correct HUD masking or complete historical evidence.
- **Actions:** model Python snippets call `action(actions)` through normalized guarded host callbacks. Raw actions remain available. Per-position no-op/stale-state guards, terminal batch stops and reset-position checks are present; RESET exposure is opt-in, while `EXPOSE_UNDO=on` is set. No-op/stale-state guards are enabled from level 2. A completed level stops an old-level plan; the Dhia patch adds level-boundary priority yield. Do not equate reset-first-in-snippet with a global consecutive-reset guard.
- **Search and checked execution (Dhia overlay):** `search_plan` searches caller-supplied pure model callbacks with bounded BFS, default 50,000 nodes/depth 128/2s/250,000 edges and hard clamps. It never dispatches returned actions. `execute_plan` checks supplied start state, predicts and dispatches one action at a time, observes results, and stops on mismatches, refusal/no-op, errors or terminal boundaries. It is optional and checks the caller's representation, not full board correctness or mandatory non-HUD cell coverage.
- **Replay limitation:** `replay_transitions(observe, predict, records=None, max_cases=16, timeout_s=2.0)` defaults to current transitions, scans at most the latest 128 records, selects at most 64 eligible recent cases and clamps time to 5s. It reports matched/mismatched/errors/skipped and truncation. This is bounded diagnostic replay, not a gate proving 100% consistency with all `log.txt` history before every live probe. `matches_observed` is not universal mechanic verification.
- **Scheduling:** cell 5 enables priority scheduling, ten active streams and `ARC3_PRIORITY_PACE=0`. Cell 17 sets competition concurrency to 120 and per-game maximum to 532 minutes; rehearsal is one remaining game with 5 minutes per game. Cell 21 builds game lists from the competition API or offline host wrapper. Existing scheduler/level-boundary yield must be profiled before replacement; these settings do not prove the proposed level-proportional budget or cache-hit target.
- **Saved evidence:** cell 21 output records ft09-0d8bbf25, `state=gave_up`, level 4/6, score 47.62, 55 actions, 41,657 tokens, 307 seconds; one game/one pass/zero wins. These are archived outputs, not a rerun here or 28.00 hidden leaderboard verification. Cell 17 excludes other rehearsal games; cell 23 embeds diagnostics. None substitutes for current containment or paired-seed promotion evidence.
- **Containment scope gap:** the current audit scans only `agent/*.py` and optionally probes `.venv_agent_sandbox`. It does not inspect the archived notebook's embedded sandbox or execute its negative access tests. Restricted imports in the patch are a design mechanism, not demonstrated resistance to traversal, symlink or subprocess escape. Sprint S1-S must validate the actual port and deployed execution boundary.

## Failure triage and observed evidence

| Area | Evidence | Classification / owner |
|---|---|---|
| Gate 1 | Audit exits 1; deployed policy contains forbidden `ls20` | Immediate REJECT; S1-S prerequisite |
| Sandbox | `.venv_agent_sandbox` absent; audit defaults isolation to true and skips import probe | False assurance; no filesystem denial test; S1-S |
| Type A | Active arclog empty; archived baseline has segmentation/diffs/animation/evidence | Port and validate existing perception before extensions; S1-P |
| Type B | Active agent lacks model subsystem; notebook has drains, section memory and optional replay/checked execution | Integrate parent, enforce full-history and prediction gates; S1-W |
| Type C | Active builder T4; notebook has external inference and priority scheduler | Deployment divergence and missing verified containment; S1-S |
| Claimed 28 RHAE / L3 cap | Archived notebook includes a one-game ft09 rehearsal (4/6, 47.62); no 28.00 leaderboard scorecard or multi-seed comparison supplied | 28.00 and prior L3-cap assertion unverified; critic hold |
| Diagnostics | Summarizer reports zero files; implementation counts files, not outcomes or metrics | No measured gameplay evidence |
| Local smoke | `make verify-local` denied at `three.arcprize.org` by proxy; no game executes | Environment access blocker, not score evidence |

## Experiment and ablation log

| ID | Command / artifact | Actual outcome | Verdict |
|---|---|---|---|
| EXP-000 historical claim | Prior ledger: 28.00 RHAE, PASS, L3 cap | Python snapshot is random starter; separate notebook now supplies LLM code and one-game rehearsal, not proof of 28.00 | Unverified score; previous containment PASS withdrawn |
| AUDIT-S1 | `python scripts/audit_containment.py`; `containment.json` | `containment_passed=false`; `sandbox_isolated=true`; forbidden `ls20` | Gate 1 REJECT; isolation field insufficient |
| SETUP-S1 | Python 3.12 install, `pip check`, `make notebook` | Dependencies consistent; notebook built with T4; game smoke blocked | Build capability only |

No agent implementation, audit logic, parent snapshot or notebook builder changed during this documentation audit. No new gameplay score, measured prediction coverage, simulator success, sandbox containment or Kaggle execution is claimed. Archived notebook outputs are identified separately above.

## S1-S implementation — 2026-10-20 (Asia/Calcutta)

**Mutation: [EDIT] fail-closed containment and offline deployment.** Executed on
explicit user instruction. Architectural parent remains
`research_lab/top_programs/baseline_28_score.ipynb`; its SHA256 is preserved in
`agent/baseline/manifest.json` and it is unchanged. The prior documentation-only
audit above describes the pre-implementation checkout. This section supersedes
its current Gate 1, sandbox, builder and network-blocker statuses.

### Implemented systems and handoff

- Removed the game-specific branch and its identifier-bearing comments from
  `agent/my_agent.py`. Recursive scanning of every `agent/**/*.py` checks all 25
  current public IDs, a generic ID pattern and forbidden source-path tokens.
- Verified primary Python 3.12.14; created `.venv_agent_sandbox` without system
  site packages. Its isolated interpreter reports the engine module absent.
  `agent/sandbox.py` establishes a mandatory Linux seccomp syscall allowlist
  before generated code executes. File opens/listing/writes, exec/fork and socket
  creation are denied, including direct native calls. Host code mediates evidence
  and persistence via stdin/stdout. The worker cannot directly access scratch.
  Landlock and user namespaces are unavailable on this host; their failure is
  never treated as successful isolation. Missing libseccomp or filter failures
  prevent generated code execution and produce failed telemetry.
- `MyAgent.execute_planner` uses the engine-free worker. Default-on
  `ENABLE_SANDBOX_LIMITS` provides 15-second wall timeout, 512 MiB address-space,
  process/descriptor/output/core limits, explicit errors, process-group kill and
  reaping. Turning limits off requires `controlled_fixture=True`; seccomp stays
  mandatory. Synthetic resource-off access is denied. Extra disk-backed imports
  are blocked; future scientific dependencies require reviewed bootstrap support.
- Preserved Cell 3 and decoded Cell 5 Dhia 0.1.21 patches as inspectable text in
  `agent/baseline/`. Extracted 11 complete new Python modules: frame diff,
  animation, no-op guard, priority scheduler, retained functions, bounded search,
  checked execution, grid coverage, replay comparison, state contracts and
  transition evidence. Added a local hexadecimal display adapter for animation's
  absent upstream display dependency. Manifest records parent/patch/original/
  extracted/ported hashes and all 49 archived expected final-source hashes.
  Re-running `scripts/extract_baseline.py` verifies existing ports and exact patch
  extraction without overwriting reviewed changes.
- **Extraction boundary:** the patch only supplies hunks of existing FrameView,
  segmentation, Python tool sandbox and tool-agent implementations. Their base
  files are absent; faithfully reconstructing those entire files is blocked by
  the external source bundle. Exact hunks and a per-file gap inventory are
  retained, not represented as working modules. S1-P/S1-W can directly import the
  complete helpers; remaining integration contracts and caveats are documented
  in `agent/baseline/README.md`. Archived replay limits remain unchanged pending
  S1-W's full-history gate. No LLM reasoning subsystem is claimed integrated.
- `agent/runtime.py` provides immutable serialized per-context prefix snapshots;
  callers must validate complete requests immediately before inference. Reusing
  closed IDs, deletion, reorder or mutation fails under `ENABLE_PREFIX_LOCK=1`;
  whole resets use fresh IDs. The current random agent has no model requests;
  S1-W must connect this API at its backend boundary. Cache hit rate is N/A.
- `LevelBudget` allocates `env_seconds * l / sum(1..n)`, carries unused time (and
  overruns), reduces generation tokens near deadline, and exposes best-effort
  execution at exhaustion. It consumes public level-count metadata in MyAgent.
  `ARC3_ENV_SECONDS` supplies the environment allocation; the default is the run
  target, so a future concurrent backend scheduler must divide the shared budget
  across environments. `ENABLE_LEVEL_BUDGET=0` removes level weighting while
  retaining environment/global deadlines. Shared target is 29,700 seconds under
  the 32,400-second cap. Notebook process timeout includes setup and kills/reaps
  the framework process group at the target; no throughput result is inferred.
- Builder uses `ACCELERATOR = "rtx6000"`, emits `nvidiaRtx6000`, GPU enabled and
  internet disabled. Payload includes `my_agent.py`, `arclog.py` and every Python
  support module, validates SHA256 hashes, creates an isolated venv and runs the
  same behavioral audit before gameplay. All three requested toggles default to
  `"1"`. Empty `arclog.py` remains the S1-P integration target.
- `research_lab/offline_inventory.json` records the absent source bundle,
  Pennyroyal v253 wheelhouse, Intel AutoRound target weights and draft weights.
  Archived inference API is OpenAI-compatible `flashnext` on localhost:8001.
  Source/package/model installation, package hashes and serving flags cannot be
  verified without those mounts. No server launch or offline LLM readiness claim.

### Actual validation

Evidence: `containment.json` and `research_lab/traces/s1_s_validation.json`.

| Check | Actual result |
|---|---|
| `ruff check agent/ scripts/` | PASS; Ruff 0.17.1 |
| `pyright agent/` | PASS, 0 errors / warnings; Pyright 1.1.414; project venv/vendor resolution configured |
| `python scripts/audit_containment.py` | PASS; `containment_passed=true`, no violations |
| `python scripts/test_systems.py -v` | 8/8 fixtures pass |
| Missing sandbox / failed interpreter / synthetic importable engine | All rejected; no skipped-probe success |
| Synthetic absolute/traversal/symlink reads; writes/listing; subprocess/fork/exec/native-open attempts | Denied; canary unchanged, forbidden new file absent |
| Compute / planner exception / 600 MiB allocation / infinite loop | Compute works; exception visible; allocation denied; loop killed near 15.001 s and worker reaped |
| Prefix mutation, alias mutation, context reset; prefix toggle 0 | Enforced / reference path verified on fixtures |
| Level weighting, rollover, token degradation, caps; budget toggle 0 | Deterministic fixtures pass |
| Notebook source payload / metadata / generated preflight | Both entry modules and helpers present; RTX metadata correct; Python payloads parse; generated audit passes |
| Local agent import | PASS via existing importlib loader |
| `play_local.py --game ls20,ft09,vc33 --max-steps 2` | All 3 run without exception |
| `play_local.py --max-steps 2` | All 25 available games run without exception; 0 levels, 3 framework iterations each, aggregate 0.0 |

These short smokes verify compatibility only. The existing framework's `<=` cap
means the two-step argument permits three iterations. No paired seeds, score
improvement, late-level performance, prediction coverage, cache reuse or simulator
success is claimed. Network access now succeeds through the configured proxy;
only the engine host downloaded/loaded game assets. No game file was inspected by
the researcher or accessed by a worker.

Repair accounting: three bounded implementation repair passes covered (1)
import/style issues and inherited dynamic types, (2) remaining helper type and
formatting diagnostics, and (3) callback annotations / SDK reasoning metadata and
final lint cleanup. Interim autofix diagnostics were retained in `work/`; final
lint/type checks pass. The unrelated diagnostics script required only import
ordering for the mandated whole-directory lint command. Synthetic subprocess
canary handling explicitly recognizes this host's blocked-shell status 127;
separate native-open, fork and exec denial checks still run.

**Critic decision:** Gate 1 PASS for this systems implementation. S1-P/S1-W
integration, unavailable upstream base/LLM assets, Gates 2–3 and statistical
promotion remain on hold. No champion promotion or Kaggle submission performed.
