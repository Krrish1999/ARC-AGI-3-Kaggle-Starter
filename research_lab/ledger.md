# ARC-AGI-3 PhD Research Laboratory Ledger

## [EDIT] Hidden-evaluation submission survival — 2026-10-20

User-requested reliability mutation of the Sprint 1 archived harness at
`6f4dc442f587ac5ba26fea2fdf1de889291dfc6f` (deployed agent hash matches the
archive before this edit); no score-promoted parent exists. Preserve all archived
sources. `ENABLE_ACTION_FALLBACK=1` catches the complete action-selection path,
logs tracebacks under `scratch/` (or `ARC3_LOG_DIR`), clears pending plans and
uses public legal actions in emergency mode for the remaining game. This mode
bypasses the failed evidence pipeline during dispatch and makes no replay claims.
Setting the toggle to `0` restores exception propagation for controlled fixtures.

The notebook's first executable cell atomically writes a zero-score placeholder
in both public and competition-rerun modes, preserving existing gateway output.
All setup shares one start time. Keep the 29,700-second operating target; bound
the global runner at the user-requested 30,600 seconds. Request framework SIGINT
cleanup to close its public scorecard before killing/reaping remaining children;
retain the early parquet if initialization, gameplay or cleanup fails. Deadlines
and placeholder generation are unconditional safety guarantees, not ablations.
Validation: `pytest -q` passes 95/95 cases (71 existing + 24 survival fixtures);
`ruff check agent/ scripts/` and `pyright agent/` pass. The survival fixtures
exercise first-cell SIGKILL in both modes, preservation of real output, injected
budget/perception/world-model/Tier-1 failures, fallback dispatch, ablation,
deadlines, failed/hung cleanup, real SIGINT finalization, and the generated
rerun Python/registry including preflight failure. Notebook regeneration succeeds.
Containment passes all 20 checks. Fixed-seed (`19`) public smoke runs on `ls20`,
`ft09`, `vc33` finish normally at the existing 81-iteration cap; zero levels
complete and no policy improvement is claimed. Private Kaggle acceptance and
the hidden-test score remain unverified; no competition submission was made.

## Sprint 2 Task Packet — offline bundle and RTX 6000 validation

Issued 2026-10-20 (Asia/Calcutta) after all three Sprint 1 implementation PRs
merged into main `6f4dc44`. **Sprint 1 is COMPLETE for functional integration**
(S1-S, S1-P, S1-W). Gate 1 passes; real-inference Gates 2–3 and statistical
promotion remain open. The historical audits below retain their original scope
and dates; this packet and the closeout supersede their current-status claims.

**Selected engineering parent:**
`research_lab/top_programs/v1_retrodict_world_model_harness.py`, an exact archive
of the unified `agent/my_agent.py` and `agent/arclog.py` at
`6f4dc442f587ac5ba26fea2fdf1de889291dfc6f` (use the archive's `SOURCE_COMMIT`
as the authoritative full revision). It records module and support-module hashes;
restore the two `SOURCE_FILES` entries under their original package paths with
support modules from that revision. This user-requested checkpoint is not a
score-promoted champion. No highest-scoring verified parent exists yet. Keep
`baseline_28_score.ipynb` and `v0_franzen_m2_baseline.py` unchanged as references.

**Failure triage for Sprint 2**

| Area | Current evidence | Priority / owner |
|---|---|---|
| Type A — perception | Immutable objects/diffs, animation/HUD/dead-target/reset contracts pass deterministic fixtures; short public runs provide no late-level evidence | Preserve S1-P contracts; no new perception mutation ticket |
| Type B — memory/planning | Full-history replay, expect queue, scoped playbook and Tier-1 simulator fixtures pass; `WorldModel.backend` is still `None` in default gameplay | Connect the real local backend in S2-B; phd-world-model contract review |
| Type C — deployment/throughput | Builder bundles Python sources and containment, but launches no model server; source bundle, wheelhouse and target/draft mounts are absent in `offline_inventory.json` | Blocking deployment gap; phd-systems, S2-B then S2-V |
| Evaluation | Historical public coverage is 0 predictions / 328 nonboundary responses per configuration; match, replay-rejection and simulator rates N/A; latest diagnostic file scan finds 0 logs | No per-level causal diagnosis or score claim possible; critic retains Gates 2–3 hold |

Execute these two bounded tickets in dependency order. S2-V may prepare its
validator while S2-B is implemented, but its final run must use the exact generated
bundle. Shared builder edits require coordination. No prompt-only changes.

### S2-B: phd-systems | [EDIT] complete offline inference bundle

- Targets: `scripts/build_notebook.py`, `notebooks/kernel-metadata.json`,
  `research_lab/offline_inventory.json`, and a minimal adapter to the existing
  `WorldModel.backend(messages)` interface. Preserve the S1 source archive and
  sandbox/evidence/expect interfaces; include both entry modules and every needed
  support module, launcher and configuration in the generated notebook.
- Independent default-`1` ablations: `ENABLE_OFFLINE_BACKEND` connects the local
  backend; `ENABLE_SPECULATIVE_DRAFT` selects the verified target-plus-draft path.
  Backend `0` restores the explicit logged exploration control for controlled
  comparisons; draft `0` serves the same target without speculative decoding.
  Neither toggle may disable containment, asset verification or deadlines.
- Package the **Pennyroyal SGLang v2 launcher**, **Qwen 3.8 / Intel AutoRound
  target weights mount**, compatible draft mount when enabled, and the complete
  offline wheelhouse. Resolve exact upstream release, supported launcher entry
  point, model/draft versions, quantization and serving flags from the actual
  artifacts before pinning them. The historical inventory says Pennyroyal v253;
  do not silently equate that label with v2 or claim mount availability.
- Record immutable source/model/wheel versions, licenses/provenance, file hashes,
  required files and resolved Kaggle mount paths in a bundle manifest. Verify
  the archived source/patch hashes if reusing that bundle. Add the real dataset
  and model attachments to kernel metadata; never invent attachment identifiers
  or treat local fixture paths as actual mounts.
- Resolve dependencies completely for the Kaggle Python/CUDA/PyTorch ABI. Install
  only from attached wheels with `--no-index` and explicit `--find-links`, with
  hash verification and no downloads or runtime source builds. Preflight fails
  visibly on missing, corrupt or incompatible source, weights or wheels before
  live dispatch. Preserve the stdlib perception fallback and engine-free worker.
- Start the verified launcher on loopback, using the archived OpenAI-compatible
  `flashnext` contract at `127.0.0.1:8001` if supported by v2. Keep it distinct
  from the competition `gateway:8001`. Bound startup/readiness and inference
  requests; record server errors and clean up its process group on all exits.
- Connect complete immutable context requests through PrefixLock to the backend;
  parse structured hypotheses/actions/simulator replies through the existing
  gates. An enabled backend must fail visibly on unavailable/malformed output
  rather than report an exploration run as model inference. Record actual
  model identity, first response, throughput, cache metrics when available and
  simulator/fallback telemetry; missing cache data remains N/A.
- Validation targets: synthetic backend/asset failure fixtures first, then public
  `ls20`, `ft09`, `vc33`, `re86` through host-owned interfaces with fixed environment
  and policy seeds `{7,19}` and equal action budgets. Pass: all 71 existing tests,
  lint/types and behavioral containment; source hash round-trip; offline install,
  local readiness and at least one actual structured response reaching the
  unchanged world-model gates. Performance lift is evaluated separately.

### S2-V: phd-systems | [EDIT] generated submission acceptance validator

- Targets: `scripts/build_notebook.py`, a generated-notebook validation entry
  point under `scripts/`, and machine-readable evidence under
  `research_lab/traces/`. Critic reviews results against Gates 1–3. Depends on
  S2-B's complete bundle and its exact content hashes.
- Default-`1` ablation: `ENABLE_DEPLOYMENT_PREFLIGHT`. Setting `0` may skip only
  the extra deployment diagnostics in isolated synthetic comparisons; mandatory
  containment, source/asset hashes, offline installation and deadline checks
  remain enforced for all evaluation runs.
- Validate the **generated** `notebooks/submission.ipynb` and
  `notebooks/kernel-metadata.json` together: nbformat/cell syntax and payload
  hashes; both agent modules/support imports; GPU enabled; internet disabled;
  `ACCELERATOR = "rtx6000"`, notebook accelerator `nvidiaRtx6000`, target
  `g4-standard-48`. Confirm the actual Kaggle accelerator/schema at execution
  and record GPU model, VRAM, driver, Python, CUDA, PyTorch and SGLang versions;
  metadata alone cannot prove hardware compatibility.
- Execute from a clean Kaggle offline RTX 6000 session using only attached
  assets and the competition gateway. Prove dependency closure, AutoRound load,
  kernel/quantization compatibility, bounded readiness, real generation and
  generated containment preflight before a live action. Run the notebook's
  competition-rerun branch: dummy save-and-run parquet output is not acceptance.
- Test missing/corrupt/incompatible wheel and weight manifests, absent GPU,
  failed/OOM server, startup/request timeout, malformed responses and worker
  failure. Each must expose a diagnostic and stop or record the defined failure
  path without fabricating inference or weakening containment.
- Measure setup + server load + gameplay + cleanup against one shared start:
  29,700-second operating target and 32,400-second hard wall limit. Account for
  both server and framework children, aggregate concurrent environment budgets,
  and verify process-group cleanup and no deadline rollover. Require a complete
  offline rehearsal with measured runtime/peak VRAM; an accelerated timeout
  fixture alone is insufficient evidence for a nine-hour run.
- Use the same four public targets, fixed paired seeds `{7,19}`, dependencies,
  weights, action caps and scoring rules for default-on versus each backend/draft
  ablation. Record non-probing coverage/match numerators and denominators,
  zero-action rejection rate, compile/replay/search successes, visible fallbacks,
  wasted actions, late-level opportunities/completion and paired RHAE deltas.
  Add seed pairs under a prespecified protocol if needed for confidence; do not
  infer significance from the eight-pair smoke or synthetic cases.
- Pass requires reproducible generated-notebook validation and offline RTX 6000
  execution artifacts bound to the source/notebook/asset hashes. If hardware or
  attachments are unavailable, report **BLOCKED** with exact missing inputs;
  static checks can pass while runtime acceptance remains pending.

**Sprint 2 definition of done:** verified complete offline bundle, real local
backend connected, generated-notebook checks and clean offline RTX 6000 rehearsal
with resource/telemetry evidence. All 71 Sprint 1 regression cases and Gate 1 stay
green. Score promotion additionally requires >=60% zero-action rejection,
>=75% non-probing prediction coverage, >=85% settled non-HUD match, positive
paired causal lift, and >=5 RHAE improvement or >=20% Level 3–5 action reduction
against a measured parent. Report confidence intervals; PI's stronger two-sided
+8 target remains a research objective. This packet does not claim those results
or consume a Kaggle competition submission slot.

## Sprint 1 closeout — COMPLETE, functional acceptance

Closed 2026-10-20 (Asia/Calcutta), after merged S1-S / S1-P / S1-W at `6f4dc44`.
**Mutation: [EDIT] archival checkpoint and research ledger only.** Deployed
`agent/` code and the notebook builder are unchanged by this closeout, so no new
runtime ablation is introduced. The explicit user instruction authorizes saving
the combined checkpoint under `top_programs/` and synchronizing main; it does
not convert the archive into a statistically promoted strategy.

Fresh evidence: `containment.json` and
`research_lab/traces/sprint1_closeout_validation.json`. Primary Python 3.12.14
and the separate engine-free sandbox were verified. The behavioral audit passes
20/20 checks, including engine import denial, synthetic absolute/traversal/symlink
access, writes/listing, native/subprocess/fork/exec denial, useful compute,
resource limits, visible planner errors and timeout/worker cleanup. No real game
implementation files were read.

### 71/71 regression coverage matrix

Counts refer to collected parametrized test cases, not measured line/branch
coverage or public-game skill acquisition. Every case was rerun on the merged
checkout for this closeout.

| Sprint / test suite | Passing / collected | Behaviors covered |
|---|---|---|
| S1-S — `scripts/test_systems.py` | 8/8 | Missing/broken/importable-engine workers fail closed; resource-off remains contained; prefix alias/reset and budget rollover/caps; notebook payload/RTX metadata; extracted module imports/BFS |
| S1-P — `tests/test_perception_s1p.py` | 20/20 | Reverting animation, translation/components, row/column HUD and negatives, exact dead-signature thresholds/scope/boundaries, reset guard and reference ablations, immutable evidence, final-response interception, stale/dead/no-action/dispatch-error paths |
| S1-W — `tests/test_world_model_s1w.py` | 43/43 | Zero-action rejection/one-probe acceptance; full-history 145-transition replay; malformed/crashed/empty/limited/timeout workers; whole non-HUD state/status checks; expect validation/mismatch/boundaries/stale scope; scoped playbook/prefixes; simulator compile/replay/BFS and 40-waste trigger; fallback visibility; direct-dispatch and mutable-enum regressions |
| **Total — `pytest -q`** | **71/71** | All three Sprint 1 contracts integrated; zero failures, errors or skipped cases |

Additional fresh checks: `ruff check agent/ scripts/`, `pyright agent/`,
`pip check`, archived baseline extraction/hash verification, generated notebook
build/payload checks and `git diff --check`. Both archived source strings
round-trip byte for byte with their deployed files and compile independently.
All support sources are identified by SHA256 at the snapshot's parent commit.

Historical public evidence is retained, not rerun as part of this documentation
closeout: S1-P 30/30 and S1-W 40/40 paired compatibility runs; 25/25 full public
short smokes. They complete no levels and establish no learned-policy lift.
The latest diagnostics scan has zero files and cannot classify stalled levels.
Actual offline source/model/wheel mounts and GPU runtime validation remain
pending; existing builder metadata/source checks are not proof of a working
Pennyroyal/Qwen deployment.

**Critic disposition:** Gate 1 PASS; Sprint 1 functional integration accepted.
Gate 2 public prediction coverage remains unsatisfied and Gate 3 statistical
promotion is deferred. The root cause is no connected offline reasoning backend
and no late-level paired evidence, addressed by Sprint 2. Preserve the checkpoint
as an archive; no new score, champion or Kaggle execution is claimed.

---

## Historical Sprint 1 Task Packet — original PI issue; critic hold

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

## S1-P implementation — 2026-10-20 (Asia/Calcutta)

**Mutation: [EDIT] observation and action interception.** Local implementation
branch `feat/s1-p-perception`, based on S1-S commit `c283ccd`. The architectural
parent remains the unmodified archived notebook; the missing base FrameView and
segmentation file bodies are adapted from the preserved contracts and complete
signature hunks. No verified score champion or promotion is inferred.

**Prespecified validation protocol:** five required synthetic fixture categories
plus boundary, immutability and reference-path regressions. Public smoke/ablation
selection: `ls20`, `ft09`, `vc33`; agent RNG seeds `{7, 19}`; per-game framework cap
40 (41 iterations under the existing `<=` loop). Reference is the candidate with
exactly one requested flag set to `0`, versus all four flags `1`; other flags,
package/data versions and cap are held fixed. This yields six paired samples for
each independent flag (30 total runs with the six shared all-on controls).
Environmental randomness will be seeded only if the public wrapper exposes a
seed option; otherwise that limitation will be recorded. Report exact eligible
live-action denominators, zero settled-and-transient causal-effect counts,
blocked proposals and observed reset regressions. These small exploratory runs
are not a promotion benchmark. Results follow below after execution.

### Implemented observation and dispatch contracts

- `agent/arclog.py` provides a FrameView adapter with immutable integer grids and
  lazy connected components; object fields include frame-local ID, color, inclusive
  `(xmin,ymin,xmax,ymax)`, area, centroid, translation-invariant color-free shape
  hash, rotation-canonical hash/pose, and adjacency. Signature canonicalization is
  ported from Cell 3's complete hunks. SciPy `ndimage.label` and NetworkX provide
  the accelerated graph path; an equivalent stdlib path is tested for offline
  environments without those wheels. NumPy is already an engine dependency.
- Reuses `baseline.frame_diff.compute_frame_diff`, animation normalization,
  collapse/summary helpers, `transition_evidence`, and `NoopRepeatGuard`. Missing
  original FrameView/segmentation bodies are explicitly adapted, not described
  as faithfully reconstructed upstream files. Both raw object diffs and a causal
  object-diff view with the inferred HUD delta neutralized are retained.
- Confirmed response transitions are recorded at `MyAgent.do_action_request`,
  including the final action when the main loop stops. Each frozen record stores
  pre/post grid, the exact ordered response intermediates, action/coordinates,
  source object signature, level/episode, reset/full-reset/terminal flags,
  settled and reverting transient evidence, HUD mask, and helper diagnostics.
  No response means a visible dispatch-error event and re-raised failure, not an
  invented successful transition. Empty rendered responses carry unavailable
  causal-evidence status and cannot teach inert signatures.
- Per-agent `scratch/perception-*/log.txt` is append-only JSONL. `[DIFF]` lists exact
  raw `(x,y): old -> new` cells; `[CAUSAL_BOARD_DIFF]`,
  `[TRANSIENT_ANIMATION_DIFF]` and `[HUD_STEP_BUDGET_STRIP]` remain distinct.
  Refused proposals, explicit replans and dispatch failures are separate events;
  they never consume a live-action denominator or fabricate transitions.
- Wrapper ordering verified from public installed interfaces: generic
  `arcengine/base_game.py:perform_action` appends render results in loop order;
  `arc_agi/local_wrapper.py:step` returns that result; remote conversion and
  framework `_convert_raw_frame_data` preserve array order. Only one response's
  `FrameData.frame[:-1]` contains its intermediates. Outer accumulated `frames`
  are never interpreted as animation. No private game source was inspected.
- Handoff: `agent.arclog.get_board_objects(view_or_log)`,
  `get_step_diffs(log, index=-1)`, `get_hud_mask(log)`, and
  `is_dead_signature(log, action_type, target_shape_hash, target_color)`; equivalent
  methods live on `agent.perception`. Tuples/frozen records carry the full
  evidence; JSON-derived dictionaries returned to callers are independent copies.
  Worker access remains host-mediated under S1-S containment.

### Independent default-on ablations and reference paths

- `ENABLE_HUD_MASK=1`: requires four same-scope settled observations (three
  monotonic transitions), at least two action labels, an outer full row/column,
  a contiguous two-color strip and nested shrinking occupied sets. Microframes
  must also be monotonic; a reverting border flash invalidates classification.
  Initial warmup remains unmasked and prior records are never rewritten. Scalar
  `remaining_step_budget_estimate` is emitted only for a constant observed
  per-action drain; this is provisional visual evidence, not a proven timer or
  a game-specific resource decoder. With `0`, no HUD cells are removed.
- `ENABLE_ANIMATION_DIFF=1`: captures ordered changes at coordinates that change
  at least twice during the response, including flashes reverting to the original
  color. The pre-action board is included when identifying these changes.
  Transient event indices refer to the collapsed animation chain; the exact raw
  intermediates are separately preserved. With `0`, the policy/reference diff
  sees only settled changes; raw animation remains in the audit evidence.
- `ENABLE_DEAD_SIGNATURE=1`: per-level/episode `(action,shape_hash,color)` counters
  hard-block after two consecutive interactions with zero settled AND transient
  non-HUD effect. One inert observation does not block; positive effects clear
  that signature's count. Reset, terminal, full-reset and level boundaries clear
  scope. Targets come from the pre-action component under the requested cell.
  Untargeted directional actions do not invent an object identity. With `0`,
  generalized dead-signature blocking is disabled and the archived state/position
  no-op block-then-insist path is preserved from level 2. Its override is not
  consumed twice by choose/dispatch interception. Stale-observation validation
  remains exposed for future planned-action dispatch.
- `ENABLE_DOUBLE_RESET_GUARD=1`: checks the last confirmed live action, not the
  last proposal. Blocks a second RESET in both choose and final dispatch paths,
  selecting an advertised non-reset action. With `0`, consecutive RESET proposals
  dispatch unchanged. If hard signatures exhaust every advertised target, an
  explicit logged replan may issue one first RESET; it cannot revive a hard-dead
  target or bypass the consecutive-reset guard. If neither an admissible action
  nor a first RESET is possible, execution fails visibly for replanning.
- Generic sampling respects advertised actions and uses a private seeded RNG;
  `ARC3_SEED` supports paired controls without shared-global RNG interference.
  The public `Arcade.make(..., seed=...)` interface seeds fresh environments.
  No game IDs or fixed level solutions were added to deployed Python.

### Actual validation and denominators

Evidence: `research_lab/traces/s1_p_validation.json`,
`research_lab/traces/s1_p_smoke.json`, `containment.json`; raw per-action logs are
under the recorded ignored `scratch/` paths with SHA256 values in the smoke
report. The archived notebook and all 11 extracted parent modules are unchanged.

| Required / regression check | Actual outcome |
|---|---|
| `pytest tests/test_perception_s1p.py` | PASS, 20/20 parametrized cases |
| `ruff check agent/` | PASS |
| `pyright agent/` | PASS, 0 errors / warnings |
| `python scripts/audit_containment.py` | PASS, `containment_passed=true`, 20/20 behavioral/presence checks |
| Existing `scripts/test_systems.py` | PASS, 8/8 fixtures |
| Notebook payload parse, bundled containment preflight | PASS; both entry modules and all support sources packaged |
| Parent hash/extraction verification; `pip check`; diff whitespace check | PASS |
| Public independent ablation matrix | PASS, 30/30 runs, 41 observed responses each; six paired samples per flag |
| Full public compatibility smoke, cap 2 | PASS, 25/25 games, three framework iterations each, zero completed levels; not a benchmark |

Required synthetic fixture denominators: reverting flash has two positive
transient-effect responses out of two despite 0/2 settled effects; animation-off
sees 2/2 zero policy effects. Translation matches one moved object and exactly
its two changed cells. Row and column timer fixtures each infer one 64-cell
border after three transitions (two warmup transitions remain unmasked).
Inert target remains allowed after 1/1 zero effects, becomes hard-blocked after
2/2, and is cleared by positive effects and tested boundary kinds. Default-on
consecutive-reset fixture dispatches one RESET and one non-reset alternative for
two proposals; toggle-off dispatches both RESET proposals. Additional cases cover
border flash, non-border/action-diversity negatives, immutable evidence,
reference insistence, stale observations, final-response recording, dispatch
failure, no-admissible-action failure, and signature-exhaustion replan.

Public paired results below are exact numerators/denominators. “Policy zero” uses
that ablation's visible causal representation. “Common zero” always includes raw
transient evidence and the same HUD-assessment rule; HUD-off traces matched the
paired all-on evidence exactly, allowing the same mask assessment. Thus animation
visibility is not silently relabeled as an action-efficiency gain. Blocked counts
include internal alternative-candidate proposals, not live actions. The large
animation-off block count includes repeated candidate cells belonging to the same
hard-dead signatures during exploratory fallback.

| Flag | Policy zero ON | Policy zero OFF | Common zero ON | Common zero OFF | Blocked ON / OFF | RESETs ON / OFF | Paired common zero-rate reduction, 95% bootstrap |
|---|---|---|---|---|---|---|---|
| `HUD_MASK` | 1/246 | 1/246 | 1/246 | 1/246 | 0 / 0 | 0 / 0 | 0.000 pp [0.000, 0.000] |
| `ANIMATION_DIFF` | 1/246 | 81/244 | 1/246 | 6/244 | 0 / 35735 | 0 / 2 | 2.093 pp [0.000, 4.593] |
| `DEAD_SIGNATURE` | 1/246 | 1/246 | 1/246 | 1/246 | 0 / 0 | 0 / 0 | 0.000 pp [0.000, 0.000] |
| `DOUBLE_RESET_GUARD` | 1/246 | 1/246 | 1/246 | 1/246 | 0 / 0 | 0 / 0 | 0.000 pp [0.000, 0.000] |

Bootstrap uses 5,000 resamples of six matched game/seed pairs with seed 917;
positive reduction means ON has fewer common zero-effect actions. The interval
includes zero for the animation comparison. Other intervals are degenerate from
identical short traces and do not establish equivalence or generalization.
All 30 runs complete zero levels: no late-level evidence or score lift. No public
run reached a reset with completed progress at risk, so reset-loss opportunity
coverage is **0/0 (N/A)**. Observed losses were 0/2 RESETs in the animation-off
runs and 0/0 RESETs for all-on controls. All observed consecutive-reset dispatch
counts were zero. Synthetic fixtures, rather than these short public runs,
establish the guard and HUD behaviors. S1-W prediction coverage/match/simulator
metrics remain N/A because its inference/expect pipeline is not implemented.

Repair/failure log: initial fixture run 11/12 passed (wrong archived summary key
in the test) and Pyright reported three annotation issues. Repair 1 passed 12/12
fixtures with two object-diff annotation errors remaining. Repair 2 passed 19/19
fixtures and type checking; one test-style diagnostic was corrected. The first
public matrix then completed 12/13 attempted runs and failed on `ft09`, seed 7,
`ENABLE_ANIMATION_DIFF=0` when all legal targets were hard-dead. Repair 3 added
visible first-RESET replanning while preserving hard blocks and reference
insistence; its new regression brings the fixture total to 20/20 and the matrix
rerun completes 30/30. No exhausted guard was bypassed or error hidden.

Environment versions: arc-agi 0.9.9, arcengine 0.9.3, NumPy 2.5.4, SciPy 1.18.1,
NetworkX 3.7, pytest 9.1.1, Ruff 0.17.1, Pyright 1.1.414. Public wrapper log versions:
`ls20-9607627b`, `ft09-0d8bbf25`, `vc33-5430563c`. Both environment and policy seeds
are `{7,19}`. Source/package/data settings were fixed across each pair. Optional
scientific packages were installed only in the primary environment; the sandbox
remains engine-free and offline deployment can use the tested stdlib fallback.

**Critic decision:** S1-P functional acceptance and Gate 1 PASS. Handoff is ready
for S1-W. Gates 2–3, causal performance promotion, champion selection and Kaggle
submission remain on hold; no PR push or submission was performed for S1-P.


### S1-P publication preparation — 2026-10-20 (Asia/Calcutta)

Explicit user authorization now includes committing and pushing
`feat/s1-p-perception` and opening a PR against `main`. The existing branch was
fast-forwarded to main's S1-S merge (`f48f9f1`) before committing S1-P; the
validated implementation and archived source hashes are unchanged. Publication
revalidation: `pytest tests/test_perception_s1p.py` PASS 20/20;
`ruff check agent/` PASS; `pyright agent/` PASS, 0 errors / warnings;
`python scripts/audit_containment.py` PASS, `containment_passed=true`, 20/20 checks.
The prior 30/30 paired smoke runs and 25/25 compatibility smokes remain recorded
above; they were not rerun for publication. No score promotion or Kaggle
submission is authorized or claimed.

## S1-W implementation and Sprint 1 integration — 2026-10-20 (Asia/Calcutta)

**Mutation: [REPLACE] evidence-based memory and planning**, with four independent
default-`1` ablations. Executed under explicit user authorization, including
branch/commit/push/PR publication. Integration parent is main `cf020b6`, containing
the merged S1-S and S1-P contracts. The architectural reference remains
`top_programs/baseline_28_score.ipynb`; its recorded SHA256 and all extracted
baseline module hashes are unchanged. No verified champion or score promotion is
inferred from the notebook filename.

### Implemented pipeline

- `agent/my_agent.py` now hosts `WorldModel` and `Playbook`, consuming immutable
  S1-P `Transition`/`FrameView` records, public object/diff/HUD interfaces and
  the S1-S engine-free worker. Generated source never executes in the engine
  host. The archived replay comparison and checked-execution designs are adapted
  to full-history gating and asynchronous framework responses. The extracted BFS
  executes in the worker with its host-only type annotation removed; archived
  helpers themselves remain unchanged.
- `ENABLE_RETRODICTION=1`: host-generated `verify_hypothesis(log_history)` invokes
  each candidate's `step(state, action)` in explicit chunks (default 16, maximum
  chunk 64) over **all** relevant transitions, including earlier episodes and
  observed level/terminal transitions. There is no last-128 scan or total-case
  clamp. Reports bind source/history SHA256, level/game scope, input/excluded/
  eligible/checked/matched counts, every completed chunk and counterexamples.
  Comparison covers the entire settled non-HUD board plus public level/status,
  including unchanged cells. Reset/full-reset/unrendered cases are explicitly
  excluded; this is endpoint consistency, not proof of hidden/transient rules.
  Empty, timed-out, malformed, incomplete and crashed runs cannot grant a pass.
  A consistent source may authorize exactly one probe with concrete differing
  cell predictions; expected cells must also match the candidate's forward
  prediction. New/stale evidence invalidates its dispatch scope. Rejected
  hypotheses cannot fall through to an alternative `actions` response field.
  Toggle `0` exposes an `ablation_bypass`, retaining containment and forward
  prediction validation without labeling the source verified.
- `ENABLE_EXPECT_QUEUE=1`: entire concrete enum/coordinate plans validate before
  enqueueing. Every item requires nonempty settled non-HUD cells and an explicit
  `expect_level_up` boolean. One action is dispatched, then its settled response
  is checked before another. First mismatch, reset, terminal, level boundary,
  stale scope, dispatch failure or perception substitution flushes the queue.
  The final host interception prevents direct-call bypass and silent retargeting
  through mutable complex-action enums. Toggle `0` permits missing/unchecked
  expectations while boundary invalidation and containment remain mandatory.
- `ENABLE_PLAYBOOK_RESET=1`: evidence-backed section memory distinguishes scoped
  historical consistency, counterexamples and unresolved hypotheses, retaining
  source files and coverage references. Live probe counterexamples demote the
  corresponding scoped fact. Distillation at boundaries and conservative
  serialized-byte token ceilings writes immutable `playbook.vNNNN.md` plus
  `playbook.md`, then starts a fresh context from that snapshot. Same-source
  facts on different levels remain separate. Turns append within each context;
  copied complete requests pass S1-S PrefixLock immediately before an optional
  backend callback. Full `log.txt` replay evidence remains outside context.
  Toggle `0` restores oldest-exchange drain with persistent section memory,
  using a fresh context ID to preserve immutable submitted prefixes.
- `ENABLE_TIER1_SIMULATOR=1`: after 40 cumulative zero settled-and-transient
  causal-effect moves on level >=3, require simulator synthesis, executable
  `step` definition, full current-level replay and only then bounded BFS
  (2,000 nodes, depth 32, 10,000 edges, two seconds; worker hard timeout 15s).
  Compile and replay successes are separate events. Goal is a modeled level
  advance/WIN. Without a backend, a partial exact-transition lookup is synthesized
  from observations; unknown/conflicting edges remain unknown. Failed/exhausted
  search is exposed before exploration fallback. Toggle `0` suppresses this
  trigger. Simulator replay remains mandatory even if general retrodiction is off.
- The structured backend handoff (`agent.world.backend(messages)`) is documented
  in `agent/baseline/README.md`. External source/model/wheel mounts remain absent
  as recorded by S1-S; no inference server or remote API is invented. Default
  public gameplay therefore records available-action exploration fallback.
  Cache reuse, actual model throughput and learned-policy score lift remain N/A.

### Final verification and measured denominators

Primary Python 3.12.14 and the separate sandbox interpreter were verified before
code changes. The sandbox lacks the engine package; behavioral containment
passed before candidate runs and again after final agent changes. Only the engine
host downloaded/loaded public games; researcher/worker code never inspected game
implementation files.

| Check | Final result |
|---|---|
| `pytest -q` | PASS, 71/71: 43 world-model, 20 perception, 8 systems cases |
| `ruff check agent/` | PASS; broader `agent/ scripts/` and new fixture lint also PASS |
| `pyright agent/` | PASS, 0 errors / warnings |
| `python scripts/audit_containment.py` | PASS, `containment_passed=true`, 20/20 checks |
| Archived extraction/hash verification; `pip check`; `git diff --check` | PASS |
| Toy incorrect-mechanic replay | 3/3 rejected by zero-action replay; 0 live actions |
| Toy consistent-mechanic replay | 1/1 complete pass; separate fixture permits exactly one probe |
| Toy non-probing rollout coverage / match | 2/2 covered, 2/2 matched; synthetic only |
| Toy simulator compile / replay | Compile 2/3; replay 1/2 compiled candidates; failures deliberately injected |
| Long-history fixture | 145/145 transitions checked in 12 chunks; old counterexample rejected; consistent source passes all 145 |
| Public paired matrix | PASS, 40/40 runs: four games x two seeds x five flag configurations |
| Full public compatibility smoke, cap 2 | PASS, 25/25 games; 3 framework iterations each, 0 levels, score 0.0 |

Evidence: `research_lab/traces/s1_w_validation.json`,
`research_lab/traces/s1_w_smoke.json`, `containment.json` and the synthetic fixtures
in `tests/test_world_model_s1w.py`. Runtime evidence/source/playbook files stay
beside each ignored perception log; public trace hashes are recorded. Reproduce
the toy and matrix validation with
`python scripts/validate_world_model.py --public`; the containment audit must pass
first. `pytest.ini` discovers project `tests/` and `scripts/`, excluding the
separately vendored framework's optional SDK tests.

Public protocol was fixed before running: environment and policy seeds `{7,19}`;
`ls20`, `ft09`, `vc33`, `re86`; cap argument 40 (the inherited `<=` loop executes
41 iterations); all-on versus one world-model flag off at a time; unchanged
dependencies/data/action budgets within pairs. There were eight matched samples
per flag. Every configuration records 328/328 observed responses, 1/328 zero
causal effects, 0 RESETs, and 0 completed levels. All eight paired zero-effect
rate differences per flag are zero; paired bootstrap 95% intervals are [0,0]
(5,000 resamples, seed 917). These degenerate intervals reflect dormant action
gates and do **not** establish equivalence or causal improvement. Four playbook
distillations occur per configuration except playbook-off (zero versions).

Public prediction coverage is absent: 0 predictions across 328 nonboundary
responses per configuration. Hypothesis rejection, match and simulator rates are
**N/A (0/0)**; no backend emitted mechanics/plans and no run reached level >=3
(0/328 late-level opportunities). Synthetic positives and intentional negatives
are reported separately and cannot satisfy game-performance promotion gates.
No late-level completion, >=75% public prediction coverage, >=85% public match
rate, >=5 RHAE lift or >=20% Level 3–5 action reduction has been demonstrated.

Repair accounting: pre-check compilation caught and fixed an f-string typo.
Initial tooling also exposed vendored pytest collection, one fixture style issue,
an inferred dictionary type and missing optional primary-environment scientific
packages. Repair 1 fixed discovery/style/types and installed the existing S1-P
optional dependencies; 54/66 fixtures then passed, with failures tracing to a
blocked disk-backed `__future__` import in the generated driver. Repair 2 removed
that dependency and stripped only the trusted BFS annotation: 66/66 passed with
lint/types clean. Final dispatch/evidence review added five passing regression
cases, bringing the suite to 71/71; no containment weakening or failed-game
fallback was hidden. No third repair cycle was needed.

**Sprint 1 status:** S1-S/S1-P/S1-W functional integration and Gate 1 PASS on this
checkout. Gates 2–3 and statistical promotion remain on hold pending real offline
inference assets and paired late-level policy evidence. This task authorizes the
feature PR only; no champion promotion, merge or Kaggle submission was performed.
