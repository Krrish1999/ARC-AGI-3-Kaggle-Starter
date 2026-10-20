# S1-P / S1-W source handoff

`cell3.patch` is the exact Cell 3 unified diff (minus the writefile directive).
`dhia_0_1_21.patch` is the Cell 5 base64/LZMA literal decoded without executing
notebook code. `manifest.json` retains notebook/patch/source hashes and all 49
expected final upstream hashes. The notebook parent has not been modified.

Complete additions are importable as `agent.baseline.<module>` locally and
`agents.templates.baseline.<module>` in the built framework notebook:

- `frame_diff.compute_frame_diff`, `animation`, `noop_repeat_guard`: perception.
- `grid_coverage`, `state_contracts`, `transition_evidence`: Dhia evidence helpers.
- `bounded_search.bounded_bfs`, `checked_execution.checked_execute`,
  `replay_compare.replay_named_transitions`: search, execution and replay.
- `retained_functions`, `priority_scheduler`: retention and scheduler reference.

These are reference mechanisms, not acceptance of their behavioral assumptions.
The replay module retains the archived limited-history semantics; S1-W must add
its mandatory full-history gate. `retained_functions` executes restored source
and must only run inside the secured worker. `grid_display` is a small local
hexadecimal display adapter because upstream `grid_utils` is absent.

The existing FrameView, segmentation and tool-agent files are *modified*, rather
than supplied in full, by these patches. They cannot be faithfully reconstructed
without the external base source bundle. Their exact hunks are preserved, and the
manifest lists these gaps. Do not concatenate fragments or claim those classes
are available. `research_lab/offline_inventory.json` records missing mounts.

`MyAgent.execute_planner(source)` delegates to `sandbox.run_sandbox` with a
15-second limit. Pass immutable evidence as literals or JSON in source; workers
cannot read files, launch children or open sockets. Host code stores results.
The worker preloads math/json/collections/ctypes; further disk-backed imports are
blocked. To use these helper modules in a worker, embed their source and any
required stdlib dependencies before execution or extend the trusted bootstrap
with an explicitly reviewed dependency. A venv alone is never the boundary.

S1-W now integrates these mechanisms in `my_agent.WorldModel`. Archived modules
and their provenance hashes stay unchanged: full-history `replay_transitions`
replaces the reference's tail/case clamps, `execute_plan` adapts checked execution
to one framework response per dispatch, and `search_plan` embeds the extracted
bounded BFS in the secured worker (only its host typing annotation is stripped).
Replay reports include source/history hashes, scope, exclusions, chunks and exact
coverage. Empty, limited and failed runs cannot authorize probes or simulator
search. Every probe requires differing concrete predictions and permits one move.

Synthesized source must define `step(state, action)` and return a JSON-compatible
dictionary containing `grid` (integer colors), `level` and public status `state`.
Actions use actual enum names, e.g. `ACTION1` or `ACTION6` plus `x`/`y`.
Plans contain `expect: [{"x": 1, "y": 2, "value": 4}]` and an explicit boolean
`expect_level_up`. These are settled non-HUD predictions, not animation cells.
The entire queue is flushed on mismatch, reset, terminal or level boundary.

An optional `agent.world.backend(messages)` callback returns a dictionary with
`hypotheses` (source, probe, alternative_expect, optional level/game scope),
`actions`, or `simulator_source`. Requests pass PrefixLock immediately before
the callback; copied requests cannot mutate retained prefixes. No remote API or
unavailable model installation is implied. Without a backend, the agent records
exploration fallback. After 40 wasted moves on level >= 3 it synthesizes a partial
exact-transition lookup from observed evidence, compiles/replays it, and searches
only known edges. An exhausted/failed search is recorded before fallback.

Runtime artifacts live beside each perception `log.txt`: `world_model.jsonl`,
source-hashed mechanic files, versioned `playbook.vNNNN.md`, and `playbook.md`.
Playbooks distinguish scoped historical consistency, counterexamples and open
hypotheses. Immutable snapshots begin fresh contexts at boundaries/token ceilings;
all transition evidence remains outside context. The ceiling uses a conservative
serialized-byte bound because model tokenizer assets are unavailable. Each of the
four S1-W toggles defaults to `"1"`; `0` restores FIFO drain with persistent section
memory, optional replay, unchecked expectations, or no Tier-1 trigger respectively.
Even in ablations containment and reset/boundary queue invalidation remain active.

`MyAgent.prefix_lock.validate(context_id, complete_messages)` is the mandatory
call immediately before S1-W backend requests. Fresh contexts require fresh IDs.
`MyAgent.level_budget` uses public level counts and proportional budgets with
rollover. Pass the intended environment allocation with `ARC3_ENV_SECONDS`;
`ARC3_RUN_STARTED` anchors the shared run ceiling. No model/cache benchmark is
claimed before the source bundle, wheels, target weights and draft weights exist.
