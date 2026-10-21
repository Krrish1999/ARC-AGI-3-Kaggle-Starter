# agent/AGENTS.md — Deployed Harness Guardrails

## 1. Scope of `agent/`
Files in this directory (`my_agent.py` and `arclog.py`) are bundled directly into the offline Kaggle submission notebook.
- **No External Network Calls:** Never add `requests`, `urllib`, or external API SDK calls that trigger during evaluation without a local offline SGLang (`Pennyroyal`) endpoint.
- **No Engine Leakage:** Never import `environment_files` or inspect private attributes of the `arc_agi` environment wrapper inside `my_agent.py`. All code executed by the LLM's internal REPL must run inside `.venv_agent_sandbox` or a restricted globals dictionary that blocks filesystem traversal outside `scratch/` and `log.txt`.

## 2. Required Environment-Variable Ablation Toggles
Every heuristic or subsystem added to `agent/my_agent.py` or `agent/arclog.py` must be controllable via environment variables (defaulting to `"1"`):
- `ENABLE_HUD_MASK`: Masks monotonic border strips (`1x64` or `64x1`) from causal action diffs.
- `ENABLE_ANIMATION_DIFF`: Captures transient cell changes across intermediate frames (`frames[:-1]`).
- `ENABLE_DEAD_SIGNATURE`: Blocks repeated interactions with object signatures that yield 0 non-HUD diffs twice consecutively.
- `ENABLE_DOUBLE_RESET_GUARD`: Prevents two consecutive `RESET` commands from resetting the environment back to Level 1.
- `ENABLE_PLAYBOOK_RESET`: Replaces sliding-window FIFO eviction with structured `playbook.md` distillation at level boundaries.
- `ENABLE_RETRODICTION`: Requires candidate mechanic scripts to pass 100% historical replay over `log.txt` before live probing.
- `ENABLE_EXPECT_QUEUE`: Executes multi-step `[ACTIONS]` plans gated by per-step `expect` cell predictions.
- `ENABLE_TIER1_SIMULATOR`: Triggers `step(state, action)` Python synthesis and bounded BFS after 40 wasted moves on Level >= 3.
- `ENABLE_ZERO_ACTION_REPLAN`: Allows at most three proposal attempts under one shared reasoning allowance; retries classified proposal errors and historical counterexamples without weakening replay, exact predictions or containment. Setting `0` retains one proposal; infrastructure failures remain visible.
