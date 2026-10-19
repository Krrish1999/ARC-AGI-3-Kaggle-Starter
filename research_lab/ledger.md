# ARC-AGI-3 PhD Research Laboratory Ledger

## Current Baseline Status
- **Active Champion Program:** `research_lab/top_programs/v0_franzen_m2_baseline.py`
- **Kaggle Leaderboard Score:** 28.00 RHAE (Franzen M2 Baseline Cluster: 25.77 ± 3.93 SD)
- **Target Promotion Threshold:** >= 33.80 RHAE local multi-seed equivalent (+8.0 two-sided statistical threshold) and >= 75% Forward Prediction Coverage.

---

## Active Sprint Task Packet (Issued by `phd-pi`)
- **Parent Program:** `v0_franzen_m2_baseline.py`
- **Ticket 1 (`phd-perception` | `[EDIT]`):** Implement `ENABLE_HUD_MASK`, `ENABLE_ANIMATION_DIFF`, `ENABLE_DEAD_SIGNATURE`, and `ENABLE_DOUBLE_RESET_GUARD` in `agent/arclog.py` and `agent/my_agent.py`.
- **Ticket 2 (`phd-world-model` | `[REPLACE]`):** Replace FIFO message eviction with `ENABLE_PLAYBOOK_RESET` (`playbook.md` distillation) and add `ENABLE_RETRODICTION` + `ENABLE_EXPECT_QUEUE`.
- **Ticket 3 (`phd-systems` | `[EDIT]`):** Set `ACCELERATOR = "rtx6000"` in `scripts/build_notebook.py`, enforce append-only KV-cache prefix locking, and add 15s subprocess timeouts for `.venv_agent_sandbox`.

---

## Experiment & Ablation Log

| Exp ID | Lead Skill | Mutation Op & Toggle | Containment (`Gate 1`) | Expect Coverage / Match (`Gate 2`) | Local Multi-Seed RHAE (L4+ Completion %) | Critic Verdict (`phd-critic`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-000` | Baseline | Franzen M2 (`All Toggles=0`) | PASS | 0.0% / N/A | 28.00 (L3 Cap on `ls20`, `vc33`) | Baseline Reference |