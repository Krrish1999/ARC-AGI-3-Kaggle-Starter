"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

The default body below is a port of the Stochastic Goose / random_agent
sample — a known-good baseline that produces a valid submission and
proves your end-to-end pipeline works. Replace `choose_action` with your
real strategy.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

try:
    from .runtime import TARGET_SECONDS, LevelBudget, PrefixLock, run_expired
    from .sandbox import run_sandbox
except ImportError:
    from agent.runtime import TARGET_SECONDS, LevelBudget, PrefixLock, run_expired
    from agent.sandbox import run_sandbox


class MyAgent(Agent):
    """Picks legal actions uniformly at random. Replace with your strategy."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seed per game_id so replays from the same game are reproducible but
        # different games explore independently.
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)
        self.prefix_lock = PrefixLock()
        self.level_budget: LevelBudget | None = None

    def execute_planner(self, source: str) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[1]
        python = Path(os.getenv("ARC3_SANDBOX_PYTHON", str(root / ".venv_agent_sandbox/bin/python")))
        result = run_sandbox(source, python=python, cwd=root)
        if not result["ok"]:
            logging.getLogger(__name__).warning("planner failure: %s", result)
        return result

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        return run_expired() or latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # Public level metadata supplies weights; an unavailable count defers
        # scheduling rather than inventing an environment structure.
        if latest_frame.win_levels > 0:
            if self.level_budget is None:
                self.level_budget = LevelBudget(
                    env_seconds=float(os.getenv("ARC3_ENV_SECONDS", str(TARGET_SECONDS))),
                    levels=latest_frame.win_levels,
                    run_started=float(os.getenv("ARC3_RUN_STARTED", str(time.time()))),
                )
            level = min(latest_frame.levels_completed + 1, latest_frame.win_levels)
            if level > self.level_budget.level:
                self.level_budget.begin_level(level)

        # First call or after a death → reset the level.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET

        candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        action: Any = random.choice(candidate_actions)  # SDK permits dynamic reasoning metadata

        if action.is_complex():
            # ACTION6 takes (x, y) coordinates on a 64×64 grid.
            action.set_data(
                {"x": random.randint(0, 63), "y": random.randint(0, 63)}
            )
            action.reasoning = {"why": "random complex action"}
        else:
            action.reasoning = f"random simple action: {action.value}"
        if self.level_budget is not None and self.level_budget.generation_tokens(1) == 0:
            action.reasoning = {"why": "level reasoning budget exhausted; best-effort action"}
        return action
