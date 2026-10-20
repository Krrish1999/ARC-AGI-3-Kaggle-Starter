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
import tempfile
import time
from pathlib import Path
from typing import Any

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

try:
    from .arclog import PerceptionLog
    from .runtime import TARGET_SECONDS, LevelBudget, PrefixLock, run_expired
    from .sandbox import run_sandbox
except ImportError:
    from agent.arclog import PerceptionLog
    from agent.runtime import TARGET_SECONDS, LevelBudget, PrefixLock, run_expired
    from agent.sandbox import run_sandbox


class MyAgent(Agent):
    """Picks legal actions uniformly at random. Replace with your strategy."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # A fixed control seed supports paired ablations without shared RNG state.
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        self._rng = random.Random(int(os.getenv("ARC3_SEED", str(seed))))
        log_root = Path(os.getenv("ARC3_LOG_DIR", "scratch"))
        log_root.mkdir(parents=True, exist_ok=True)
        self.perception = PerceptionLog(Path(tempfile.mkdtemp(prefix="perception-", dir=log_root)) / "log.txt")
        self._perception_started = False
        self._observation: FrameData | None = None
        self._dispatch_approval: tuple[str, tuple[int, int] | None, int] | None = None
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

        self._ensure_perception(latest_frame)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            proposal = GameAction.RESET
        else:
            legal = self._legal_actions(latest_frame)
            if not legal:
                raise RuntimeError("no advertised non-reset action; replan required")
            proposal = self._sample(self._rng.choice(legal))
        action: Any = self._resolve_action(proposal, latest_frame)
        self._dispatch_approval = (action.name, self._coordinates(action), self.perception.current.step)
        action.reasoning = {"why": "generic available-action exploration", "evidence_step": self.perception.current.step}
        if self.level_budget is not None and self.level_budget.generation_tokens(1) == 0:
            action.reasoning["budget"] = "level reasoning budget exhausted; best-effort action"
        return action

    def _ensure_perception(self, observation: FrameData) -> None:
        self._observation = observation
        if not self._perception_started:
            self.perception.start(observation.frame, level=observation.levels_completed+1,
                                  state=observation.state.name)
            self._perception_started = True

    @staticmethod
    def _legal_actions(observation: FrameData) -> list[GameAction]:
        return [a for a in GameAction if a is not GameAction.RESET
                and a.value in observation.available_actions]

    def _sample(self, action: GameAction) -> GameAction:
        if action.is_complex():
            grid = self.perception.current.grid
            height = len(grid) or 64
            width = len(grid[0]) if grid else 64
            action.set_data({"x": self._rng.randrange(width), "y": self._rng.randrange(height)})
        return action

    @staticmethod
    def _coordinates(action: GameAction) -> tuple[int, int] | None:
        if not action.is_complex():
            return None
        data = action.action_data.model_dump()
        return int(data['x']), int(data['y'])

    def _resolve_action(self, proposal: GameAction, observation: FrameData,
                        *, reference: bool = True) -> GameAction:
        coordinates = self._coordinates(proposal)
        legal = self._legal_actions(observation)
        reason = (self.perception.blocked_reason(proposal.name, coordinates, reference=reference)
                  if proposal is GameAction.RESET or proposal in legal else 'unavailable_action')
        if reason is None:
            return proposal
        self.perception.note_blocked(proposal.name, coordinates, reason)
        self._rng.shuffle(legal)
        for candidate in legal:
            grid = self.perception.current.grid
            h, w = len(grid), len(grid[0]) if grid else 0
            attempts = w*h if candidate.is_complex() else 1
            offset = self._rng.randrange(attempts) if attempts else 0
            for i in range(attempts):
                if candidate.is_complex():
                    cell = (offset+i) % attempts
                    candidate.set_data({'x': cell % w, 'y': cell // w})
                coordinates = self._coordinates(candidate)
                reason = self.perception.blocked_reason(candidate.name, coordinates)
                if reason is None:
                    return candidate
                self.perception.note_blocked(candidate.name, coordinates, reason)
        # The archived reference guard explicitly permits immediate insistence.
        # Reconsider only the final refused reference action, never a hard dead
        # signature or consecutive RESET. The event remains visible in the log.
        if legal and reason == 'reference_noop':
            candidate = legal[-1]
            coordinates = self._coordinates(candidate)
            if self.perception.blocked_reason(candidate.name, coordinates) is None:
                self.perception.note_replan('reference_noop_override', candidate.name)
                return candidate
        # If all object signatures are exhausted, a single first RESET starts a
        # fresh attempt. Keep hard refusals intact; log the replan rather than
        # silently reviving a dead target. A repeated RESET is still forbidden.
        if legal and self.perception.last_action != 'RESET':
            self.perception.note_replan('admissible_targets_exhausted', 'RESET')
            return GameAction.RESET
        raise RuntimeError("all advertised actions blocked; replan required before dispatch")

    def do_action_request(self, action: GameAction) -> FrameData:
        # Final interception also protects direct framework/tool calls that did
        # not originate in choose_action. Snapshot mutable enum data before step.
        if self._observation is None:
            self._ensure_perception(self._convert_raw_frame_data(self.arc_env.observation_space))
        assert self._observation is not None
        approval = (action.name, self._coordinates(action), self.perception.current.step)
        selected = self._resolve_action(action, self._observation,
                                        reference=approval != self._dispatch_approval)
        self._dispatch_approval = None
        coordinates = self._coordinates(selected)
        try:
            response = super().do_action_request(selected)
        except Exception as error:
            self.perception.note_dispatch_error(selected.name, coordinates, type(error).__name__)
            logging.getLogger(__name__).exception("live dispatch failed; no observed transition recorded")
            raise
        self.perception.record(selected.name, coordinates, response.frame,
                               level=response.levels_completed+1, state=response.state.name,
                               full_reset=response.full_reset)
        self._observation = response
        return response
