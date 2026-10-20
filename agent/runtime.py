"""Ablatable prefix contract and level-weighted deployment budgets."""
from __future__ import annotations

import json
import os
import time
from typing import Any

TARGET_SECONDS = 29_700.0
HARD_CAP_SECONDS = 32_400.0
_RUN_STARTED = time.time()


class PrefixLock:
    """Validate full backend requests. Reset requires a fresh context identity.

    Call validate immediately before each backend request, passing its complete
    message list. Serialized snapshots prevent mutation through caller aliases.
    """

    def __init__(self) -> None:
        self._contexts: dict[str, tuple[str, ...]] = {}
        self._active: str | None = None

    def validate(self, context_id: str, messages: list[dict[str, Any]]) -> None:
        if not context_id or not messages:
            raise ValueError("context identity and messages are required")
        current = tuple(json.dumps(m, sort_keys=True, separators=(",", ":")) for m in messages)
        previous = self._contexts.get(context_id)
        if os.getenv("ENABLE_PREFIX_LOCK", "1") == "1":
            if previous is not None and context_id != self._active:
                raise ValueError("closed context cannot be reopened")
            if previous is not None and current[:len(previous)] != previous:
                raise ValueError("prefix mutation requires a new context identity")
        self._contexts[context_id] = current
        self._active = context_id


class LevelBudget:
    """Per-environment weighted allowance with unused-time rollover.

    The caller supplies the measured environment allocation and actual level
    count; no game-specific assumptions. begin_level returns a deadline suitable
    for request timeouts. expired enforces both that deadline and global caps.
    """

    def __init__(self, *, env_seconds: float, levels: int,
                 run_started: float | None = None, now: float | None = None) -> None:
        if not 0 < env_seconds <= TARGET_SECONDS or levels < 1:
            raise ValueError("invalid environment allocation or level count")
        now = time.time() if now is None else now
        self.run_started = now if run_started is None else run_started
        self.target_deadline = self.run_started + TARGET_SECONDS
        self.hard_deadline = self.run_started + HARD_CAP_SECONDS
        self.env_deadline = min(now + env_seconds, self.target_deadline)
        self.env_seconds = env_seconds
        self.levels = levels
        self.weight_sum = levels * (levels + 1) / 2
        self.rollover = 0.0
        self.level = 0
        self.level_deadline = self.env_deadline

    def begin_level(self, level: int, *, now: float | None = None) -> float:
        now = time.time() if now is None else now
        if not 1 <= level <= self.levels or level <= self.level:
            raise ValueError("levels must advance monotonically")
        if self.level:
            self.rollover = self.level_deadline - now
        self.level = level
        enabled = os.getenv("ENABLE_LEVEL_BUDGET", "1") == "1"
        self.level_deadline = min(
            self.env_deadline,
            now + max(0.0, self.env_seconds * level / self.weight_sum + self.rollover)
        ) if enabled else self.env_deadline
        return self.level_deadline

    def remaining(self, *, now: float | None = None) -> float:
        now = time.time() if now is None else now
        deadline = min(self.hard_deadline, self.target_deadline, self.env_deadline)
        if os.getenv("ENABLE_LEVEL_BUDGET", "1") == "1":
            deadline = min(deadline, self.level_deadline)
        return max(0.0, deadline - now)

    def generation_tokens(self, requested: int, *, now: float | None = None) -> int:
        """Degrade reasoning near exhaustion; zero requests best-effort actions."""
        left = self.remaining(now=now)
        if left <= 0:
            return 0
        return min(requested, 256) if left < 15 else requested


def run_expired() -> bool:
    """Shared notebook start includes setup overhead and survives process launch."""
    started = float(os.getenv("ARC3_RUN_STARTED", str(_RUN_STARTED)))
    return time.time() >= started + min(TARGET_SECONDS, HARD_CAP_SECONDS)
