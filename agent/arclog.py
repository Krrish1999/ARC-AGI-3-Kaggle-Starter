"""Immutable visual evidence and pre-dispatch perception guards.

Grid/cell internals use (row, column); public coordinates and bounding boxes use
(x, y). Animation belongs to one response's ``frame`` list, never outer history.
The missing upstream FrameView base is adapted here; complete baseline diff,
animation, no-op and transition-evidence modules remain the shared algorithms.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from .baseline.animation import collapse_frames, normalize_frames, summarize_animation
from .baseline.frame_diff import compute_frame_diff
from .baseline.grid_display import ARC_COLOR_CHARS, format_grid_ascii
from .baseline.noop_repeat_guard import NoopRepeatGuard, action_signature
from .baseline.transition_evidence import transition_evidence

# Optional accelerators. Equivalent stdlib fallbacks keep the offline notebook
# usable without assuming scientific wheels exist in the competition dataset.
try:
    import numpy as np
    from scipy import ndimage
except ImportError:
    np = None
    ndimage = None
try:
    import networkx as nx
except ImportError:
    nx = None

Grid = tuple[tuple[int, ...], ...]
XY = tuple[int, int]
Signature = tuple[str, str, int]


def enabled(name: str) -> bool:
    return os.getenv(name, "1") == "1"


def auditable() -> bool:
    """Default-on Generation 3 contract; 0 is the archived harness control."""
    return enabled('ENABLE_AUDITABLE_WORLD_MODEL')


def freeze_grid(raw: Any) -> Grid:
    grid: Grid = normalize_frames([raw])[0]
    if grid and (not grid[0] or any(len(row) != len(grid[0]) for row in grid)):
        raise ValueError("grid must be rectangular")
    if len(grid) > 256 or (grid and len(grid[0]) > 256):
        raise ValueError("grid too large")
    if any(not 0 <= cell < 16 for row in grid for cell in row):
        raise ValueError("grid colors must be in [0, 15]")
    return grid


def _object_signature(cells: tuple[XY, ...], color: int) -> tuple[str, int | None, int, str, str]:
    """Port the notebook's translation/rotation canonicalization, color-free pose.

    Exact hash convention matches Cell 3's segmentation hunks. SHA1 is only an
    opaque geometry identifier, never a security or provenance decision.
    """
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r-r0, c-c0) for r, c in cells)
    variants = [norm]
    for _ in range(3):
        max_r = max(r for r, _ in variants[-1])
        variants.append(sorted((c, max_r-r) for r, c in variants[-1]))

    def cell_hash(shape: list[XY], label: str) -> str:
        return hashlib.sha1(repr((label, shape)).encode(), usedforsecurity=False).hexdigest()[:16]

    hashes = [cell_hash(v, ARC_COLOR_CHARS[color]) for v in variants]
    shape_hashes = [cell_hash(v, "__shape__") for v in variants]
    k = shape_hashes.index(min(shape_hashes))
    symmetry = {1: 4, 2: 2, 4: 1}[len(set(shape_hashes))]
    rotation = None if symmetry == 4 else (((4-k) % 4)*90) % (180 if symmetry == 2 else 360)
    return hashes[k], rotation, symmetry, hashes[0], shape_hashes[0]


@dataclass(frozen=True)
class BoardObject:
    object_id: int
    color: int
    bounding_box: tuple[int, int, int, int]
    area: int
    centroid: tuple[float, float]
    shape_hash: str
    adjacency_list: tuple[int, ...]
    cells_rc: tuple[XY, ...]
    canonical_hash: str
    rotation: int | None
    rotational_symmetry: int
    pose_hash: str

    def node(self) -> dict[str, Any]:
        """Compatibility with the archived segmentation/frame-diff contract."""
        return {"id": self.object_id, "color": ARC_COLOR_CHARS[self.color],
                "hash": self.canonical_hash, "shape_hash": self.shape_hash,
                "pose_hash": self.pose_hash, "rotation": self.rotation,
                "rotational_symmetry": self.rotational_symmetry,
                "pixels": self.area, "boundary": self.cells_rc,
                "bbox_rc": (self.bounding_box[1], self.bounding_box[0],
                            self.bounding_box[3], self.bounding_box[2])}


def _components(grid: Grid) -> list[tuple[XY, ...]]:
    if not grid:
        return []
    if ndimage is not None and np is not None:
        array = np.asarray(grid)
        groups = []
        for color in np.unique(array):
            labeled: Any = ndimage.label(array == color)
            labels, count = labeled
            for label in range(1, count+1):
                groups.append(tuple((int(r), int(c)) for r, c in np.argwhere(labels == label)))
        return sorted(groups, key=lambda cells: cells[0])
    seen: set[XY] = set()
    groups = []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if (r, c) in seen:
                continue
            queue = deque([(r, c)])
            seen.add((r, c))
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if (0 <= nr < len(grid) and 0 <= nc < len(row)
                            and (nr, nc) not in seen and grid[nr][nc] == color):
                        seen.add((nr, nc))
                        queue.append((nr, nc))
            groups.append(tuple(sorted(cells)))
    return groups


@dataclass(frozen=True)
class FrameView:
    grid: Grid
    level: int = 1
    episode: int = 0
    step: int = 0
    state: str = "NOT_FINISHED"

    @property
    def ascii(self) -> str:
        return format_grid_ascii(self.grid)

    @cached_property
    def segmentation(self) -> tuple[BoardObject, ...]:
        groups = _components(self.grid)
        owner = {cell: i for i, cells in enumerate(groups) for cell in cells}
        edges: set[tuple[int, int]] = set()
        for (r, c), i in owner.items():
            for other in ((r+1, c), (r, c+1)):
                j = owner.get(other)
                if j is not None and j != i:
                    edges.add((min(i, j), max(i, j)))
        graph = nx.Graph() if nx is not None else None
        if graph is not None:
            graph.add_nodes_from(range(len(groups)))
            graph.add_edges_from(edges)
        objects = []
        for i, cells in enumerate(groups):
            rows, cols = zip(*cells)
            color = self.grid[rows[0]][cols[0]]
            signature = _object_signature(cells, color)
            adjacent = (tuple(sorted(graph.neighbors(i))) if graph is not None else
                        tuple(sorted(b if a == i else a for a, b in edges if i in (a, b))))
            objects.append(BoardObject(
                i, color, (min(cols), min(rows), max(cols), max(rows)), len(cells),
                (sum(cols)/len(cells), sum(rows)/len(cells)), signature[4], adjacent,
                cells, signature[0], signature[1], signature[2], signature[3]))
        return tuple(objects)

    @cached_property
    def _owners(self) -> dict[XY, BoardObject]:
        return {cell: obj for obj in self.segmentation for cell in obj.cells_rc}

    def object_at(self, x: int, y: int) -> BoardObject | None:
        return self._owners.get((y, x))


def _cell(grid: Grid, x: int, y: int) -> int | None:
    return grid[y][x] if y < len(grid) and x < len(grid[y]) else None


@dataclass(frozen=True)
class CellChange:
    x: int
    y: int
    old: int | None
    new: int | None


@dataclass(frozen=True)
class AnimationChange:
    frame_index: int
    x: int
    y: int
    old: int | None
    new: int | None


def _diff(before: Grid, after: Grid) -> tuple[CellChange, ...]:
    return tuple(CellChange(x, y, _cell(before, x, y), _cell(after, x, y))
                 for y in range(max(len(before), len(after)))
                 for x in range(max(len(before[y]) if y < len(before) else 0,
                                    len(after[y]) if y < len(after) else 0))
                 if _cell(before, x, y) != _cell(after, x, y))


@dataclass(frozen=True)
class Transition:
    step: int
    action: str
    coordinates: XY | None
    signature: Signature | None
    before: FrameView
    intermediate: tuple[Grid, ...]
    after: FrameView
    raw_settled: tuple[CellChange, ...]
    settled: tuple[CellChange, ...]
    raw_transient: tuple[AnimationChange, ...]
    transient: tuple[AnimationChange, ...]
    hud_mask: frozenset[XY]
    reset: bool
    terminal: bool
    level_changed: bool
    full_reset: bool
    object_diffs_json: str
    evidence_json: str
    remaining_step_budget_estimate: int | None

    @property
    def zero_causal_diff(self) -> bool:
        return not self.settled and not self.transient

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result['hud_mask'] = sorted(self.hud_mask)
        result['object_diffs'] = json.loads(result.pop('object_diffs_json'))
        result['evidence'] = json.loads(result.pop('evidence_json'))
        result['zero_causal_diff'] = self.zero_causal_diff
        return result


class PerceptionLog:
    """One environment's append-only evidence; no mutable process-global state."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path
        self.current = FrameView(())
        self._records: list[Transition] = []
        self._blocked: list[str] = []
        self._scope: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        self._zeros: dict[Signature, int] = {}
        self._reference = NoopRepeatGuard()
        self.last_action: str | None = None
        self._hud: frozenset[XY] = frozenset()
        self.remaining_step_budget_estimate: int | None = None
        self._journal_hash = hashlib.sha256()

    @property
    def records(self) -> tuple[Transition, ...]:
        return tuple(self._records)

    def history(self) -> tuple[Transition, ...]:
        """Read and authenticate the complete journal before issuing evidence.

        The host owns this read; generated code receives JSON through stdin.
        Missing, edited, extra or truncated transitions cannot authorize a move.
        Non-transition audit events are deliberately kept out of mechanics.
        """
        if self.log_path is None:
            raise ValueError('retrodiction requires persistent log.txt')
        raw = self.log_path.read_bytes()
        if hashlib.sha256(raw).digest() != self._journal_hash.digest():
            raise ValueError('log.txt append-only evidence was modified')
        entries = [json.loads(line) for line in raw.splitlines()]
        logged = [entry for entry in entries if entry.get('type') == 'transition']
        if len(logged) != len(self._records):
            raise ValueError('log.txt transition count differs from immutable history')
        for entry, transition in zip(logged, self._records, strict=True):
            expected = json.loads(json.dumps(transition.to_dict()))
            if any(entry.get(key) != value for key, value in expected.items()):
                raise ValueError(f'log.txt evidence changed at step {transition.step}')
        return self.records

    @property
    def dead_signatures(self) -> tuple[Signature, ...]:
        return tuple(sorted(key for key, count in self._zeros.items() if count >= 2))

    @property
    def blocked_proposals(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(entry) for entry in self._blocked)

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(payload, separators=(',', ':'))+'\n'
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(line)
            self._journal_hash.update(line.encode())

    @staticmethod
    def _snapshot(frame: FrameView, action: str = '', **flags: Any) -> dict[str, Any]:
        return {'grid': frame.grid, 'step': frame.step, 'level': frame.level,
                'episode': frame.episode, 'action': action, 'result': flags}

    def start(self, grids: Any, *, level: int, state: str = 'NOT_FINISHED') -> None:
        if self._records:
            raise ValueError('cannot restart an existing log')
        frames = normalize_frames(grids)
        self.current = FrameView(freeze_grid(frames[-1]) if frames else (), level, state=state)
        self._scope = [self._snapshot(self.current)]
        self._history = list(self._scope)
        self._reference.note_level(level)
        if auditable():
            self._emit({'type': 'initial_observation', **self._snapshot(self.current)})

    def synchronize(self, grids: Any, *, level: int, state: str) -> None:
        """An out-of-band observation is evidence, never an invented action."""
        frames = normalize_frames(grids)
        before = self.current
        self.current = FrameView(freeze_grid(frames[-1]) if frames else before.grid,
                                 level, before.episode+1, before.step, state)
        self._zeros.clear()
        self._hud = frozenset()
        self.remaining_step_budget_estimate = None
        self._scope = [self._snapshot(self.current)]
        self._history.append(self._scope[0])
        self._emit({'type': 'observation_resync', 'live_action': False,
                    'before': asdict(before), 'after': asdict(self.current)})

    def _state_hash(self) -> str:
        cells = tuple((x, y, color) for y, row in enumerate(self.current.grid)
                      for x, color in enumerate(row) if (x, y) not in self.get_hud_mask())
        return hashlib.sha256(repr(cells).encode()).hexdigest()

    def signature(self, action: str, coordinates: XY | None) -> Signature | None:
        if coordinates is None:
            return None
        obj = self.current.object_at(*coordinates)
        return (action, obj.shape_hash, obj.color) if obj is not None else None

    def is_dead_signature(self, action_type: str, target_shape_hash: str, target_color: int) -> bool:
        return (enabled('ENABLE_DEAD_SIGNATURE')
                and self._zeros.get((action_type, target_shape_hash, target_color), 0) >= 2)

    def blocked_reason(self, action: str, coordinates: XY | None = None,
                       *, expected_step: int | None = None, reference: bool = True) -> str | None:
        if expected_step is not None and expected_step != self.current.step:
            return 'stale_observation'
        if (enabled('ENABLE_DOUBLE_RESET_GUARD') and action == 'RESET'
                and self.last_action == 'RESET'):
            return 'consecutive_reset'
        signature = self.signature(action, coordinates)
        if signature is not None and self.is_dead_signature(*signature):
            return 'dead_signature'
        # Preserve the archived state/position block-then-insist reference path.
        # It is weaker/different than the new hard shape+color filter and retains
        # the parent activation boundary (level >= 2).
        if (reference and not enabled('ENABLE_DEAD_SIGNATURE') and self.current.level >= 2
                and action != 'RESET' and self._reference.should_block(
                    self._state_hash(), self._action_sig(action, coordinates))):
            return 'reference_noop'
        return None

    @staticmethod
    def _action_sig(action: str, coordinates: XY | None) -> str:
        return action_signature({'action': action, 'row': coordinates[1], 'col': coordinates[0]}
                                if coordinates is not None else action)

    def note_blocked(self, action: str, coordinates: XY | None, reason: str) -> None:
        payload = {'type': 'blocked_proposal', 'step': self.current.step,
                   'level': self.current.level, 'episode': self.current.episode,
                   'action': action, 'coordinates': coordinates, 'reason': reason,
                   'live_action': False}
        self._blocked.append(json.dumps(payload))
        self._emit(payload)

    def note_replan(self, reason: str, action: str) -> None:
        self._emit({'type': 'replan', 'step': self.current.step, 'reason': reason,
                    'proposed_action': action, 'live_action': False})

    def note_dispatch_error(self, action: str, coordinates: XY | None, error_type: str) -> None:
        self._emit({'type': 'dispatch_error', 'step': self.current.step,
                    'action': action, 'coordinates': coordinates,
                    'error_type': error_type, 'execution_status': 'unknown'})

    def _detect_hud(self) -> frozenset[XY]:
        self.remaining_step_budget_estimate = None
        if not enabled('ENABLE_HUD_MASK') or len(self._scope) < 4:
            return frozenset()
        history = self._scope[-4:]
        varied_actions = len({f['action'] for f in history[1:]}) >= 2
        if not varied_actions and (not auditable() or not self._hud):
            return frozenset()
        grids: list[Grid] = [f['grid'] for f in history]
        h, w = len(grids[-1]), len(grids[-1][0]) if grids[-1] else 0
        if not h or not w or any((len(g), len(g[0]) if g else 0) != (h, w) for g in grids):
            return frozenset()
        timeline = [grids[0], *(g for f in history[1:]
                                for g in (f.get('animation_grids') or (f['grid'],)))]
        if any((len(g), len(g[0]) if g else 0) != (h, w) for g in timeline):
            return frozenset()
        mask: set[XY] = set()
        estimates = []
        borders = [tuple((x, y) for x in range(w)) for y in sorted({0, h-1})]
        borders += [tuple((x, y) for y in range(h)) for x in sorted({0, w-1})]
        for cells in borders:
            if not varied_actions and not set(cells) <= self._hud:
                continue  # A learned strip still needs monotonic visual evidence.
            stripes = [tuple(g[y][x] for x, y in cells) for g in grids]
            animated = [tuple(g[y][x] for x, y in cells) for g in timeline]
            palette = set().union(*(set(line) for line in animated))
            if len(palette) != 2 or any(sum(a != b for a, b in itertools.pairwise(line)) > 1
                                       for line in stripes):
                continue
            for color in sorted(palette):
                occupied = [{i for i, v in enumerate(line) if v == color} for line in stripes]
                micro_occupied = [{i for i, v in enumerate(line) if v == color} for line in animated]
                if (not all(b < a for a, b in itertools.pairwise(occupied))
                        or not all(b <= a for a, b in itertools.pairwise(micro_occupied))):
                    continue
                mask.update(cells)
                counts = [len(part) for part in occupied]
                deltas = [a-b for a, b in itertools.pairwise(counts)]
                if len(set(deltas)) == 1:
                    estimates.append(math.ceil(counts[-1]/deltas[0]))
        self.remaining_step_budget_estimate = min(estimates) if estimates else None
        return frozenset(mask)

    def record(self, action: str, coordinates: XY | None, grids: Any, *,
               level: int, state: str = 'NOT_FINISHED', full_reset: bool = False,
               level_completed: bool = False) -> Transition:
        """Record a confirmed response. Refusals/failures never create transitions."""
        before = self.current
        frames = tuple(freeze_grid(g) for g in normalize_frames(grids))
        # Empty terminal responses carry no rendered animation; preserve the last
        # board and terminal flags without inventing an observed board effect.
        post = frames[-1] if frames else before.grid
        reset = action == 'RESET'
        terminal = state in ('GAME_OVER', 'WIN')
        level_changed = level != before.level or level_completed
        boundary = reset or terminal or level_changed or full_reset
        after = FrameView(post, level, before.episode + int(boundary), before.step+1, state)
        target = self.signature(action, coordinates)
        snapshot = self._snapshot(after, action, executed=True, game_over=state == 'GAME_OVER',
                                  level_completed=level_changed, run_complete=state == 'WIN')
        snapshot['animation_grids'] = frames
        self._history.append(snapshot)
        self._scope.append(snapshot)
        self._hud = self._detect_hud() if not boundary else frozenset()
        raw_settled = _diff(before.grid, post)
        chain = collapse_frames(before.grid, frames)
        summary = summarize_animation(chain)
        repeated: set[XY] = set()
        events = [AnimationChange(i, d.x, d.y, d.old, d.new)
                  for i, (a, b) in enumerate(itertools.pairwise(chain)) for d in _diff(a, b)]
        seen: set[XY] = set()
        for event in events:
            xy = (event.x, event.y)
            if xy in seen:
                repeated.add(xy)
            seen.add(xy)
        raw_transient = tuple(e for e in events if (e.x, e.y) in repeated
                              or (auditable() and frames[:-1]))
        transient = tuple(e for e in raw_transient if (e.x, e.y) not in self._hud
                          ) if enabled('ENABLE_ANIMATION_DIFF') else ()
        settled = tuple(d for d in raw_settled if (d.x, d.y) not in self._hud)
        same_shape = (len(before.grid), len(before.grid[0]) if before.grid else 0) == (
            len(post), len(post[0]) if post else 0)
        object_diffs: dict[str, Any] = (compute_frame_diff(before.grid, post,
                                          [o.node() for o in before.segmentation],
                                          [o.node() for o in after.segmentation], max_group_match=64)
                        if same_shape else {'status': 'shape_changed'})
        # The archived object diff is raw evidence; explicitly provide a second
        # view in which only the currently inferred HUD delta is neutralized.
        if self._hud and same_shape:
            causal_grid = tuple(tuple(before.grid[y][x] if (x, y) in self._hud else value
                                      for x, value in enumerate(row)) for y, row in enumerate(post))
            causal_view = FrameView(causal_grid)
            object_diffs['causal_board'] = compute_frame_diff(
                before.grid, causal_grid, [o.node() for o in before.segmentation],
                [o.node() for o in causal_view.segmentation], max_group_match=64)
        else:
            object_diffs['causal_board'] = dict(object_diffs)
        evidence = transition_evidence(self._history[-64:], ARC_COLOR_CHARS)
        evidence['animation'] = summary if enabled('ENABLE_ANIMATION_DIFF') else None
        evidence['hud_classification'] = 'provisional monotonic border evidence; not a proven resource'
        evidence['rendered_response'] = bool(frames)
        evidence['causal_effect_status'] = 'observed' if frames and before.grid else 'unavailable'
        record = Transition(after.step, action, coordinates, target, before, frames[:-1], after,
                            raw_settled, settled, raw_transient, transient, self._hud,
                            reset, terminal, level_changed, full_reset,
                            json.dumps(object_diffs), json.dumps(evidence),
                            self.remaining_step_budget_estimate if not boundary else None)
        if not boundary and frames and before.grid:
            if target is not None:
                self._zeros[target] = self._zeros.get(target, 0)+1 if record.zero_causal_diff else 0
            self._reference.observe(self._state_hash(), self._action_sig(action, coordinates),
                                    gameplay_changed=not record.zero_causal_diff)
            self._reference.note_executed(self._state_hash(), self._action_sig(action, coordinates))
        self.last_action = action
        self.current = after
        if boundary:
            self._zeros.clear()
            self._scope = [snapshot]
            self._hud = frozenset()
            self.remaining_step_budget_estimate = None
        self._reference.note_level(level)
        self._records.append(record)
        payload = record.to_dict()
        payload.update(type='transition', live_action=True)
        payload['[DIFF]'] = [f'({d.x}, {d.y}): {d.old} -> {d.new}'
                             for d in (settled if auditable() else raw_settled)]
        payload['[RAW_SETTLED_DIFF]'] = [asdict(d) for d in raw_settled]
        payload['[CAUSAL_BOARD_DIFF]'] = [asdict(d) for d in settled]
        payload['[TRANSIENT_ANIMATION_DIFF]'] = [asdict(e) for e in transient]
        payload['[HUD_STEP_BUDGET_STRIP]'] = sorted(record.hud_mask)
        payload['dead_signatures'] = self.dead_signatures
        self._emit(payload)
        return record

    def get_board_objects(self) -> tuple[BoardObject, ...]:
        return self.current.segmentation

    def get_step_diffs(self, index: int = -1) -> Transition | None:
        return self._records[index] if self._records else None

    def get_hud_mask(self) -> frozenset[XY]:
        return self._hud if enabled('ENABLE_HUD_MASK') else frozenset()


def get_board_objects(source: PerceptionLog | FrameView) -> tuple[BoardObject, ...]:
    return source.get_board_objects() if isinstance(source, PerceptionLog) else source.segmentation


def get_step_diffs(log: PerceptionLog, index: int = -1) -> Transition | None:
    return log.get_step_diffs(index)


def get_hud_mask(log: PerceptionLog) -> frozenset[XY]:
    return log.get_hud_mask()


def is_dead_signature(log: PerceptionLog, action_type: str,
                      target_shape_hash: str, target_color: int) -> bool:
    return log.is_dead_signature(action_type, target_shape_hash, target_color)
