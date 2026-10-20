"""Generic unit-cost BFS, also embedded verbatim into the isolated Python tool.

No environment access, puzzle identifiers, transition rules or action execution.
Time limits are cooperative; the sandbox's outer timeout remains authoritative.
"""

from collections import deque as _bfs_deque
from collections.abc import Callable, Iterable
from copy import deepcopy as _bfs_copy
from math import isfinite as _bfs_finite
from time import monotonic as _bfs_clock
from typing import Any


def bounded_bfs(start, successors: Callable[[Any], Iterable[tuple[Any, Any]]], is_goal, *, key=None, max_nodes=50000,
                max_depth=128, timeout_s=2.0, max_edges=250000):
    """Search a model supplied by the caller, never the real environment.

    successors(state) yields (action, next_state). is_goal(state) is a predicate.
    States must be deepcopy-able and hashable, or key(state) must be hashable.
    Keys must distinguish every state variable relevant to transitions and goals.
    Callbacks must be pure. Copies isolate ordinary mutable state/action aliases.

    Result status: found, exhausted, or limit (reason: nodes/edges/depth/time).
    Only found has a list of actions; [] means the start already satisfies goal.
    Exhausted refers only to the fully explored supplied model, NOT the real game.
    Hitting max_depth conservatively reports limit even if its boundary is a leaf.
    Unit-cost shortest paths only; there is no heuristic or weighted-edge mode.
    Valid oversized requests are capped at unchanged hard maxima: 100000 nodes,
    1024 depth, 1000000 edges and 5 seconds. Defaults remain unchanged.
    Results include effective_limits and clamped_limits (names, not input values).
    Zero depth remains valid; other limits must be positive. No type coercion.
    Callback exceptions propagate; no automatic retry or partial plan is returned.
    """
    # Validate types/lower bounds before capping: do not coerce malformed inputs.
    effective_limits = {}
    clamped_limits = []
    for name, value, lower, upper in (
        ('max_nodes', max_nodes, 1, 100000),
        ('max_depth', max_depth, 0, 1024),
        ('max_edges', max_edges, 1, 1000000),
    ):
        if type(value) is not int or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
        effective_limits[name] = min(value, upper)
        if value > upper:
            clamped_limits.append(name)
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or (isinstance(timeout_s, float) and not _bfs_finite(timeout_s))
            or timeout_s <= 0):
        raise ValueError('timeout_s must be finite and greater than 0')
    # Python integers are finite; cap before converting even very large integers.
    effective_limits['timeout_s'] = float(min(timeout_s, 5.0))
    if timeout_s > 5.0:
        clamped_limits.append('timeout_s')
    max_nodes = effective_limits['max_nodes']
    max_depth = effective_limits['max_depth']
    max_edges = effective_limits['max_edges']
    timeout_s = effective_limits['timeout_s']
    if not callable(successors) or not callable(is_goal) or (key is not None and not callable(key)):
        raise TypeError('successors and is_goal must be callable; key must be callable or None')

    began = _bfs_clock()
    deadline = began + timeout_s
    states, parents, incoming, depths = [], [], [], []
    seen = {}
    expanded = 0
    edges = 0
    depth_cut = False

    def timed_out():
        return _bfs_clock() >= deadline

    def result(status, reason=None, node=None):
        actions = None
        depth = None
        if status == 'found':
            assert node is not None
            actions = []
            depth = depths[node]
            while parents[node] != -1:
                actions.append(_bfs_copy(incoming[node]))
                node = parents[node]
            actions.reverse()
            # Reconstruction/copying is charged to the same cooperative budget.
            if timed_out():
                status, reason, actions, depth = 'limit', 'time', None, None
        return {'status': status, 'reason': reason, 'actions': actions,
                'depth': depth, 'expanded': expanded, 'discovered': len(states),
                'edges': edges, 'elapsed_s': _bfs_clock() - began,
                'effective_limits': dict(effective_limits),
                'clamped_limits': list(clamped_limits)}

    def state_key(state):
        value = key(_bfs_copy(state)) if key is not None else state
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError('Unhashable search state/key; provide key(state) with a hashable, complete state identity') from exc
        return _bfs_copy(value)

    root = _bfs_copy(start)
    root_key = state_key(root)
    states.append(root); parents.append(-1); incoming.append(None); depths.append(0)
    seen[root_key] = 0
    if timed_out():
        return result('limit', 'time')
    solved = is_goal(_bfs_copy(root))
    if timed_out():
        return result('limit', 'time')
    if solved:
        return result('found', node=0)
    frontier = _bfs_deque([0])

    while frontier:
        if timed_out():
            return result('limit', 'time')
        node = frontier.popleft()
        if depths[node] >= max_depth:
            depth_cut = True
            continue
        expanded += 1
        iterator = iter(successors(_bfs_copy(states[node])))
        if timed_out():
            return result('limit', 'time')
        while True:
            if timed_out():
                return result('limit', 'time')
            if edges >= max_edges:
                return result('limit', 'edges')
            try:
                edge = next(iterator)
            except StopIteration:
                break
            if timed_out():
                return result('limit', 'time')
            edges += 1
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ValueError('successors(state) must yield pairs (action, next_state)')
            action, candidate = edge
            candidate = _bfs_copy(candidate)
            identity = state_key(candidate)
            if timed_out():
                return result('limit', 'time')
            if identity in seen:
                continue
            if len(states) >= max_nodes:
                return result('limit', 'nodes')
            child = len(states)
            # Store parent indices separately: no model-written predecessor tuples.
            states.append(candidate); parents.append(node)
            incoming.append(_bfs_copy(action)); depths.append(depths[node] + 1)
            seen[identity] = child
            solved = is_goal(_bfs_copy(candidate))
            if timed_out():
                return result('limit', 'time')
            if solved:
                return result('found', node=child)
            frontier.append(child)
    return result('limit', 'depth') if depth_cut else result('exhausted')
