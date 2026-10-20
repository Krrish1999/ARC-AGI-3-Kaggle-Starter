"""Evidence-gated memory, sandbox retrodiction and checked action planning.

Generated mechanics implement step(state, action), returning a complete named
state: grid, level and state (public framework status). Actions use framework
enum names and optional x/y. No generated Python executes in the engine host.
An optional injected reasoning backend returns structured hypotheses, [ACTIONS]
or simulator source. Without one, exact observed transitions provide a partial
simulator; unknown edges remain unknown and exploration fallbacks are recorded.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import os
import random
import tempfile
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

try:
    from .arclog import FrameView, PerceptionLog, Transition, enabled, freeze_grid
    from .baseline.bounded_search import bounded_bfs
    from .baseline.replay_compare import _rc_differences
    from .runtime import TARGET_SECONDS, LevelBudget, PrefixLock, run_expired
    from .sandbox import run_sandbox
except ImportError:
    from agent.arclog import FrameView, PerceptionLog, Transition, enabled, freeze_grid
    from agent.baseline.bounded_search import bounded_bfs
    from agent.baseline.replay_compare import _rc_differences
    from agent.runtime import TARGET_SECONDS, LevelBudget, PrefixLock, run_expired
    from agent.sandbox import run_sandbox


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _state(frame: FrameView) -> dict[str, Any]:
    return {'grid': [list(row) for row in frame.grid], 'level': frame.level, 'state': frame.state}


def _action(name: str, coordinates: tuple[int, int] | None) -> dict[str, Any]:
    return {'action': name, **({'x': coordinates[0], 'y': coordinates[1]} if coordinates else {})}


def _bfs_source() -> str:
    # Remove the trusted helper's host-only typing annotation. The secured
    # worker does not need disk-backed typing or future imports to run BFS.
    tree = ast.parse(inspect.getsource(bounded_bfs))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    for argument in function.args.args:
        argument.annotation = None
    return ast.unparse(tree)


def _masked(state: dict[str, Any], mask: frozenset[tuple[int, int]]) -> dict[str, Any]:
    raw = state['grid']
    if (not isinstance(raw, list) or not raw or any(not isinstance(row, list) or not row for row in raw)
            or any(type(cell) is not int for row in raw for cell in row)):
        raise ValueError('step grid must contain nonempty rows of integer colors')
    grid = freeze_grid(state['grid'])
    if type(state['level']) is not int or not isinstance(state['state'], str):
        raise ValueError('step must return grid, integer level and public state name')
    return {'grid': [[None if (x, y) in mask else value for x, value in enumerate(row)]
                     for y, row in enumerate(grid)], 'level': state['level'], 'state': state['state']}


class Playbook:
    """Evidence-backed section memory, with append-only turns per context.

    The reference ablation drains oldest exchanges into a fresh context, since
    FIFO deletion inside an already submitted prefix would violate S1-S. Both
    paths retain the external complete evidence log and section memory.
    """

    def __init__(self, directory: Path, lock: PrefixLock, token_ceiling: int = 58 * 1024) -> None:
        self.directory = directory
        self.lock = lock
        self.token_ceiling = token_ceiling
        self.facts: dict[str, dict[str, Any]] = {}
        self.version = 0
        self.context = 0
        self.messages: list[dict[str, Any]] = []
        self._fresh('No replay evidence yet.')

    def _fresh(self, snapshot: str, retained: list[dict[str, Any]] | None = None) -> None:
        self.context += 1
        self.messages = [{'role': 'system', 'content':
                          'Use observed evidence only. Mechanics require complete replay; '
                          'emit concrete [ACTIONS] with non-HUD expect cells and expect_level_up.\n' + snapshot}]
        self.messages.extend(_copy(retained or []))
        self.request()

    def request(self) -> list[dict[str, Any]]:
        self.lock.validate(str(self.context), self.messages)
        return _copy(self.messages)

    def remember(self, key: str, **fact: Any) -> None:
        self.facts[key] = _copy(fact)

    def append(self, role: str, content: Any) -> None:
        self.messages.append({'role': role, 'content': json.dumps(content, sort_keys=True)})
        self.request()
        # Conservative UTF-8 byte bound: no tokenizer/model assets are assumed.
        if len(json.dumps(self.messages).encode()) >= self.token_ceiling:
            self.distill('token_ceiling')

    def distill(self, reason: str) -> None:
        if not enabled('ENABLE_PLAYBOOK_RESET'):
            retained = self.messages[1:]
            # Archived oldest-block drain, always through a new prefix identity.
            while retained and len(json.dumps(retained).encode()) >= self.token_ceiling // 2:
                retained = retained[2:]
            self._fresh('Persistent section memory:\n' + json.dumps(self.facts, sort_keys=True), retained)
            return
        self.version += 1
        lines = [f'# Playbook v{self.version}', f'Reset reason: {reason}',
                 'Replay consistency is scoped evidence, not a universal proof.']
        for status in ('verified', 'counterexample', 'unresolved'):
            lines.append('\n## ' + status)
            lines.extend(json.dumps({'id': key, **value}, sort_keys=True)
                         for key, value in sorted(self.facts.items()) if value['status'] == status)
        snapshot = '\n'.join(lines) + '\n'
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f'playbook.v{self.version:04d}.md').write_text(snapshot)
        temporary = self.directory / 'playbook.tmp'
        temporary.write_text(snapshot)
        temporary.replace(self.directory / 'playbook.md')
        self._fresh(snapshot)


class WorldModel:
    """Host-owned evidence gates. Only this object issues live plan/probe tickets."""

    def __init__(self, perception: PerceptionLog, planner: Callable[[str], dict[str, Any]],
                 directory: Path, lock: PrefixLock) -> None:
        self.log = perception
        self.planner = planner
        self.memory = Playbook(directory, lock)
        self.directory = directory
        self.events: list[dict[str, Any]] = []
        self.queue: deque[dict[str, Any]] = deque()
        self.pending: dict[str, Any] | None = None
        self.queue_scope: tuple[int, int, int] | None = None
        self.candidates: dict[str, dict[str, Any]] = {}
        self.wasted_moves = 0
        self.backend: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
        self._seen = 0
        self._tier_attempt: tuple[int, int] | None = None

    def event(self, kind: str, **payload: Any) -> None:
        entry = {'type': kind, 'step': self.log.current.step, 'live_action': False, **payload}
        self.events.append(_copy(entry))
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / 'world_model.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        self.memory.append('user', entry)

    def _worker(self, source: str, body: str) -> dict[str, Any]:
        # Candidate namespace is separate from the verification driver. The host
        # checks output shape, coverage and cells; a candidate cannot grant itself
        # permission by returning a boolean "verified" flag.
        program = ('import json\n'
                   f'namespace = {{}}\nexec(compile({source!r}, "<mechanic>", "exec"), namespace)\n'
                   'step = namespace["step"]\n' + body)
        result = self.planner(program)
        if not result.get('ok'):
            return {'ok': False, 'error': result}
        try:
            lines = result['stdout'].splitlines()
            if len(lines) != 1:
                raise ValueError('worker must return exactly one JSON result')
            return {'ok': True, 'value': json.loads(lines[0])}
        except (ValueError, KeyError, TypeError) as error:
            return {'ok': False, 'error': 'invalid_worker_output: ' + str(error)}

    def _relevant(self, scope: str) -> list[Transition]:
        if scope not in ('level', 'game'):
            raise ValueError('scope must be level or game')
        return [t for t in self.log.records if not t.reset and not t.full_reset
                and t.before.grid and json.loads(t.evidence_json).get('rendered_response')
                and (scope == 'game' or t.before.level == self.log.current.level)]

    def replay_transitions(self, source: str, *, scope: str = 'level', chunk_size: int = 16,
                           timeout_s: float = 30.0) -> dict[str, Any]:
        """Full-history upgrade of archived replay; no tail scan or case clamp.

        Empty/limited/crashed/malformed replay never grants a complete pass.
        Chunk outputs contain complete successor states; comparison is host-owned.
        Reset/unrendered evidence is counted as excluded, never silently matched.
        """
        if not 1 <= chunk_size <= 64 or not 0 < timeout_s <= 120:
            raise ValueError('invalid replay limits')
        records = self._relevant(scope)
        digest = hashlib.sha256(json.dumps([t.to_dict() for t in records], sort_keys=True).encode()).hexdigest()
        report: dict[str, Any] = {
            'status': 'empty', 'complete': False, 'scope': scope, 'level': self.log.current.level,
            'source_hash': hashlib.sha256(source.encode()).hexdigest(), 'history_hash': digest,
            'history_count': len(self.log.records), 'eligible': len(records),
            'excluded': len(self.log.records)-len(records), 'checked': 0, 'matched': 0,
            'errors': 0, 'chunks': [], 'counterexamples': [], 'live_actions': 0}
        began = time.monotonic()
        for offset in range(0, len(records), chunk_size):
            if time.monotonic()-began >= timeout_s:
                report['status'] = 'limited'
                break
            chunk = records[offset:offset+chunk_size]
            evidence = [{'step': t.step, 'before': _state(t.before),
                         'action': _action(t.action, t.coordinates)} for t in chunk]
            body = (f'log_history = json.loads({json.dumps(evidence)!r})\n'
                    'def verify_hypothesis(log_history):\n'
                    '    return [step(t["before"], t["action"]) for t in log_history]\n'
                    'print(json.dumps(verify_hypothesis(log_history)))\n')
            result = self._worker(source, body)
            predictions = result.get('value')
            if (not result['ok'] or not isinstance(predictions, list)
                    or len(predictions) != len(chunk)):
                report.update(status='error', errors=report['errors']+1, failure=result)
                break
            try:
                for transition, predicted in zip(chunk, predictions, strict=True):
                    expected = _masked(predicted, transition.hud_mask)
                    actual = _masked(_state(transition.after), transition.hud_mask)
                    differences, count = _rc_differences(expected, actual)
                    report['checked'] += 1
                    if count:
                        report['counterexamples'].append({'step': transition.step,
                            'expected_cells': differences, 'actual_state': actual,
                            'difference_count': count})
                    else:
                        report['matched'] += 1
            except (KeyError, TypeError, ValueError, IndexError) as error:
                report.update(status='error', errors=report['errors']+1, failure=str(error))
                break
            report['chunks'].append({'offset': offset, 'checked': len(chunk)})
        if report['status'] not in ('error', 'limited'):
            report['complete'] = bool(records) and report['checked'] == len(records)
            report['status'] = ('counterexample' if report['counterexamples'] else
                                'pass' if report['complete'] else 'empty')
        if time.monotonic()-began >= timeout_s:
            report.update(status='limited', complete=False)
        self.event('replay', report=report)
        return report

    def _cells(self, value: Any, *, required: bool = True) -> list[dict[str, int]]:
        if isinstance(value, dict):
            value = [{'x': xy[0], 'y': xy[1], 'value': color} for xy, color in value.items()]
        if not isinstance(value, list) or (required and not value):
            raise ValueError('non-empty expected settled non-HUD cells required')
        grid, mask = self.log.current.grid, self.log.get_hud_mask()
        seen: set[tuple[int, int]] = set()
        for cell in value:
            if not isinstance(cell, dict) or set(cell) != {'x', 'y', 'value'}:
                raise ValueError('expect needs x, y, value cells')
            x, y, color = cell['x'], cell['y'], cell['value']
            if (any(type(n) is not int for n in (x, y, color)) or not 0 <= color < 16
                    or not 0 <= y < len(grid) or not 0 <= x < len(grid[y])
                    or (x, y) in mask or (x, y) in seen):
                raise ValueError('invalid, duplicate or HUD prediction cell')
            seen.add((x, y))
        return _copy(value)

    def _plan_item(self, item: dict[str, Any], legal: list[GameAction]) -> dict[str, Any]:
        name = item.get('action')
        if not isinstance(name, str) or name not in GameAction.__members__:
            raise ValueError('concrete framework action enum required')
        action = GameAction[name]
        if action is GameAction.RESET or action not in legal:
            raise ValueError('unavailable action or reset in plan')
        result: dict[str, Any] = {'action': name}
        if action.is_complex():
            x, y = item.get('x'), item.get('y')
            grid = self.log.current.grid
            if (type(x) is not int or type(y) is not int or not 0 <= y < len(grid)
                    or not 0 <= x < len(grid[y])):
                raise ValueError('concrete in-board x/y required')
            result.update(x=x, y=y)
        elif 'x' in item or 'y' in item:
            raise ValueError('simple action cannot carry coordinates')
        result['expect'] = self._cells(item.get('expect', []), required=enabled('ENABLE_EXPECT_QUEUE'))
        up = item.get('expect_level_up', False if not enabled('ENABLE_EXPECT_QUEUE') else None)
        if type(up) is not bool:
            raise ValueError('explicit expect_level_up boolean required')
        result['expect_level_up'] = up
        return result

    def execute_plan(self, actions: list[dict[str, Any]], legal: list[GameAction]) -> None:
        """Install an immutable concrete queue; actual dispatch stays in the host.

        Like checked_execute, terminal flags interrupt before old-level decoding.
        No callback can dispatch from within replay/search's isolated process.
        """
        if self.pending is not None:
            raise ValueError('cannot replace plan awaiting settled observation')
        if not isinstance(actions, list) or not 1 <= len(actions) <= 128:
            raise ValueError('plan requires 1..128 concrete actions')
        plan = [self._plan_item(item, legal) for item in actions]
        self.queue = deque(plan)
        current = self.log.current
        self.queue_scope = (current.level, current.episode, current.step)
        self.event('plan_queued', count=len(plan))

    def flush(self, reason: str) -> None:
        count = len(self.queue) + int(self.pending is not None)
        self.queue.clear()
        self.pending = None
        self.queue_scope = None
        if count:
            self.event('queue_flushed', reason=reason, remaining=count)

    def next_action(self, legal: list[GameAction]) -> dict[str, Any] | None:
        current = self.log.current
        if self.pending is not None:
            raise RuntimeError('settled observation required before another dispatch')
        if not self.queue:
            return None
        if self.queue_scope != (current.level, current.episode, current.step):
            self.flush('stale_state')
            return None
        try:
            item = self._plan_item(self.queue[0], legal)
        except ValueError:
            self.flush('plan_no_longer_admissible')
            return None
        stored = self.queue.popleft()
        item['probe'] = bool(stored.get('probe'))
        if 'mechanic_key' in stored:
            item['mechanic_key'] = stored['mechanic_key']
        self.pending = item
        return _copy(item)

    def observe(self, transition: Transition) -> None:
        if transition.step <= self._seen:
            return
        self._seen = transition.step
        boundary = transition.reset or transition.full_reset or transition.level_changed or transition.terminal
        if self.pending is not None:
            item = self.pending
            action_matches = _action(transition.action, transition.coordinates) == {
                k: v for k, v in item.items() if k in ('action', 'x', 'y')}
            cells = item['expect']
            mask = transition.hud_mask
            grid = transition.after.grid
            match = action_matches and bool(transition.level_changed) == item['expect_level_up']
            match = match and all((c['x'], c['y']) not in mask
                                 and c['y'] < len(grid) and c['x'] < len(grid[c['y']])
                                 and grid[c['y']][c['x']] == c['value'] for c in cells)
            self.event('prediction', matched=match, predicted_cells=len(cells),
                       probing=bool(item.get('probe')), boundary=boundary)
            if not match and item.get('mechanic_key') in self.memory.facts:
                key = item['mechanic_key']
                fact = _copy(self.memory.facts[key])
                fact['status'] = 'counterexample'
                fact['counterexamples'].append({'step': transition.step,
                    'expected_cells': cells, 'actual_state': _state(transition.after)})
                self.memory.remember(key, **fact)
                self.candidates[key]['status'] = 'counterexample'
                self.event('live_counterexample', mechanic_key=key, transition=transition.step)
            if boundary or (enabled('ENABLE_EXPECT_QUEUE') and not match):
                self.flush('boundary' if boundary else 'cell_mismatch')
            else:
                self.pending = None
                self.queue_scope = (transition.after.level, transition.after.episode, transition.after.step)
        elif boundary:
            self.flush('boundary')
        if boundary:
            self.wasted_moves = 0
            self._tier_attempt = None
        elif transition.zero_causal_diff:
            self.wasted_moves += 1
        # Record observations separately; a replay pass remains explicitly tied
        # to its evidence revision and must be refreshed before a new probe.
        self.memory.append('user', {'transition': transition.step, 'level': transition.before.level,
                                  'diff': [vars(cell) for cell in transition.settled],
                                  'transient_cells': len(transition.transient), 'boundary': boundary})
        if boundary:
            self.memory.distill('level_boundary' if transition.level_changed else 'reset_or_terminal')

    def _predict(self, source: str, action: dict[str, Any]) -> dict[str, Any]:
        result = self._worker(source, f'print(json.dumps(step(json.loads({json.dumps(_state(self.log.current))!r}), '
                              f'json.loads({json.dumps(action)!r}))))\n')
        if not result['ok'] or not isinstance(result.get('value'), dict):
            raise ValueError('prediction failed: ' + str(result))
        _masked(result['value'], self.log.get_hud_mask())
        return result['value']

    def propose_hypothesis(self, source: str, probe: dict[str, Any], alternative_expect: Any,
                           legal: list[GameAction], *, scope: str = 'level') -> dict[str, Any]:
        """Replay first, then authorize exactly one differing, evidence-seeking probe."""
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        key = f'{source_hash}:{scope}:{self.log.current.level if scope == "level" else "all"}'
        report = (self.replay_transitions(source, scope=scope) if enabled('ENABLE_RETRODICTION')
                  else {'status': 'ablation_bypass', 'complete': False})
        status = ('verified' if report['status'] == 'pass' else 'counterexample'
                  if report['status'] == 'counterexample' else 'unresolved')
        fact = {'status': status, 'scope': scope, 'level': self.log.current.level,
                'source': source, 'report': report}
        self.candidates[key] = fact
        self.directory.mkdir(parents=True, exist_ok=True)
        source_path = self.directory / f'mechanic.{source_hash}.py'
        source_path.write_text(source)
        self.memory.remember(key, status=status, scope=scope, level=self.log.current.level,
                             source_file=source_path.name, evidence={
                                 k: report[k] for k in ('status', 'complete', 'history_hash', 'checked', 'eligible')
                                 if k in report},
                             counterexamples=[{'step': c['step'], 'expected_cells': c['expected_cells']}
                                              for c in report.get('counterexamples', [])[:1]])
        if enabled('ENABLE_RETRODICTION') and not (report['status'] == 'pass' and report['complete']):
            self.event('hypothesis_rejected', source_hash=source_hash, reason=report['status'], live_actions=0)
            return report
        item = self._plan_item(probe, legal)
        alternative = self._cells(alternative_expect)
        if not item['expect'] or { (c['x'], c['y']) for c in alternative } != {
                (c['x'], c['y']) for c in item['expect']}:
            raise ValueError('probe alternatives must predict the same nonempty cells')
        predicted = self._predict(source, {k: v for k, v in item.items() if k in ('action', 'x', 'y')})
        if (not all(predicted['grid'][c['y']][c['x']] == c['value'] for c in item['expect'])
                or (predicted['level'] > self.log.current.level) != item['expect_level_up']):
            raise ValueError('probe expectation disagrees with mechanic')
        if sorted(alternative, key=lambda c: (c['x'], c['y'])) == sorted(item['expect'], key=lambda c: (c['x'], c['y'])):
            raise ValueError('probe must disambiguate concrete predictions')
        self.execute_plan([item], legal)
        self.queue[0]['probe'] = True
        self.queue[0]['mechanic_key'] = key
        self.event('probe_authorized', source_hash=source_hash, count=1, replay_status=report['status'])
        return report

    def _lookup_source(self) -> str:
        # Conservative synthesis from observation only: no invented unknown edges.
        edges = [{'before': _state(t.before), 'action': _action(t.action, t.coordinates),
                  'after': _state(t.after)} for t in self._relevant('level')]
        return (f'edges = {edges!r}\n'
                'def step(state, action):\n'
                '    matches = [e["after"] for e in edges if e["before"] == state and e["action"] == action]\n'
                '    if matches and all(m == matches[0] for m in matches):\n'
                '        return matches[0]\n'
                '    return None\n')

    def search_plan(self, source: str, legal: list[GameAction]) -> dict[str, Any]:
        """Compile and replay current-level history before bounded sandbox BFS."""
        compile_result = self._worker(source, 'assert callable(step)\nprint(json.dumps(True))\n')
        self.event('simulator_compile', success=compile_result['ok'], result=compile_result)
        if not compile_result['ok']:
            return {'status': 'compile_failed'}
        # Simulator validation is mandatory even with the general replay ablated.
        replay = self.replay_transitions(source, scope='level')
        self.event('simulator_replay', success=replay['status'] == 'pass', report=replay)
        if replay['status'] != 'pass' or not replay['complete']:
            return {'status': 'replay_failed', 'replay': replay}
        actions = [_action(a.name, None) for a in legal if not a.is_complex()]
        actions.extend(_action(t.action, t.coordinates) for t in self._relevant('level')
                       if GameAction[t.action] in legal and t.coordinates is not None)
        actions = list({json.dumps(a, sort_keys=True): a for a in actions}.values())
        # Reuse the extracted BFS verbatim, providing already-bootstrapped aliases.
        body = ('from collections import deque as _bfs_deque\nfrom time import monotonic as _bfs_clock\n'
                '_bfs_copy = lambda value: json.loads(json.dumps(value))\n'
                '_bfs_finite = __import__("math").isfinite\n' + _bfs_source() + '\n'
                f'actions = json.loads({json.dumps(actions)!r})\n'
                f'start = json.loads({json.dumps(_state(self.log.current))!r})\n'
                'def successors(state):\n'
                '    for action in actions:\n'
                '        candidate = step(_bfs_copy(state), _bfs_copy(action))\n'
                '        if candidate is not None:\n'
                '            yield action, candidate\n'
                'report = bounded_bfs(start, successors, lambda s: s["level"] > start["level"] '
                'or s["state"] == "WIN", key=lambda s: json.dumps(s, sort_keys=True), '
                'max_nodes=2000, max_depth=32, max_edges=10000, timeout_s=2.0)\n'
                'plan = []\nstate = start\n'
                'for action in report.get("actions") or []:\n'
                '    after = step(_bfs_copy(state), _bfs_copy(action))\n'
                '    cells = [{"x": x, "y": y, "value": v} for y, row in enumerate(after["grid"]) '
                'for x, v in enumerate(row)]\n'
                '    plan.append(dict(action, expect=cells, expect_level_up=after["level"] > state["level"]))\n'
                '    state = after\n'
                'report["plan"] = plan\nprint(json.dumps(report))\n')
        result = self._worker(source, body)
        if not result['ok']:
            self.event('search_failed', result=result)
            return {'status': 'search_error'}
        report = result['value']
        self.event('search', report=report)
        if report['status'] == 'found' and report['plan']:
            mask = self.log.get_hud_mask()
            for item in report['plan']:
                item['expect'] = [c for c in item['expect'] if (c['x'], c['y']) not in mask]
            self.execute_plan(report['plan'], legal)
        return report

    def reason(self, legal: list[GameAction]) -> None:
        """Backend integration boundary: validate immutable prefix before inference."""
        tier_due = (enabled('ENABLE_TIER1_SIMULATOR') and self.log.current.level >= 3
                    and self.wasted_moves >= 40)
        response: dict[str, Any] = {}
        if self.backend is not None:
            self.memory.append('user', {'request': 'simulator' if tier_due else 'hypotheses_or_actions',
                                      'state': _state(self.log.current),
                                      'available_actions': [a.name for a in legal],
                                      'objects': [o.node() for o in self.log.get_board_objects()],
                                      'hud_mask': sorted(self.log.get_hud_mask()),
                                      'transition_count': len(self.log.records)})
            response = self.backend(self.memory.request())
            self.memory.append('assistant', response)
        if tier_due:
            revision = (self.log.current.level, len(self.log.records))
            if self._tier_attempt != revision:
                self._tier_attempt = revision
                source = response.get('simulator_source') or self._lookup_source()
                self.event('tier1_required', wasted_moves=self.wasted_moves,
                           synthesis='backend' if response.get('simulator_source') else 'observed_lookup')
                report = self.search_plan(source, legal)
                if report['status'] != 'found':
                    self.event('fallback', reason='tier1_' + report['status'])
            return
        for hypothesis in response.get('hypotheses', []):
            self.propose_hypothesis(hypothesis['source'], hypothesis['probe'],
                                    hypothesis['alternative_expect'], legal,
                                    scope=hypothesis.get('scope', 'level'))
            if self.queue:
                return
        if response.get('hypotheses'):
            # Rejected mechanics cannot fall through to an unchecked live probe
            # disguised as a second response field.
            return
        if response.get('actions'):
            self.execute_plan(response['actions'], legal)


class MyAgent(Agent):
    """Framework adapter; perception owns actions, world model owns reasoning."""

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
        self._recovery_mode = False
        self._observation: FrameData | None = None
        self._dispatch_approval: tuple[str, tuple[int, int] | None, int] | None = None
        self.prefix_lock = PrefixLock()
        self.level_budget: LevelBudget | None = None
        assert self.perception.log_path is not None
        self.world = WorldModel(self.perception, self.execute_planner,
                                self.perception.log_path.parent, self.prefix_lock)

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
        try:
            if self._recovery_mode:
                return self._safe_fallback(latest_frame)
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
            for transition in self.perception.records:
                if transition.step > self.world._seen:
                    self.world.observe(transition)
            planned = None
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self.world.flush('reset_proposal')
                proposal = GameAction.RESET
            else:
                legal = self._legal_actions(latest_frame)
                if not legal:
                    raise RuntimeError("no advertised non-reset action; replan required")
                planned = self.world.next_action(legal)
                if planned is None:
                    try:
                        self.world.reason(legal)
                    except Exception as error:
                        self.world.flush('reasoning_failure')
                        self.world.event('reasoning_failure', error=type(error).__name__, detail=str(error))
                        raise
                    planned = self.world.next_action(legal)
                if planned is not None:
                    proposal = GameAction[planned['action']]
                    if proposal.is_complex():
                        proposal.set_data({'x': planned['x'], 'y': planned['y']})
                else:
                    if self.world.backend is not None:
                        raise RuntimeError('backend produced no admissible evidence-gated plan')
                    self.world.event('fallback', reason='no_backend_or_validated_route',
                                     wasted_moves=self.world.wasted_moves)
                    proposal = self._sample(self._rng.choice(legal))
            action: Any = self._resolve_action(proposal, latest_frame)
            if planned is not None and _action(action.name, self._coordinates(action)) != {
                    k: v for k, v in planned.items() if k in ('action', 'x', 'y')}:
                self.world.flush('perception_blocked_plan')
                raise RuntimeError('planned action blocked; replan before dispatch')
            self._dispatch_approval = (action.name, self._coordinates(action), self.perception.current.step)
            action.reasoning = {"why": "expect-checked plan" if planned else "visible exploration fallback",
                                "evidence_step": self.perception.current.step,
                                "context_id": self.world.memory.context}
            if self.level_budget is not None and self.level_budget.generation_tokens(1) == 0:
                action.reasoning["budget"] = "level reasoning budget exhausted; best-effort action"
            return action
        except Exception as e:
            if not enabled('ENABLE_ACTION_FALLBACK'):
                raise
            # Do not call world.flush/event or perception here: these may be
            # exactly the subsystem that failed. No stale plan may dispatch.
            self._recovery_mode = True
            self._dispatch_approval = None
            self.world.queue.clear()
            self.world.pending = None
            self.world.queue_scope = None
            self._log_action_failure(e)
            return self._safe_fallback(latest_frame)

    @staticmethod
    def _log_action_failure(error: Exception) -> None:
        detail = ''.join(traceback.format_exception(error))
        try:
            directory = Path(os.getenv('ARC3_LOG_DIR', 'scratch'))
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'choose_action_errors.log').open('a', encoding='utf-8') as stream:
                stream.write(detail + '\n')
        except OSError:
            # A full or read-only disk must not turn recovery into another crash.
            logging.getLogger(__name__).error('Unable to save action traceback: %s', detail)

    @staticmethod
    def _safe_fallback(observation: FrameData) -> GameAction:
        # ACTION1 is the framework's UP action. Use public action metadata only;
        # emergency mode never re-enters world-model or perception helpers.
        if observation.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        legal = [a for a in GameAction if a is not GameAction.RESET
                 and a.value in observation.available_actions]
        action: Any = next((a for a in legal if not a.is_complex()),
                      legal[0] if legal else GameAction.ACTION1)
        if action.is_complex():
            action.set_data({'x': 0, 'y': 0})
        action.reasoning = {'why': 'choose_action exception recovery; unverified fallback'}
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
        if self._recovery_mode:
            # Remain in best-effort mode for this game. Failed perception must
            # not crash dispatch or invent transitions used as replay evidence.
            response = super().do_action_request(action)
            self._observation = response
            return response
        # Final interception also protects direct framework/tool calls that did
        # not originate in choose_action. Snapshot mutable enum data before step.
        if self._observation is None:
            self._ensure_perception(self._convert_raw_frame_data(self.arc_env.observation_space))
        assert self._observation is not None
        if action is GameAction.RESET:
            self.world.flush('reset_proposal')
        elif self.world.queue and self.world.pending is None:
            self.world.flush('unapproved_dispatch')
            raise RuntimeError('queued actions require next_action authorization')
        approval = (action.name, self._coordinates(action), self.perception.current.step)
        if self.world.pending is not None:
            pending = self.world.pending
            expected = {k: v for k, v in pending.items() if k in ('action', 'x', 'y')}
            frame = self.perception.current
            if (_action(action.name, self._coordinates(action)) != expected
                    or self.world.queue_scope != (frame.level, frame.episode, frame.step)):
                self.world.flush('dispatch_mismatch')
                raise RuntimeError('dispatch does not match evidence-gated plan')
        selected = self._resolve_action(action, self._observation,
                                        reference=approval != self._dispatch_approval)
        if self.world.pending is not None and (selected.name, self._coordinates(selected)) != approval[:2]:
            self.world.flush('perception_blocked_plan')
            raise RuntimeError('planned action blocked; replan before dispatch')
        self._dispatch_approval = None
        coordinates = self._coordinates(selected)
        try:
            response = super().do_action_request(selected)
        except Exception as error:
            self.world.flush('dispatch_error')
            self.perception.note_dispatch_error(selected.name, coordinates, type(error).__name__)
            logging.getLogger(__name__).exception("live dispatch failed; no observed transition recorded")
            raise
        transition = self.perception.record(selected.name, coordinates, response.frame,
                                            level=response.levels_completed+1, state=response.state.name,
                                            full_reset=response.full_reset)
        self.world.observe(transition)
        self._observation = response
        return response
