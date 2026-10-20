"""Sprint 1 unified retrodiction/world-model harness: exact two-module archive.

Captured 2026-10-20 (Asia/Calcutta) after the S1-S, S1-P and S1-W merges.
This is a functional integration checkpoint, not a statistically promoted champion.
SOURCE_FILES preserves both deployed modules byte for byte. Restore each entry
under its named path in a separate checkout of SOURCE_COMMIT; keep the support
modules listed in SUPPORT_SHA256 from that commit. Do not concatenate or execute
the embedded modules as a single strategy: relative imports require their package.
No inference server, model weights or wheelhouse is included in this archive.
The running agent still needs the Sprint 2 offline backend connection.
"""

SOURCE_COMMIT = '6f4dc442f587ac5ba26fea2fdf1de889291dfc6f'

SOURCE_SHA256 = {'agent/my_agent.py': '1e34d0cbbedbb4b72c26dc7b0902950e17be0768eeb90f054c89c8f1dcaf6777',
 'agent/arclog.py': '3043cf1829bf61621dc717226cc71d2a8cfc2a61039ae25c3fdad79e28a94589'}

SUPPORT_SHA256 = {'agent/baseline/__init__.py': 'af3b0c639320a3a3559a3b41dfd7b3060205b8ad94b1e2b6d46071784fb3c8dc',
 'agent/baseline/animation.py': 'cb47d19f35e2ceeee160541963c12cc0d82ef2d608215db91128184c1e3fce86',
 'agent/baseline/bounded_search.py': 'dfa22cd0a686cf2315fdb942b387c572eb756fb9ce73db8b5439129710709ef2',
 'agent/baseline/checked_execution.py': '829aebfef19e039d7994ecd84bff6fd918244138fe0b1e941a076b5b9c297621',
 'agent/baseline/frame_diff.py': '843be0ce9225ae19de91dd34409bd7b7cf69ebad1ff73e69759c506c161bdeee',
 'agent/baseline/grid_coverage.py': '1d0b524149a146bc029d03029076aa865cafa2bde0296db61d678d481fbb957b',
 'agent/baseline/grid_display.py': 'c536ffaa0a1250bdea7a2f9a29b6f56dfe3b5f46abb9495b76dfbad51a730b0a',
 'agent/baseline/noop_repeat_guard.py': 'efad18a4c803b8a778f9561ebb13335d3d47372ee5c80fd6297a5f78258c5957',
 'agent/baseline/priority_scheduler.py': '46d77544a6285bdb5dad0e0c7fcd6e546f06961640fa13341f38b2cd5043fd19',
 'agent/baseline/replay_compare.py': '3cd7873bf52eac5edab570ac92f29ce309dbe9f530adec8a857717ab6abe90c7',
 'agent/baseline/retained_functions.py': 'aaf7750b02c4b56a9bb3253d69e52992a18cf6bc65a7ed88e0d6aedf81852c9f',
 'agent/baseline/state_contracts.py': '437a7ccef673e68c16c4b4ccc5c9fdd863c294eb38cc6c61e0728f7832044457',
 'agent/baseline/transition_evidence.py': '0c82056d9a21ca7ed573213e786e0a93828834925611df705d465eb22c376947',
 'agent/runtime.py': '2b59095e38911d5c0de52c9104990a601f89b4730e6ba05aa7a3b358c29c342e',
 'agent/sandbox.py': '59a26176fbd45f3022110a7b7edf0f04b916a074141d1cd585f2882e635b4dfe'}

SOURCE_FILES = {
    'agent/my_agent.py': r'''"""Evidence-gated memory, sandbox retrodiction and checked action planning.

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
''',
    'agent/arclog.py': r'''"""Immutable visual evidence and pre-dispatch perception guards.

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

    @property
    def records(self) -> tuple[Transition, ...]:
        return tuple(self._records)

    @property
    def blocked_proposals(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(entry) for entry in self._blocked)

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(payload, separators=(',', ':'))+'\n')

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
        if len({f['action'] for f in history[1:]}) < 2:
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
               level: int, state: str = 'NOT_FINISHED', full_reset: bool = False) -> Transition:
        """Record a confirmed response. Refusals/failures never create transitions."""
        before = self.current
        frames = tuple(freeze_grid(g) for g in normalize_frames(grids))
        # Empty terminal responses carry no rendered animation; preserve the last
        # board and terminal flags without inventing an observed board effect.
        post = frames[-1] if frames else before.grid
        reset = action == 'RESET'
        terminal = state in ('GAME_OVER', 'WIN')
        level_changed = level != before.level
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
        raw_transient = tuple(e for e in events if (e.x, e.y) in repeated)
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
        payload['[DIFF]'] = [f'({d.x}, {d.y}): {d.old} -> {d.new}' for d in raw_settled]
        payload['[CAUSAL_BOARD_DIFF]'] = [asdict(d) for d in settled]
        payload['[TRANSIENT_ANIMATION_DIFF]'] = [asdict(e) for e in transient]
        payload['[HUD_STEP_BUDGET_STRIP]'] = sorted(record.hud_mask)
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
''',
}
