"""Submission survival fixtures using synthetic observations and child runners."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from arcengine import FrameData, GameAction, GameState

from agent.my_agent import MyAgent
from scripts.build_notebook import build


def setup_source(tmp_path):
    cells = [c['source'] for c in build()['cells'] if c['cell_type'] == 'code']
    return cells[0].replace('/kaggle/working', str(tmp_path)).replace(
        '/tmp/agent_bundle', str(tmp_path / 'bundle'))


@pytest.mark.parametrize('rerun', ['', 'True'])
def test_first_cell_parquet_survives_sigkill(tmp_path, rerun):
    source = setup_source(tmp_path)
    result = subprocess.run(
        [sys.executable, '-c', source + '\nimport signal\nos.kill(os.getpid(), signal.SIGKILL)'],
        env=dict(os.environ, KAGGLE_IS_COMPETITION_RERUN=rerun), timeout=10,
    )
    assert result.returncode == -signal.SIGKILL
    submission = pd.read_parquet(tmp_path / 'submission.parquet')
    assert list(submission.columns) == ['row_id', 'game_id', 'end_of_game', 'score']
    assert submission.iloc[0].to_dict() == {
        'row_id': '1_0', 'game_id': '1', 'end_of_game': True, 'score': 0}


def test_setup_preserves_existing_gateway_submission(tmp_path, monkeypatch):
    monkeypatch.setenv('ARC3_RUN_STARTED', '0')
    pd.DataFrame({'real_result': [42]}).to_parquet(tmp_path / 'submission.parquet')
    exec(setup_source(tmp_path), {})
    assert pd.read_parquet(tmp_path / 'submission.parquet').real_result.tolist() == [42]
    assert float(os.environ['ARC3_RUN_STARTED']) > 0


def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv('ARC3_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('ENABLE_ACTION_FALLBACK', '1')
    return MyAgent(card_id='fixture', game_id='synthetic', agent_name='fixture',
                   ROOT_URL='', record=False, arc_env=SimpleNamespace())


def observation(*, actions=(1, 2), level=1, state=GameState.NOT_FINISHED):
    return FrameData(frame=[[[0, 0], [0, 0]]], state=state,
                     levels_completed=level-1, win_levels=6, available_actions=list(actions))


@pytest.mark.parametrize('failure', ['budget', 'perception', 'world_model', 'tier1', 'resolver'])
def test_action_failure_logs_traceback_and_dispatches_without_failed_helpers(tmp_path, monkeypatch, failure):
    agent = make_agent(tmp_path, monkeypatch)
    frame = observation(level=3)
    if failure == 'budget':
        target, name = 'agent.my_agent.LevelBudget', None
    elif failure == 'perception':
        target, name = agent, '_ensure_perception'
    elif failure == 'world_model':
        target, name = agent.world, 'reason'
    elif failure == 'tier1':
        agent.world.wasted_moves = 40
        target, name = agent.world, 'search_plan'
    else:
        target, name = agent, '_resolve_action'
    crash = ValueError('unexpected grid size / empty diff')
    context = patch(target, side_effect=crash) if name is None else patch.object(target, name, side_effect=crash)
    with context as broken:
        selected = agent.choose_action([], frame)
        assert selected is GameAction.ACTION1
        assert agent._recovery_mode
        assert agent.world.pending is None and not agent.world.queue
        assert 'Traceback' in (tmp_path / 'choose_action_errors.log').read_text()
        assert str(crash) in (tmp_path / 'choose_action_errors.log').read_text()
        count = broken.call_count
        with patch('agents.agent.Agent.do_action_request', return_value=frame) as dispatch:
            assert agent.do_action_request(selected) is frame
            assert agent.choose_action([], frame) is GameAction.ACTION1
            dispatch.assert_called_once_with(selected)
        assert broken.call_count == count


@pytest.mark.parametrize('actions, state, expected', [
    ((2,), GameState.NOT_FINISHED, GameAction.ACTION2),
    ((6,), GameState.NOT_FINISHED, GameAction.ACTION6),
    ((), GameState.NOT_FINISHED, GameAction.ACTION1),
    ((1,), GameState.NOT_PLAYED, GameAction.RESET),
    ((1,), GameState.GAME_OVER, GameAction.RESET),
])
def test_fallback_uses_public_actions_only(actions, state, expected):
    action = MyAgent._safe_fallback(observation(actions=actions, state=state))
    assert action is expected
    if action.is_complex():
        assert action.action_data.model_dump()['x'] == 0
        assert action.action_data.model_dump()['y'] == 0


def test_traceback_write_failure_does_not_prevent_fallback(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv('ARC3_LOG_DIR', '/dev/null/not-a-directory')
    with patch.object(agent.world, 'reason', side_effect=ValueError('fixture error')):
        assert agent.choose_action([], observation()) is GameAction.ACTION1


def test_action_fallback_ablation_restores_exception(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    monkeypatch.setenv('ENABLE_ACTION_FALLBACK', '0')
    with patch.object(agent.world, 'reason', side_effect=ValueError('fixture error')):
        with pytest.raises(ValueError, match='fixture error'):
            agent.choose_action([], observation())
    assert not agent._recovery_mode


def monitor_source():
    source = next(c['source'] for c in build()['cells'] if 'hard_limit_seconds = ' in c['source'])
    start = source.index('        # Include all setup')
    end = source.index('\nexcept Exception:', start)
    return dedent(source[start:end])


@pytest.mark.parametrize('mode', ['normal', 'crash', 'timeout', 'hung_cleanup', 'expired_setup', 'hard_expiry'])
def test_shared_deadline_and_cleanup_paths(monkeypatch, mode):
    clock = [30_601.0 if mode == 'expired_setup' else 29_699.0]
    events = []
    monkeypatch.setenv('ARC3_RUN_STARTED', '0')
    monkeypatch.setattr(time, 'time', lambda: clock[0])

    class Process:
        pid = 123456789
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            events.append(('signal', sig))
            if mode != 'hung_cleanup':
                self.returncode = 0  # Synthetic public scorecard close.
                events.append(('finalized', clock[0]))

        def wait(self, timeout=None):
            events.append(('wait', timeout))
            if self.returncode is not None:
                return self.returncode
            if mode in ('normal', 'crash'):
                self.returncode = 0 if mode == 'normal' else 1
                return self.returncode
            clock[0] += timeout
            if mode == 'hard_expiry':
                clock[0] = 30_601.0
            raise subprocess.TimeoutExpired('synthetic runner', timeout)

    process = Process()

    def killpg(pid, sig):
        assert pid == process.pid and sig == signal.SIGKILL
        events.append(('kill', sig))
        process.returncode = -sig if process.returncode is None else process.returncode

    with (patch.object(subprocess, 'Popen', return_value=process) as launch,
          patch.object(os, 'killpg', side_effect=killpg)):
        exec(monitor_source(), {'os': os, 'subprocess': subprocess, 'sys': sys})
    if mode == 'expired_setup':
        launch.assert_not_called()
        assert not events
        return
    assert events[-1] == ('wait', None)
    assert ('kill', signal.SIGKILL) in events
    if mode in ('timeout', 'hung_cleanup', 'hard_expiry'):
        assert ('signal', signal.SIGINT) in events
        assert events.index(('signal', signal.SIGINT)) < events.index(('kill', signal.SIGKILL))
        cleanup_wait = events[events.index(('signal', signal.SIGINT)) + (2 if mode != 'hung_cleanup' else 1)]
        assert cleanup_wait == ('wait', 0.0 if mode == 'hard_expiry' else 60.0)
    else:
        assert not any(event[0] == 'signal' for event in events)


def test_real_child_finalizes_on_sigint(tmp_path, monkeypatch):
    marker = tmp_path / 'finalized'
    child_source = (
        'import signal, sys, time\nfrom pathlib import Path\n'
        'def finalize(sig, frame):\n'
        f'    Path({str(marker)!r}).write_text("scorecard closed")\n'
        '    sys.exit(0)\n'
        'signal.signal(signal.SIGINT, finalize)\n'
        'time.sleep(10)\n'
    )
    real_popen = subprocess.Popen

    def launch(*args, **kwargs):
        return real_popen([sys.executable, '-c', child_source], start_new_session=True)

    monkeypatch.setenv('ARC3_RUN_STARTED', str(time.time()))
    source = monitor_source().replace('29_700', '0.5').replace('30_600', '2.0')
    with patch.object(subprocess, 'Popen', side_effect=launch):
        exec(source, {'os': os, 'subprocess': subprocess, 'sys': sys})
    assert marker.read_text() == 'scorecard closed'


def python_runner_source(tmp_path):
    source = next(c['source'] for c in build()['cells'] if 'hard_limit_seconds = ' in c['source'])
    # Shell commands are exercised by Kaggle; replace only IPython shell syntax
    # so the exact generated Python and file writes can run in a clean fixture.
    lines = []
    continuation = False
    for line in source.splitlines():
        if line.lstrip().startswith('!') or continuation:
            continued = line.rstrip().endswith('\\')
            if not continuation:
                lines.append(line[:len(line)-len(line.lstrip())] + 'pass')
            continuation = continued
        else:
            lines.append(line)
    return '\n'.join(lines).replace('/kaggle/working', str(tmp_path)).replace(
        '/tmp/agent_preflight', str(tmp_path / 'preflight')).replace(
        '/tmp/agent_bundle', str(tmp_path / 'bundle'))


@pytest.mark.parametrize('audit_failure', [False, True])
def test_generated_rerun_python_and_registry_are_valid(tmp_path, monkeypatch, audit_failure):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', 'True')
    monkeypatch.setenv('ARC3_RUN_STARTED', str(time.time()))
    framework = tmp_path / 'ARC-AGI-3-Agents'
    (framework / 'agents').mkdir(parents=True)
    exec(setup_source(tmp_path), {})
    before = (tmp_path / 'submission.parquet').read_bytes()
    runner = python_runner_source(tmp_path)
    compile(runner, '<generated-run-cell>', 'exec')
    process = SimpleNamespace(pid=123456789, returncode=0, poll=lambda: 0, wait=lambda: 0)
    failure = subprocess.CalledProcessError(1, 'audit') if audit_failure else None
    with (patch.object(subprocess, 'run', side_effect=failure),
          patch.object(subprocess, 'Popen', return_value=process),
          patch.object(os, 'killpg', side_effect=ProcessLookupError)):
        exec(runner, {})
    assert (tmp_path / 'submission.parquet').read_bytes() == before
    if audit_failure:
        assert 'CalledProcessError' in (tmp_path / 'scratch/runner_errors.log').read_text()
    else:
        registry = (framework / 'agents/__init__.py').read_text()
        compile(registry, '<generated-registry>', 'exec')
        assert registry.startswith('from typing import Type\nfrom dotenv')
        assert '\nHOST=gateway\n' in (framework / '.env').read_text()
