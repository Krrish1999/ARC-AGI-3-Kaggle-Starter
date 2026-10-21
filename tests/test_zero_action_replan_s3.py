"""Counterexample-guided repair contracts; synthetic evidence, never game source."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from arcengine import FrameData, GameAction, GameState

from agent.my_agent import MyAgent
from agent.runtime import TARGET_SECONDS

GOOD = """def step(state, action):
    state['grid'][1][1] = (state['grid'][1][1] + 1) % 16
    return state
"""
WRONG = 'def step(state, action): return state\n'


def board(value=0):
    return [[0, 0, 0], [0, value, 0], [0, 0, 0]]


def observation(value=0, *, level=1):
    return FrameData(frame=[board(value)], state=GameState.NOT_FINISHED,
                     levels_completed=level-1, win_levels=6, available_actions=[1, 2])


def plan(value=1, **updates):
    return {'action': 'ACTION1', 'expect': [{'x': 1, 'y': 1, 'value': value}],
            'expect_level_up': False, **updates}


def hypothesis(source=WRONG):
    return {'hypotheses': [{'source': source, 'probe': plan(2),
                           'alternative_expect': [{'x': 1, 'y': 1, 'value': 3}]}]}


@pytest.fixture
def agent(tmp_path, monkeypatch):
    # Explicit reference control; strict Generation 3 has separate acceptance fixtures.
    monkeypatch.setenv('ENABLE_AUDITABLE_WORLD_MODEL', '0')
    monkeypatch.setenv('ARC3_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('ARC3_SANDBOX_PYTHON', str(
        Path(__file__).resolve().parents[1] / '.venv_agent_sandbox/bin/python'))
    for name in ('ENABLE_ZERO_ACTION_REPLAN', 'ENABLE_ACTION_FALLBACK', 'ENABLE_RETRODICTION',
                 'ENABLE_EXPECT_QUEUE', 'ENABLE_PREFIX_LOCK', 'ENABLE_SANDBOX_LIMITS'):
        monkeypatch.setenv(name, '1')
    candidate = MyAgent(card_id='fixture', game_id='synthetic', agent_name='fixture',
                        ROOT_URL='', record=False, arc_env=SimpleNamespace())
    return candidate


def install_backend(agent, responses):
    sequence = iter(responses)
    histories = []

    def backend(messages):
        histories.append(copy.deepcopy(messages))
        return copy.deepcopy(next(sequence))

    agent.world.backend = backend
    return histories


def kinds(agent, kind):
    return [event for event in agent.world.events if event['type'] == kind]


def test_counterexample_then_correct_mechanic_uses_real_sandbox_and_no_live_action(agent):
    agent._ensure_perception(observation())
    agent.perception.record('ACTION1', None, [board(1)], level=1)
    rejected = hypothesis()
    rejected['actions'] = [plan(1)]  # Must not smuggle this sibling plan through.
    histories = install_backend(agent, [rejected, hypothesis(GOOD)])
    with patch('agents.agent.Agent.do_action_request') as dispatch:
        chosen = agent.choose_action([], observation(1))
        dispatch.assert_not_called()
    assert chosen is GameAction.ACTION1 and not agent._recovery_mode
    assert len(histories) == 2 and len(agent.perception.records) == 1
    assert agent.world.pending['probe']
    assert agent.world.pending['expect'][0]['value'] == 2
    assert histories[1][:len(histories[0])] == histories[0]
    feedback = [json.loads(message['content']) for message in histories[1]
                if message['role'] == 'user']
    counterexample = next(item for item in feedback if item.get('type') == 'replan_rejected')
    assert counterexample['counterexamples'][0]['step'] == 1
    assert counterexample['live_actions'] == 0
    reports = [event['report'] for event in kinds(agent, 'replay')]
    assert [report['status'] for report in reports] == ['counterexample', 'pass']
    assert all(report['checked'] == report['eligible'] == 1 for report in reports)
    assert kinds(agent, 'replan_accepted')[-1]['attempt'] == 2


@pytest.mark.parametrize('bad', [plan(expect=[]), plan(action='RESET'), plan(x=1),
                                 plan(expect_level_up=None), 'not an action'])
def test_invalid_action_proposal_then_valid_plan_is_repaired_without_weakening_gates(agent, bad):
    histories = install_backend(agent, [{'actions': [bad]}, {'actions': [plan()]}])
    assert agent.choose_action([], observation()) is GameAction.ACTION1
    assert len(histories) == 2 and not agent._recovery_mode
    assert agent.world.pending['expect'] == [{'x': 1, 'y': 1, 'value': 1}]
    assert len(agent.perception.records) == 0
    assert kinds(agent, 'replan_rejected')[0]['reason']


@pytest.mark.parametrize('enabled,calls', [('1', 3), ('0', 1)])
def test_bad_proposals_exhaust_exact_cap_and_toggle_off_restores_reference(agent, monkeypatch, enabled, calls):
    monkeypatch.setenv('ENABLE_ZERO_ACTION_REPLAN', enabled)
    histories = install_backend(agent, [{'actions': [plan(expect=[])]}] * calls)
    assert agent.choose_action([], observation()) is GameAction.ACTION1
    assert agent._recovery_mode and len(histories) == calls
    assert agent.world.pending is None and not agent.world.queue
    assert agent._reasoning_deadline is None
    assert all(event['live_actions'] == 0 for event in kinds(agent, 'replan_rejected'))
    if enabled == '1':
        assert kinds(agent, 'replan_exhausted')[-1]['reason'] == 'attempt_cap'
    else:
        assert not kinds(agent, 'replan_attempt')


def test_retry_deadline_shrinks_backend_and_worker_allowance(agent, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr('agent.my_agent.time.monotonic', lambda: clock.now)
    agent._ensure_perception(observation())
    limits = []

    def backend(messages):
        limits.append(agent._inference_limits()[1])
        clock.now += 12
        return {'actions': [plan(expect=[])]}

    agent.world.backend = backend
    agent._reason_with_replanning([GameAction.ACTION1])
    assert limits == [30, 18, 6]
    assert agent._reasoning_deadline is None
    # Same shared deadline also bounds each real planner subprocess.
    agent._reasoning_deadline = 40
    with patch('agent.my_agent.run_sandbox', return_value={'ok': True}) as worker:
        agent.execute_planner('print(1)')
    assert worker.call_args.kwargs['timeout'] == 4
    clock.now = 40
    with pytest.raises(TimeoutError), patch('agent.my_agent.run_sandbox') as worker:
        agent.execute_planner('print(1)')
    worker.assert_not_called()


def test_late_valid_proposal_is_flushed_before_authorization(agent, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr('agent.my_agent.time.monotonic', lambda: clock.now)

    def backend(messages):
        clock.now = 31
        return {'actions': [plan()]}

    agent.world.backend = backend
    assert agent.choose_action([], observation()) is GameAction.ACTION1
    assert agent._recovery_mode and agent.world.pending is None and not agent.world.queue
    assert kinds(agent, 'replan_exhausted')[-1]['reason'] == 'budget'


@pytest.mark.parametrize('deadline', ['level', 'global'])
def test_no_inference_when_level_or_global_budget_is_exhausted(agent, monkeypatch, deadline):
    if deadline == 'level':
        agent.level_budget = SimpleNamespace(level=1, remaining=lambda: 0,
                                             generation_tokens=lambda tokens: 0)
    else:
        monkeypatch.setenv('ARC3_RUN_STARTED', '0')
        monkeypatch.setattr('agent.my_agent.time.time', lambda: TARGET_SECONDS + 1)
    backend = Mock(return_value={'actions': [plan()]})
    agent.world.backend = backend
    assert agent.choose_action([], observation()) is GameAction.ACTION1
    backend.assert_not_called()
    assert agent._recovery_mode


@pytest.mark.parametrize('error', [ValueError('prefix or perception error'), OSError('disk/transport error'),
                                  RuntimeError('tool failed')])
def test_infrastructure_errors_are_not_classified_as_model_repairs(agent, error):
    backend = Mock(side_effect=error)
    agent.world.backend = backend
    assert agent.choose_action([], observation()) is GameAction.ACTION1
    assert agent._recovery_mode and backend.call_count == 1
    assert not kinds(agent, 'replan_rejected')


def test_prefix_mutation_remains_fatal_on_first_attempt(agent):
    def backend(messages):
        agent.world.memory.messages[0]['content'] = 'mutated prefix'
        return {'actions': [plan(expect=[])]}

    agent.world.backend = Mock(side_effect=backend)
    assert agent.choose_action([], observation()) is GameAction.ACTION1
    assert agent._recovery_mode and agent.world.backend.call_count == 1
    assert not kinds(agent, 'replan_rejected')


def test_failed_worker_is_not_retried_as_an_ordinary_counterexample(agent):
    agent._ensure_perception(observation())
    agent.perception.record('ACTION1', None, [board(1)], level=1)
    agent.world.planner = Mock(return_value={'ok': False, 'error': 'missing_sandbox'})
    histories = install_backend(agent, [hypothesis(GOOD)])
    assert agent.choose_action([], observation(1)) is GameAction.ACTION1
    assert agent._recovery_mode and len(histories) == 1
    assert not kinds(agent, 'replan_rejected')
    assert kinds(agent, 'hypothesis_rejected')[-1]['reason'] == 'error'


@pytest.mark.parametrize('tail', ['malformed', 'valid'])
def test_failed_worker_cannot_be_hidden_by_a_later_proposal(agent, tail):
    agent._ensure_perception(observation())
    agent.perception.record('ACTION1', None, [board(1)], level=1)
    real_worker = agent.world.planner
    attempts = []

    def worker(source):
        attempts.append(source)
        if len(attempts) == 1:
            return {'ok': False, 'error': 'missing_sandbox'}
        return real_worker(source)

    agent.world.planner = worker
    response = hypothesis(GOOD)
    response['hypotheses'].append({} if tail == 'malformed' else hypothesis(GOOD)['hypotheses'][0])
    histories = install_backend(agent, [response])
    assert agent.choose_action([], observation(1)) is GameAction.ACTION1
    assert agent._recovery_mode and len(histories) == 1
    assert agent.world.pending is None and not agent.world.queue
    assert not kinds(agent, 'replan_rejected') and not kinds(agent, 'replan_accepted')
    assert kinds(agent, 'hypothesis_rejected')[0]['reason'] == 'error'


def test_pending_action_cannot_be_overwritten(agent):
    agent._ensure_perception(observation())
    agent.world.execute_plan([plan()], [GameAction.ACTION1])
    agent.world.next_action([GameAction.ACTION1])
    agent.world.backend = Mock(return_value={'actions': [plan(2)]})
    with pytest.raises(RuntimeError, match='settled observation'):
        agent._reason_with_replanning([GameAction.ACTION1])
    agent.world.backend.assert_not_called()
    assert agent.world.pending['expect'][0]['value'] == 1


def test_tier1_revision_guard_and_no_backend_reference_are_unchanged(agent):
    agent._ensure_perception(observation(level=3))
    agent.world.wasted_moves = 40
    histories = install_backend(agent, [{'simulator_source': GOOD}])
    with patch.object(agent.world, 'search_plan', return_value={'status': 'exhausted'}) as search:
        agent._reason_with_replanning([GameAction.ACTION1])
    assert search.call_count == 1 and len(histories) == 1
    assert not kinds(agent, 'replan_attempt')
    assert agent.world._tier_attempt == (3, 0)
    agent.world.backend = None
    agent.world.wasted_moves = 0
    agent._reason_with_replanning([GameAction.ACTION1])
    assert not kinds(agent, 'replan_attempt')
