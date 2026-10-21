"""Synthetic evidence only: tests never load private game implementations."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/ARC-AGI-3-Agents')]

from arcengine import FrameData, GameAction, GameState

from agent.arclog import PerceptionLog
from agent.my_agent import MyAgent, Playbook, WorldModel
from agent.runtime import PrefixLock
from agent.sandbox import run_sandbox

GOOD = '''
def step(state, action):
    state['grid'][1][1] = (state['grid'][1][1] + 1) % 16
    return state
'''
WRONG = '''
def step(state, action):
    return state
'''
LEGAL = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6]


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    # Explicit reference control; strict Generation 3 has separate acceptance fixtures.
    monkeypatch.setenv('ENABLE_AUDITABLE_WORLD_MODEL', '0')
    for name in ('ENABLE_PLAYBOOK_RESET', 'ENABLE_RETRODICTION', 'ENABLE_EXPECT_QUEUE',
                 'ENABLE_TIER1_SIMULATOR', 'ENABLE_PREFIX_LOCK', 'ENABLE_SANDBOX_LIMITS'):
        monkeypatch.setenv(name, '1')


def board(value=0):
    return [[0, 0, 0], [0, value, 0], [0, 0, 0]]


def plan(value=2, **kwargs):
    return {'action': 'ACTION1', 'expect': [{'x': 1, 'y': 1, 'value': value}],
            'expect_level_up': False, **kwargs}


def world(tmp_path, *, level=1, transitions=1, planner=None):
    log = PerceptionLog(tmp_path / 'log.txt')
    log.start([board()], level=level)
    for i in range(transitions):
        log.record('ACTION1', None, [board((i+1) % 16)], level=level)
    def worker(source):
        return run_sandbox(source, python=ROOT / '.venv_agent_sandbox/bin/python', cwd=ROOT)
    return WorldModel(log, planner or worker, tmp_path, PrefixLock())


def test_counterexample_rejected_at_zero_live_action_cost(tmp_path):
    model = world(tmp_path)
    count = len(model.log.records)
    with patch('agents.agent.Agent.do_action_request') as dispatch:
        report = model.propose_hypothesis(WRONG, plan(1), {(1, 1): 2}, LEGAL)
        dispatch.assert_not_called()
    assert report['status'] == 'counterexample'
    assert report['checked'] == 1 and report['live_actions'] == 0
    assert report['counterexamples'][0]['step'] == 1
    assert not model.queue and len(model.log.records) == count
    assert next(iter(model.memory.facts.values()))['status'] == 'counterexample'


def test_consistent_mechanic_gets_exactly_one_disambiguating_probe(tmp_path):
    model = world(tmp_path)
    report = model.propose_hypothesis(GOOD, plan(), {(1, 1): 3}, LEGAL)
    assert report['status'] == 'pass' and report['complete']
    assert len(model.queue) == 1
    item = model.next_action(LEGAL)
    assert item['probe']
    with pytest.raises(RuntimeError, match='settled observation'):
        model.next_action(LEGAL)
    record = model.log.record('ACTION1', None, [board(2)], level=1)
    model.observe(record)
    assert model.next_action(LEGAL) is None
    assert [e for e in model.events if e['type'] == 'prediction'][-1]['probing']


def test_full_history_including_early_counterexample_and_chunks(tmp_path):
    model = world(tmp_path, transitions=145)
    # Only the oldest transition is incompatible; archived last-128/last-16
    # replay would certify this source incorrectly.
    source = GOOD.replace("return state", "\n    if state['grid'][1][1] == 1:\n        state['grid'][0][0] = 9\n    return state")
    report = model.replay_transitions(source, chunk_size=13)
    assert report['checked'] == report['eligible'] == 145
    assert len(report['chunks']) == 12
    assert report['status'] == 'counterexample'
    assert report['counterexamples'][0]['step'] == 1
    consistent = model.replay_transitions(GOOD, chunk_size=64)
    assert consistent['status'] == 'pass' and consistent['matched'] == 145


@pytest.mark.parametrize('source', ['def step(:', 'def step(s,a): raise RuntimeError("bad")',
                                    'def step(s,a): return True', 'def step(s,a): return {"grid": []}'])
def test_compile_crash_or_malformed_result_never_passes(tmp_path, source):
    model = world(tmp_path)
    report = model.replay_transitions(source)
    assert report['status'] == 'error' and not report['complete']
    assert report['errors'] == 1 and not model.queue


def test_empty_limited_timeout_and_missing_worker_are_not_passes(tmp_path):
    model = world(tmp_path, transitions=0)
    assert model.replay_transitions(GOOD)['status'] == 'empty'
    model.log.record('ACTION1', None, [board(1)], level=1)
    assert model.replay_transitions(GOOD, timeout_s=1e-12)['status'] == 'limited'
    real = model.planner
    model.planner = lambda source: run_sandbox(source, python=ROOT / '.venv_agent_sandbox/bin/python',
                                              cwd=ROOT, timeout=0.2)
    report = model.replay_transitions('def step(s,a):\n    while True: pass')
    assert report['status'] == 'error' and report['failure']['error']['timeout']
    model.planner = lambda source: run_sandbox(source, python=tmp_path / 'missing', cwd=ROOT)
    assert model.replay_transitions(GOOD)['status'] == 'error'
    model.planner = real


def test_explicit_level_and_game_coverage_across_episodes(tmp_path):
    model = world(tmp_path)
    model.log.record('RESET', None, [board()], level=1)
    model.log.record('ACTION1', None, [board(1)], level=1)
    model.log.record('ACTION1', None, [board(2)], level=2)
    # The level boundary is relevant to its starting level, not dropped.
    source = GOOD.replace('return state', "\n    if state['grid'][1][1] == 2: state['level'] += 1\n    return state")
    level = model.replay_transitions(source, scope='level')
    assert level['status'] == 'empty' and level['eligible'] == 0
    game = model.replay_transitions(source, scope='game')
    assert game['status'] == 'pass' and game['eligible'] == 3 and game['excluded'] == 1
    model.log.record('ACTION1', None, [board(3)], level=2)
    assert model.replay_transitions(source, scope='level')['eligible'] == 1


def test_replay_compares_whole_nonhud_board_and_status(tmp_path):
    model = world(tmp_path)
    # Correct causal cell plus an invented change at an unchanged board cell.
    wrong_elsewhere = GOOD.replace('return state', "state['grid'][2][2] = 9\n    return state")
    assert model.replay_transitions(wrong_elsewhere)['status'] == 'counterexample'
    wrong_level = GOOD.replace('return state', "state['level'] += 1\n    return state")
    assert model.replay_transitions(wrong_level)['status'] == 'counterexample'


def test_queue_interrupts_on_first_mismatch_before_next_dispatch(tmp_path):
    model = world(tmp_path)
    model.execute_plan([plan(2), plan(3), plan(4)], LEGAL)
    assert model.next_action(LEGAL)['expect'][0]['value'] == 2
    model.observe(model.log.record('ACTION1', None, [board(9)], level=1))
    assert not model.queue and model.pending is None
    assert model.next_action(LEGAL) is None
    flushed = [e for e in model.events if e['type'] == 'queue_flushed']
    assert flushed[-1]['reason'] == 'cell_mismatch'


@pytest.mark.parametrize('kind', ['reset', 'level', 'terminal', 'full_reset'])
def test_queue_flushes_on_every_boundary_even_predicted_level_up(tmp_path, kind):
    model = world(tmp_path)
    model.execute_plan([plan(2, expect_level_up=kind == 'level'), plan(3)], LEGAL)
    model.next_action(LEGAL)
    record = model.log.record('RESET' if kind == 'reset' else 'ACTION1', None, [board(2)],
                              level=2 if kind == 'level' else 1,
                              state='WIN' if kind == 'terminal' else 'NOT_FINISHED',
                              full_reset=kind == 'full_reset')
    model.observe(record)
    assert not model.queue and model.pending is None


@pytest.mark.parametrize('item', [
    plan(expect=[]), plan(action='ACTION_RIGHT'), plan(action='RESET'),
    plan(action='ACTION7'), plan(action='ACTION6'), plan(action='ACTION6', x=64, y=1),
    plan(x=1), plan(expect_level_up=None), plan(expect=[{'x': 1, 'y': 1, 'value': 16}]),
])
def test_invalid_plan_rejected_before_dispatch(tmp_path, item):
    model = world(tmp_path)
    with pytest.raises(ValueError):
        model.execute_plan([item], LEGAL)
    assert not model.queue


def test_hud_prediction_disallowed_and_stale_scope_flushes(tmp_path):
    model = world(tmp_path)
    with (patch.object(model.log, 'get_hud_mask', return_value=frozenset({(1, 1)})),
          pytest.raises(ValueError, match='HUD')):
        model.execute_plan([plan()], LEGAL)
    model.execute_plan([plan()], LEGAL)
    model.log.record('ACTION2', None, [board(1)], level=1)
    assert model.next_action(LEGAL) is None
    assert model.events[-1]['reason'] == 'stale_state'


def test_expected_queue_reference_ablation_keeps_actions_after_mismatch(tmp_path, monkeypatch):
    model = world(tmp_path)
    monkeypatch.setenv('ENABLE_EXPECT_QUEUE', '0')
    model.execute_plan([{'action': 'ACTION1'}, {'action': 'ACTION2'}], LEGAL)
    model.next_action(LEGAL)
    model.observe(model.log.record('ACTION1', None, [board(9)], level=1))
    assert len(model.queue) == 1
    assert model.next_action(LEGAL)['action'] == 'ACTION2'
    model.observe(model.log.record('RESET', None, [board()], level=1))
    assert not model.queue and model.pending is None


def test_retrodiction_ablation_exposes_bypass(tmp_path, monkeypatch):
    model = world(tmp_path)
    monkeypatch.setenv('ENABLE_RETRODICTION', '0')
    report = model.propose_hypothesis(WRONG, plan(1), {(1, 1): 2}, LEGAL)
    assert report['status'] == 'ablation_bypass' and len(model.queue) == 1
    assert next(iter(model.memory.facts.values()))['status'] == 'unresolved'


def test_probe_must_distinguish_and_match_candidate(tmp_path):
    model = world(tmp_path)
    with pytest.raises(ValueError, match='disambiguate'):
        model.propose_hypothesis(GOOD, plan(), {(1, 1): 2}, LEGAL)
    with pytest.raises(ValueError, match='disagrees'):
        model.propose_hypothesis(GOOD, plan(4), {(1, 1): 2}, LEGAL)
    assert not model.queue


def test_cross_level_playbook_preserves_facts_and_immutable_prefix(tmp_path):
    model = world(tmp_path)
    model.propose_hypothesis(GOOD, plan(), {(1, 1): 3}, LEGAL)
    model.propose_hypothesis(WRONG, plan(1), {(1, 1): 3}, LEGAL)
    model.memory.remember('open', status='unresolved', claim='direction not observed')
    first = model.memory.request()
    old_id = str(model.memory.context)
    model.memory.append('assistant', 'new turn')
    assert model.memory.request()[:len(first)] == first
    old_full = model.memory.request()
    log_bytes = model.log.log_path.read_bytes()
    model.observe(model.log.record('ACTION1', None, [board(2)], level=2))
    assert model.memory.version == 1
    text = (tmp_path / 'playbook.md').read_text()
    assert 'verified' in text and 'counterexample' in text and 'direction not observed' in text
    assert (tmp_path / 'playbook.v0001.md').read_text() == text
    assert text in model.memory.messages[0]['content']
    assert model.log.log_path.read_bytes().startswith(log_bytes)
    assert str(model.memory.context) != old_id
    with pytest.raises(ValueError, match='closed context'):
        model.memory.lock.validate(old_id, old_full)
    model.memory.distill('token_ceiling')
    assert model.memory.version == 2 and (tmp_path / 'playbook.v0001.md').read_text() == text


def test_token_ceiling_and_reference_drain_keep_prefixes_immutable(tmp_path, monkeypatch):
    memory = Playbook(tmp_path, PrefixLock(), token_ceiling=500)
    first = memory.request()
    memory.remember('rule', status='verified', evidence='synthetic step 1')
    memory.append('user', 'a' * 600)
    assert memory.version == 1 and 'synthetic step 1' in memory.messages[0]['content']
    monkeypatch.setenv('ENABLE_PLAYBOOK_RESET', '0')
    version, context = memory.version, memory.context
    memory.append('user', 'b' * 600)
    assert memory.version == version and memory.context > context
    assert 'Persistent section memory' in memory.messages[0]['content']
    assert first[0]['content'].startswith('Use observed evidence')


def test_simulator_compile_then_full_current_level_replay_then_bfs(tmp_path):
    model = world(tmp_path, level=3, transitions=2)
    # Known toy mechanic reaches the next level when center color becomes 3.
    source = GOOD.replace('return state', "\n    if state['grid'][1][1] == 3: state['level'] += 1\n    return state")
    result = model.search_plan(source, LEGAL)
    assert result['status'] == 'found'
    assert result['depth'] == 1 and len(model.queue) == 1
    kinds = [e['type'] for e in model.events]
    assert kinds.index('simulator_compile') < kinds.index('simulator_replay') < kinds.index('search')
    assert model.queue[0]['expect_level_up']


def test_simulator_failure_never_searches_even_with_replay_toggle_off(tmp_path, monkeypatch):
    model = world(tmp_path, level=3)
    monkeypatch.setenv('ENABLE_RETRODICTION', '0')
    assert model.search_plan('def step(:', LEGAL)['status'] == 'compile_failed'
    assert model.search_plan(WRONG, LEGAL)['status'] == 'replay_failed'
    assert not any(e['type'] == 'search' for e in model.events)


def test_tier1_trigger_counts_forty_wasted_moves_and_ablation(tmp_path, monkeypatch):
    model = world(tmp_path, level=3, transitions=0)
    for _ in range(39):
        model.observe(model.log.record('ACTION1', None, [board()], level=3))
    with patch.object(model, 'search_plan', return_value={'status': 'exhausted'}) as search:
        model.reason(LEGAL)
        search.assert_not_called()
        model.observe(model.log.record('ACTION1', None, [board()], level=3))
        assert model.wasted_moves == 40
        model.reason(LEGAL)
        assert search.call_count == 1
        model.reason(LEGAL)
        assert search.call_count == 1
        monkeypatch.setenv('ENABLE_TIER1_SIMULATOR', '0')
        model.observe(model.log.record('ACTION1', None, [board()], level=3))
        model.reason(LEGAL)
        assert search.call_count == 1
    assert any(e.get('reason') == 'tier1_exhausted' for e in model.events)


def test_lookup_synthesizes_only_observed_edges_and_is_not_universal_claim(tmp_path):
    model = world(tmp_path, level=3, transitions=3)
    source = model._lookup_source()
    assert model.replay_transitions(source)['status'] == 'pass'
    assert model.search_plan(source, LEGAL)['status'] == 'exhausted'
    assert not model.queue


def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv('ARC3_LOG_DIR', str(tmp_path))
    return MyAgent(card_id='fixture', game_id='synthetic', agent_name='fixture',
                   ROOT_URL='', record=False, arc_env=SimpleNamespace())


def observation(value, *, level=1):
    return FrameData(frame=[board(value)], state=GameState.NOT_FINISHED,
                     levels_completed=level-1, win_levels=6, available_actions=[1, 2])


def test_framework_backend_rejects_hypothesis_without_live_dispatch(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    agent._ensure_perception(observation(0))
    agent.perception.record('ACTION1', None, [board(1)], level=1)
    agent.world.backend = lambda messages: {'hypotheses': [
        {'source': WRONG, 'probe': plan(1), 'alternative_expect': [{'x': 1, 'y': 1, 'value': 2}]}],
        'actions': [plan(1)]}
    with patch('agents.agent.Agent.do_action_request') as dispatch:
        assert agent.choose_action([], observation(1)) is GameAction.ACTION1
        assert agent._recovery_mode
        assert 'no admissible' in (tmp_path / 'choose_action_errors.log').read_text()
        dispatch.assert_not_called()
    assert not agent.world.queue and len(agent.perception.records) == 1


def test_framework_plan_flushes_after_first_live_cell_mismatch(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    agent.world.backend = lambda messages: {'actions': [plan(1), plan(2)]}
    first = agent.choose_action([], observation(0))
    with patch('agents.agent.Agent.do_action_request', return_value=observation(9)) as dispatch:
        agent.do_action_request(first)
        assert dispatch.call_count == 1
    assert not agent.world.queue and agent.world.pending is None
    events = [e for e in agent.world.events if e['type'] == 'queue_flushed']
    assert events[-1]['reason'] == 'cell_mismatch'


def test_backend_request_prefix_is_immutable_and_returned_alias_cannot_change_it(tmp_path):
    model = world(tmp_path)
    requests = []
    def backend(messages):
        requests.append(json.loads(json.dumps(messages)))
        messages[0]['content'] = 'caller mutation'
        return {'actions': [plan()]}
    model.backend = backend
    model.reason(LEGAL)
    model.queue.clear()
    model.reason(LEGAL)
    assert requests[1][:len(requests[0])] == requests[0]
    assert 'caller mutation' not in model.memory.messages[0]['content']


def test_hud_changes_are_excluded_from_replay_comparison(tmp_path):
    model = world(tmp_path)
    transition = model.log.records[0]
    # Synthetic HUD designation; do not infer it from puzzle-specific positions.
    changed = replace(transition, hud_mask=frozenset({(2, 2)}))
    with patch.object(model, '_relevant', return_value=[changed]):
        source = GOOD.replace('return state', "state['grid'][2][2] = 9\n    return state")
        assert model.replay_transitions(source)['status'] == 'pass'


def test_live_probe_counterexample_demotes_scoped_playbook_fact(tmp_path):
    model = world(tmp_path)
    model.propose_hypothesis(GOOD, plan(), {(1, 1): 3}, LEGAL)
    model.next_action(LEGAL)
    model.observe(model.log.record('ACTION1', None, [board(9)], level=1))
    fact = next(iter(model.memory.facts.values()))
    assert fact['status'] == 'counterexample' and fact['counterexamples'][-1]['step'] == 2
    assert not model.queue


def test_same_source_has_independent_cross_level_evidence(tmp_path):
    model = world(tmp_path)
    model.propose_hypothesis(GOOD, plan(), {(1, 1): 3}, LEGAL)
    model.flush('fixture_boundary')
    model.log.record('ACTION1', None, [board(2)], level=2)
    model.log.record('ACTION1', None, [board(9)], level=2)
    report = model.propose_hypothesis(GOOD, plan(10), {(1, 1): 11}, LEGAL)
    assert report['status'] == 'counterexample'
    assert {v['status'] for v in model.memory.facts.values()} == {'verified', 'counterexample'}
    assert {v['level'] for v in model.memory.facts.values()} == {1, 2}


def test_float_color_cannot_pass_through_normalizer_coercion(tmp_path):
    model = world(tmp_path)
    source = GOOD.replace("+ 1)", "+ 1.5)")
    assert model.replay_transitions(source)['status'] == 'error'


def test_direct_framework_call_cannot_bypass_waiting_queue(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    agent._ensure_perception(observation(0))
    agent.world.execute_plan([plan(1), plan(2)], LEGAL)
    with patch('agents.agent.Agent.do_action_request') as dispatch:
        with pytest.raises(RuntimeError, match='authorization'):
            agent.do_action_request(GameAction.ACTION2)
        dispatch.assert_not_called()
    assert not agent.world.queue


def test_mutable_complex_enum_cannot_silently_retarget_queued_action(tmp_path, monkeypatch):
    agent = make_agent(tmp_path, monkeypatch)
    agent._ensure_perception(observation(0))
    agent.world.execute_plan([plan(1, action='ACTION6', x=1, y=1)], LEGAL)
    agent.world.next_action(LEGAL)
    action = GameAction.ACTION6
    action.set_data({'x': 1, 'y': 1})
    def resolver(proposal, observation, **kwargs):
        proposal.set_data({'x': 2, 'y': 2})
        return proposal
    with (patch.object(agent, '_resolve_action', resolver),
          patch('agents.agent.Agent.do_action_request') as dispatch,
          pytest.raises(RuntimeError, match='blocked')):
        agent.do_action_request(action)
    dispatch.assert_not_called()
