"""Generation 3 contracts exercised with real seccomp workers and toy grids."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from arcengine import FrameData, GameAction, GameState

from agent.arclog import PerceptionLog, freeze_grid
from agent.my_agent import MyAgent, PlanningHalted, ProposalValidationError, WorldModel
from agent.runtime import PrefixLock
from agent.sandbox import run_sandbox

ROOT = Path(__file__).resolve().parents[1]
LEGAL = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION6]
GOOD = '''def step(state, action):
    if action['action'] == 'RESET':
        state['grid'][1][1] = 0
        state['state'] = 'NOT_FINISHED'
    elif action['action'] == 'ACTION1':
        state['grid'][1][1] = (state['grid'][1][1] + 1) % 16
    return state
'''
WRONG = 'def step(state, action): return state\n'


@pytest.fixture(autouse=True)
def strict_contract(monkeypatch):
    for flag in ('ENABLE_AUDITABLE_WORLD_MODEL', 'ENABLE_RETRODICTION', 'ENABLE_EXPECT_QUEUE',
                 'ENABLE_TIER1_SIMULATOR', 'ENABLE_ZERO_ACTION_REPLAN', 'ENABLE_PLAYBOOK_RESET',
                 'ENABLE_ANIMATION_DIFF', 'ENABLE_HUD_MASK', 'ENABLE_DEAD_SIGNATURE',
                 'ENABLE_DOUBLE_RESET_GUARD', 'ENABLE_PREFIX_LOCK', 'ENABLE_SANDBOX_LIMITS'):
        monkeypatch.setenv(flag, '1')
    monkeypatch.setenv('ARC3_SANDBOX_PYTHON', str(ROOT / '.venv_agent_sandbox/bin/python'))
    monkeypatch.setenv('ENABLE_OFFLINE_BACKEND', '0')


def board(value=0):
    return [[0, 0, 0], [0, value, 0], [0, 0, 0]]


def observation(value=0, *, level=1, state=GameState.NOT_FINISHED):
    return FrameData(frame=[board(value)], state=state, levels_completed=level-1,
                     win_levels=6, available_actions=[1, 2, 6])


def plan(value, **updates):
    return dict(action='ACTION1', expect={(1, 1): value}, expect_level_up=False, **updates)


def model(tmp_path, *, level=1, transitions=1):
    log = PerceptionLog(tmp_path / 'log.txt')
    log.start([board()], level=level)
    for i in range(transitions):
        log.record('ACTION1', None, [board((i+1) % 16)], level=level)
    return WorldModel(log, lambda source: run_sandbox(
        source, python=ROOT / '.venv_agent_sandbox/bin/python', cwd=ROOT), tmp_path, PrefixLock())


def agent(tmp_path, monkeypatch, *, level=1, transitions=1):
    monkeypatch.setenv('ARC3_LOG_DIR', str(tmp_path))
    candidate = MyAgent(card_id='fixture', game_id='synthetic', agent_name='fixture',
                        ROOT_URL='', record=False, arc_env=SimpleNamespace())
    candidate._ensure_perception(observation(level=level))
    for i in range(transitions):
        candidate.perception.record('ACTION1', None, [board((i+1) % 16)], level=level)
    return candidate


def backend(candidate, responses):
    responses = iter(responses)
    requests = []
    def call(messages):
        requests.append(copy.deepcopy(messages))
        return next(responses)
    candidate.world.backend = call
    return requests


def json_plan(value):
    return {'action': 'ACTION1', 'expect': [{'x': 1, 'y': 1, 'value': value}],
            'expect_level_up': False}


def test_default_contract_is_enabled(monkeypatch):
    from agent.arclog import auditable
    monkeypatch.delenv('ENABLE_AUDITABLE_WORLD_MODEL')
    assert auditable()


def test_earliest_counterexample_stops_complete_game_authorization(tmp_path):
    world = model(tmp_path, transitions=0)
    world.log.record('ACTION1', None, [board(2)], level=1)
    world.log.record('ACTION2', None, [board(2)], level=2)
    world.log.record('ACTION1', None, [board(3)], level=2)
    source = GOOD.replace("    return state", "    if action['action'] == 'ACTION2': state['level'] += 1\n    return state")
    with patch('agents.agent.Agent.do_action_request') as live:
        result = world.propose_hypothesis(source, plan(4), {(1, 1): 5}, LEGAL, scope='level')
    live.assert_not_called()
    assert result['scope'] == 'game' and result['eligible'] == 3
    assert result['status'] == 'counterexample' and result['checked'] == 1
    assert result['counterexamples'][0]['step'] == 1 and result['live_actions'] == 0
    assert not world.queue


def test_resets_and_unrendered_responses_are_never_dropped(tmp_path):
    world = model(tmp_path)
    world.log.record('RESET', None, [board()], level=1)
    world.log.record('ACTION1', None, [board(1)], level=1)
    source = GOOD.replace("if action['action'] == 'RESET':", "if False:")
    result = world.replay_transitions(source, scope='game')
    assert result['eligible'] == 3 and result['excluded'] == 0
    assert result['status'] == 'counterexample' and result['counterexamples'][0]['step'] == 2
    assert world.replay_transitions(GOOD, scope='game')['matched'] == 3
    world.log.record('ACTION2', None, [], level=1, state='GAME_OVER')
    result = world.replay_transitions(GOOD, scope='game')
    assert result['eligible'] == 4 and result['counterexamples'][0]['step'] == 4


@pytest.mark.parametrize('tamper', ['edit', 'truncate', 'extra'])
def test_journal_tampering_cannot_authorize_action(tmp_path, tamper):
    world = model(tmp_path)
    path = world.log.log_path
    lines = path.read_text().splitlines()
    if tamper == 'edit':
        entry = json.loads(lines[-1]); entry['after']['grid'][1][1] = 9
        lines[-1] = json.dumps(entry)
    elif tamper == 'truncate':
        lines.pop()
    else:
        lines.append(lines[-1])
    path.write_text('\n'.join(lines)+'\n')
    with pytest.raises(ValueError, match='log.txt'):
        world.propose_hypothesis(GOOD, plan(2), {(1, 1): 3}, LEGAL)
    assert not world.queue


def test_log_keeps_animation_raw_evidence_and_exact_causal_diff(tmp_path):
    world = model(tmp_path, transitions=0)
    before = world.log.log_path.read_bytes()
    flash = board(7)
    transition = world.log.record('ACTION6', (1, 1), [flash, board()], level=1)
    assert transition.intermediate == (freeze_grid(flash),)
    assert transition.transient and not transition.settled
    assert world.log.log_path.read_bytes().startswith(before)
    entry = json.loads(world.log.log_path.read_text().splitlines()[-1])
    assert entry['before']['grid'] == board() and entry['after']['grid'] == board()
    assert entry['intermediate'] == [flash] and entry['[DIFF]'] == []
    assert entry['[TRANSIENT_ANIMATION_DIFF]'] and not entry['dead_signatures']


def test_hud_is_not_part_of_exact_diff_or_dead_object_mechanics(tmp_path):
    def timer(count):
        grid = [[0]*64 for _ in range(64)]
        grid[20][20] = 4
        grid[0] = [6 if x < count else 0 for x in range(64)]
        return grid
    log = PerceptionLog(tmp_path / 'log.txt'); log.start([timer(64)], level=1)
    for action, count in [('ACTION1', 63), ('ACTION2', 62), ('ACTION1', 61)]:
        log.record(action, None, [timer(count)], level=1)
    for count in (60, 59):
        log.record('ACTION6', (20, 20), [timer(count)], level=1)
    entry = json.loads(log.log_path.read_text().splitlines()[-1])
    assert entry['[HUD_STEP_BUDGET_STRIP]'] and entry['[RAW_SETTLED_DIFF]']
    assert entry['[DIFF]'] == entry['[CAUSAL_BOARD_DIFF]'] == []
    assert entry['dead_signatures'] and not entry['settled']


def test_unknown_history_only_buys_one_explicit_bootstrap_probe(tmp_path):
    world = model(tmp_path, transitions=0)
    with pytest.raises(ProposalValidationError, match='mechanic_source'):
        world.execute_plan([plan(1)], LEGAL)
    assert world.verified_actions(GOOD, [plan(1)], LEGAL)['status'] == 'empty'
    result = world.propose_hypothesis(GOOD, plan(1), {(1, 1): 2}, LEGAL)
    assert result['status'] == 'bootstrap' and len(world.queue) == 1
    assert not any(f['status'] == 'verified' for f in world.memory.facts.values())


def test_forward_rollout_rejects_inconsistent_expect_at_zero_cost(tmp_path):
    world = model(tmp_path)
    with (patch('agents.agent.Agent.do_action_request') as live,
          pytest.raises(ProposalValidationError, match='index 1')):
        world.verified_actions(GOOD, [plan(2), plan(9)], LEGAL)
    live.assert_not_called(); assert not world.queue


def test_expect_queue_has_at_most_one_bad_live_action_and_reprompts_counterexample(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch)
    requests = backend(candidate, [
        {'mechanic_source': GOOD, 'actions': [json_plan(2), json_plan(3), json_plan(4)]},
        {'mechanic_source': WRONG, 'actions': [json_plan(9)]}]*2)
    chosen = candidate.choose_action([], observation(1))
    assert len(requests) == 1
    with patch('agents.agent.Agent.do_action_request', return_value=observation(9)) as live:
        candidate.do_action_request(chosen)
        assert live.call_count == 1
        assert not candidate.world.queue and candidate.world.pending is None
        with pytest.raises(PlanningHalted):
            candidate.choose_action([], observation(9))
        assert live.call_count == 1
    assert any('prediction_counterexample' in message['content'] for message in requests[1])
    assert not candidate._recovery_mode


def test_valid_queue_dispatches_without_more_inference(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch)
    requests = backend(candidate, [{'mechanic_source': GOOD, 'actions': [json_plan(2), json_plan(3)]}])
    first = candidate.choose_action([], observation(1))
    with patch('agents.agent.Agent.do_action_request', return_value=observation(2)) as live:
        candidate.do_action_request(first)
        second = candidate.choose_action([], observation(2))
        live.return_value = observation(3)
        candidate.do_action_request(second)
    assert len(requests) == 1 and live.call_count == 2
    assert not candidate.world.queue and candidate.world.pending is None


@pytest.mark.parametrize('boundary', ['reset', 'level', 'full_reset', 'level_completed'])
def test_every_boundary_flushes_checked_queue(tmp_path, boundary):
    world = model(tmp_path)
    world.verified_actions(GOOD, [plan(2), plan(3)], LEGAL)
    world.next_action(LEGAL)
    transition = world.log.record('RESET' if boundary == 'reset' else 'ACTION1', None,
                                  [board(2)], level=2 if boundary == 'level' else 1,
                                  full_reset=boundary == 'full_reset',
                                  level_completed=boundary == 'level_completed')
    world.observe(transition)
    assert not world.queue and world.pending is None


def test_external_latest_frame_cannot_continue_stale_queue(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch)
    backend(candidate, [{'mechanic_source': GOOD, 'actions': [json_plan(2), json_plan(3)]}])
    first = candidate.choose_action([], observation(1))
    with patch('agents.agent.Agent.do_action_request', return_value=observation(2)):
        candidate.do_action_request(first)
    candidate.world.backend = None
    with pytest.raises(PlanningHalted):
        candidate.choose_action([], observation(8))
    assert not candidate.world.queue and candidate.perception.current.grid == freeze_grid(board(8))
    assert 'observation_resync' in candidate.perception.log_path.read_text()


def test_two_section_playbook_proofs_persist_and_working_memory_clears(tmp_path):
    world = model(tmp_path)
    world.verified_actions(GOOD, [plan(2)], LEGAL)
    world.flush('fixture')
    world.memory.working.update(active_coordinates=[(1, 1)], sub_goals=['old coordinate'],
                                dead_signatures=[['inert']], falsified_paths=['wrong turn'])
    old = world.memory.request()
    world.memory.append('user', {'note': 'append only'})
    assert world.memory.request()[:len(old)] == old
    world.memory.distill('token_ceiling')
    first = (tmp_path / 'playbook.md').read_text()
    assert '[Verified: steps 1]' in first and 'old coordinate' in first
    world.observe(world.log.record('ACTION2', None, [board(1)], level=2))
    second = (tmp_path / 'playbook.md').read_text()
    assert [line for line in second.splitlines() if line.startswith('## ')] == [
        '## 1. Verified World Model', '## 2. Current Level Working Memory']
    assert '[Verified: steps 1]' in second
    assert 'old coordinate' not in second and 'wrong turn' not in second and 'inert' not in second
    assert world.memory.working == {} and first == (tmp_path / 'playbook.v0001.md').read_text()
    assert second in world.memory.request()[0]['content']


def test_falsified_model_is_not_promoted_to_persistent_section(tmp_path):
    world = model(tmp_path)
    world.propose_hypothesis(GOOD, plan(2), {(1, 1): 3}, LEGAL)
    world.next_action(LEGAL)
    world.observe(world.log.record('ACTION1', None, [board(9)], level=1))
    world.memory.distill('token_ceiling')
    persistent = (tmp_path / 'playbook.md').read_text().split('## 2.')[0]
    assert '[Verified:' not in persistent
    assert world.memory.working['falsified_paths']


def test_fifo_never_evicts_turns_even_with_memory_ablation(tmp_path, monkeypatch):
    world = model(tmp_path)
    world.memory.token_ceiling = 50_000
    for i in range(20):
        world.memory.append('user', {'turn': i})
    old = world.memory.request()
    assert all(json.loads(message['content'])['turn'] == i for i, message in enumerate(old[1:]))
    monkeypatch.setenv('ENABLE_PLAYBOOK_RESET', '0')
    world.memory.token_ceiling = 1
    world.memory.append('user', {'turn': 20})
    assert world.memory.request()[:len(old)] == old


def test_total_actions_trigger_even_when_every_action_changes_board(tmp_path):
    world = model(tmp_path, level=3, transitions=39)
    assert not world.tier_due() and world.wasted_moves == 0
    world.log.record('ACTION1', None, [board(8)], level=3)
    assert world.tier_due()
    world._install_plan([plan(9)], LEGAL)
    assert world.next_action(LEGAL) is None and not world.queue


@pytest.mark.parametrize('before_state', ['NOT_FINISHED', 'GAME_OVER'])
def test_one_self_reset_triggers_and_cannot_erase_escalation(tmp_path, before_state):
    world = model(tmp_path, level=3)
    if before_state == 'GAME_OVER':
        world.log.synchronize([board(1)], level=3, state=before_state)
    world.log.record('RESET', None, [board()], level=3)
    assert world.tier_due()
    world.observe(world.log.records[-1])
    assert world.tier_due()
    world.observe(world.log.record('ACTION2', None, [board()], level=4))
    assert not world.tier_due()


def test_lifecycle_start_reset_does_not_trigger_simulator(tmp_path):
    world = model(tmp_path, level=3, transitions=0)
    world.log.synchronize([board()], level=3, state='NOT_PLAYED')
    world.log.record('RESET', None, [board()], level=3)
    assert not world.tier_due()


def test_stuck_simulator_real_replay_and_bfs_yields_winning_checked_queue(tmp_path):
    world = model(tmp_path, level=3, transitions=2)
    world.log.record('RESET', None, [board()], level=3)
    source = GOOD.replace('    return state',
                          "    if state['grid'][1][1] == 3: state['level'] += 1\n    return state")
    world.backend = lambda messages: {'simulator_source': source}
    world.reason(LEGAL)
    assert world.tier_due() and world.queue_kind == 'simulator' and len(world.queue) == 3
    searches = [event for event in world.events if event['type'] == 'search']
    assert searches[-1]['report']['effective_limits']['timeout_s'] == 14
    assert world.next_action(LEGAL)['simulator']
    assert world.queue[-1]['expect_level_up']


def test_tier2_repairs_failed_synthesis_at_zero_live_cost(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch, level=3, transitions=2)
    candidate.perception.record('RESET', None, [board()], level=3)
    source = GOOD.replace('    return state',
                          "    if state['grid'][1][1] == 3: state['level'] += 1\n    return state")
    requests = backend(candidate, [{'simulator_source': WRONG}, {'simulator_source': source}])
    with patch('agents.agent.Agent.do_action_request') as live:
        chosen = candidate.choose_action([], observation(0, level=3))
    live.assert_not_called()
    assert chosen is GameAction.ACTION1 and len(requests) == 2
    assert candidate.world.pending['simulator'] and not candidate._recovery_mode
    assert any(event['type'] == 'tier2_required' for event in candidate.world.events)
    assert requests[1][:len(requests[0])] == requests[0]


@pytest.mark.parametrize('response', [{'simulator_source': WRONG}, {'simulator_source': 'def step(:'},
                                     {'actions': [json_plan(1)]}])
def test_failed_stuck_planner_halts_without_fallback_action(tmp_path, monkeypatch, response):
    candidate = agent(tmp_path, monkeypatch, level=3)
    candidate.perception.record('RESET', None, [board()], level=3)
    candidate.world.backend = Mock(return_value=response)
    with patch('agents.agent.Agent.do_action_request') as live:
        with pytest.raises(PlanningHalted):
            candidate.choose_action([], observation(0, level=3))
        with pytest.raises(PlanningHalted):
            candidate.do_action_request(GameAction.ACTION1)
    live.assert_not_called()
    assert candidate.world.backend.call_count == 3 and not candidate._recovery_mode
    assert candidate.world.halted and not candidate.world.queue


def test_no_backend_or_direct_action_never_spends_blind_move(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch)
    with patch('agents.agent.Agent.do_action_request') as live:
        with pytest.raises(PlanningHalted):
            candidate.do_action_request(GameAction.ACTION1)
        with pytest.raises(PlanningHalted):
            candidate.choose_action([], observation(1))
    live.assert_not_called()
    assert not candidate._recovery_mode


def test_journal_display_block_tampering_is_also_detected(tmp_path):
    world = model(tmp_path)
    path = world.log.log_path
    path.write_text(path.read_text().replace('0 -> 1', '0 -> 9'))
    with pytest.raises(ValueError, match='append-only'):
        world.replay_transitions(GOOD, scope='game')


def test_successful_complete_history_has_no_fifo_case_limit(tmp_path):
    world = model(tmp_path, transitions=145)
    report = world.replay_transitions(GOOD, scope='game', chunk_size=16)
    assert report['complete'] and report['status'] == 'pass'
    assert report['eligible'] == report['matched'] == report['checked'] == 145
    assert len(report['chunks']) == 10


def test_hud_mask_survives_long_runs_of_the_same_action(tmp_path):
    def grid(count):
        result = [[0]*64 for _ in range(64)]
        result[0] = [6 if x < count else 0 for x in range(64)]
        return result
    log = PerceptionLog(tmp_path / 'log.txt'); log.start([grid(64)], level=1)
    for name, count in [('ACTION1', 63), ('ACTION2', 62), ('ACTION1', 61),
                        ('ACTION6', 60), ('ACTION6', 59), ('ACTION6', 58), ('ACTION6', 57)]:
        transition = log.record(name, None, [grid(count)], level=1)
    assert len(transition.hud_mask) == 64 and not transition.settled


def test_long_full_board_simulator_plan_fits_seccomp_output_cap(tmp_path):
    grid = [[0]*64 for _ in range(64)]
    log = PerceptionLog(tmp_path / 'log.txt'); log.start([grid], level=3)
    grid[1][1] = 1
    log.record('ACTION1', None, [grid], level=3)
    world = WorldModel(log, lambda source: run_sandbox(
        source, python=ROOT / '.venv_agent_sandbox/bin/python', cwd=ROOT), tmp_path, PrefixLock())
    source = '''def step(state, action):
    if action['action'] == 'ACTION1':
        value = state['grid'][1][1] + 16 * state['grid'][1][2] + 1
        state['grid'][1][1] = value % 16
        state['grid'][1][2] = value // 16
        if value == 24: state['level'] += 1
    return state
'''
    report = world.search_plan(source, [GameAction.ACTION1])
    assert report['status'] == 'found' and len(world.queue) == 23
    assert all(len(item['expect']) == 4096 for item in world.queue)
    assert world.queue[-1]['expect_level_up']


def test_actions_without_mechanic_are_repaired_before_authorization(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch)
    requests = backend(candidate, [{'actions': [json_plan(2)]},
                                   {'mechanic_source': GOOD, 'actions': [json_plan(2), json_plan(3)]}])
    with patch('agents.agent.Agent.do_action_request') as live:
        candidate.choose_action([], observation(1))
    live.assert_not_called()
    assert len(requests) == 2 and candidate.world.pending
    assert 'mechanic_source' in next(event['reason'] for event in candidate.world.events
                                     if event['type'] == 'replan_rejected')


def test_framework_halt_closes_scorecard_without_dispatch(tmp_path, monkeypatch):
    candidate = agent(tmp_path, monkeypatch)
    with (patch('agents.agent.Agent.main', side_effect=PlanningHalted('no verified route')),
          patch.object(candidate, 'cleanup') as cleanup,
          patch('agents.agent.Agent.do_action_request') as live):
        candidate.main()
    cleanup.assert_called_once(); live.assert_not_called()
