"""S1-P fixtures use synthetic observations; never read game implementations."""
from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/ARC-AGI-3-Agents')]

from arcengine import FrameData, GameAction, GameState

from agent import arclog
from agent.arclog import FrameView, PerceptionLog, freeze_grid
from agent.my_agent import MyAgent


@pytest.fixture(autouse=True)
def default_flags(monkeypatch):
    # Explicit reference control; strict Generation 3 has separate acceptance fixtures.
    monkeypatch.setenv('ENABLE_AUDITABLE_WORLD_MODEL', '0')
    for name in ('ENABLE_HUD_MASK', 'ENABLE_ANIMATION_DIFF', 'ENABLE_DEAD_SIGNATURE',
                 'ENABLE_DOUBLE_RESET_GUARD'):
        monkeypatch.setenv(name, '1')


def board(size=8, *, x=2, y=3, color=4):
    grid = [[0]*size for _ in range(size)]
    grid[y][x] = color
    return grid


def observation(grid, *, state=GameState.NOT_FINISHED, level=1, actions=(1, 2, 6), animation=None):
    return FrameData(frame=animation if animation is not None else [grid], state=state,
                     levels_completed=level-1, win_levels=6, available_actions=list(actions))


def logger(grid, *, level=1, path=None):
    log = PerceptionLog(path)
    log.start([grid], level=level)
    return log


def test_reverting_flash_animation(monkeypatch, tmp_path):
    before = board()
    flash = board()
    flash[3][2] = 7
    log = logger(before, path=tmp_path/'log.txt')
    target = log.current.object_at(2, 3)
    assert target is not None
    for _ in range(2):
        record = log.record('ACTION6', (2, 3), [flash, before], level=1)
        assert record.settled == ()
        assert [(e.old, e.new) for e in record.transient] == [(4, 7), (7, 4)]
        assert not record.zero_causal_diff
    assert not log.is_dead_signature('ACTION6', target.shape_hash, target.color)
    assert record.intermediate == (freeze_grid(flash),)
    assert json.loads(record.evidence_json)['animation']['transient_pixels'] == 1
    monkeypatch.setenv('ENABLE_ANIMATION_DIFF', '0')
    reference = logger(before)
    for _ in range(2):
        record = reference.record('ACTION6', (2, 3), [flash, before], level=1)
        assert record.transient == ()
        assert record.zero_causal_diff
    assert reference.is_dead_signature('ACTION6', target.shape_hash, target.color)


def test_object_translation_and_component_contract(monkeypatch):
    before = board(x=2)
    after = board(x=4)
    log = logger(before)
    obj = next(o for o in arclog.get_board_objects(log) if o.color == 4)
    record = log.record('ACTION2', None, [after], level=1)
    translated = next(o for o in arclog.get_board_objects(log.current) if o.color == 4)
    assert obj.shape_hash == translated.shape_hash
    assert obj.bounding_box == (2, 3, 2, 3)
    assert obj.area == 1 and obj.centroid == (2.0, 3.0)
    assert obj.adjacency_list
    diff = json.loads(record.object_diffs_json)
    assert any(item['from'] == [3, 2] and item['to'] == [3, 4] for item in diff['moved'])
    assert {(d.x, d.y, d.old, d.new) for d in record.settled} == {(2, 3, 4, 0), (4, 3, 0, 4)}
    # Same public graph and signature contract with no optional scientific wheels.
    accelerated = FrameView(freeze_grid(before)).segmentation
    with patch.object(arclog, 'ndimage', None), patch.object(arclog, 'nx', None):
        assert FrameView(freeze_grid(before)).segmentation == accelerated


def obj_frame(grid):
    return FrameView(freeze_grid(grid))


def timer_grid(count, *, column=False, border=True):
    grid = board(64, x=20, y=20)
    for i in range(64):
        if column:
            grid[i][0 if border else 10] = int(i < count)*6
        else:
            grid[0 if border else 10][i] = int(i < count)*6
    return grid


@pytest.mark.parametrize('column', [False, True])
def test_monotonic_border_timer_and_ablation(monkeypatch, column):
    log = logger(timer_grid(64, column=column))
    for action, count in [('ACTION1', 63), ('ACTION2', 62), ('ACTION1', 61)]:
        record = log.record(action, None, [timer_grid(count, column=column)], level=1)
    mask = arclog.get_hud_mask(log)
    expected = frozenset((0, y) if column else (x, 0) for x in range(64) for y in ([x] if column else [0]))
    assert mask == expected
    assert record.raw_settled and not record.settled
    assert json.loads(record.object_diffs_json)['causal_board']['changed_cell_count'] == 0
    assert record.remaining_step_budget_estimate == 61
    assert len(record.hud_mask) == 64
    monkeypatch.setenv('ENABLE_HUD_MASK', '0')
    reference = logger(timer_grid(64, column=column))
    for action, count in [('ACTION1', 63), ('ACTION2', 62), ('ACTION1', 61)]:
        record = reference.record(action, None, [timer_grid(count, column=column)], level=1)
    assert not arclog.get_hud_mask(reference)
    assert len(record.settled) == 1 and record.remaining_step_budget_estimate is None


@pytest.mark.parametrize('border,actions', [(False, ['ACTION1','ACTION2','ACTION1']),
                                            (True, ['ACTION1']*3)])
def test_timer_requires_border_and_action_diversity(border, actions):
    log = logger(timer_grid(64, border=border))
    for action, count in zip(actions, (63, 62, 61)):
        log.record(action, None, [timer_grid(count, border=border)], level=1)
    assert not log.get_hud_mask()


def test_inert_signature_exact_threshold_action_scope_and_boundaries(monkeypatch):
    grid = board()
    log = logger(grid)
    target = log.current.object_at(2, 3)
    assert target is not None
    key = ('ACTION6', target.shape_hash, target.color)
    log.record('ACTION6', (2, 3), [grid], level=1)
    assert not arclog.is_dead_signature(log, *key)
    log.record('ACTION6', (2, 3), [grid], level=1)
    assert arclog.is_dead_signature(log, *key)
    assert not arclog.is_dead_signature(log, 'ACTION1', *key[1:])
    assert log.blocked_reason('ACTION6', (2, 3)) == 'dead_signature'
    monkeypatch.setenv('ENABLE_DEAD_SIGNATURE', '0')
    assert not arclog.is_dead_signature(log, *key)
    assert log.blocked_reason('ACTION6', (2, 3)) is None
    monkeypatch.setenv('ENABLE_DEAD_SIGNATURE', '1')
    log.record('ACTION6', (2, 3), [board(color=5)], level=1)
    assert not log.is_dead_signature(*key)
    for _ in range(2):
        log.record('ACTION6', (2, 3), [board(color=5)], level=1)
    assert log.is_dead_signature('ACTION6', target.shape_hash, 5)
    log.record('RESET', None, [grid], level=1)
    assert not log.is_dead_signature('ACTION6', target.shape_hash, 5)
    for _ in range(2):
        log.record('ACTION6', (2, 3), [grid], level=1)
    log.record('ACTION2', None, [grid], level=2)
    assert not log.is_dead_signature(*key)
    assert log.current.episode > 0


def test_archived_noop_reference_blocks_then_allows_insistence(monkeypatch):
    monkeypatch.setenv('ENABLE_DEAD_SIGNATURE', '0')
    grid = board()
    log = logger(grid, level=2)
    log.record('ACTION6', (2, 3), [grid], level=2)
    assert log.blocked_reason('ACTION6', (2, 3)) == 'reference_noop'
    assert log.blocked_reason('ACTION6', (2, 3)) is None
    assert log.blocked_reason('ACTION6', (2, 3), expected_step=0) == 'stale_observation'


def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv('ARC3_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('ARC3_SEED', '19')
    return MyAgent(card_id='fixture', game_id='synthetic', agent_name='fixture',
                   ROOT_URL='', record=False, arc_env=SimpleNamespace())


def test_consecutive_reset_blocked_before_dispatch_and_ablation(monkeypatch, tmp_path):
    agent = make_agent(tmp_path, monkeypatch)
    initial = observation(board(), state=GameState.GAME_OVER, actions=(1,))
    agent._ensure_perception(initial)
    calls = []

    def dispatch(self, action):
        calls.append(action.name)
        return initial

    with patch('agents.agent.Agent.do_action_request', dispatch):
        agent.do_action_request(GameAction.RESET)
        agent.do_action_request(GameAction.RESET)
        assert calls == ['RESET', 'ACTION1']
        assert len(agent.perception.records) == 2
        assert agent.perception.blocked_proposals[-1]['reason'] == 'consecutive_reset'
        monkeypatch.setenv('ENABLE_DOUBLE_RESET_GUARD', '0')
        agent.do_action_request(GameAction.RESET)
        agent.do_action_request(GameAction.RESET)
        assert calls[-2:] == ['RESET', 'RESET']


def test_choose_action_intercepts_dead_target_and_records_final_response(monkeypatch, tmp_path):
    agent = make_agent(tmp_path, monkeypatch)
    grid = board()
    frame = observation(grid, actions=(6,))
    agent._ensure_perception(frame)
    for _ in range(2):
        agent.perception.record('ACTION6', (2, 3), [grid], level=1)
    proposal = GameAction.ACTION6
    proposal.set_data({'x': 2, 'y': 3})
    calls = []
    def dispatch(self, action):
        calls.append(dict(action.action_data.model_dump()))
        return frame
    with patch('agents.agent.Agent.do_action_request', dispatch):
        agent.do_action_request(proposal)
    assert calls[0] != {'x':2,'y':3}
    assert len(agent.perception.records) == 3
    assert agent.perception.blocked_proposals[-1]['reason'] == 'dead_signature'


def test_immutable_complete_evidence_and_separate_refusals(tmp_path):
    before, after = board(), board(x=4)
    log = logger(before, path=tmp_path/'log.txt')
    record = log.record('ACTION2', None, [after], level=1)
    before[3][2] = 9
    after[3][4] = 9
    assert record.before.grid[3][2] == 4 and record.after.grid[3][4] == 4
    with pytest.raises(FrozenInstanceError):
        record.action = 'RESET'
    original = (tmp_path/'log.txt').read_bytes()
    log.note_blocked('RESET', None, 'consecutive_reset')
    assert (tmp_path/'log.txt').read_bytes().startswith(original)
    assert len(log.records) == 1
    lines = [json.loads(line) for line in (tmp_path/'log.txt').read_text().splitlines()]
    assert lines[0]['[DIFF]'] and lines[0]['live_action']
    assert lines[1]['type'] == 'blocked_proposal' and not lines[1]['live_action']
    assert arclog.get_step_diffs(log) is record


def test_outer_history_is_not_animation(monkeypatch, tmp_path):
    agent = make_agent(tmp_path, monkeypatch)
    history = [observation(board(color=7))]
    current = observation(board(), actions=(1,))
    agent.choose_action(history, current)
    assert not agent.perception.records
    assert agent.perception.current.grid == freeze_grid(board())


def test_border_flash_is_not_masked_as_a_timer():
    log = logger(timer_grid(64))
    for action, count in [('ACTION1', 63), ('ACTION2', 62)]:
        log.record(action, None, [timer_grid(count)], level=1)
    flash = timer_grid(61)
    flash[0][5] = 7
    record = log.record('ACTION1', None, [flash, timer_grid(61)], level=1)
    assert not record.hud_mask
    assert record.transient


@pytest.mark.parametrize('state,full_reset', [('GAME_OVER',False), ('WIN',False),
                                             ('NOT_FINISHED',True)])
def test_dead_signature_cleared_on_terminal_or_full_reset(state, full_reset):
    grid = board()
    log = logger(grid)
    for _ in range(2):
        record = log.record('ACTION6', (2, 3), [grid], level=1)
    assert record.signature is not None and log.is_dead_signature(*record.signature)
    log.record('ACTION1', None, [grid], level=1, state=state, full_reset=full_reset)
    assert not log.is_dead_signature(*record.signature)


def test_no_available_nonreset_action_fails_without_dispatch(monkeypatch, tmp_path):
    agent = make_agent(tmp_path, monkeypatch)
    frame = observation(board(), state=GameState.GAME_OVER, actions=())
    agent._ensure_perception(frame)
    agent.perception.record('RESET', None, [board()], level=1)
    with patch('agents.agent.Agent.do_action_request') as dispatch:
        with pytest.raises(RuntimeError, match='replan required'):
            agent.do_action_request(GameAction.RESET)
        dispatch.assert_not_called()
    assert len(agent.perception.records) == 1


def test_choose_action_reference_override_is_not_consumed_twice(monkeypatch, tmp_path):
    monkeypatch.setenv('ENABLE_DEAD_SIGNATURE', '0')
    agent = make_agent(tmp_path, monkeypatch)
    frame = observation(board(), actions=(1,), level=2)
    agent._ensure_perception(frame)
    agent.perception.record('ACTION1', None, [board()], level=2)
    assert agent.perception.blocked_reason('ACTION1') == 'reference_noop'
    proposal = agent.choose_action([], frame)
    with patch('agents.agent.Agent.do_action_request', return_value=frame) as dispatch:
        agent.do_action_request(proposal)
        dispatch.assert_called_once()
    assert len(agent.perception.records) == 2


def test_dispatch_error_is_visible_and_never_fabricates_transition(monkeypatch, tmp_path):
    agent = make_agent(tmp_path, monkeypatch)
    frame = observation(board(), actions=(1,))
    agent._ensure_perception(frame)
    with (patch('agents.agent.Agent.do_action_request', side_effect=RuntimeError('fixture-error')),
          pytest.raises(RuntimeError, match='fixture-error')):
        agent.do_action_request(GameAction.ACTION1)
    assert not agent.perception.records
    payload = json.loads(agent.perception.log_path.read_text().splitlines()[-1])
    assert payload['type'] == 'dispatch_error' and payload['execution_status'] == 'unknown'


def test_exhausted_dead_signatures_replan_once_without_reviving_targets(monkeypatch, tmp_path):
    agent = make_agent(tmp_path, monkeypatch)
    frame = observation(board(), actions=(6,))
    agent._ensure_perception(frame)
    for xy in [(0, 0), (2, 3)]:
        for _ in range(2):
            agent.perception.record('ACTION6', xy, [board()], level=1)
    proposal = GameAction.ACTION6
    proposal.set_data({'x':2,'y':3})
    with patch('agents.agent.Agent.do_action_request', return_value=frame) as dispatch:
        agent.do_action_request(proposal)
        assert dispatch.call_args.args[0] is GameAction.RESET
        agent.do_action_request(GameAction.RESET)
        assert dispatch.call_args.args[0] is GameAction.ACTION6
    lines = [json.loads(line) for line in agent.perception.log_path.read_text().splitlines()]
    assert any(line.get('reason') == 'admissible_targets_exhausted' for line in lines)
    assert [t.action for t in agent.perception.records[-2:]] == ['RESET', 'ACTION6']
