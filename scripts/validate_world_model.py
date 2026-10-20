"""Reproducible S1-W toy evidence and optional paired public compatibility runs.

Use --public for seeds {7,19}, four named public targets, cap 40, all-on versus
each independent world-model ablation. These runs have no inference backend;
dormant mechanisms are reported as such, never as causal performance evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/ARC-AGI-3-Agents')]

from arcengine import GameAction

from agent.arclog import PerceptionLog
from agent.my_agent import MyAgent, WorldModel
from agent.runtime import PrefixLock
from agent.sandbox import run_sandbox

FLAGS = ('ENABLE_PLAYBOOK_RESET', 'ENABLE_RETRODICTION', 'ENABLE_EXPECT_QUEUE',
         'ENABLE_TIER1_SIMULATOR')
GOOD = """def step(state, action):
    state['grid'][1][1] = (state['grid'][1][1] + 1) % 16
    if state['grid'][1][1] == 5: state['level'] += 1
    return state
"""


def board(value):
    return [[0, 0, 0], [0, value, 0], [0, 0, 0]]


def synthetic():
    directory = ROOT / 'work/s1w-synthetic'
    directory.mkdir(parents=True, exist_ok=True)
    log = PerceptionLog()
    log.start([board(0)], level=3)
    for value in (1, 2):
        log.record('ACTION1', None, [board(value)], level=3)
    def worker(source):
        return run_sandbox(source, python=ROOT / '.venv_agent_sandbox/bin/python', cwd=ROOT)
    model = WorldModel(log, worker, directory, PrefixLock())
    reports = [model.replay_transitions(source) for source in (
        GOOD, 'def step(state, action): return state',
        GOOD.replace('return state', "state['grid'][2][2] = 9\n    return state"),
        GOOD.replace('return state', "state['level'] += 1\n    return state"))]
    before = len(log.records)
    plan = [{'action': 'ACTION1', 'expect': [{'x': 1, 'y': 1, 'value': value}],
             'expect_level_up': False} for value in (3, 4)]
    model.execute_plan(plan, [GameAction.ACTION1])
    for value in (3, 4):
        assert model.next_action([GameAction.ACTION1]) is not None
        model.observe(log.record('ACTION1', None, [board(value)], level=3))
    search = model.search_plan(GOOD, [GameAction.ACTION1])
    model.flush('validation_boundary')
    bad_compile = model.search_plan('def step(:', [GameAction.ACTION1])
    bad_replay = model.search_plan('def step(s,a): return s', [GameAction.ACTION1])
    predictions = [e for e in model.events if e['type'] == 'prediction']
    assert [r['status'] for r in reports] == ['pass', 'counterexample', 'counterexample', 'counterexample']
    assert search['status'] == 'found'
    return {'kind': 'synthetic toy; not game performance', 'replay_reports': reports,
            'rejected_by_zero_action_replay': 3, 'rejected_hypotheses': 3,
            'replay_live_actions': 0, 'prediction_covered': len(predictions),
            'nonprobing_transitions': len(log.records)-before,
            'prediction_matched': sum(e['matched'] for e in predictions),
            'prediction_checked': len(predictions),
            'simulator_compile_success': 2, 'simulator_compile_attempts': 3,
            'simulator_replay_success': 1, 'simulator_replay_attempts': 2,
            'search_reports': [search, bad_compile, bad_replay]}


def public_runs():
    import arc_agi
    from arc_agi import OperationMode
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    available = {e.game_id.split('-')[0]: e.game_id for e in arcade.get_environments()}
    target_games = ('ls20', 'ft09', 'vc33', 're86')
    results = []
    MyAgent.MAX_ACTIONS = 40
    for disabled in (None, *FLAGS):
        for game in target_games:
            for seed in (7, 19):
                if game not in available:
                    results.append({'game': game, 'seed': seed, 'disabled': disabled, 'status': 'missing'})
                    continue
                flags = {name: '0' if name == disabled else '1' for name in FLAGS}
                with patch.dict(os.environ, {**flags, 'ARC3_SEED': str(seed),
                                            'ARC3_LOG_DIR': str(ROOT / 'work/s1w-public')}):
                    environment = arcade.make(game, seed=seed)
                    if environment is None:
                        raise RuntimeError('available environment failed to initialize: ' + game)
                    agent = MyAgent(card_id='s1w-validation', game_id=game,
                                    agent_name='s1w-validation', ROOT_URL='', record=False,
                                    arc_env=environment, tags=['s1w-validation'])
                    agent.main()
                    records = agent.perception.records
                    relevant = [t for t in records if not t.reset and not t.terminal and not t.level_changed]
                    predictions = [e for e in agent.world.events if e['type'] == 'prediction']
                    log_path = agent.perception.log_path
                    assert log_path is not None
                    results.append({'game': game, 'version': available[game], 'seed': seed,
                        'disabled': disabled, 'status': 'pass', 'flags': flags,
                        'live_responses': len(records), 'framework_iterations': agent.action_counter,
                        'levels_completed': agent.frames[-1].levels_completed,
                        'zero_causal': sum(t.zero_causal_diff for t in relevant),
                        'nonboundary_responses': len(relevant), 'resets': sum(t.reset for t in records),
                        'level3plus_responses': sum(t.before.level >= 3 for t in relevant),
                        'predictions': len(predictions), 'prediction_matches': sum(e['matched'] for e in predictions),
                        'hypotheses_rejected': sum(e['type'] == 'hypothesis_rejected' for e in agent.world.events),
                        'simulator_compiles': sum(e['type'] == 'simulator_compile' for e in agent.world.events),
                        'playbook_versions': agent.world.memory.version,
                        'log_path': str(log_path.relative_to(ROOT)),
                        'log_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest()})
                    print(f'{game} seed={seed} disabled={disabled}: pass', flush=True)
    comparisons = {}
    baseline = {(r['game'], r['seed']): r for r in results if r['disabled'] is None and r['status'] == 'pass'}
    for flag in FLAGS:
        differences = []
        for run in results:
            key = (run['game'], run['seed'])
            if run['disabled'] != flag or run['status'] != 'pass' or key not in baseline:
                continue
            control = baseline[key]
            differences.append(run['zero_causal']/run['nonboundary_responses']
                               - control['zero_causal']/control['nonboundary_responses'])
        if differences:
            generator = random.Random(917)
            means = sorted(sum(generator.choices(differences, k=len(differences)))/len(differences)
                           for _ in range(5000))
            comparisons[flag] = {'pairs': len(differences), 'mean_zero_rate_reduction': sum(differences)/len(differences),
                                 'paired_bootstrap_95_ci': [means[125], means[4874]],
                                 'bootstrap_resamples': 5000, 'bootstrap_seed': 917,
                                 'interpretation': 'Dormant policy gates; degenerate intervals do not establish equivalence or causal lift.'}
    return {'prespecified_seeds': [7, 19], 'cap_argument': 40, 'paired_comparisons': comparisons,
            'backend': None, 'runs': results,
            'limits': 'No inference backend; prediction/rejection/simulator rates N/A when denominator is zero. '
                      'Dormant ablations and short runs cannot establish score lift or promotion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--out', default='research_lab/traces/s1_w_validation.json')
    args = parser.parse_args()
    if not json.loads((ROOT / 'containment.json').read_text())['containment_passed']:
        raise SystemExit('Run successful containment audit before candidate validation')
    logging.basicConfig(level=logging.ERROR)
    with patch.dict(os.environ, dict.fromkeys(FLAGS, '1')):
        report = {'synthetic': synthetic(), 'source_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('agent/my_agent.py', 'agent/arclog.py')},
            'versions': {name: importlib.metadata.version(name)
                         for name in ('arc-agi', 'arcengine', 'pytest', 'ruff', 'pyright')}}
        if args.public:
            report['public'] = public_runs()
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
