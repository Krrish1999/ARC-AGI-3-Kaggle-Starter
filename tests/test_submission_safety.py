"""Gateway-only output and diagnostic fixtures; no hidden games or real gateway."""
from __future__ import annotations

import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from arcengine import FrameData, GameAction, GameState

from agent.my_agent import MyAgent
from scripts.build_notebook import build
from scripts.notebook_runtime import NotebookRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', 'True')
    monkeypatch.setenv('ENABLE_NOTEBOOK_DIAGNOSTICS', '1')
    # Do not let hardware discovery influence synthetic fixtures.
    monkeypatch.setattr('scripts.notebook_runtime.shutil.which', lambda name: None)
    run = NotebookRunner('fixture-build', working=tmp_path, bundle=tmp_path / 'bundle',
                         competition=tmp_path / 'input', preflight=tmp_path / 'preflight',
                         audit_script=tmp_path / 'audit.py')
    run.submission_wait_seconds = 0
    run.start()
    yield run
    for handler in list(run.logger.handlers):
        handler.close()
        run.logger.removeHandler(handler)


def status(run):
    return json.loads((run.diagnostics / 'status.json').read_text())


def events(run):
    return [json.loads(line) for line in (run.diagnostics / 'events.jsonl').read_text().splitlines()]


def write_result(path):
    pd.DataFrame([{'row_id': 'synthetic_0', 'game_id': 'synthetic',
                   'end_of_game': True, 'score': 0.0}]).to_parquet(path, index=False)


@pytest.mark.parametrize('flag', ['', '0', 'False', 'false', 'no'])
def test_ordinary_run_records_not_evaluated_with_empty_selector(runner, monkeypatch, flag):
    runner.rerun = False
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', flag)
    with patch.object(runner, 'prepare_framework') as prepare:
        runner.run()
    prepare.assert_not_called()
    assert status(runner)['status'] == 'NOT_EVALUATED'
    assert pd.read_parquet(runner.submission).empty
    bootstrap = next(e for e in events(runner) if e['event'] == 'submission_bootstrap')
    assert bootstrap['rows'] == 0 and bootstrap['evaluated'] is False
    assert events(runner)[-1]['event'] == 'not_evaluated'


@pytest.mark.parametrize('flag,expected', [('', False), ('0', False), ('False', False),
                                         ('True', True), ('1', True), ('yes', True)])
def test_mode_parsing_is_explicit(tmp_path, monkeypatch, flag, expected):
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', flag)
    monkeypatch.setenv('ENABLE_NOTEBOOK_DIAGNOSTICS', '0')
    run = NotebookRunner('mode', working=tmp_path, bundle=tmp_path / 'bundle')
    try:
        run.start()
        assert run.rerun is expected
        assert not run.submission.exists()
        assert events(run)[1]['mode'] == ('COMPETITION_RERUN' if expected else 'SAVE_OR_INTERACTIVE')
    finally:
        for handler in list(run.logger.handlers):
            handler.close()
            run.logger.removeHandler(handler)


def test_generated_cells_compile_without_fabricated_score_rows():
    notebook = build()
    assert notebook['metadata']['arc3']['submission_policy'] == 'gateway-only'
    for cell in notebook['cells']:
        source = cell['source']
        assert 'to_parquet(' not in source
        assert 'placeholder.to_' not in source
        if cell['cell_type'] == 'code' and not source.startswith('%%writefile '):
            compile(source, '<generated-cell>', 'exec')
    first = next(c['source'] for c in notebook['cells'] if c['cell_type'] == 'code')
    assert 'NotebookRunner' in first and 'runner.start()' in first


def test_first_cell_diagnostics_survive_sigkill(tmp_path):
    source = next(c['source'] for c in build()['cells'] if c['cell_type'] == 'code')
    source = source.replace('/kaggle/working', str(tmp_path)).replace(
        '/tmp/agent_bundle', str(tmp_path / 'bundle'))
    result = subprocess.run([sys.executable, '-c', source + '\nos.kill(os.getpid(), signal.SIGKILL)'],
                            env=dict(os.environ, ENABLE_NOTEBOOK_DIAGNOSTICS='0'), timeout=10)
    assert result.returncode == -signal.SIGKILL
    assert (tmp_path / 'diagnostics/notebook.log').is_file()
    assert (tmp_path / 'diagnostics/status.json').is_file()
    assert not (tmp_path / 'submission.parquet').exists()


def test_existing_submission_is_preserved_but_cannot_count_as_fresh(runner):
    write_result(runner.submission)
    before = runner.submission.read_bytes()
    runner.initial_submission = runner._signature()
    with pytest.raises(RuntimeError, match='unchanged'):
        runner.validate_submission()
    assert runner.submission.read_bytes() == before
    assert status(runner)['status'] == 'FAILED'


@pytest.mark.parametrize('kind', ['missing', 'dummy', 'corrupt', 'empty', 'wrong_schema'])
def test_missing_or_invalid_result_fails_loudly_with_traceback(runner, kind):
    if kind == 'dummy':
        pd.DataFrame([{'row_id': '1_0', 'game_id': '1', 'end_of_game': True,
                       'score': 1}]).to_parquet(runner.submission)
    elif kind == 'corrupt':
        runner.submission.write_bytes(b'not parquet')
    elif kind == 'empty':
        pd.DataFrame(columns=['row_id', 'game_id', 'end_of_game', 'score']).to_parquet(runner.submission)
    elif kind == 'wrong_schema':
        pd.DataFrame({'other': [1]}).to_parquet(runner.submission)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        runner.validate_submission()
    assert status(runner)['status'] == 'FAILED'
    assert status(runner)['stage'] == 'submission_validation'
    assert 'Traceback' in (runner.diagnostics / 'traceback.log').read_text()
    assert any(e['event'] == 'stage_failed' for e in events(runner))


def test_real_result_is_retained_and_summarized(runner):
    write_result(runner.submission)
    before = runner.submission.read_bytes()
    runner.validate_submission()
    assert status(runner)['status'] == 'COMPLETE'
    assert runner.submission.read_bytes() == before
    summary = next(e for e in events(runner) if e['event'] == 'submission_validated')
    assert summary['rows'] == 1 and summary['bytes'] == len(before)
    assert summary['sha256'] == hashlib.sha256(before).hexdigest()


@pytest.mark.parametrize('failure', ['gateway_readiness', 'framework_setup', 'containment_audit',
                                     'framework_execution'])
def test_failed_stage_is_visible_and_exception_is_not_swallowed(runner, failure):
    with pytest.raises(RuntimeError, match='fixture failure'):
        with runner.stage(failure):
            raise RuntimeError('fixture failure')
    report = status(runner)
    assert report['status'] == 'FAILED' and report['stage'] == failure
    assert 'fixture failure' in (runner.diagnostics / 'notebook.log').read_text()
    assert failure in (runner.diagnostics / 'traceback.log').read_text()
    assert not runner.submission.exists()


def test_real_command_failure_captures_stderr_and_return_code(runner):
    with pytest.raises(subprocess.CalledProcessError):
        with runner.stage('offline_dependencies'):
            runner.command('fixture', [sys.executable, '-c',
                           'import sys; print("actual child failure", file=sys.stderr); sys.exit(7)'], timeout=3)
    assert 'actual child failure' in (runner.diagnostics / 'fixture.log').read_text()
    assert status(runner)['status'] == 'FAILED'
    assert 'returned non-zero exit status 7' in (runner.diagnostics / 'traceback.log').read_text()


def test_real_command_timeout_leaves_diagnostics(runner):
    with pytest.raises(subprocess.TimeoutExpired):
        with runner.stage('gateway_readiness'):
            runner.command('fixture', [sys.executable, '-u', '-c',
                           'import time; print("waiting for gateway"); time.sleep(5)'], timeout=0.3)
    assert status(runner)['status'] == 'FAILED'
    assert 'waiting for gateway' in (runner.diagnostics / 'fixture.log').read_text()


def test_hash_mismatch_is_logged_before_dispatch(runner):
    (runner.bundle / 'example.py').write_text('print(1)')
    with pytest.raises(ValueError, match='hash mismatch'):
        runner.verify_sources({'example.py': 'wrong-hash'})
    assert status(runner)['stage'] == 'bundle_verification'


def test_diagnostic_toggle_keeps_failures_and_deadlines(runner):
    runner.verbose = False
    assert runner.target_seconds == 29_700 and runner.hard_limit_seconds == 30_600
    with pytest.raises(RuntimeError):
        with runner.stage('fixture'):
            raise RuntimeError('always visible')
    assert status(runner)['status'] == 'FAILED'
    assert 'always visible' in (runner.diagnostics / 'traceback.log').read_text()


@pytest.mark.parametrize('mode', ['normal', 'crash', 'timeout', 'hung_cleanup', 'hard_expiry'])
def test_shared_deadline_and_cleanup_paths(runner, monkeypatch, mode):
    clock = [29_699.0]
    recorded = []
    monkeypatch.setattr(time, 'time', lambda: clock[0])
    runner.started = 0

    class Process:
        pid = 123456789
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            recorded.append(('signal', sig))
            if mode != 'hung_cleanup':
                self.returncode = 0
                recorded.append(('finalized', clock[0]))

        def wait(self, timeout=None):
            recorded.append(('wait', timeout))
            if self.returncode is not None:
                return self.returncode
            if mode in ('normal', 'crash'):
                self.returncode = 0 if mode == 'normal' else 7
                return self.returncode
            clock[0] += timeout
            if mode == 'hard_expiry':
                clock[0] = 30_601.0
            raise subprocess.TimeoutExpired('fixture', timeout)

    process = Process()

    def killpg(pid, sig):
        recorded.append(('kill', sig))
        if process.returncode is None:
            process.returncode = -sig

    with patch.object(os, 'killpg', side_effect=killpg):
        if mode in ('crash', 'hung_cleanup'):
            with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
                with runner.stage('framework_execution'):
                    runner.monitor(process, io.StringIO('fixture framework output'))
            assert status(runner)['status'] == 'FAILED'
        else:
            runner.monitor(process, io.StringIO('fixture framework output'))
    assert ('kill', signal.SIGKILL) in recorded
    if mode in ('timeout', 'hung_cleanup', 'hard_expiry'):
        assert recorded.index(('signal', signal.SIGINT)) < recorded.index(('kill', signal.SIGKILL))
        assert ('wait', 0.0 if mode == 'hard_expiry' else 60.0) in recorded


def test_expired_setup_does_not_launch_framework(runner, monkeypatch):
    monkeypatch.setattr(runner, 'elapsed', lambda: 29_701)
    with patch.object(subprocess, 'Popen') as launch, pytest.raises(TimeoutError):
        runner.run_framework()
    launch.assert_not_called()
    assert status(runner)['status'] == 'FAILED'


def test_real_runner_streams_traceback_and_fails(runner):
    runner.framework.mkdir()
    (runner.framework / 'main.py').write_text('raise ValueError("actual framework traceback")')
    with pytest.raises(RuntimeError, match='Framework exited'):
        runner.run_framework()
    assert 'actual framework traceback' in (runner.diagnostics / 'framework.log').read_text()
    assert status(runner)['status'] == 'FAILED'


def test_framework_child_uses_competition_mode_without_changing_kaggle_flag(runner, monkeypatch):
    monkeypatch.setenv('OPERATION_MODE', 'online')
    flag = os.environ['KAGGLE_IS_COMPETITION_RERUN']
    runner.framework.mkdir()
    report = runner.working / 'child-environment.json'
    (runner.framework / 'main.py').write_text(
        'import json, os\nfrom pathlib import Path\n'
        f'Path({str(report)!r}).write_text(json.dumps({{name: os.environ[name] for name in '
        '("OPERATION_MODE", "KAGGLE_IS_COMPETITION_RERUN")}))\n')
    runner.run_framework()
    assert json.loads(report.read_text()) == {'OPERATION_MODE': 'competition',
                                             'KAGGLE_IS_COMPETITION_RERUN': flag}
    assert os.environ['OPERATION_MODE'] == 'online'


def test_real_child_finalizes_on_sigint_and_produces_original_result(runner):
    runner.framework.mkdir()
    child = ('import signal, sys, time\n'
             'def finalize(sig, frame):\n'
             f'    open({str(runner.submission)!r}, "wb").write(bytes.fromhex("PAYLOAD"))\n'
             '    print("scorecard finalized", flush=True)\n'
             '    sys.exit(0)\n'
             'signal.signal(signal.SIGINT, finalize)\n'
             'print("framework ready", flush=True)\n'
             'time.sleep(10)\n')
    # Serialize synthetic gateway output in the fixture, never in notebook code.
    buf = io.BytesIO()
    pd.DataFrame([{'row_id': 'synthetic_0', 'game_id': 'synthetic',
                   'end_of_game': True, 'score': 0}]).to_parquet(buf)
    (runner.framework / 'main.py').write_text(child.replace('PAYLOAD', buf.getvalue().hex()))
    runner.started = time.time()
    runner.target_seconds = 0.5
    runner.hard_limit_seconds = 3
    runner.run_framework()
    runner.validate_submission()
    assert status(runner)['status'] == 'COMPLETE'
    assert 'scorecard finalized' in (runner.diagnostics / 'framework.log').read_text()


@pytest.mark.parametrize('audit_failure', [False, True])
def test_prepare_framework_registers_agent_and_preserves_audit_report(runner, audit_failure):
    source = runner.competition / 'ARC-AGI-3-Agents'
    (source / 'agents').mkdir(parents=True)
    (source / 'main.py').write_text('# synthetic framework')

    def command(name, args, **kwargs):
        if name == 'containment':
            (runner.preflight / 'containment.json').write_text(
                json.dumps({'containment_passed': not audit_failure}))
            if audit_failure:
                raise subprocess.CalledProcessError(1, args)

    with patch.object(runner, 'command', side_effect=command):
        if audit_failure:
            with pytest.raises(subprocess.CalledProcessError):
                runner.prepare_framework()
            assert status(runner)['status'] == 'FAILED'
        else:
            runner.prepare_framework()
            registry = (runner.framework / 'agents/__init__.py').read_text()
            compile(registry, '<registry>', 'exec')
            assert 'HOST=gateway' in (runner.framework / '.env').read_text()
            # Exercise the real SDK using the launcher's generated .env. The
            # gateway distinguishes competition scorecards by this wire field.
            from arc_agi import Arcade, OperationMode
            from dotenv import dotenv_values
            config = dotenv_values(runner.framework / '.env')
            response = SimpleNamespace(raise_for_status=lambda: None,
                                       json=lambda: {'card_id': 'fixture-card'})
            games = SimpleNamespace(raise_for_status=lambda: None, json=lambda: [])
            with patch.dict(os.environ, {k: v for k, v in config.items() if v is not None}), \
                    patch('requests.get', return_value=games), \
                    patch('requests.Session.post', return_value=response) as post:
                arcade = Arcade(environments_dir=str(runner.working / 'empty-environments'),
                                recordings_dir=str(runner.working / 'fixture-recordings'))
                assert arcade.open_scorecard() == 'fixture-card'
                assert post.call_args.kwargs['json'].get('competition_mode') is True
                assert arcade.operation_mode is OperationMode.COMPETITION
    assert json.loads((runner.diagnostics / 'containment.json').read_text())['containment_passed'] is not audit_failure


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



@pytest.mark.parametrize('mode', ['ordinary', 'rerun', 'runner_failure', 'preflight_failure'])
def test_exact_generated_notebook_executes_and_records_correct_outcome(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('KAGGLE_IS_COMPETITION_RERUN', '' if mode == 'ordinary' else 'True')
    monkeypatch.setenv('ENABLE_NOTEBOOK_DIAGNOSTICS', '0')
    notebook = build()
    namespace = {'__name__': 'generated_notebook_fixture'}
    run = None
    buf = io.BytesIO()
    pd.DataFrame([{'row_id': 'synthetic_0', 'game_id': 'synthetic',
                   'end_of_game': True, 'score': 0}]).to_parquet(buf)

    def prepare():
        with run.stage('gateway_readiness'):
            if mode == 'preflight_failure':
                raise ConnectionError('fixture gateway unreachable')
        run.framework.mkdir()
        child = ('raise ValueError("fixture runner failure")' if mode == 'runner_failure' else
                 f'open({str(run.submission)!r}, "wb").write(bytes.fromhex({buf.getvalue().hex()!r}))')
        (run.framework / 'main.py').write_text(child)

    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = cell['source'].replace('/kaggle/working', str(tmp_path)).replace(
                '/tmp/agent_bundle', str(tmp_path / 'bundle')).replace(
                '/tmp/submission_audit.py', str(tmp_path / 'audit.py'))
            if source.startswith('%%writefile '):
                header, body = source.split('\n', 1)
                path = Path(header.removeprefix('%%writefile '))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
                continue
            if source == 'runner.run()\n' and mode.endswith('failure'):
                with pytest.raises((RuntimeError, ConnectionError)):
                    exec(source, namespace)
                break
            exec(source, namespace)
            if run is None:
                run = namespace['runner']
                run.submission_wait_seconds = 0
                monkeypatch.setattr(run, 'install_dependencies', lambda: run.event('fixture_install'))
                monkeypatch.setattr(run, 'prepare_framework', prepare)
        assert status(run)['status'] == ('NOT_EVALUATED' if mode == 'ordinary' else
                                         'FAILED' if mode.endswith('failure') else 'COMPLETE')
        assert run.submission.exists() is (mode in ('ordinary', 'rerun'))
        if mode == 'ordinary':
            assert pd.read_parquet(run.submission).empty
        assert any(e['event'] == 'bundle_verified' for e in events(run))
        if mode.endswith('failure'):
            assert 'Traceback' in (run.diagnostics / 'traceback.log').read_text()
    finally:
        if run is not None:
            for handler in list(run.logger.handlers):
                handler.close()
                run.logger.removeHandler(handler)


def test_sidecar_partial_write_is_retried_within_deadline(runner, monkeypatch):
    runner.submission.write_bytes(b'incomplete footer')
    runner.submission_wait_seconds = 2
    sleeps = []

    def finish_write(seconds):
        sleeps.append(seconds)
        write_result(runner.submission)

    monkeypatch.setattr(time, 'sleep', finish_write)
    runner.validate_submission()
    assert status(runner)['status'] == 'COMPLETE'
    assert len(sleeps) == 1
    assert any(e['event'] == 'submission_write_pending' for e in events(runner))


def test_checked_in_notebook_is_the_current_reviewed_build():
    # A stale ignored notebook caused the real Kaggle upload mismatch. The
    # reviewed artifact must stay synchronized when source or runner changes.
    path = Path(__file__).resolve().parents[1] / 'notebooks/submission.ipynb'
    assert json.loads(path.read_text()) == build()
