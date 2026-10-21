"""Notebook diagnostics and gateway-only evaluation; embedded by the builder.

Only the competition gateway writes evaluated submission rows. Save runs emit
an empty schema-only output so Kaggle can select the file for a code rerun.
All runtime imports outside submission validation use the Python standard library.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent


class NotebookRunner:
    """Keep notebook phases, subprocess output and failures visible on disk."""

    def __init__(self, build_id: str, *, working: Path = Path('/kaggle/working'),
                 bundle: Path = Path('/tmp/agent_bundle'),
                 competition: Path = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-3'),
                 preflight: Path = Path('/tmp/agent_preflight'),
                 audit_script: Path = Path('/tmp/submission_audit.py')) -> None:
        self.build_id = build_id
        self.working, self.bundle = working, bundle
        self.competition, self.preflight = competition, preflight
        self.audit_script = audit_script
        self.framework = working / 'ARC-AGI-3-Agents'
        self.diagnostics = working / 'diagnostics'
        self.submission = working / 'submission.parquet'
        self.started = time.time()
        self.target_seconds = 29_700
        self.hard_limit_seconds = 30_600
        self.submission_wait_seconds = 30.0
        self.rerun = False
        self.verbose = os.getenv('ENABLE_NOTEBOOK_DIAGNOSTICS', '1') == '1'
        self.initial_submission: tuple[int, int, int] | None = None
        self.llm = None
        self.state: dict = {}
        self.logger = logging.getLogger(f'arc3.notebook.{id(self)}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def elapsed(self) -> float:
        return time.time() - self.started

    def event(self, name: str, **details) -> None:
        record = {'event': name, 'elapsed_seconds': round(self.elapsed(), 3),
                  'build_id': self.build_id, **details}
        line = json.dumps(record, sort_keys=True)
        self.logger.info('%s', line)
        with (self.diagnostics / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    def status(self, state: str, **details) -> None:
        self.state.update(status=state, elapsed_seconds=round(self.elapsed(), 3),
                          build_id=self.build_id, competition_rerun=self.rerun, **details)
        temporary = self.diagnostics / 'status.json.tmp'
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(self.diagnostics / 'status.json')

    @contextmanager
    def stage(self, name: str):
        began = time.time()
        self.status('RUNNING', stage=name)
        self.event('stage_started', stage=name)
        try:
            yield
        except Exception as error:
            detail = traceback.format_exc()
            with (self.diagnostics / 'traceback.log').open('a', encoding='utf-8') as stream:
                stream.write(f'\nStage: {name}\n' + detail)
            self.logger.exception('Stage %s failed', name)
            self.status('FAILED', stage=name, error_type=type(error).__name__, error=str(error),
                        submission_exists=self.submission.is_file())
            self.event('stage_failed', stage=name, error_type=type(error).__name__, error=str(error))
            raise
        else:
            self.event('stage_completed', stage=name, duration_seconds=round(time.time() - began, 3))

    def _signature(self) -> tuple[int, int, int] | None:
        if not self.submission.exists():
            return None
        stat = self.submission.stat()
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    def start(self) -> None:
        self.started = time.time()
        self.diagnostics.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        for handler in (logging.StreamHandler(sys.stdout),
                        logging.FileHandler(self.diagnostics / 'notebook.log', encoding='utf-8')):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        with self.stage('startup'):
            raw_flag = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '')
            flag = raw_flag.strip().lower()
            if flag not in ('', '0', 'false', 'no', '1', 'true', 'yes'):
                raise ValueError('Unrecognized KAGGLE_IS_COMPETITION_RERUN value')
            self.rerun = flag in ('1', 'true', 'yes')
            os.environ['ARC3_RUN_STARTED'] = str(self.started)
            os.environ['ARC3_LOG_DIR'] = str(self.working / 'scratch')
            for name in ('ENABLE_SANDBOX_LIMITS', 'ENABLE_PREFIX_LOCK', 'ENABLE_LEVEL_BUDGET',
                         'ENABLE_ACTION_FALLBACK', 'ENABLE_NOTEBOOK_DIAGNOSTICS'):
                os.environ.setdefault(name, '1')
            (self.bundle / 'baseline').mkdir(parents=True, exist_ok=True)
            self.initial_submission = self._signature()
            self.event('execution_mode', mode='COMPETITION_RERUN' if self.rerun else 'SAVE_OR_INTERACTIVE',
                       rerun_flag=raw_flag, kernel_run_type=os.getenv('KAGGLE_KERNEL_RUN_TYPE', ''),
                       python=sys.version.split()[0], existing_submission=self.initial_submission is not None,
                       submission_policy='gateway-only', diagnostics=str(self.diagnostics))
            if self.verbose:
                gpu = shutil.which('nvidia-smi')
                if gpu:
                    try:
                        result = subprocess.run(
                            [gpu, '--query-gpu=name,memory.total', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=5, check=False)
                        self.event('gpu_probe', returncode=result.returncode,
                                   output=(result.stdout + result.stderr)[-2000:])
                    except (OSError, subprocess.TimeoutExpired) as error:
                        self.event('gpu_probe_unavailable', error=str(error))
                else:
                    self.event('gpu_probe_unavailable', reason='nvidia-smi not installed')

    def command(self, name: str, args: list[str], *, timeout: float, cwd: Path | None = None) -> None:
        remaining = self.target_seconds - self.elapsed()
        if remaining <= 0:
            raise TimeoutError(f'Global run target exhausted before {name}')
        path = self.diagnostics / f'{name}.log'
        self.event('command_started', command=name, args=args, log=str(path))
        try:
            with path.open('w', encoding='utf-8') as stream:
                result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=min(timeout, remaining), check=True)
            self.event('command_completed', command=name, returncode=result.returncode)
        finally:
            # The complete stream stays on disk, including when a command fails.
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 16_384))
                    tail = stream.read().decode(errors='replace')
                if tail:
                    self.logger.info('%s output (tail):\n%s', name, tail)

    def install_dependencies(self) -> None:
        with self.stage('offline_dependencies'):
            wheelhouse = self.competition / 'arc_agi_3_wheels'
            if not wheelhouse.is_dir():
                raise FileNotFoundError(f'Competition wheelhouse is missing: {wheelhouse}')
            self.command('pip', [sys.executable, '-m', 'pip', 'install', '--no-index',
                                '--find-links', str(wheelhouse), 'arc-agi', 'python-dotenv'], timeout=600)
            self.event('dependency_versions', versions={name: importlib.metadata.version(name)
                       for name in ('arc-agi', 'arcengine', 'python-dotenv')})

    def verify_sources(self, expected: dict[str, str]) -> None:
        with self.stage('bundle_verification'):
            for name, digest in expected.items():
                path = self.bundle / name
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Bundled module hash mismatch: {name}')
                compile(path.read_text(), str(path), 'exec')
            self.event('bundle_verified', modules=len(expected))

    def prepare_framework(self) -> None:
        with self.stage('gateway_readiness'):
            self.command('gateway', ['curl', '--fail', '--show-error', '--silent',
                         '--connect-timeout', '5', '--max-time', '15', '--retry', '999',
                         '--retry-all-errors', '--retry-delay', '5', '--retry-max-time', '600',
                         'http://gateway:8001/api/games'], timeout=620)
        self.prepare_offline_framework()

    def prepare_offline_framework(self) -> None:
        """Exercise the deployed setup without contacting or starting a gateway."""
        with self.stage('framework_setup'):
            source = self.competition / 'ARC-AGI-3-Agents'
            if not (source / 'main.py').is_file():
                raise FileNotFoundError(f'Competition framework is missing: {source}')
            shutil.copytree(source, self.framework, dirs_exist_ok=True)
            shutil.copytree(self.bundle, self.framework / 'agents/templates', dirs_exist_ok=True)
            sandbox = self.working / '.venv_agent_sandbox'
            # The worker needs only stdlib. Avoid ensurepip and bundled pip
            # availability as an unnecessary dependency of isolated startup.
            self.command('sandbox_setup', [sys.executable, '-m', 'venv', '--without-pip',
                                          str(sandbox)], timeout=120)
            os.environ['ARC3_SANDBOX_PYTHON'] = str(sandbox / 'bin/python')
            self.preflight.mkdir(parents=True, exist_ok=True)
            for name, target in [('agent', self.bundle), ('.venv_agent_sandbox', sandbox)]:
                link = self.preflight / name
                if link.is_symlink():
                    if link.resolve() != target.resolve():
                        raise ValueError(f'Preflight symlink points at the wrong directory: {link}')
                else:
                    link.symlink_to(target, target_is_directory=True)
        with self.stage('containment_audit'):
            report_path = self.preflight / 'containment.json'
            # Do not mistake an earlier audit report for this invocation's result.
            report_path.unlink(missing_ok=True)
            try:
                self.command('containment', [sys.executable, str(self.audit_script)], timeout=90)
            finally:
                if report_path.is_file():
                    shutil.copy2(report_path, self.diagnostics / 'containment.json')
            report = json.loads(report_path.read_text())
            if report.get('containment_passed') is not True:
                raise RuntimeError('Containment audit did not pass; see diagnostics/containment.json')
        with self.stage('agent_registration'):
            registry = ('from typing import Type\nfrom dotenv import load_dotenv\n'
                        'from .agent import Agent, Playback\nfrom .swarm import Swarm\n'
                        'from .templates.random_agent import Random\n'
                        'from .templates.my_agent import MyAgent\n\nload_dotenv()\n\n'
                        "AVAILABLE_AGENTS: dict[str, Type[Agent]] = {'random': Random, 'myagent': MyAgent}\n")
            (self.framework / 'agents/__init__.py').write_text(registry)
            (self.framework / '.env').write_text(
                'SCHEME=http\nHOST=gateway\nPORT=8001\nARC_API_KEY=test-key-123\n'
                'ARC_BASE_URL=http://gateway:8001/\nOPERATION_MODE=competition\nENVIRONMENTS_DIR=\n'
                f'RECORDINGS_DIR={self.working / "server_recording"}\n')
            self.event('agent_registered', agent='myagent', action_log=os.environ['ARC3_LOG_DIR'],
                       operation_mode='competition',
                       backend='local flashnext' if os.getenv('ARC3_LLM_READY') == '1' else 'exploration')

    def preflight_framework(self) -> None:
        """Validate Kaggle's actual interpreter, containment and framework imports."""
        self.prepare_offline_framework()
        with self.stage('framework_imports'):
            self.command('framework_imports', [sys.executable, '-c',
                         dedent('''\
                         import os
                         os.environ['ENABLE_OFFLINE_BACKEND'] = '0'
                         from pathlib import Path
                         import numpy as np
                         from agents import AVAILABLE_AGENTS
                         from agents.agent import Agent
                         from arcengine import FrameDataRaw, GameAction, GameState
                         agent_class = AVAILABLE_AGENTS['myagent']
                         assert issubclass(agent_class, Agent)
                         print('FRAMEWORK_IMPORTS_OK', flush=True)
                         # Synthetic public API data only; never instantiate an
                         # Arcade or read environment source during this check.
                         class Fixture:
                             def __init__(self):
                                 self.observation_space = FrameDataRaw(
                                     game_id='runtime-fixture', frame=[np.zeros((4, 4), dtype=np.int8)],
                                     state=GameState.NOT_PLAYED, win_levels=1, available_actions=[1])
                             def step(self, action, data=None, reasoning=None):
                                 assert action is GameAction.RESET
                                 self.observation_space.state = GameState.WIN
                                 self.observation_space.levels_completed = 1
                                 return self.observation_space
                         os.environ['ARC3_LOG_DIR'] = str(Path.cwd().parent / 'diagnostics/synthetic-agent')
                         agent = agent_class(card_id='fixture', game_id='runtime-fixture',
                                             agent_name='fixture', ROOT_URL='', record=False,
                                             arc_env=Fixture())
                         agent.main()
                         assert agent.action_counter == 1 and not agent._recovery_mode
                         print('FRAMEWORK_RUNTIME_OK synthetic_actions=1 gateway_actions=0', flush=True)
                         ''')],
                         timeout=120, cwd=self.framework)
        self.event('save_preflight_passed', gateway_contacted=False, gameplay_actions=0)

    def start_llm(self, *, startup_seconds: float = 2400.0) -> None:
        with self.stage('llm_startup'):
            spec = importlib.util.spec_from_file_location('arc3_llm_daemon', '/tmp/arc3_llm_daemon.py')
            if spec is None or spec.loader is None:
                raise RuntimeError('Bundled LLM supervisor is unavailable')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            remaining = self.target_seconds - self.elapsed()
            if remaining <= 0:
                raise TimeoutError('Run budget exhausted before LLM startup')
            self.llm = module.LLMDaemon(Path('/tmp/arc3_llm_launcher.py'), self.working,
                                        self.event, startup_seconds=min(startup_seconds, remaining))
            self.llm.start()

    def stop_llm(self) -> None:
        if self.llm is not None:
            self.llm.stop()

    def monitor(self, process: subprocess.Popen, output) -> None:
        last_heartbeat = -60.0
        try:
            while process.poll() is None:
                elapsed = self.elapsed()
                if elapsed >= self.hard_limit_seconds or elapsed >= self.target_seconds:
                    self.event('runtime_limit', elapsed=elapsed, action='request_scorecard_finalization')
                    break
                if self.verbose:
                    chunk = output.read(65_536)
                    if chunk:
                        print(chunk, end='', flush=True)
                    if elapsed - last_heartbeat >= 60:
                        self.event('framework_heartbeat', pid=process.pid,
                                   submission_exists=self.submission.is_file())
                        last_heartbeat = elapsed
                try:
                    process.wait(timeout=min(1.0, self.target_seconds - elapsed))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                if process.poll() is None:
                    self.event('cleanup_started', pid=process.pid, signal='SIGINT')
                    try:
                        process.send_signal(signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    remaining = max(0.0, self.hard_limit_seconds - self.elapsed())
                    process.wait(timeout=min(60.0, remaining))
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if self.verbose:
                    print(output.read(), end='', flush=True)
                self.event('framework_exited', pid=process.pid, returncode=process.returncode)
        if process.returncode:
            raise RuntimeError(f'Framework exited with code {process.returncode}; see diagnostics/framework.log')

    def run_framework(self) -> None:
        with self.stage('framework_execution'):
            if self.elapsed() >= self.target_seconds:
                raise TimeoutError('Global run target exhausted during setup')
            path = self.diagnostics / 'framework.log'
            with path.open('w', encoding='utf-8') as stream, path.open(errors='replace') as output:
                process = subprocess.Popen(
                    [sys.executable, '-u', 'main.py', '--agent', 'myagent'], cwd=self.framework,
                    env=dict(os.environ, MPLBACKEND='agg', PYTHONUNBUFFERED='1',
                             OPERATION_MODE='competition'),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.event('framework_started', pid=process.pid, log=str(path))
                self.monitor(process, output)

    def validate_submission(self) -> None:
        with self.stage('submission_validation'):
            # Give the sidecar a bounded interval to finish writing after close.
            deadline = min(self.started + self.hard_limit_seconds,
                           time.time() + self.submission_wait_seconds)
            while self._signature() == self.initial_submission and time.time() < deadline:
                time.sleep(min(0.5, max(0.0, deadline - time.time())))
            if not self.submission.is_file():
                raise FileNotFoundError('Gateway produced no submission.parquet; inspect diagnostics/framework.log')
            if self._signature() == self.initial_submission:
                raise RuntimeError('submission.parquet is unchanged from startup; no fresh gateway result')
            import pyarrow.parquet as pq
            while True:
                try:
                    parquet = pq.ParquetFile(self.submission)
                    break
                except (OSError, ValueError):
                    # The sidecar may have created the file before its footer
                    # was flushed. Retry only inside the shared hard deadline.
                    if time.time() >= deadline:
                        raise
                    self.event('submission_write_pending')
                    time.sleep(min(0.5, max(0.0, deadline - time.time())))
            required = {'row_id', 'game_id', 'end_of_game', 'score'}
            if not required.issubset(parquet.schema_arrow.names) or parquet.metadata.num_rows < 1:
                raise ValueError('Gateway submission has missing columns or no rows')
            if parquet.metadata.num_rows == 1:
                row = parquet.read(columns=sorted(required)).to_pylist()[0]
                if row['row_id'] == '1_0' and row['game_id'] == '1' and row['end_of_game']:
                    raise ValueError('submission.parquet contains the old dummy record; not a gateway result')
            self.event('submission_validated', rows=parquet.metadata.num_rows,
                       bytes=self.submission.stat().st_size, columns=parquet.schema_arrow.names,
                       sha256=hashlib.sha256(self.submission.read_bytes()).hexdigest())
        self.status('COMPLETE', stage='submission_validation', submission_exists=True)

    def run(self) -> None:
        try:
            self._run()
        finally:
            self.stop_llm()

    def _run(self) -> None:
        if not self.rerun:
            if os.getenv('ENABLE_SUBMISSION_BOOTSTRAP', '1') == '1':
                # Create no scores or gameplay records. Kaggle requires a saved
                # output with this name before accepting a code submission.
                # Evaluation must replace it with a fresh, nonempty gateway file.
                import pyarrow as pa
                import pyarrow.parquet as pq
                schema = pa.schema([('row_id', pa.string()), ('game_id', pa.string()),
                                    ('end_of_game', pa.bool_()), ('score', pa.float64())])
                if self.submission.exists():
                    raise FileExistsError('Refusing to overwrite an existing submission during a save run')
                pq.write_table(pa.Table.from_batches([], schema=schema), self.submission)
                self.event('submission_bootstrap', rows=0, evaluated=False,
                           path=str(self.submission), purpose='Kaggle code submission output selector')
            self.status('NOT_EVALUATED', stage='competition_dispatch',
                        reason='Competition rerun flag is not enabled; gameplay gateway is unavailable')
            self.event('not_evaluated', message='Save & Run All does not play hidden games. '
                       'No gameplay rows were generated. The competition gateway runs during Submit to Competition.')
            return
        self.prepare_framework()
        self.run_framework()
        self.validate_submission()
