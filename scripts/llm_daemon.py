"""Bounded loopback-only supervisor for the extracted Pennyroyal launcher."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

EPHEMERAL_MODEL_ROOT = Path('/tmp/models/arc3')


class LLMDaemon:
    def __init__(self, launcher: Path, working: Path, event: Callable[..., Any],
                 *, startup_seconds: float = 2400.0) -> None:
        if not 0 < startup_seconds <= 2400:
            raise ValueError('LLM startup must fit within 2400 seconds')
        self.launcher, self.working, self.event = launcher, working, event
        self.startup_seconds = startup_seconds
        self.process: subprocess.Popen | None = None
        self.ready = False
        self.ever_ready = False
        self.reason = 'not_started'
        self.startup_elapsed_seconds: float | None = None
        self.log: Any = None
        self._staging_signature: tuple[int, int] | None = None
        self._staging_owner = uuid.uuid4().hex

    def models_ready(self, timeout: float) -> bool:
        connection = http.client.HTTPConnection('127.0.0.1', 8001, timeout=timeout)
        try:
            connection.request('GET', '/v1/models')
            response = connection.getresponse()
            if response.status != 200:
                return False
            raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                return False
            data = json.loads(raw)
            return any(item.get('id') == 'flashnext' for item in data.get('data', []))
        finally:
            connection.close()

    def record(self) -> None:
        path = self.working / 'diagnostics/llm_status.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'ready': self.ready, 'ever_ready': self.ever_ready,
                                  'reason': self.reason,
                                  'endpoint': 'http://127.0.0.1:8001/v1',
                                  'mode': 'llm' if self.ready else 'exploration',
                                  'startup_seconds': self.startup_seconds,
                                  'startup_elapsed_seconds': self.startup_elapsed_seconds}, indent=2) + '\n')

    def startup_logs(self) -> None:
        """Keep this run's own startup failures visible in its notebook log."""
        for path in (self.working / 'diagnostics/llm_launcher.log', self.working / 'serve.log'):
            if not path.is_file():
                continue
            try:
                with path.open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 16_384))
                    tail = stream.read(16_384).decode(errors='replace')
            except OSError as error:
                self.event('llm_startup_log_unavailable', file=path.name, error=str(error))
            else:
                if tail:
                    self.event('llm_startup_log', file=path.name, tail=tail)

    def staging_report(self) -> None:
        path = self.working / 'diagnostics/model_staging.json'
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature == self._staging_signature:
                return
            report = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            self.event('model_staging_report_error', error=str(error))
        else:
            self._staging_signature = signature
            self.event('model_staging', report=report)

    def start(self) -> bool:
        os.environ['ARC3_LLM_READY'] = '0'
        os.environ.pop('ARC3_LLM_FAILURE', None)
        if os.getenv('ENABLE_OFFLINE_BACKEND', '1') != '1':
            self.reason = 'disabled_by_ablation'
            self.record()
            self.event('llm_disabled', reason=self.reason)
            return False
        began = time.monotonic()
        deadline = began + self.startup_seconds
        next_heartbeat = began + 30.0
        try:
            # An occupied port is not evidence that our model is running. Never
            # adopt or terminate another service, including gateway:8001.
            import socket
            with socket.socket() as probe:
                probe.settimeout(min(1.0, self.startup_seconds))
                if probe.connect_ex(('127.0.0.1', 8001)) == 0:
                    raise RuntimeError('loopback_port_occupied')
            digest = hashlib.sha256(self.launcher.read_bytes()).hexdigest()
            (self.working / 'diagnostics').mkdir(parents=True, exist_ok=True)
            (self.working / 'diagnostics/model_staging.json').unlink(missing_ok=True)
            self.log = (self.working / 'diagnostics/llm_launcher.log').open('ab', buffering=0)
            environment = dict(os.environ, ARC3_LLM_WORKING=str(self.working),
                               ARC3_MODEL_OWNER=self._staging_owner,
                               ARC3_LLM_PREFIX=f'/tmp/arc3-pennyroyal-{digest[:16]}',
                               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UV_OFFLINE='1',
                               UV_PYTHON_DOWNLOADS='never')
            self.process = subprocess.Popen([sys.executable, '-u', str(self.launcher)],
                                            env=environment, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            self.event('llm_started', pid=self.process.pid, launcher_sha256=digest)
            while time.monotonic() < deadline:
                self.staging_report()
                if time.monotonic() >= next_heartbeat:
                    self.event('llm_startup_wait', elapsed_seconds=time.monotonic()-began)
                    next_heartbeat = time.monotonic() + 30.0
                if self.process.poll() is not None:
                    raise RuntimeError(f'launcher_exited:{self.process.returncode}')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    if self.models_ready(min(2.0, remaining)):
                        self.ready, self.ever_ready = True, True
                        self.reason = 'models_endpoint_ready'
                        self.startup_elapsed_seconds = time.monotonic() - began
                        os.environ['ARC3_LLM_READY'] = '1'
                        os.environ['ARC3_LLM_MODEL'] = 'flashnext'
                        self.record()
                        self.event('llm_ready', elapsed_seconds=self.startup_elapsed_seconds)
                        return True
                except (OSError, ValueError, TypeError, AttributeError, http.client.HTTPException):
                    pass
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
            raise TimeoutError(f'startup_deadline_{self.startup_seconds:g}s')
        except (OSError, ValueError, RuntimeError, TypeError, AttributeError,
                subprocess.SubprocessError, http.client.HTTPException) as error:
            self.reason = f'{type(error).__name__}: {error}'
            self.startup_elapsed_seconds = time.monotonic() - began
            self.stop()
            self.record()
            self.staging_report()
            self.startup_logs()
            os.environ['ARC3_LLM_FAILURE'] = self.reason
            self.event('llm_fallback', reason=self.reason, mode='exploration')
            return False
        except BaseException:
            # Notebook interruption must not leave a half-installed launcher
            # or detached CUDA workers consuming resources.
            self.stop()
            raise

    def cleanup_output_staging(self) -> None:
        """Never save model copies as outputs; remove only our stopped run."""
        try:
            report = json.loads((self.working / 'diagnostics/model_staging.json').read_text())
            paths = report.get('orphaned_transactions', [])
            if 'transaction' in report:
                paths = [report['transaction'], *paths]
            roots = {(self.working / '.arc3-models').resolve(), EPHEMERAL_MODEL_ROOT.resolve()}
            for path in paths:
                try:
                    transaction = Path(path)
                    if (transaction.is_symlink() or transaction.parent.resolve() not in roots
                            or not transaction.name.startswith('run-')):
                        continue
                    marker = transaction / '.arc3-owner'
                    if marker.is_symlink() or marker.read_text() != self._staging_owner:
                        continue
                    shutil.rmtree(transaction)
                    self.event('model_staging_cleaned', transaction=str(transaction))
                except FileNotFoundError:
                    continue
                except (OSError, ValueError, TypeError) as error:
                    self.event('model_staging_cleanup_error', error=str(error))
        except (FileNotFoundError, KeyError):
            pass
        except (OSError, ValueError, TypeError) as error:
            self.event('model_staging_cleanup_error', error=str(error))

    def stop(self) -> None:
        self.ready = False
        os.environ['ARC3_LLM_READY'] = '0'
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
                self.event('llm_stopped', returncode=self.process.returncode)
                self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.cleanup_output_staging()
        if self.ever_ready:
            self.reason = 'stopped_after_ready'
            self.record()
