"""S2 failure boundaries and immutable wire histories, using synthetic inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import socket
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from arcengine import GameAction

from agent.arclog import PerceptionLog
from agent.local_backend import LocalBackend
from agent.my_agent import MyAgent, Playbook, WorldModel
from agent.runtime import PrefixLock
from scripts import llm_daemon
from scripts.build_notebook import build
from scripts.extract_llm_launcher import extract_launcher
from scripts.llm_assets import SGLANG_WHEEL, verify_offline_assets
from scripts.notebook_runtime import NotebookRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    monkeypatch.setenv('ENABLE_OFFLINE_BACKEND', '1')
    monkeypatch.delenv('ARC3_LLM_READY', raising=False)
    launcher = tmp_path / 'launcher.py'
    launcher.write_text('raise SystemExit(1)\n')
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(llm_daemon.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(llm_daemon.time, 'sleep', lambda seconds: setattr(clock, 'now', clock.now + seconds))
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock(return_value=False)
    probe.connect_ex.return_value = 111
    monkeypatch.setattr(socket, 'socket', Mock(return_value=probe))
    process = Mock(pid=123456, returncode=None)
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr(llm_daemon.subprocess, 'Popen', spawn)
    kills = Mock()
    monkeypatch.setattr(llm_daemon.os, 'killpg', kills)
    event = Mock()
    supervisor = llm_daemon.LLMDaemon(launcher, tmp_path, event)
    return SimpleNamespace(supervisor=supervisor, process=process, clock=clock,
                           spawn=spawn, kills=kills, event=event, probe=probe)


def test_readiness_precedes_backend_enable_and_cleanup_owns_all_children(daemon):
    supervisor = daemon.supervisor
    supervisor.models_ready = Mock(side_effect=lambda timeout: daemon.clock.now >= 619)
    assert supervisor.start()
    assert daemon.clock.now == 619
    assert os.environ['ARC3_LLM_READY'] == '1'
    assert daemon.spawn.call_args.kwargs['start_new_session'] is True
    assert daemon.spawn.call_args.kwargs['env']['HF_HUB_OFFLINE'] == '1'
    # An exited launcher can leave descendants: still terminate its process group.
    daemon.process.poll.return_value = 0
    supervisor.stop()
    assert [call.args for call in daemon.kills.call_args_list] == [
        (123456, signal.SIGTERM), (123456, signal.SIGKILL)]
    assert os.environ['ARC3_LLM_READY'] == '0'
    status = json.loads((supervisor.working / 'diagnostics/llm_status.json').read_text())
    assert status['ever_ready'] and not status['ready']


@pytest.mark.parametrize('failure', ['deadline', 'exit', 'occupied', 'disabled'])
def test_startup_failure_is_bounded_logged_exploration(daemon, monkeypatch, failure):
    supervisor = daemon.supervisor
    supervisor.models_ready = Mock(return_value=False)
    if failure == 'exit':
        daemon.process.poll.return_value = 1
        daemon.process.returncode = 1
    elif failure == 'occupied':
        daemon.probe.connect_ex.return_value = 0
    elif failure == 'disabled':
        monkeypatch.setenv('ENABLE_OFFLINE_BACKEND', '0')
    assert not supervisor.start()
    assert daemon.clock.now <= 900
    if failure == 'deadline':
        assert daemon.clock.now == 900
        assert 'startup_deadline_900s' in supervisor.reason
    assert os.environ['ARC3_LLM_READY'] == '0'
    status = json.loads((supervisor.working / 'diagnostics/llm_status.json').read_text())
    assert status['mode'] == 'exploration' and not status['ever_ready']
    if failure in ('occupied', 'disabled'):
        daemon.spawn.assert_not_called()
        daemon.kills.assert_not_called()
    else:
        assert daemon.kills.call_count == 2
    if failure != 'disabled':
        assert daemon.event.call_args.args[0] == 'llm_fallback'


@pytest.mark.parametrize('body,status,ready', [
    ({'data': [{'id': 'flashnext'}]}, 200, True),
    ({'data': [{'id': 'other'}]}, 200, False),
    ({'data': [{'id': 'flashnext'}]}, 503, False),
])
def test_health_checks_served_identity_on_loopback(tmp_path, monkeypatch, body, status, ready):
    response = Mock(status=status)
    response.read.return_value = json.dumps(body).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    factory = Mock(return_value=connection)
    monkeypatch.setattr(llm_daemon.http.client, 'HTTPConnection', factory)
    supervisor = llm_daemon.LLMDaemon(tmp_path / 'launcher', tmp_path, Mock())
    assert supervisor.models_ready(0.25) is ready
    factory.assert_called_once_with('127.0.0.1', 8001, timeout=0.25)
    connection.request.assert_called_once_with('GET', '/v1/models')
    connection.close.assert_called_once()


def test_no_unbounded_startup_configuration(tmp_path):
    for timeout in (0, -1, 901):
        with pytest.raises(ValueError):
            llm_daemon.LLMDaemon(tmp_path / 'launcher', tmp_path, Mock(), startup_seconds=timeout)


def test_notebook_interrupt_during_startup_stops_owned_children(daemon):
    daemon.supervisor.models_ready = Mock(side_effect=KeyboardInterrupt)
    with pytest.raises(KeyboardInterrupt):
        daemon.supervisor.start()
    assert daemon.supervisor.process is None and daemon.supervisor.log is None
    assert daemon.kills.call_args.args == (123456, signal.SIGKILL)


def fake_endpoint(monkeypatch, proposals, *, usage=None, status=200):
    wire = []
    responses = iter(proposals)

    class Connection:
        def __init__(self, host, port, timeout):
            assert (host, port) == ('127.0.0.1', 8001)
            assert 0 < timeout <= 30

        def request(self, method, path, body, headers):
            assert (method, path) == ('POST', '/v1/chat/completions')
            wire.append(json.loads(body))

        def getresponse(self):
            data = {'choices': [{'message': {'content': json.dumps(next(responses))}}]}
            if usage is not None:
                data['usage'] = usage
            return SimpleNamespace(status=status, read=lambda size: json.dumps(data).encode())

        def close(self):
            pass

    monkeypatch.setattr('agent.local_backend.http.client.HTTPConnection', Connection)
    return wire


def proposal(value=2):
    return {'actions': [{'action': 'ACTION1', 'expect': [{'x': 1, 'y': 1, 'value': value}],
                         'expect_level_up': False}]}


def test_wire_histories_append_only_until_explicit_playbook_reset(tmp_path, monkeypatch):
    wire = fake_endpoint(monkeypatch, [proposal()] * 8)
    memory = Playbook(tmp_path, PrefixLock())
    backend = LocalBackend(tmp_path, lambda: (256, 2.0), lambda: str(memory.context))
    context = memory.context
    previous = []
    for turn in range(8):
        memory.append('user', {'observation': turn})
        request = memory.request()
        frozen = copy.deepcopy(request)
        response = backend(request)
        assert request == frozen  # Adapter must not rewrite caller-owned objects.
        current = wire[-1]['messages']
        assert current[:len(previous)] == previous
        assert memory.context == context
        previous = current
        memory.append('assistant', response)
    # Neither accumulated facts nor external playbook changes mutate a live prefix.
    memory.remember('synthetic', status='unresolved')
    (tmp_path / 'playbook.md').write_text('external replacement')
    assert memory.request()[:len(previous)] == previous
    memory.distill('level_boundary')
    assert memory.context != context
    assert 'synthetic' in memory.request()[0]['content']
    with pytest.raises(ValueError, match='prefix'):
        memory.messages[0]['content'] = 'mid-context mutation'
        memory.request()
    records = [json.loads(line) for line in (tmp_path / 'llm_requests.jsonl').read_text().splitlines()]
    assert all(record['cache_hit_rate'] is None for record in records)


def test_token_ceiling_keeps_latest_request_for_inference(tmp_path):
    memory = Playbook(tmp_path, PrefixLock(), token_ceiling=2500)
    context = memory.context
    latest = {'state': {'grid': [[3] * 64] * 64}, 'request': 'hypotheses_or_actions'}
    memory.append('user', latest)
    assert memory.context > context
    assert json.loads(memory.request()[-1]['content']) == latest


@pytest.mark.parametrize('usage,rate', [
    ({'prompt_tokens': 1000, 'completion_tokens': 20,
      'prompt_tokens_details': {'cached_tokens': 970}}, 0.97),
    ({'prompt_tokens': 1000}, None),
    ({'prompt_tokens': 1000, 'prompt_tokens_details': {'cached_tokens': 2000}}, None),
])
def test_cache_metrics_use_reported_tokens_only(tmp_path, monkeypatch, usage, rate):
    fake_endpoint(monkeypatch, [proposal()], usage=usage)
    LocalBackend(tmp_path, lambda: (256, 2.0), lambda: 'one')([])
    record = json.loads((tmp_path / 'llm_requests.jsonl').read_text())
    assert record['cache_hit_rate'] == rate


@pytest.mark.parametrize('response,status,error', [
    ({}, 200, ValueError),
    (proposal(), 500, RuntimeError),
])
def test_model_failures_propagate_with_diagnostics(tmp_path, monkeypatch, response, status, error):
    fake_endpoint(monkeypatch, [response], status=status)
    with pytest.raises(error):
        LocalBackend(tmp_path, lambda: (256, 2.0), lambda: 'one')([])
    assert json.loads((tmp_path / 'llm_requests.jsonl').read_text())['status'] == 'error'


def test_reasoning_budget_exhaustion_makes_no_request(tmp_path, monkeypatch):
    wire = fake_endpoint(monkeypatch, [])
    with pytest.raises(TimeoutError):
        LocalBackend(tmp_path, lambda: (0, 0.0), lambda: 'one')([])
    assert not wire


def test_model_response_still_requires_forward_predictions(tmp_path, monkeypatch):
    invalid = proposal()
    invalid['actions'][0]['expect'] = []
    fake_endpoint(monkeypatch, [invalid])
    perception = PerceptionLog(tmp_path / 'log.txt')
    perception.start([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], level=1)
    world = WorldModel(perception, Mock(), tmp_path, PrefixLock())
    world.backend = LocalBackend(tmp_path, lambda: (256, 2.0), lambda: str(world.memory.context))
    with pytest.raises(ValueError, match='non-empty'):
        world.reason([GameAction.ACTION1])
    assert not world.queue


def test_model_plan_reaches_existing_queue_and_mismatch_flush(tmp_path, monkeypatch):
    monkeypatch.setenv('ENABLE_AUDITABLE_WORLD_MODEL', '0')
    wire = fake_endpoint(monkeypatch, [proposal()])
    perception = PerceptionLog(tmp_path / 'log.txt')
    perception.start([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], level=1)
    world = WorldModel(perception, Mock(), tmp_path, PrefixLock())
    world.backend = LocalBackend(tmp_path, lambda: (256, 2.0), lambda: str(world.memory.context))
    world.reason([GameAction.ACTION1])
    assert json.loads(wire[0]['messages'][-1]['content'])['state']['grid'][1][1] == 1
    assert world.next_action([GameAction.ACTION1])['expect'][0]['value'] == 2
    world.observe(perception.record('ACTION1', None, [[[0, 0, 0], [0, 9, 0], [0, 0, 0]]], level=1))
    assert world.pending is None and not world.queue
    assert any(event['type'] == 'prediction' and not event['matched'] for event in world.events)


@pytest.mark.parametrize('enabled,ready,connected', [('1', '1', True), ('0', '1', False), ('1', '0', False)])
def test_backend_connection_obeys_ablation_and_readiness(tmp_path, monkeypatch, enabled, ready, connected):
    monkeypatch.setenv('ARC3_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('ENABLE_OFFLINE_BACKEND', enabled)
    monkeypatch.setenv('ARC3_LLM_READY', ready)
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr('agents.agent.Agent.__init__', lambda self: setattr(self, 'game_id', 'synthetic'))
        agent = MyAgent()
    assert isinstance(agent.world.backend, LocalBackend) is connected
    assert agent.world.events[-1]['type'] == ('backend_connected' if connected else 'backend_unavailable')


def test_notebook_waits_for_daemon_before_gateway_and_always_cleans_up(tmp_path):
    notebook = build()
    cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert cells.index('runner.start_llm(startup_seconds=900)\n') < cells.index('runner.run()\n')
    assert notebook['metadata']['arc3']['llm_bundle']['startup_seconds'] == 900
    run = NotebookRunner('fixture', working=tmp_path)
    run.llm = Mock()
    run._run = Mock(side_effect=RuntimeError('gateway failed'))
    with pytest.raises(RuntimeError, match='gateway failed'):
        run.run()
    run.llm.stop.assert_called_once()


def test_launcher_reuses_reviewed_cells_and_supported_flags():
    source, manifest = extract_launcher()
    compile(source, '<fixture-launcher>', 'exec')
    assert manifest['baseline_notebook_sha256'] == hashlib.sha256(
        (ROOT / 'research_lab/top_programs/baseline_28_score.ipynb').read_bytes()).hexdigest()
    assert manifest['launcher_sha256'] == hashlib.sha256(source.encode()).hexdigest()
    for flag in ('--quantization', 'auto-round', '--speculative-algorithm', 'NEXTN',
                 '--enable-cache-report', '--enable-metrics', '--schedule-policy', 'lpm',
                 '--no-index', '--only-binary'):
        assert flag in source
    assert '--disable-radix-cache' not in source
    assert 'precache_model_thread' not in source
    assert 'start_new_session=True' not in source  # Parent owns the whole process group.
    assert 'pickle' not in source
    assert 'SPEC=os.getenv("ENABLE_SPECULATIVE_DRAFT", "1") == "1"' in source
    metadata = json.loads((ROOT / 'notebooks/kernel-metadata.json').read_text())
    assert metadata['dataset_sources'] == ['dfranzen/pennyroyal-v253']
    assert metadata['model_sources'] == [
        'dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1',
        'dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1']
    assert metadata['enable_internet'] is False


@pytest.fixture
def assets(tmp_path):
    wheels = tmp_path / 'bundle/wheels'
    wheels.mkdir(parents=True)
    for name in (SGLANG_WHEEL, 'uv-0.1-py3-none-linux_x86_64.whl'):
        with zipfile.ZipFile(wheels / name, 'w') as archive:
            archive.writestr('fixture', 'synthetic wheel only')
    (wheels.parent / 'requirements.lock').write_text('sglang==0.5.19\n')
    target = tmp_path / 'target'
    draft = tmp_path / 'draft'
    for path in (target, draft):
        path.mkdir()
        (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'one.safetensors'}}))
        (path / 'one.safetensors').write_bytes(b'synthetic weights')
    (target / 'config.json').write_text(json.dumps({'quantization_config': {'quant_method': 'auto-round', 'bits': 4}}))
    (draft / 'config.json').write_text(json.dumps({'quantization_config': {
        'quant_method': 'compressed-tensors', 'config_groups': {'mtp_routed_experts': {}}}}))
    return wheels.parent, target, draft


def test_assets_validate_required_formats_and_record_observed_hashes(assets):
    report = verify_offline_assets(*assets)
    assert report['wheel_sha256_observed'][SGLANG_WHEEL]
    assert report['target']['shard_sizes'] == {'one.safetensors': 17}
    assert 'draft' in report
    assert 'draft' not in verify_offline_assets(assets[0], assets[1], None)


@pytest.mark.parametrize('failure', ['wheel', 'lock', 'shard', 'traversal', 'quantization', 'draft'])
def test_bad_assets_fail_before_install(assets, failure):
    wheelhouse, target, draft = assets
    if failure == 'wheel':
        (wheelhouse / 'wheels' / SGLANG_WHEEL).write_bytes(b'not a wheel')
    elif failure == 'lock':
        (wheelhouse / 'requirements.lock').unlink()
    elif failure == 'shard':
        (target / 'one.safetensors').unlink()
    elif failure == 'traversal':
        (target / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': '../outside'}}))
    elif failure == 'quantization':
        (target / 'config.json').write_text('{}')
    else:
        (draft / 'config.json').write_text('{}')
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_offline_assets(*assets)


def test_real_failed_launcher_is_reaped_without_blocking_exploration(tmp_path, monkeypatch):
    # Real child lifecycle; mock only the port probe so no network is required.
    monkeypatch.setenv('ENABLE_OFFLINE_BACKEND', '1')
    probe = Mock()
    probe.__enter__ = Mock(return_value=probe)
    probe.__exit__ = Mock(return_value=False)
    probe.connect_ex.return_value = 111
    monkeypatch.setattr(socket, 'socket', Mock(return_value=probe))
    launcher = tmp_path / 'fail.py'
    launcher.write_text('raise RuntimeError("synthetic missing wheel")\n')
    supervisor = llm_daemon.LLMDaemon(launcher, tmp_path, Mock(), startup_seconds=2)
    supervisor.models_ready = Mock(side_effect=ConnectionRefusedError)
    assert not supervisor.start()
    assert supervisor.process is None and supervisor.log is None
    assert 'launcher_exited' in supervisor.reason
    assert 'synthetic missing wheel' in (tmp_path / 'diagnostics/llm_launcher.log').read_text()


def test_cleanup_escalates_for_unresponsive_owned_process(daemon):
    daemon.supervisor.models_ready = Mock(return_value=True)
    assert daemon.supervisor.start()
    daemon.process.wait.side_effect = [subprocess.TimeoutExpired('launcher', 2), 0]
    daemon.supervisor.stop()
    assert daemon.kills.call_args.args == (123456, signal.SIGKILL)
