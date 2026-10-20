"""Deterministic S1-S acceptance fixtures; no live games or engine internals."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.baseline.bounded_search import bounded_bfs
from agent.runtime import HARD_CAP_SECONDS, TARGET_SECONDS, LevelBudget, PrefixLock
from agent.sandbox import run_sandbox
from scripts.audit_containment import audit
from scripts.build_notebook import build


class SystemsTests(unittest.TestCase):
    def test_missing_sandbox_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'agent').mkdir()
            for name in ('my_agent.py', 'arclog.py'):
                (root / 'agent' / name).touch()
            report = audit(root)
            self.assertFalse(report['containment_passed'])
            self.assertFalse(report['checks']['sandbox_present'])

    def test_engine_import_success_fails_closed(self):
        site = next((ROOT / '.venv_agent_sandbox/lib').glob('python*/site-packages'))
        canary = site / 'arc_agi.py'
        self.assertFalse(canary.exists())
        try:
            canary.write_text('# synthetic engine-import canary\n')
            report = audit(loop_timeout=0.2)
            self.assertFalse(report['containment_passed'])
            self.assertFalse(report['checks']['engine_import_denied'])
        finally:
            canary.unlink(missing_ok=True)

    def test_worker_startup_failure_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'agent').mkdir()
            for name in ('my_agent.py', 'arclog.py'):
                (root / 'agent' / name).touch()
            executable = root / '.venv_agent_sandbox/bin/python'
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/sh\nexit 2\n')
            executable.chmod(0o700)
            report = audit(root, loop_timeout=0.2)
            self.assertFalse(report['containment_passed'])
            self.assertFalse(report['checks']['engine_import_denied'])

    def test_prefix_alias_mutation_and_reset(self):
        lock = PrefixLock()
        messages = [{'role': 'system', 'content': 'fixed'}]
        lock.validate('one', messages)
        messages.append({'role': 'user', 'content': 'next'})
        lock.validate('one', messages)
        messages[0]['content'] = 'mutated'
        with self.assertRaises(ValueError):
            lock.validate('one', messages)
        lock.validate('two', messages)
        with self.assertRaises(ValueError):
            lock.validate('one', messages)
        with patch.dict(os.environ, ENABLE_PREFIX_LOCK='0'):
            lock.validate('two', [{'role': 'system', 'content': 'ablation'}])

    def test_budget_rollover_and_caps(self):
        budget = LevelBudget(env_seconds=600, levels=3, run_started=0, now=0)
        self.assertEqual(budget.begin_level(1, now=0), 100)
        self.assertEqual(budget.begin_level(2, now=20), 300)
        self.assertEqual(budget.begin_level(3, now=100), 600)
        self.assertEqual(budget.generation_tokens(4096, now=590), 256)
        self.assertEqual(budget.generation_tokens(4096, now=600), 0)
        self.assertLess(TARGET_SECONDS, HARD_CAP_SECONDS)
        with patch.dict(os.environ, ENABLE_LEVEL_BUDGET='0'):
            budget = LevelBudget(env_seconds=600, levels=3, run_started=0, now=0)
            self.assertEqual(budget.begin_level(1, now=0), 600)
            self.assertEqual(budget.remaining(now=HARD_CAP_SECONDS), 0)

    def test_limits_off_remains_contained_and_requires_fixture(self):
        python = ROOT / '.venv_agent_sandbox/bin/python'
        with patch.dict(os.environ, ENABLE_SANDBOX_LIMITS='0'):
            with self.assertRaises(ValueError):
                run_sandbox('print(1)', python=python, cwd=ROOT)
            result = run_sandbox("open('/etc/passwd').read()", python=python, cwd=ROOT,
                                 controlled_fixture=True)
            self.assertTrue(result['ready'])
            self.assertFalse(result['ok'])
            self.assertIn('PermissionError', result['stderr'])

    def test_notebook_payload_and_metadata(self):
        notebook = build()
        metadata = notebook['metadata']['kaggle']
        self.assertEqual(metadata['accelerator'], 'nvidiaRtx6000')
        self.assertTrue(metadata['isGpuEnabled'])
        self.assertFalse(metadata['isInternetEnabled'])
        cells = [cell['source'] for cell in notebook['cells']]
        for path in (ROOT / 'agent').rglob('*.py'):
            name = path.relative_to(ROOT / 'agent').as_posix()
            self.assertIn('%%writefile /tmp/agent_bundle/' + name + '\n' + path.read_text(), cells)
        self.assertTrue(any('submission_audit.py' in cell and 'check=True' in cell for cell in cells))

    def test_extracted_modules_import_and_search(self):
        manifest = json.loads((ROOT / 'agent/baseline/manifest.json').read_text())
        for name in manifest['modules']:
            importlib.import_module('agent.baseline.' + Path(name).stem)
        result = bounded_bfs(0, lambda n: [('next', n+1)] if n < 3 else [], lambda n: n == 3)
        self.assertEqual(result['status'], 'found')
        self.assertEqual(result['actions'], ['next'] * 3)


if __name__ == '__main__':
    unittest.main()
