"""Submission lifecycle fixtures; no live gateway or game engine access."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.notebook_runtime import NotebookRunner


class NotebookWorkflowTests(unittest.TestCase):
    def make_runner(self, root: Path) -> NotebookRunner:
        runner = NotebookRunner('fixture', working=root, bundle=root / 'bundle')
        runner.verbose = False
        self.addCleanup(lambda: [handler.close() for handler in runner.logger.handlers])
        runner.start()
        return runner

    def test_save_exposes_empty_output_without_running_gateway(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            runner = self.make_runner(Path(directory))
            with patch.object(runner, 'prepare_framework') as gateway:
                runner.run()
                gateway.assert_not_called()
            self.assertEqual(pq.ParquetFile(runner.submission).metadata.num_rows, 0)
            self.assertEqual(json.loads((runner.diagnostics / 'status.json').read_text())['status'],
                             'NOT_EVALUATED')
            runner.submission_wait_seconds = 0
            with self.assertRaisesRegex(ValueError, 'no rows'):
                runner.validate_submission()

    def test_stale_output_fails_and_fresh_gateway_result_passes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            pq.write_table(pa.table({'row_id': ['fixture_0'], 'game_id': ['fixture'],
                                     'end_of_game': [True], 'score': [0.5]}), root / 'submission.parquet')
            runner = self.make_runner(root)
            runner.rerun = True
            runner.submission_wait_seconds = 0
            with self.assertRaisesRegex(RuntimeError, 'unchanged'):
                runner.validate_submission()
            temporary = root / 'fresh.parquet'
            pq.write_table(pa.table({'row_id': ['fixture_1'], 'game_id': ['fixture'],
                                     'end_of_game': [True], 'score': [0.75]}), temporary)
            temporary.replace(runner.submission)
            runner.validate_submission()
            self.assertEqual(runner.state['status'], 'COMPLETE')

    def test_bootstrap_can_be_disabled_and_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            runner = self.make_runner(Path(directory))
            with patch.dict(os.environ, ENABLE_SUBMISSION_BOOTSTRAP='0'):
                runner.run()
            self.assertFalse(runner.submission.exists())
            runner.submission.write_bytes(b'existing evidence')
            with self.assertRaises(FileExistsError):
                runner.run()
            self.assertEqual(runner.submission.read_bytes(), b'existing evidence')


if __name__ == '__main__':
    unittest.main()
