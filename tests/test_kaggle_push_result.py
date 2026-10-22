"""Submission acceptance must come from the SDK response, not a CLI footer."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.kaggle_cli import push_kernel


@pytest.mark.parametrize('failure', ['quota', 'empty', 'dataset', 'version'])
def test_rejected_push_never_reports_success(failure, capsys):
    result = SimpleNamespace(error='', invalid_dataset_sources=[], invalid_competition_sources=[],
                             invalid_kernel_sources=[], version_number=13, url='https://www.kaggle.com/code/fixture')
    if failure == 'quota':
        result.error = 'Maximum batch GPU session count of 2 reached'
    elif failure == 'empty':
        result = None
    elif failure == 'dataset':
        result.invalid_dataset_sources = ['required/weights']
    else:
        result.version_number = 0
    with pytest.raises(SystemExit):
        push_kernel(Mock(kernels_push=Mock(return_value=result)), 'notebooks/')
    assert 'successfully pushed' not in capsys.readouterr().out


def test_accepted_push_prints_confirmed_saved_version(capsys):
    result = SimpleNamespace(error='', invalid_dataset_sources=[], invalid_competition_sources=[],
                             invalid_kernel_sources=[], version_number=13, url='https://www.kaggle.com/code/fixture')
    api = Mock(kernels_push=Mock(return_value=result))
    push_kernel(api, 'notebooks/')
    api.kernels_push.assert_called_once_with('notebooks/', timeout=None, acc=None)
    assert 'Kernel version 13 successfully pushed' in capsys.readouterr().out
