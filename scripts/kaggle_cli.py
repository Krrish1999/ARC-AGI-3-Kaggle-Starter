"""Run the Kaggle CLI with project-local config and injected token support."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def push_kernel(api, folder: str) -> None:
    """The upstream CLI prints push errors but exits zero; fail closed here."""
    result = api.kernels_push(folder, timeout=None, acc=None)
    if result is None or result.error:
        raise SystemExit('Kaggle push failed: ' + (result.error if result else 'no response'))
    rejected = {name: getattr(result, name) for name in (
        'invalid_dataset_sources', 'invalid_competition_sources', 'invalid_kernel_sources')
        if getattr(result, name)}
    if rejected:
        raise SystemExit(f'Kaggle push rejected sources: {rejected}')
    if result.version_number <= 0 or not result.url:
        raise SystemExit('Kaggle push did not confirm a saved version')
    print(f'Kernel version {result.version_number} successfully pushed. '
          f'Please check progress at {result.url}', flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = root / '.kaggle'
    config.mkdir(exist_ok=True)
    os.environ.setdefault('KAGGLE_CONFIG_DIR', str(config))
    if not os.getenv('KAGGLE_API_TOKEN'):
        token_file = config / 'access_token'
        token = os.getenv('KAGGLE_KEY')
        if not token and token_file.is_file():
            token = token_file.read_text().strip()
        if token:
            os.environ['KAGGLE_API_TOKEN'] = token
    if sys.argv[1:4] == ['kernels', 'push', '-p'] and len(sys.argv) == 5:
        from kaggle import api
        push_kernel(api, sys.argv[4])
        return
    executable = root / '.venv/bin/kaggle'
    os.execv(str(executable), [str(executable), *sys.argv[1:]])


if __name__ == '__main__':
    main()
