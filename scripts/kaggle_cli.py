"""Run the Kaggle CLI with project-local config and injected token support."""
from __future__ import annotations

import os
import sys
from pathlib import Path


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
    executable = root / '.venv/bin/kaggle'
    os.execv(str(executable), [str(executable), *sys.argv[1:]])


if __name__ == '__main__':
    main()
