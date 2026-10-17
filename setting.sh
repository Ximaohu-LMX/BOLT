#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == "--help" ]]; then
    echo 'BOLT_PYTHON=python3.9 bash setting.sh [--check]'
    echo 'Creates .venv using Python 3.9–3.11; the paper used Python 3.9.25.'
    exit 0
fi
if [[ "${1:-}" == "--check" ]]; then
    source scripts/common.sh
    exec "$BOLT_PY" -m repro.doctor
fi
BOLT_BOOTSTRAP_PYTHON="${BOLT_PYTHON:-python3}"
"$BOLT_BOOTSTRAP_PYTHON" -c 'import sys; assert (3,9) <= sys.version_info[:2] <= (3,11), "Use Python 3.9–3.11; set BOLT_PYTHON=python3.9"'
if [[ ! -x .venv/bin/python ]]; then
    "$BOLT_BOOTSTRAP_PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m repro.doctor
