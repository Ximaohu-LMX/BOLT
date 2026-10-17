#!/usr/bin/env bash
set -euo pipefail
BOLT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$BOLT_ROOT"
if [[ -n "${BOLT_PYTHON:-}" ]]; then
    BOLT_PY="$BOLT_PYTHON"
elif [[ -x "$BOLT_ROOT/.venv/bin/python" ]]; then
    BOLT_PY="$BOLT_ROOT/.venv/bin/python"
else
    BOLT_PY=python3
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${BOLT_THREADS:-1}"
export MKL_NUM_THREADS="${BOLT_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${BOLT_THREADS:-1}"

# Historical CLIQUE scripts use presence-based flags; public runners set modes explicitly.
unset MODE EST DEBUG EFFICIENCY_PROFILE EFFICIENCY_TIMING_FILE EFFICIENCY_CACHE_ROOT || true
