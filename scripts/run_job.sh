#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
exec "$BOLT_PY" -m repro.job "$@"
