#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Quote each argument for the container's shell
ARGS=""
if [ $# -gt 0 ]; then
  ARGS=$(printf '%q ' "$@")
fi

# No .pytest_cache in the tree
./docker/exec.sh "pytest -p no:cacheprovider tests/ ${ARGS}"
