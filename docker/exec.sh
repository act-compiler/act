#!/bin/bash
set -euo pipefail

# Work in the docker/ directory
cd "$(dirname "$0")"

# A leading --cpu or --cuda13 overrides the host's pick
source ./image.sh "${1:-}"
case "${1:-}" in
  --cpu | --cuda13) shift ;;
esac

HOST_MOUNT="$(pwd)/.."
# One name per call: the trap removes this run's container
CONTAINER_NAME="act-exec-$$-${RANDOM}"

# At exit: remove the container, restore the caller's ownership
restore_ownership() {
  local status=$?
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  ./chown.sh >/dev/null 2>&1 || echo "warning: could not restore the ownership of the files this run wrote" >&2
  exit "$status"
}
trap restore_ownership EXIT
# A signal ends the run at once, with its status
trap 'exit 130' INT
trap 'exit 143' TERM

# Created in the foreground: a signal waits for it
docker create --rm --name "${CONTAINER_NAME}" --entrypoint bash \
  ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -ilc "$*" >/dev/null
# In the background, so a signal reaches the trap
docker start -a "${CONTAINER_NAME}" &
wait $!
