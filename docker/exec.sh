#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# A leading --cpu or --cuda13 overrides the host's pick
source ./image.sh "${1:-}"
case "${1:-}" in
  --cpu | --cuda13) shift ;;
esac

HOST_MOUNT="$(pwd)/.."
# One name per call, so the trap removes this run's container alone
CONTAINER_NAME="act-exec-$$-${RANDOM}"

# The container writes as root: remove it and restore the caller's ownership at exit
restore_ownership() {
  local status=$?
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  ./chown.sh >/dev/null 2>&1 || echo "warning: could not restore the ownership of the files this run wrote" >&2
  exit "$status"
}
trap restore_ownership EXIT
# A signal ends the run at once, with the signal's status
trap 'exit 130' INT
trap 'exit 143' TERM

# Made first and in the foreground: a signal waits for the create to return
docker create --rm --name "${CONTAINER_NAME}" --entrypoint bash \
  ${GPU_FLAGS[@]+"${GPU_FLAGS[@]}"} \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -ilc "$*" >/dev/null
# In the background, so a signal reaches its trap while the container runs
docker start -a "${CONTAINER_NAME}" &
wait $!
