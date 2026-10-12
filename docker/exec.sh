#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Pick the cuda13 image and pass the GPU on a GPU host
VARIANT=""
GPU_FLAGS=()
if nvidia-smi >/dev/null 2>&1 &&
  docker info -f '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  VARIANT="-cuda13"
  GPU_FLAGS=(--gpus all)
fi
# --cpu or --cuda13 overrides the host's pick
case "${1:-}" in
  --cpu)
    VARIANT=""
    shift
    ;;
  --cuda13)
    VARIANT="-cuda13"
    shift
    ;;
esac
# The GPU is passed to the cuda13 image alone
[ -n "${VARIANT}" ] || GPU_FLAGS=()

ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
  IMAGE_NAME="devanshdvj/act:v1.2${VARIANT}-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  IMAGE_NAME="devanshdvj/act:v1.2${VARIANT}-arm64"
else
  echo "Error: Unsupported architecture: $ARCH"
  exit 1
fi

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
