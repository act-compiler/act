#!/bin/bash
set -euo pipefail

# Work in the docker/ directory
cd "$(dirname "$0")"

CONTAINER_MOUNT="/workspace"
# One name per call, so concurrent calls never collide
CHOWN_CONTAINER_NAME="act-chown-$$-${RANDOM}"

UID_N="$(id -u)"
GID_N="$(id -g)"

# Return the repository to the caller in a busybox container
docker run --rm --name "${CHOWN_CONTAINER_NAME}" \
  -v "$(pwd)/..:${CONTAINER_MOUNT}:rw" \
  busybox:1.36 \
  sh -c "chown -R ${UID_N}:${GID_N} ${CONTAINER_MOUNT} || (echo \"Failed to change ownership.\" >&2 && exit 1)"

echo "Ownership changed successfully."
