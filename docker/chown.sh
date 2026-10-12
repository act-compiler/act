#!/bin/bash
set -euo pipefail

# Run from docker/ directory
cd "$(dirname "$0")"

CONTAINER_MOUNT="/workspace"
# One name per call, so concurrent calls never collide
CHOWN_CONTAINER_NAME="act-chown-$$-${RANDOM}"

UID_N="$(id -u)"
GID_N="$(id -g)"

# Use busybox (tiny image) to perform chown
docker run --rm --name "${CHOWN_CONTAINER_NAME}" \
  -v "$(pwd)/..:${CONTAINER_MOUNT}:rw" \
  busybox:1.36 \
  sh -c "chown -R ${UID_N}:${GID_N} ${CONTAINER_MOUNT} || (echo \"Failed to change ownership.\" && exit 1)"

echo "Ownership changed successfully."
