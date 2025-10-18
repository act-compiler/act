#!/bin/bash
set -euo pipefail

# Run from docker/ directory
cd "$(dirname "$0")"

CONTAINER_MOUNT="/workspace"
CHOWN_CONTAINER_NAME="act-chown"

UID_N="$(id -u)"
GID_N="$(id -g)"

# Remove any previous leftover act-chown container
if docker ps -a --format '{{.Names}}' | grep -xq "${CHOWN_CONTAINER_NAME}"; then
    echo "Removing existing leftover container named ${CHOWN_CONTAINER_NAME}..."
    docker rm -f ${CHOWN_CONTAINER_NAME} >/dev/null 2>&1
fi

# Use busybox (tiny image) to perform chown
docker run --rm --name "${CHOWN_CONTAINER_NAME}" \
  -v "$(pwd)/..:${CONTAINER_MOUNT}:rw" \
  busybox:1.36 \
  sh -c "chown -R ${UID_N}:${GID_N} ${CONTAINER_MOUNT} || (echo \"Failed to change ownership.\" && exit 1)"

echo "Ownership changed successfully."
