#!/bin/bash

set -e

docker stop act-main >/dev/null 2>&1 || true

cd "$(dirname "$0")/../"
HOST_MOUNT="$(pwd)"

UID_N="$(id -u)"
GID_N="$(id -g)"

ACT_IMAGE_NAME="devanshdvj/act-artifact:amd64"

$HOST_MOUNT/scripts/setup.sh

echo "Starting interactive ACT container"

docker run --rm -it \
    -v "$HOST_MOUNT:/act" \
    -w /act \
    $ACT_IMAGE_NAME \
    bash

# Fix ownership
docker run --rm --name act-main \
    -v "$HOST_MOUNT:/act" \
    $ACT_IMAGE_NAME \
    bash -c "chown -R ${UID_N}:${GID_N} /act/*"
