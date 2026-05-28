#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    IMAGE_NAME="devanshdvj/act:v1.1-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    IMAGE_NAME="devanshdvj/act:v1.1-arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

HOST_MOUNT="$(pwd)/.."

docker run --rm --entrypoint bash \
  -v "${HOST_MOUNT}:/workspace:rw" \
  -w /workspace \
  "${IMAGE_NAME}" \
  -ilc "$*"
