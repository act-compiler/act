#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"

# Detect architecture
ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    IMAGE_NAME="devanshdvj/act:v1.2-arm64"
else
    IMAGE_NAME="devanshdvj/act:v1.2-amd64"
fi

# Parse arguments
for arg in "$@"; do
    case $arg in
        --setup)
            echo "Setup mode: pulling Docker image ${IMAGE_NAME}..."
            docker pull "${IMAGE_NAME}"
            echo "Setup complete."
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

CONTAINER_NAME="act-tutorials-$(whoami)"
HOST_MOUNT="$(pwd)/../.."
CONTAINER_MOUNT="/workspace"

# Check if image exists locally
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found. Pulling..."
    docker pull "${IMAGE_NAME}"
fi

# Launch ephemeral container (removed on exit)
echo "Launching ACT tutorial environment..."
echo "Container: ${CONTAINER_NAME}"
echo "Working directory: ${CONTAINER_MOUNT}"
echo ""

docker run -it --rm \
    --name "${CONTAINER_NAME}" \
    -v "${HOST_MOUNT}:${CONTAINER_MOUNT}:rw" \
    -w "${CONTAINER_MOUNT}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${IMAGE_NAME}"
