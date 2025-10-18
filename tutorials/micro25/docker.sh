#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"

# Detect architecture
ARCH="$(uname -m)"
MODE=""
PLATFORM_FLAG=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --sim)
            MODE="sim"
            ;;
        --compile)
            MODE="compile"
            ;;
        --setup)
            echo "Setup mode selected. Pulling necessary Docker images..."
            if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
                docker pull devanshdvj/act-tutorials:micro25-arm64
                docker pull devanshdvj/act-tutorials:micro25-amd64
            else
                docker pull devanshdvj/act-tutorials:micro25-amd64
            fi
            echo "Setup complete."
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# Handle architecture-specific logic
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    # arm64 requires a flag
    if [[ -z "$MODE" ]]; then
        echo "Error: On arm64, you must specify either --sim or --compile"
        exit 1
    fi

    if [[ "$MODE" == "sim" ]]; then
        IMAGE_NAME="devanshdvj/act-tutorials:micro25-arm64"
    else
        IMAGE_NAME="devanshdvj/act-tutorials:micro25-amd64"
        PLATFORM_FLAG="--platform linux/amd64"
    fi
else
    # amd64 - use default image
    IMAGE_NAME="devanshdvj/act-tutorials:micro25-amd64"
fi

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
    ${PLATFORM_FLAG} \
    -v "${HOST_MOUNT}:${CONTAINER_MOUNT}:rw" \
    -w "${CONTAINER_MOUNT}" \
    -e HOST_UID="$(id -u)" \
    -e HOST_GID="$(id -g)" \
    "${IMAGE_NAME}"
