#!/bin/bash
set -e

IMAGE_NAME="devanshdvj/act-artifact:amd64"
CONTAINER_NAME="act-artifact-dev"

cd "$(dirname "$0")/../"
REPO_DIR="$(pwd)"

# Ensure image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Image $IMAGE_NAME not found. Run ./scripts/setup.sh first."
    exit 1
fi

# Start persistent container if not running
if ! docker ps --format '{{.Names}}' | grep -xq "$CONTAINER_NAME"; then
    if docker ps -a --format '{{.Names}}' | grep -xq "$CONTAINER_NAME"; then
        docker start "$CONTAINER_NAME" >/dev/null
    else
        docker run -d --name "$CONTAINER_NAME" \
            -v "$REPO_DIR:/act" \
            -w /act \
            "$IMAGE_NAME" \
            sleep infinity >/dev/null
    fi
    echo "Container $CONTAINER_NAME started."
fi

UID_N="$(id -u)"
GID_N="$(id -g)"

fix_permissions() {
    docker exec "$CONTAINER_NAME" chown -R "$UID_N:$GID_N" /act 2>/dev/null || true
}
trap fix_permissions EXIT

# Use -it only if stdin is a terminal
TTY_FLAG=""
if [ -t 0 ]; then
    TTY_FLAG="-it"
fi

# Run command inside container
if [ $# -eq 0 ]; then
    docker exec $TTY_FLAG "$CONTAINER_NAME" bash
else
    docker exec $TTY_FLAG "$CONTAINER_NAME" bash -c "$*"
fi
