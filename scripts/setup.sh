#!/bin/bash

set -e

ACT_IMAGE_NAME="devanshdvj/act-artifact:amd64"
ACT_IMAGE_TARGZ="act-artifact-amd64.tar.gz"
ACT_BUILD_FOLDER="artifact-act"

cd "$(dirname "$0")/../"
REPO_DIR="$(pwd)"

if [[ "$1" == "--build" ]]; then
    echo "Building Docker image locally."
    echo

    $REPO_DIR/$ACT_BUILD_FOLDER/build.sh

    echo
    echo "Cleaning up dangling Docker images..."
    docker image prune -f

    exit 0
fi

if ! docker image inspect "$ACT_IMAGE_NAME" >/dev/null 2>&1; then
    echo
    echo "Searching for $ACT_IMAGE_TARGZ in $REPO_DIR and its parent directory."
    if [[ -f "$REPO_DIR/$ACT_IMAGE_TARGZ" ]]; then
        echo
        echo "Found $REPO_DIR/$ACT_IMAGE_TARGZ. Loading..."
        docker load -i "$REPO_DIR/$ACT_IMAGE_TARGZ"
    elif [[ -f "$REPO_DIR/../$ACT_IMAGE_TARGZ" ]]; then
        echo
        echo "Found $REPO_DIR/../$ACT_IMAGE_TARGZ. Loading..."
        docker load -i "$REPO_DIR/../$ACT_IMAGE_TARGZ"
    else
        echo
        echo "Info: $ACT_IMAGE_TARGZ not found locally in $REPO_DIR or its parent directory."
        echo "Attempting to pull from Docker Hub instead."

        docker pull $ACT_IMAGE_NAME || {
            echo "Error: Failed to pull $ACT_IMAGE_NAME from Docker Hub."
            echo "You can build the image locally using: ./scripts/setup.sh --build"
            exit 1
        }
    fi

    echo
    echo "Cleaning up dangling Docker images..."
    docker image prune -f
fi
