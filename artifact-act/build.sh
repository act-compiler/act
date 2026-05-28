#!/bin/bash

IMAGE_NAME="devanshdvj/act-artifact:amd64"

cd $(dirname "$0")

echo "Building ACT artifact image $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .
