#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"

# Detect architecture
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    IMAGE_NAME="devanshdvj/act:latest-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    IMAGE_NAME="devanshdvj/act:latest-arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

# collect candidate containers (all containers using the image)
mapfile -t CONTAINERS < <(docker ps -a --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}' | awk -v img="${IMAGE_NAME}" '$3 == img')

NUM_CONTAINERS=${#CONTAINERS[@]}
if [ ${NUM_CONTAINERS} -eq 0 ]; then
  echo "No containers using image '$IMAGE_NAME' found."
  exit 0
fi

echo "Found ${NUM_CONTAINERS} container(s) using image '${IMAGE_NAME}':"
for line in "${CONTAINERS[@]}"; do
  echo "$line"
done
echo
for line in "${CONTAINERS[@]}"; do
  CID=$(echo "$line" | cut -d' ' -f1)
  CNAME=$(echo "$line" | cut -d' ' -f2)
  CINFO=$(echo "$line" | cut -d' ' -f3-)
  read -rp "Remove container ${CNAME} (${CID})? [y/N] " RESP
  case "${RESP}" in
    y|Y)
      echo "Removing ${CNAME}..."
      docker rm -f "${CID}" || { echo "Failed to remove ${CNAME}"; }
      ;;
    *)
      echo "Keeping ${CNAME}."
      ;;
  esac
done

echo
echo "Done."
