#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"

usage() {
  echo "Usage: $0 [--cpu|--cuda13] [-h|--help]"
}

# Any argument but --cpu, --cuda13 and --help is refused
for arg in "$@"; do
  case "$arg" in
    --cpu | --cuda13) ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# The image of this host, or of --cpu or --cuda13
source ./image.sh "$@"

# Every container of the image
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
  read -rp "Remove container ${CNAME} (${CID})? [y/N] " RESP
  case "${RESP}" in
    y | Y)
      echo "Removing ${CNAME}..."
      docker rm -f "${CID}" || {
        echo "Failed to remove ${CNAME}" >&2
        FAILED=1
      }
      ;;
    *)
      echo "Keeping ${CNAME}."
      ;;
  esac
done

echo
echo "Done."
exit "${FAILED:-0}"
