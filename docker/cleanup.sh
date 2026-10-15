#!/bin/bash
set -euo pipefail

# Work in the docker/ directory
cd "$(dirname "$0")"

# The image of this host, or of --cpu or --cuda13
source ./image.sh "$@"

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
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Every container of the image
LISTED=$(docker ps -a --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}')
mapfile -t CONTAINERS < <(awk -v img="${IMAGE_NAME}" '$3 == img' <<<"${LISTED}")

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
      docker rm -f "${CID}"
      ;;
    *)
      echo "Keeping ${CNAME}."
      ;;
  esac
done

echo
echo "Done."
