#!/bin/bash
set -euo pipefail

# Work in the docker/ directory
cd "$(dirname "$0")"

# The image of this host, or of --cpu or --cuda13
source ./image.sh "$@"

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
