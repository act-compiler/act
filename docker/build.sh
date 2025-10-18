#!/bin/bash
set -euo pipefail

# Change to script directory
cd "$(dirname "$0")"

# Detect architecture
ARCH=$(uname -m)

if [ "$ARCH" = "x86_64" ]; then
    DOCKERFILE="Dockerfile.amd64"
    IMAGE_NAME="devanshdvj/act:latest-amd64"
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    DOCKERFILE="Dockerfile.arm64"
    IMAGE_NAME="devanshdvj/act:latest-arm64"
else
    echo "Error: Unsupported architecture: $ARCH"
    exit 1
fi

usage() {
  cat <<EOF
Usage: $0 [-r|--rebuild] [-f|--force] [-y|--no-confirm] [-h|--help]

Options:
  -r, --rebuild     Remove old image (and optionally containers with --force) and rebuild.
  -f, --force       Remove containers referencing ${IMAGE_NAME} (only useful with --rebuild).
  -y, --no-confirm  Do not prompt for confirmation when --force is used.
  -h, --help        Show this help message.
EOF
}

# Parse args
REBUILD=0
FORCE=0
NO_CONFIRM=0
while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild) REBUILD=1; shift ;;
        --force) FORCE=1; shift ;;
        --no-confirm) NO_CONFIRM=1; shift ;;
        --help) usage; exit 0 ;;
        -[!-]*)
            # support combined short flags e.g. -rf or -rfy
            flags="${1#-}"
            for (( i=0; i<${#flags}; i++ )); do
                ch="${flags:i:1}"
                case "$ch" in
                    r) REBUILD=1 ;;
                    f) FORCE=1 ;;
                    y) NO_CONFIRM=1 ;;
                    h) usage; exit 0 ;;
                    *)
                        echo "Unknown flag: -${ch}"
                        echo
                        usage
                        exit 1
                    ;;
                esac
            done
            shift
        ;;
        *)
            echo "Unknown argument: $1"
            echo
            usage
            exit 1
        ;;
    esac
done

# Does an old image exist?
if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Found existing image ${IMAGE_NAME}."
    IMAGE_EXISTS=1
else
    echo "No existing image named ${IMAGE_NAME} found."
    IMAGE_EXISTS=0
fi

# If image exists and user didn't ask to rebuild, just inform and exit
if [ "${IMAGE_EXISTS}" -eq 1 ] && [ "${REBUILD}" -eq 0 ]; then
    echo
    echo "Image ${IMAGE_NAME} already exists. To replace it, re-run with --rebuild. Aborting."
    exit 0
fi

# If image exists and user asked to rebuild, attempt to remove it
if [ "${IMAGE_EXISTS}" -eq 1 ] && [ "${REBUILD}" -eq 1 ]; then
    echo "--rebuild specified: will attempt to remove existing image ${IMAGE_NAME}."
    # List containers that reference this image (do not remove automatically unless --force)
    CONTAINERS=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}	{{.Names}}	{{.Status}}') || true
    if [ -n "${CONTAINERS}" ]; then # Containers reference this image
        echo
        echo "Containers referencing image ${IMAGE_NAME}:"
        echo -e "CONTAINER_ID\tNAME\tSTATUS"
        echo "${CONTAINERS}"
        echo
        if [ "${FORCE}" -eq 1 ]; then
            if [ "${NO_CONFIRM}" -eq 0 ]; then
                read -rp "You passed --force: remove the above containers now? [y/N] " RESP
                case "${RESP}" in
                    y|Y)
                        echo "--force confirmed: removing containers that reference ${IMAGE_NAME}."
                    ;;
                    *)
                        echo "--force not confirmed: Aborting."
                        exit 1
                    ;;
                esac
            else
                echo "--force and --no-confirm specified: removing containers that reference ${IMAGE_NAME}."
            fi
            # remove by container id
            docker ps -a --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}' | xargs -r docker rm -f
        else
            echo "Rebuild requested but containers reference this image. Re-run with --rebuild --force to remove them automatically, or remove them manually. Aborting."
            exit 1
        fi
    else # No containers reference this image
        echo
        echo "No containers reference ${IMAGE_NAME}."
    fi
    echo
    echo "Proceeding to remove image ${IMAGE_NAME}..."
    docker image rm "${IMAGE_NAME}"
fi

# Build
echo
echo "Building ${IMAGE_NAME}..."
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" .

# Prune dangling images
echo "Pruning dangling images..."
docker image prune -f || true

echo
echo "Docker image ${IMAGE_NAME} built successfully."
