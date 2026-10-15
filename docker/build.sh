#!/bin/bash
set -euo pipefail

# Work in the docker/ directory
cd "$(dirname "$0")"
source ./act-env.env

# The image of this host, or of --cpu or --cuda13
source ./image.sh "$@"
# The Dockerfile and environment image of this architecture
DOCKERFILE="Dockerfile.${ARCH}"
ENV_IMAGE_NAME="devanshdvj/act-env:${ACT_ENV_BUILD_TAG}${VARIANT}-${ARCH}"

usage() {
  cat <<EOF
Usage: $0 [-e|--env] [--cpu|--cuda13] [-r|--rebuild] [-f|--force] [-y|--no-confirm] [-h|--help]

Options:
  -e, --env         Build only the environment image ${ENV_IMAGE_NAME}.
  --cpu             Build the CPU images, the default unless the host has a GPU and the nvidia runtime.
  --cuda13          Build the cuda13 images, with jax's CUDA 13 runtime, the default on such a host.
  -r, --rebuild     Remove old image (and optionally containers with --force) and rebuild.
  -f, --force       Remove containers referencing ${IMAGE_NAME} (only useful with --rebuild).
  -y, --no-confirm  Do not prompt for confirmation when --force is used.
  -h, --help        Show this help message.
EOF
}

# The options; image.sh reads --cpu and --cuda13
ENV_ONLY=0
REBUILD=0
FORCE=0
NO_CONFIRM=0
while [ $# -gt 0 ]; do
  case "$1" in
    --env)
      ENV_ONLY=1
      shift
      ;;
    --cpu | --cuda13) shift ;;
    --rebuild)
      REBUILD=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-confirm)
      NO_CONFIRM=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    -[!-]*)
      # Combined short flags, as in -rf or -rfy
      flags="${1#-}"
      for ((i = 0; i < ${#flags}; i++)); do
        ch="${flags:i:1}"
        case "$ch" in
          e) ENV_ONLY=1 ;;
          r) REBUILD=1 ;;
          f) FORCE=1 ;;
          y) NO_CONFIRM=1 ;;
          h)
            usage
            exit 0
            ;;
          *)
            echo "Unknown flag: -${ch}" >&2
            echo >&2
            usage >&2
            exit 1
            ;;
        esac
      done
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo >&2
      usage >&2
      exit 1
      ;;
  esac
done

# The environment alone is the image to check and build
if [ "${ENV_ONLY}" -eq 1 ]; then
  IMAGE_NAME="${ENV_IMAGE_NAME}"
fi

# An existing image of the name
if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Found existing image ${IMAGE_NAME}."
  IMAGE_EXISTS=1
else
  echo "No existing image named ${IMAGE_NAME} found."
  IMAGE_EXISTS=0
fi

# An existing image stays unless --rebuild is given
if [ "${IMAGE_EXISTS}" -eq 1 ] && [ "${REBUILD}" -eq 0 ]; then
  echo
  echo "Image ${IMAGE_NAME} already exists. To replace it, re-run with --rebuild. Aborting."
  exit 0
fi

# --rebuild removes the existing image first
if [ "${IMAGE_EXISTS}" -eq 1 ] && [ "${REBUILD}" -eq 1 ]; then
  echo "--rebuild specified: will attempt to remove existing image ${IMAGE_NAME}."
  # Containers of the image, removed only with --force
  CONTAINERS=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}	{{.Names}}	{{.Status}}')
  if [ -n "${CONTAINERS}" ]; then
    echo
    echo "Containers referencing image ${IMAGE_NAME}:"
    echo -e "CONTAINER_ID\tNAME\tSTATUS"
    echo "${CONTAINERS}"
    echo
    if [ "${FORCE}" -eq 1 ]; then
      if [ "${NO_CONFIRM}" -eq 0 ]; then
        read -rp "You passed --force: remove the above containers now? [y/N] " RESP
        case "${RESP}" in
          y | Y)
            echo "--force confirmed: removing containers that reference ${IMAGE_NAME}."
            ;;
          *)
            echo "--force not confirmed: Aborting." >&2
            exit 1
            ;;
        esac
      else
        echo "--force and --no-confirm specified: removing containers that reference ${IMAGE_NAME}."
      fi
      # Remove by container id
      docker ps -a --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}' | xargs -r docker rm -f
    else
      echo "Rebuild requested but containers reference this image. Re-run with --rebuild --force to remove them automatically, or remove them manually. Aborting." >&2
      exit 1
    fi
  else
    echo
    echo "No containers reference ${IMAGE_NAME}."
  fi
  echo
  echo "Proceeding to remove image ${IMAGE_NAME}..."
  docker image rm "${IMAGE_NAME}"
fi

# Build one stage of the Dockerfile as an image
build_stage() {
  echo
  echo "Building $2..."
  docker build -f "${DOCKERFILE}" --target "$1" --build-arg ACT_ENV_TAG="${ACT_ENV_TAG}" --build-arg VARIANT="${VARIANT}" -t "$2" ..
}

# The environment image alone, or the ecosystem on it
if [ "${ENV_ONLY}" -eq 1 ]; then
  build_stage "env${VARIANT}" "${IMAGE_NAME}"
else
  # The environment: local, pulled, else built if the Dockerfiles' own
  if [ "${ACT_ENV_TAG}" = "${ACT_ENV_BUILD_TAG}" ] && ! docker image inspect "${ENV_IMAGE_NAME}" >/dev/null 2>&1 && ! docker pull "${ENV_IMAGE_NAME}"; then
    build_stage "env${VARIANT}" "${ENV_IMAGE_NAME}"
  fi
  build_stage ecosystem "${IMAGE_NAME}"
fi

echo
echo "Docker image ${IMAGE_NAME} built successfully."
