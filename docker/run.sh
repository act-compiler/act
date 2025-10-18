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

USER_NAME="$(id -un)"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_MOUNT="$(pwd)/.."
CONTAINER_MOUNT="/workspace"
EPHEMERAL_NAME="act-rm-${USER_NAME}"
DEFAULT_PERSISTENT_NAME="act-default-${USER_NAME}"

usage() {
  cat <<EOF
Usage: $0 [--persistent] [--no-default] [-h|--help]

Options:
  --persistent    Use or create a persistent container (default name: ${DEFAULT_PERSISTENT_NAME}).
  --no-default    Do NOT use the default name; present interactive selection/creation of persistent containers.
  -h, --help      Show this help message.
EOF
}

# Parse args
PERSISTENT=0
NO_DEFAULT=0
while [ $# -gt 0 ]; do
    case "$1" in
        --persistent) PERSISTENT=1; shift ;;
        --no-default) NO_DEFAULT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

# Helper: List all persistent containers of form act-<base>-<username>
list_persistent_all() {
    docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}' \
      | awk -v user="${USER_NAME}" -F '\t' '$1 ~ ("^act-.*-"user"$") { print $0 }'
}

# Helper: Ensure name is normalized to: act-<base>-<username>
normalize_name() {
    local raw="$1"
    local prefix="act-"
    local suffix="-${USER_NAME}"

    # remove leading prefix if present
    local base="${raw#${prefix}}"

    # remove trailing suffix if present
    if [[ "${base}" == *"${suffix}" ]]; then
        base="${base%${suffix}}"
    fi

    printf '%s' "${prefix}${base}${suffix}"
}

# Helper: Create persistent container
create_persistent() {
    local name="$1"
    echo "Creating persistent container (will not be removed on exit) with name ${name}."

    docker run -it --name "${name}" \
      -v "${HOST_MOUNT}:${CONTAINER_MOUNT}:rw" \
      -w "${CONTAINER_MOUNT}" \
      -e HOST_UID="${HOST_UID}" -e HOST_GID="${HOST_GID}" \
      "${IMAGE_NAME}"
}

# Helper: Restart an existing container
attach_existing() {
    local name="$1"
    echo "Restarting persistent container (will not be removed on exit) with name ${name}."

    docker start -ai "${name}"
    ./chown.sh >/dev/null
}


# Ensure image exists
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "Image ${IMAGE_NAME} not found locally. Build it first with ./build.sh"
    exit 1
fi

# Ephemeral flow: if --persistent is not set, run an ephemeral container
if [ "${PERSISTENT}" -eq 0 ]; then
    echo "Launching ephemeral container (will be removed on exit) with name ${EPHEMERAL_NAME}."
    docker run -it --rm \
      --name "${EPHEMERAL_NAME}" \
      -v "${HOST_MOUNT}:${CONTAINER_MOUNT}:rw" \
      -w "${CONTAINER_MOUNT}" \
      -e HOST_UID="${HOST_UID}" -e HOST_GID="${HOST_GID}" \
      "${IMAGE_NAME}"
    exit 0
fi

# Persistent flow: if --persistent is set, we handle persistent containers
if [ "${NO_DEFAULT}" -eq 0 ]; then
    # default behavior: use default name
    if docker ps -a --format '{{.Names}}' | grep -xq "${DEFAULT_PERSISTENT_NAME}"; then
        attach_existing "${DEFAULT_PERSISTENT_NAME}"
        exit 0
    else
        create_persistent "${DEFAULT_PERSISTENT_NAME}"
        exit 0
    fi
fi

# If --no-default: interactive selection/creation
EXISTING="$(list_persistent_all)"
if [ -z "${EXISTING}" ]; then
    read -rp "No persistent containers found. Enter a name for the new persistent container (without suffix): " NAME_IN
    if [ -z "${NAME_IN}" ]; then
      echo "No name provided. Aborting."
      exit 1
    fi
    if [[ ! "$NAME_IN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        echo "Invalid name: $NAME_IN. Aborting."
        echo "Supported Format: [a-zA-Z0-9][a-zA-Z0-9_.-]*"
        exit 1
    fi
    NAME="$(normalize_name "${NAME_IN}")"
    create_persistent "${NAME}"
    exit 0
fi

# Present choices: create new or choose existing
echo "Existing persistent containers:"
echo -e "INDEX\tNAME\tIMAGE\tSTATUS"
mapfile -t LINES < <(printf '%s\n' "${EXISTING}")
i=1
for ln in "${LINES[@]}"; do
  name=$(awk '{print $1}' <<<"$ln")
  rest=$(awk '{$1=""; sub(/^ /,""); print}' <<<"$ln")
  echo -e "${i}\t${name}\t${rest}"
  NAMES[$i]="$name"
  ((i++))
done

echo
echo "Options:"
echo "  0) Create a new persistent container"
for idx in "${!NAMES[@]}"; do
  echo "  ${idx}) Attach to ${NAMES[$idx]}"
done

read -rp "Enter choice [0-${#NAMES[@]}]: " CHOICE
case "$CHOICE" in
  0)
    read -rp "Enter a name for the new persistent container (without suffix): " NAME_IN
    if [ -z "${NAME_IN}" ]; then
      echo "No name provided. Aborting."
      exit 1
    fi

    if [[ ! "$NAME_IN" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]]; then
        echo "Invalid name: $NAME_IN. Aborting."
        echo "Supported Format: [a-zA-Z0-9][a-zA-Z0-9_.-]*"
        exit 1
    fi
    NAME="$(normalize_name "${NAME_IN}")"
    if docker ps -a --format '{{.Names}}' | grep -xq "${NAME}"; then
      echo "Container ${NAME} already exists. Aborting."
      exit 1
    fi
    create_persistent "${NAME}"
    ;;
  *)
    if ! [[ "$CHOICE" =~ ^[0-9]+$ ]]; then
      echo "Invalid choice. Aborting."
      exit 1
    fi
    if [ -z "${NAMES[$CHOICE]:-}" ]; then
      echo "Invalid choice. Aborting."
      exit 1
    fi
    SEL="${NAMES[$CHOICE]}"
    echo "Selected existing container: ${SEL}"
    attach_existing "${SEL}"
    ;;
esac
