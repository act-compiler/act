# The image of this host: IMAGE_NAME, VARIANT, ARCH and GPU_FLAGS

# The cuda13 image and the GPUs on a GPU host
VARIANT=""
GPU_FLAGS=()
if nvidia-smi >/dev/null 2>&1 &&
  docker info -f '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  VARIANT="-cuda13"
  GPU_FLAGS=(--gpus all)
fi
# --cpu or --cuda13 overrides the host's pick
for arg in "$@"; do
  case "$arg" in
    --cpu) VARIANT="" ;;
    --cuda13) VARIANT="-cuda13" ;;
  esac
done
# The GPU is passed to the cuda13 image alone
[ -n "${VARIANT}" ] || GPU_FLAGS=()

# amd64 or arm64, by the machine
case "$(uname -m)" in
  x86_64) ARCH=amd64 ;;
  arm64 | aarch64) ARCH=arm64 ;;
  *)
    echo "Error: Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac
IMAGE_NAME="devanshdvj/act:v1.3${VARIANT}-${ARCH}"
