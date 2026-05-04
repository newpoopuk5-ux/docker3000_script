#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-docker3000_script:local}"
CONTAINER_NAME="${CONTAINER_NAME:-muse-flask-3000}"
UI_PORT="${UI_PORT:-3000}"
COMFY_PORT="${COMFY_PORT:-8188}"
COMFY_URL="${COMFY_URL:-http://127.0.0.1:${COMFY_PORT}}"
PULL_IMAGE="${PULL_IMAGE:-0}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
BUILD_IF_MISSING="${BUILD_IF_MISSING:-1}"
MOUNT_CODE="${MOUNT_CODE:-0}"

detect_volume_root() {
  if [ -n "${VOLUME_ROOT:-}" ]; then
    echo "$VOLUME_ROOT"
    return
  fi
  for candidate in /workspace /root/volume /runpod-volume /mnt/data /volume; do
    if [ -d "$candidate" ]; then
      echo "$candidate"
      return
    fi
  done
  echo "/workspace"
}

VOLUME_ROOT="$(detect_volume_root)"
COMFY_ROOT_HOST="${COMFY_ROOT_HOST:-$VOLUME_ROOT/ComfyUI}"

mkdir -p "$COMFY_ROOT_HOST/output"

if [ "$PULL_IMAGE" = "1" ]; then
  echo "Pulling Docker image: $IMAGE"
  docker pull "$IMAGE" || true
fi

if [ "$BUILD_IMAGE" = "1" ]; then
  echo "Building Docker image from local repo: $IMAGE"
  docker build -t "$IMAGE" "$SCRIPT_DIR"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [ "$BUILD_IF_MISSING" != "1" ]; then
    echo "Docker image not found and BUILD_IF_MISSING=0: $IMAGE"
    exit 1
  fi
  echo "Building Docker image locally: $IMAGE"
  docker build -t "$IMAGE" "$SCRIPT_DIR"
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

VOLUME_ARGS=(-v "$COMFY_ROOT_HOST:/workspace/ComfyUI")
if [ "$MOUNT_CODE" = "1" ]; then
  VOLUME_ARGS+=(-v "$SCRIPT_DIR:/app")
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network host \
  "${VOLUME_ARGS[@]}" \
  -e UI_PORT="$UI_PORT" \
  -e COMFY_URL="$COMFY_URL" \
  -e COMFY_ROOT="/workspace/ComfyUI" \
  -e APP_PASSWORD="${APP_PASSWORD:-}" \
  -e UI_PASSWORD="${UI_PASSWORD:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e WORKFLOW_PATH="${WORKFLOW_PATH:-workflow.json}" \
  -e WORKFLOW_IMG2IMG_PATH="${WORKFLOW_IMG2IMG_PATH:-workflow_img2img.json}" \
  -e WORKFLOW_NODE_MAP_PATH="${WORKFLOW_NODE_MAP_PATH:-workflow_map.json}" \
  -e FAVORITES_PATH="${FAVORITES_PATH:-/workspace/ComfyUI/output/favorites.json}" \
  "$IMAGE"

echo "Running $CONTAINER_NAME"
echo "Flask bridge: http://127.0.0.1:$UI_PORT"
echo "ComfyUI:      $COMFY_URL"
echo "Comfy mount:  $COMFY_ROOT_HOST -> /workspace/ComfyUI"
if [ "$MOUNT_CODE" = "1" ]; then
  echo "Code mount:   $SCRIPT_DIR -> /app"
fi
