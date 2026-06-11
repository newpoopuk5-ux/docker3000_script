#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

RAW_ARG="${1:-${MODEL_SET:-basic}}"
INSTALL_ONLY=0
SET_NAMES="$RAW_ARG"

if [ "$RAW_ARG" = "--install-only" ] || [ "$RAW_ARG" = "none" ]; then
  INSTALL_ONLY=1
  SET_NAMES=""
fi

VOLUME_ROOT="${VOLUME_ROOT:-}"

echo "Disk layout:"
df -h

if [ -z "$VOLUME_ROOT" ]; then
  for candidate in /workspace /root/volume /runpod-volume /mnt/data /volume; do
    if [ -d "$candidate" ]; then
      VOLUME_ROOT="$candidate"
      break
    fi
  done
fi

if [ -z "$VOLUME_ROOT" ]; then
  echo "Set VOLUME_ROOT first. Example: export VOLUME_ROOT=/workspace"
  exit 1
fi

export VOLUME_ROOT
export COMFY_ROOT="${COMFY_ROOT:-$VOLUME_ROOT/ComfyUI}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"

apt update
apt install -y aria2 ca-certificates curl git python3-pip python3-venv unzip wget

pip install -r requirements.txt

if [ ! -f "$COMFY_ROOT/main.py" ]; then
  echo "Installing ComfyUI into $COMFY_ROOT"
  mkdir -p "$(dirname "$COMFY_ROOT")"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"
else
  echo "ComfyUI already exists: $COMFY_ROOT"
fi

mkdir -p \
  "$COMFY_ROOT/models/checkpoints" \
  "$COMFY_ROOT/models/loras" \
  "$COMFY_ROOT/models/loras/flux" \
  "$COMFY_ROOT/models/vae" \
  "$COMFY_ROOT/models/controlnet" \
  "$COMFY_ROOT/models/upscale_models" \
  "$COMFY_ROOT/models/unet" \
  "$COMFY_ROOT/models/diffusion_models" \
  "$COMFY_ROOT/models/text_encoders" \
  "$COMFY_ROOT/custom_nodes" \
  "$COMFY_ROOT/output"

if [ -f "$COMFY_ROOT/requirements.txt" ]; then
  pip install -r "$COMFY_ROOT/requirements.txt"
fi

GGUF_NODE_DIR="$COMFY_ROOT/custom_nodes/ComfyUI-GGUF"
if [ ! -d "$GGUF_NODE_DIR/.git" ]; then
  echo "Installing ComfyUI-GGUF custom node"
  git clone --depth 1 https://github.com/city96/ComfyUI-GGUF "$GGUF_NODE_DIR"
else
  echo "ComfyUI-GGUF already exists: $GGUF_NODE_DIR"
  git -C "$GGUF_NODE_DIR" pull --ff-only || true
fi

if [ -f "$GGUF_NODE_DIR/requirements.txt" ]; then
  pip install -r "$GGUF_NODE_DIR/requirements.txt"
fi

if [ "$INSTALL_ONLY" = "1" ] || [ -z "$SET_NAMES" ]; then
  echo "Skipping model downloads (install-only / MODEL_SET=none)"
else
  echo "Downloading model set(s): $SET_NAMES"
  for SET_NAME in ${SET_NAMES//,/ }; do
    if [ "$SET_NAME" = "none" ]; then
      continue
    fi
    python download_models.py --set "$SET_NAME"
  done
fi

echo "Bootstrap complete."
echo "Start ComfyUI:"
echo "cd \"$COMFY_ROOT\" && python main.py --listen 0.0.0.0 --port 8188"
echo ""
echo "Then run the app container:"
echo "export VOLUME_ROOT=\"$VOLUME_ROOT\""
echo "bash docker-run.sh"
