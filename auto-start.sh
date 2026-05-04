#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

load_env() {
  if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
  fi
}

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

wait_for_comfy() {
  local url="$1"
  for _ in $(seq 1 90); do
    if curl -fsS "$url/system_stats" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

load_env

MODEL_SET="${MODEL_SET:-all,flux_user}"
BOOTSTRAP="${BOOTSTRAP:-1}"
COMFY_PORT="${COMFY_PORT:-8188}"
UI_PORT="${UI_PORT:-3000}"

VOLUME_ROOT="$(detect_volume_root)"
export VOLUME_ROOT
export COMFY_ROOT="${COMFY_ROOT:-$VOLUME_ROOT/ComfyUI}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:${COMFY_PORT}}"
export APP_DIR="$SCRIPT_DIR"
export UI_PORT

mkdir -p "$VOLUME_ROOT"

echo "VOLUME_ROOT=$VOLUME_ROOT"
echo "COMFY_ROOT=$COMFY_ROOT"
echo "APP_DIR=$APP_DIR"
echo "COMFY_URL=$COMFY_URL"

if [ "$BOOTSTRAP" = "1" ]; then
  echo "Running bootstrap.sh with MODEL_SET=$MODEL_SET"
  (cd "$SCRIPT_DIR" && bash bootstrap.sh "$MODEL_SET")
else
  echo "Skipping bootstrap because BOOTSTRAP=$BOOTSTRAP"
fi

if ! curl -fsS "$COMFY_URL/system_stats" >/dev/null 2>&1; then
  echo "Starting ComfyUI on port $COMFY_PORT"
  cd "$COMFY_ROOT"
  nohup python main.py --listen 0.0.0.0 --port "$COMFY_PORT" > "$VOLUME_ROOT/comfyui.log" 2>&1 &
  cd "$SCRIPT_DIR"
  if ! wait_for_comfy "$COMFY_URL"; then
    echo "ComfyUI did not become ready. Last log lines:"
    tail -80 "$VOLUME_ROOT/comfyui.log" || true
    exit 1
  fi
else
  echo "ComfyUI is already online"
fi

bash "$SCRIPT_DIR/docker-run.sh"

echo ""
echo "Ready."
echo "Flask bridge: http://<vast-host>:$UI_PORT"
echo "ComfyUI:      http://<vast-host>:$COMFY_PORT"
