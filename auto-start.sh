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

AUTOSTART_MODE="${AUTOSTART_MODE:-none}"
BOOTSTRAP="${BOOTSTRAP:-0}"
MODEL_SET="${MODEL_SET:-none}"
COMFY_PORT="${COMFY_PORT:-8188}"
UI_PORT="${UI_PORT:-3000}"

case "$AUTOSTART_MODE" in
  none|comfy) ;;
  *)
    echo "AUTOSTART_MODE must be none or comfy (got: $AUTOSTART_MODE)"
    exit 1
    ;;
esac

VOLUME_ROOT="$(detect_volume_root)"
export VOLUME_ROOT
export COMFY_ROOT="${COMFY_ROOT:-$VOLUME_ROOT/ComfyUI}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:${COMFY_PORT}}"
export APP_DIR="$SCRIPT_DIR"
export UI_PORT
export WORKER_MODE="$AUTOSTART_MODE"
export AUTOSTART_MODE
export BOOTSTRAP
export MODEL_SET

mkdir -p "$VOLUME_ROOT/.muse-worker"
printf '%s' "$AUTOSTART_MODE" > "$VOLUME_ROOT/.muse-worker/mode"

echo "VOLUME_ROOT=$VOLUME_ROOT"
echo "COMFY_ROOT=$COMFY_ROOT"
echo "APP_DIR=$APP_DIR"
echo "COMFY_URL=$COMFY_URL"
echo "AUTOSTART_MODE=$AUTOSTART_MODE"
echo "BOOTSTRAP=$BOOTSTRAP"
echo "MODEL_SET=$MODEL_SET"

if [ "$BOOTSTRAP" = "1" ]; then
  if [ "$MODEL_SET" = "none" ]; then
    echo "Running bootstrap.sh --install-only (no model downloads)"
    (cd "$SCRIPT_DIR" && bash bootstrap.sh --install-only)
  else
    echo "Running bootstrap.sh with MODEL_SET=$MODEL_SET"
    (cd "$SCRIPT_DIR" && bash bootstrap.sh "$MODEL_SET")
  fi
else
  echo "Skipping bootstrap because BOOTSTRAP=$BOOTSTRAP"
fi

if [ "$AUTOSTART_MODE" = "comfy" ]; then
  if ! curl -fsS "$COMFY_URL/system_stats" >/dev/null 2>&1; then
    if [ ! -f "$COMFY_ROOT/main.py" ]; then
      echo "ComfyUI is not installed at $COMFY_ROOT. Set BOOTSTRAP=1 or run bootstrap.sh first."
      exit 1
    fi
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
else
  echo "AUTOSTART_MODE=none — Flask control plane only (ComfyUI not started)"
fi

bash "$SCRIPT_DIR/docker-run.sh"

echo ""
echo "Ready."
echo "Worker mode: $AUTOSTART_MODE"
echo "Flask bridge: http://<vast-host>:$UI_PORT"
if [ "$AUTOSTART_MODE" = "comfy" ]; then
  echo "ComfyUI:      http://<vast-host>:$COMFY_PORT"
fi
echo "Worker status: http://<vast-host>:$UI_PORT/api/worker/status"
