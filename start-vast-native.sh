#!/usr/bin/env bash
# Native Vast startup (no Docker). Keeps Flask on :3000; Comfy starts via Muse /api/mode/comfy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export AUTOSTART_MODE="${AUTOSTART_MODE:-none}"
export BOOTSTRAP="${BOOTSTRAP:-0}"
export MODEL_SET="${MODEL_SET:-none}"
export VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
export COMFY_ROOT="${COMFY_ROOT:-$VOLUME_ROOT/ComfyUI}"
export COMFY_PORT="${COMFY_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:${COMFY_PORT}}"
export FLASK_PORT="${FLASK_PORT:-3000}"
export UI_PORT="${UI_PORT:-$FLASK_PORT}"
export WORKER_CONTROL="${WORKER_CONTROL:-1}"
export APP_DIR="$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

mkdir -p "$VOLUME_ROOT/.muse-worker"
printf '%s' "$AUTOSTART_MODE" > "$VOLUME_ROOT/.muse-worker/mode"

echo "=== Muse worker native start ==="
echo "VOLUME_ROOT=$VOLUME_ROOT"
echo "COMFY_ROOT=$COMFY_ROOT"
echo "FLASK_PORT=$FLASK_PORT COMFY_PORT=$COMFY_PORT"
echo "AUTOSTART_MODE=$AUTOSTART_MODE"

VENV_DIR="$VOLUME_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Running bootstrap.sh --install-only (ComfyUI folders, no model downloads)"
bash "$SCRIPT_DIR/bootstrap.sh" --install-only

if [ "$AUTOSTART_MODE" = "comfy" ]; then
  if ! curl -fsS "$COMFY_URL/system_stats" >/dev/null 2>&1; then
    if [ -f "$COMFY_ROOT/main.py" ]; then
      echo "AUTOSTART_MODE=comfy — starting ComfyUI"
      nohup python "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port "$COMFY_PORT" \
        > "$VOLUME_ROOT/comfyui.log" 2>&1 &
      echo $! > "$VOLUME_ROOT/.muse-worker/comfy.pid"
    else
      echo "WARN: ComfyUI missing at $COMFY_ROOT — start via Muse Start Comfy after bootstrap"
    fi
  fi
fi

echo "Starting Flask on 0.0.0.0:$FLASK_PORT"
exec python -u app.py
