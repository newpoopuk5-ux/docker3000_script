#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
export COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
export COMFY_ROOT="${COMFY_ROOT:-/workspace/ComfyUI}"
export UI_PORT="${UI_PORT:-3000}"
export FAVORITES_PATH="${FAVORITES_PATH:-/workspace/ComfyUI/output/favorites.json}"

cd "$APP_DIR"

if [ ! -f app.py ]; then
  echo "app.py not found in $APP_DIR. Mount customui_comfy to /app."
  exit 1
fi

mkdir -p "$COMFY_ROOT/output"

echo "Starting customui_comfy Flask bridge"
echo "UI_PORT=$UI_PORT"
echo "COMFY_URL=$COMFY_URL"
echo "COMFY_ROOT=$COMFY_ROOT"
exec python app.py
