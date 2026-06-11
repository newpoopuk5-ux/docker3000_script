#!/usr/bin/env bash
# Native Vast startup (no Docker). Keeps Flask on :3000; Comfy starts via Muse /api/mode/comfy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/worker-env.sh"

# 1) keep process env  2) /etc/environment  3) /workspace/.env  4) repo .env  5) defaults
load_worker_env "$SCRIPT_DIR"
apply_worker_env_defaults

export APP_DIR="$SCRIPT_DIR"
export PYTHONUNBUFFERED=1

mkdir -p "$VOLUME_ROOT/.muse-worker"
printf '%s' "$AUTOSTART_MODE" > "$VOLUME_ROOT/.muse-worker/mode"

echo "=== Muse worker native start ==="
echo "VOLUME_ROOT=$VOLUME_ROOT"
echo "COMFY_ROOT=$COMFY_ROOT"
echo "FLASK_PORT=$FLASK_PORT COMFY_PORT=$COMFY_PORT"
echo "AUTOSTART_MODE=$AUTOSTART_MODE BOOTSTRAP=$BOOTSTRAP MODEL_SET=$MODEL_SET"
print_worker_secret_summary

VENV_DIR="$VOLUME_ROOT/venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"
VENV_PYTHON="$VENV_DIR/bin/python"

install_python_venv_package() {
  command -v apt-get >/dev/null 2>&1 || return 1
  local py_minor=""
  py_minor="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  echo "Installing python venv support via apt..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  if [ -n "$py_minor" ]; then
    apt-get install -y "python${py_minor}-venv" python3-venv || apt-get install -y python3-venv
  else
    apt-get install -y python3-venv
  fi
}

create_venv() {
  rm -rf "$VENV_DIR"
  python3 -m venv "$VENV_DIR"
}

if [ ! -f "$VENV_ACTIVATE" ] || [ ! -x "$VENV_PYTHON" ]; then
  echo "Creating venv at $VENV_DIR"
  if ! create_venv; then
    echo "WARN: python3 -m venv failed - trying apt install python3-venv..."
    install_python_venv_package || true
    if ! create_venv; then
      echo "ERROR: python3 -m venv still failed."
      echo "Run manually: apt-get update && apt-get install -y python3.12-venv"
      exit 1
    fi
  fi
fi
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "ERROR: venv incomplete - missing $VENV_ACTIVATE"
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_ACTIVATE"

python -m pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Running bootstrap.sh --install-only (ComfyUI folders, no model downloads)"
bash "$SCRIPT_DIR/bootstrap.sh" --install-only

if [ "$AUTOSTART_MODE" = "comfy" ]; then
  if ! curl -fsS "$COMFY_URL/system_stats" >/dev/null 2>&1; then
    if [ -f "$COMFY_ROOT/main.py" ]; then
      echo "AUTOSTART_MODE=comfy - starting ComfyUI"
      nohup python "$COMFY_ROOT/main.py" --listen 0.0.0.0 --port "$COMFY_PORT" \
        > "$VOLUME_ROOT/comfyui.log" 2>&1 &
      echo $! > "$VOLUME_ROOT/.muse-worker/comfy.pid"
    else
      echo "WARN: ComfyUI missing at $COMFY_ROOT - start via Muse Start Comfy after bootstrap"
    fi
  fi
fi

python "$SCRIPT_DIR/worker_startup_banner.py" || true

flask_probe_code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${FLASK_PORT}/api/health" 2>/dev/null || echo "000")"
if [ "$flask_probe_code" = "200" ] || [ "$flask_probe_code" = "401" ]; then
  echo "Flask already listening on 0.0.0.0:${FLASK_PORT} (HTTP ${flask_probe_code}) - skip second start."
  echo "Re-print Muse URLs: python worker_startup_banner.py --probe"
  exit 0
fi

echo "Starting Flask on 0.0.0.0:$FLASK_PORT"
exec python -u app.py
