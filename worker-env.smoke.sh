#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# shellcheck disable=SC1091
source "$ROOT/worker-env.sh"

export APP_PASSWORD="from-process"
unset HF_TOKEN CIVITAI_TOKEN MODEL_SET

cat > "$TMP/workspace.env" <<'EOF'
APP_PASSWORD=from-workspace
HF_TOKEN=from-workspace-hf
CIVITAI_TOKEN=from-workspace-civitai
AUTOSTART_MODE=comfy
EOF

cat > "$TMP/repo.env" <<'EOF'
APP_PASSWORD=from-repo
HF_TOKEN=from-repo-hf
MODEL_SET=basic
EOF

load_env_file_if_unset "$TMP/workspace.env"
load_env_file_if_unset "$TMP/repo.env"

[ "$APP_PASSWORD" = "from-process" ] || { echo "process APP_PASSWORD should win"; exit 1; }
[ "$HF_TOKEN" = "from-workspace-hf" ] || { echo "workspace HF_TOKEN should load"; exit 1; }
[ "$CIVITAI_TOKEN" = "from-workspace-civitai" ] || { echo "workspace CIVITAI_TOKEN should load"; exit 1; }
[ "$AUTOSTART_MODE" = "comfy" ] || { echo "workspace AUTOSTART_MODE should load"; exit 1; }
[ "${MODEL_SET:-}" = "basic" ] || { echo "repo MODEL_SET should fill unset"; exit 1; }

unset AUTOSTART_MODE BOOTSTRAP MODEL_SET VOLUME_ROOT FLASK_PORT COMFY_PORT WORKER_CONTROL
apply_worker_env_defaults
[ "$AUTOSTART_MODE" = "none" ] || exit 1
[ "$BOOTSTRAP" = "0" ] || exit 1
[ "$MODEL_SET" = "none" ] || exit 1
[ "$VOLUME_ROOT" = "/workspace" ] || exit 1
[ "$FLASK_PORT" = "3000" ] || exit 1
[ "$COMFY_PORT" = "8188" ] || exit 1
[ "$WORKER_CONTROL" = "1" ] || exit 1

summary="$(print_worker_secret_summary)"
echo "$summary" | grep -q "APP_PASSWORD configured: yes" || exit 1
echo "$summary" | grep -q "HF_TOKEN configured: yes" || exit 1
echo "$summary" | grep -q "CIVITAI_TOKEN configured: yes" || exit 1
echo "$summary" | grep -q "from-process" && exit 1
echo "$summary" | grep -q "from-workspace" && exit 1

echo "worker-env.smoke: ok"
