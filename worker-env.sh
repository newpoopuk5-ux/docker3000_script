#!/usr/bin/env bash
# Source from start-vast-native.sh — load worker env without overriding existing exports.

trim_ws() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

strip_env_quotes() {
  local v="$1"
  if [ "${#v}" -ge 2 ]; then
    if [ "${v:0:1}" = '"' ] && [ "${v: -1}" = '"' ]; then
      v="${v:1:-1}"
    elif [ "${v:0:1}" = "'" ] && [ "${v: -1}" = "'" ]; then
      v="${v:1:-1}"
    fi
  fi
  printf '%s' "$v"
}

# Load KEY=VALUE lines from file only when the key is not already set in the shell.
load_env_file_if_unset() {
  local file="$1"
  [ -f "$file" ] || return 0
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line//$'\r'/}"
    case "$line" in
      ''|\#*) continue ;;
    esac
    case "$line" in
      export\ *) line="${line#export }" ;;
    esac
    case "$line" in
      *"="*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    key="$(trim_ws "$key")"
    value="$(strip_env_quotes "$(trim_ws "$value")")"
    [ -n "$key" ] || continue
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}

load_worker_env() {
  local script_dir="${1:-}"
  load_env_file_if_unset /etc/environment
  load_env_file_if_unset /workspace/.env
  if [ -n "$script_dir" ]; then
    load_env_file_if_unset "$script_dir/.env"
  else
    load_env_file_if_unset /workspace/docker3000_script/.env
  fi
}

apply_worker_env_defaults() {
  export AUTOSTART_MODE="${AUTOSTART_MODE:-none}"
  export BOOTSTRAP="${BOOTSTRAP:-0}"
  export MODEL_SET="${MODEL_SET:-none}"
  export VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
  export FLASK_PORT="${FLASK_PORT:-3000}"
  export COMFY_PORT="${COMFY_PORT:-8188}"
  export WORKER_CONTROL="${WORKER_CONTROL:-1}"
  export COMFY_ROOT="${COMFY_ROOT:-$VOLUME_ROOT/ComfyUI}"
  export COMFY_URL="${COMFY_URL:-http://127.0.0.1:${COMFY_PORT}}"
  export UI_PORT="${UI_PORT:-$FLASK_PORT}"
}

env_configured_yes_no() {
  local value="${1:-}"
  value="$(trim_ws "$value")"
  if [ -n "$value" ]; then
    echo "yes"
  else
    echo "no"
  fi
}

print_worker_secret_summary() {
  echo "APP_PASSWORD configured: $(env_configured_yes_no "${APP_PASSWORD:-}${UI_PASSWORD:-}")"
  echo "HF_TOKEN configured: $(env_configured_yes_no "${HF_TOKEN:-}")"
  echo "CIVITAI_TOKEN configured: $(env_configured_yes_no "${CIVITAI_TOKEN:-}")"
}
