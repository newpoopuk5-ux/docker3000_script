#!/usr/bin/env bash
# Idempotent llama.cpp CUDA build for Muse worker.
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
LLAMA_ROOT="${LLAMA_ROOT:-$VOLUME_ROOT/llama.cpp}"
BIN_PATH="$LLAMA_ROOT/build/bin/llama-server"

if [ -x "$BIN_PATH" ]; then
  echo "llama-server already installed: $BIN_PATH"
  exit 0
fi

echo "Installing build deps for llama.cpp..."
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y build-essential cmake git curl
fi

mkdir -p "$VOLUME_ROOT/models/llm"

if [ ! -d "$LLAMA_ROOT/.git" ]; then
  echo "Cloning llama.cpp into $LLAMA_ROOT"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_ROOT"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. CUDA build is required for Muse LLM worker."
  echo "Use a GPU Vast template with CUDA drivers, or pre-build llama-server manually."
  exit 1
fi

echo "Building llama-server with GGML_CUDA=ON"
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build "$LLAMA_ROOT/build" --config Release -j "$(nproc)"

if [ ! -x "$BIN_PATH" ]; then
  echo "ERROR: CUDA build finished but $BIN_PATH is missing."
  exit 1
fi

echo "llama-server ready: $BIN_PATH"
