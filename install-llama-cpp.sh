#!/usr/bin/env bash
# Idempotent llama.cpp CUDA build for Muse worker (single-GPU arch + Ninja).
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
  apt-get install -y build-essential cmake git curl ninja-build
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

detect_cuda_arch() {
  local cap=""
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if [ -z "$cap" ]; then
    return 1
  fi
  local major="${cap%%.*}"
  local minor="${cap#*.}"
  minor="${minor:-0}"
  echo "${major}${minor}"
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs || true)"
CMAKE_ARCH="${LLM_CUDA_ARCHITECTURES:-}"
if [ -z "$CMAKE_ARCH" ]; then
  CMAKE_ARCH="$(detect_cuda_arch || true)"
fi

CMAKE_ARCH_ARGS=()
if [ -n "$CMAKE_ARCH" ]; then
  echo "GPU: ${GPU_NAME:-unknown} · compile for CMAKE_CUDA_ARCHITECTURES=${CMAKE_ARCH} only"
  CMAKE_ARCH_ARGS=(-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_ARCH}" -DGGML_NATIVE=OFF)
else
  echo "WARN: could not detect GPU arch — falling back to GGML_NATIVE=ON"
  CMAKE_ARCH_ARGS=(-DGGML_NATIVE=ON)
fi

GENERATOR=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR=(-G Ninja)
else
  echo "WARN: ninja not found — using default generator (slower)"
fi

JOBS="$(nproc 2>/dev/null || echo 4)"

echo "Configuring llama-server (GGML_CUDA=ON, FA_ALL_QUANTS=OFF, Ninja=${GENERATOR:+yes})"
cmake -S "$LLAMA_ROOT" -B "$LLAMA_ROOT/build" \
  "${GENERATOR[@]}" \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  "${CMAKE_ARCH_ARGS[@]}"

echo "Building llama-server (-j ${JOBS})..."
cmake --build "$LLAMA_ROOT/build" --config Release -j "$JOBS"

if [ ! -x "$BIN_PATH" ]; then
  echo "ERROR: CUDA build finished but $BIN_PATH is missing."
  exit 1
fi

echo "llama-server ready: $BIN_PATH"
