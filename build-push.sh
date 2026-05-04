#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/newpoopuk5-ux/docker3000_script:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -t "$IMAGE" "$SCRIPT_DIR"
docker push "$IMAGE"

echo "Pushed $IMAGE"
