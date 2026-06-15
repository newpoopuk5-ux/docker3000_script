#!/usr/bin/env python3
"""Local smoke test for worker_status (no Comfy/Docker required)."""
import json
import os
import sys

os.environ.setdefault("WORKER_MODE", "none")
os.environ.setdefault("VOLUME_ROOT", "/workspace")
os.environ.setdefault("COMFY_ROOT", "/workspace/ComfyUI")
os.environ.setdefault("UI_PORT", "3000")

import smoke_bootstrap  # noqa: F401

from worker_status import build_worker_status

payload = build_worker_status()
assert payload.get("ok") is True
assert payload.get("mode") in ("none", "comfy")
assert "services" in payload
assert payload["services"]["flask"]["online"] is True
assert "storage" in payload
assert "tokens" in payload
assert "hf_token_configured" in payload["tokens"]
assert "env" in payload
assert "HF_TOKEN" not in json.dumps(payload)
print(json.dumps(payload, indent=2))
print("worker_status.smoke: ok")
