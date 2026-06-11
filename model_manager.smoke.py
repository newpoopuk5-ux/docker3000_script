"""Smoke test for model_manager catalog + safe delete guards."""
import json
import tempfile
from pathlib import Path

from model_manager import ALLOWED_DELETE_DIRS, _safe_delete_path, load_catalog

payload = load_catalog("basic")
assert payload.get("ok") is True, "catalog should load"
assert isinstance(payload.get("installed"), list), "installed list"
assert isinstance(payload.get("catalog"), list), "catalog list"

try:
    _safe_delete_path("checkpoints", "../outside.safetensors")
    raise AssertionError("path traversal should fail")
except ValueError:
    pass

print("model_manager.smoke: ok")
