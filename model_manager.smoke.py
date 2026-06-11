"""Smoke test for model_manager catalog + safe delete guards."""
import json
import tempfile
from pathlib import Path

from model_manager import ALLOWED_DELETE_DIRS, _safe_delete_path, load_catalog

payload = load_catalog()
assert payload.get("ok") is True, "catalog should load"
assert payload.get("recommended_set") == "basic_no_flux", "recommended_set"
assert payload.get("active_set") == "basic_no_flux", "default active_set"
no_flux = load_catalog("sdxl_no_flux")
assert len(no_flux.get("catalog") or []) >= 4, "sdxl_no_flux catalog"
hidden = load_catalog("*", include_flux=False)
assert not any((row.get("set") or "").startswith("flux_") for row in hidden.get("catalog") or []), "flux hidden in merge"
assert isinstance(payload.get("installed"), list), "installed list"
assert isinstance(payload.get("catalog"), list), "catalog list"

try:
    _safe_delete_path("checkpoints", "../outside.safetensors")
    raise AssertionError("path traversal should fail")
except ValueError:
    pass

print("model_manager.smoke: ok")
