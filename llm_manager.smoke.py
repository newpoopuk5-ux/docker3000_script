import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from llm_manager import load_profiles, manager_supported, profile_storage_dir, resolve_profile

assert manager_supported(), "llm_models.json missing"
payload = load_profiles()
assert payload.get("ok"), payload
profiles = {row["id"]: row for row in payload.get("profiles") or []}
assert "qwen3_14b_uncensored_q4" in profiles
assert profiles["qwen3_14b_uncensored_q4"].get("recommended") is True
assert "qwen35_uncensored_q4_5090" in profiles
assert "gemma4_31b_uncensored_q4_5090" in profiles
assert payload.get("recommended_profile") == "qwen3_14b_uncensored_q4"
dir_path = profile_storage_dir("qwen3_14b_uncensored_q4", {"custom": False})
assert str(dir_path).endswith("qwen3_14b_uncensored_q4")
assert resolve_profile("missing_profile_xyz") is None
print(json.dumps({"ok": True, "profile_count": len(profiles), "recommended": payload.get("recommended_profile")}))
