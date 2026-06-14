"""Smoke test model_registry JSON + installed merge."""
from model_registry import load_entry, load_index, merge_index_installed, registry_index_payload, registry_supported

assert registry_supported(), "registry files missing"
index = load_index()
assert index.get("ok"), "index load failed"
assert len(index.get("entries") or {}) >= 1, "expected catalog entries"
payload = registry_index_payload()
assert payload.get("ok"), "registry_index_payload failed"
assert isinstance(payload.get("installed_by_ref"), dict), "installed_by_ref dict"
first_ref = next(iter((index.get("entries") or {}).keys()))
entry = load_entry(first_ref)
assert entry, f"entry missing for {first_ref}"
merge = merge_index_installed(index)
assert first_ref in merge, "merge missing first ref"
print(f"model_registry.smoke ok: {len(index.get('entries') or {})} entries")
