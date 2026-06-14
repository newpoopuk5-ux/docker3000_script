"""Refresh model_registry entries from Civitai (versions, sizes, previews)."""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import requests

from build_registry import (
    _infer_tags,
    _preview_url_from_meta,
    _version_file_row,
)
from download_models import civitai_lookup, resolve_civitai_file
from model_registry import load_entries, load_index, write_registry_files

LOOKUP_DELAY_SEC = 0.25
LOOKUP_RETRIES = 4


def _get_json(url: str) -> dict:
    last_err: Exception | None = None
    for attempt in range(LOOKUP_RETRIES):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ConnectionError, OSError) as e:
            last_err = e
            wait = 0.8 * (attempt + 1)
            print(f"  retry {attempt + 1}/{LOOKUP_RETRIES} after error: {e}")
            time.sleep(wait)
    raise SystemExit(f"Civitai request failed: {last_err}")


def _lookup_version(version_id: int) -> dict:
    last_err: Exception | None = None
    for attempt in range(LOOKUP_RETRIES):
        try:
            return civitai_lookup(version_id)
        except (requests.RequestException, ConnectionError, OSError) as e:
            last_err = e
            wait = 0.8 * (attempt + 1)
            print(f"  version {version_id} retry {attempt + 1}/{LOOKUP_RETRIES}: {e}")
            time.sleep(wait)
    raise SystemExit(f"Civitai version lookup failed for {version_id}: {last_err}")


def refs_for_kind(kind: str) -> list[str]:
    index = load_index()
    if not index.get("ok"):
        raise SystemExit("Registry missing — run build_registry.py first")
    refs: list[str] = []
    for ref, summary in (index.get("entries") or {}).items():
        if str(summary.get("kind") or "") == kind:
            refs.append(ref)
    return sorted(refs)


def refs_for_all_civitai() -> list[str]:
    index = load_index()
    entries = load_entries()
    if not index.get("ok") or not entries.get("ok"):
        raise SystemExit("Registry missing — run build_registry.py first")
    by_ref = entries.get("by_ref") or {}
    refs: list[str] = []
    for ref in sorted((index.get("entries") or {}).keys()):
        entry = by_ref.get(ref) or {}
        if entry.get("source") == "civitai":
            refs.append(ref)
    return refs


def refresh_ref(ref: str) -> None:
    ref = ref.strip()
    index = load_index()
    entries = load_entries()
    if not index.get("ok") or not entries.get("ok"):
        raise SystemExit("Registry missing — run build_registry.py first")

    by_ref = dict(entries.get("by_ref") or {})
    entry = by_ref.get(ref)
    if not entry:
        raise SystemExit(f"Unknown ref: {ref}")
    if entry.get("source") != "civitai":
        raise SystemExit(f"Refresh only supports Civitai entries (got {entry.get('source')})")

    model_id = entry.get("model_id")
    if not model_id:
        raise SystemExit("Entry has no model_id")

    full = _get_json(f"https://civitai.com/api/v1/models/{int(model_id)}")
    time.sleep(LOOKUP_DELAY_SEC)
    model_name = str(full.get("name") or entry.get("label") or f"model_{model_id}")

    version_rows = []
    for v in full.get("modelVersions") or []:
        if not isinstance(v, dict):
            continue
        v_id = int(v.get("id") or 0)
        if not v_id:
            continue
        meta = _lookup_version(v_id)
        time.sleep(LOOKUP_DELAY_SEC)
        file_entry = resolve_civitai_file(meta)
        fid = file_entry.get("id")
        fname = str(file_entry.get("name") or f"civitai_{v_id}.safetensors")
        from download_models import _file_size_bytes

        size_bytes = _file_size_bytes(file_entry)
        version_rows.append(
            _version_file_row(
                version_id=v_id,
                version_name=str(meta.get("name") or v.get("name") or ""),
                base_model=str(meta.get("baseModel") or v.get("baseModel") or ""),
                catalog_name=fname,
                download_url=(
                    f"https://civitai.com/api/download/models/{v_id}?fileId={fid}"
                    if fid
                    else f"https://civitai.com/api/download/models/{v_id}"
                ),
                model_name=model_name,
                file_id=int(fid) if fid else None,
                size_bytes=size_bytes,
                preview_remote_url=_preview_url_from_meta(meta),
            )
        )

    if not version_rows:
        raise SystemExit("No versions returned from Civitai")

    default_version = version_rows[0]
    default_vid = default_version["version_id"]
    default_base = str(default_version.get("base_model") or "")
    kind = str(entry.get("kind") or "checkpoint")

    entry = dict(entry)
    entry["label"] = model_name
    entry["default_version_id"] = default_vid
    entry["versions"] = version_rows
    by_ref[ref] = entry

    index_entries = dict(index.get("entries") or {})
    summary = dict(index_entries.get(ref) or {})
    summary.update({
        "label": model_name,
        "default_version_id": default_vid,
        "default_base_model": default_base,
        "preview_remote_url": default_version.get("preview_remote_url"),
        "version_count": len(version_rows),
        "tags": _infer_tags(model_name, kind, default_base),
        "civitai_version_id": default_vid if isinstance(default_vid, int) else None,
    })
    index_entries[ref] = summary

    now = datetime.now(timezone.utc).isoformat()
    index["generated_at"] = now
    index["entries"] = index_entries
    entries["generated_at"] = now
    entries["by_ref"] = by_ref
    write_registry_files(index, entries)
    print(f"refreshed {ref}: {len(version_rows)} versions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh registry entries from Civitai")
    parser.add_argument("--ref", help="Registry ref, e.g. civitai:589918")
    parser.add_argument("--kind", help="Refresh all entries of this kind, e.g. lora")
    parser.add_argument("--all", action="store_true", help="Refresh every Civitai entry in the registry")
    args = parser.parse_args()

    if args.all:
        refs = refs_for_all_civitai()
        if not refs:
            raise SystemExit("No Civitai entries in registry")
        print(f"refreshing {len(refs)} Civitai entries…")
        for ref in refs:
            refresh_ref(ref)
            time.sleep(0.5)
        print(f"done: {len(refs)} entries")
        return

    if args.kind:
        refs = refs_for_kind(args.kind.strip())
        if not refs:
            raise SystemExit(f"No entries with kind={args.kind!r}")
        print(f"refreshing {len(refs)} {args.kind} entries…")
        for ref in refs:
            refresh_ref(ref)
            time.sleep(0.5)
        print(f"done: {len(refs)} entries")
        return

    if not args.ref:
        raise SystemExit("Provide --ref, --kind, or --all")
    refresh_ref(args.ref)


if __name__ == "__main__":
    main()
