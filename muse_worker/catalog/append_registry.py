"""Append one Civitai model to model_registry/index.json + entries.json."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from build_registry import (
    CATEGORY_META,
    GROUP_TO_KIND,
    KINDS,
    RECOMMENDED_CATEGORIES,
    _infer_tags,
    _kind_for_group,
    _kind_label,
    _preview_url_from_meta,
    _slug,
    _version_file_row,
    build as rebuild_all,
    write_registry,
)
from download_models import (
    civitai_lookup,
    civitai_pretty_filename,
    civitai_trigger_words,
    extract_civitai_file_id,
    extract_civitai_version_id,
    is_civitai_model_page_url,
    resolve_civitai_file,
)
from model_registry import ENTRIES_JSON, INDEX_JSON, load_entries, load_index, write_registry_files

SCHEMA_VERSION = 1


def _load_pair() -> tuple[dict, dict]:
    index = load_index()
    entries = load_entries()
    if not index.get("ok") or not entries.get("ok"):
        raise SystemExit("Registry missing — run build_registry.py first")
    return index, entries


def _add_category_ref(categories: dict, category: str, ref: str) -> None:
    if category not in categories:
        categories[category] = {
            "label": category.replace("_", " "),
            "description": CATEGORY_META.get(category, ""),
            "refs": [],
        }
    if ref not in categories[category]["refs"]:
        categories[category]["refs"].append(ref)


def _rebuild_recommended(categories: dict) -> None:
    refs: list[str] = []
    for cat in RECOMMENDED_CATEGORIES:
        for ref in (categories.get(cat) or {}).get("refs") or []:
            if ref not in refs:
                refs.append(ref)
    categories["recommended"] = {
        "label": "Recommended",
        "description": "Anime starter + Anima + MiaoMiao bundles",
        "refs": refs,
    }


def _fetch_model(model_id: int) -> dict:
    url = f"https://civitai.com/api/v1/models/{int(model_id)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _ref_for_model(model_id: int, group: str, catalog_name: str, file_id: int | None) -> str:
    return f"civitai:{model_id}:f{file_id}" if file_id else f"civitai:{model_id}:{_slug(catalog_name, 32)}"


def append_from_url(url: str, category: str, *, kind: str | None = None, group: str | None = None) -> str:
    url = url.strip()
    index, entries = _load_pair()
    by_ref = dict(entries.get("by_ref") or {})
    index_entries = dict(index.get("entries") or {})
    categories = dict(index.get("categories") or {})

    if is_civitai_model_page_url(url):
        import re

        match = re.search(r"/models/(\d+)", url.replace("civitai.red", "civitai.com"), flags=re.I)
        if not match:
            raise SystemExit("Could not parse model id from URL")
        model_id = int(match.group(1))
        full = _fetch_model(model_id)
        model_name = str(full.get("name") or f"model_{model_id}")
        model_type = str(full.get("type") or "")
        main_url = f"https://civitai.com/models/{model_id}"
        versions_raw = full.get("modelVersions") or []
        if not versions_raw:
            raise SystemExit(f"No versions for model {model_id}")
        latest = versions_raw[0]
        version_id = int(latest.get("id") or 0)
        meta = civitai_lookup(version_id)
        time.sleep(0.12)
        file_entry = resolve_civitai_file(meta)
        catalog_name = str(file_entry.get("name") or f"civitai_{version_id}.safetensors")
        file_id = file_entry.get("id")
        resolved_group = group or ("loras" if model_type.upper() == "LORA" else "checkpoints")
        resolved_kind = kind or _kind_for_group(resolved_group, model_type)
        ref = f"civitai:{model_id}"
        if ref in by_ref and len({str(v.get("group")) for v in (by_ref[ref].get("versions") or [])}) > 1:
            ref = _ref_for_model(model_id, resolved_group, catalog_name, int(file_id) if file_id else None)

        version_rows = []
        for v in versions_raw:
            if not isinstance(v, dict):
                continue
            v_id = int(v.get("id") or 0)
            if not v_id:
                continue
            vmeta = civitai_lookup(v_id)
            time.sleep(0.12)
            fentry = resolve_civitai_file(vmeta)
            fid = fentry.get("id")
            fname = str(fentry.get("name") or catalog_name)
            version_rows.append(
                _version_file_row(
                    version_id=v_id,
                    version_name=str(vmeta.get("name") or v.get("name") or ""),
                    base_model=str(vmeta.get("baseModel") or v.get("baseModel") or ""),
                    catalog_name=fname,
                    download_url=(
                        f"https://civitai.com/api/download/models/{v_id}?fileId={fid}"
                        if fid
                        else f"https://civitai.com/api/download/models/{v_id}"
                    ),
                    model_name=model_name,
                    file_id=int(fid) if fid else None,
                    size_bytes=int(fentry.get("size") or 0) or None,
                    preview_remote_url=_preview_url_from_meta(vmeta),
                    trigger_words=civitai_trigger_words(vmeta),
                )
            )
    else:
        version_id = extract_civitai_version_id(url)
        if not version_id:
            raise SystemExit("URL must be a Civitai model page or /api/download/models/… link")
        meta = civitai_lookup(int(version_id))
        time.sleep(0.12)
        model_id = int(meta.get("modelId") or (meta.get("model") or {}).get("id") or 0)
        model_name = str((meta.get("model") or {}).get("name") or meta.get("name") or f"model_{model_id}")
        model_type = str((meta.get("model") or {}).get("type") or "")
        file_entry = resolve_civitai_file(meta, extract_civitai_file_id(url))
        catalog_name = str(file_entry.get("name") or f"civitai_{version_id}.safetensors")
        file_id = file_entry.get("id")
        resolved_group = group or ("loras" if model_type.upper() == "LORA" else "checkpoints")
        resolved_kind = kind or _kind_for_group(resolved_group, model_type)
        ref = f"civitai:{model_id}" if model_id else f"civitai:v:{version_id}"
        main_url = f"https://civitai.com/models/{model_id}" if model_id else url
        version_rows = [
            _version_file_row(
                version_id=int(version_id),
                version_name=str(meta.get("name") or ""),
                base_model=str(meta.get("baseModel") or ""),
                catalog_name=catalog_name,
                download_url=url,
                model_name=model_name,
                file_id=int(file_id) if file_id else extract_civitai_file_id(url),
                size_bytes=int(file_entry.get("size") or 0) or None,
                preview_remote_url=_preview_url_from_meta(meta),
                trigger_words=civitai_trigger_words(meta),
            )
        ]

    default_version = version_rows[0]
    default_vid = default_version["version_id"]
    default_base = str(default_version.get("base_model") or "")
    folder = (KINDS.get(resolved_kind) or {}).get("folder") or "checkpoints"

    index_entries[ref] = {
        "ref": ref,
        "source": "civitai",
        "kind": resolved_kind,
        "label": model_name,
        "model_id": model_id if model_id else None,
        "main_url": main_url,
        "default_version_id": default_vid,
        "default_base_model": default_base,
        "preview_remote_url": default_version.get("preview_remote_url"),
        "version_count": len(version_rows),
        "tags": _infer_tags(model_name, resolved_kind, default_base),
        "bundle_id": None,
        "civitai_version_id": default_vid if isinstance(default_vid, int) else None,
    }

    by_ref[ref] = {
        "ref": ref,
        "source": "civitai",
        "kind": resolved_kind,
        "model_id": model_id if model_id else None,
        "label": model_name,
        "main_url": main_url,
        "default_version_id": default_vid,
        "folder": folder,
        "bundle_id": None,
        "versions": version_rows,
    }

    _add_category_ref(categories, category, ref)
    _rebuild_recommended(categories)

    now = datetime.now(timezone.utc).isoformat()
    index["generated_at"] = now
    index["categories"] = categories
    index["entries"] = index_entries
    entries["generated_at"] = now
    entries["by_ref"] = by_ref
    write_registry_files(index, entries)
    return ref


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one model to model_registry JSON")
    parser.add_argument("--url", required=True, help="Civitai model page or direct download URL")
    parser.add_argument("--category", default="uncategorized", help="Category id, e.g. all_loras")
    parser.add_argument("--kind", default="", help="Optional kind override (lora, checkpoint, …)")
    parser.add_argument("--group", default="", help="Optional models.json group (loras, checkpoints, …)")
    parser.add_argument("--rebuild-all", action="store_true", help="Run full build_registry instead")
    args = parser.parse_args()
    if args.rebuild_all:
        index_payload, entries_payload = rebuild_all(refresh=False, merge_existing=True)
        write_registry(index_payload, entries_payload)
        print(f"rebuilt registry (merge existing versions): {len(index_payload['entries'])} entries")
        return
    ref = append_from_url(
        args.url,
        args.category.strip() or "uncategorized",
        kind=(args.kind.strip() or None),
        group=(args.group.strip() or None),
    )
    print(f"appended {ref} to category {args.category}")


if __name__ == "__main__":
    main()
