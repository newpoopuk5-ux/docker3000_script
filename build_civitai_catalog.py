"""Build civitai_catalog.json from Civitai download URLs in models.json."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from download_models import (
    civitai_lookup,
    extract_civitai_file_id,
    extract_civitai_version_id,
    resolve_civitai_file,
)

MODELS_JSON = Path(__file__).resolve().parent / "models.json"
OUTPUT_JSON = Path(__file__).resolve().parent / "civitai_catalog.json"
CATALOG_GROUPS = [
    "upscalers",
    "checkpoints",
    "loras",
    "vae",
    "flux_diffusion_models",
    "flux_gguf_unet",
    "text_encoders",
    "text_encoders_gguf",
    "flux_loras",
    "auto",
]


def _slug(text: str) -> str:
    raw = re.sub(r"[^\w\s-]+", "", str(text or ""), flags=re.UNICODE).strip().lower()
    raw = re.sub(r"[\s_]+", "-", raw)
    return (raw[:56] or "version").strip("-") or "version"


def _preview_name(meta: dict, filename: str) -> str:
    vid = int(meta.get("id") or 0)
    return f"{vid}__{_slug(meta.get('name') or 'version')}__{Path(filename).stem}.jpeg"


def _scan_models_json() -> list[dict]:
    cfg = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    refs: list[dict] = []
    for set_name, selected in (cfg.get("sets") or {}).items():
        if not isinstance(selected, dict):
            continue
        for group in CATALOG_GROUPS:
            for item in selected.get(group, []) or []:
                if not isinstance(item, dict):
                    continue
                url = (item.get("url") or "").strip()
                if "civitai." not in url:
                    continue
                version_id = extract_civitai_version_id(url)
                if not version_id:
                    continue
                refs.append({
                    "set": set_name,
                    "group": group,
                    "catalog_name": item.get("name") or "",
                    "url": url,
                    "version_id": int(version_id),
                    "file_id": extract_civitai_file_id(url),
                })
    return refs


def _fetch_model(model_id: int, cache: dict[int, dict]) -> dict:
    if model_id in cache:
        return cache[model_id]
    url = f"https://civitai.com/api/v1/models/{model_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache[model_id] = data
    time.sleep(0.15)
    return data


def build() -> dict:
    refs = _scan_models_json()
    version_meta_cache: dict[int, dict] = {}
    model_cache: dict[int, dict] = {}
    by_model: dict[int, dict] = {}

    for ref in refs:
        vid = int(ref["version_id"])
        if vid not in version_meta_cache:
            version_meta_cache[vid] = civitai_lookup(vid)
            time.sleep(0.15)
        ver = version_meta_cache[vid]
        model_id = int(ver.get("modelId") or (ver.get("model") or {}).get("id") or 0)
        if not model_id:
            continue
        if model_id not in by_model:
            full = _fetch_model(model_id, model_cache)
            model_info = full.get("model") or {}
            by_model[model_id] = {
                "model_id": model_id,
                "model_name": full.get("name") or model_info.get("name") or f"model_{model_id}",
                "model_type": full.get("type") or model_info.get("type") or "",
                "page_url": f"https://civitai.com/models/{model_id}",
                "versions": [],
            }
        bucket = by_model[model_id]
        if not bucket["versions"]:
            full = model_cache[model_id]
            for v in full.get("modelVersions") or []:
                if not isinstance(v, dict):
                    continue
                v_id = int(v.get("id") or 0)
                if not v_id:
                    continue
                if v_id not in version_meta_cache:
                    version_meta_cache[v_id] = civitai_lookup(v_id)
                    time.sleep(0.15)
                vmeta = version_meta_cache[v_id]
                file_entry = resolve_civitai_file(vmeta)
                fid = file_entry.get("id")
                fname = str(file_entry.get("name") or "")
                dl = f"https://civitai.com/api/download/models/{v_id}?fileId={fid}" if fid else f"https://civitai.com/api/download/models/{v_id}"
                images = vmeta.get("images") or v.get("images") or []
                thumb = None
                if images and isinstance(images[0], dict):
                    thumb = images[0].get("url")
                bucket["versions"].append({
                    "version_id": v_id,
                    "version_name": vmeta.get("name") or v.get("name") or "",
                    "base_model": vmeta.get("baseModel") or v.get("baseModel") or "",
                    "file_id": fid,
                    "filename": fname,
                    "download_url": dl,
                    "page_url": f"https://civitai.com/models/{model_id}?modelVersionId={v_id}",
                    "preview_thumb": thumb,
                    "preview_cache_name": _preview_name(vmeta, fname or f"civitai_{v_id}.safetensors"),
                    "selected": False,
                    "catalog_refs": [],
                })

    selected_ids = {int(r["version_id"]) for r in refs}
    ref_map: dict[int, list[dict]] = {}
    for ref in refs:
        ref_map.setdefault(int(ref["version_id"]), []).append({
            "set": ref["set"],
            "group": ref["group"],
            "catalog_name": ref["catalog_name"],
            "url": ref["url"],
        })

    for bucket in by_model.values():
        for ver in bucket["versions"]:
            vid = int(ver["version_id"])
            if vid in selected_ids:
                ver["selected"] = True
                ver["catalog_refs"] = ref_map.get(vid, [])

    models = sorted(by_model.values(), key=lambda row: (row.get("model_type") or "", row.get("model_name") or ""))
    for bucket in models:
        bucket["versions"].sort(key=lambda v: int(v.get("version_id") or 0), reverse=True)
        bucket["selected_count"] = sum(1 for v in bucket["versions"] if v.get("selected"))
        bucket["version_count"] = len(bucket["versions"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "models.json",
        "model_count": len(models),
        "selected_version_count": len(selected_ids),
        "models": models,
    }


def main() -> None:
    payload = build()
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT_JSON.name}: {payload['model_count']} models, "
        f"{payload['selected_version_count']} selected versions"
    )


if __name__ == "__main__":
    main()
