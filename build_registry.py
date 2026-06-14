"""Build model_registry/index.json + entries.json from civitai_model_roots.json and HF rows in models.json."""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from download_models import (
    civitai_lookup,
    civitai_pretty_filename,
    extract_civitai_file_id,
    resolve_civitai_file,
    _pick_primary_civitai_preview,
)

ROOTS_JSON = Path(__file__).resolve().parent / "civitai_model_roots.json"
MODELS_JSON = Path(__file__).resolve().parent / "models.json"
REGISTRY_DIR = Path(__file__).resolve().parent / "model_registry"
INDEX_JSON = REGISTRY_DIR / "index.json"
ENTRIES_JSON = REGISTRY_DIR / "entries.json"

SCHEMA_VERSION = 1

KINDS = {
    "checkpoint": {"label": "Checkpoint", "folder": "checkpoints", "tab": "models"},
    "lora": {"label": "LoRA", "folder": "loras", "tab": "loras"},
    "vae": {"label": "VAE", "folder": "vae", "tab": "vae"},
    "text_encoder": {"label": "Text Encoder", "folder": "text_encoders", "tab": "te"},
    "flux_diffusion": {"label": "Flux Diffusion", "folder": "diffusion_models", "tab": "flux"},
    "flux_gguf_unet": {"label": "Flux GGUF UNet", "folder": "unet", "tab": "flux"},
    "flux_lora": {"label": "Flux LoRA", "folder": "loras", "tab": "flux"},
    "controlnet": {"label": "ControlNet", "folder": "controlnet", "tab": "controlnet"},
    "upscaler": {"label": "Upscaler", "folder": "upscale_models", "tab": "upscalers"},
}

GROUP_TO_KIND = {
    "checkpoints": "checkpoint",
    "loras": "lora",
    "vae": "vae",
    "text_encoders": "text_encoder",
    "text_encoders_gguf": "text_encoder",
    "flux_diffusion_models": "flux_diffusion",
    "flux_gguf_unet": "flux_gguf_unet",
    "flux_loras": "flux_lora",
    "upscalers": "upscaler",
    "controlnet": "controlnet",
}

CIVITAI_TYPE_TO_KIND = {
    "Checkpoint": "checkpoint",
    "LORA": "lora",
    "LoCon": "lora",
    "LyCORIS": "lora",
    "VAE": "vae",
    "TextualInversion": "text_encoder",
    "Controlnet": "controlnet",
    "Upscaler": "upscaler",
    "Hypernetwork": "lora",
    "AestheticGradient": "lora",
    "Poses": "other",
    "Wildcards": "other",
    "Workflows": "other",
    "Other": "other",
}

CATALOG_GROUPS = list(GROUP_TO_KIND.keys())

HF_SETS = ("anima", "miaomiao")

CATEGORY_META = {
    "anime_starter": "Anime starter checkpoints",
    "anima": "Anima Base (HF UNET + TE + VAE)",
    "miaomiao": "MiaoMiao Harem Anima 1.2 bundle",
    "all_models": "All checkpoints",
    "all_loras": "All LoRAs",
    "all_vae": "All VAE",
    "all_te": "All text encoders",
    "uncategorized": "Uncategorized",
}


def _preview_url_from_meta(meta: dict) -> str | None:
    picked = _pick_primary_civitai_preview(meta)
    return (picked or {}).get("url")

RECOMMENDED_CATEGORIES = ("anime_starter", "anima", "miaomiao")


def _slug(text: str, limit: int = 48) -> str:
    raw = re.sub(r"[^\w\s-]+", "", str(text or ""), flags=re.UNICODE).strip().lower()
    raw = re.sub(r"[\s_]+", "-", raw)
    return (raw[:limit] or "item").strip("-") or "item"


def _kind_for_group(group: str, model_type: str = "") -> str:
    if group in GROUP_TO_KIND:
        return GROUP_TO_KIND[group]
    return CIVITAI_TYPE_TO_KIND.get(model_type or "", "checkpoint")


def _kind_label(kind: str) -> str:
    return (KINDS.get(kind) or {}).get("label") or kind


def _base_model_tags(base_model: str) -> list[str]:
    raw = (base_model or "").strip()
    if not raw:
        return []
    tags = [_slug(raw, 32)]
    lower = raw.lower()
    if "illustrious" in lower:
        tags.append("illustrious")
    if "pony" in lower:
        tags.append("pony")
    if "sdxl" in lower:
        tags.append("sdxl")
    if "flux" in lower:
        tags.append("flux")
    if "anima" in lower:
        tags.append("anima")
    return list(dict.fromkeys(tags))


def _infer_tags(label: str, kind: str, base_model: str) -> list[str]:
    tags = _base_model_tags(base_model)
    tags.append(_slug(_kind_label(kind), 24))
    for token in re.split(r"[\s_\-]+", label.lower()):
        if len(token) >= 4 and token not in tags:
            tags.append(token)
    return tags[:8]


def _file_id_from_url(url: str) -> int | None:
    fid = extract_civitai_file_id(url or "")
    if fid:
        return int(fid)
    return None


def _version_file_row(
    *,
    version_id: int,
    version_name: str,
    base_model: str,
    catalog_name: str,
    download_url: str,
    model_name: str,
    file_id: int | None = None,
    size_bytes: int | None = None,
    preview_remote_url: str | None = None,
) -> dict:
    fid = file_id if file_id is not None else _file_id_from_url(download_url)
    pretty = civitai_pretty_filename(model_name, version_name, catalog_name)
    dl = download_url.strip()
    if fid and "fileId=" not in dl:
        sep = "&" if "?" in dl else "?"
        dl = f"{dl}{sep}fileId={fid}"
    return {
        "version_id": int(version_id),
        "name": version_name or f"v{version_id}",
        "base_model": base_model or "",
        "preview_remote_url": preview_remote_url,
        "files": [
            {
                "file_id": fid,
                "download_url": dl,
                "catalog_filename": catalog_name,
                "pretty_filename": pretty,
                "size_bytes": size_bytes,
                "primary": True,
            }
        ],
    }


def _split_model_refs(model: dict) -> list[tuple[str, list[dict]]]:
    """Return [(ref, catalog_entries_subset)] for one civitai roots model row."""
    model_id = int(model.get("model_id") or 0)
    entries = list(model.get("catalog_entries") or [])
    if not model_id or not entries:
        return []

    groups = {str(e.get("group") or "") for e in entries}
    if len(groups) <= 1:
        return [(f"civitai:{model_id}", entries)]

    by_sig: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        sig = (str(entry.get("group") or ""), str(entry.get("catalog_name") or ""))
        by_sig.setdefault(sig, []).append(entry)

    rows: list[tuple[str, list[dict]]] = []
    for (group, catalog_name), subset in sorted(by_sig.items(), key=lambda row: (row[0][0], row[0][1])):
        fid = _file_id_from_url(subset[0].get("download_url") or "")
        suffix = f"f{fid}" if fid else _slug(catalog_name, 32)
        ref = f"civitai:{model_id}:{suffix}"
        rows.append((ref, subset))
    return rows


def _aggregate_versions(
    model: dict,
    catalog_subset: list[dict],
    *,
    refresh: bool,
    version_meta_cache: dict[int, dict],
) -> list[dict]:
    model_name = str(model.get("model_name") or "")
    model_id = int(model.get("model_id") or 0)
    by_vid: dict[int, dict] = {}

    for entry in catalog_subset:
        vid = int(entry.get("version_id") or 0)
        if not vid:
            continue
        catalog_name = str(entry.get("catalog_name") or "")
        version_name = str(entry.get("version_name") or "")
        base_model = str(entry.get("base_model") or "")
        download_url = str(entry.get("download_url") or "")
        file_id = _file_id_from_url(download_url)
        size_bytes = None
        preview_remote_url = None

        if refresh:
            if vid not in version_meta_cache:
                version_meta_cache[vid] = civitai_lookup(vid)
                time.sleep(0.12)
            meta = version_meta_cache[vid]
            file_entry = resolve_civitai_file(meta, file_id)
            catalog_name = str(file_entry.get("name") or catalog_name)
            version_name = str(meta.get("name") or version_name)
            base_model = str(meta.get("baseModel") or base_model)
            fid = file_entry.get("id")
            if fid:
                file_id = int(fid)
            size_bytes = file_entry.get("size") or file_entry.get("sizeKB")
            if isinstance(size_bytes, (int, float)) and size_bytes < 10_000_000:
                size_bytes = int(float(size_bytes) * 1024) if size_bytes else None
            elif isinstance(size_bytes, (int, float)):
                size_bytes = int(size_bytes)
            preview_remote_url = _preview_url_from_meta(meta)
            download_url = (
                f"https://civitai.com/api/download/models/{vid}?fileId={file_id}"
                if file_id
                else f"https://civitai.com/api/download/models/{vid}"
            )

        row = _version_file_row(
            version_id=vid,
            version_name=version_name,
            base_model=base_model,
            catalog_name=catalog_name,
            download_url=download_url,
            model_name=model_name,
            file_id=file_id,
            size_bytes=size_bytes,
            preview_remote_url=preview_remote_url,
        )
        existing = by_vid.get(vid)
        if not existing or int(row["files"][0].get("file_id") or 0) == int(file_id or 0):
            by_vid[vid] = row

    versions = sorted(by_vid.values(), key=lambda v: int(v.get("version_id") or 0), reverse=True)
    if refresh and model_id and versions:
        try:
            if model_id not in version_meta_cache:
                pass
            url = f"https://civitai.com/api/v1/models/{model_id}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            full = r.json()
            time.sleep(0.12)
            known_vids = {int(v["version_id"]) for v in versions}
            for v in full.get("modelVersions") or []:
                if not isinstance(v, dict):
                    continue
                v_id = int(v.get("id") or 0)
                if not v_id or v_id in known_vids:
                    continue
                if v_id not in version_meta_cache:
                    version_meta_cache[v_id] = civitai_lookup(v_id)
                    time.sleep(0.12)
                meta = version_meta_cache[v_id]
                file_entry = resolve_civitai_file(meta)
                fid = file_entry.get("id")
                versions.append(
                    _version_file_row(
                        version_id=v_id,
                        version_name=str(meta.get("name") or v.get("name") or ""),
                        base_model=str(meta.get("baseModel") or v.get("baseModel") or ""),
                        catalog_name=str(file_entry.get("name") or f"civitai_{v_id}.safetensors"),
                        download_url=(
                            f"https://civitai.com/api/download/models/{v_id}?fileId={fid}"
                            if fid
                            else f"https://civitai.com/api/download/models/{v_id}"
                        ),
                        model_name=model_name,
                        file_id=int(fid) if fid else None,
                        size_bytes=int(file_entry.get("size") or 0) or None,
                        preview_remote_url=static_preview_url_from_version_meta(meta),
                    )
                )
            versions.sort(key=lambda row: int(row.get("version_id") or 0), reverse=True)
        except Exception:
            pass
    return versions


def _hf_ref(repo_id: str, repo_path: str) -> str:
    stem = Path(repo_path).stem
    return f"hf:{_slug(repo_id, 40)}:{_slug(stem, 40)}"


def _scan_hf_sets(cfg: dict) -> list[dict]:
    rows: list[dict] = []
    for set_name in HF_SETS:
        selected = (cfg.get("sets") or {}).get(set_name) or {}
        for group in CATALOG_GROUPS:
            for item in selected.get(group, []) or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("source") or "").lower() != "hf":
                    continue
                repo_id = str(item.get("repo_id") or "").strip()
                repo_path = str(item.get("repo_path") or "").strip()
                name = str(item.get("name") or Path(repo_path).name)
                if not repo_id or not repo_path:
                    continue
                rows.append({
                    "set": set_name,
                    "group": group,
                    "name": name,
                    "repo_id": repo_id,
                    "repo_path": repo_path,
                })
    return rows


def build(*, refresh: bool = False) -> tuple[dict, dict]:
    roots = json.loads(ROOTS_JSON.read_text(encoding="utf-8"))
    cfg = json.loads(MODELS_JSON.read_text(encoding="utf-8")) if MODELS_JSON.is_file() else {"sets": {}}

    version_meta_cache: dict[int, dict] = {}
    categories: dict[str, dict] = {}
    index_entries: dict[str, dict] = {}
    by_ref: dict[str, dict] = {}

    def add_category_ref(category: str, ref: str) -> None:
        if category not in categories:
            categories[category] = {
                "label": category.replace("_", " "),
                "description": CATEGORY_META.get(category, ""),
                "refs": [],
            }
        if ref not in categories[category]["refs"]:
            categories[category]["refs"].append(ref)

    for model in roots.get("models") or []:
        if not isinstance(model, dict):
            continue
        model_id = int(model.get("model_id") or 0)
        model_name = str(model.get("model_name") or f"model_{model_id}")
        model_type = str(model.get("model_type") or "")
        main_url = str(model.get("main_url") or f"https://civitai.com/models/{model_id}")

        for ref, subset in _split_model_refs(model):
            primary_group = str(subset[0].get("group") or "checkpoints")
            kind = _kind_for_group(primary_group, model_type)
            versions = _aggregate_versions(
                model,
                subset,
                refresh=refresh,
                version_meta_cache=version_meta_cache,
            )
            if not versions:
                continue

            default_version = versions[0]
            default_vid = int(default_version["version_id"])
            default_base = str(default_version.get("base_model") or "")
            preview = default_version.get("preview_remote_url")
            label = model_name
            if len(_split_model_refs(model)) > 1:
                catalog_name = str(subset[0].get("catalog_name") or "")
                if catalog_name and catalog_name.lower() not in label.lower():
                    label = f"{model_name} · {Path(catalog_name).stem}"

            bundle_id = None
            for entry in subset:
                if entry.get("set") in ("anima", "miaomiao"):
                    bundle_id = str(entry.get("set"))

            index_entries[ref] = {
                "ref": ref,
                "source": "civitai",
                "kind": kind,
                "label": label,
                "model_id": model_id,
                "main_url": main_url,
                "default_version_id": default_vid,
                "default_base_model": default_base,
                "preview_remote_url": preview,
                "version_count": len(versions),
                "tags": _infer_tags(label, kind, default_base),
                "bundle_id": bundle_id,
            }
            if isinstance(default_vid, int):
                index_entries[ref]["civitai_version_id"] = default_vid

            by_ref[ref] = {
                "ref": ref,
                "source": "civitai",
                "kind": kind,
                "model_id": model_id,
                "label": label,
                "main_url": main_url,
                "default_version_id": default_vid,
                "folder": (KINDS.get(kind) or {}).get("folder") or "checkpoints",
                "bundle_id": bundle_id,
                "versions": versions,
            }

            for entry in subset:
                add_category_ref(str(entry.get("set") or "uncategorized"), ref)

    for hf_row in _scan_hf_sets(cfg):
        ref = _hf_ref(hf_row["repo_id"], hf_row["repo_path"])
        kind = _kind_for_group(hf_row["group"])
        name = hf_row["name"]
        label = name
        if hf_row["set"] == "anima":
            label = f"Anima · {Path(name).stem}"

        version_row = {
            "version_id": "v1",
            "name": "v1",
            "base_model": "Flux" if kind.startswith("flux") else "",
            "preview_remote_url": None,
            "files": [
                {
                    "repo_id": hf_row["repo_id"],
                    "repo_path": hf_row["repo_path"],
                    "catalog_filename": name,
                    "pretty_filename": name,
                    "size_bytes": None,
                    "primary": True,
                }
            ],
        }

        index_entries[ref] = {
            "ref": ref,
            "source": "hf",
            "kind": kind,
            "label": label,
            "default_version_id": "v1",
            "default_base_model": version_row["base_model"],
            "preview_remote_url": None,
            "version_count": 1,
            "tags": _infer_tags(label, kind, version_row["base_model"]),
            "bundle_id": hf_row["set"],
        }

        by_ref[ref] = {
            "ref": ref,
            "source": "hf",
            "kind": kind,
            "label": label,
            "default_version_id": "v1",
            "folder": (KINDS.get(kind) or {}).get("folder") or "checkpoints",
            "repo_id": hf_row["repo_id"],
            "bundle_id": hf_row["set"],
            "versions": [version_row],
        }
        add_category_ref(hf_row["set"], ref)

    recommended_refs: list[str] = []
    for cat in RECOMMENDED_CATEGORIES:
        for ref in (categories.get(cat) or {}).get("refs") or []:
            if ref not in recommended_refs:
                recommended_refs.append(ref)

    if "recommended" not in categories:
        categories["recommended"] = {
            "label": "Recommended",
            "description": "Anime starter + Anima + MiaoMiao bundles",
            "refs": recommended_refs,
        }

    index_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_roots": str(ROOTS_JSON.name),
        "recommended_category": "recommended",
        "kinds": KINDS,
        "categories": categories,
        "entries": index_entries,
    }

    entries_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": index_payload["generated_at"],
        "by_ref": by_ref,
    }
    return index_payload, entries_payload


def write_registry(index_payload: dict, entries_payload: dict) -> None:
    from model_registry import write_registry_files

    write_registry_files(index_payload, entries_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model_registry JSON from civitai_model_roots.json")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch Civitai API for sizes, previews, and extra versions (slow, PC only)",
    )
    args = parser.parse_args()
    index_payload, entries_payload = build(refresh=bool(args.refresh))
    write_registry(index_payload, entries_payload)
    print(
        f"wrote {INDEX_JSON.relative_to(REGISTRY_DIR.parent)}: "
        f"{len(index_payload['entries'])} entries, "
        f"{len(index_payload['categories'])} categories"
    )
    print(f"wrote {ENTRIES_JSON.relative_to(REGISTRY_DIR.parent)}")


if __name__ == "__main__":
    main()
