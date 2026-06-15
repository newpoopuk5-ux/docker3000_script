"""Read model_registry JSON, match installed files, start registry downloads."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

from download_models import (
    build_item_from_url,
    download_item,
    extract_civitai_file_id,
    extract_civitai_version_id,
    primary_cached_preview_filename,
)
from metadata import list_files
from model_manager import (
    SCAN_DIRS,
    _clear_job_handles,
    _download_kwargs,
    _job_progress,
    _mark_job_cancelled,
    _model_extensions,
    manager_supported,
)
from muse_worker.paths import CATALOG_REGISTRY_DIR

REGISTRY_DIR = CATALOG_REGISTRY_DIR
INDEX_JSON = REGISTRY_DIR / "index.json"
ENTRIES_JSON = REGISTRY_DIR / "entries.json"
ASSETS_DIR = REGISTRY_DIR / "assets"
PREVIEW_OVERRIDES_JSON = REGISTRY_DIR / "preview_overrides.json"

FOLDER_TO_CATALOG_KEY = {
    "checkpoints": "checkpoints",
    "loras": "loras",
    "vae": "vae",
    "text_encoders": "text_encoders",
    "diffusion_models": "flux_diffusion_models",
    "unet": "flux_gguf_unet",
    "controlnet": "controlnet",
    "upscale_models": "upscalers",
}

KINDS = {
    "checkpoint": {"label": "Checkpoint", "folder": "checkpoints"},
    "lora": {"label": "LoRA", "folder": "loras"},
    "vae": {"label": "VAE", "folder": "vae"},
    "text_encoder": {"label": "Text Encoder", "folder": "text_encoders"},
    "flux_diffusion": {"label": "Flux Diffusion", "folder": "diffusion_models"},
    "flux_gguf_unet": {"label": "Flux GGUF UNet", "folder": "unet"},
    "flux_lora": {"label": "Flux LoRA", "folder": "loras"},
    "controlnet": {"label": "ControlNet", "folder": "controlnet"},
    "upscaler": {"label": "Upscaler", "folder": "upscale_models"},
}


def registry_supported() -> bool:
    return INDEX_JSON.is_file() and ENTRIES_JSON.is_file()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_index() -> dict:
    if not INDEX_JSON.is_file():
        return {"ok": False, "supported": False, "error": "model_registry/index.json not found"}
    payload = _read_json(INDEX_JSON)
    payload["ok"] = True
    payload["supported"] = True
    return payload


def load_entries() -> dict:
    if not ENTRIES_JSON.is_file():
        return {"ok": False, "supported": False, "error": "model_registry/entries.json not found"}
    payload = _read_json(ENTRIES_JSON)
    payload["ok"] = True
    payload["supported"] = True
    return payload


_PREVIEW_OVERRIDE_BY_REF: dict[str, str] | None = None
_PREVIEW_OVERRIDE_BY_BUNDLE: dict[str, str] | None = None


def _load_preview_override_maps() -> tuple[dict[str, str], dict[str, str]]:
    global _PREVIEW_OVERRIDE_BY_REF, _PREVIEW_OVERRIDE_BY_BUNDLE
    if _PREVIEW_OVERRIDE_BY_REF is not None and _PREVIEW_OVERRIDE_BY_BUNDLE is not None:
        return _PREVIEW_OVERRIDE_BY_REF, _PREVIEW_OVERRIDE_BY_BUNDLE

    by_ref: dict[str, str] = {}
    by_bundle: dict[str, str] = {}
    if PREVIEW_OVERRIDES_JSON.is_file():
        try:
            payload = _read_json(PREVIEW_OVERRIDES_JSON)
        except (OSError, json.JSONDecodeError):
            payload = {}
        for filename, row in (payload.get("assets") or {}).items():
            if not isinstance(row, dict):
                continue
            safe_name = Path(str(filename)).name
            if not safe_name or not (ASSETS_DIR / safe_name).is_file():
                continue
            for ref in row.get("refs") or []:
                if ref:
                    by_ref[str(ref)] = safe_name
            for bundle_id in row.get("bundle_ids") or []:
                if bundle_id:
                    by_bundle[str(bundle_id)] = safe_name

    _PREVIEW_OVERRIDE_BY_REF = by_ref
    _PREVIEW_OVERRIDE_BY_BUNDLE = by_bundle
    return by_ref, by_bundle


def preview_static_file_for_summary(summary: dict) -> str | None:
    by_ref, by_bundle = _load_preview_override_maps()
    ref = str(summary.get("ref") or "")
    if ref and ref in by_ref:
        return by_ref[ref]
    bundle_id = str(summary.get("bundle_id") or "")
    if bundle_id and bundle_id in by_bundle:
        return by_bundle[bundle_id]
    return None


def apply_preview_override(summary: dict) -> dict:
    static_file = preview_static_file_for_summary(summary)
    if not static_file:
        return summary
    out = dict(summary)
    out["preview_static_file"] = static_file
    return out


def registry_asset_file_path(filename: str) -> Path | None:
    safe_name = Path(str(filename or "")).name
    if not safe_name or safe_name.startswith("."):
        return None
    target = (ASSETS_DIR / safe_name).resolve()
    assets_root = ASSETS_DIR.resolve()
    if not str(target).startswith(str(assets_root)):
        return None
    if not target.is_file():
        return None
    return target


def load_entry(ref: str) -> dict | None:
    if not ENTRIES_JSON.is_file():
        return None
    payload = _read_json(ENTRIES_JSON)
    row = (payload.get("by_ref") or {}).get(ref)
    return row


def _primary_file(version: dict) -> dict | None:
    files = version.get("files") or []
    if not files:
        return None
    return next((row for row in files if row.get("primary")), files[0])


def _file_disk_names(file_row: dict) -> list[str]:
    names: list[str] = []
    for key in ("catalog_filename", "pretty_filename"):
        value = str(file_row.get(key) or "").strip()
        if value:
            names.append(value)
    return names


def match_file_installed(file_row: dict, disk_index: dict[str, list[dict]], entry: dict | None = None) -> dict | None:
    folder = str(file_row.get("folder") or "").strip()
    if not folder and entry:
        kind = str(entry.get("kind") or "")
        folder = str(entry.get("folder") or (KINDS.get(kind) or {}).get("folder") or "checkpoints")
    if not folder:
        folder = "checkpoints"
    names = _file_disk_names(file_row)
    hit = _match_names(names, disk_index.get(folder) or [])
    if hit:
        return {"disk_row": hit, "match_kind": "filename", "folder": folder}
    for alt_folder, disk_rows in disk_index.items():
        if alt_folder == folder:
            continue
        hit = _match_names(names, disk_rows)
        if hit:
            return {"disk_row": hit, "match_kind": "filename", "folder": alt_folder}
    return None


def match_version_installed(
    version: dict,
    disk_index: dict[str, list[dict]] | list[dict],
) -> dict | None:
    if isinstance(disk_index, list):
        disk_index = {"checkpoints": disk_index}
    primary = _primary_file(version)
    if primary:
        hit = match_file_installed(primary, disk_index, None)
        if hit:
            return hit
    names = _version_filenames(version)
    for folder, disk_rows in disk_index.items():
        hit = _match_names(names, disk_rows)
        if hit:
            return {"disk_row": hit, "match_kind": "filename", "folder": folder}
        vid = version.get("version_id")
        try:
            vid_int = int(vid)
        except (TypeError, ValueError):
            vid_int = None
        if vid_int is not None:
            sidecar_hit = next((row for row in disk_rows if row.get("version_id") == vid_int), None)
            if sidecar_hit:
                return {"disk_row": sidecar_hit, "match_kind": "sidecar", "folder": folder}
    return None


def version_has_bundle_files(version: dict) -> bool:
    files = version.get("files") or []
    return len(files) > 1


def _file_install_key(file_row: dict, index: int = 0) -> str:
    fid = file_row.get("file_id")
    if fid is not None:
        return str(fid)
    name = str(file_row.get("catalog_filename") or file_row.get("pretty_filename") or "").strip()
    if name:
        return name
    return str(index)


def version_bundle_installed(version: dict, disk_index: dict[str, list[dict]], entry: dict | None = None) -> bool:
    files = version.get("files") or []
    if not files:
        return False
    if len(files) == 1:
        return match_version_installed(version, disk_index) is not None
    return all(match_file_installed(file_row, disk_index, entry) for file_row in files)


def list_installed_version_ids(entry: dict, disk_index: dict[str, list[dict]] | None = None) -> list:
    disk_index = disk_index if disk_index is not None else _scan_disk_index()
    installed: list = []
    for version in entry.get("versions") or []:
        if match_version_installed(version, disk_index):
            installed.append(version.get("version_id"))
    return installed


def enrich_entry_install(entry: dict, disk_index: dict[str, list[dict]] | None = None) -> dict:
    disk_index = disk_index if disk_index is not None else _scan_disk_index()
    installed_ids: list = []
    version_install: dict[str, dict] = {}
    for version in entry.get("versions") or []:
        vid = version.get("version_id")
        file_status: dict[str, dict] = {}
        primary_hit = None
        for index, file_row in enumerate(version.get("files") or []):
            key = _file_install_key(file_row, index)
            hit = match_file_installed(file_row, disk_index, entry)
            file_status[key] = {
                "installed": hit is not None,
                "disk_name": hit["disk_row"].get("name") if hit else None,
                "disk_path": hit["disk_row"].get("path") if hit else None,
                "folder": hit.get("folder") if hit else file_row.get("folder"),
                "catalog_filename": file_row.get("catalog_filename"),
            }
            if file_row.get("primary") and hit:
                primary_hit = hit
        if not primary_hit and not any(row.get("installed") for row in file_status.values()):
            continue
        installed_ids.append(vid)
        row = primary_hit["disk_row"] if primary_hit else {}
        version_install[str(vid)] = {
            "disk_name": row.get("name") if primary_hit else None,
            "disk_path": row.get("path") if primary_hit else None,
            "size_bytes": row.get("size_bytes") if primary_hit else None,
            "folder": primary_hit.get("folder") if primary_hit else None,
            "bundle_complete": version_bundle_installed(version, disk_index, entry),
            "files": file_status,
        }
    out = dict(entry)
    out["installed_version_ids"] = installed_ids
    if version_install:
        out["version_install"] = version_install
    return out


def load_entries_batch(refs: list[str]) -> dict:
    if not ENTRIES_JSON.is_file():
        return {"ok": False, "supported": False, "error": "model_registry/entries.json not found", "by_ref": {}}
    payload = _read_json(ENTRIES_JSON)
    all_rows = payload.get("by_ref") or {}
    disk_index = _scan_disk_index()
    selected = {ref: enrich_entry_install(all_rows[ref], disk_index) for ref in refs if ref in all_rows}
    return {
        "ok": True,
        "supported": True,
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "by_ref": selected,
    }


def _normalize_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def _read_sidecar_version(path: Path) -> int | None:
    sidecar = path.with_suffix(path.suffix + ".civitai.json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = data.get("id") or data.get("versionId") or data.get("version_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _scan_disk_index() -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {}
    for folder_key, base in SCAN_DIRS.items():
        if not base.exists():
            continue
        recursive = folder_key in ("loras", "text_encoders", "diffusion_models", "unet")
        names = list_files(base, _model_extensions(), recursive=recursive)
        bucket: list[dict] = []
        for name in names:
            target = base / name
            if not target.is_file():
                continue
            bucket.append({
                "folder": folder_key,
                "name": name,
                "path": name,
                "size_bytes": target.stat().st_size,
                "stem_norm": _normalize_stem(name),
                "version_id": _read_sidecar_version(target),
            })
        rows[folder_key] = bucket
    return rows


def _match_names(names: list[str], disk_rows: list[dict]) -> dict | None:
    wanted = {n for n in names if n}
    wanted_stems = {_normalize_stem(n) for n in names if n}
    for row in disk_rows:
        if row["name"] in wanted:
            return row
    for row in disk_rows:
        if row["stem_norm"] in wanted_stems:
            return row
    return None


def _version_filenames(version: dict) -> list[str]:
    names: list[str] = []
    for file_row in version.get("files") or []:
        for key in ("catalog_filename", "pretty_filename"):
            value = str(file_row.get(key) or "").strip()
            if value:
                names.append(value)
    return names


def _resolve_version(entry: dict, version_id) -> dict | None:
    versions = entry.get("versions") or []
    if version_id is None:
        default_id = entry.get("default_version_id")
        for row in versions:
            if row.get("version_id") == default_id:
                return row
        return versions[0] if versions else None
    for row in versions:
        if str(row.get("version_id")) == str(version_id):
            return row
    return None


def match_entry_installed(entry: dict, disk_index: dict[str, list[dict]] | None = None) -> dict:
    folder = str(entry.get("folder") or (KINDS.get(entry.get("kind") or "") or {}).get("folder") or "checkpoints")
    disk_index = disk_index if disk_index is not None else _scan_disk_index()
    disk_rows = disk_index.get(folder) or []

    default_version = _resolve_version(entry, None)
    versions = entry.get("versions") or []
    installed_version = None
    disk_row = None
    match_kind = None

    for version in versions:
        hit = match_version_installed(version, disk_index)
        if hit:
            installed_version = version
            disk_row = hit["disk_row"]
            match_kind = hit["match_kind"]
            break

    if not disk_row and default_version:
        names = _version_filenames(default_version)
        stems = {_normalize_stem(n) for n in names}
        for row in disk_rows:
            if row["stem_norm"] in stems:
                installed_version = default_version
                disk_row = row
                match_kind = "fuzzy"
                break

    installed = disk_row is not None
    out = {
        "installed": installed,
        "missing": not installed,
        "folder": folder,
        "match_kind": match_kind,
        "disk_name": disk_row.get("name") if disk_row else None,
        "disk_path": disk_row.get("path") if disk_row else None,
        "size_bytes": disk_row.get("size_bytes") if disk_row else None,
        "installed_version_id": installed_version.get("version_id") if installed_version else None,
    }
    if installed and entry.get("source") == "civitai" and installed_version:
        vid = installed_version.get("version_id")
        try:
            vid_int = int(vid)
            thumb = primary_cached_preview_filename(vid_int)
            if thumb:
                out["civitai_version_id"] = vid_int
                out["preview_thumb_file"] = thumb
        except (TypeError, ValueError):
            pass
    return out


def merge_index_installed(index: dict, disk_index: dict[str, list[dict]] | None = None) -> dict[str, dict]:
    entries_payload = load_entries()
    by_ref = entries_payload.get("by_ref") or {}
    disk_index = disk_index if disk_index is not None else _scan_disk_index()
    installed_by_ref: dict[str, dict] = {}
    for ref, summary in (index.get("entries") or {}).items():
        entry = by_ref.get(ref) or {"ref": ref, "kind": summary.get("kind"), "folder": (KINDS.get(summary.get("kind") or "") or {}).get("folder")}
        if ref in by_ref:
            entry = by_ref[ref]
        status = match_entry_installed(entry, disk_index)
        installed_by_ref[ref] = status
    return installed_by_ref


def registry_index_payload() -> dict:
    if not registry_supported():
        return {"ok": False, "supported": False, "error": "model registry not found"}
    index = load_index()
    disk_index = _scan_disk_index()
    entries_payload = load_entries()
    by_ref = entries_payload.get("by_ref") or {}
    installed_by_ref = merge_index_installed(index, disk_index)
    entries = dict(index.get("entries") or {})
    for ref, summary in entries.items():
        status = installed_by_ref.get(ref) or {}
        summary = dict(summary)
        summary["installed"] = bool(status.get("installed"))
        summary["disk_name"] = status.get("disk_name")
        summary["disk_path"] = status.get("disk_path")
        summary["installed_version_id"] = status.get("installed_version_id")
        entry_row = by_ref.get(ref)
        if entry_row:
            summary["installed_version_ids"] = list_installed_version_ids(entry_row, disk_index)
        if status.get("folder"):
            summary["folder"] = status["folder"]
        if status.get("size_bytes") is not None:
            summary["size_bytes"] = status["size_bytes"]
        if status.get("civitai_version_id"):
            summary["civitai_version_id"] = status["civitai_version_id"]
        if status.get("preview_thumb_file"):
            summary["preview_thumb_file"] = status["preview_thumb_file"]
        vid = summary.get("civitai_version_id")
        if vid is None:
            default_vid = summary.get("default_version_id")
            try:
                vid = int(default_vid)
                summary["civitai_version_id"] = vid
            except (TypeError, ValueError):
                vid = None
        if vid and not summary.get("preview_thumb_file"):
            thumb = primary_cached_preview_filename(int(vid))
            if thumb:
                summary["preview_thumb_file"] = thumb
        summary["ref"] = ref
        summary = apply_preview_override(summary)
        entries[ref] = summary
    index = dict(index)
    index["entries"] = entries
    index["installed_by_ref"] = installed_by_ref
    index["manager_supported"] = manager_supported()
    return index


def registry_installed_payload() -> dict:
    if not registry_supported():
        return {"ok": False, "supported": False, "error": "model registry not found", "installed_by_ref": {}, "disk": []}
    index = load_index()
    disk_index = _scan_disk_index()
    installed_by_ref = merge_index_installed(index, disk_index)
    disk_rows: list[dict] = []
    refs_installed = []
    for ref, status in installed_by_ref.items():
        if not status.get("installed"):
            continue
        refs_installed.append(ref)
        summary = (index.get("entries") or {}).get(ref) or {}
        disk_rows.append({
            "ref": ref,
            "label": summary.get("label") or ref,
            "kind": summary.get("kind"),
            "folder": status.get("folder"),
            "name": status.get("disk_name"),
            "path": status.get("disk_path"),
            "size_bytes": status.get("size_bytes"),
            "installed_version_id": status.get("installed_version_id"),
            "civitai_version_id": status.get("civitai_version_id"),
            "preview_thumb_file": status.get("preview_thumb_file"),
        })
    return {
        "ok": True,
        "supported": True,
        "installed_by_ref": installed_by_ref,
        "refs_installed": refs_installed,
        "disk": disk_rows,
    }


def _catalog_folder_key(entry: dict) -> str:
    folder = str(entry.get("folder") or "").strip()
    return FOLDER_TO_CATALOG_KEY.get(folder, folder or "checkpoints")


def build_download_item(entry: dict, version: dict, file_row: dict | None = None) -> tuple[dict, str]:
    files = version.get("files") or []
    if not files:
        raise RuntimeError("No files for selected version")
    file_row = file_row or _primary_file(version) or files[0]
    folder_key = str(file_row.get("folder") or _catalog_folder_key(entry))
    folder_key = FOLDER_TO_CATALOG_KEY.get(folder_key, folder_key)
    source = str(entry.get("source") or "civitai")
    if source == "hf":
        item = {
            "source": "hf",
            "repo_id": str(entry.get("repo_id") or file_row.get("repo_id") or ""),
            "repo_path": str(file_row.get("repo_path") or ""),
            "name": str(file_row.get("catalog_filename") or file_row.get("pretty_filename") or Path(file_row.get("repo_path") or "").name),
        }
        if not item["repo_id"] or not item["repo_path"]:
            raise RuntimeError("HF entry missing repo_id/repo_path")
        return item, folder_key

    url = str(file_row.get("download_url") or "").strip()
    if not url:
        raise RuntimeError("Missing download URL")
    pretty = str(file_row.get("pretty_filename") or file_row.get("catalog_filename") or "").strip() or None
    item = build_item_from_url(url, folder_key, name=pretty)
    fid = file_row.get("file_id")
    if fid:
        item["civitai_file_id"] = int(fid)
    vid = version.get("version_id")
    try:
        item["civitai_version_id"] = int(vid)
    except (TypeError, ValueError):
        pass
    return item, folder_key


def build_download_items(
    entry: dict,
    version: dict,
    *,
    bundle: bool = False,
    file_id=None,
) -> list[tuple[dict, str]]:
    files = version.get("files") or []
    if not files:
        raise RuntimeError("No files for selected version")
    if file_id is not None:
        file_row = None
        for index, row in enumerate(files):
            if str(row.get("file_id") or "") == str(file_id):
                file_row = row
                break
            if str(row.get("catalog_filename") or "") == str(file_id):
                file_row = row
                break
            if _file_install_key(row, index) == str(file_id):
                file_row = row
                break
        if not file_row:
            raise RuntimeError(f"Unknown file_id for version: {file_id}")
        hit = match_file_installed(file_row, _scan_disk_index(), entry)
        if hit:
            raise RuntimeError("File already installed")
        return [build_download_item(entry, version, file_row)]
    if not bundle:
        return [build_download_item(entry, version)]
    pending = []
    disk_index = _scan_disk_index()
    for file_row in files:
        if match_file_installed(file_row, disk_index, entry):
            continue
        pending.append(build_download_item(entry, version, file_row))
    if not pending:
        raise RuntimeError("All bundle files already installed")
    return pending


def _run_registry_download_job(job_id: str, ref: str, version_id, bundle: bool = False, file_id=None) -> None:
    kwargs = _download_kwargs(job_id)
    if kwargs["cancel_event"].is_set():
        _mark_job_cancelled(job_id)
        return
    from model_manager import _update_job

    _update_job(job_id, status="running", started_at=time.time(), progress="starting")
    progress = _job_progress(job_id)
    try:
        entry = load_entry(ref)
        if not entry:
            raise RuntimeError(f"Unknown registry ref: {ref}")
        version = _resolve_version(entry, version_id)
        if not version:
            raise RuntimeError(f"Unknown version for {ref}")
        items = build_download_items(entry, version, bundle=bundle, file_id=file_id)
        total = len(items)
        for index, (item, folder_key) in enumerate(items, start=1):
            if kwargs["cancel_event"].is_set():
                _mark_job_cancelled(job_id)
                return
            if total > 1:
                _update_job(job_id, progress=f"file {index}/{total}")
            download_item(item, folder_key, progress=progress, **kwargs)
        if kwargs["cancel_event"].is_set():
            _mark_job_cancelled(job_id)
            return
        _update_job(job_id, status="done", finished_at=time.time(), ok=True, progress="done")
    except Exception as e:
        from model_manager import _update_job

        err = str(e)
        if not err.startswith("Download failed"):
            err = f"Download failed: {err}"
        _update_job(
            job_id,
            status="error",
            finished_at=time.time(),
            ok=False,
            error=err,
            progress="error",
        )
    finally:
        _clear_job_handles(job_id)


def start_registry_download(ref: str, version_id=None, bundle: bool = False, file_id=None) -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "Model downloads are not available on this host."}
    if not registry_supported():
        return {"ok": False, "error": "Model registry not found"}
    ref = (ref or "").strip()
    if not ref:
        return {"ok": False, "error": "ref is required"}
    entry = load_entry(ref)
    if not entry:
        return {"ok": False, "error": f"Unknown registry ref: {ref}"}
    version = _resolve_version(entry, version_id)
    if not version:
        return {"ok": False, "error": "Unknown version"}
    disk_index = _scan_disk_index()
    if file_id is not None:
        files = version.get("files") or []
        target = None
        for index, row in enumerate(files):
            if str(row.get("file_id") or "") == str(file_id):
                target = row
                break
            if str(row.get("catalog_filename") or "") == str(file_id):
                target = row
                break
            if _file_install_key(row, index) == str(file_id):
                target = row
                break
        if target and match_file_installed(target, disk_index, entry):
            return {"ok": True, "already_installed": True, "ref": ref, "version_id": version.get("version_id"), "file_id": file_id}
    elif bundle:
        if version_bundle_installed(version, disk_index, entry):
            return {"ok": True, "already_installed": True, "ref": ref, "version_id": version.get("version_id"), "bundle": True}
    elif match_version_installed(version, disk_index):
        return {"ok": True, "already_installed": True, "ref": ref, "version_id": version.get("version_id")}

    label = str(entry.get("label") or ref)
    if file_id is not None:
        label = f"{label} · {file_id}"
    elif bundle and version_has_bundle_files(version):
        label = f"{label} (bundle)"
    job_id = uuid.uuid4().hex
    from model_manager import _DOWNLOAD_JOBS, _DOWNLOAD_LOCK

    with _DOWNLOAD_LOCK:
        _DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "catalog_id": f"registry:{ref}",
            "display_name": label,
            "source_url": (_primary_file(version) or {}).get("download_url"),
            "status": "queued",
            "ok": None,
            "error": None,
            "progress": "queued",
            "created_at": time.time(),
            "registry_ref": ref,
            "version_id": version.get("version_id"),
            "bundle": bool(bundle),
            "file_id": file_id,
        }
    thread = threading.Thread(
        target=_run_registry_download_job,
        args=(job_id, ref, version.get("version_id"), bundle, file_id),
        daemon=True,
    )
    thread.start()
    return {
        "ok": True,
        "job_id": job_id,
        "ref": ref,
        "version_id": version.get("version_id"),
        "display_name": label,
        "status": "queued",
        "bundle": bool(bundle),
        "file_id": file_id,
    }


def collect_registry_preview_items(include_flux: bool = True) -> list[dict]:
    """Civitai preview prefetch targets from committed model_registry JSON."""
    if not registry_supported():
        return []
    index = load_index()
    entries_payload = load_entries()
    by_ref = entries_payload.get("by_ref") or {}
    recommended_cat = index.get("recommended_category") or "recommended"
    recommended_refs = set((index.get("categories") or {}).get(recommended_cat, {}).get("refs") or [])

    items: list[dict] = []
    seen: set[int] = set()
    flux_kinds = {"flux_diffusion", "flux_gguf_unet", "flux_lora"}

    for ref, summary in (index.get("entries") or {}).items():
        if str(summary.get("source") or "") != "civitai":
            continue
        entry = by_ref.get(ref)
        if not entry:
            continue
        kind = str(entry.get("kind") or summary.get("kind") or "")
        if not include_flux and kind in flux_kinds:
            continue
        default_vid = entry.get("default_version_id") or summary.get("default_version_id")
        version = _resolve_version(entry, default_vid)
        if not version:
            continue
        try:
            vid = int(version.get("version_id"))
        except (TypeError, ValueError):
            continue
        if vid in seen:
            continue
        seen.add(vid)
        items.append({
            "version_id": vid,
            "name": str(entry.get("label") or summary.get("label") or ref),
            "ref": ref,
            "recommended": ref in recommended_refs,
        })
    return items


def write_registry_files(index_payload: dict, entries_payload: dict) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    index_clean = {k: v for k, v in index_payload.items() if k not in ("ok", "supported", "installed_by_ref")}
    entries_clean = {k: v for k, v in entries_payload.items() if k not in ("ok", "supported")}
    INDEX_JSON.write_text(json.dumps(index_clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ENTRIES_JSON.write_text(json.dumps(entries_clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
