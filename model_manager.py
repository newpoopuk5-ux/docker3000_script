import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    COMFY_ROOT,
    DIFFUSION_MODEL_DIR,
    LORA_DIR,
    PRESETS_PATH,
    TEXT_ENCODER_DIR,
    UPSCALE_DIR,
    VAE_DIR,
)
from download_models import (
    TARGET_DIRS,
    DownloadCancelled,
    build_item_from_url,
    download_item,
    preview_civitai_url,
    redact_download_secrets,
)
from metadata import list_files

MODELS_JSON = Path(os.environ.get("MODELS_JSON", "models.json"))

CATALOG_FOLDER_TO_DELETE_KEY = {
    "upscalers": "upscale_models",
    "checkpoints": "checkpoints",
    "loras": "loras",
    "vae": "vae",
    "flux_diffusion_models": "diffusion_models",
    "flux_gguf_unet": "diffusion_models",
    "text_encoders": "text_encoders",
    "text_encoders_gguf": "text_encoders",
    "flux_loras": "loras",
    "auto": "checkpoints",
}

ALLOWED_DELETE_DIRS: dict[str, Path] = {
    "checkpoints": CHECKPOINT_DIR,
    "loras": LORA_DIR,
    "vae": VAE_DIR,
    "controlnet": COMFY_ROOT / "models" / "controlnet",
    "upscale_models": UPSCALE_DIR,
    "text_encoders": TEXT_ENCODER_DIR,
    "diffusion_models": DIFFUSION_MODEL_DIR,
}

SCAN_DIRS: dict[str, Path] = {
    **ALLOWED_DELETE_DIRS,
    "unet": TARGET_DIRS["flux_gguf_unet"],
}

CATALOG_GROUPS = [
    ("upscalers", "upscalers"),
    ("checkpoints", "checkpoints"),
    ("loras", "loras"),
    ("vae", "vae"),
    ("flux_diffusion_models", "flux_diffusion_models"),
    ("flux_gguf_unet", "flux_gguf_unet"),
    ("text_encoders", "text_encoders"),
    ("text_encoders_gguf", "text_encoders_gguf"),
    ("flux_loras", "flux_loras"),
    ("auto", "auto"),
]

_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_JOBS: dict[str, dict] = {}
_JOB_CANCEL: dict[str, threading.Event] = {}
_JOB_PROCS: dict[str, subprocess.Popen | None] = {}

DEFAULT_CATALOG_SET = "basic_no_flux"


def is_flux_set(name: str) -> bool:
    return str(name or "").strip().lower().startswith("flux_")


def recommended_set_name(cfg: dict) -> str:
    name = str(cfg.get("recommended_set") or DEFAULT_CATALOG_SET).strip()
    if name in (cfg.get("sets") or {}):
        return name
    if DEFAULT_CATALOG_SET in (cfg.get("sets") or {}):
        return DEFAULT_CATALOG_SET
    return next(iter((cfg.get("sets") or {}).keys()), DEFAULT_CATALOG_SET)


def manager_supported() -> bool:
    return MODELS_JSON.is_file()


def _model_extensions() -> tuple[str, ...]:
    return (".safetensors", ".ckpt", ".pt", ".bin", ".pth", ".gguf")


def _relative_within(base: Path, path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError:
        return None
    return rel.as_posix()


def _install_base_for_folder_key(folder_key: str):
    if folder_key in TARGET_DIRS:
        return TARGET_DIRS[folder_key]
    delete_key = CATALOG_FOLDER_TO_DELETE_KEY.get(folder_key, folder_key)
    return ALLOWED_DELETE_DIRS.get(delete_key)


def _installed_entries() -> list[dict]:
    entries: list[dict] = []
    for folder_key, base in SCAN_DIRS.items():
        if not base.exists():
            continue
        recursive = folder_key in ("loras", "text_encoders", "diffusion_models", "unet")
        names = list_files(base, _model_extensions(), recursive=recursive)
        for name in names:
            target = base / name
            if not target.is_file():
                continue
            entries.append({
                "id": f"{folder_key}:{name}",
                "folder": folder_key,
                "name": name,
                "path": name,
                "size_bytes": target.stat().st_size,
                "installed": True,
            })
    return entries


def _catalog_item_id(set_name: str, folder_key: str, item: dict, index: int) -> str:
    name = item.get("name") or item.get("repo_id") or f"item_{index}"
    return f"{set_name}:{folder_key}:{name}"


def _target_name_for_item(folder_key: str, item: dict) -> str:
    if item.get("name"):
        return str(item["name"])
    if item.get("target"):
        return Path(str(item["target"])).name
    if item.get("repo_path"):
        return Path(str(item["repo_path"])).name
    return "model.bin"


def _is_installed(folder_key: str, item: dict) -> bool:
    if item.get("target"):
        target = Path(str(item["target"]))
        if target.is_file() and target.stat().st_size > 1024 * 1024:
            return True
    base = _install_base_for_folder_key(folder_key)
    if not base:
        return False
    name = _target_name_for_item(folder_key, item)
    candidate = base / name
    if candidate.is_file() and candidate.stat().st_size > 1024 * 1024:
        return True
    if folder_key in ("loras", "flux_loras"):
        nested = (TARGET_DIRS.get("flux_loras") or base) / name
        if nested.is_file() and nested.stat().st_size > 1024 * 1024:
            return True
    return False


def _catalog_rows_for_set(cfg: dict, set_name: str) -> list[dict]:
    rows: list[dict] = []
    selected = (cfg.get("sets") or {}).get(set_name) or {}
    for group_key, folder_key in CATALOG_GROUPS:
        for index, item in enumerate(selected.get(group_key, []) or []):
            if not isinstance(item, dict):
                continue
            cid = _catalog_item_id(set_name, folder_key, item, index)
            delete_key = CATALOG_FOLDER_TO_DELETE_KEY.get(folder_key, folder_key)
            rows.append({
                "id": cid,
                "set": set_name,
                "group": group_key,
                "folder": delete_key,
                "name": _target_name_for_item(folder_key, item),
                "source": item.get("source", "direct"),
                "installed": _is_installed(folder_key, item),
                "recommended": True,
            })
    return rows


def load_catalog(set_name: str | None = None, include_flux: bool = False) -> dict:
    if not MODELS_JSON.is_file():
        return {"ok": False, "supported": False, "error": "models.json not found", "sets": {}, "catalog": []}

    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    recommended = recommended_set_name(cfg)
    sets_info = {}
    for key, value in (cfg.get("sets") or {}).items():
        experimental = is_flux_set(key)
        sets_info[key] = {
            "description": (value or {}).get("description", ""),
            "experimental": experimental,
            "recommended": key == recommended,
            "hidden_by_default": experimental,
        }

    active_set = set_name or (os.environ.get("MODEL_SET") or recommended)
    merge_all = active_set in ("*", "__all__")
    if merge_all:
        active_set = "__all__"
        catalog: list[dict] = []
        seen: set[str] = set()
        for name in cfg.get("sets", {}):
            if not include_flux and is_flux_set(name):
                continue
            for row in _catalog_rows_for_set(cfg, name):
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                catalog.append(row)
    else:
        if active_set not in cfg.get("sets", {}):
            active_set = recommended
        catalog = _catalog_rows_for_set(cfg, active_set)

    installed = _installed_entries()
    missing = [row for row in catalog if not row["installed"]]
    installed_catalog = [row for row in catalog if row["installed"]]

    presets = {}
    if PRESETS_PATH.is_file():
        try:
            presets = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            presets = {}

    return {
        "ok": True,
        "supported": True,
        "recommended_set": recommended,
        "active_set": active_set,
        "include_flux": include_flux,
        "sets": sets_info,
        "catalog": catalog,
        "installed": installed,
        "installed_catalog": installed_catalog,
        "missing": missing,
        "presets": presets,
    }


def _find_catalog_item(catalog_id: str) -> tuple[str, str, dict] | None:
    if not MODELS_JSON.is_file():
        return None
    parts = catalog_id.split(":", 2)
    if len(parts) != 3:
        return None
    set_name, folder_key, name = parts
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    selected = (cfg.get("sets") or {}).get(set_name)
    if not selected:
        return None
    for index, item in enumerate(selected.get(folder_key, []) or []):
        if _catalog_item_id(set_name, folder_key, item, index) == catalog_id:
            return set_name, folder_key, item
        if _target_name_for_item(folder_key, item) == name:
            return set_name, folder_key, item
    return None


def _update_job(job_id: str, **fields) -> None:
    with _DOWNLOAD_LOCK:
        job = _DOWNLOAD_JOBS.get(job_id)
        if job:
            job.update(fields)


def _job_cancel_event(job_id: str) -> threading.Event:
    with _DOWNLOAD_LOCK:
        ev = _JOB_CANCEL.get(job_id)
        if ev is None:
            ev = threading.Event()
            _JOB_CANCEL[job_id] = ev
        return ev


def _register_job_proc(job_id: str, proc: subprocess.Popen) -> None:
    with _DOWNLOAD_LOCK:
        _JOB_PROCS[job_id] = proc


def _clear_job_handles(job_id: str) -> None:
    with _DOWNLOAD_LOCK:
        _JOB_CANCEL.pop(job_id, None)
        _JOB_PROCS.pop(job_id, None)


def _kill_job_proc(job_id: str) -> None:
    with _DOWNLOAD_LOCK:
        proc = _JOB_PROCS.get(job_id)
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _mark_job_cancelled(job_id: str) -> None:
    _update_job(
        job_id,
        status="cancelled",
        finished_at=time.time(),
        ok=False,
        error="Cancelled by user",
        progress="cancelled",
    )
    _clear_job_handles(job_id)


def _download_kwargs(job_id: str) -> dict:
    cancel = _job_cancel_event(job_id)
    return {
        "cancel_event": cancel,
        "proc_cb": lambda proc: _register_job_proc(job_id, proc),
    }


def _job_progress(job_id: str):
    cancel = _job_cancel_event(job_id)

    def progress(msg: str, **stats) -> None:
        if cancel.is_set():
            raise DownloadCancelled()
        fields: dict = {"progress": msg}
        for key in (
            "progress_pct",
            "progress_done",
            "progress_total",
            "progress_speed",
            "progress_eta",
            "progress_connections",
        ):
            if key in stats and stats[key] is not None:
                fields[key] = stats[key]
        _update_job(job_id, **fields)

    return progress


def _run_url_download_job(job_id: str, url: str, display_name: str, folder_key: str | None = None) -> None:
    kwargs = _download_kwargs(job_id)
    if kwargs["cancel_event"].is_set():
        _mark_job_cancelled(job_id)
        return
    _update_job(job_id, status="running", started_at=time.time(), progress="starting")
    progress = _job_progress(job_id)

    try:
        item = build_item_from_url(url, folder_key)
        download_item(item, "auto", progress=progress, **kwargs)
        if kwargs["cancel_event"].is_set():
            _mark_job_cancelled(job_id)
            return
        _update_job(job_id, status="done", finished_at=time.time(), ok=True, progress="done")
    except DownloadCancelled:
        _mark_job_cancelled(job_id)
    except Exception as e:
        err = redact_download_secrets(str(e))
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


def _run_download_job(job_id: str, catalog_id: str) -> None:
    kwargs = _download_kwargs(job_id)
    if kwargs["cancel_event"].is_set():
        _mark_job_cancelled(job_id)
        return
    _update_job(job_id, status="running", started_at=time.time(), progress="starting")
    progress = _job_progress(job_id)

    try:
        found = _find_catalog_item(catalog_id)
        if not found:
            raise RuntimeError(f"Unknown catalog id: {catalog_id}")
        _set_name, folder_key, item = found
        download_item(item, folder_key, progress=progress, **kwargs)
        if kwargs["cancel_event"].is_set():
            _mark_job_cancelled(job_id)
            return
        _update_job(job_id, status="done", finished_at=time.time(), ok=True, progress="done")
    except DownloadCancelled:
        _mark_job_cancelled(job_id)
    except Exception as e:
        err = redact_download_secrets(str(e))
        if not err.startswith("Download failed"):
            err = f"Download failed: {err}"
        _update_job(job_id, status="error", finished_at=time.time(), ok=False, error=err, progress="error")
    finally:
        _clear_job_handles(job_id)


def preview_url(url: str, folder_key: str | None = None) -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "Model downloads are not available on this host."}
    return preview_civitai_url(url, folder_key)


def start_url_download(url: str, folder_key: str | None = None) -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "Model downloads are not available on this host."}
    preview = preview_civitai_url(url, folder_key)
    if not preview.get("ok"):
        return preview
    if preview.get("already_installed"):
        return {
            "ok": True,
            "already_installed": True,
            "display_name": preview.get("display_name"),
            "source_url": url,
        }

    display_name = str(preview.get("display_name") or preview.get("filename") or "Civitai model")
    job_id = uuid.uuid4().hex
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "catalog_id": f"url:{display_name}",
            "display_name": display_name,
            "source_url": url,
            "status": "queued",
            "ok": None,
            "error": None,
            "progress": "queued",
            "created_at": time.time(),
        }
    thread = threading.Thread(
        target=_run_url_download_job,
        args=(job_id, url, display_name, folder_key),
        daemon=True,
    )
    thread.start()
    return {
        "ok": True,
        "job_id": job_id,
        "display_name": display_name,
        "source_url": url,
        "status": "queued",
    }


def start_download(catalog_id: str) -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "Model downloads are not available on this host."}
    found = _find_catalog_item(catalog_id)
    if not found:
        return {"ok": False, "error": f"Unknown model id: {catalog_id}"}
    _set_name, folder_key, item = found
    if _is_installed(folder_key, item):
        return {"ok": True, "already_installed": True, "catalog_id": catalog_id}

    job_id = uuid.uuid4().hex
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "catalog_id": catalog_id,
            "status": "queued",
            "ok": None,
            "error": None,
            "progress": "queued",
            "created_at": time.time(),
        }
    thread = threading.Thread(target=_run_download_job, args=(job_id, catalog_id), daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id, "catalog_id": catalog_id, "status": "queued"}


def list_downloads() -> dict:
    with _DOWNLOAD_LOCK:
        jobs = list(_DOWNLOAD_JOBS.values())
    jobs.sort(key=lambda row: row.get("created_at", 0), reverse=True)
    return {"ok": True, "jobs": jobs[:50]}


def cancel_download(job_id: str) -> dict:
    job_id = (job_id or "").strip()
    if not job_id:
        return {"ok": False, "error": "job id is required"}
    with _DOWNLOAD_LOCK:
        job = _DOWNLOAD_JOBS.get(job_id)
        if not job:
            return {"ok": False, "error": f"Unknown job: {job_id}"}
        status = str(job.get("status") or "")
        if status not in ("queued", "running"):
            return {"ok": False, "error": f"Job is not active ({status})"}
    cancel = _job_cancel_event(job_id)
    cancel.set()
    _kill_job_proc(job_id)
    if status == "queued":
        _mark_job_cancelled(job_id)
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


def _safe_delete_path(folder_key: str, relative_path: str) -> Path:
    if folder_key not in ALLOWED_DELETE_DIRS:
        raise ValueError(f"Delete not allowed for folder: {folder_key}")
    base = ALLOWED_DELETE_DIRS[folder_key]
    rel = Path(relative_path.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Invalid model path")
    target = (base / rel).resolve()
    if _relative_within(base, target) is None:
        raise ValueError("Path escapes model directory")
    if not target.is_file():
        raise FileNotFoundError(f"Model not found: {relative_path}")
    return target


def delete_model(model_id: str) -> dict:
    if ":" not in model_id:
        return {"ok": False, "error": "Model id must be folder:relative/path"}
    folder_key, relative_path = model_id.split(":", 1)
    try:
        target = _safe_delete_path(folder_key, relative_path)
        target.unlink()
        sidecar = target.with_suffix(target.suffix + ".civitai.json")
        if sidecar.is_file():
            sidecar.unlink()
        return {"ok": True, "deleted": model_id}
    except FileNotFoundError as e:
        return {"ok": False, "error": str(e)}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
