import json
import os
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
from download_models import download_item
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


def _installed_entries() -> list[dict]:
    entries: list[dict] = []
    for folder_key, base in ALLOWED_DELETE_DIRS.items():
        if not base.exists():
            continue
        recursive = folder_key in ("loras", "text_encoders", "diffusion_models")
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
    delete_key = CATALOG_FOLDER_TO_DELETE_KEY.get(folder_key, folder_key)
    base = ALLOWED_DELETE_DIRS.get(delete_key)
    if not base:
        return False
    if item.get("target"):
        target = Path(str(item["target"]))
        if target.is_file() and target.stat().st_size > 1024 * 1024:
            return True
    name = _target_name_for_item(folder_key, item)
    candidate = base / name
    if candidate.is_file() and candidate.stat().st_size > 1024 * 1024:
        return True
    if delete_key == "loras":
        nested = base / "flux" / name
        if nested.is_file() and nested.stat().st_size > 1024 * 1024:
            return True
    return False


def load_catalog(set_name: str | None = None) -> dict:
    if not MODELS_JSON.is_file():
        return {"ok": False, "supported": False, "error": "models.json not found", "sets": {}, "catalog": []}

    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sets_info = {
        key: {"description": (value or {}).get("description", "")}
        for key, value in (cfg.get("sets") or {}).items()
    }
    active_set = set_name or (os.environ.get("MODEL_SET") or "basic")
    if active_set not in cfg.get("sets", {}):
        active_set = next(iter(cfg.get("sets", {}).keys()), "basic")

    catalog: list[dict] = []
    selected = cfg["sets"][active_set]
    for group_key, folder_key in CATALOG_GROUPS:
        for index, item in enumerate(selected.get(group_key, []) or []):
            if not isinstance(item, dict):
                continue
            cid = _catalog_item_id(active_set, folder_key, item, index)
            delete_key = CATALOG_FOLDER_TO_DELETE_KEY.get(folder_key, folder_key)
            catalog.append({
                "id": cid,
                "set": active_set,
                "group": group_key,
                "folder": delete_key,
                "name": _target_name_for_item(folder_key, item),
                "source": item.get("source", "direct"),
                "installed": _is_installed(folder_key, item),
                "recommended": True,
            })

    installed = _installed_entries()
    installed_ids = {row["id"] for row in installed}
    missing = [row for row in catalog if not row["installed"]]

    presets = {}
    if PRESETS_PATH.is_file():
        try:
            presets = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            presets = {}

    return {
        "ok": True,
        "supported": True,
        "active_set": active_set,
        "sets": sets_info,
        "catalog": catalog,
        "installed": installed,
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


def _run_download_job(job_id: str, catalog_id: str) -> None:
    with _DOWNLOAD_LOCK:
        job = _DOWNLOAD_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = time.time()

    try:
        found = _find_catalog_item(catalog_id)
        if not found:
            raise RuntimeError(f"Unknown catalog id: {catalog_id}")
        _set_name, folder_key, item = found
        download_item(item, folder_key)
        with _DOWNLOAD_LOCK:
            job = _DOWNLOAD_JOBS[job_id]
            job["status"] = "done"
            job["finished_at"] = time.time()
            job["ok"] = True
    except Exception as e:
        with _DOWNLOAD_LOCK:
            job = _DOWNLOAD_JOBS[job_id]
            job["status"] = "error"
            job["finished_at"] = time.time()
            job["ok"] = False
            job["error"] = str(e)


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
