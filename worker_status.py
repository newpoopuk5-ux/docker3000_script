import os
import shutil
from pathlib import Path

from download_models import aria2_available
from model_manager import manager_supported
from studio_comfy_api import health_status

WORKER_API_REVISION = 3

MODE_FILE = Path("/workspace/.muse-worker/mode")


def _read_mode() -> str:
    env_mode = (os.environ.get("WORKER_MODE") or os.environ.get("AUTOSTART_MODE") or "").strip().lower()
    if env_mode in ("none", "comfy"):
        return env_mode
    try:
        if MODE_FILE.is_file():
            mode = MODE_FILE.read_text(encoding="utf-8").strip().lower()
            if mode in ("none", "comfy"):
                return mode
    except OSError:
        pass
    comfy = health_status()
    return "comfy" if comfy.get("comfy_ok") else "none"


def _volume_root() -> str:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return explicit
    comfy_root = (os.environ.get("COMFY_ROOT") or "/workspace/ComfyUI").strip()
    path = Path(comfy_root)
    if path.name == "ComfyUI":
        return str(path.parent)
    return "/workspace"


def _disk_usage(path: str) -> dict:
    try:
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError as e:
        return {"path": path, "error": str(e)}


def _token_configured(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def _safe_env_summary() -> dict:
    return {
        "ui_port": int(os.environ.get("UI_PORT") or "3000"),
        "comfy_port": int(os.environ.get("COMFY_PORT") or "8188"),
        "comfy_url": (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/"),
        "comfy_root": os.environ.get("COMFY_ROOT") or "/workspace/ComfyUI",
        "bootstrap": os.environ.get("BOOTSTRAP", ""),
        "model_set": os.environ.get("MODEL_SET", ""),
        "image": os.environ.get("IMAGE", ""),
        "app_password_configured": _token_configured("APP_PASSWORD") or _token_configured("UI_PASSWORD"),
        "hf_token_configured": _token_configured("HF_TOKEN"),
        "civitai_token_configured": _token_configured("CIVITAI_TOKEN"),
    }


def build_worker_status() -> dict:
    mode = _read_mode()
    volume_root = _volume_root()
    comfy_health = health_status()
    comfy_online = bool(comfy_health.get("comfy_ok"))

    return {
        "ok": True,
        "mode": mode,
        "volume_root": volume_root,
        "services": {
            "flask": {
                "online": True,
                "port": int(os.environ.get("UI_PORT") or "3000"),
            },
            "comfy": {
                "online": comfy_online,
                "port": int(os.environ.get("COMFY_PORT") or "8188"),
                "url": comfy_health.get("comfy_url"),
                "detail": comfy_health.get("detail"),
            },
        },
        "storage": _disk_usage(volume_root),
        "tokens": {
            "hf_token_configured": _token_configured("HF_TOKEN"),
            "civitai_token_configured": _token_configured("CIVITAI_TOKEN"),
            "aria2_available": aria2_available(),
        },
        "features": {
            "api_revision": WORKER_API_REVISION,
            "model_manager": manager_supported(),
            "model_download": manager_supported(),
            "download_cancel": True,
        },
        "env": _safe_env_summary(),
    }
