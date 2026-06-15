import os
import shutil
from pathlib import Path

from download_models import aria2_available
from model_manager import manager_supported
from llm_install import llama_server_bin
from llm_manager import load_profiles, manager_supported as llm_manager_supported
from llm_runner import llm_health
from studio_comfy_api import health_status
from worker_startup_banner import mapped_port, muse_llm_urls, muse_urls, public_host

WORKER_API_REVISION = 5


def _volume_root() -> str:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return explicit
    comfy_root = (os.environ.get("COMFY_ROOT") or "/workspace/ComfyUI").strip()
    path = Path(comfy_root)
    if path.name == "ComfyUI":
        return str(path.parent)
    return "/workspace"


def _mode_file() -> Path:
    return Path(_volume_root()) / ".muse-worker" / "mode"


def _service_online(mode: str) -> bool:
    if mode == "llm":
        return bool(llm_health().get("llm_ok"))
    if mode == "comfy":
        return bool(health_status().get("comfy_ok"))
    if mode == "none":
        return not llm_health().get("llm_ok") and not health_status().get("comfy_ok")
    return False


def _read_mode() -> str:
    env_mode = (os.environ.get("WORKER_MODE") or os.environ.get("AUTOSTART_MODE") or "").strip().lower()
    if env_mode in ("none", "comfy", "llm") and _service_online(env_mode):
        return env_mode
    mode_file = _mode_file()
    try:
        if mode_file.is_file():
            mode = mode_file.read_text(encoding="utf-8").strip().lower()
            if mode in ("none", "comfy", "llm") and _service_online(mode):
                return mode
    except OSError:
        pass
    llm = llm_health()
    if llm.get("llm_ok"):
        return "llm"
    comfy = health_status()
    return "comfy" if comfy.get("comfy_ok") else "none"


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
        "llm_port": int(os.environ.get("LLM_PORT") or "8080"),
        "comfy_url": (os.environ.get("COMFY_URL") or "http://127.0.0.1:8188").rstrip("/"),
        "llm_url": (os.environ.get("LLM_URL") or "http://127.0.0.1:8080").rstrip("/"),
        "comfy_root": os.environ.get("COMFY_ROOT") or "/workspace/ComfyUI",
        "bootstrap": os.environ.get("BOOTSTRAP", ""),
        "model_set": os.environ.get("MODEL_SET", ""),
        "image": os.environ.get("IMAGE", ""),
        "app_password_configured": _token_configured("APP_PASSWORD") or _token_configured("UI_PASSWORD"),
        "hf_token_configured": _token_configured("HF_TOKEN"),
        "civitai_token_configured": _token_configured("CIVITAI_TOKEN"),
    }


def build_muse_external_urls() -> dict:
    flask_url, comfy_url = muse_urls()
    llm_urls = muse_llm_urls()
    flask_internal = int(os.environ.get("FLASK_PORT") or os.environ.get("UI_PORT") or "3000")
    comfy_internal = int(os.environ.get("COMFY_PORT") or "8188")
    llm_internal = int(os.environ.get("LLM_PORT") or "8080")
    return {
        "host": public_host(),
        "flask_url": flask_url,
        "comfy_url": comfy_url,
        "llm_openai_base_url": llm_urls.get("external_openai_base_url") or "",
        "ports": {
            "flask_internal": flask_internal,
            "flask_external": mapped_port(flask_internal),
            "comfy_internal": comfy_internal,
            "comfy_external": mapped_port(comfy_internal),
            "llm_internal": llm_internal,
            "llm_external": mapped_port(llm_internal),
        },
    }


def build_worker_status() -> dict:
    mode = _read_mode()
    volume_root = _volume_root()
    comfy_health = health_status()
    comfy_online = bool(comfy_health.get("comfy_ok"))
    llm_status = llm_health()
    llm_online = bool(llm_status.get("llm_ok"))
    llm_catalog = load_profiles()
    muse_external = build_muse_external_urls()

    return {
        "ok": True,
        "mode": mode,
        "volume_root": volume_root,
        "services": {
            "flask": {
                "online": True,
                "port": int(os.environ.get("UI_PORT") or "3000"),
                "external_port": muse_external["ports"]["flask_external"],
                "external_url": muse_external["flask_url"],
            },
            "comfy": {
                "online": comfy_online,
                "port": int(os.environ.get("COMFY_PORT") or "8188"),
                "external_port": muse_external["ports"]["comfy_external"],
                "external_url": muse_external["comfy_url"],
                "url": comfy_health.get("comfy_url"),
                "detail": comfy_health.get("detail"),
            },
            "llm": {
                "online": llm_online,
                "port": llm_status.get("llm_port") or int(os.environ.get("LLM_PORT") or "8080"),
                "external_port": llm_status.get("external_llm_port"),
                "url": llm_status.get("llm_url"),
                "external_url": llm_status.get("external_llm_url"),
                "openai_base_url": llm_status.get("openai_base_url"),
                "external_openai_base_url": llm_status.get("external_openai_base_url"),
                "detail": llm_status.get("detail"),
                "active_profile": llm_status.get("active_profile") or {},
                "binary_installed": bool(llama_server_bin()),
                "llm_api_key_configured": _token_configured("LLM_API_KEY"),
                "model_path": (llm_status.get("active_profile") or {}).get("model_path"),
            },
        },
        "storage": _disk_usage(volume_root),
        "tokens": {
            "hf_token_configured": _token_configured("HF_TOKEN"),
            "civitai_token_configured": _token_configured("CIVITAI_TOKEN"),
            "llm_api_key_configured": _token_configured("LLM_API_KEY"),
            "aria2_available": aria2_available(),
        },
        "features": {
            "api_revision": WORKER_API_REVISION,
            "model_manager": manager_supported(),
            "model_download": manager_supported(),
            "download_cancel": True,
            "llm_runtime": llm_manager_supported(),
            "llm_model_manager": llm_manager_supported(),
        },
        "llm": {
            "catalog": llm_catalog,
        },
        "env": _safe_env_summary(),
        "muse_external": muse_external,
    }
