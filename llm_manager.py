import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from download_models import DownloadCancelled, download_url, hf_download, redact_download_secrets
LLM_MODELS_JSON = Path(os.environ.get("LLM_MODELS_JSON", "llm_models.json"))

_DOWNLOAD_LOCK = threading.Lock()
_DOWNLOAD_JOBS: dict[str, dict] = {}
_JOB_CANCEL: dict[str, threading.Event] = {}


def volume_root() -> Path:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    return Path("/workspace")


def llm_models_dir() -> Path:
    explicit = (os.environ.get("LLM_MODELS_DIR") or "").strip()
    if explicit:
        return Path(explicit)
    return volume_root() / "models" / "llm"


def profile_storage_dir(profile_id: str, meta: dict | None = None) -> Path:
    if meta and meta.get("custom"):
        return llm_models_dir() / "custom"
    return llm_models_dir() / profile_id


def profile_model_path(profile_id: str, meta: dict) -> Path:
    filename = str(meta.get("filename") or "").strip()
    if meta.get("custom"):
        return profile_storage_dir(profile_id, meta)
    if not filename:
        raise ValueError("filename is required")
    return profile_storage_dir(profile_id, meta) / filename


def manager_supported() -> bool:
    return LLM_MODELS_JSON.is_file()


def _catalog_raw() -> dict:
    if not LLM_MODELS_JSON.is_file():
        return {"profiles": {}}
    with open(LLM_MODELS_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024 * 1024


def _size_bytes(path: Path | None) -> int | None:
    try:
        if path and path.is_file():
            return path.stat().st_size
    except OSError:
        pass
    return None


def list_installed() -> dict[str, str]:
    base = llm_models_dir()
    if not base.is_dir():
        return {}
    out: dict[str, str] = {}
    cfg = _catalog_raw()
    for profile_id, meta in (cfg.get("profiles") or {}).items():
        if meta.get("custom"):
            custom_dir = profile_storage_dir(profile_id, meta)
            if custom_dir.is_dir():
                for path in sorted(custom_dir.glob("*.gguf")):
                    if _file_ok(path):
                        out[profile_id] = str(path)
                        break
            continue
        filename = str(meta.get("filename") or "").strip()
        if not filename:
            continue
        target = profile_storage_dir(profile_id, meta) / filename
        if _file_ok(target):
            out[profile_id] = str(target)
    return out


def load_profiles() -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "llm_models.json not found", "profiles": []}
    cfg = _catalog_raw()
    installed = list_installed()
    rows = []
    for profile_id, meta in (cfg.get("profiles") or {}).items():
        model_path = installed.get(profile_id)
        path_obj = Path(model_path) if model_path else None
        rows.append({
            "id": profile_id,
            "display_name": meta.get("display_name") or profile_id,
            "description": meta.get("description") or "",
            "recommended": bool(meta.get("recommended")),
            "gpu_target": meta.get("gpu_target") or "",
            "gpu_warning": meta.get("gpu_warning") or "",
            "custom": bool(meta.get("custom")),
            "ctx_size": meta.get("ctx_size"),
            "port": meta.get("port") or int(os.environ.get("LLM_PORT") or "8080"),
            "installed": profile_id in installed,
            "model_path": model_path,
            "size_bytes": _size_bytes(path_obj),
            "filename": meta.get("filename"),
            "hf_model": meta.get("hf_model"),
            "repo_id": meta.get("repo_id"),
        })
    rows.sort(key=lambda row: (not row.get("recommended"), row.get("display_name") or ""))
    from llm_install import llama_server_bin

    return {
        "ok": True,
        "supported": True,
        "recommended_profile": cfg.get("recommended_profile") or "",
        "models_dir": str(llm_models_dir()),
        "llama_root": str(volume_root() / "llama.cpp"),
        "llama_server_installed": bool(llama_server_bin()),
        "profiles": rows,
        "installed": list(installed.values()),
    }


def resolve_profile(profile_id: str, custom_filename: str | None = None) -> dict | None:
    profile_id = (profile_id or "").strip()
    if not profile_id:
        return None
    cfg = _catalog_raw()
    meta = (cfg.get("profiles") or {}).get(profile_id)
    if not meta:
        return None
    if meta.get("custom"):
        custom_dir = profile_storage_dir(profile_id, meta)
        name = (custom_filename or "").strip()
        if name:
            target = custom_dir / Path(name).name
        else:
            ggufs = sorted(custom_dir.glob("*.gguf")) if custom_dir.is_dir() else []
            target = ggufs[-1] if ggufs else None
        if not target or not _file_ok(target):
            return None
        return _profile_runtime(profile_id, meta, str(target))
    try:
        target = profile_model_path(profile_id, meta)
    except ValueError:
        return None
    if not _file_ok(target):
        return None
    return _profile_runtime(profile_id, meta, str(target))


def _profile_runtime(profile_id: str, meta: dict, model_path: str) -> dict:
    return {
        "id": profile_id,
        "display_name": meta.get("display_name") or profile_id,
        "model_path": model_path,
        "ctx_size": int(meta.get("ctx_size") or os.environ.get("LLM_CTX_SIZE") or 8192),
        "n_gpu_layers": int(
            meta.get("n_gpu_layers")
            if meta.get("n_gpu_layers") is not None
            else os.environ.get("LLM_GPU_LAYERS") or os.environ.get("LLM_N_GPU_LAYERS") or 999
        ),
        "hf_model": meta.get("hf_model"),
    }


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


def _clear_job_handles(job_id: str) -> None:
    with _DOWNLOAD_LOCK:
        _JOB_CANCEL.pop(job_id, None)


def _progress_cb(job_id: str):
    def _cb(msg: str, **stats) -> None:
        fields = {"progress": msg}
        fields.update(stats)
        _update_job(job_id, **fields)
    return _cb


def _hf_repo_and_file(url: str) -> tuple[str, str] | None:
    raw = (url or "").strip()
    if "huggingface.co" not in raw:
        return None
    parsed = urlparse(raw)
    parts = [p for p in parsed.path.split("/") if p]
    if parts and parts[0] in ("models", "datasets"):
        parts = parts[1:]
    if len(parts) < 2:
        return None
    repo_id = f"{parts[0]}/{parts[1]}"
    if "resolve" in parts:
        filename = parts[-1]
    elif len(parts) >= 3:
        filename = parts[-1]
    else:
        return None
    if not str(filename).endswith(".gguf"):
        return None
    return repo_id, filename


def _hf_download_profile(meta: dict, target: Path, progress, cancel) -> None:
    repo_path = str(meta.get("repo_path") or meta.get("filename") or "").strip()
    repo_ids: list[str] = []
    primary = str(meta.get("repo_id") or "").strip()
    if primary:
        repo_ids.append(primary)
    for alt in meta.get("alt_repo_ids") or []:
        alt_id = str(alt or "").strip()
        if alt_id and alt_id not in repo_ids:
            repo_ids.append(alt_id)
    if not repo_ids:
        raise RuntimeError("Profile missing repo_id")

    last_error: Exception | None = None
    for repo_id in repo_ids:
        try:
            hf_download(repo_id, repo_path, target, progress=progress, cancel_event=cancel)
            return
        except DownloadCancelled:
            raise
        except Exception as e:
            last_error = e
    if last_error:
        raise last_error
    raise RuntimeError("Hugging Face download failed")


def _run_profile_download(job_id: str, profile_id: str) -> None:
    cancel = _job_cancel_event(job_id)
    _update_job(job_id, status="running", progress="preparing")
    try:
        cfg = _catalog_raw()
        meta = (cfg.get("profiles") or {}).get(profile_id)
        if not meta:
            _update_job(job_id, status="failed", ok=False, error=f"Unknown profile: {profile_id}")
            return
        if meta.get("custom"):
            _update_job(job_id, status="failed", ok=False, error="Use URL download for custom_gguf")
            return

        repo_id = str(meta.get("repo_id") or "").strip()
        repo_path = str(meta.get("repo_path") or meta.get("filename") or "").strip()
        filename = str(meta.get("filename") or Path(repo_path).name).strip()
        if not repo_id or not repo_path:
            _update_job(job_id, status="failed", ok=False, error="Profile missing repo_id/repo_path")
            return

        target_dir = profile_storage_dir(profile_id, meta)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        if _file_ok(target):
            _update_job(job_id, status="done", ok=True, progress="already installed", model_path=str(target))
            return

        _update_job(job_id, progress=f"downloading {filename}")
        _hf_download_profile(meta, target, progress=_progress_cb(job_id), cancel=cancel)
        if not _file_ok(target):
            raise RuntimeError("Download finished but GGUF is missing or too small")
        _update_job(job_id, status="done", ok=True, progress="complete", model_path=str(target))
    except DownloadCancelled:
        _update_job(job_id, status="cancelled", ok=False, error="cancelled", progress="cancelled")
    except Exception as e:
        _update_job(job_id, status="failed", ok=False, error=redact_download_secrets(str(e)), progress="failed")
    finally:
        _clear_job_handles(job_id)


def _run_url_download(job_id: str, url: str) -> None:
    cancel = _job_cancel_event(job_id)
    _update_job(job_id, status="running", progress="preparing")
    try:
        custom_dir = llm_models_dir() / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)
        resolved = _hf_repo_and_file(url)
        if resolved:
            repo_id, filename = resolved
            target = custom_dir / filename
            if _file_ok(target):
                _update_job(job_id, status="done", ok=True, progress="already installed", model_path=str(target))
                return
            _update_job(job_id, progress=f"downloading {filename} from Hugging Face")
            hf_download(repo_id, filename, target, progress=_progress_cb(job_id), cancel_event=cancel)
        else:
            name = Path(urlparse(url).path).name
            if not name.endswith(".gguf"):
                name = f"custom-{int(time.time())}.gguf"
            target = custom_dir / name
            if _file_ok(target):
                _update_job(job_id, status="done", ok=True, progress="already installed", model_path=str(target))
                return
            _update_job(job_id, progress=f"downloading {name}")
            download_url(url, target, progress=_progress_cb(job_id), cancel_event=cancel)

        if not _file_ok(target):
            raise RuntimeError("Download finished but GGUF is missing or too small")
        _update_job(job_id, status="done", ok=True, progress="complete", model_path=str(target))
    except DownloadCancelled:
        _update_job(job_id, status="cancelled", ok=False, error="cancelled", progress="cancelled")
    except Exception as e:
        _update_job(job_id, status="failed", ok=False, error=redact_download_secrets(str(e)), progress="failed")
    finally:
        _clear_job_handles(job_id)


def start_profile_download(profile_id: str) -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "LLM model manager not available on this host."}
    profile_id = (profile_id or "").strip()
    if not profile_id:
        return {"ok": False, "error": "profile_id is required"}
    meta = (_catalog_raw().get("profiles") or {}).get(profile_id) or {}
    if meta.get("custom"):
        return {"ok": False, "error": "Use URL download for custom_gguf profile"}
    installed = list_installed()
    if profile_id in installed:
        return {"ok": True, "already_installed": True, "profile_id": profile_id, "model_path": installed[profile_id]}

    job_id = uuid.uuid4().hex
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "profile_id": profile_id,
            "display_name": meta.get("display_name") or profile_id,
            "status": "queued",
            "ok": None,
            "error": None,
            "progress": "queued",
            "created_at": time.time(),
        }
    threading.Thread(target=_run_profile_download, args=(job_id, profile_id), daemon=True).start()
    return {"ok": True, "job_id": job_id, "profile_id": profile_id, "status": "queued"}


def start_url_download(url: str) -> dict:
    if not manager_supported():
        return {"ok": False, "supported": False, "error": "LLM model manager not available on this host."}
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "url is required"}
    display = url if len(url) < 80 else f"{url[:77]}..."
    job_id = uuid.uuid4().hex
    with _DOWNLOAD_LOCK:
        _DOWNLOAD_JOBS[job_id] = {
            "id": job_id,
            "profile_id": "custom_gguf",
            "display_name": display,
            "source_url": url,
            "status": "queued",
            "ok": None,
            "error": None,
            "progress": "queued",
            "created_at": time.time(),
        }
    threading.Thread(target=_run_url_download, args=(job_id, url), daemon=True).start()
    return {"ok": True, "job_id": job_id, "status": "queued", "source_url": url}


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
    _job_cancel_event(job_id).set()
    if status == "queued":
        _update_job(job_id, status="cancelled", ok=False, error="cancelled", progress="cancelled")
        _clear_job_handles(job_id)
    return {"ok": True, "job_id": job_id, "status": "cancelled"}


def delete_model(profile_id: str) -> dict:
    profile_id = (profile_id or "").strip()
    if not profile_id:
        return {"ok": False, "error": "profile id is required"}
    cfg = _catalog_raw()
    meta = (cfg.get("profiles") or {}).get(profile_id)
    if not meta:
        return {"ok": False, "error": f"Unknown profile: {profile_id}"}
    removed: list[str] = []
    if meta.get("custom"):
        custom_dir = profile_storage_dir(profile_id, meta)
        if custom_dir.is_dir():
            for path in custom_dir.glob("*.gguf"):
                try:
                    path.unlink()
                    removed.append(str(path))
                except OSError as e:
                    return {"ok": False, "error": str(e)}
        return {"ok": True, "profile_id": profile_id, "removed": removed}
    try:
        target = profile_model_path(profile_id, meta)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if target.is_file():
        try:
            target.unlink()
            removed.append(str(target))
        except OSError as e:
            return {"ok": False, "error": str(e)}
    profile_dir = profile_storage_dir(profile_id, meta)
    if profile_dir.is_dir() and not any(profile_dir.iterdir()):
        try:
            profile_dir.rmdir()
        except OSError:
            pass
    if not removed:
        return {"ok": False, "error": "Model file not found"}
    return {"ok": True, "profile_id": profile_id, "removed": removed}


load_catalog = load_profiles
