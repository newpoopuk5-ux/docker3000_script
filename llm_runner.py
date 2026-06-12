import json
import os
import shlex
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from llm_install import llama_server_bin, volume_root
from llm_manager import llm_models_dir, resolve_profile
from worker_bootstrap import ensure_llama_runtime


def llm_port() -> int:
    return int(os.environ.get("LLM_PORT") or "8080")


def _pid_file() -> Path:
    return volume_root() / ".muse-worker" / "llama.pid"


def _log_file() -> Path:
    return volume_root() / "llm.log"


def _active_profile_file() -> Path:
    return volume_root() / ".muse-worker" / "llm-profile.json"


def _cuda_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _resolve_n_gpu_layers(profile: dict) -> int:
    for key in ("LLM_GPU_LAYERS", "LLM_N_GPU_LAYERS"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return int(raw)
    if profile.get("n_gpu_layers") is not None:
        return int(profile["n_gpu_layers"])
    return 999 if _cuda_available() else 0


def _read_pid() -> Optional[int]:
    try:
        if _pid_file().is_file():
            return int(_pid_file().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    return None


def _clear_pid() -> None:
    try:
        _pid_file().unlink(missing_ok=True)
    except OSError:
        pass


def _save_active_profile(profile: dict) -> None:
    path = _active_profile_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": profile.get("id"),
                "display_name": profile.get("display_name"),
                "model_path": profile.get("model_path"),
                "hf_model": profile.get("hf_model"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def active_profile() -> dict:
    try:
        if _active_profile_file().is_file():
            return json.loads(_active_profile_file().read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    return {}


def llm_api_key_configured() -> bool:
    return bool((os.environ.get("LLM_API_KEY") or "").strip())


def llm_health() -> dict:
    port = llm_port()
    base = f"http://127.0.0.1:{port}"
    ok = False
    detail = "offline"
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"{base}{path}", timeout=3) as resp:
                ok = 200 <= resp.status < 300
                detail = "ok" if ok else f"status {resp.status}"
                if ok:
                    break
        except urllib.error.HTTPError as e:
            if e.code < 500:
                ok = True
                detail = "ok"
                break
            detail = f"http {e.code}"
        except Exception as e:
            detail = str(e)
    binary = llama_server_bin()
    return {
        "llm_ok": ok,
        "llm_url": base,
        "llm_port": port,
        "openai_base_url": f"{base}/v1",
        "detail": detail,
        "pid": _read_pid(),
        "models_dir": str(llm_models_dir()),
        "llama_root": str(volume_root() / "llama.cpp"),
        "active_profile": active_profile(),
        "binary": str(binary or ""),
        "binary_installed": bool(binary),
        "llm_api_key_configured": llm_api_key_configured(),
    }


def _wait_ready(timeout_sec: int = 180) -> bool:
    for _ in range(max(1, timeout_sec // 2)):
        if llm_health().get("llm_ok"):
            return True
        time.sleep(2)
    return False


def _extra_args() -> list[str]:
    raw = (os.environ.get("LLM_EXTRA_ARGS") or "").strip()
    if not raw:
        return []
    return shlex.split(raw)


def start_llm(profile_id: str, custom_filename: str | None = None) -> dict:
    if llm_health().get("llm_ok"):
        return {"ok": True, "already_running": True, **llm_health()}

    install = ensure_llama_runtime()
    if not install.get("ok"):
        return {"ok": False, "error": install.get("error") or "llama.cpp is not installed", "install": install}

    profile = resolve_profile(profile_id, custom_filename=custom_filename)
    if not profile:
        return {
            "ok": False,
            "error": (
                f"LLM profile not ready: {profile_id}. Download the GGUF first "
                f"(stored under {llm_models_dir()})."
            ),
        }

    binary = llama_server_bin()
    if not binary:
        return {"ok": False, "error": "llama-server binary missing after install"}

    port = llm_port()
    log_path = _log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _pid_file().parent.mkdir(parents=True, exist_ok=True)

    ngl = _resolve_n_gpu_layers(profile)
    cmd = [
        str(binary),
        "-m",
        profile["model_path"],
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "-c",
        str(profile.get("ctx_size") or int(os.environ.get("LLM_CTX_SIZE") or 8192)),
        "-ngl",
        str(ngl),
    ]
    api_key = (os.environ.get("LLM_API_KEY") or "").strip()
    if api_key:
        cmd.extend(["--api-key", api_key])
    cmd.extend(_extra_args())

    with open(log_path, "ab", buffering=0) as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _pid_file().write_text(str(proc.pid), encoding="utf-8")
    _save_active_profile(profile)

    if _wait_ready():
        return {"ok": True, "started": True, "pid": proc.pid, "profile": profile, **llm_health()}

    tail = ""
    try:
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        pass
    return {
        "ok": False,
        "error": "llama-server did not become ready",
        "pid": proc.pid,
        "profile": profile,
        "log_tail": tail,
    }


def stop_llm() -> dict:
    was_online = bool(llm_health().get("llm_ok"))
    pid = _read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        time.sleep(1)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    if os.name != "nt":
        port = llm_port()
        subprocess.run(["pkill", "-f", f"llama-server.*--port {port}"], check=False)

    _clear_pid()
    time.sleep(1)
    still_online = bool(llm_health().get("llm_ok"))
    return {
        "ok": not still_online,
        "stopped": was_online or pid is not None,
        "already_stopped": not was_online and pid is None,
        "llm_online": still_online,
        **llm_health(),
    }


def test_llm() -> dict:
    health = llm_health()
    if not health.get("llm_ok"):
        return {"ok": False, "error": "LLM is not running", **health}
    base = health.get("openai_base_url") or f"http://127.0.0.1:{llm_port()}/v1"
    headers = {}
    api_key = (os.environ.get("LLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        req = urllib.request.Request(f"{base}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body) if body else {}
        models = payload.get("data") if isinstance(payload, dict) else None
        return {
            "ok": True,
            "detail": "models endpoint ok",
            "model_count": len(models) if isinstance(models, list) else None,
            **health,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), **health}
