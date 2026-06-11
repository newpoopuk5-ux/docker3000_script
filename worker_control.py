import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from studio_comfy_api import health_status
from worker_bootstrap import ensure_comfy_installed, python_bin

MODE_NAMES = ("none", "comfy")


def _volume_root() -> Path:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    comfy_root = Path(os.environ.get("COMFY_ROOT", "/workspace/ComfyUI"))
    if comfy_root.name == "ComfyUI":
        return comfy_root.parent
    return Path("/workspace")


def _mode_file() -> Path:
    return _volume_root() / ".muse-worker" / "mode"


def _pid_file() -> Path:
    return _volume_root() / ".muse-worker" / "comfy.pid"


def control_supported() -> bool:
    flag = (os.environ.get("WORKER_CONTROL") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if (os.environ.get("VOLUME_ROOT") or "").strip():
        return True
    mode_parent = _mode_file().parent
    try:
        mode_parent.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False


def unsupported_payload(action: str) -> dict:
    return {
        "ok": False,
        "supported": False,
        "action": action,
        "error": "Worker mode control is only available on the cloud worker host (Vast/RunPod).",
        "hint": "Use Local mode in Muse, or deploy docker3000_script with VOLUME_ROOT/.muse-worker.",
    }


def _write_mode(mode: str) -> None:
    mode = mode.strip().lower()
    if mode not in MODE_NAMES:
        raise ValueError(f"mode must be one of {MODE_NAMES}")
    path = _mode_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mode, encoding="utf-8")
    os.environ["WORKER_MODE"] = mode
    os.environ["AUTOSTART_MODE"] = mode


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


def _comfy_root() -> Path:
    return Path(os.environ.get("COMFY_ROOT", "/workspace/ComfyUI"))


def _comfy_port() -> int:
    return int(os.environ.get("COMFY_PORT") or "8188")


def _wait_comfy_ready(timeout_sec: int = 90) -> bool:
    for _ in range(max(1, timeout_sec // 2)):
        if health_status().get("comfy_ok"):
            return True
        time.sleep(2)
    return False


def start_comfy() -> dict:
    health = health_status()
    if health.get("comfy_ok"):
        return {"ok": True, "already_running": True, "detail": health.get("detail")}

    comfy_root = _comfy_root()
    if not (comfy_root / "main.py").exists():
        prep = ensure_comfy_installed()
        if not prep.get("ok"):
            return {
                "ok": False,
                "error": prep.get("error") or f"ComfyUI is not installed at {comfy_root}",
                "bootstrap": prep,
            }

    log_path = _volume_root() / "comfyui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _mode_file().parent.mkdir(parents=True, exist_ok=True)

    py = python_bin()
    with open(log_path, "ab", buffering=0) as log:
        proc = subprocess.Popen(
            [py, "main.py", "--listen", "0.0.0.0", "--port", str(_comfy_port())],
            cwd=str(comfy_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _pid_file().write_text(str(proc.pid), encoding="utf-8")

    if _wait_comfy_ready():
        return {"ok": True, "started": True, "pid": proc.pid, "detail": health_status().get("detail")}

    tail = ""
    try:
        if log_path.exists():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    except OSError:
        pass
    return {
        "ok": False,
        "error": "ComfyUI did not become ready",
        "pid": proc.pid,
        "log_tail": tail,
    }


def stop_comfy() -> dict:
    was_online = bool(health_status().get("comfy_ok"))
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

    # Fallback for processes started outside our pid file.
    if os.name != "nt":
        subprocess.run(["pkill", "-f", "python main.py --listen"], check=False)

    _clear_pid()
    time.sleep(1)
    still_online = bool(health_status().get("comfy_ok"))
    return {
        "ok": not still_online,
        "stopped": was_online or pid is not None,
        "already_stopped": not was_online and pid is None,
        "comfy_online": still_online,
        "detail": health_status().get("detail"),
    }


def set_mode_comfy() -> dict:
    if not control_supported():
        return unsupported_payload("comfy")
    _write_mode("comfy")
    start_result = start_comfy()
    mode = _mode_file().read_text(encoding="utf-8").strip() if _mode_file().is_file() else "comfy"
    return {
        "ok": bool(start_result.get("ok")),
        "supported": True,
        "mode": mode,
        "comfy": start_result,
        "services": {
            "flask": {"online": True},
            "comfy": {
                "online": bool(health_status().get("comfy_ok")),
                "detail": health_status().get("detail"),
            },
        },
    }


def set_mode_none() -> dict:
    if not control_supported():
        return unsupported_payload("none")
    stop_result = stop_comfy()
    _write_mode("none")
    mode = _mode_file().read_text(encoding="utf-8").strip() if _mode_file().is_file() else "none"
    return {
        "ok": bool(stop_result.get("ok", True)),
        "supported": True,
        "mode": mode,
        "comfy": stop_result,
        "services": {
            "flask": {"online": True},
            "comfy": {
                "online": bool(health_status().get("comfy_ok")),
                "detail": health_status().get("detail"),
            },
        },
    }
