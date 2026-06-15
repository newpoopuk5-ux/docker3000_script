import os
import subprocess
from pathlib import Path

from muse_worker.paths import WORKER_ROOT


def volume_root() -> Path:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    return Path("/workspace")


def llama_root() -> Path:
    explicit = (os.environ.get("LLAMA_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    return volume_root() / "llama.cpp"


def llama_server_bin() -> Path | None:
    env_bin = (os.environ.get("LLAMA_SERVER_BIN") or "").strip()
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin)
    candidate = llama_root() / "build" / "bin" / "llama-server"
    return candidate if candidate.is_file() else None


def ensure_llama_installed() -> dict:
    existing = llama_server_bin()
    if existing:
        return {"ok": True, "already_installed": True, "binary": str(existing)}

    script = WORKER_ROOT / "install-llama-cpp.sh"
    if not script.is_file():
        return {"ok": False, "error": f"install script missing: {script}"}

    env = os.environ.copy()
    env.setdefault("VOLUME_ROOT", str(volume_root()))
    env.setdefault("LLAMA_ROOT", str(llama_root()))

    try:
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(script.parent),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        return {"ok": False, "error": str(e)}

    binary = llama_server_bin()
    if proc.returncode != 0 or not binary:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return {
            "ok": False,
            "error": "llama.cpp install failed",
            "log_tail": tail,
            "exit_code": proc.returncode,
        }
    return {"ok": True, "installed": True, "binary": str(binary)}
