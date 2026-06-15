import os
import subprocess
from pathlib import Path

from muse_worker.paths import WORKER_ROOT


def volume_root() -> Path:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    comfy_root = Path(os.environ.get("COMFY_ROOT", "/workspace/ComfyUI"))
    if comfy_root.name == "ComfyUI":
        return comfy_root.parent
    return Path("/workspace")


def repo_root() -> Path:
    return Path(os.environ.get("APP_DIR") or WORKER_ROOT)


def python_bin() -> str:
    venv_py = volume_root() / "venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    return os.environ.get("PYTHON", "python3")


def comfy_root() -> Path:
    return Path(os.environ.get("COMFY_ROOT", str(volume_root() / "ComfyUI")))


def comfy_installed() -> bool:
    return (comfy_root() / "main.py").is_file()


def run_install_only(timeout_sec: int = 3600) -> dict:
    root = repo_root()
    script = root / "bootstrap.sh"
    if not script.is_file():
        return {"ok": False, "error": f"bootstrap.sh not found in {root}"}

    env = os.environ.copy()
    venv_bin = volume_root() / "venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(volume_root() / "venv")

    env.setdefault("VOLUME_ROOT", str(volume_root()))
    env.setdefault("COMFY_ROOT", str(comfy_root()))

    try:
        subprocess.run(
            ["bash", str(script), "--install-only"],
            cwd=str(root),
            env=env,
            check=True,
            timeout=timeout_sec,
        )
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": f"bootstrap failed: {e}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "bootstrap timed out"}

    if not comfy_installed():
        return {"ok": False, "error": f"ComfyUI still missing at {comfy_root()}"}
    return {
        "ok": True,
        "comfy_root": str(comfy_root()),
        "llama_installed": llama_installed(),
    }


def ensure_comfy_installed() -> dict:
    if comfy_installed():
        return {"ok": True, "already_installed": True, "comfy_root": str(comfy_root())}
    result = run_install_only()
    if result.get("ok"):
        result["prepared"] = True
    return result


def llama_installed() -> bool:
    from llm_install import llama_server_bin

    return bool(llama_server_bin())


def ensure_llama_runtime(timeout_sec: int = 7200) -> dict:
    from llm_install import ensure_llama_installed, llama_server_bin

    binary = llama_server_bin()
    if binary:
        return {"ok": True, "already_installed": True, "binary": str(binary)}

    result = ensure_llama_installed()
    if result.get("ok"):
        return result

    prep = run_install_only(timeout_sec=timeout_sec)
    binary = llama_server_bin()
    if binary:
        return {"ok": True, "prepared": True, "binary": str(binary), "bootstrap": prep}

    if not result.get("ok"):
        return result
    return {"ok": False, "error": "llama-server binary missing after bootstrap"}
