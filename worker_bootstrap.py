import os
import subprocess
from pathlib import Path


def volume_root() -> Path:
    explicit = (os.environ.get("VOLUME_ROOT") or "").strip()
    if explicit:
        return Path(explicit)
    comfy_root = Path(os.environ.get("COMFY_ROOT", "/workspace/ComfyUI"))
    if comfy_root.name == "ComfyUI":
        return comfy_root.parent
    return Path("/workspace")


def repo_root() -> Path:
    return Path(os.environ.get("APP_DIR", Path(__file__).resolve().parent))


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
    return {"ok": True, "comfy_root": str(comfy_root())}


def ensure_comfy_installed() -> dict:
    if comfy_installed():
        return {"ok": True, "already_installed": True, "comfy_root": str(comfy_root())}
    result = run_install_only()
    if result.get("ok"):
        result["prepared"] = True
    return result
