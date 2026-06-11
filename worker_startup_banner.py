import os
import shutil
import sys
import urllib.error
import urllib.request
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


def public_host() -> str:
    for key in ("PUBLIC_IPADDR", "VAST_PUBLIC_IP", "EXTERNAL_IP", "VAST_TCP_HOST"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    try:
        with urllib.request.urlopen("https://ifconfig.me/ip", timeout=4) as resp:
            text = resp.read().decode("utf-8", errors="ignore").strip()
            if text:
                return text
    except (OSError, urllib.error.URLError, ValueError):
        pass
    return "YOUR_VAST_IP"


def mapped_port(internal_port: int) -> int:
    for key in (f"VAST_TCP_PORT_{internal_port}", f"PORT_{internal_port}"):
        raw = (os.environ.get(key) or "").strip()
        if raw.isdigit():
            return int(raw)
    return internal_port


def muse_urls() -> tuple[str, str]:
    host = public_host()
    flask_port = mapped_port(int(os.environ.get("FLASK_PORT") or os.environ.get("UI_PORT") or "3000"))
    comfy_port = mapped_port(int(os.environ.get("COMFY_PORT") or "8188"))
    flask_url = f"http://{host}:{flask_port}"
    comfy_url = f"http://{host}:{comfy_port}"
    return flask_url, comfy_url


def _probe(url: str, path: str, timeout: float = 2.5) -> tuple[bool, str]:
    target = f"{url.rstrip('/')}{path}"
    try:
        req = urllib.request.Request(target, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if 200 <= resp.status < 300:
                return True, f"ok ({resp.status})"
            return False, f"status {resp.status}"
    except Exception as e:
        return False, str(e)


def _line(ok: bool | None, label: str, detail: str = "") -> str:
    if ok is True:
        mark = "OK "
    elif ok is False:
        mark = "NO "
    else:
        mark = "-- "
    suffix = f" — {detail}" if detail else ""
    return f"  [{mark}] {label}{suffix}"


def gather_checks(probe_services: bool = False) -> list[tuple[bool | None, str, str]]:
    root = repo_root()
    vol = volume_root()
    comfy_root = Path(os.environ.get("COMFY_ROOT", str(vol / "ComfyUI")))
    venv_py = vol / "venv" / "bin" / "python"
    models_json = root / "models.json"
    flask_port = int(os.environ.get("FLASK_PORT") or os.environ.get("UI_PORT") or "3000")
    comfy_port = int(os.environ.get("COMFY_PORT") or "8188")
    comfy_url = (os.environ.get("COMFY_URL") or f"http://127.0.0.1:{comfy_port}").rstrip("/")

    rows: list[tuple[bool | None, str, str]] = []
    rows.append((venv_py.is_file(), "Python venv", str(venv_py)))
    rows.append(((comfy_root / "main.py").is_file(), "ComfyUI installed", str(comfy_root / "main.py")))
    rows.append((models_json.is_file(), "models.json", str(models_json)))
    rows.append((shutil.which("aria2c") is not None, "aria2c", shutil.which("aria2c") or "missing"))
    rows.append((bool((os.environ.get("HF_TOKEN") or "").strip()), "HF_TOKEN", "configured" if (os.environ.get("HF_TOKEN") or "").strip() else "missing"))
    rows.append((
        bool((os.environ.get("CIVITAI_TOKEN") or "").strip()),
        "CIVITAI_TOKEN",
        "configured" if (os.environ.get("CIVITAI_TOKEN") or "").strip() else "missing",
    ))
    rows.append((os.environ.get("WORKER_CONTROL", "1") == "1", "WORKER_CONTROL", os.environ.get("WORKER_CONTROL", "1")))

    if probe_services:
        flask_ok, flask_detail = _probe(f"http://127.0.0.1:{flask_port}", "/api/health")
        rows.append((flask_ok, f"Flask :{flask_port}", flask_detail))
        comfy_ok, comfy_detail = _probe(comfy_url, "/system_stats")
        rows.append((comfy_ok, f"ComfyUI :{comfy_port}", comfy_detail))
    else:
        rows.append((None, f"Flask :{flask_port}", "starting next"))
        comfy_running = False
        try:
            comfy_running, _ = _probe(comfy_url, "/system_stats", timeout=1.0)
        except Exception:
            comfy_running = False
        if comfy_running:
            rows.append((True, f"ComfyUI :{comfy_port}", "online"))
        else:
            rows.append((None, f"ComfyUI :{comfy_port}", "start from Muse → Start Comfy mode"))

    return rows


def format_banner(probe_services: bool = False, title: str = "Muse worker startup summary") -> str:
    flask_url, comfy_url = muse_urls()
    host = public_host()
    lines = [
        "",
        "=" * 72,
        title,
        "=" * 72,
        "",
        "Readiness",
    ]
    for ok, label, detail in gather_checks(probe_services=probe_services):
        lines.append(_line(ok, label, detail))

    lines.extend([
        "",
        "Copy into Muse → More → Cloud GPU (Cloud mode)",
        "-" * 72,
        f"Flask  {flask_url}",
        f"Comfy  {comfy_url}",
        "-" * 72,
        f"Host detected: {host}",
        f"Internal ports: Flask {os.environ.get('FLASK_PORT', '3000')} · Comfy {os.environ.get('COMFY_PORT', '8188')}",
        "If copy-paste fails, add http:// before host:port in Muse.",
        "",
        "Worker status (on instance):",
        f"  curl -s http://127.0.0.1:{os.environ.get('FLASK_PORT', '3000')}/api/worker/status",
        "",
        "=" * 72,
        "",
    ])
    return "\n".join(lines)


def print_banner(probe_services: bool = False, title: str = "Muse worker startup summary") -> None:
    sys.stdout.write(format_banner(probe_services=probe_services, title=title))
    sys.stdout.flush()


def main() -> int:
    probe = "--probe" in sys.argv
    print_banner(probe_services=probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
