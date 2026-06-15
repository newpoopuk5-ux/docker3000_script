import os
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from muse_worker.paths import WORKER_ROOT


def load_etc_environment() -> None:
    path = Path("/etc/environment")
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


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
    load_etc_environment()
    for key in (f"VAST_TCP_PORT_{internal_port}", f"PORT_{internal_port}"):
        raw = (os.environ.get(key) or "").strip()
        if raw.isdigit():
            return int(raw)
    return internal_port


def vast_port_mapping_warning(flask_internal: int, comfy_internal: int) -> str:
    load_etc_environment()
    flask_external = mapped_port(flask_internal)
    comfy_external = mapped_port(comfy_internal)
    vast_keys = [k for k in os.environ if re.match(r"^VAST_TCP_PORT_\d+$", k)]
    if vast_keys and (flask_external != flask_internal or comfy_external != comfy_internal):
        return ""
    if not vast_keys and public_host() not in ("YOUR_VAST_IP", "127.0.0.1", "localhost"):
        return (
            "WARN: VAST_TCP_PORT_* not found - URLs below use internal ports and will NOT work from your PC. "
            "Copy Open / direct ports from the Vast instance page, or run: env | grep VAST_TCP_PORT"
        )
    if flask_external == flask_internal or comfy_external == comfy_internal:
        return (
            "WARN: external port equals internal port - on Vast this is usually wrong from outside the instance. "
            "Use mapped ports from the Vast UI (e.g. 34346 not 3000)."
        )
    return ""


def muse_urls() -> tuple[str, str]:
    host = public_host()
    flask_port = mapped_port(int(os.environ.get("FLASK_PORT") or os.environ.get("UI_PORT") or "3000"))
    comfy_port = mapped_port(int(os.environ.get("COMFY_PORT") or "8188"))
    flask_url = f"http://{host}:{flask_port}"
    comfy_url = f"http://{host}:{comfy_port}"
    return flask_url, comfy_url


def muse_llm_urls() -> dict:
    host = public_host()
    llm_internal = int(os.environ.get("LLM_PORT") or "8080")
    llm_external = mapped_port(llm_internal)
    base = f"http://{host}:{llm_external}"
    return {
        "external_llm_port": llm_external,
        "external_llm_url": base,
        "external_openai_base_url": f"{base}/v1",
        "llm_internal_port": llm_internal,
    }


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
    suffix = f" - {detail}" if detail else ""
    return f"  [{mark}] {label}{suffix}"


def gather_checks(probe_services: bool = False) -> list[tuple[bool | None, str, str]]:
    root = repo_root()
    vol = volume_root()
    comfy_root = Path(os.environ.get("COMFY_ROOT", str(vol / "ComfyUI")))
    venv_py = vol / "venv" / "bin" / "python"
    models_json = root / "muse_worker" / "catalog" / "sources" / "models.json"
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
            rows.append((None, f"ComfyUI :{comfy_port}", "start from Muse -> Start Comfy mode"))

    return rows


def format_banner(probe_services: bool = False, title: str = "Muse worker startup summary") -> str:
    load_etc_environment()
    flask_url, comfy_url = muse_urls()
    llm_urls = muse_llm_urls()
    host = public_host()
    flask_internal = int(os.environ.get("FLASK_PORT") or os.environ.get("UI_PORT") or "3000")
    comfy_internal = int(os.environ.get("COMFY_PORT") or "8188")
    llm_internal = int(llm_urls.get("llm_internal_port") or os.environ.get("LLM_PORT") or "8080")
    port_warn = vast_port_mapping_warning(flask_internal, comfy_internal)
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
        "Copy into Muse -> More -> Cloud GPU (Cloud mode)",
        "-" * 72,
        f"Flask  {flask_url}",
        f"Comfy  {comfy_url}",
        f"LLM    {llm_urls['external_openai_base_url']}",
        "-" * 72,
        f"Host detected: {host}",
        f"Internal ports: Flask {flask_internal} / Comfy {comfy_internal} / LLM {llm_internal}",
        *( [port_warn] if port_warn else [] ),
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
    text = format_banner(probe_services=probe_services, title=title)
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding))
    sys.stdout.flush()


def main() -> int:
    probe = "--probe" in sys.argv
    print_banner(probe_services=probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
