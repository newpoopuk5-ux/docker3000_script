import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse, parse_qs
import requests
import zipfile

from config import COMFY_ROOT
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ARIA2_CONNECTIONS = max(1, min(32, int(os.environ.get("ARIA2_CONNECTIONS", "16") or 16)))


class DownloadCancelled(Exception):
    pass


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelled()


def _abort_if_cancelled(
    cancel_event: threading.Event | None,
    proc: subprocess.Popen | None = None,
    target: Path | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        _stop_proc(proc)
        if target is not None:
            _remove_partial_target(target)
        raise DownloadCancelled()


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

TARGET_DIRS = {
    "checkpoints": COMFY_ROOT / "models" / "checkpoints",
    "loras": COMFY_ROOT / "models" / "loras",
    "upscalers": COMFY_ROOT / "models" / "upscale_models",
    "vae": COMFY_ROOT / "models" / "vae",
    "flux_diffusion_models": COMFY_ROOT / "models" / "diffusion_models",
    "flux_gguf_unet": COMFY_ROOT / "models" / "unet",
    "text_encoders": COMFY_ROOT / "models" / "text_encoders",
    "text_encoders_gguf": COMFY_ROOT / "models" / "text_encoders",
    "flux_loras": COMFY_ROOT / "models" / "loras" / "flux",
}

def run(cmd):
    print("+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


ARIA2_PROGRESS_RX = re.compile(
    r"\[?#?(?P<gid>[0-9a-f]{4,})\s+"
    r"(?P<done>[\d.]+)(?P<done_unit>GiB|MiB|KiB|B)/"
    r"(?P<total>[\d.]+)(?P<total_unit>GiB|MiB|KiB|B)\((?P<pct>\d+)%\)"
    r"(?:\s+CN:(?P<cn>\d+))?"
    r"\s+DL:(?P<speed>[\d.]+)(?P<speed_unit>GiB|MiB|KiB|B)"
    r"\s+ETA:(?P<eta>[^\]\s]+)",
    re.IGNORECASE,
)


def redact_download_secrets(text: str) -> str:
    out = str(text or "")
    if CIVITAI_TOKEN:
        out = out.replace(CIVITAI_TOKEN, "[redacted]")
    if HF_TOKEN:
        out = out.replace(HF_TOKEN, "[redacted]")
    out = re.sub(r"([?&]token=)[^&\s\"']+", r"\1[redacted]", out, flags=re.IGNORECASE)
    out = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", out, flags=re.IGNORECASE)
    return out


def _safe_log_line(text: str) -> None:
    print(redact_download_secrets(text))


def _emit_progress(progress: Optional[Callable[..., None]], msg: str, **stats) -> None:
    if not progress:
        return
    try:
        progress(msg, **stats)
    except TypeError:
        progress(msg)


def _head_content_length(url: str) -> int:
    token_url = add_token(url)
    headers = _civitai_request_headers(url)
    try:
        resp = requests.head(token_url, headers=headers, allow_redirects=True, timeout=20)
        if resp.status_code >= 400:
            with requests.get(token_url, headers=headers, stream=True, timeout=20, allow_redirects=True) as stream:
                stream.raise_for_status()
                return int(stream.headers.get("content-length") or 0)
        return int(resp.headers.get("content-length") or 0)
    except Exception:
        return 0


def _emit_byte_progress(
    progress: Optional[Callable[..., None]],
    downloaded: int,
    total: int,
    started: float,
    *,
    connections: str | int | None = None,
) -> None:
    if not progress:
        return
    elapsed = max(time.time() - started, 0.1)
    speed_bps = downloaded / elapsed
    done = _human_bytes(downloaded)
    speed = f"{_human_bytes(int(speed_bps))}/s"
    if total > 0:
        pct = min(100, int(downloaded * 100 / total))
        total_h = _human_bytes(total)
        remaining = max(total - downloaded, 0)
        eta_sec = int(remaining / speed_bps) if speed_bps > 0 else 0
        eta = f"{eta_sec}s" if eta_sec < 3600 else f"{eta_sec // 60}m"
        cn = connections if connections is not None else "?"
        msg = f"{done}/{total_h} ({pct}%) CN:{cn} DL:{speed} ETA:{eta}"
        _emit_progress(
            progress,
            msg,
            progress_pct=pct,
            progress_done=done,
            progress_total=total_h,
            progress_speed=speed,
            progress_eta=eta,
            progress_connections=cn,
        )
        return
    msg = f"{done}/? DL:{speed}"
    _emit_progress(
        progress,
        msg,
        progress_done=done,
        progress_total="?",
        progress_speed=speed,
    )


def _human_bytes(num: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(max(num, 0))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TiB"


def _handle_aria2_chunk(chunk: str, progress: Optional[Callable[..., None]]) -> None:
    for raw in re.split(r"[\r\n]+", chunk):
        line = raw.strip()
        if not line:
            continue
        match = ARIA2_PROGRESS_RX.search(line)
        if not match:
            continue
        data = match.groupdict()
        done = f"{data['done']}{data['done_unit']}"
        total = f"{data['total']}{data['total_unit']}"
        speed = f"{data['speed']}{data['speed_unit']}"
        msg = f"{done}/{total} ({data['pct']}%) CN:{data.get('cn') or '?'} DL:{speed} ETA:{data['eta']}"
        _emit_progress(
            progress,
            msg,
            progress_pct=int(data["pct"]),
            progress_done=done,
            progress_total=total,
            progress_speed=speed,
            progress_eta=data["eta"],
            progress_connections=data.get("cn"),
        )


def _civitai_request_headers(url: str) -> dict[str, str]:
    headers = {
        "User-Agent": "muse-worker/1.0 (compatible; requests)",
        "Accept": "*/*",
    }
    if CIVITAI_TOKEN and "civitai." in (url or ""):
        headers["Authorization"] = f"Bearer {CIVITAI_TOKEN}"
    return headers


def _remove_partial_target(target: Path) -> None:
    try:
        if target.exists() and not file_ok(target):
            target.unlink()
    except OSError:
        pass


def _winget_aria2_paths() -> list[Path]:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    paths: list[Path] = []
    link = local / "Microsoft" / "WinGet" / "Links" / "aria2c.exe"
    if link.is_file():
        paths.append(link)
    pkg_root = local / "Microsoft" / "WinGet" / "Packages"
    if pkg_root.is_dir():
        for pkg_dir in sorted(pkg_root.glob("aria2.aria2_*")):
            direct = pkg_dir / "aria2c.exe"
            if direct.is_file():
                paths.append(direct)
            for nested in sorted(pkg_dir.rglob("aria2c.exe")):
                if nested.is_file():
                    paths.append(nested)
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_aria2() -> str | None:
    env_path = (os.environ.get("ARIA2C_PATH") or os.environ.get("ARIA2_PATH") or "").strip()
    if env_path and Path(env_path).is_file():
        return str(Path(env_path).resolve())
    found = shutil.which("aria2c")
    if found:
        return found
    if sys.platform != "win32":
        return None
    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "aria2" / "aria2c.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "aria2" / "aria2c.exe",
        *_winget_aria2_paths(),
    ):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def aria2_available() -> bool:
    return _resolve_aria2() is not None


def download_url_aria2(
    url: str,
    target: Path,
    progress: Optional[Callable[..., None]] = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen], None] | None = None,
):
    aria2_bin = _resolve_aria2() or "aria2c"
    splits = str(ARIA2_CONNECTIONS)
    cmd = [
        aria2_bin, "-x", splits, "-s", splits,
        "--summary-interval=1",
        "--console-log-level=notice",
        "--max-redirect=10",
        "--timeout=120",
        "--file-allocation=none",
        url,
        "-d", str(target.parent),
        "-o", target.name,
    ]
    if CIVITAI_TOKEN and "civitai." in url:
        cmd[1:1] = [f"--header=Authorization: Bearer {CIVITAI_TOKEN}"]
    _safe_log_line("+ " + " ".join(str(x) for x in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc_cb:
        proc_cb(proc)
    carry = ""
    if proc.stdout is not None:
        while True:
            _abort_if_cancelled(cancel_event, proc, target)
            if proc.poll() is not None:
                piece = proc.stdout.read(512)
                if not piece:
                    break
            else:
                piece = proc.stdout.read(512)
                if not piece:
                    time.sleep(0.2)
                    continue
            carry += piece
            while True:
                split_at = -1
                for sep in ("\r", "\n"):
                    idx = carry.find(sep)
                    if idx != -1 and (split_at == -1 or idx < split_at):
                        split_at = idx
                if split_at == -1:
                    break
                line = carry[:split_at]
                carry = carry[split_at + 1 :]
                _handle_aria2_chunk(line, progress)
        if carry.strip():
            _handle_aria2_chunk(carry, progress)
    _check_cancel(cancel_event)
    code = proc.wait()
    if cancel_event is not None and cancel_event.is_set():
        _stop_proc(proc)
        _remove_partial_target(target)
        raise DownloadCancelled()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)

def add_token(url: str) -> str:
    if "civitai." not in url or "token=" in url or not CIVITAI_TOKEN:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}token={CIVITAI_TOKEN}"


_CIVITAI_VERSION_RX = re.compile(r"/api/download/models/(\d+)|/model-versions/(\d+)|[?&]modelVersionId=(\d+)")


def extract_civitai_version_id(url: str):
    m = _CIVITAI_VERSION_RX.search(url or "")
    if not m:
        return None
    for group in m.groups():
        if group:
            return int(group)
    return None


def extract_civitai_file_id(url: str):
    qs = parse_qs(urlparse(url or "").query)
    raw = (qs.get("fileId") or qs.get("fileid") or [None])[0]
    if raw and str(raw).isdigit():
        return int(raw)
    return None


def resolve_civitai_file(meta: dict, file_id: int | None = None) -> dict:
    files = meta.get("files") or []
    if not files:
        return {}
    if file_id is not None:
        match = next((f for f in files if f.get("id") == file_id), None)
        if match:
            return match
    return next((f for f in files if f.get("primary")), files[0])


def folder_key_for_path(target_dir: Path) -> str:
    target = target_dir.resolve()
    for key, root in TARGET_DIRS.items():
        root_resolved = root.resolve()
        if target == root_resolved or root_resolved in target.parents:
            return key
    embeddings = (COMFY_ROOT / "models" / "embeddings").resolve()
    if embeddings in target.parents or target.parent == embeddings:
        return "auto"
    return "checkpoints"


def civitai_target_path(url: str) -> tuple[Path, dict, dict]:
    version_id = extract_civitai_version_id(url)
    if not version_id:
        raise ValueError("Could not parse Civitai model version from URL.")
    file_id = extract_civitai_file_id(url)
    meta = civitai_lookup(version_id)
    file_entry = resolve_civitai_file(meta, file_id)
    if not file_entry:
        raise ValueError("No files found for this Civitai version.")
    target_dir = auto_target_dir(meta, file_entry)
    filename = file_entry.get("name") or f"civitai_{version_id}.safetensors"
    return target_dir / filename, meta, file_entry


def preview_civitai_url(url: str) -> dict:
    url = (url or "").strip()
    if "civitai." not in url:
        return {"ok": False, "error": "Only Civitai URLs are supported for custom download."}
    try:
        target, meta, file_entry = civitai_target_path(url)
    except Exception as e:
        return {"ok": False, "error": redact_download_secrets(str(e))}
    model_info = meta.get("model") or {}
    model_name = (model_info.get("name") or "").strip()
    version_name = (meta.get("name") or "").strip()
    display_parts = [part for part in (model_name, version_name) if part]
    size_kb = file_entry.get("sizeKB")
    size_bytes = int(size_kb * 1024) if isinstance(size_kb, (int, float)) else file_entry.get("size")
    return {
        "ok": True,
        "url": url,
        "model_name": model_name,
        "version_name": version_name,
        "display_name": " · ".join(display_parts) or (file_entry.get("name") or "Civitai model"),
        "filename": file_entry.get("name"),
        "file_size_bytes": size_bytes,
        "model_type": model_info.get("type"),
        "base_model": meta.get("baseModel"),
        "folder": folder_key_for_path(target.parent),
        "target_path": str(target),
        "civitai_version_id": meta.get("id") or extract_civitai_version_id(url),
        "civitai_file_id": extract_civitai_file_id(url),
        "already_installed": file_ok(target),
    }


def build_item_from_url(url: str) -> dict:
    return {
        "source": "civitai",
        "url": (url or "").strip(),
        "target_auto": True,
    }


def civitai_lookup(version_id: int) -> dict:
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    r = requests.get(url, headers=_civitai_request_headers(url), timeout=20)
    r.raise_for_status()
    return r.json()


def variant_from_basemodel(base_model: str) -> str:
    base = (base_model or "").strip().lower()
    if not base:
        return "unknown"
    if "flux 2" in base or "flux2" in base or "flux.2" in base:
        return "flux2"
    if "schnell" in base or "flux.1 s" in base or "flux1 s" in base:
        return "flux1_schnell"
    if "kontext" in base or "flux.1 d" in base or "flux1 d" in base or "flux 1 d" in base:
        return "flux1_dev"
    if base.startswith("flux"):
        return "flux1_dev"
    if "xl" in base or "pony" in base or "illustrious" in base or "noobai" in base:
        return "sdxl"
    if "sd 1" in base or "sd1" in base:
        return "sd15"
    return "unknown"


def auto_target_dir(meta: dict, file_entry: dict | None = None) -> Path:
    """Choose the right ComfyUI models subfolder from Civitai version metadata."""
    model_type = ((meta.get("model") or {}).get("type") or "").lower()
    base = (meta.get("baseModel") or "").lower()
    if file_entry is None:
        file_entry = resolve_civitai_file(meta)
    fmt = (file_entry.get("metadata") or {}).get("format", "").lower()
    fname = (file_entry.get("name") or "").lower()

    if model_type == "lora" or model_type == "locon":
        return TARGET_DIRS["flux_loras"] if "flux" in base else TARGET_DIRS["loras"]
    if model_type == "vae":
        return TARGET_DIRS["vae"]
    if model_type == "textualinversion":
        return COMFY_ROOT / "models" / "embeddings"
    if model_type == "upscaler":
        return TARGET_DIRS["upscalers"]
    if model_type == "checkpoint":
        if "flux" in base:
            if fname.endswith(".gguf") or fmt == "gguf":
                return TARGET_DIRS["flux_gguf_unet"]
            # All-in-one Flux safetensors usually go into checkpoints/ for CheckpointLoaderSimple
            return TARGET_DIRS["checkpoints"]
        return TARGET_DIRS["checkpoints"]
    return TARGET_DIRS["checkpoints"]


def write_civitai_sidecar(target: Path, meta: dict):
    """Drop a small sidecar JSON next to the model so the UI can read variant info later."""
    try:
        sidecar = target.with_suffix(target.suffix + ".civitai.json")
        payload = {
            "filename": target.name,
            "baseModel": meta.get("baseModel"),
            "variant": variant_from_basemodel(meta.get("baseModel") or ""),
            "model_type": ((meta.get("model") or {}).get("type") or "").lower(),
            "civitai_version_id": meta.get("id"),
            "civitai_model_id": (meta.get("modelId") or (meta.get("model") or {}).get("id")),
            "name": meta.get("name"),
            "model_name": (meta.get("model") or {}).get("name"),
        }
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  + sidecar: {sidecar.name} (variant={payload['variant']})")
    except Exception as e:
        print(f"  ! sidecar write failed: {e}")

def file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1024 * 1024


def _have_aria2() -> bool:
    return aria2_available()


def _have_curl() -> bool:
    return shutil.which("curl") is not None


def download_url_curl(
    url: str,
    target: Path,
    progress: Optional[Callable[..., None]] = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen], None] | None = None,
):
    target.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl") or "curl"
    total = _head_content_length(url)
    started = time.time()
    last_emit = 0.0
    _emit_progress(progress, f"downloading {target.name} via curl")
    cmd = [
        curl, "-fL", "--retry", "3", "--retry-delay", "2",
        "--connect-timeout", "30", "--max-time", "0",
        "-o", str(target),
    ]
    if CIVITAI_TOKEN and "civitai." in url:
        cmd.extend(["-H", f"Authorization: Bearer {CIVITAI_TOKEN}"])
    cmd.append(url)
    _safe_log_line("+ " + " ".join(redact_download_secrets(str(x)) for x in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc_cb:
        proc_cb(proc)
    while proc.poll() is None:
        _abort_if_cancelled(cancel_event, proc, target)
        now = time.time()
        if progress and now - last_emit >= 1.0:
            last_emit = now
            downloaded = target.stat().st_size if target.exists() else 0
            _emit_byte_progress(progress, downloaded, total, started, connections=1)
        time.sleep(0.25)
    if proc.stdout is not None:
        proc.stdout.read()
    _check_cancel(cancel_event)
    code = proc.wait()
    if code == 0 and progress:
        downloaded = target.stat().st_size if target.exists() else 0
        _emit_byte_progress(progress, downloaded, total or downloaded, started, connections=1)
    if cancel_event is not None and cancel_event.is_set():
        _remove_partial_target(target)
        raise DownloadCancelled()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def download_url_requests(
    url: str,
    target: Path,
    progress: Optional[Callable[..., None]] = None,
    cancel_event: threading.Event | None = None,
):
    target.parent.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress, f"downloading {target.name} via requests")
    headers = _civitai_request_headers(url)
    session = requests.Session()
    session.headers.update(headers)
    with session.get(url, stream=True, timeout=(30, 600), allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        downloaded = 0
        started = time.time()
        last_emit = 0.0
        with open(target, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                _check_cancel(cancel_event)
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if not progress or now - last_emit < 1.0:
                    continue
                last_emit = now
                _emit_byte_progress(progress, downloaded, total, started, connections=1)


def download_url(
    url: str,
    target: Path,
    progress: Optional[Callable[..., None]] = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen], None] | None = None,
):
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_ok(target):
        print(f"SKIP exists: {target}")
        return
    url = add_token(url)
    dl_kwargs = {"progress": progress, "cancel_event": cancel_event, "proc_cb": proc_cb}
    aria2_error: str | None = None
    if _have_aria2():
        _emit_progress(progress, f"downloading {target.name} via aria2")
        try:
            download_url_aria2(url, target, **dl_kwargs)
            if file_ok(target):
                return
            aria2_error = "aria2 finished but output file is missing or too small"
        except subprocess.CalledProcessError as e:
            aria2_error = f"aria2 failed (exit {e.returncode})"
        except DownloadCancelled:
            _remove_partial_target(target)
            raise
        except OSError as e:
            aria2_error = f"aria2 failed ({redact_download_secrets(str(e))})"
        _remove_partial_target(target)
        _emit_progress(progress, f"aria2 failed, retrying {target.name} via curl")

    curl_error: str | None = None
    if _have_curl():
        try:
            download_url_curl(url, target, **dl_kwargs)
            if file_ok(target):
                return
            curl_error = "curl finished but output file is missing or too small"
        except subprocess.CalledProcessError as e:
            curl_error = f"curl failed (exit {e.returncode})"
        except DownloadCancelled:
            _remove_partial_target(target)
            raise
        except OSError as e:
            curl_error = f"curl failed ({redact_download_secrets(str(e))})"
        _remove_partial_target(target)
        _emit_progress(progress, f"curl failed, retrying {target.name} via requests (slow)")

    try:
        download_url_requests(url, target, progress=progress, cancel_event=cancel_event)
    except DownloadCancelled:
        _remove_partial_target(target)
        raise
    except Exception as e:
        _remove_partial_target(target)
        parts: list[str] = []
        if aria2_error:
            parts.append(aria2_error)
        if curl_error:
            parts.append(curl_error)
        parts.append(f"requests failed ({redact_download_secrets(str(e))})")
        raise RuntimeError(f"Download failed for {target.name}: " + "; ".join(parts)) from e

    if not file_ok(target):
        parts = [p for p in (aria2_error, curl_error) if p]
        parts.append("requests finished but output file is missing or too small")
        raise RuntimeError(f"Download failed for {target.name}: " + "; ".join(parts))


def _normalize_hf_target(target: Path, repo_path: str) -> None:
    candidates = (
        target,
        target.parent / repo_path,
        target.parent / Path(repo_path).name,
    )
    for candidate in candidates:
        if not candidate.is_file() or not file_ok(candidate):
            continue
        if candidate.resolve() == target.resolve():
            return
        if target.exists():
            target.unlink()
        candidate.rename(target)
        return


def hf_download(
    repo_id: str,
    repo_path: str,
    target: Path,
    progress: Optional[Callable[..., None]] = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen], None] | None = None,
):
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_ok(target):
        print(f"SKIP exists: {target}")
        return
    _emit_progress(progress, f"downloading {target.name} from Hugging Face")
    errors: list[str] = []

    _check_cancel(cancel_event)
    try:
        from huggingface_hub import hf_hub_download

        fetched = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            local_dir=str(target.parent),
            token=HF_TOKEN or None,
        )
        fetched_path = Path(fetched)
        if fetched_path.is_file() and fetched_path.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            fetched_path.rename(target)
        _normalize_hf_target(target, repo_path)
        if file_ok(target):
            return
        errors.append("huggingface_hub finished but output file is missing or too small")
    except DownloadCancelled:
        raise
    except Exception as e:
        errors.append(f"huggingface_hub: {redact_download_secrets(str(e))}")

    cli = shutil.which("hf") or shutil.which("huggingface-cli")
    if cli:
        _check_cancel(cancel_event)
        cmd = [cli, "download", repo_id, repo_path, "--local-dir", str(target.parent)]
        if HF_TOKEN:
            cmd += ["--token", HF_TOKEN]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc_cb:
            proc_cb(proc)
        while proc.poll() is None:
            _check_cancel(cancel_event)
            time.sleep(0.25)
        _check_cancel(cancel_event)
        stdout = (proc.stdout.read() if proc.stdout else "") or ""
        stderr = (proc.stderr.read() if proc.stderr else "") or ""
        if proc.returncode == 0:
            _normalize_hf_target(target, repo_path)
            if file_ok(target):
                return
            errors.append("hf cli finished but output file is missing or too small")
        else:
            detail = redact_download_secrets((stderr or stdout).strip()) or f"exit {proc.returncode}"
            errors.append(f"hf cli: {detail}")

    hint = "Check HF_TOKEN in /workspace/.env, free disk space, and network."
    raise RuntimeError(
        f"Hugging Face download failed for {repo_id}/{repo_path}: " + "; ".join(errors) + f" {hint}"
    )


def download_item(
    item: dict,
    folder_key: str,
    progress: Optional[Callable[..., None]] = None,
    cancel_event: threading.Event | None = None,
    proc_cb: Callable[[subprocess.Popen], None] | None = None,
):
    source = item.get("source", "direct")

    civitai_meta = None
    auto_dir = None
    auto_name = None
    if "civitai." in (item.get("url") or "") or item.get("civitai_version_id"):
        version_id = item.get("civitai_version_id") or extract_civitai_version_id(item.get("url") or "")
        if version_id:
            try:
                civitai_meta = civitai_lookup(version_id)
                file_id = item.get("civitai_file_id") or extract_civitai_file_id(item.get("url") or "")
                primary = resolve_civitai_file(civitai_meta, file_id)
                auto_name = primary.get("name")
                if item.get("target_auto") or folder_key == "auto":
                    auto_dir = auto_target_dir(civitai_meta, primary)
            except Exception as e:
                print(f"  ! Civitai lookup failed for version {version_id}: {e}")

    if item.get("target"):
        target = Path(item["target"])
    elif auto_dir is not None:
        target = auto_dir / (item.get("name") or auto_name or f"civitai_{extract_civitai_version_id(item.get('url') or '')}.safetensors")
    else:
        target = TARGET_DIRS[folder_key] / item["name"]

    dl_kwargs = {"progress": progress, "cancel_event": cancel_event, "proc_cb": proc_cb}
    if source in ("direct", "civitai", "url"):
        download_url(item["url"], target, **dl_kwargs)
    elif source in ("hf", "huggingface", "huggingface_hub"):
        hf_download(item["repo_id"], item.get("repo_path", item["name"]), target, **dl_kwargs)
    else:
        raise RuntimeError(f"Unsupported source in {item}: {source}")

    if civitai_meta and target.exists():
        write_civitai_sidecar(target, civitai_meta)

def unzip_if_needed(zip_path: Path, dest: Path):
    if not zip_path.exists():
        print(f"ZIP not found: {zip_path}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Unzipping {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

def install_aria2():
    try:
        subprocess.run(["aria2c", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        run(["apt", "update"])
        run(["apt", "install", "-y", "aria2", "wget", "unzip"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="basic", help="models.json set name, e.g. basic, all, flux_user, flux_min, flux_official, flux_all")
    parser.add_argument("--models", default="models.json")
    args = parser.parse_args()

    with open(args.models, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.set not in cfg["sets"]:
        raise SystemExit(f"Unknown set '{args.set}'. Available: {', '.join(cfg['sets'].keys())}")

    selected = cfg["sets"][args.set]

    download_groups = [
        ("upscalers", "upscalers"),
        ("checkpoints", "checkpoints"),
        ("loras", "loras"),
        ("vae", "vae"),
        ("flux_diffusion_models", "flux_diffusion_models"),
        ("flux_gguf_unet", "flux_gguf_unet"),
        ("text_encoders", "text_encoders"),
        ("text_encoders_gguf", "text_encoders_gguf"),
        ("flux_loras", "flux_loras"),
        ("auto", "auto"),
    ]
    needs_aria2 = any(
        item.get("source", "direct") in ("direct", "civitai", "url")
        for key, _folder_key in download_groups
        for item in selected.get(key, [])
    )
    if needs_aria2:
        install_aria2()

    for key, folder_key in download_groups:
        for item in selected.get(key, []):
            download_item(item, folder_key)

    for group in selected.get("private_hf", []):
        if not group.get("enabled", False):
            continue
        repo_id = group["repo_id"]
        for f in group.get("files", []):
            target = Path(f["target"])
            hf_download(repo_id, f["repo_path"], target)
            if "unzip_to" in f:
                unzip_if_needed(target, Path(f["unzip_to"]))

    print("All downloads done.")

if __name__ == "__main__":
    main()
