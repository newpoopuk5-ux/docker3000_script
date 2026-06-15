import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
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
    _safe_log_line("+ " + " ".join(str(x) for x in cmd))
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
    line = redact_download_secrets(text)
    try:
        print(line)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((line + "\n").encode(encoding, errors="replace"))
        sys.stdout.flush()


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
_CIVITAI_MODEL_RX = re.compile(r"/models/(\d+)")


def normalize_civitai_url(url: str) -> str:
    return (url or "").strip().replace("civitai.red", "civitai.com")


def extract_civitai_model_id(url: str):
    m = _CIVITAI_MODEL_RX.search(normalize_civitai_url(url))
    if not m:
        return None
    return int(m.group(1))


def is_civitai_model_page_url(url: str) -> bool:
    normalized = normalize_civitai_url(url)
    if "/api/download/" in normalized:
        return False
    return extract_civitai_model_id(normalized) is not None


def build_civitai_download_url(version_id: int, file_id: int | None = None) -> str:
    base = f"https://civitai.com/api/download/models/{int(version_id)}"
    if file_id is not None:
        return f"{base}?fileId={int(file_id)}"
    return base


def _sanitize_filename_part(text: str, *, max_len: int = 120) -> str:
    cleaned = str(text or "")
    kept: list[str] = []
    for ch in cleaned:
        cat = unicodedata.category(ch)
        if cat in ("So", "Sk", "Cs", "Co", "Cn", "Cf"):
            continue
        kept.append(ch)
    cleaned = "".join(kept)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        return "model"
    return cleaned[:max_len]


def civitai_pretty_filename(model_name: str, version_name: str, original_filename: str) -> str:
    ext = Path(str(original_filename or "model.safetensors")).suffix or ".safetensors"
    model_part = _sanitize_filename_part(model_name)
    version_part = _sanitize_filename_part(version_name)
    if version_part and version_part.lower() not in model_part.lower():
        return f"{model_part} {version_part}{ext}"
    return f"{model_part}{ext}"


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


def civitai_target_path(
    url: str,
    folder_key: str | None = None,
    *,
    save_as: str | None = None,
) -> tuple[Path, dict, dict]:
    version_id = extract_civitai_version_id(url)
    if not version_id:
        raise ValueError("Could not parse Civitai model version from URL.")
    file_id = extract_civitai_file_id(url)
    meta = civitai_lookup(version_id)
    file_entry = resolve_civitai_file(meta, file_id)
    if not file_entry:
        raise ValueError("No files found for this Civitai version.")
    key = (folder_key or "").strip()
    if key and key != "auto" and key in TARGET_DIRS:
        target_dir = TARGET_DIRS[key]
    else:
        target_dir = auto_target_dir(meta, file_entry)
    raw_name = file_entry.get("name") or f"civitai_{version_id}.safetensors"
    model_info = meta.get("model") or {}
    pretty = save_as or civitai_pretty_filename(
        (model_info.get("name") or "").strip(),
        (meta.get("name") or "").strip(),
        raw_name,
    )
    return target_dir / pretty, meta, file_entry


CIVITAI_PREVIEW_LIMIT = 3
CIVITAI_PRIMARY_PREVIEW_SLOT = 2
CIVITAI_PREVIEW_MAX_BYTES = 12 * 1024 * 1024
CATALOG_PREVIEW_DIR = (COMFY_ROOT / "models" / ".catalog_previews").resolve()
PREVIEW_MANIFEST_NAME = ".preview.json"

_PREVIEW_IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".webp")


def civitai_preview_cache_dir(version_id: int) -> Path:
    return CATALOG_PREVIEW_DIR / str(int(version_id))


def _model_stem(model_name: str) -> str:
    stem = Path(str(model_name or "model").strip()).stem
    return stem or "model"


def _named_preview_base(stem: str, slot: int) -> str:
    slot = int(slot)
    if slot == CIVITAI_PRIMARY_PREVIEW_SLOT:
        return stem
    if slot == 1:
        return f"{stem}.preview1"
    if slot == CIVITAI_PRIMARY_PREVIEW_SLOT + 1:
        return f"{stem}.alt"
    return f"{stem}.preview{slot}"


def _read_preview_manifest(dest_dir: Path) -> dict:
    path = dest_dir / PREVIEW_MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_preview_manifest(
    dest_dir: Path,
    *,
    model_name: str,
    primary: str | None,
    fallback: str | None,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "model_stem": _model_stem(model_name),
        "primary": primary,
        "fallback": fallback,
    }
    (dest_dir / PREVIEW_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _preview_slot_basename(slot: int) -> str:
    return f"{int(slot):02d}"


def _find_legacy_preview_file(dest_dir: Path, slot: int) -> Path | None:
    if not dest_dir.is_dir():
        return None
    prefix = _preview_slot_basename(slot)
    matches = [
        p for p in dest_dir.iterdir()
        if p.is_file()
        and p.name.startswith(prefix)
        and p.suffix.lower() in _PREVIEW_IMAGE_SUFFIXES
        and p.stat().st_size > 512
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[0]


def _find_named_preview_file(dest_dir: Path, stem: str, slot: int) -> Path | None:
    if not dest_dir.is_dir():
        return None
    want_stem = _named_preview_base(stem, slot)
    matches = [
        p for p in dest_dir.iterdir()
        if p.is_file()
        and p.stem == want_stem
        and p.suffix.lower() in _PREVIEW_IMAGE_SUFFIXES
        and p.stat().st_size > 512
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[0]


def _find_any_cached_preview_file(dest_dir: Path, *, alt: bool = False) -> Path | None:
    if not dest_dir.is_dir():
        return None
    matches: list[Path] = []
    for path in dest_dir.iterdir():
        if not path.is_file() or path.name == PREVIEW_MANIFEST_NAME:
            continue
        if path.suffix.lower() not in _PREVIEW_IMAGE_SUFFIXES:
            continue
        if not _is_valid_cached_preview(path):
            continue
        is_alt = path.stem.endswith(".alt")
        if alt and not is_alt:
            continue
        if not alt and is_alt:
            continue
        matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda p: p.name)
    return matches[0]


def _resolve_preview_file(dest_dir: Path, slot: int, model_name: str | None = None) -> Path | None:
    manifest = _read_preview_manifest(dest_dir)
    key = "primary" if slot == CIVITAI_PRIMARY_PREVIEW_SLOT else "fallback"
    listed = str(manifest.get(key) or "").strip()
    if listed:
        path = dest_dir / Path(listed).name
        if path.is_file() and _is_valid_cached_preview(path):
            return path
    stem = str(manifest.get("model_stem") or "").strip() or (
        _model_stem(model_name) if model_name else ""
    )
    if stem:
        found = _find_named_preview_file(dest_dir, stem, slot)
        if found and _is_valid_cached_preview(found):
            return found
    legacy = _find_legacy_preview_file(dest_dir, slot)
    if legacy and _is_valid_cached_preview(legacy):
        return legacy
    if slot == CIVITAI_PRIMARY_PREVIEW_SLOT:
        return _find_any_cached_preview_file(dest_dir, alt=False)
    if slot == CIVITAI_PRIMARY_PREVIEW_SLOT + 1:
        return _find_any_cached_preview_file(dest_dir, alt=True)
    return None


def civitai_model_filename(meta: dict, file_id: int | None = None) -> str:
    file_entry = resolve_civitai_file(meta, file_id)
    version_id = meta.get("id") or "unknown"
    return str(file_entry.get("name") or f"civitai_{version_id}.safetensors")


def primary_cached_preview_filename(version_id: int, model_name: str | None = None) -> str | None:
    dest = civitai_preview_cache_dir(int(version_id))
    for slot in (CIVITAI_PRIMARY_PREVIEW_SLOT, 1, 3, CIVITAI_PRIMARY_PREVIEW_SLOT + 1):
        path = _resolve_preview_file(dest, slot, model_name)
        if path and _is_valid_cached_preview(path):
            return path.name
    return None


def fallback_cached_preview_filename(version_id: int, model_name: str | None = None) -> str | None:
    path = _resolve_preview_file(
        civitai_preview_cache_dir(int(version_id)),
        CIVITAI_PRIMARY_PREVIEW_SLOT + 1,
        model_name,
    )
    return path.name if path else None


def first_cached_preview_filename(version_id: int, model_name: str | None = None) -> str | None:
    return primary_cached_preview_filename(version_id, model_name)


def civitai_preview_cache_ready(version_id: int, model_name: str | None = None) -> bool:
    del model_name
    return primary_cached_preview_filename(int(version_id), None) is not None


def civitai_preview_file_path(version_id: int, filename: str) -> Path | None:
    safe = Path(str(filename or "")).name
    if not safe or safe != filename:
        return None
    path = civitai_preview_cache_dir(int(version_id)) / safe
    return path if path.is_file() else None


def _looks_like_displayable_image(path: Path) -> bool:
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return False
    if len(head) >= 3 and head[:3] == b"\xff\xd8\xff":
        return True
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return False
    return False


def _is_valid_cached_preview(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size <= 512 or size > CIVITAI_PREVIEW_MAX_BYTES:
        return False
    return _looks_like_displayable_image(path)


def civitai_preview_entries(meta: dict, limit: int = CIVITAI_PREVIEW_LIMIT) -> list[dict]:
    limit = max(0, int(limit))
    if limit == 0:
        return []
    gallery = _normalize_civitai_gallery(meta)
    if not gallery:
        return []
    gallery = [row for row in gallery if not row.get("animated")]
    fetch = min(limit, 3, len(gallery))
    return [
        {key: value for key, value in row.items() if key != "animated"}
        for row in gallery[:fetch]
    ]


def _is_animated_civitai_image(img: dict) -> bool:
    if not isinstance(img, dict):
        return True
    kind = str(img.get("type") or "").strip().lower()
    if kind in ("video", "animated", "animation"):
        return True
    url = str(img.get("url") or "").lower()
    if any(token in url for token in (".gif", ".mp4", ".webm", ".mov", ".m4v")):
        return True
    mime = str(img.get("mimeType") or img.get("mime") or "").strip().lower()
    if mime.startswith("video/") or mime == "image/gif":
        return True
    return False


def _normalize_civitai_gallery(meta: dict) -> list[dict]:
    out: list[dict] = []
    for img in meta.get("images") or []:
        if not isinstance(img, dict):
            continue
        url = str(img.get("url") or "").strip()
        if not url:
            continue
        out.append({
            "url": url,
            "width": img.get("width"),
            "height": img.get("height"),
            "type": img.get("type"),
            "animated": _is_animated_civitai_image(img),
        })
    return out


def _pick_primary_civitai_preview(meta: dict) -> dict | None:
    """Prefer Civitai gallery image #2; fall back through #1/#3 then any static image."""
    gallery = _normalize_civitai_gallery(meta)
    if not gallery:
        return None
    order: list[dict] = []
    if len(gallery) >= 2:
        order.append(gallery[1])
    if gallery:
        order.append(gallery[0])
    if len(gallery) >= 3:
        order.append(gallery[2])
    order.extend(gallery)
    seen: set[str] = set()
    for row in order:
        url = str(row.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        if row["animated"]:
            continue
        return {key: value for key, value in row.items() if key != "animated"}
    for row in gallery:
        if not row["animated"]:
            return {key: value for key, value in row.items() if key != "animated"}
    return None


def civitai_preview_dir_for(target: Path) -> Path:
    return target.parent / f"{target.stem}.previews"


def _preview_image_ext(url: str) -> str:
    lower = (url or "").lower().split("?")[0]
    if lower.endswith(".png") or ".png/" in lower:
        return ".png"
    if lower.endswith(".webp") or ".webp/" in lower:
        return ".webp"
    if lower.endswith(".gif") or ".gif/" in lower:
        return ".gif"
    if lower.endswith(".jpg") or lower.endswith(".jpeg") or ".jpg/" in lower or ".jpeg/" in lower:
        return ".jpeg"
    return ".jpeg"


def _maybe_migrate_legacy_previews(dest_dir: Path, model_name: str) -> None:
    if _read_preview_manifest(dest_dir):
        return
    stem = _model_stem(model_name)
    primary_fn: str | None = None
    fallback_fn: str | None = None
    for slot in (CIVITAI_PRIMARY_PREVIEW_SLOT, CIVITAI_PRIMARY_PREVIEW_SLOT + 1):
        legacy = _find_legacy_preview_file(dest_dir, slot)
        if not legacy or _find_named_preview_file(dest_dir, stem, slot):
            if slot == CIVITAI_PRIMARY_PREVIEW_SLOT:
                existing = _find_named_preview_file(dest_dir, stem, slot)
                if existing:
                    primary_fn = existing.name
            else:
                existing = _find_named_preview_file(dest_dir, stem, slot)
                if existing:
                    fallback_fn = existing.name
            continue
        new_path = dest_dir / f"{_named_preview_base(stem, slot)}{legacy.suffix}"
        try:
            if not new_path.exists():
                legacy.rename(new_path)
            else:
                legacy.unlink(missing_ok=True)
            if slot == CIVITAI_PRIMARY_PREVIEW_SLOT:
                primary_fn = new_path.name
            else:
                fallback_fn = new_path.name
        except OSError:
            if legacy.is_file():
                if slot == CIVITAI_PRIMARY_PREVIEW_SLOT:
                    primary_fn = legacy.name
                else:
                    fallback_fn = legacy.name
    if primary_fn or fallback_fn:
        _write_preview_manifest(dest_dir, model_name=model_name, primary=primary_fn, fallback=fallback_fn)


def _download_civitai_preview_entries(
    entries: list[dict],
    dest_dir: Path,
    model_name: str,
) -> list[str]:
    if not entries:
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(model_name or "model.safetensors").strip() or "model.safetensors"
    _maybe_migrate_legacy_previews(dest_dir, model_name)
    stem = _model_stem(model_name)
    saved: list[str] = []
    primary_fn: str | None = None
    fallback_fn: str | None = None
    for idx, entry in enumerate(entries, start=1):
        url = entry.get("url") or ""
        if _is_animated_civitai_image({"url": url, "type": entry.get("type")}):
            continue
        existing = _resolve_preview_file(dest_dir, idx, model_name)
        if existing:
            if not _is_valid_cached_preview(existing):
                try:
                    existing.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                saved.append(str(existing))
                if idx == CIVITAI_PRIMARY_PREVIEW_SLOT:
                    primary_fn = existing.name
                elif idx == CIVITAI_PRIMARY_PREVIEW_SLOT + 1:
                    fallback_fn = existing.name
                continue
        ext = _preview_image_ext(url)
        out_path = dest_dir / f"{_named_preview_base(stem, idx)}{ext}"
        try:
            r = requests.get(url, headers=_civitai_request_headers(url), timeout=90, stream=True)
            r.raise_for_status()
            content_len = r.headers.get("Content-Length")
            if content_len:
                try:
                    if int(content_len) > CIVITAI_PREVIEW_MAX_BYTES:
                        continue
                except (TypeError, ValueError):
                    pass
            downloaded = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(65536):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > CIVITAI_PREVIEW_MAX_BYTES:
                        break
                    f.write(chunk)
            if downloaded > CIVITAI_PREVIEW_MAX_BYTES or not _is_valid_cached_preview(out_path):
                out_path.unlink(missing_ok=True)
                continue
            saved.append(str(out_path))
            if idx == CIVITAI_PRIMARY_PREVIEW_SLOT:
                primary_fn = out_path.name
            elif idx == CIVITAI_PRIMARY_PREVIEW_SLOT + 1:
                fallback_fn = out_path.name
        except Exception as e:
            _safe_log_line(f"  ! preview {out_path.name} failed: {redact_download_secrets(str(e))}")
    slot_names: dict[int, str] = {}
    for idx in (1, 2, 3):
        hit = _resolve_preview_file(dest_dir, idx, model_name)
        if hit and _is_valid_cached_preview(hit):
            slot_names[idx] = hit.name
    primary_fn = slot_names.get(CIVITAI_PRIMARY_PREVIEW_SLOT) or slot_names.get(1) or slot_names.get(3)
    fallback_fn = slot_names.get(CIVITAI_PRIMARY_PREVIEW_SLOT + 1) or slot_names.get(1) or slot_names.get(3)
    if primary_fn and fallback_fn == primary_fn:
        fallback_fn = None
    if saved:
        _safe_log_line(f"  + previews: {len(saved)} saved under {dest_dir.name}/ ({_sanitize_filename_part(model_name)})")
    if primary_fn or fallback_fn:
        _write_preview_manifest(dest_dir, model_name=model_name, primary=primary_fn, fallback=fallback_fn)
    return saved


def download_civitai_previews_to_cache(
    meta: dict,
    model_name: str | None = None,
    limit: int = CIVITAI_PREVIEW_LIMIT,
) -> list[str]:
    version_id = meta.get("id")
    if not version_id:
        return []
    resolved_name = (model_name or civitai_model_filename(meta)).strip()
    entries = civitai_preview_entries(meta, limit=limit)
    return _download_civitai_preview_entries(
        entries,
        civitai_preview_cache_dir(int(version_id)),
        resolved_name,
    )


def download_civitai_preview_images(
    target: Path,
    meta: dict,
    limit: int = CIVITAI_PREVIEW_LIMIT,
) -> list[str]:
    entries = civitai_preview_entries(meta, limit=limit)
    return _download_civitai_preview_entries(entries, civitai_preview_dir_for(target), target.name)


def _file_size_bytes(file_entry: dict) -> int | None:
    size_kb = file_entry.get("sizeKB")
    if isinstance(size_kb, (int, float)):
        return int(size_kb * 1024)
    raw = file_entry.get("size")
    return int(raw) if isinstance(raw, (int, float)) else None


def _preview_from_version_meta(
    meta: dict,
    *,
    source_url: str,
    folder_key: str | None = None,
    file_id: int | None = None,
    source_kind: str = "download_url",
    model_id: int | None = None,
    main_url: str | None = None,
    versions: list[dict] | None = None,
) -> dict:
    file_entry = resolve_civitai_file(meta, file_id)
    if not file_entry:
        return {"ok": False, "error": "No files found for this Civitai version."}
    model_info = meta.get("model") or {}
    model_name = (model_info.get("name") or "").strip()
    version_name = (meta.get("name") or "").strip()
    display_parts = [part for part in (model_name, version_name) if part]
    raw_filename = file_entry.get("name") or civitai_model_filename(meta, file_id)
    pretty_filename = civitai_pretty_filename(model_name, version_name, raw_filename)
    version_id = meta.get("id") or extract_civitai_version_id(source_url)
    resolved_file_id = file_id if file_id is not None else file_entry.get("id")
    download_url = build_civitai_download_url(int(version_id), resolved_file_id) if version_id else source_url
    try:
        target, _, _ = civitai_target_path(
            download_url,
            folder_key,
            save_as=pretty_filename,
        )
    except Exception as e:
        return {"ok": False, "error": redact_download_secrets(str(e))}
    suggested_dir = auto_target_dir(meta, file_entry)
    selected_key = (folder_key or "").strip() or "auto"
    preview_images = civitai_preview_entries(meta, limit=CIVITAI_PREVIEW_LIMIT)
    picked = preview_images[0] if preview_images else None
    preview_remote_url = (picked or {}).get("url")
    if version_id and preview_images:
        try:
            download_civitai_previews_to_cache(meta, model_name=pretty_filename, limit=CIVITAI_PREVIEW_LIMIT)
        except Exception as e:
            _safe_log_line(f"  ! lookup preview cache failed: {redact_download_secrets(str(e))}")
    vid = int(version_id) if version_id else None
    primary_file = primary_cached_preview_filename(vid) if vid else None
    cached_previews: list[dict] = []
    if vid and primary_file:
        cached_previews.append({
            "slot": CIVITAI_PRIMARY_PREVIEW_SLOT,
            "filename": primary_file,
            "width": picked.get("width") if picked else None,
            "height": picked.get("height") if picked else None,
        })
    payload = {
        "ok": True,
        "source_kind": source_kind,
        "url": source_url,
        "download_url": download_url,
        "model_name": model_name,
        "version_name": version_name,
        "display_name": " · ".join(display_parts) or pretty_filename,
        "filename": raw_filename,
        "pretty_filename": pretty_filename,
        "file_size_bytes": _file_size_bytes(file_entry),
        "model_type": model_info.get("type"),
        "file_type": file_entry.get("type"),
        "base_model": meta.get("baseModel"),
        "folder": folder_key_for_path(target.parent),
        "suggested_folder": folder_key_for_path(suggested_dir),
        "selected_folder": selected_key,
        "target_path": str(target),
        "civitai_version_id": vid,
        "civitai_file_id": resolved_file_id,
        "already_installed": file_ok(target),
        "preview_thumb_file": primary_file,
        "preview_fallback_file": None,
        "preview_remote_url": preview_remote_url,
        "preview_images": cached_previews,
        "preview_image_count": len(cached_previews),
    }
    if model_id is not None:
        payload["model_id"] = model_id
    if main_url:
        payload["main_url"] = main_url
    if versions is not None:
        payload["versions"] = versions
        payload["selected_version_id"] = vid
    return payload


def preview_civitai_url(url: str, folder_key: str | None = None) -> dict:
    url = normalize_civitai_url(url)
    if "civitai." not in url:
        return {"ok": False, "error": "Only Civitai URLs are supported for custom download."}
    if is_civitai_model_page_url(url):
        return preview_civitai_model_page(url, folder_key)
    try:
        version_id = extract_civitai_version_id(url)
        if not version_id:
            raise ValueError("Could not parse Civitai model version from URL.")
        meta = civitai_lookup(version_id)
    except Exception as e:
        return {"ok": False, "error": redact_download_secrets(str(e))}
    return _preview_from_version_meta(
        meta,
        source_url=url,
        folder_key=folder_key,
        file_id=extract_civitai_file_id(url),
        source_kind="download_url",
    )


def preview_civitai_version(
    version_id: int,
    file_id: int | None = None,
    folder_key: str | None = None,
) -> dict:
    try:
        meta = civitai_lookup(int(version_id))
    except Exception as e:
        return {"ok": False, "error": redact_download_secrets(str(e))}
    download_url = build_civitai_download_url(int(version_id), file_id)
    return _preview_from_version_meta(
        meta,
        source_url=download_url,
        folder_key=folder_key,
        file_id=file_id,
        source_kind="model_page",
        model_id=(meta.get("model") or {}).get("id"),
    )


def preview_civitai_model_page(url: str, folder_key: str | None = None) -> dict:
    url = normalize_civitai_url(url)
    model_id = extract_civitai_model_id(url)
    if not model_id:
        return {"ok": False, "error": "Could not parse Civitai model id from URL."}
    try:
        model_data = civitai_lookup_model(model_id)
    except Exception as e:
        return {"ok": False, "error": redact_download_secrets(str(e))}
    raw_versions = list(model_data.get("modelVersions") or [])
    if not raw_versions:
        return {"ok": False, "error": "This Civitai model has no published versions."}
    raw_versions.sort(key=lambda row: str(row.get("createdAt") or ""), reverse=True)
    latest_id = raw_versions[0].get("id")
    hint_id = extract_civitai_version_id(url)
    selected = next((row for row in raw_versions if row.get("id") == hint_id), None) or raw_versions[0]
    version_options: list[dict] = []
    for row in raw_versions:
        file_entry = resolve_civitai_file(row)
        version_options.append({
            "id": row.get("id"),
            "name": row.get("name"),
            "base_model": row.get("baseModel"),
            "created_at": row.get("createdAt"),
            "file_size_bytes": _file_size_bytes(file_entry) if file_entry else None,
            "is_latest": row.get("id") == latest_id,
        })
    model_name = (model_data.get("name") or "").strip()
    model_type = model_data.get("type")
    main_url = f"https://civitai.com/models/{model_id}"
    enriched = dict(selected)
    enriched["model"] = {
        "id": model_id,
        "name": model_name,
        "type": model_type,
    }
    preview = _preview_from_version_meta(
        enriched,
        source_url=url,
        folder_key=folder_key,
        file_id=None,
        source_kind="model_page",
        model_id=model_id,
        main_url=main_url,
        versions=version_options,
    )
    if preview.get("ok"):
        preview["model_type"] = model_type
    return preview


def build_item_from_url(url: str, folder_key: str | None = None, name: str | None = None) -> dict:
    key = (folder_key or "").strip()
    item = {
        "source": "civitai",
        "url": normalize_civitai_url(url),
    }
    if name:
        item["name"] = str(name).strip()
    if key and key != "auto" and key in TARGET_DIRS:
        item["folder_key"] = key
    else:
        item["target_auto"] = True
    return item


def civitai_lookup_model(model_id: int) -> dict:
    url = f"https://civitai.com/api/v1/models/{int(model_id)}"
    r = requests.get(url, headers=_civitai_request_headers(url), timeout=20)
    r.raise_for_status()
    return r.json()


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
    file_type = (file_entry.get("type") or "").strip().lower()

    if file_type == "vae":
        return TARGET_DIRS["vae"]
    if file_type in ("text encoder", "text_encoder"):
        return TARGET_DIRS["text_encoders"]
    if base == "anima" and model_type == "checkpoint" and file_type == "model":
        return TARGET_DIRS["flux_diffusion_models"]

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


def write_civitai_sidecar(target: Path, meta: dict, preview_paths: list[str] | None = None):
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
            "preview_images": civitai_preview_entries(meta),
            "preview_paths": preview_paths or [],
        }
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _safe_log_line(f"  + sidecar: {sidecar.name} (variant={payload['variant']})")
    except Exception as e:
        _safe_log_line(f"  ! sidecar write failed: {e}")

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
        _safe_log_line(f"SKIP exists: {target}")
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


def _hf_file_total_bytes(repo_id: str, repo_path: str) -> int:
    try:
        from huggingface_hub import get_hf_file_metadata

        meta = get_hf_file_metadata(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="model",
            token=HF_TOKEN or None,
        )
        return int(getattr(meta, "size", 0) or 0)
    except Exception:
        try:
            from huggingface_hub import hf_hub_url

            url = hf_hub_url(repo_id=repo_id, filename=repo_path, repo_type="model")
            if HF_TOKEN:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}token={HF_TOKEN}"
            return _head_content_length(url)
        except Exception:
            return 0


def _poll_hf_partial_bytes(target: Path, repo_path: str) -> int:
    rel_name = Path(repo_path).name
    candidates = [
        target,
        target.parent / repo_path,
        target.parent / rel_name,
    ]
    best = 0
    seen: set[str] = set()
    for path in candidates:
        for variant in (path, Path(f"{path}.incomplete")):
            key = str(variant)
            if key in seen:
                continue
            seen.add(key)
            try:
                if variant.is_file():
                    best = max(best, variant.stat().st_size)
            except OSError:
                pass
    try:
        for path in target.parent.rglob("*.incomplete"):
            if path.is_file():
                best = max(best, path.stat().st_size)
    except OSError:
        pass
    return best


def _watch_hf_download_progress(
    progress: Optional[Callable[..., None]],
    target: Path,
    repo_path: str,
    total: int,
    cancel_event: threading.Event | None,
    *,
    thread: threading.Thread | None = None,
    proc: subprocess.Popen | None = None,
) -> None:
    started = time.time()
    last_emit = 0.0
    while True:
        _check_cancel(cancel_event)
        alive = thread.is_alive() if thread is not None else proc is not None and proc.poll() is None
        if not alive:
            break
        now = time.time()
        if progress and now - last_emit >= 1.0:
            last_emit = now
            downloaded = _poll_hf_partial_bytes(target, repo_path)
            _emit_byte_progress(progress, downloaded, total, started, connections=1)
        time.sleep(0.25)
    if progress:
        downloaded = _poll_hf_partial_bytes(target, repo_path)
        if downloaded > 0:
            _emit_byte_progress(progress, downloaded, total or downloaded, started, connections=1)


def _hf_hub_download_worker(repo_id: str, repo_path: str, target: Path, out: dict) -> None:
    try:
        from huggingface_hub import hf_hub_download

        out["fetched"] = hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            local_dir=str(target.parent),
            token=HF_TOKEN or None,
        )
    except Exception as e:
        out["error"] = e


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
        _safe_log_line(f"SKIP exists: {target}")
        return
    _emit_progress(progress, f"downloading {target.name} from Hugging Face")
    errors: list[str] = []
    total = _hf_file_total_bytes(repo_id, repo_path)

    _check_cancel(cancel_event)
    try:
        hub_out: dict = {}
        worker = threading.Thread(
            target=_hf_hub_download_worker,
            args=(repo_id, repo_path, target, hub_out),
            daemon=True,
        )
        worker.start()
        _watch_hf_download_progress(
            progress,
            target,
            repo_path,
            total,
            cancel_event,
            thread=worker,
        )
        worker.join()
        _check_cancel(cancel_event)
        if hub_out.get("error") is not None:
            raise hub_out["error"]
        fetched = hub_out.get("fetched")
        if fetched:
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
        _watch_hf_download_progress(
            progress,
            target,
            repo_path,
            total,
            cancel_event,
            proc=proc,
        )
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

    folder_override = str(item.get("folder_key") or "").strip()
    if folder_override in TARGET_DIRS:
        auto_dir = TARGET_DIRS[folder_override]

    if item.get("target"):
        target = Path(item["target"])
    elif auto_dir is not None:
        fallback = auto_name or f"civitai_{extract_civitai_version_id(item.get('url') or '')}.safetensors"
        target = auto_dir / (item.get("name") or fallback)
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
        preview_paths = download_civitai_preview_images(target, civitai_meta)
        write_civitai_sidecar(target, civitai_meta, preview_paths=preview_paths)

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
    parser.add_argument("--set", default="anime_starter", help="models.json set name, e.g. anime_starter, all, flux_user, flux_min, flux_official, flux_all")
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
