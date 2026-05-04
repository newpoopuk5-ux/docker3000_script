import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import requests
import zipfile

COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/workspace/ComfyUI"))
CIVITAI_TOKEN = os.environ.get("CIVITAI_TOKEN", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

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


def civitai_lookup(version_id: int) -> dict:
    headers = {"Authorization": f"Bearer {CIVITAI_TOKEN}"} if CIVITAI_TOKEN else {}
    r = requests.get(f"https://civitai.com/api/v1/model-versions/{version_id}",
                     headers=headers, timeout=20)
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


def auto_target_dir(meta: dict) -> Path:
    """Choose the right ComfyUI models subfolder from Civitai version metadata."""
    model_type = ((meta.get("model") or {}).get("type") or "").lower()
    base = (meta.get("baseModel") or "").lower()
    files = meta.get("files") or []
    primary = next((f for f in files if f.get("primary")), files[0] if files else {})
    fmt = (primary.get("metadata") or {}).get("format", "").lower()
    fname = (primary.get("name") or "").lower()

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

def download_url(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_ok(target):
        print(f"SKIP exists: {target}")
        return
    url = add_token(url)
    run(["aria2c", "-x", "8", "-s", "8", url, "-d", str(target.parent), "-o", target.name])

def hf_download(repo_id: str, repo_path: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if file_ok(target):
        print(f"SKIP exists: {target}")
        return
    cmd = ["huggingface-cli", "download", repo_id, repo_path, "--local-dir", str(target.parent), "--local-dir-use-symlinks", "False"]
    if HF_TOKEN:
        cmd += ["--token", HF_TOKEN]
    run(cmd)
    downloaded = target.parent / repo_path
    flat_downloaded = target.parent / Path(repo_path).name
    if downloaded.exists() and downloaded != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.rename(target)
    elif flat_downloaded.exists() and flat_downloaded != target:
        flat_downloaded.rename(target)

def download_item(item: dict, folder_key: str):
    source = item.get("source", "direct")

    civitai_meta = None
    auto_dir = None
    auto_name = None
    if "civitai." in (item.get("url") or "") or item.get("civitai_version_id"):
        version_id = item.get("civitai_version_id") or extract_civitai_version_id(item.get("url") or "")
        if version_id:
            try:
                civitai_meta = civitai_lookup(version_id)
                files = civitai_meta.get("files") or []
                primary = next((f for f in files if f.get("primary")), files[0] if files else {})
                auto_name = primary.get("name")
                if item.get("target_auto") or folder_key == "auto":
                    auto_dir = auto_target_dir(civitai_meta)
            except Exception as e:
                print(f"  ! Civitai lookup failed for version {version_id}: {e}")

    if item.get("target"):
        target = Path(item["target"])
    elif auto_dir is not None:
        target = auto_dir / (item.get("name") or auto_name or f"civitai_{extract_civitai_version_id(item.get('url') or '')}.safetensors")
    else:
        target = TARGET_DIRS[folder_key] / item["name"]

    if source in ("direct", "civitai", "url"):
        download_url(item["url"], target)
    elif source in ("hf", "huggingface", "huggingface_hub"):
        hf_download(item["repo_id"], item.get("repo_path", item["name"]), target)
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
