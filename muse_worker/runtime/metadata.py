import json
from datetime import datetime
from pathlib import Path

from config import FAVORITES_PATH, IMAGE_SUFFIXES, OUTPUT_DIR

try:
    from PIL import Image
except Exception:
    Image = None

IMAGE_INFO_CACHE = {}


def list_files(folder: Path, exts, recursive=False):
    if not folder.exists():
        return []
    files = folder.rglob("*") if recursive else folder.iterdir()
    return sorted([
        p.relative_to(folder).as_posix() if recursive else p.name
        for p in files
        if p.is_file() and p.name.lower().endswith(exts)
    ])


def safe_output_path(filename: str):
    if not filename:
        return None
    path = (OUTPUT_DIR / filename).resolve()
    try:
        path.relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return path


def read_favorites():
    try:
        with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(str(x) for x in data)
    except Exception:
        pass
    return set()


def write_favorites(favorites):
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(favorites), f, indent=2)


def image_records(limit=200):
    if not OUTPUT_DIR.exists():
        return []
    favorites = read_favorites()
    files = [p for p in OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    records = []
    for p in files[:limit]:
        stat = p.stat()
        records.append({
            "filename": p.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "modified_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"/local_output/{p.name}",
            "download_url": f"/download_output/{p.name}",
            "favorite": p.name in favorites,
        })
    return records


def read_image_metadata(path: Path):
    if Image is None or not path.exists():
        return {}
    try:
        with Image.open(path) as img:
            info = dict(img.info or {})
        clean = {}
        for k, v in info.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean
    except Exception:
        return {}


def maybe_json(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def link_node_id(value):
    if isinstance(value, list) and value:
        return str(value[0])
    return None


def parse_comfy_metadata(meta: dict):
    result = {
        "positive": "",
        "negative": "",
        "model": "",
        "loras": [],
        "seed": "",
        "steps": "",
        "cfg": "",
        "sampler_name": "",
        "scheduler": "",
        "width": "",
        "height": "",
        "raw": meta,
    }

    parsed_prompt = maybe_json(meta.get("prompt"))
    parsed_workflow = maybe_json(meta.get("workflow"))
    if parsed_prompt is None and isinstance(meta.get("prompt"), str):
        result["positive"] = meta.get("prompt", "")[:4000]

    if isinstance(parsed_prompt, dict):
        sampler_inputs = {}
        positive_node_id = None
        negative_node_id = None

        for node_id, node in parsed_prompt.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {}) or {}
            if class_type in ("KSampler", "KSamplerAdvanced") and not sampler_inputs:
                sampler_inputs = inputs
                positive_node_id = link_node_id(inputs.get("positive"))
                negative_node_id = link_node_id(inputs.get("negative"))
                result["seed"] = inputs.get("seed", result["seed"])
                result["steps"] = inputs.get("steps", result["steps"])
                result["cfg"] = inputs.get("cfg", result["cfg"])
                result["sampler_name"] = inputs.get("sampler_name", result["sampler_name"])
                result["scheduler"] = inputs.get("scheduler", result["scheduler"])

        for node_id, node in parsed_prompt.items():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {}) or {}
            title = ((node.get("_meta", {}) or {}).get("title") or "").lower()

            if class_type == "CheckpointLoaderSimple":
                result["model"] = inputs.get("ckpt_name", result["model"])

            if class_type == "LoraLoader":
                lora_name = inputs.get("lora_name")
                if lora_name:
                    lora_info = {
                        "lora_name": lora_name,
                        "name": lora_name,
                        "strength_model": inputs.get("strength_model", ""),
                        "strength_clip": inputs.get("strength_clip", ""),
                    }
                    if lora_info not in result["loras"]:
                        result["loras"].append(lora_info)

            if class_type == "EmptyLatentImage":
                result["width"] = inputs.get("width", result["width"])
                result["height"] = inputs.get("height", result["height"])

            if class_type == "VAELoader":
                result["vae"] = inputs.get("vae_name", "")

            if class_type == "CLIPLoader":
                clip_name = inputs.get("clip_name", "")
                if "t5" in clip_name.lower():
                    result["t5xxl"] = clip_name
                else:
                    result["clip_l"] = clip_name

            if class_type == "DualCLIPLoader":
                result["clip_l"] = inputs.get("clip_name1", "")
                result["t5xxl"] = inputs.get("clip_name2", "")
                
            if class_type == "UNETLoader":
                result["model"] = inputs.get("unet_name", result["model"])

            if class_type == "CLIPTextEncode":
                txt = inputs.get("text", "")
                is_positive = str(node_id) == positive_node_id or "positive" in title
                is_negative = str(node_id) == negative_node_id or "negative" in title
                if is_negative and not result["negative"]:
                    result["negative"] = txt
                elif is_positive and not result["positive"]:
                    result["positive"] = txt

        if not result["positive"] or not result["negative"]:
            for node in parsed_prompt.values():
                if node.get("class_type") != "CLIPTextEncode":
                    continue
                inputs = node.get("inputs", {}) or {}
                txt = inputs.get("text", "")
                title = ((node.get("_meta", {}) or {}).get("title") or "").lower()
                if "negative" in title and not result["negative"]:
                    result["negative"] = txt
                elif not result["positive"]:
                    result["positive"] = txt

    if isinstance(parsed_workflow, dict):
        result["raw"] = dict(meta)
        result["raw"]["workflow"] = parsed_workflow

    return result


def cached_image_info(path: Path, want_raw=False):
    stat = path.stat()
    cache_key = (path.name, stat.st_mtime, stat.st_size)
    info = IMAGE_INFO_CACHE.get(cache_key)
    if info is None:
        stale_keys = [key for key in IMAGE_INFO_CACHE if key[0] == path.name and key != cache_key]
        for key in stale_keys:
            IMAGE_INFO_CACHE.pop(key, None)
        meta = read_image_metadata(path)
        info = parse_comfy_metadata(meta)
        IMAGE_INFO_CACHE[cache_key] = info
    response = dict(info)
    if not want_raw:
        response.pop("raw", None)
    return response
