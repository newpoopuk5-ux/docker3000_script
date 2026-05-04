import json
import os
import random

from config import CHECKPOINT_DIR, DEFAULT_NODE_MAP, WORKFLOW_IMG2IMG_PATH, WORKFLOW_NODE_MAP_PATH, WORKFLOW_PATH
from metadata import link_node_id


def load_workflow(path=WORKFLOW_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_node_map(workflow):
    found = {}
    clip_nodes = []
    sampler_inputs = {}
    for node_id, node in workflow.items():
        class_type = node.get("class_type", "")
        title = ((node.get("_meta", {}) or {}).get("title") or "").lower()
        if class_type == "CheckpointLoaderSimple":
            found.setdefault("checkpoint", str(node_id))
        elif class_type in ("KSampler", "KSamplerAdvanced"):
            found.setdefault("sampler", str(node_id))
            sampler_inputs = node.get("inputs", {}) or {}
        elif class_type == "VAEDecode":
            found.setdefault("vae_decode", str(node_id))
        elif class_type == "SaveImage":
            found.setdefault("save", str(node_id))
        elif class_type == "EmptyLatentImage":
            found.setdefault("latent", str(node_id))
        elif class_type == "LoadImage":
            found.setdefault("load_image", str(node_id))
        elif class_type == "VAEEncode":
            found.setdefault("vae_encode", str(node_id))
        elif class_type == "UpscaleModelLoader":
            found.setdefault("upscale_model", str(node_id))
        elif class_type == "ImageUpscaleWithModel":
            found.setdefault("upscale", str(node_id))
        elif class_type == "CLIPTextEncode":
            clip_nodes.append((str(node_id), title))

    for node_id, title in clip_nodes:
        if "positive" in title:
            found.setdefault("positive", node_id)
        elif "negative" in title:
            found.setdefault("negative", node_id)
    if "positive" not in found and clip_nodes:
        found["positive"] = link_node_id(sampler_inputs.get("positive")) or clip_nodes[0][0]
    if "negative" not in found and len(clip_nodes) > 1:
        found["negative"] = link_node_id(sampler_inputs.get("negative")) or clip_nodes[1][0]
    return found


def load_node_map(workflow):
    node_map = dict(DEFAULT_NODE_MAP)
    node_map.update(discover_node_map(workflow))

    try:
        if WORKFLOW_NODE_MAP_PATH.exists():
            with open(WORKFLOW_NODE_MAP_PATH, "r", encoding="utf-8") as f:
                file_map = json.load(f)
            if isinstance(file_map, dict):
                node_map.update({str(k): str(v) for k, v in file_map.items() if v is not None})
    except Exception:
        pass

    env_map = os.environ.get("WORKFLOW_NODE_MAP")
    if env_map:
        try:
            parsed = json.loads(env_map)
            if isinstance(parsed, dict):
                node_map.update({str(k): str(v) for k, v in parsed.items() if v is not None})
        except Exception:
            pass
    return node_map


def require_node(workflow, node_map, key):
    node_id = node_map.get(key)
    if not node_id or node_id not in workflow:
        raise KeyError(f"workflow node '{key}' not found; set WORKFLOW_NODE_MAP or {WORKFLOW_NODE_MAP_PATH}")
    return node_id


def normalize_loras(data):
    raw_loras = data.get("loras")
    loras = []
    if isinstance(raw_loras, list):
        for item in raw_loras:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("lora_name") or "").strip()
            if not name:
                continue
            strength_model = item.get("strength_model", item.get("strength", 1.0))
            strength_clip = item.get("strength_clip", item.get("strength", strength_model))
            loras.append({
                "name": name,
                "strength_model": float(strength_model),
                "strength_clip": float(strength_clip),
            })

    if not loras and data.get("lora_name"):
        strength = float(data.get("lora_strength", 1.0))
        loras.append({
            "name": str(data.get("lora_name")).strip(),
            "strength_model": strength,
            "strength_clip": strength,
        })
    return loras


def apply_loras(workflow, node_map, loras):
    if not loras:
        return workflow

    checkpoint_id = require_node(workflow, node_map, "checkpoint")
    positive_id = require_node(workflow, node_map, "positive")
    negative_id = require_node(workflow, node_map, "negative")
    sampler_id = require_node(workflow, node_map, "sampler")

    used = {int(k) for k in workflow.keys() if str(k).isdigit()}
    next_id = max(used) + 1 if used else 100
    model_link = [checkpoint_id, 0]
    clip_link = [checkpoint_id, 1]

    for index, lora in enumerate(loras, start=1):
        lora_id = str(next_id)
        next_id += 1
        workflow[lora_id] = {
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": float(lora["strength_model"]),
                "strength_clip": float(lora["strength_clip"]),
                "model": model_link,
                "clip": clip_link,
            },
            "class_type": "LoraLoader",
            "_meta": {"title": f"Load LoRA {index}"},
        }
        model_link = [lora_id, 0]
        clip_link = [lora_id, 1]

    workflow[positive_id]["inputs"]["clip"] = clip_link
    workflow[negative_id]["inputs"]["clip"] = clip_link
    workflow[sampler_id]["inputs"]["model"] = model_link

    return workflow


def next_node_id(workflow):
    used = {int(k) for k in workflow.keys() if str(k).isdigit()}
    return max(used) + 1 if used else 1


def add_node(workflow, class_type, inputs, title=None):
    node_id = str(next_node_id(workflow))
    workflow[node_id] = {
        "inputs": inputs,
        "class_type": class_type,
        "_meta": {"title": title or class_type},
    }
    return node_id


def random_or_fixed_seed(value):
    return random.randint(1, 999999999999999) if value in [-1, "-1", None, ""] else int(value)


def flux_variant(data):
    raw = str(data.get("flux_variant") or data.get("model_family") or "flux1_schnell").strip().lower()
    aliases = {
        "flux": "flux1_dev",
        "flux1": "flux1_dev",
        "dev": "flux1_dev",
        "schnell": "flux1_schnell",
        "custom": "flux_custom",
    }
    return aliases.get(raw, raw)


def is_flux_payload(data):
    family = str(data.get("model_family") or "sdxl").lower()
    return family.startswith("flux") or str(data.get("model_format") or "").lower() == "gguf"


def flux_default_steps(variant, value):
    if value not in (None, ""):
        return int(value)
    return 4 if variant == "flux1_schnell" else 28


def flux_default_cfg(variant, value):
    if value not in (None, ""):
        return float(value)
    return 1.0 if variant == "flux1_schnell" else 1.0


def flux_default_guidance(variant, value):
    if value not in (None, ""):
        return float(value)
    return 0.0 if variant == "flux1_schnell" else 3.5


def add_flux_loras(workflow, model_link, clip_link, loras, model_only=False):
    for index, lora in enumerate(loras, start=1):
        if model_only:
            node_id = add_node(workflow, "LoraLoaderModelOnly", {
                "model": model_link,
                "lora_name": lora["name"],
                "strength_model": float(lora["strength_model"]),
            }, f"Flux LoRA {index} (Model Only)")
            model_link = [node_id, 0]
            continue

        node_id = add_node(workflow, "LoraLoader", {
            "model": model_link,
            "clip": clip_link,
            "lora_name": lora["name"],
            "strength_model": float(lora["strength_model"]),
            "strength_clip": float(lora["strength_clip"]),
        }, f"Flux LoRA {index}")
        model_link = [node_id, 0]
        clip_link = [node_id, 1]
    return model_link, clip_link


def add_optional_upscale_or_save(workflow, image_link, data):
    upscale_model = str(data.get("upscale_model") or "").strip()
    if upscale_model:
        upscaler_id = add_node(workflow, "UpscaleModelLoader", {
            "model_name": upscale_model,
        }, "Load Upscale Model")
        upscale_id = add_node(workflow, "ImageUpscaleWithModel", {
            "upscale_model": [upscaler_id, 0],
            "image": image_link,
        }, "Upscale Image")
        image_link = [upscale_id, 0]

    add_node(workflow, "SaveImage", {
        "filename_prefix": "ComfyUI",
        "images": image_link,
    }, "Save Image")
    return workflow


def build_flux1_workflow(data, variant, model_format):
    workflow = {}
    is_gguf = model_format == "gguf"
    model_name = str(data.get("flux_model") or data.get("model") or "").strip()
    if not model_name:
        raise KeyError("Flux model is required")

    is_checkpoint_flux = (not is_gguf) and (CHECKPOINT_DIR / model_name).exists()

    if is_checkpoint_flux:
        ckpt_id = add_node(workflow, "CheckpointLoaderSimple", {
            "ckpt_name": model_name,
        }, "Load Flux Checkpoint (all-in-one)")
        model_id = ckpt_id
        clip_id = ckpt_id
        vae_id = ckpt_id
        model_link_initial = [ckpt_id, 0]
        clip_link_initial = [ckpt_id, 1]
        vae_link = [ckpt_id, 2]
    else:
        clip_l = str(data.get("clip_l") or data.get("flux_clip_l") or "clip_l.safetensors").strip()
        t5xxl = str(data.get("t5xxl") or data.get("flux_t5xxl") or "").strip()
        vae_name = str(data.get("vae") or data.get("flux_vae") or "ae.safetensors").strip()
        if not clip_l:
            raise KeyError("Flux clip_l text encoder is required")
        if not t5xxl:
            raise KeyError("Flux t5xxl text encoder is required")
        if not vae_name:
            raise KeyError("Flux VAE is required")

        if is_gguf:
            model_id = add_node(workflow, "UnetLoaderGGUF", {
                "unet_name": model_name,
            }, "Load Flux GGUF Model")
            clip_id = add_node(workflow, "DualCLIPLoaderGGUF", {
                "clip_name1": t5xxl,
                "clip_name2": clip_l,
                "type": "flux",
            }, "Load Flux GGUF Text Encoders")
        else:
            model_id = add_node(workflow, "UNETLoader", {
                "unet_name": model_name,
                "weight_dtype": str(data.get("weight_dtype") or "default"),
            }, "Load Flux Diffusion Model")
            clip_id = add_node(workflow, "DualCLIPLoader", {
                "clip_name1": t5xxl,
                "clip_name2": clip_l,
                "type": "flux",
            }, "Load Flux Text Encoders")

        vae_id = add_node(workflow, "VAELoader", {
            "vae_name": vae_name,
        }, "Load Flux VAE")
        model_link_initial = [model_id, 0]
        clip_link_initial = [clip_id, 0]
        vae_link = [vae_id, 0]

    model_link, clip_link = add_flux_loras(
        workflow,
        model_link_initial,
        clip_link_initial,
        normalize_loras(data),
        model_only=is_gguf,
    )

    model_sampling_id = add_node(workflow, "ModelSamplingFlux", {
        "model": model_link,
        "max_shift": float(data.get("max_shift") or 1.15),
        "base_shift": float(data.get("base_shift") or 0.5),
        "width": int(data.get("width", 1024)),
        "height": int(data.get("height", 1024)),
    }, "Flux Model Sampling")

    guidance = flux_default_guidance(variant, data.get("guidance"))
    positive_id = add_node(workflow, "CLIPTextEncodeFlux", {
        "clip": clip_link,
        "clip_l": data.get("prompt", ""),
        "t5xxl": data.get("prompt", ""),
        "guidance": guidance,
    }, "Flux Positive Prompt")
    negative_id = add_node(workflow, "CLIPTextEncodeFlux", {
        "clip": clip_link,
        "clip_l": data.get("negative", ""),
        "t5xxl": data.get("negative", ""),
        "guidance": guidance,
    }, "Flux Negative Prompt")
    latent_id = add_node(workflow, "EmptyLatentImage", {
        "width": int(data.get("width", 1024)),
        "height": int(data.get("height", 1024)),
        "batch_size": max(1, min(8, int(data.get("batch_size", 1) or 1))),
    }, "Flux Empty Latent")
    sampler_id = add_node(workflow, "KSampler", {
        "seed": random_or_fixed_seed(data.get("seed", -1)),
        "steps": flux_default_steps(variant, data.get("steps")),
        "cfg": flux_default_cfg(variant, data.get("cfg")),
        "sampler_name": data.get("sampler_name") or ("euler" if variant == "flux1_schnell" else "euler"),
        "scheduler": data.get("scheduler") or "simple",
        "denoise": 1,
        "model": [model_sampling_id, 0],
        "positive": [positive_id, 0],
        "negative": [negative_id, 0],
        "latent_image": [latent_id, 0],
    }, "Flux KSampler")
    decode_id = add_node(workflow, "VAEDecode", {
        "samples": [sampler_id, 0],
        "vae": vae_link,
    }, "Flux VAE Decode")
    return add_optional_upscale_or_save(workflow, [decode_id, 0], data)


def build_flux2_workflow(data, model_format):
    workflow = {}
    is_gguf = model_format == "gguf"
    model_name = str(data.get("flux_model") or data.get("model") or "").strip()
    text_encoder = str(data.get("flux2_text_encoder") or data.get("t5xxl") or data.get("flux_t5xxl") or "").strip()
    vae_name = str(data.get("vae") or data.get("flux_vae") or "ae.safetensors").strip()
    if not model_name:
        raise KeyError("Flux 2 model is required")
    if not text_encoder:
        raise KeyError("Flux 2 text encoder is required")
    if not vae_name:
        raise KeyError("Flux 2 VAE is required")

    if is_gguf:
        model_id = add_node(workflow, "UnetLoaderGGUF", {
            "unet_name": model_name,
        }, "Load Flux 2 GGUF Model")
        clip_id = add_node(workflow, "CLIPLoaderGGUF", {
            "clip_name": text_encoder,
            "type": "flux2",
        }, "Load Flux 2 GGUF Text Encoder")
    else:
        model_id = add_node(workflow, "UNETLoader", {
            "unet_name": model_name,
            "weight_dtype": str(data.get("weight_dtype") or "default"),
        }, "Load Flux 2 Diffusion Model")
        clip_id = add_node(workflow, "CLIPLoader", {
            "clip_name": text_encoder,
            "type": "flux2",
        }, "Load Flux 2 Text Encoder")

    vae_id = add_node(workflow, "VAELoader", {
        "vae_name": vae_name,
    }, "Load Flux 2 VAE")
    model_link, clip_link = add_flux_loras(
        workflow,
        [model_id, 0],
        [clip_id, 0],
        normalize_loras(data),
        model_only=is_gguf,
    )
    positive_id = add_node(workflow, "CLIPTextEncode", {
        "clip": clip_link,
        "text": data.get("prompt", ""),
    }, "Flux 2 Prompt")
    latent_id = add_node(workflow, "EmptyFlux2LatentImage", {
        "width": int(data.get("width", 1024)),
        "height": int(data.get("height", 1024)),
        "batch_size": max(1, min(8, int(data.get("batch_size", 1) or 1))),
    }, "Flux 2 Empty Latent")
    noise_id = add_node(workflow, "RandomNoise", {
        "noise_seed": random_or_fixed_seed(data.get("seed", -1)),
    }, "Flux 2 Random Noise")
    guider_id = add_node(workflow, "BasicGuider", {
        "model": model_link,
        "conditioning": [positive_id, 0],
    }, "Flux 2 Basic Guider")
    sampler_select_id = add_node(workflow, "KSamplerSelect", {
        "sampler_name": data.get("sampler_name") or "euler",
    }, "Flux 2 Sampler")
    sigmas_id = add_node(workflow, "Flux2Scheduler", {
        "steps": flux_default_steps("flux2", data.get("steps")),
        "width": int(data.get("width", 1024)),
        "height": int(data.get("height", 1024)),
    }, "Flux 2 Scheduler")
    sampler_id = add_node(workflow, "SamplerCustomAdvanced", {
        "noise": [noise_id, 0],
        "guider": [guider_id, 0],
        "sampler": [sampler_select_id, 0],
        "sigmas": [sigmas_id, 0],
        "latent_image": [latent_id, 0],
    }, "Flux 2 Custom Sampler")
    decode_id = add_node(workflow, "VAEDecode", {
        "samples": [sampler_id, 0],
        "vae": [vae_id, 0],
    }, "Flux 2 VAE Decode")
    return add_optional_upscale_or_save(workflow, [decode_id, 0], data)


def build_sdxl_workflow(data):
    source_image = str(data.get("source_image") or "").strip()
    is_img2img = bool(source_image)
    workflow = load_workflow(WORKFLOW_IMG2IMG_PATH if is_img2img else WORKFLOW_PATH)
    node_map = load_node_map(workflow)

    checkpoint_id = require_node(workflow, node_map, "checkpoint")
    positive_id = require_node(workflow, node_map, "positive")
    negative_id = require_node(workflow, node_map, "negative")
    sampler_id = require_node(workflow, node_map, "sampler")
    save_id = require_node(workflow, node_map, "save")

    workflow[checkpoint_id]["inputs"]["ckpt_name"] = data.get("model", workflow[checkpoint_id]["inputs"]["ckpt_name"])
    workflow[positive_id]["inputs"]["text"] = data.get("prompt", workflow[positive_id]["inputs"]["text"])
    workflow[negative_id]["inputs"]["text"] = data.get("negative", workflow[negative_id]["inputs"]["text"])

    workflow[sampler_id]["inputs"]["seed"] = random_or_fixed_seed(data.get("seed", -1))
    workflow[sampler_id]["inputs"]["steps"] = int(data.get("steps", workflow[sampler_id]["inputs"]["steps"]))
    workflow[sampler_id]["inputs"]["cfg"] = float(data.get("cfg", workflow[sampler_id]["inputs"]["cfg"]))
    workflow[sampler_id]["inputs"]["sampler_name"] = data.get("sampler_name") or workflow[sampler_id]["inputs"].get("sampler_name", "euler")
    workflow[sampler_id]["inputs"]["scheduler"] = data.get("scheduler") or workflow[sampler_id]["inputs"].get("scheduler", "karras")
    if is_img2img:
        load_image_id = require_node(workflow, node_map, "load_image")
        workflow[load_image_id]["inputs"]["image"] = source_image
        workflow[sampler_id]["inputs"]["denoise"] = float(data.get("denoise", workflow[sampler_id]["inputs"].get("denoise", 0.6)))
    else:
        latent_id = require_node(workflow, node_map, "latent")
        workflow[latent_id]["inputs"]["width"] = int(data.get("width", workflow[latent_id]["inputs"]["width"]))
        workflow[latent_id]["inputs"]["height"] = int(data.get("height", workflow[latent_id]["inputs"]["height"]))

        batch_size = max(1, min(8, int(data.get("batch_size", 1) or 1)))
        workflow[latent_id]["inputs"]["batch_size"] = batch_size

    upscale_model = data.get("upscale_model", "")
    if upscale_model:
        upscale_model_id = require_node(workflow, node_map, "upscale_model")
        workflow[upscale_model_id]["inputs"]["model_name"] = upscale_model
    else:
        # If no upscaler selected, bypass upscale node by saving decoded image directly.
        vae_decode_id = require_node(workflow, node_map, "vae_decode")
        workflow[save_id]["inputs"]["images"] = [vae_decode_id, 0]

    return apply_loras(workflow, node_map, normalize_loras(data))


def build_workflow(data):
    if is_flux_payload(data):
        variant = flux_variant(data)
        model_format = str(data.get("model_format") or "official").strip().lower()
        if variant == "flux2":
            return build_flux2_workflow(data, model_format)
        return build_flux1_workflow(data, variant, model_format)
    return build_sdxl_workflow(data)
