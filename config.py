import os
from pathlib import Path

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

def _looks_like_comfy_root(path: Path) -> bool:
    return (path / "main.py").exists() or (path / "models").exists()

def _discover_comfy_root() -> Path:
    explicit = os.environ.get("COMFY_ROOT")
    if explicit:
        return Path(explicit)

    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "ComfyUI",
        here / "ComfyUI",
        Path("C:/ComfyUI_windows_portable/ComfyUI"),
        Path("/workspace/ComfyUI"),
        Path("/root/volume/ComfyUI"),
        Path("/runpod-volume/ComfyUI"),
        Path("/mnt/data/ComfyUI"),
        Path("/volume/ComfyUI"),
    ]
    for candidate in candidates:
        if _looks_like_comfy_root(candidate):
            return candidate
    return Path("/workspace/ComfyUI")

COMFY_ROOT = _discover_comfy_root()
WORKFLOW_PATH = Path(os.environ.get("WORKFLOW_PATH", "workflow.json"))
WORKFLOW_IMG2IMG_PATH = Path(os.environ.get("WORKFLOW_IMG2IMG_PATH", "workflow_img2img.json"))
WORKFLOW_NODE_MAP_PATH = Path(os.environ.get("WORKFLOW_NODE_MAP_PATH", "workflow_map.json"))
PRESETS_PATH = Path(os.environ.get("PRESETS_PATH", "presets.json"))
UI_TITLE = os.environ.get("UI_TITLE", "Comfy Vast Studio")
APP_PASSWORD = os.environ.get("APP_PASSWORD") or os.environ.get("UI_PASSWORD")

CHECKPOINT_DIR = COMFY_ROOT / "models" / "checkpoints"
LORA_DIR = COMFY_ROOT / "models" / "loras"
UPSCALE_DIR = COMFY_ROOT / "models" / "upscale_models"
DIFFUSION_MODEL_DIR = COMFY_ROOT / "models" / "diffusion_models"
UNET_DIR = COMFY_ROOT / "models" / "unet"
TEXT_ENCODER_DIR = COMFY_ROOT / "models" / "text_encoders"
VAE_DIR = COMFY_ROOT / "models" / "vae"
OUTPUT_DIR = COMFY_ROOT / "output"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
FAVORITES_PATH = Path(os.environ.get("FAVORITES_PATH", "favorites.json"))

DEFAULT_NODE_MAP = {
    "checkpoint": "5",
    "positive": "7",
    "negative": "8",
    "sampler": "9",
    "vae_decode": "10",
    "save": "12",
    "latent": "13",
    "load_image": "14",
    "vae_encode": "15",
    "upscale_model": "27",
    "upscale": "28",
}
