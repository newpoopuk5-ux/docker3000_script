import argparse
import json
import os
from pathlib import Path

DEFAULT_REPO_ID = "lisabew/comfy-loras"
DEFAULT_LORA_DIR = Path("C:/ComfyUI_windows_portable/ComfyUI/models/loras")
LORA_SUFFIXES = {".safetensors", ".pt", ".bin"}


def fail(message: str) -> None:
    raise SystemExit(message)


def find_loras(source: Path) -> list[Path]:
    if not source.exists():
        fail(f"LoRA source folder not found: {source}")
    return sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in LORA_SUFFIXES
    )


def repo_path_for(filename: str, path_prefix: str) -> str:
    name = filename
    lower = name.lower()
    if lower.endswith("_txt.safetensors") or lower.startswith("qwen_3_"):
        folder = "text_encoders"
    elif "vae" in lower:
        folder = "vae"
    else:
        folder = path_prefix.rstrip("/") or "loras"
    return f"{folder}/{name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload local model files to a private Hugging Face repo.")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help="Hugging Face repo id, e.g. lisabew/comfy-loras")
    parser.add_argument("--source", default=str(DEFAULT_LORA_DIR), help="Local folder with .safetensors files")
    parser.add_argument("--path-prefix", default="loras", help="Repo folder prefix for LoRA-style files")
    parser.add_argument("--public", action="store_true", help="Create repo as public instead of private")
    parser.add_argument("--smart-paths", action="store_true", help="Route VAE/TE files to vae/ and text_encoders/")
    args = parser.parse_args()

    token = (os.environ.get("HF_TOKEN_W") or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        fail("HF_TOKEN_W or HF_TOKEN is required. Set it in your terminal environment before running this script.")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        fail("Missing huggingface_hub. Install dependencies first: python -m pip install huggingface_hub")

    source = Path(args.source).expanduser().resolve()
    loras = find_loras(source)
    if not loras:
        fail(f"No model files found in: {source}")

    api = HfApi(token=token)
    create_repo(repo_id=args.repo, repo_type="model", private=not args.public, exist_ok=True, token=token)

    entries = []
    print(f"Uploading {len(loras)} file(s) to {args.repo}...")
    for path in loras:
        repo_path = (
            repo_path_for(path.name, args.path_prefix)
            if args.smart_paths
            else f"{args.path_prefix.rstrip('/')}/{path.name}"
        )
        print(f"Uploading {path.name} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload {path.name}",
        )
        entries.append({
            "name": path.name,
            "source": "hf",
            "repo_id": args.repo,
            "repo_path": repo_path,
        })

    print("Upload complete.")
    print("models.json entries:")
    print(json.dumps(entries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
