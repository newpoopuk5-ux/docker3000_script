"""Match on-disk models vs civitai_model_roots + suggest renames."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from config import COMFY_ROOT
from model_manager import SCAN_DIRS

ROOTS_JSON = Path(__file__).resolve().parent / "civitai_model_roots.json"
MIN_BYTES = 1024 * 1024
EXTS = {".safetensors", ".ckpt", ".pt", ".bin", ".pth", ".gguf"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def scan_disk() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for folder, root in SCAN_DIRS.items():
        names: list[str] = []
        if root.is_dir():
            for p in root.iterdir():
                if p.is_file() and p.suffix.lower() in EXTS and p.stat().st_size >= MIN_BYTES:
                    names.append(p.name)
        flux = COMFY_ROOT / "models" / "loras" / "flux"
        if folder == "loras" and flux.is_dir():
            for p in flux.iterdir():
                if p.is_file() and p.suffix.lower() in EXTS and p.stat().st_size >= MIN_BYTES:
                    names.append(p.name)
        out[folder] = sorted(set(names), key=str.lower)
    return out


def folder_for_group(group: str) -> str:
    return {
        "checkpoints": "checkpoints",
        "loras": "loras",
        "vae": "vae",
        "text_encoders": "text_encoders",
        "upscalers": "upscale_models",
        "flux_diffusion_models": "diffusion_models",
        "flux_gguf_unet": "unet",
    }.get(group, "checkpoints")


def load_targets() -> list[dict]:
    data = json.loads(ROOTS_JSON.read_text(encoding="utf-8"))
    targets: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for e in data.get("entries") or []:
        key = (e.get("group") or "", e.get("catalog_name") or "")
        if key in seen:
            continue
        seen.add(key)
        targets.append({
            "catalog_name": e.get("catalog_name") or "",
            "group": e.get("group") or "",
            "folder": folder_for_group(e.get("group") or ""),
            "model_name": e.get("model_name") or "",
            "version_name": e.get("version_name") or "",
            "main_url": e.get("main_url") or "",
            "version_id": e.get("version_id"),
        })
    return targets


def best_fuzzy(name: str, candidates: list[str]) -> tuple[str, float] | None:
    n = norm(name)
    best = ("", 0.0)
    for c in candidates:
        cn = norm(c)
        if n == cn:
            return c, 1.0
        if n in cn or cn in n:
            score = 0.85 + 0.1 * SequenceMatcher(None, n, cn).ratio()
        else:
            score = SequenceMatcher(None, n, cn).ratio()
        if score > best[1]:
            best = (c, score)
    if best[1] >= 0.72:
        return best
    return None


def main() -> None:
    if not ROOTS_JSON.is_file():
        raise SystemExit(f"missing {ROOTS_JSON.name} — run build_civitai_catalog.py first")
    disk = scan_disk()
    targets = load_targets()
    all_disk = [(f, n) for f, names in disk.items() for n in names]

    exact: list[dict] = []
    rename: list[dict] = []
    missing: list[dict] = []
    orphan: list[dict] = []
    matched_disk: set[tuple[str, str]] = set()

    for t in targets:
        folder = t["folder"]
        want = t["catalog_name"]
        pool = disk.get(folder, [])
        if want in pool:
            exact.append({**t, "status": "exact", "on_disk": want})
            matched_disk.add((folder, want))
            continue
        hit = best_fuzzy(want, pool)
        if hit:
            on_disk, score = hit
            if on_disk != want:
                rename.append({
                    **t,
                    "status": "rename",
                    "on_disk": on_disk,
                    "rename_to": want,
                    "score": round(score, 3),
                })
                matched_disk.add((folder, on_disk))
            else:
                exact.append({**t, "status": "exact", "on_disk": want})
                matched_disk.add((folder, want))
        else:
            missing.append({**t, "status": "missing"})

    for folder, name in all_disk:
        if (folder, name) not in matched_disk:
            orphan.append({"folder": folder, "on_disk": name, "status": "orphan"})

    print(f"COMFY_ROOT: {COMFY_ROOT}")
    print(f"catalog targets (unique): {len(targets)}")
    print(f"exact match: {len(exact)}")
    print(f"can rename: {len(rename)}")
    print(f"missing: {len(missing)}")
    print(f"on disk, no catalog: {len(orphan)}")
    print()

    if exact:
        print("=== HAVE (name matches catalog) ===")
        for r in sorted(exact, key=lambda x: (x["folder"], x["catalog_name"].lower())):
            print(f"  [{r['folder']}] {r['catalog_name']}")
            print(f"    Civitai: {r['model_name']} ({r['version_name']})")
        print()

    if rename:
        print("=== RENAME SUGGESTIONS (on disk -> catalog name) ===")
        for r in sorted(rename, key=lambda x: -x["score"]):
            print(f"  [{r['folder']}] {r['on_disk']}")
            print(f"    -> {r['rename_to']}  (score {r['score']})")
            print(f"    Civitai: {r['model_name']} ({r['version_name']})")
            print(f"    ren \"C:\\ComfyUI_windows_portable\\ComfyUI\\models\\{r['folder']}\\{r['on_disk']}\" \"{r['rename_to']}\"")
        print()

    if missing:
        print("=== MISSING (in catalog, not on disk) ===")
        for r in sorted(missing, key=lambda x: (x["folder"], x["catalog_name"].lower())):
            print(f"  [{r['folder']}] {r['catalog_name']}  — {r['model_name']} ({r['version_name']})")
        print()

    if orphan:
        print("=== ON DISK, NOT IN CIVITAI CATALOG ===")
        for r in sorted(orphan, key=lambda x: (x["folder"], x["on_disk"].lower())):
            print(f"  [{r['folder']}] {r['on_disk']}")


if __name__ == "__main__":
    main()
