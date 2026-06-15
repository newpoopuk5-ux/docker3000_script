from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
WORKER_ROOT = PACKAGE_DIR.parent
RUNTIME_DIR = PACKAGE_DIR / "runtime"
CATALOG_DIR = PACKAGE_DIR / "catalog"
CATALOG_SOURCES_DIR = CATALOG_DIR / "sources"
CATALOG_REGISTRY_DIR = CATALOG_DIR / "registry"
LLM_DIR = PACKAGE_DIR / "llm"
WORKER_DIR = PACKAGE_DIR / "worker"


def bootstrap_module_paths() -> None:
    for path in (RUNTIME_DIR, CATALOG_DIR, LLM_DIR, WORKER_DIR):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def resolve_packaged_path(raw: str | None, default: Path, *fallback_dirs: Path) -> Path:
    if raw:
        path = Path(raw)
        if path.is_absolute():
            return path
        for base in (WORKER_ROOT, *fallback_dirs):
            candidate = base / path
            if candidate.exists():
                return candidate
        if fallback_dirs:
            return fallback_dirs[0] / path.name
        return WORKER_ROOT / path
    return default

