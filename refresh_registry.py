import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muse_worker.paths import bootstrap_module_paths

bootstrap_module_paths()

from refresh_registry import main


if __name__ == "__main__":
    main()
