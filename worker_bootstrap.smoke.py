import os

os.environ.setdefault("VOLUME_ROOT", "/workspace")
os.environ.setdefault("COMFY_ROOT", "/workspace/ComfyUI")
os.environ.setdefault("APP_DIR", os.path.dirname(__file__))

from worker_bootstrap import comfy_installed, python_bin, repo_root

assert repo_root().is_dir(), "repo_root"
assert python_bin(), "python_bin"
assert isinstance(comfy_installed(), bool), "comfy_installed bool"
print("worker_bootstrap.smoke: ok")
