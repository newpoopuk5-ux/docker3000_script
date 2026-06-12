import base64
import hmac
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))

from studio_comfy_api import comfy_image_url, get_history, get_object_info, health_status, interrupt_comfy, queue_prompt, queue_status_payload, upload_image
from config import APP_PASSWORD, CHECKPOINT_DIR, DIFFUSION_MODEL_DIR, IMAGE_SUFFIXES, LORA_DIR, OUTPUT_DIR, PRESETS_PATH, TEXT_ENCODER_DIR, UI_TITLE, UNET_DIR, UPSCALE_DIR, VAE_DIR
from metadata import cached_image_info, image_records, list_files, read_favorites, safe_output_path, write_favorites
from workflow_utils import build_workflow
from worker_status import build_worker_status
from worker_control import control_supported, set_mode_comfy, set_mode_llm, set_mode_none
from llm_manager import (
    cancel_download as llm_cancel_download,
    delete_model as llm_delete_model,
    list_downloads as llm_list_downloads,
    load_profiles,
    start_profile_download,
    start_url_download,
)
from llm_runner import llm_health, stop_llm, test_llm
from model_manager import (
    cancel_download,
    delete_model,
    list_downloads,
    load_catalog,
    manager_supported,
    preview_url,
    start_download,
    start_url_download,
)

app = Flask(__name__)

JOB_LOOKUP = {}
JOB_LOOKUP_TTL_SECONDS = 60 * 60


def prune_job_lookup(now=None):
    now = now or time.time()
    stale = [
        key for key, value in JOB_LOOKUP.items()
        if now - float(value.get("created", 0)) > JOB_LOOKUP_TTL_SECONDS
    ]
    for key in stale:
        JOB_LOOKUP.pop(key, None)


def image_payload_from_comfy_image(image):
    filename = image.get("filename")
    if not filename:
        return None
    subfolder = image.get("subfolder") or ""
    image_type = image.get("type") or "output"
    saved = safe_output_path(filename) or (OUTPUT_DIR / filename)
    modified = saved.stat().st_mtime if saved.exists() else time.time()
    size = saved.stat().st_size if saved.exists() else None
    image_url = (
        f"/local_output/{filename}"
        if saved.exists()
        else comfy_image_url(filename, subfolder=subfolder, image_type=image_type)
    )
    return {
        "filename": filename,
        "image_url": image_url,
        "download_url": f"/download_output/{filename}" if saved.exists() else image_url,
        "modified": modified,
        "modified_text": datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M:%S"),
        "size": size,
    }


def prompt_history_payload(prompt_id):
    hist = get_history(prompt_id)
    if prompt_id not in hist:
        return {"ok": True, "done": False, "prompt_id": prompt_id}

    prompt_history = hist[prompt_id]
    outputs = prompt_history.get("outputs", {})
    collected = []
    for _node_id, out in outputs.items():
        for image in out.get("images", []):
            payload = image_payload_from_comfy_image(image)
            if payload:
                collected.append(payload)

    if collected:
        first = collected[0]
        return {
            "ok": True,
            "done": True,
            "prompt_id": prompt_id,
            "images": collected,
            "filename": first["filename"],
            "image_url": first["image_url"],
            "download_url": first["download_url"],
            "modified": first["modified"],
            "modified_text": first["modified_text"],
            "size": first["size"],
        }

    status = prompt_history.get("status", {}) or {}
    return {
        "ok": False,
        "done": True,
        "prompt_id": prompt_id,
        "error": status.get("status_str") or "finished without image output",
        "status": status,
    }


def node_status_payload():
    required = [
        "UNETLoader",
        "DualCLIPLoader",
        "CLIPLoader",
        "VAELoader",
        "CLIPTextEncodeFlux",
        "FluxGuidance",
        "ModelSamplingFlux",
        "EmptyFlux2LatentImage",
        "Flux2Scheduler",
        "RandomNoise",
        "BasicGuider",
        "KSamplerSelect",
        "SamplerCustomAdvanced",
        "UnetLoaderGGUF",
        "DualCLIPLoaderGGUF",
        "CLIPLoaderGGUF",
    ]
    try:
        object_info = get_object_info()
        nodes = {name: name in object_info for name in required}
        return {
            "ok": True,
            "comfy_online": True,
            "nodes": nodes,
            "gguf_ready": all(nodes.get(name) for name in ("UnetLoaderGGUF", "DualCLIPLoaderGGUF", "CLIPLoaderGGUF")),
            "flux_ready": all(nodes.get(name) for name in ("UNETLoader", "DualCLIPLoader", "VAELoader", "CLIPTextEncodeFlux", "ModelSamplingFlux")),
            "flux2_ready": all(nodes.get(name) for name in ("UNETLoader", "CLIPLoader", "VAELoader", "EmptyFlux2LatentImage", "Flux2Scheduler", "RandomNoise", "BasicGuider", "KSamplerSelect", "SamplerCustomAdvanced")),
        }
    except Exception as e:
        return {
            "ok": False,
            "comfy_online": False,
            "error": str(e),
            "nodes": {name: False for name in required},
            "gguf_ready": False,
            "flux_ready": False,
            "flux2_ready": False,
        }


def auth_required():
    if not APP_PASSWORD:
        return False
    header = request.headers.get("Authorization", "")
    if not header.startswith("Basic "):
        return True
    try:
        raw = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        _user, password = raw.split(":", 1)
    except Exception:
        return True
    return not hmac.compare_digest(password, APP_PASSWORD)


@app.before_request
def require_basic_auth():
    if auth_required():
        return Response(
            "Authentication required",
            status=401,
            headers={"WWW-Authenticate": 'Basic realm="Comfy Vast Studio"'},
        )


@app.route("/")
def index():
    return render_template("index.html", title=UI_TITLE)


@app.get("/api/health")
def health():
    return jsonify(health_status())


@app.get("/api/worker/status")
def worker_status():
    return jsonify(build_worker_status())


@app.post("/api/mode/comfy")
def mode_comfy():
    return jsonify(set_mode_comfy())


@app.post("/api/mode/none")
def mode_none():
    return jsonify(set_mode_none())


@app.post("/api/mode/llm")
def mode_llm():
    body = request.json or {}
    profile_id = (body.get("profile_id") or body.get("id") or "").strip() or None
    custom_filename = (body.get("custom_filename") or body.get("filename") or "").strip() or None
    ctx_raw = body.get("ctx_size")
    ctx_size = None
    if ctx_raw is not None and str(ctx_raw).strip() != "":
        try:
            ctx_size = int(ctx_raw)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "ctx_size must be an integer"}), 400
    return jsonify(set_mode_llm(profile_id=profile_id, custom_filename=custom_filename, ctx_size=ctx_size))


@app.get("/api/llm/status")
def llm_status():
    return jsonify({"ok": True, **llm_health()})


@app.get("/api/llm/profiles")
def llm_profiles():
    return jsonify(load_profiles())


@app.get("/api/llm/models")
def llm_models_alias():
    return jsonify(load_profiles())


@app.post("/api/llm/download")
def llm_download():
    body = request.json or {}
    url = (body.get("url") or "").strip()
    profile_id = (body.get("profile_id") or body.get("id") or "").strip()
    if url:
        result = start_url_download(url)
    elif profile_id:
        result = start_profile_download(profile_id)
    else:
        return jsonify({"ok": False, "error": "profile_id or url is required"}), 400
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/llm/models/download")
def llm_models_download_alias():
    return llm_download()


@app.get("/api/llm/downloads")
def llm_downloads():
    return jsonify(llm_list_downloads())


@app.post("/api/llm/downloads/<job_id>/cancel")
def llm_download_cancel(job_id):
    result = llm_cancel_download(job_id)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.post("/api/llm/stop")
def llm_stop():
    from worker_control import stop_llm_mode

    return jsonify(stop_llm_mode())


@app.post("/api/llm/test")
def llm_test():
    return jsonify(test_llm())


@app.delete("/api/llm/models/<profile_id>")
def llm_delete(profile_id):
    result = llm_delete_model(profile_id)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.get("/api/models")
def models_overview():
    set_name = (request.args.get("set") or "").strip() or None
    include_flux = str(request.args.get("include_flux") or "").strip().lower() in ("1", "true", "yes")
    payload = load_catalog(set_name, include_flux=include_flux)
    payload["control_supported"] = control_supported()
    payload["manager_supported"] = manager_supported()
    return jsonify(payload)


@app.post("/api/models/preview-url")
def models_preview_url():
    body = request.json or {}
    url = (body.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "url is required"}), 400
    result = preview_url(url)
    status = 200 if result.get("ok") else (501 if result.get("supported") is False else 400)
    return jsonify(result), status


@app.post("/api/models/download")
def models_download():
    body = request.json or {}
    url = (body.get("url") or "").strip()
    if url:
        result = start_url_download(url)
        status = 200 if result.get("ok") else (501 if result.get("supported") is False else 400)
        return jsonify(result), status
    catalog_id = (body.get("id") or body.get("catalog_id") or "").strip()
    if not catalog_id:
        return jsonify({"ok": False, "error": "id or url is required"}), 400
    result = start_download(catalog_id)
    status = 200 if result.get("ok") else (501 if result.get("supported") is False else 400)
    return jsonify(result), status


@app.delete("/api/models/<path:model_id>")
def models_delete(model_id):
    result = delete_model(model_id)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.get("/api/models/downloads")
def models_downloads():
    return jsonify(list_downloads())


@app.post("/api/models/downloads/<job_id>/cancel")
def models_download_cancel(job_id):
    result = cancel_download(job_id)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.get("/api/queue")
def queue_status():
    try:
        return jsonify(queue_status_payload())
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "pending_count": 0,
            "running_count": 0,
            "current_prompt_id": "",
            "raw": {},
        }), 502


@app.get("/api/job/<prompt_id>")
def job_status(prompt_id):
    try:
        return jsonify(prompt_history_payload(prompt_id))
    except Exception as e:
        return jsonify({"ok": False, "done": False, "prompt_id": prompt_id, "error": str(e)}), 502


@app.get("/api/job_lookup/<client_job_id>")
def job_lookup(client_job_id):
    prune_job_lookup()
    job = JOB_LOOKUP.get(client_job_id)
    if not job:
        return jsonify({"ok": False, "found": False}), 404
    return jsonify({"ok": True, "found": True, **job})


@app.post("/api/interrupt")
def interrupt():
    try:
        ok, detail = interrupt_comfy()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 502
    return jsonify({"ok": ok, "detail": detail}), (200 if ok else 502)


@app.get("/api/checkpoints")
def checkpoints():
    return jsonify(list_files(CHECKPOINT_DIR, (".safetensors", ".ckpt")))


@app.get("/api/loras")
def loras():
    return jsonify(list_files(LORA_DIR, (".safetensors", ".pt", ".bin"), recursive=True))


@app.get("/api/upscalers")
def upscalers():
    return jsonify(list_files(UPSCALE_DIR, (".pth", ".safetensors", ".pt")))


@app.get("/api/node_status")
def node_status():
    payload = node_status_payload()
    return jsonify(payload), (200 if payload.get("ok") else 502)


def _variant_from_name(name: str) -> str:
    n = (name or "").lower()
    if "flux2" in n or "flux.2" in n or "flux-2" in n:
        return "flux2"
    if "schnell" in n:
        return "flux1_schnell"
    if "kontext" in n or "dev" in n or "fluxd" in n or "flux1d" in n or "flux.1.d" in n:
        return "flux1_dev"
    if "flux" in n:
        return "flux1_dev"
    return "unknown"


def _read_sidecar_variant(folder, filename) -> str:
    try:
        sidecar = (folder / filename).with_suffix((folder / filename).suffix + ".civitai.json")
        if sidecar.exists():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            v = (data.get("variant") or "").strip()
            if v and v != "unknown":
                return v
    except Exception:
        pass
    return ""


def _build_variants_map(folders, names):
    """Return {filename: variant} using sidecar JSON first, falling back to filename heuristic."""
    out = {}
    for name in names:
        variant = ""
        for folder in folders:
            if (folder / name).exists():
                variant = _read_sidecar_variant(folder, name)
                if variant:
                    break
        if not variant:
            variant = _variant_from_name(name)
        out[name] = variant
    return out


@app.get("/api/model_inventory")
def model_inventory():
    loras = list_files(LORA_DIR, (".safetensors", ".pt", ".bin"), recursive=True)
    all_checkpoints = list_files(CHECKPOINT_DIR, (".safetensors", ".ckpt"))
    flux_checkpoints = [name for name in all_checkpoints if "flux" in name.lower()]
    diffusion_models = list_files(DIFFUSION_MODEL_DIR, (".safetensors", ".ckpt", ".pt", ".bin"))
    gguf_unet = list_files(UNET_DIR, (".gguf",), recursive=True)
    flux_models = diffusion_models + flux_checkpoints
    variants = _build_variants_map([DIFFUSION_MODEL_DIR, CHECKPOINT_DIR], flux_models)
    gguf_variants = _build_variants_map([UNET_DIR], gguf_unet)
    return jsonify({
        "sdxl": {
            "checkpoints": all_checkpoints,
        },
        "flux": {
            "diffusion_models": flux_models,
            "checkpoint_models": flux_checkpoints,
            "gguf_unet": gguf_unet,
            "text_encoders": list_files(TEXT_ENCODER_DIR, (".safetensors", ".ckpt", ".pt", ".bin", ".gguf")),
            "text_encoders_gguf": list_files(TEXT_ENCODER_DIR, (".gguf",)),
            "vae": list_files(VAE_DIR, (".safetensors", ".ckpt", ".pt", ".bin")),
            "loras": loras,
            "variants": variants,
            "gguf_variants": gguf_variants,
        },
        "loras": loras,
        "upscalers": list_files(UPSCALE_DIR, (".pth", ".safetensors", ".pt")),
    })


@app.get("/api/presets")
def presets():
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({})


@app.get("/api/images")
def images():
    return jsonify(image_records())


@app.get("/local_output/<path:filename>")
def local_output(filename):
    path = safe_output_path(filename)
    if not path or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return Response("not found", status=404)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    return Response(path.read_bytes(), mimetype=mime)


@app.get("/download_output/<path:filename>")
def download_output(filename):
    path = safe_output_path(filename)
    if not path or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return Response("not found", status=404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.post("/api/favorite/<path:filename>")
def favorite_image(filename):
    path = safe_output_path(filename)
    if not path or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return jsonify({"ok": False, "error": "not found"}), 404
    favorites = read_favorites()
    desired = (request.json or {}).get("favorite")
    if desired is None:
        desired = path.name not in favorites
    if desired:
        favorites.add(path.name)
    else:
        favorites.discard(path.name)
    write_favorites(favorites)
    return jsonify({"ok": True, "filename": path.name, "favorite": path.name in favorites})


@app.delete("/api/images/<path:filename>")
def delete_image(filename):
    path = safe_output_path(filename)
    if not path or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return jsonify({"ok": False, "error": "not found"}), 404
    path.unlink()
    favorites = read_favorites()
    if path.name in favorites:
        favorites.discard(path.name)
        write_favorites(favorites)
    return jsonify({"ok": True, "filename": path.name})


@app.get("/api/image_info/<path:filename>")
def image_info(filename):
    path = safe_output_path(filename)
    if not path or not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return jsonify({"error": "not found"}), 404
    want_raw = request.args.get("raw") in ("1", "true", "yes")
    return jsonify(cached_image_info(path, want_raw=want_raw))


@app.post("/api/upload_image")
def upload_source_image():
    image = request.files.get("image")
    if not image or not image.filename:
        return jsonify({"ok": False, "error": "missing image"}), 400
    original = secure_filename(image.filename)
    suffix = Path(original).suffix.lower() or ".png"
    if suffix not in IMAGE_SUFFIXES:
        return jsonify({"ok": False, "error": "unsupported image type"}), 400
    stem = Path(original).stem or "source"
    filename = f"source_{int(time.time() * 1000)}_{stem}{suffix}"
    try:
        result = upload_image(image.stream, filename)
    except Exception as e:
        return jsonify({"ok": False, "error": f"ComfyUI upload failed: {e}"}), 502
    uploaded_name = result.get("name") or filename
    subfolder = result.get("subfolder") or ""
    image_type = result.get("type") or "input"
    return jsonify({
        "ok": True,
        "filename": uploaded_name,
        "preview_url": comfy_image_url(uploaded_name, subfolder=subfolder, image_type=image_type),
        "raw": result,
    })


@app.post("/generate")
def generate():
    data = request.json or {}
    client_job_id = str(data.get("client_job_id") or "").strip()
    family = str(data.get("model_family") or "sdxl").lower()
    model_format = str(data.get("model_format") or "official").lower()
    if family.startswith("flux") or model_format == "gguf":
        status = node_status_payload()
        if not status.get("comfy_online"):
            return jsonify({"ok": False, "error": "ComfyUI is offline; cannot validate Flux nodes"}), 502
        if model_format == "gguf" and not status.get("gguf_ready"):
            return jsonify({"ok": False, "error": "GGUF node pack not installed"}), 400
        if family == "flux2" and not status.get("flux2_ready"):
            return jsonify({"ok": False, "error": "Flux 2 nodes are not available in this ComfyUI install"}), 400
        if family != "flux2" and not status.get("flux_ready"):
            return jsonify({"ok": False, "error": "Flux nodes are not available in this ComfyUI install"}), 400
    try:
        workflow = build_workflow(data)
    except KeyError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    try:
        result = queue_prompt(workflow)
    except Exception as e:
        return jsonify({"ok": False, "error": f"ComfyUI queue failed: {e}"}), 502
    prompt_id = result["prompt_id"]
    if client_job_id:
        prune_job_lookup()
        JOB_LOOKUP[client_job_id] = {
            "prompt_id": prompt_id,
            "created": time.time(),
        }
    return jsonify({
        "ok": True,
        "queued": True,
        "prompt_id": prompt_id,
        "client_job_id": client_job_id,
    })


if __name__ == "__main__":
    import threading
    import time

    from worker_startup_banner import print_banner

    port = int(os.environ.get("FLASK_PORT") or os.environ.get("UI_PORT") or "3000")

    def _print_ready_banner() -> None:
        time.sleep(2.0)
        print_banner(probe_services=True, title="Muse worker ready - copy URLs into Muse")

    threading.Thread(target=_print_ready_banner, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False)
