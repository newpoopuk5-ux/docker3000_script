import requests
from urllib.parse import urlencode

from config import COMFY_URL


def health_status():
    comfy_ok = False
    detail = "unknown"
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        comfy_ok = r.ok
        detail = "ok" if comfy_ok else f"status {r.status_code}"
    except Exception as e:
        detail = str(e)
    return {"comfy_ok": comfy_ok, "comfy_url": COMFY_URL, "detail": detail}


def extract_prompt_id(item):
    if isinstance(item, (list, tuple)) and len(item) > 1:
        return item[1]
    if isinstance(item, dict):
        return item.get("prompt_id") or item.get("id")
    return None


def extract_prompt_ids(items):
    prompt_ids = []
    for item in items:
        prompt_id = extract_prompt_id(item)
        if prompt_id:
            prompt_ids.append(prompt_id)
    return prompt_ids


def queue_status_payload():
    r = requests.get(f"{COMFY_URL}/queue", timeout=10)
    r.raise_for_status()
    queue = r.json()
    running = queue.get("queue_running") or queue.get("running") or []
    pending = queue.get("queue_pending") or queue.get("pending") or []
    current_prompt_id = extract_prompt_id(running[0]) if running else ""
    return {
        "ok": True,
        "pending_count": len(pending),
        "running_count": len(running),
        "current_prompt_id": current_prompt_id or "",
        "pending_prompt_ids": extract_prompt_ids(pending),
        "running_prompt_ids": extract_prompt_ids(running),
        "raw": queue,
    }


def interrupt_comfy():
    r = requests.post(f"{COMFY_URL}/interrupt", timeout=10)
    return r.ok, r.text[:1000]


def queue_prompt(workflow):
    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": workflow, "client_id": "custom-ui"}, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as exc:
        detail = (r.text or "").strip()
        if len(detail) > 2000:
            detail = detail[:2000] + "..."
        raise RuntimeError(f"{exc}; Comfy response: {detail}") from exc
    return r.json()


def get_object_info():
    r = requests.get(f"{COMFY_URL}/object_info", timeout=20)
    r.raise_for_status()
    return r.json()


def upload_image(file_obj, filename):
    files = {"image": (filename, file_obj)}
    data = {"overwrite": "true", "type": "input"}
    r = requests.post(f"{COMFY_URL}/upload/image", files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def comfy_image_url(filename, subfolder="", image_type="input"):
    query = {"filename": filename, "type": image_type}
    if subfolder:
        query["subfolder"] = subfolder
    return f"{COMFY_URL}/view?{urlencode(query)}"


def get_history(prompt_id):
    r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
    r.raise_for_status()
    return r.json()
