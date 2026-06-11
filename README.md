# docker3000_script

Clean Vast/RunPod worker startup bundle for Muse. This repo includes the Flask bridge on port **3000**, plus scripts to install/start ComfyUI on port **8188**.

This repo intentionally contains only worker-side files:

- `auto-start.sh` - one command setup/start for a new Vast machine
- `docker-run.sh` - run the Flask bridge container on port 3000
- `Dockerfile` - small Python runtime image with this Flask bridge copied in
- `start-flask.sh` - container entrypoint
- `app.py`, `worker_status.py`, `workflow_utils.py`, `studio_comfy_api.py`, `config.py`, `metadata.py` - Flask bridge
- `workflow*.json`, `presets.json`, `models.json`, `download_models.py`, `bootstrap.sh` - workflow/model bootstrap support
- `.env.example` - optional config
- `.gitignore` / `.dockerignore` - prevents models, outputs, secrets, logs, and app data from being committed

It does not include Muse React, Muse Express, `chatbot.db`, images, models, LoRAs, or ComfyUI output.

## Ports

| Port | Service | Muse setting |
|------|---------|--------------|
| **3000** | Flask bridge | `COMFY_STUDIO_URL=http://VAST_HOST:3000` |
| **8188** | ComfyUI (when `AUTOSTART_MODE=comfy`) | `COMFY_URL=http://VAST_HOST:8188` with `COMFY_BACKEND_MODE=hybrid` |

Worker status (no secrets): `GET http://VAST_HOST:3000/api/worker/status`

## Native Vast startup (no Docker)

Most Vast **Interactive shell server, SSH** templates do **not** include `docker`. Use:

```bash
bash start-vast-native.sh
```

This script:

- creates `/workspace/venv` and installs `requirements.txt`
- runs `bootstrap.sh --install-only` (ComfyUI at `/workspace/ComfyUI`, no model downloads)
- starts Flask on port **3000** in the foreground
- leaves ComfyUI stopped when `AUTOSTART_MODE=none` (start later from Muse **Start Comfy mode**)

Recommended Vast on-start script — see Muse `docs/COMFY_WORKER.md` for the full template (image, ports, env).

`auto-start.sh` + `docker-run.sh` remain for Docker hosts.

## Clean startup (default)

New `.env.example` defaults boot quickly without ComfyUI or model downloads:

```bash
AUTOSTART_MODE=none
BOOTSTRAP=0
MODEL_SET=none
```

```bash
git clone https://github.com/newpoopuk5-ux/docker3000_script.git
cd docker3000_script
cp .env.example .env
bash auto-start.sh
```

This starts the **Flask control plane** on port 3000 only. ComfyUI is **not** started until you switch to comfy mode (see below).

## Comfy startup

For full image generation on the worker:

```bash
export AUTOSTART_MODE=comfy
export BOOTSTRAP=1
export MODEL_SET=all,flux_user   # or basic, all, etc.
bash auto-start.sh
```

Or set those values in `.env` and run `bash auto-start.sh`.

First run with `BOOTSTRAP=1` installs ComfyUI, Python deps, GGUF custom node, and model folders under `$VOLUME_ROOT/ComfyUI`. Use `MODEL_SET=none` with `BOOTSTRAP=1` for **install-only** (folders + ComfyUI, no downloads).

## Muse PC settings

Expose ports **3000** and **8188** from Vast when using hybrid mode:

```env
COMFY_BACKEND_MODE=hybrid
COMFY_STUDIO_URL=http://VAST_HOST:3000
COMFY_URL=http://VAST_HOST:8188
COMFY_OUTPUT_DIR=C:\ComfyUI_windows_portable\ComfyUI\output
```

If you expose only port **3000**:

```env
COMFY_BACKEND_MODE=flask
COMFY_STUDIO_URL=http://VAST_HOST:3000
COMFY_OUTPUT_DIR=C:\ComfyUI_windows_portable\ComfyUI\output
```

## What `auto-start.sh` does

1. Loads `.env` if present.
2. Detects persistent volume root (`/workspace`, `/root/volume`, `/runpod-volume`, `/mnt/data`, `/volume`).
3. Writes worker mode to `$VOLUME_ROOT/.muse-worker/mode`.
4. If `BOOTSTRAP=1`, runs `bootstrap.sh` (respects `MODEL_SET=none` for install-only).
5. If `AUTOSTART_MODE=comfy`, starts ComfyUI on `COMFY_PORT` (default 8188) when not already online.
6. If `AUTOSTART_MODE=none`, skips ComfyUI and starts Flask only.
7. Builds/runs the Flask Docker container on port 3000.

## Useful env

```bash
VOLUME_ROOT=/workspace
AUTOSTART_MODE=none
BOOTSTRAP=0
MODEL_SET=none
UI_PORT=3000
COMFY_PORT=8188
IMAGE=docker3000_script:local
BUILD_IMAGE=1
PULL_IMAGE=0
APP_PASSWORD=
HF_TOKEN=
CIVITAI_TOKEN=
```

## Test commands

On the Vast instance after `auto-start.sh`:

```bash
# Flask bridge health (Comfy may be offline in none mode)
curl -s http://127.0.0.1:3000/api/health | python3 -m json.tool

# Worker status (mode, disk, token flags — no secret values)
curl -s http://127.0.0.1:3000/api/worker/status | python3 -m json.tool

# ComfyUI (only when AUTOSTART_MODE=comfy and Comfy is running)
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool

docker logs -f muse-flask-3000
```

With Basic Auth enabled (`APP_PASSWORD` set):

```bash
curl -s -u "muse:$APP_PASSWORD" http://127.0.0.1:3000/api/worker/status | python3 -m json.tool
```

## Build and push image

```bash
docker login ghcr.io
IMAGE=ghcr.io/newpoopuk5-ux/docker3000_script:latest bash build-push.sh
```

If the image is not pushed yet, `auto-start.sh` still works because `docker-run.sh` builds the image locally.

## Vast quick start (returning machine)

```bash
cd /workspace/docker3000_script || cd ~/docker3000_script
git pull
cp .env.example .env   # first time only; edit AUTOSTART_MODE as needed
bash auto-start.sh
```

For generation after a clean boot, set `AUTOSTART_MODE=comfy` (and `BOOTSTRAP=1` on first install) in `.env`, then re-run `auto-start.sh`.
