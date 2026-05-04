# docker3000_script

Clean Vast/RunPod worker startup bundle for Muse. This repo includes the Flask bridge code needed for port `3000`, plus scripts to install/start ComfyUI on port `8188`.

This repo intentionally contains only worker-side files:

- `auto-start.sh` - one command setup/start for a new Vast machine
- `docker-run.sh` - run the Flask bridge container on port 3000
- `Dockerfile` - small Python runtime image with this Flask bridge copied in
- `start-flask.sh` - container entrypoint
- `app.py`, `workflow_utils.py`, `studio_comfy_api.py`, `config.py`, `metadata.py` - Flask bridge
- `workflow*.json`, `presets.json`, `models.json`, `download_models.py`, `bootstrap.sh` - workflow/model bootstrap support
- `.env.example` - optional config
- `.gitignore` / `.dockerignore` - prevents models, outputs, secrets, logs, and app data from being committed

It does not include Muse React, Muse Express, `chatbot.db`, images, models, LoRAs, or ComfyUI output.

## Vast quick start

```bash
git clone https://github.com/newpoopuk5-ux/docker3000_script.git
cd docker3000_script
cp .env.example .env
bash auto-start.sh
```

After that, expose these ports from Vast:

- `3000` - Flask bridge for Muse `COMFY_STUDIO_URL`
- `8188` - ComfyUI for Muse `COMFY_URL` when using `COMFY_BACKEND_MODE=hybrid`

On the Muse PC:

```env
COMFY_BACKEND_MODE=hybrid
COMFY_STUDIO_URL=http://VAST_HOST:3000
COMFY_URL=http://VAST_HOST:8188
COMFY_OUTPUT_DIR=C:\ComfyUI_windows_portable\ComfyUI\output
```

If you expose only port `3000`, use:

```env
COMFY_BACKEND_MODE=flask
COMFY_STUDIO_URL=http://VAST_HOST:3000
COMFY_OUTPUT_DIR=C:\ComfyUI_windows_portable\ComfyUI\output
```

## What `auto-start.sh` does

1. Detects persistent volume root (`/workspace`, `/root/volume`, `/runpod-volume`, `/mnt/data`, `/volume`).
2. Runs local `bootstrap.sh` to install ComfyUI, Python deps, custom nodes, and selected model sets.
4. Starts ComfyUI on port `8188` if it is not already online.
5. Builds the small Flask runtime Docker image from this repo.
6. Runs Flask bridge on port `3000`, mounting `$VOLUME_ROOT/ComfyUI` to `/workspace/ComfyUI`.

## Useful env

```bash
VOLUME_ROOT=/workspace
MODEL_SET=all,flux_user
BOOTSTRAP=1
UI_PORT=3000
COMFY_PORT=8188
IMAGE=docker3000_script:local
BUILD_IMAGE=1
PULL_IMAGE=0
APP_PASSWORD=
HF_TOKEN=
```

## Build and push image

```bash
docker login ghcr.io
IMAGE=ghcr.io/newpoopuk5-ux/docker3000_script:latest bash build-push.sh
```

If the image is not pushed yet, `auto-start.sh` still works because `docker-run.sh` builds the image locally.

## GitHub setup from local folder

```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/newpoopuk5-ux/docker3000_script.git
git push -u origin main
```
