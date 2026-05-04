FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UI_PORT=3000 \
    COMFY_URL=http://127.0.0.1:8188 \
    COMFY_ROOT=/workspace/ComfyUI

RUN apt-get update \
    && apt-get install -y --no-install-recommends aria2 bash curl git unzip wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' /app/*.sh \
    && chmod +x /app/start-flask.sh /app/auto-start.sh /app/docker-run.sh /app/bootstrap.sh /app/build-push.sh \
    && ln -sf /app/start-flask.sh /usr/local/bin/start-flask \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 3000

CMD ["start-flask"]
