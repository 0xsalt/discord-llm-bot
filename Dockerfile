# llmcord Dockerfile (Docker-only execution per architecture)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# App files (copy code and assets; config.yaml will be mounted at runtime)
COPY . /app

# Runtime env (secrets provided by docker-compose)
# - DISCORD_BOT_TOKEN
# - OPENAI_API_KEY
ENV CONFIG_PATH=/app/config.yaml \
    LOG_DIR=/app/logs

# Default command (compose will mount /app/config.yaml)
# Run the top-level script directly (project layout is a single-file app inside this directory).
CMD ["python", "/app/discord-llm-bot.py"]