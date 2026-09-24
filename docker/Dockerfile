# slrag API image — multi-stage, non-root, corpus baked in for demo use.
# Build:  docker build -f docker/Dockerfile -t slrag:latest .

# ---------- stage 1: build wheels ----------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- stage 2: runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 slrag

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=slrag:slrag src/ ./src/
COPY --chown=slrag:slrag scripts/ ./scripts/
COPY --chown=slrag:slrag config/ ./config/
COPY --chown=slrag:slrag pyproject.toml ./

RUN mkdir -p data/raw data/processed data/chroma data/cache reports \
    && chown -R slrag:slrag /app

USER slrag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "slrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
