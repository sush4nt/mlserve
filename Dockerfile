# ── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
# Uses the committed src/config/models.generated.json (regenerate with
# `make frontend-config`). Produces /fe/dist.
RUN npm run build

# ── Stage 2: Python inference server ─────────────────────────────────────────
FROM python:3.11-slim AS app
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MLSERVE_ROOT=/app \
    MLSERVE_FRONTEND_DIST=/app/frontend/dist
WORKDIR /app

# curl is used by the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install dependencies first (cached unless pyproject/lock change).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY configs ./configs
COPY --from=frontend /fe/dist ./frontend/dist
# Baked model artifacts (may be empty on first build; compose mounts ./artifacts
# to serve host-trained models, and the HF deploy bakes real ones in).
COPY artifacts ./artifacts

EXPOSE 8080
CMD ["uv", "run", "--no-dev", "mlserve-serve"]
