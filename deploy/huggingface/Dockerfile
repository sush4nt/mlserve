# Hugging Face Spaces (Docker SDK) image — Tier 3 deployment.
#
# Self-contained and free-tier friendly: it trains a SMALL synthetic model at
# BUILD time so no dataset download and no external services are needed. The
# running container serves the API + UI + /metrics on port 7860 (HF default).
#
# To deploy:
#   1. Create a new Space -> SDK: Docker.
#   2. Copy this file to the Space repo root as `Dockerfile`.
#   3. Copy README-space.md to the Space root as `README.md` (it has the Space card).
#   4. Push pyproject.toml, uv.lock, src/, configs/, frontend/ too.
#
# Grafana/Prometheus/MLflow are intentionally omitted here — they don't fit a
# single free container. The /metrics endpoint is still exposed for scraping.

# ── Stage 1: frontend ────────────────────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: train + serve ───────────────────────────────────────────────────
FROM python:3.11-slim AS app
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    MLSERVE_ROOT=/app \
    MLSERVE_PORT=7860 \
    MLSERVE_FRONTEND_DIST=/app/frontend/dist \
    DEMO_ROWS=60000
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

# Train a small model at build so the image is self-contained.
RUN uv run --no-dev mlserve-prepare --dataset all --source synthetic --rows ${DEMO_ROWS} \
 && uv run --no-dev mlserve-train  --dataset all \
 && uv run --no-dev mlserve-export --dataset avazu \
 && uv run --no-dev mlserve-frontend-config

COPY --from=frontend /fe/dist ./frontend/dist

EXPOSE 7860
CMD ["uv", "run", "--no-dev", "mlserve-serve"]
