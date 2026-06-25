---
title: MLServe
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# mlserve

A production-grade ML inference platform: XGBoost vs ONNX Runtime comparison on a
synthetic Avazu CTR model, plus a NYC Taxi fare regressor, served over the
KServe V2 inference protocol with a FastAPI backend and a React frontend.

This Space runs the single-container build. It trains a small synthetic model at
build time, so it needs no dataset download and no external services. Prometheus
metrics are exposed at `/metrics`; the full Grafana/Prometheus/MLflow stack runs
locally via Docker Compose (see the main repository README).

- `GET /` — React UI
- `POST /v2/models/{name}/infer` — V2 inference
- `GET /metrics` — Prometheus metrics
- `GET /docs` — OpenAPI docs
