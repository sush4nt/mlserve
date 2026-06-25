"""The inference server.

A small FastAPI app that:
  * loads all models from artifacts/models/ at startup,
  * serves the KServe V2 inference protocol (health, ready, metadata, infer),
  * exposes Prometheus metrics at /metrics,
  * serves the built React frontend (if present) as static files.

Path operations that run the model are defined with `def` (not `async def`) so
FastAPI executes them in a threadpool — that is what lets concurrent requests
actually overlap and makes the Python-vs-ONNX load comparison meaningful.

Run it:  uv run mlserve-serve            (or: uvicorn mlserve.serving.app:app)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mlserve.common.logging import get_logger
from mlserve.common.paths import FRONTEND_DIST
from mlserve.common.protocol import InferRequest, array_to_response, request_to_array
from mlserve.serving import metrics
from mlserve.serving.registry import ModelRegistry

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry().load_all()
    yield


app = FastAPI(title="mlserve", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    endpoint, method = request.url.path, request.method
    metrics.IN_PROGRESS.labels(method, endpoint).inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        metrics.DURATION.labels(method, endpoint).observe(time.perf_counter() - start)
        metrics.REQUESTS.labels(method, endpoint, str(status)).inc()
        metrics.IN_PROGRESS.labels(method, endpoint).dec()
    return response


# --- V2 inference protocol ----------------------------------------------------
@app.get("/v2/health/live")
@app.get("/v2/health/ready")
def health() -> dict:
    return {"status": "ready"}


@app.get("/v2/models")
def list_models(request: Request) -> dict:
    return {"models": request.app.state.registry.names()}


@app.get("/v2/models/{name}/ready")
def model_ready(name: str, request: Request):
    if not request.app.state.registry.ready(name):
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"name": name, "ready": True}


@app.get("/v2/models/{name}")
def model_meta(name: str, request: Request) -> dict:
    runner = request.app.state.registry.get(name)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {
        "name": runner.name,
        "platform": runner.runtime,
        "task": runner.task,
        "inputs": [{"name": "input-0", "datatype": runner.datatype,
                    "shape": [-1, runner.n_features]}],
        "outputs": [{"name": "output-0", "datatype": runner.datatype, "shape": [-1, 1]}],
        "feature_order": runner.feature_order,
    }


@app.post("/v2/models/{name}/infer")
def infer(name: str, body: InferRequest, request: Request):
    runner = request.app.state.registry.get(name)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    arr = request_to_array(body)
    if arr.shape[1] != runner.n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {runner.n_features} features, got {arr.shape[1]}",
        )
    preds = runner.predict(arr)
    return array_to_response(name, preds, runner.datatype, body.id)


# --- Prometheus ---------------------------------------------------------------
@app.get("/metrics")
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


# --- Static frontend (served last so API routes win) --------------------------
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    log.info("Serving frontend from %s", FRONTEND_DIST)
else:
    @app.get("/")
    def root() -> dict:
        return {"service": "mlserve", "frontend": "not built", "docs": "/docs"}


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mlserve.serving.app:app",
        host=os.getenv("MLSERVE_HOST", "0.0.0.0"),
        port=int(os.getenv("MLSERVE_PORT", "8080")),
        workers=int(os.getenv("MLSERVE_WORKERS", "1")),
    )


if __name__ == "__main__":
    main()
