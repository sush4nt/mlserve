# mlserve — Production ML Inference Platform

A containerised ML inference platform that serves three model endpoints — two for
AdTech CTR classification (**XGBoost-Python** vs **XGBoost-ONNX/C++**) and one for
**NYC taxi-fare regression** — over the **KServe V2 Inference Protocol**, with a
FastAPI serving layer, an MLflow model registry, Prometheus metrics, a Grafana
dashboard, and a React frontend. Everything runs locally with `uv` (no Docker
required), scales up to a full Docker-Compose observability stack, and deploys to
a single free Hugging Face Space.

The headline feature is a **measurable, side-by-side comparison of the Python and
ONNX execution engines serving identical model weights.**

---

## What this is, and how it differs from the original blueprint

This implementation deliberately diverges from a pure MLServer/NGINX design in a
few places. Each change serves the goals of *local-first, simple, readable, and
cheap to deploy* — and none of them weaken the resume story.

| Original plan | Here | Why |
|---|---|---|
| `seldonio/mlserver` container | Custom **FastAPI** server implementing the V2 protocol | `uv`-runnable, readable, self-hosts `/metrics`, and makes the `XGBoostRunner`/`ONNXRunner` abstraction explicit. The V2 contract is preserved, so "implemented the Open Inference Protocol V2" stays fully defensible. |
| NGINX reverse proxy | FastAPI serves the built React app + API on one port | One fewer moving part; one port for both local and HF. |
| One Docker-Compose monolith | **Three tiers**: `uv`-only → Compose → single HF container | Local-first priority; HF free tier can't run 6 always-on services. |
| 6 GB Kaggle datasets, required | **Synthetic-by-default** data with planted signal; Kaggle is an opt-in switch | The whole pipeline runs end-to-end with zero downloads and no Kaggle account. |
| MLflow tracking server required | MLflow defaults to a local **file store** (`./mlruns`) | Training works on a bare checkout; the Compose stack adds the live UI. |

Everything below is verified to run end-to-end on synthetic data.

---

## Architecture

```
                         +---------------------------+
                         |      Browser / Client     |
                         +-------------+-------------+
                                       | HTTP :8080
                         +-------------v-------------+
                         |   FastAPI app (mlserve)   |
                         |  - React UI  (/)          |
                         |  - V2 API    (/v2/models) |
                         |  - Metrics   (/metrics)   |
                         +---+-------------------+---+
                  dispatch   |                   |  scrape /metrics (15s)
              +--------------v--------+   +-------v--------+
              |     ModelRegistry     |   |   Prometheus   |  (Tier 2)
              |  XGBoostRunner (FP64) |   +-------+--------+
              |  ONNXRunner    (FP32) |           | PromQL
              +-----------------------+   +-------v--------+
              artifacts/models/*  <----    |    Grafana     |  (Tier 2)
              baked or volume-mounted      +----------------+

   MLflow (file store ./mlruns, or the mlflow service in Tier 2) tracks every run.
```

**Request flow:** `POST /v2/models/{name}/infer` -> FastAPI parses the V2 request
-> `ModelRegistry` dispatches to the runner -> runner predicts -> V2 response. A
middleware records `rest_server_*` Prometheus metrics on every call.

---

## Repository layout

```
mlserve/
|-- pyproject.toml            # uv project: pinned deps + console scripts
|-- Makefile                  # one-command workflows (make help)
|-- docker-compose.yml        # Tier 2: app + mlflow + prometheus + grafana
|-- Dockerfile                # app image (build React -> run FastAPI)
|
|-- configs/                  # one YAML per model = single source of truth
|   |-- avazu.yaml
|   +-- nyc_taxi.yaml
|
|-- src/mlserve/
|   |-- common/               # paths, logging, V2 protocol (pydantic)
|   |-- config/               # YAML -> dataclasses
|   |-- data/                 # synthetic generators + kaggle loader + prepare CLI
|   |-- features/             # BasePreprocessor -> Avazu / Taxi
|   |-- training/             # train, export_onnx, register, mlflow utils
|   +-- serving/              # runners, registry, metrics, FastAPI app, frontend-config
|
|-- frontend/                 # React + Vite + Tailwind (served by FastAPI)
|-- monitoring/               # prometheus.yml + grafana provisioning + dashboard JSON
|-- load_testing/             # k6 scripts
|-- deploy/huggingface/       # Tier 3: self-contained Space Dockerfile + card
+-- tests/                    # end-to-end pytest
```

---

## Tier 1 — run locally with `uv` (the priority path)

No Docker. One command per stage. The whole thing finishes in well under a minute
on synthetic data.

```bash
# 0. Install uv if you don't have it: https://docs.astral.sh/uv/
make install                 # uv sync (creates .venv, installs everything)

# 1-4. Full pipeline: data -> train -> ONNX -> register -> frontend config
make pipeline                # ROWS=100000 SOURCE=synthetic by default

# 5. Build the UI (optional; the API works without it)
make frontend-build

# 6. Serve API + UI + /metrics on http://localhost:8080
make serve
```

Then open <http://localhost:8080>, or call the API directly:

```bash
curl -s -X POST http://localhost:8080/v2/models/avazu-ctr-xgb-py/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"input-0","shape":[1,22],"datatype":"FP64",
       "data":[[14,2,0,1005,5000,3000,3,3000,200,5,100,150,1,0,10,320,50,1722,0,35,100,79]]}]}'
# -> {"model_name":"avazu-ctr-xgb-py", ... "outputs":[{"data":[0.54...]}]}
```

Want the live MLflow UI without Docker? `make mlflow-ui` (reads `./mlruns`).

### The pipeline stages, individually

| Stage | Command | What it does |
|---|---|---|
| Prepare | `uv run mlserve-prepare` | Generate/clean data -> `data/processed/<ds>/{train,val}.parquet` + `feature_meta.json` |
| Train | `uv run mlserve-train` | Train XGBoost (both tasks), log to MLflow, save booster + `model_meta.json` |
| Export | `uv run mlserve-export` | Convert the Avazu booster to ONNX, validate vs XGBoost (< 1e-3) |
| Register | `uv run mlserve-register` | Promote latest MLflow versions to Production |
| Frontend config | `uv run mlserve-frontend-config` | Generate `models.generated.json` from artifacts (no hand-edited feature lists) |
| Serve | `uv run mlserve-serve` | FastAPI: V2 API + `/metrics` + static UI |

---

## Tier 2 — full local stack with Docker Compose

Adds the live MLflow UI, Prometheus, and Grafana around the app.

```bash
make stack-up          # trains on the host, then docker compose up --build -d
```

`stack-up` runs the full pipeline first so `./artifacts` has models (mounted into
the app container, read-only) and `./mlruns` has runs (mounted straight into the
mlflow container as its backend store). The Dockerized MLflow UI reads the same
`./mlruns` directory as `make mlflow-ui` — there's no separate store to keep in
sync, so runs from `make pipeline`/`make train` show up immediately either way.

| Service | URL | Notes |
|---|---|---|
| App (UI + API + metrics) | <http://localhost:8080> | |
| MLflow UI | <http://localhost:5001> | |
| Prometheus | <http://localhost:9090> | targets page shows the app as UP |
| Grafana | <http://localhost:3001> | admin / admin; the 6-panel dashboard auto-loads |

`make stack-down` to stop.

---

## Tier 3 — deploy to Hugging Face Spaces (free, single container)

The Space image in `deploy/huggingface/` trains a small synthetic model **at build
time**, so it needs no dataset and no external services. Prometheus `/metrics`
stays exposed; Grafana/Prometheus/MLflow are local-only.

1. Create a new Space -> **SDK: Docker**.
2. Copy `deploy/huggingface/Dockerfile` to the Space root as `Dockerfile`.
3. Copy `deploy/huggingface/README-space.md` to the Space root as `README.md`.
4. Push `pyproject.toml`, `uv.lock`, `src/`, `configs/`, `frontend/`.

The app serves on port 7860 (HF default). The same image works on Render, Railway,
or Fly.io — point them at that Dockerfile.

---

## Using the real Kaggle datasets

The synthetic default is schema-identical to the real data, so switching is just a
flag. Real metrics (Avazu AUC > 0.76, Taxi RMSE under $3.50) require the real data.

```bash
uv sync --extra kaggle                     # adds the kaggle CLI
# put your API key at ~/.kaggle/kaggle.json (chmod 600)
make download-data                         # ~11 GB across both competitions

# Re-run prepare against the real CSVs (row-capped for memory):
uv run mlserve-prepare --dataset avazu     --source kaggle --rows 5000000
uv run mlserve-prepare --dataset nyc_taxi  --source kaggle --rows 3000000
make train export register frontend-config
```

The feature-engineering and model code are identical for synthetic and real data.

---

## The Python vs ONNX comparison

Both `avazu-ctr-xgb-py` and `avazu-ctr-xgb-onnx` serve the **same trained booster**
— only the engine differs. The export step asserts the two agree to < 1e-3, so any
latency difference is purely runtime, never accuracy.

`BaseRunner` defines `predict()`; `XGBoostRunner` runs it through the Python
XGBoost API (accepts FP64), `ONNXRunner` runs it through the compiled ONNX Runtime
(expects FP32). Adding a third engine is a new subclass — nothing else changes.

**An honest note on the numbers.** With the small *synthetic* model, a single
inference is dominated by HTTP/framework overhead, so Python ~= ONNX — the engine
isn't the bottleneck at this size. The divergence the project is built to show
(ONNX winning on tail latency under concurrency) appears with the **full Avazu
model under load**, which is what the k6 spike stage exercises. Don't quote
fabricated "3x faster" numbers; run the load test on the real model and report what
you actually measure. That honesty is itself a good interview signal.

---

## Load testing (k6)

k6 is a standalone Go binary (install: <https://k6.io/docs/get-started/installation/>).

```bash
k6 run load_testing/avazu_comparison.js                 # ramps to 80 VUs, py vs onnx
k6 run --out json=load_testing/results/avazu.json load_testing/avazu_comparison.js
# Against a deployed target:
k6 run -e BASE_URL=https://<user>-mlserve.hf.space load_testing/avazu_comparison.js
```

Watch the Grafana latency panel during the spike and screenshot it — that's your
evidence.

---

## Metrics & dashboard

The app exposes these at `/metrics` (names chosen to match the Grafana queries):

| Metric | Type | Labels |
|---|---|---|
| `rest_server_requests_total` | Counter | method, endpoint, status_code |
| `rest_server_request_duration_seconds` | Histogram | method, endpoint |
| `rest_server_requests_in_progress` | Gauge | method, endpoint |

The Grafana dashboard (`monitoring/grafana/dashboards/mlserve-overview.json`) has
6 panels — request rate, p50/p95/p99 latency (Avazu), concurrency, error rate,
total predictions, and taxi p95 — and auto-provisions on `make stack-up`.

---

## Testing & linting

```bash
make test     # pytest: end-to-end pipeline + V2 round-trip + generator signal
make lint     # ruff
```

---

## Notes for interviews (defensible bullets)

- *Built a production ML inference platform serving 3 endpoints over the KServe V2
  Inference Protocol (FastAPI), with a pluggable runner abstraction enabling a
  side-by-side XGBoost-Python vs ONNX-Runtime comparison on identical weights.*
- *Trained XGBoost CTR classifiers (Avazu) and fare regressors (NYC Taxi) with a
  temporal train/val split, tracked in MLflow and promoted to Production via the
  Model Registry; exported to ONNX with numerical-equivalence validation (< 1e-3).*
- *Instrumented the server with Prometheus metrics and a provisioned-as-code
  Grafana dashboard; load-tested with k6 to measure latency percentiles under
  concurrency.*
- *Designed a config-driven frontend: adding a model endpoint is one JSON entry
  generated from training metadata, with zero component changes.*

---

## Troubleshooting

- **`uv` can't fetch Python 3.11** behind a restricted network -> it's fine on a
  normal connection; the project also runs on 3.12 (`requires-python = >=3.11,<3.13`).
- **`No module named pkg_resources`** (Python 3.12) -> already handled: `setuptools<81`
  is pinned (it ships `pkg_resources`, removed in 81+).
- **ONNX shape warnings** -> silenced via `log_severity_level=3` in `ONNXRunner`.
- **Frontend build fails on missing `models.generated.json`** -> run
  `make frontend-config` (or `make pipeline`) first; a copy is committed too.
- **404 on infer** -> the model isn't trained yet; run `make pipeline`.

## License

MIT.
