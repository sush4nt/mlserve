# mlserve: Production ML Inference Platform

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

## What this project includes

mlserve is a small but complete ML platform: it trains models, tracks and
registers them, serves them behind a standard inference API, and observes
that API in production — all runnable on a laptop with zero paid infrastructure.

| Component | What it does |
|---|---|
| **Training pipeline** | Generates data (synthetic or real Kaggle), engineers features, trains XGBoost models, and logs every run to MLflow. |
| **Model registry (MLflow)** | Versions every trained model and promotes the best one to "Production" — the same pattern used in real MLOps stacks. |
| **Serving layer (FastAPI)** | Implements the industry-standard **KServe V2 Inference Protocol**, so any V2-compatible client can call it. A pluggable `Runner` abstraction lets the same model be served through different execution engines. |
| **Python vs. ONNX comparison** | The Avazu CTR model is served two ways — via the native XGBoost Python API and via a compiled ONNX Runtime engine — on **identical weights**, so latency differences are measurable and attributable purely to the runtime. |
| **Observability (Prometheus + Grafana)** | Every request is instrumented; a provisioned-as-code dashboard shows request rate, latency percentiles, concurrency, and error rate in real time. |
| **Frontend (React)** | A config-driven UI for sending live predictions and reading model metadata — no hardcoded model list, it's generated from training artifacts. |
| **Load testing (k6)** | Scripted traffic that drives concurrency high enough to actually separate the two serving engines' tail latencies. |

Everything above is verified to run end-to-end on synthetic data with no
external accounts or downloads.

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

**Request flow:** `POST /v2/models/{name}/infer` → FastAPI parses the V2 request
→ `ModelRegistry` looks up the model by name and dispatches to its `Runner` →
the runner predicts → the result is wrapped back into a V2 response. A
middleware records `rest_server_*` Prometheus metrics on every call, regardless
of which runner served it.

**Key building blocks:**

- **`ModelRegistry`** — a simple in-memory map from model name to a loaded
  `Runner`, built once at startup from the trained artifacts. This is what
  makes adding a fourth model or a third engine a one-line registration
  instead of a new code path.
- **`Runner` abstraction** — `BaseRunner` defines one method, `predict()`.
  `XGBoostRunner` calls the native XGBoost API (FP64 input); `ONNXRunner` calls
  a compiled ONNX Runtime session (FP32 input). Both wrap the *same trained
  weights*, so switching engines never changes what the model predicts —
  only how fast it predicts it.
- **V2 Inference Protocol** — the request/response schema (`pydantic` models
  in `common/`) matches KServe's Open Inference Protocol V2, the same
  contract used by Seldon, MLServer, and Triton. Any standard V2 client can
  call this API without modification.
- **MLflow** — every training run (params, metrics, artifacts) is logged and
  versioned; the best version per model is promoted to "Production" and is
  what the serving layer loads.

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

## Why three tiers?

Not every environment needs (or can afford) the same amount of infrastructure.
mlserve is built as three progressively heavier layers on top of the *same*
code and artifacts, so you pick the tier that matches what you're trying to
do rather than always paying for the biggest setup:

| Tier | Environment | Adds | Best for |
|---|---|---|---|
| **1. Local (`uv`)** | Bare machine, no Docker | Training, ONNX export, MLflow file-store tracking, the FastAPI serving API | Fast iteration and development |
| **2. Docker Compose** | Local Docker | Live MLflow UI, Prometheus, Grafana dashboard | Full observability and side-by-side latency comparisons |
| **3. Hugging Face Spaces** | Free cloud container | A public URL, self-trained at build time | Sharing a live, working demo |

Each tier is a superset in capability, not a rewrite: the same `Runner`
classes, the same V2 API, and the same trained artifacts are used everywhere.
Only the amount of surrounding infrastructure changes — which also mirrors a
real-world progression from a developer's laptop to a fully observable
staging stack to a lightweight public deployment.

---

## Tier 1 — run locally with `uv` (the priority path)

No Docker. One command per stage. The whole thing finishes in well under a minute
on synthetic data.

```bash
# 0. Install uv if you don't have it: https://docs.astral.sh/uv/
make install                 # uv sync (creates .venv, installs everything)

# 1-4. Full pipeline: data -> train -> ONNX -> register -> frontend config
# The Makefile defaults to SOURCE=kaggle ROWS=3000000; pass SOURCE=synthetic
# for the zero-setup path that requires no Kaggle account.
make pipeline SOURCE=synthetic ROWS=100000

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
```

Response:

```json
{"model_name":"avazu-ctr-xgb-py", ... "outputs":[{"data":[0.54...]}]}
```

**What this `curl` command is doing.** It's a no-UI test of the exact same
`/v2/models/{name}/infer` endpoint the React frontend calls internally — proof
that the API works independently of the UI.

- **`-X POST`** — sends an HTTP POST request. `/infer` runs a model and
  returns a freshly computed result rather than fetching a stored resource,
  and the input is a structured JSON payload too large/complex for a URL
  query string, so POST (with a body) is the right verb — it's also what the
  KServe V2 Inference Protocol spec requires for this endpoint.
- **`-H "Content-Type: application/json"`** — sets a request *header*, i.e.
  metadata about the request rather than the request itself. This one tells
  FastAPI to parse the body as JSON so it can validate it against the
  `InferRequest` pydantic model; without it, FastAPI may reject the body.
- **`-d '...'`** — the request *body* (`-d` = "data"): the actual JSON payload
  being sent. Here it's a V2 inference request with one named input tensor:
  - `shape: [1, 22]` — 1 row, 22 features, matching the Avazu model's
    `n_features` (the server rejects mismatched shapes, see `app.py`).
  - `datatype: "FP64"` — the numeric precision the Python XGBoost runner
    expects.
  - `data` — the flat feature vector itself, in the exact order
    `features/avazu.py`'s `FEATURE_ORDER` defines.

**Calling a different model** means changing three things together: the URL
segment (e.g. `avazu-ctr-xgb-onnx` or `nyc-taxi-fare-py`), the `shape`/`data`
length (22 for Avazu, 18 for `nyc-taxi-fare-py` — see `featureOrder` in
`frontend/src/config/models.generated.json`), and `datatype` (`FP32` for the
ONNX runner, `FP64` for both Python runners).

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

Wraps the same app with the live MLflow UI, Prometheus, and Grafana — the
observability stack needed to actually see the Python-vs-ONNX latency
difference under load.

```bash
# With Kaggle data (default, requires ~/.kaggle/kaggle.json):
make stack-up

# With synthetic data (zero-setup):
make stack-up SOURCE=synthetic ROWS=100000
```

`stack-up` starts the MLflow container, runs the full pipeline against it
(`$(MAKE) pipeline MLFLOW_TRACKING_URI=http://localhost:5001`), then brings up
the rest of the stack. `./artifacts` is volume-mounted into the app container
(read-only) and `./mlruns` is mounted straight into the mlflow container, so
runs from `make pipeline`/`make train` show up in both `make mlflow-ui` and the
Dockerized MLflow UI without any sync step.

| Service | URL | Notes |
|---|---|---|
| App (UI + API + metrics) | <http://localhost:8080> | |
| MLflow UI | <http://localhost:5001> | |
| Prometheus | <http://localhost:9090> | targets page shows the app as UP |
| Grafana | <http://localhost:3001> | admin / admin; the 6-panel dashboard auto-loads |

`make stack-down` to stop.

---

## Tier 3 — deploy to Hugging Face Spaces (free, single container)

The lightest tier: a single container, no external services, meant for
sharing a working demo rather than for benchmarking. The Space image in
`deploy/huggingface/` trains a small synthetic model **at build time**, so it
needs no dataset and no Kaggle account. Prometheus's `/metrics` endpoint stays
exposed for inspection; Grafana/Prometheus-as-a-service/MLflow UI are
intentionally left out since HF's free tier can't run several always-on
containers.

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

## Frontend: how the UI is built and served

This section is written for a reader with a Python/data-science background and
**no prior JavaScript experience**. It explains what the frontend actually is,
which libraries it uses and why, how the code is organized, and how it gets
from source files to something your browser renders — the JS equivalent of
"here's your `venv`, here's `pip install`, here's how `uvicorn` serves it."

### What it is, in one sentence

A small **React single-page app** that renders a form, calls the exact same
`/v2/models/{name}/infer` HTTP endpoint you could `curl`, and displays the
JSON response — it is a UI client of the API, not a separate backend.

### The JavaScript tooling, mapped to Python equivalents

If you've never touched the JS ecosystem, these four names look interchangeable
but aren't — each solves a different problem, the same way `pip`, `venv`,
`pytest`, and a `.py` file all coexist in a Python project:

| JS concept | What it is | Closest Python analogy |
|---|---|---|
| **Node.js** | A JS runtime that can run outside the browser (e.g. on your laptop, in CI). | The Python interpreter. |
| **npm** | Node's package manager; reads `package.json`/`package-lock.json` and installs into `node_modules/`. | `pip` + `requirements.txt`/`uv.lock`. |
| **Vite** | A *build tool*: compiles JSX into plain JS, bundles all files into a few optimized `.js`/`.css` files, and (in dev) hot-reloads the browser on save. | Roughly `uv build` + a dev autoreloader in one; there's no build step in Python because the interpreter reads `.py` directly, but browsers can't read JSX, so JS projects need this compile step. | 
| **React** | A UI library: describe the page as components (functions that return HTML-like markup called JSX) and React re-renders only what changed when state updates. | The templating layer — think Jinja2, but re-run automatically whenever data changes instead of once per request. |

**Why npm/Node is involved at all**, since this is an ML project: browsers only
understand HTML/CSS/plain JS, but the source is written in JSX (HTML mixed into
JS) using modern syntax browsers can't parse directly. Node + npm + Vite exist
purely to *translate and bundle* that source into files a browser can load —
this happens once at build time. **Node is not part of the running service**:
after `npm run build` produces static files in `frontend/dist/`, FastAPI serves
those files directly (`StaticFiles` mount in `app.py`) and no JS process runs
in production, just like compiling a `.pyx` Cython file once and shipping the
compiled artifact.

### Libraries used and what each one is for

| Package | Role |
|---|---|
| `react`, `react-dom` | Component model + rendering the component tree into the browser DOM. |
| `axios` | HTTP client for calling the V2 API — the JS equivalent of Python's `requests`. |
| `recharts` | Chart library used only by the comparison panel to plot Python-vs-ONNX latency bars. |
| `vite`, `@vitejs/plugin-react` | Dev server + production bundler (see table above); the plugin teaches Vite to compile JSX. |
| `tailwindcss`, `postcss`, `autoprefixer` | Utility-class CSS (e.g. `className="rounded-xl bg-slate-800 p-6"` instead of writing separate `.css` rules) plus the tooling that generates the final stylesheet. |

### Code structure

```
frontend/
|-- package.json              # dependency list + npm scripts (dev/build/preview) — like pyproject.toml
|-- vite.config.js            # build config; also proxies /v2 to :8080 during `npm run dev`
|-- tailwind.config.js        # which files Tailwind scans for class names
|-- index.html                # the one real HTML file; React mounts into its <div id="root">
+-- src/
    |-- main.jsx               # entry point: renders <App /> into index.html
    |-- App.jsx                # top-level layout: model picker, header, wires everything together
    |-- index.css              # Tailwind's base/utility imports
    |-- config/
    |   +-- models.generated.json   # NOT hand-written — see below
    |-- services/
    |   +-- api.js              # one function: POST to /v2/models/{id}/infer via axios
    +-- components/
        |-- PredictionForm.jsx  # renders editable fields, assembles the full feature vector, calls the API
        |-- ComparisonPanel.jsx # calls the Python AND ONNX Avazu endpoints and charts latency side by side
        +-- ResultDisplay.jsx   # formats the prediction (probability vs. dollar value)
```

**Nothing here hardcodes a model name or a feature list.** `models.generated.json`
is produced by `uv run mlserve-frontend-config` (see `frontend_config.py`) from
the same `configs/*.yaml` and `feature_meta.json` the training pipeline
produces — so the UI can never drift from what the models actually expect.
Adding a model is a config change, not a React change.

### How a click becomes a prediction

1. `App.jsx` loads `models.generated.json` at import time and renders a button
   per model plus the fields for whichever one is selected.
2. `PredictionForm.jsx` shows only the fields marked `editable: true`; typing
   updates local component state (`useState`), not the file on disk.
3. On **Predict**, `assembleVector()` rebuilds the *full* ordered feature array
   — typed values for editable slots, training-median `default` for hidden
   ones — and `services/api.js` POSTs it as a KServe V2 payload via `axios`.
4. FastAPI's `/v2/models/{name}/infer` handles it exactly like a `curl` call
   would; the JSON response (predicted value + client-measured latency) is
   rendered by `ResultDisplay.jsx`.

### Dev vs. production — two different ways this gets served

| Mode | Command | What's running |
|---|---|---|
| **Development** | `cd frontend && npm run dev` | Vite's own dev server on a JS port, hot-reloading on save; it proxies `/v2` calls to `:8080` (see `vite.config.js`) so the browser sees one origin while the FastAPI server runs separately via `uv run mlserve-serve`. |
| **Production** | `make frontend-build` (`npm install && npm run build`) → `make serve` | Vite compiles everything into static files under `frontend/dist/`; FastAPI mounts that directory (`app.py`) and serves it at `/` alongside the API. One process, one port, no Node at runtime. |

`make serve` (Tier 1) and the `Dockerfile` (Tiers 2/3) both use the production
path — the React app you see at <http://localhost:8080> is just static
HTML/CSS/JS files being handed out by FastAPI's `StaticFiles`, indistinguishable
from any other static asset.

---

## Frontend: 7 editable inputs, 22 model features

The Avazu model is always called with all **22 features**. The UI only exposes
**7 editable fields** (hour, day, banner position, device type, connection type,
site frequency, app frequency). This is a deliberate UX choice, not a model
simplification.

How it works end-to-end:

1. `configs/avazu.yaml` lists 7 `form_fields` — these become `editable: true`
   entries in `models.generated.json`.
2. All 22 features appear in `models.generated.json` under `fields`; the 15
   non-listed ones carry `editable: false` and a `default` equal to the
   **training-set median** (written by `features/base.py` into `feature_meta.json`
   during preprocessing).
3. `PredictionForm.jsx`'s `assembleVector()` builds the full 22-element vector:
   editable slots use the user's typed value; hidden slots use the stored median.
4. The V2 request is sent with `shape: [1, 22]` and the server validates
   `arr.shape[1] == runner.n_features` (22) before calling the model.

The 15 hidden features are hashed ad-tech IDs (`site_domain`, `app_domain`,
`device_id`, `C14`…`C21`) whose frequency-encoded integers are only meaningful
relative to the training distribution — there is no sensible "user-entered"
value for them. Pinning them to the training median is the standard serving
strategy for features a real-time caller cannot provide.

**A `hint` field for non-obvious inputs.** The Taxi form's four coordinate
fields (`pickup_longitude/latitude`, `dropoff_longitude/latitude`) are
pre-filled with a valid Manhattan trip by default, but typing an arbitrary
number into a bare "Longitude" box is easy to get wrong (wrong sign, wrong
range, lat/lon swapped). `FeatureField` in `config/schema.py` supports an
optional `hint: str` per field, set in `configs/nyc_taxi.yaml`; it flows
through `frontend_config.py` into `models.generated.json` and is rendered by
`PredictionForm.jsx` as small helper text under the input (e.g. *"Decimal
degrees, negative = West. NYC ranges roughly -74.05 to -73.70"*). Add a `hint`
to any field in any `configs/*.yaml` and regenerate
(`make frontend-config`) to get the same treatment.

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

There are two scripts:

| Script | Purpose |
|---|---|
| `load_testing/avazu_comparison.js` | Hammers `avazu-ctr-xgb-py` and `avazu-ctr-xgb-onnx` head-to-head. Ramps to 80 VUs with a spike stage. Tracks `py_inference_ms` and `onnx_inference_ms` as separate k6 Trend metrics. |
| `load_testing/full_stack.js` | Exercises all three endpoints together (avazu-py, avazu-onnx, nyc-taxi-py) — useful for populating every Grafana panel with traffic. |

```bash
# Head-to-head Python vs ONNX comparison (ramps to 80 VUs):
k6 run load_testing/avazu_comparison.js
k6 run --out json=load_testing/results/avazu.json load_testing/avazu_comparison.js

# All three endpoints (populates all Grafana panels):
k6 run load_testing/full_stack.js

# Against a deployed target (HF Spaces, Render, etc.):
k6 run -e BASE_URL=https://<user>-mlserve.hf.space load_testing/avazu_comparison.js
```

**Why Python ≈ ONNX on synthetic data.** A model trained on the default 100 000
synthetic rows is small (few trees, shallow depth). At that size, a single
tree-prediction takes ~0.01–0.1 ms, which is completely swamped by HTTP
round-trip and FastAPI/JSON overhead (~5–20 ms). Both engines finish inference
before the framework even processes the response. The divergence the project is
built to show — ONNX winning on tail latency at concurrency — requires the
**full Avazu model** (trained on millions of real rows, deeper trees, 300 boost
rounds) hit with the 80-VU spike stage. Train on real Kaggle data first, then
run the comparison and report what you actually measure.

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
make test     # pytest tests/test_pipeline.py
make lint     # ruff check src tests
```

`tests/test_pipeline.py` is the single end-to-end test file. It runs the full
in-memory pipeline on tiny synthetic data — no external services, no disk
fixtures, no Kaggle account needed. What each test covers:

| Test | What it checks |
|---|---|
| `test_preprocess_feature_count[avazu-22]` | `AvazuPreprocessor` produces exactly 22 features, in the canonical `FEATURE_ORDER` from `features/avazu.py`; the val split has no NaN (no encoder leakage). |
| `test_preprocess_feature_count[nyc_taxi-18]` | Same guarantee for the Taxi preprocessor and its 18 haversine + temporal features. |
| `test_avazu_has_learnable_signal` | Trains a mini XGBoost on 20 000 synthetic rows and asserts `val_auc > 0.6` — guards the synthetic generator against accidentally producing random noise. |
| `test_v2_protocol_roundtrip` | Round-trips an `InferRequest` through `request_to_array` and `array_to_response`; checks shape, dtype, and output values are preserved correctly by the V2 pydantic models. |

Run a single test by name:
```bash
uv run --extra dev pytest -q -k test_v2_protocol_roundtrip
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
