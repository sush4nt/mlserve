# mlserve — common workflows. Run `make help` for the list.
# All Python commands go through uv so the environment is always reproducible.

ROWS    ?= 3000000          # rows for the demo pipeline (synthetic)
SOURCE  ?= kaggle       	# synthetic | kaggle
DATASET ?= all             	# all | avazu | nyc_taxi

# Where train/register log runs to. Overridden by stack-up so the pipeline writes
# straight into the Dockerized mlflow service instead of the local file store.
MLFLOW_TRACKING_URI ?= file:$(CURDIR)/mlruns

.DEFAULT_GOAL := help
.PHONY: help install pipeline prepare train export register frontend-config \
        serve test lint frontend-build stack-up stack-down clean download-data

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install dependencies (uv)
	uv sync --extra dev

pipeline: prepare train export register frontend-config  ## Full local pipeline end-to-end

prepare:  ## Generate/prepare processed data (SOURCE/ROWS/DATASET overridable)
	uv run mlserve-prepare --dataset $(DATASET) --source $(SOURCE) --rows $(ROWS)

train:  ## Train models and log to MLflow (MLFLOW_TRACKING_URI overridable)
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run mlserve-train --dataset $(DATASET)

export:  ## Export the Avazu model to ONNX (validated)
	uv run mlserve-export --dataset avazu

register:  ## Promote latest model versions to Production in MLflow
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run mlserve-register

frontend-config:  ## Regenerate the frontend model config from artifacts
	uv run mlserve-frontend-config

serve:  ## Run the inference server (API + UI + /metrics) on :8080
	uv run mlserve-serve

test:  ## Run the test suite
	uv run --extra dev pytest -q

lint:  ## Lint with ruff
	uv run --extra dev ruff check src tests

frontend-build:  ## Build the React frontend into frontend/dist
	cd frontend && npm install && npm run build

stack-up:  ## Start mlflow, train against it, then build + start the rest (app, prometheus, grafana)
	docker compose up --build -d --wait mlflow
	$(MAKE) pipeline MLFLOW_TRACKING_URI=http://localhost:5001
	docker compose up --build -d

stack-down:  ## Stop the Docker stack
	docker compose down

mlflow-ui:  ## Open a local MLflow UI against the file store (no Docker)
	uv run mlflow ui --backend-store-uri ./mlruns --port 5000

download-data:  ## Download the real Kaggle datasets (needs `uv sync --extra kaggle` + ~/.kaggle/kaggle.json)
	mkdir -p data/raw/avazu data/raw/nyc_taxi
	uv run --extra kaggle kaggle competitions download -c avazu-ctr-prediction -p data/raw/avazu/
	cd data/raw/avazu && unzip -o avazu-ctr-prediction.zip
	uv run --extra kaggle kaggle competitions download -c new-york-city-taxi-fare-prediction -p data/raw/nyc_taxi/
	cd data/raw/nyc_taxi && unzip -o new-york-city-taxi-fare-prediction.zip

clean:  ## Remove generated data, artifacts, mlruns, and frontend build
	rm -rf data/processed/* artifacts/models/* mlruns frontend/dist \
	       frontend/src/config/models.generated.json
