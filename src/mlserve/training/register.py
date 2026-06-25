"""Stage 2c entrypoint: promote the latest version of each registered model to
the Production stage in the MLflow Model Registry.

    uv run mlserve-register

This gives a versioned audit trail: every Production model maps to an MLflow run
with full params, metrics, and artifacts. Serving still loads the baked booster
artifacts directly, so this step is about provenance, not the request path.
"""

from __future__ import annotations

from mlflow.tracking import MlflowClient

from mlserve.common.logging import get_logger
from mlserve.config.schema import load_config
from mlserve.training.mlflow_utils import configure_tracking

log = get_logger(__name__)


def main() -> None:
    configure_tracking()
    client = MlflowClient()

    for ds in ["avazu", "nyc_taxi"]:
        name = load_config(ds).registered_model_name
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            log.warning("No registered versions for %s — train first.", name)
            continue
        latest = max(versions, key=lambda v: int(v.version))
        try:
            client.transition_model_version_stage(
                name=name, version=latest.version,
                stage="Production", archive_existing_versions=True,
            )
            log.info("%s v%s -> Production", name, latest.version)
        except Exception as exc:  # registry may be unavailable on some backends
            log.warning("Could not promote %s: %s", name, exc)


if __name__ == "__main__":
    main()
