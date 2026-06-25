"""Model registry: discover every model directory under MODELS_DIR (each holding
a model_meta.json) and load the appropriate runner. Inference is dispatched by
model name, exactly like MLServer scanning its models dir."""

from __future__ import annotations

from pathlib import Path

from mlserve.common.logging import get_logger
from mlserve.common.paths import MODELS_DIR
from mlserve.serving.runners import BaseRunner, make_runner

log = get_logger(__name__)


class ModelRegistry:
    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self._runners: dict[str, BaseRunner] = {}

    def load_all(self) -> "ModelRegistry":
        if not self.models_dir.exists():
            log.warning("Models dir %s does not exist — no models loaded", self.models_dir)
            return self
        for d in sorted(self.models_dir.iterdir()):
            if (d / "model_meta.json").exists():
                try:
                    runner = make_runner(d)
                    self._runners[runner.name] = runner
                    log.info("Loaded model '%s' (%s)", runner.name, runner.runtime)
                except Exception as exc:
                    log.error("Failed to load model in %s: %s", d, exc)
        if not self._runners:
            log.warning("No models loaded. Run training + export first.")
        return self

    def get(self, name: str) -> BaseRunner | None:
        return self._runners.get(name)

    def ready(self, name: str) -> bool:
        return name in self._runners

    def names(self) -> list[str]:
        return list(self._runners.keys())
