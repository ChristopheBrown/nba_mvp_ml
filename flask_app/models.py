from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlflow.pyfunc
import numpy as np

from config import Settings, settings


class ModelHandler:
    """Load and serve the MVP model using either MLflow or a packaged artifact."""

    def __init__(self, runtime_settings: Optional[Settings] = None) -> None:
        self._settings = runtime_settings or settings
        self.model = None

    def load_model(self):
        if self.model is not None:
            return self.model

        target = self._settings.resolved_model_target
        if self._settings.mlflow_model_uri is None:
            artifact_path = Path(target)
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Packaged model artifact not found at {artifact_path}. "
                    "Run the packaging step or set MVP_MLFLOW_MODEL_URI."
                )

        self.model = mlflow.pyfunc.load_model(target)
        return self.model

    def predict(self, input_array: np.ndarray):
        if self.model is None:
            self.load_model()
        return self.model.predict(input_array)
