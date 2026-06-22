from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT = PROJECT_ROOT / "mlops" / "artifacts" / "24-nn-1"


class Settings(BaseSettings):
    """Runtime configuration for the demo-serving stack."""

    debug: bool = Field(default=True, alias="MVP_DEBUG")
    mlflow_model_uri: Optional[str] = Field(
        default=None,
        alias="MVP_MLFLOW_MODEL_URI",
        description="If set, load the model from this MLflow URI instead of the packaged artifact.",
    )
    model_artifact_path: Path = Field(
        default=DEFAULT_ARTIFACT,
        alias="MVP_MODEL_ARTIFACT_PATH",
        description="Path to the packaged mlflow.pyfunc artifact for offline serving.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    @property
    def resolved_model_target(self) -> str:
        """Return the URI/path we should hand to mlflow.pyfunc.load_model."""
        if self.mlflow_model_uri:
            return self.mlflow_model_uri
        return str(self.model_artifact_path)
