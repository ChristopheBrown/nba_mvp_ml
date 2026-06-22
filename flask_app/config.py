from config import settings


class Config:
    """Expose strongly-typed settings to the Flask app."""

    DEBUG = settings.debug
    MVP_MODEL_TARGET = settings.resolved_model_target
    MVP_USE_MLFLOW = settings.mlflow_model_uri is not None
    MVP_MODEL_ARTIFACT_PATH = str(settings.model_artifact_path)
