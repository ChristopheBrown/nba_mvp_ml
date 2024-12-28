class Config:
    """Flask Configuration."""
    DEBUG = True
    MLFLOW_MODEL_URI = "models:/24-nn-1/1"  # Update with your MLflow model URI
    LOCAL_MODEL_PATH = "model.pkl"