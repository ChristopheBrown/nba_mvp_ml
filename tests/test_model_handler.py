import numpy as np
import pytest

from config import Settings
from flask_app.models import ModelHandler


class DummyModel:
    def predict(self, arr):
        return np.ones((arr.shape[0], 2))


def test_model_handler_errors_when_artifact_missing(tmp_path):
    missing = tmp_path / "missing"
    settings = Settings(MVP_MODEL_ARTIFACT_PATH=str(missing), MVP_MLFLOW_MODEL_URI=None)
    handler = ModelHandler(runtime_settings=settings)

    with pytest.raises(FileNotFoundError):
        handler.load_model()


def test_model_handler_prefers_mlflow_uri(monkeypatch):
    loaded = {}

    def fake_load_model(target):
        loaded["target"] = target
        return DummyModel()

    monkeypatch.setattr("flask_app.models.mlflow.pyfunc.load_model", fake_load_model)

    settings = Settings(MVP_MLFLOW_MODEL_URI="runs:/12345/abc")
    handler = ModelHandler(runtime_settings=settings)

    handler.load_model()

    assert loaded["target"] == "runs:/12345/abc"
    output = handler.predict(np.zeros((1, 24), dtype=np.float32))
    assert output.shape == (1, 2)
