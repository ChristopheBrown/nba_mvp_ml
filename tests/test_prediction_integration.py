import numpy as np

from config import Settings
from flask_app import create_app
from flask_app.models import ModelHandler


class StubModel:
    def predict(self, arr):
        return np.hstack([np.zeros((arr.shape[0], 1)), np.ones((arr.shape[0], 1))])


def test_predict_endpoint_with_real_handler(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    def fake_load_model(target):
        assert str(artifact) in target
        return StubModel()

    monkeypatch.setattr("flask_app.models.mlflow.pyfunc.load_model", fake_load_model)

    handler = ModelHandler(runtime_settings=Settings(MVP_MODEL_ARTIFACT_PATH=str(artifact)))
    app = create_app(model_handler=handler)
    app.testing = True
    client = app.test_client()

    payload = {"features": [0.0] * 24}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert data["predictions"] == [[0.0, 1.0]]
