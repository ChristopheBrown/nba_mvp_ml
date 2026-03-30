from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from flask_app import create_app
from flask_app.models import ModelHandler
from flask_app.routes import configure_routes
from src.features import CandidatePoolService
from src.features.schema import load_feature_schema


class DummyHandler(ModelHandler):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self._model_version = "dummy-v1"

    def load_model(self):
        self.model = self
        return self

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.1, 0.9]]), (input_array.shape[0], 1))


@pytest.fixture(autouse=True)
def patch_candidate_rows(monkeypatch: Any) -> None:
    schema = load_feature_schema()

    def fake_build_rows(*args: Any, **kwargs: Any):
        base_row = {feature: float(idx) for idx, feature in enumerate(schema.vector_order, start=1)}
        names = ["Player A", "Player B"]
        ids = ["A", "B"]
        metadata = [{"minutes": 40}, {"minutes": 35}]
        return [base_row, base_row], names, ids, metadata

    monkeypatch.setattr("src.features.pipeline.build_candidate_feature_rows", fake_build_rows)


def test_predict_endpoint_exposes_contract(monkeypatch: Any) -> None:
    handler = DummyHandler()
    app = create_app(model_handler=handler)
    client = app.test_client()
    response = client.post("/predict", json={"features": [0.0] * 24})

    assert response.status_code == 200
    payload = response.json
    assert payload["model_version"] == "dummy-v1"
    assert payload["count"] == 1
    assert isinstance(payload["predictions"], list)


def test_candidate_pool_contains_pagination_info(monkeypatch: Any) -> None:
    handler = DummyHandler()
    app = create_app(model_handler=handler)
    client = app.test_client()
    response = client.get("/candidate_pool?top_n=1&pool_size=2")

    assert response.status_code == 200
    payload = response.json
    assert payload["model_version"] == "dummy-v1"
    assert payload["feature_schema_version"].startswith("v")
    assert payload["scaler_version"]
    assert payload["candidate_pool"]["next_cursor"]

    first_id = payload["results"][0]["player_id"]
    second_payload = client.get(f"/candidate_pool?top_n=1&pool_size=2&cursor={first_id}")
    assert second_payload.status_code == 200
    second_data = second_payload.json
    assert second_data["candidate_pool"]["after_id"] == first_id
