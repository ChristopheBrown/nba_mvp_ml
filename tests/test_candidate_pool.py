from __future__ import annotations

from typing import Any

import math
import numpy as np
import pytest

from flask_app.models import ModelHandler
from src.features import CandidatePoolService
from src.features.schema import load_feature_schema


class DummyHandler(ModelHandler):
    def __init__(self, predictions: np.ndarray) -> None:
        super().__init__()
        self._predictions = predictions
        self.loaded = False

    def load_model(self):
        self.loaded = True

    def predict(self, arr: np.ndarray) -> np.ndarray:
        assert self.loaded
        repeats = arr.shape[0]
        tiled = np.tile(self._predictions, (math.ceil(repeats / len(self._predictions)), 1))
        return tiled[:repeats]


def _mock_build_rows(*args: Any, **kwargs: Any):
    schema = load_feature_schema()
    base_row = {feature: float(idx) for idx, feature in enumerate(schema.vector_order, start=1)}
    row_high = {feature: value + 1.0 for feature, value in base_row.items()}
    names = ["Player High", "Player Low"]
    ids = ["01", "02"]
    metadata = [{"minutes": 100}, {"minutes": 95}]
    return [row_high, base_row], names, ids, metadata


def _mock_build_rows_three(*args: Any, **kwargs: Any):
    schema = load_feature_schema()
    base_row = {feature: float(idx) for idx, feature in enumerate(schema.vector_order, start=1)}
    names = ["Player Red", "Player Blue", "Player Green"]
    ids = ["01", "02", "03"]
    metadata = [{"minutes": 30}, {"minutes": 25}, {"minutes": 20}]
    return [base_row, base_row, base_row], names, ids, metadata


def test_candidate_pool_service_ranks_descending(monkeypatch):
    monkeypatch.setattr("src.features.candidate_pool.build_candidate_feature_rows", _mock_build_rows)
    mocks = np.array([[0.1, 0.9], [0.05, 0.95]])
    handler = DummyHandler(predictions=mocks)
    service = CandidatePoolService(handler=handler)

    payload, _ = service.build_candidate_pool(top_n=2, pool_size=2)

    assert payload["candidate_pool"]["pool_size"] == 2
    assert payload["candidate_pool"]["top_n"] == 2
    results = payload["results"]
    assert len(results) == 2
    assert results[0]["player_name"] == "Player Low"
    assert results[1]["player_name"] == "Player High"
    assert results[0]["mvp_rank"] == 1
    assert results[1]["mvp_rank"] == 2


def test_candidate_pool_service_requires_scaler(monkeypatch, tmp_path):
    handler = DummyHandler(predictions=np.array([[0.5, 0.5]]))
    missing = tmp_path / "missing_scaler.json"
    with pytest.raises(FileNotFoundError):
        CandidatePoolService(handler=handler, scaler_path=missing)


def test_candidate_pool_service_reports_scaler_and_cursor(monkeypatch):
    monkeypatch.setattr("src.features.candidate_pool.build_candidate_feature_rows", _mock_build_rows_three)
    schema = load_feature_schema()
    length = schema.vector_length
    handler = DummyHandler(predictions=np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]))
    scaler_params = {"mean": [0.0] * length, "scale": [1.0] * length, "version": "test-v2"}
    service = CandidatePoolService(handler=handler, scaler_params=scaler_params)

    payload, _ = service.build_candidate_pool(pool_size=3, top_n=2)
    assert payload["scaler_version"] == "test-v2"
    assert payload["candidate_pool"]["next_cursor"]

    cursor = payload["results"][0]["player_id"]
    paged, _ = service.build_candidate_pool(pool_size=3, top_n=1, after_id=cursor)
    assert paged["candidate_pool"]["after_id"] == cursor
