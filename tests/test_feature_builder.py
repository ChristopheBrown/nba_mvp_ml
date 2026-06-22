from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from src.features import RuntimeFeatureBuilder, SchemaValidationError, load_feature_schema


def _make_scalar_params(length: int, mean: float = 0.0, scale: float = 1.0) -> dict[str, Sequence[float]]:
    return {
        "mean": [mean] * length,
        "scale": [scale] * length,
    }


def _make_ordered_features(schema) -> dict[str, float]:
    return {
        feature: float(idx)
        for idx, feature in enumerate(schema.vector_order, start=1)
    }


def test_build_vector_requires_all_features() -> None:
    schema = load_feature_schema()
    builder = RuntimeFeatureBuilder(schema=schema)

    with pytest.raises(SchemaValidationError):
        builder.build_vector("Test Player", {"FGM": 1.0})


def test_build_vector_normalizes_when_scaler_params_are_provided() -> None:
    schema = load_feature_schema()
    scalar_params = _make_scalar_params(schema.vector_length)
    builder = RuntimeFeatureBuilder(schema=schema, scaler_params=scalar_params)
    named_features = _make_ordered_features(schema)

    vector = builder.build_vector("Normalized Player", named_features)

    assert vector.schema_version == schema.version
    assert vector.normalized
    assert np.allclose(vector.vector, np.asarray(list(range(1, schema.vector_length + 1)), dtype=float))


def test_build_batch_preserves_metadata() -> None:
    schema = load_feature_schema()
    builder = RuntimeFeatureBuilder(schema=schema)
    named_features = _make_ordered_features(schema)

    metadata = {"role": "starter"}
    batch = builder.build_batch(
        named_feature_rows=[named_features],
        player_names=["Batch Player"],
        player_ids=["123"],
        metadatas=[metadata],
    )

    assert len(batch) == 1
    feature_vector = batch[0]
    assert feature_vector.player_id == "123"
    assert feature_vector.metadata["role"] == "starter"
    assert feature_vector.player_name == "Batch Player"
    assert feature_vector.normalized is False
    assert np.allclose(feature_vector.vector, np.asarray(list(range(1, schema.vector_length + 1)), dtype=float))
