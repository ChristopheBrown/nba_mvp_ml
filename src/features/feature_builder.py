"""Runtime feature builder supporting the NBA MVP v1 schema contract."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.schema import FeatureSchema, SchemaValidationError, load_feature_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureVector:
    """Structured container for a normalized feature vector and associated metadata."""

    player_id: str | None
    player_name: str
    schema_version: str
    vector: np.ndarray
    normalized: bool
    metadata: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "schema_version": self.schema_version,
            "normalized": self.normalized,
            "vector": self.vector.tolist(),
            "metadata": dict(self.metadata),
        }


class RuntimeFeatureBuilder:
    """Builds schema-aligned feature vectors for MVP scoring in production."""

    def __init__(
        self,
        schema: FeatureSchema | None = None,
        scaler_params: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.schema = schema or load_feature_schema()
        validate_result = self.schema.vector_length
        logger.debug("Loaded feature schema %s with %d entries", self.schema.version, validate_result)
        self.scaler_params = scaler_params
        if scaler_params:
            self._validate_scaler_params(scaler_params)

    def _validate_scaler_params(self, scaler_params: Mapping[str, Sequence[float]]) -> None:
        mean = scaler_params.get("mean")
        scale = scaler_params.get("scale")
        if mean is None or scale is None:
            raise SchemaValidationError("Scaler params must include both `mean` and `scale` arrays")

        if len(mean) != self.schema.vector_length or len(scale) != self.schema.vector_length:
            raise SchemaValidationError(
                f"Scaler params length mismatch: schema expects {self.schema.vector_length} features but scaler has {len(mean)}"
            )
        logger.debug("Scaler parameters validated for %d features", self.schema.vector_length)

    def _normalize(self, raw_vector: np.ndarray) -> np.ndarray:
        if self.scaler_params is None:
            logger.warning("No scaler params provided; returning raw features without normalization")
            return raw_vector

        mean = np.asarray(self.scaler_params["mean"], dtype=float)
        scale = np.asarray(self.scaler_params["scale"], dtype=float)
        safe_scale = np.where(scale == 0, 1.0, scale)
        normalized = (raw_vector - mean) / safe_scale
        return normalized

    def build_vector(
        self,
        player_name: str,
        named_features: Mapping[str, float],
        player_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FeatureVector:
        """Build a ready-to-score vector from a named feature mapping."""

        missing = [feature for feature in self.schema.vector_order if feature not in named_features]
        if missing:
            raise SchemaValidationError(f"Feature mapping missing values for {missing}")

        raw = np.array([float(named_features[feature]) for feature in self.schema.vector_order], dtype=float)
        normalized = self._normalize(raw)
        metadata_payload = dict(metadata or {})
        metadata_payload.setdefault("feature_order", self.schema.vector_order)

        return FeatureVector(
            player_id=player_id,
            player_name=player_name,
            schema_version=self.schema.version,
            vector=normalized,
            normalized=self.scaler_params is not None,
            metadata=metadata_payload,
        )

    def build_batch(
        self,
        named_feature_rows: Sequence[Mapping[str, float]],
        player_names: Sequence[str],
        player_ids: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[FeatureVector]:
        """Convert a batch of named-feature dictionaries into schema-aligned vectors."""

        if len(named_feature_rows) != len(player_names):
            raise ValueError("player_names length must match named_feature_rows")

        player_ids = player_ids or [None] * len(player_names)
        metadatas = metadatas or [None] * len(player_names)

        vectors: list[FeatureVector] = []
        for row, name, pid, meta in zip(named_feature_rows, player_names, player_ids, metadatas):
            vectors.append(self.build_vector(name, row, player_id=pid, metadata=meta))
        return vectors

    def load_player_features(self, player_id: str) -> Mapping[str, float]:
        """Hook for loading named feature dictionaries for a player. To be implemented."""

        raise NotImplementedError("Player feature loading is not implemented yet")
