from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.feature_builder import FeatureVector, RuntimeFeatureBuilder
from src.features.pipeline import POOL_DEFINITION, build_candidate_feature_rows
from src.features.schema import FeatureSchema, load_feature_schema
from src.monitoring import emit_metric

DEFAULT_SCALER_PATH = Path("json/scaler_params_v1.json")


class CandidatePoolService:
    def __init__(
        self,
        handler: Any,
        schema: FeatureSchema | None = None,
        scaler_path: Path | None = None,
        scaler_params: Mapping[str, Sequence[float]] | None = None,
        pool_definition: str = POOL_DEFINITION,
    ) -> None:
        self.handler = handler
        self.schema = schema or load_feature_schema()
        self.pool_definition = pool_definition
        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        self.scaler_params = scaler_params or self._load_scaler_params()
        self.scaler_version = self.scaler_params.get("version") or self._derive_scaler_version()

    def _derive_scaler_version(self) -> str:
        try:
            timestamp = self.scaler_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp, timezone.utc).replace(microsecond=0).isoformat()
        except OSError:
            return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _load_scaler_params(self) -> Mapping[str, Sequence[float]]:
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler params not found at {self.scaler_path}")
        with self.scaler_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        mean = payload.get("mean")
        scale = payload.get("scale")
        if mean is None or scale is None:
            raise ValueError("Scaler params must contain both `mean` and `scale` arrays")
        params: dict[str, Any] = {"mean": list(mean), "scale": list(scale)}
        if "version" in payload:
            params["version"] = payload.get("version")
        return params

    def _build_vectors(
        self,
        season: int | None,
        pool_size: int,
        historical_mode: bool = False,
    ) -> tuple[list[FeatureVector], list[str], list[str]]:
        rows, names, ids, metadata = build_candidate_feature_rows(
            season=season,
            top_n=pool_size,
            schema=self.schema,
            historical_mode=historical_mode,
        )
        builder = RuntimeFeatureBuilder(schema=self.schema, scaler_params=self.scaler_params)
        vectors: list[FeatureVector] = []
        built_names: list[str] = []
        built_ids: list[str] = []
        for row, player_name, player_id, meta in zip(rows, names, ids, metadata):
            vectors.append(
                builder.build_vector(
                    player_name=player_name,
                    named_features=row,
                    player_id=player_id,
                    metadata=meta,
                )
            )
            built_names.append(player_name)
            built_ids.append(player_id)
        return vectors, built_names, built_ids

    def build_candidate_pool(
        self,
        season: int | None = None,
        pool_size: int = 50,
        top_n: int = 5,
        mode: str = "latest",
        after_id: str | None = None,
        historical_mode: bool = False,
    ) -> tuple[Mapping[str, Any], list[FeatureVector]]:
        vectors, names, ids = self._build_vectors(season, pool_size, historical_mode=historical_mode)
        if not vectors:
            raise ValueError("No candidate vectors built for the pool")

        matrix = np.asarray([vector.vector for vector in vectors], dtype=np.float32)
        self.handler.load_model()
        predictions = np.asarray(self.handler.predict(matrix))
        if predictions.ndim != 2 or predictions.shape[1] != 2:
            raise ValueError("Model output must contain [not_mvp, mvp] probabilities")

        candidates = []
        for idx, vector in enumerate(vectors):
            p_not, p_mvp = predictions[idx]
            candidates.append(
                {
                    "player_id": ids[idx] or "",
                    "player_name": names[idx],
                    "mvp_probability": float(p_mvp),
                    "not_mvp_probability": float(p_not),
                    "vector": vector.vector.tolist(),
                    "metadata": vector.metadata,
                }
            )

        candidates.sort(key=lambda record: (-record["mvp_probability"], record["player_id"]))

        start = 0
        if after_id:
            for idx, record in enumerate(candidates):
                if record["player_id"] == after_id:
                    start = idx + 1
                    break
        sliced = candidates[start:]
        selected = sliced[:top_n]
        next_cursor = ""
        if selected and len(sliced) > top_n:
            next_cursor = selected[-1]["player_id"]

        displayed_probability_total = sum(candidate["mvp_probability"] for candidate in selected)

        results = []
        for rank_offset, candidate in enumerate(selected):
            entry = dict(candidate)
            entry["mvp_rank"] = start + rank_offset + 1
            entry["mvp_probability_raw"] = float(candidate["mvp_probability"])
            entry["mvp_share_of_top_n"] = (
                float(candidate["mvp_probability"]) / displayed_probability_total
                if displayed_probability_total > 0
                else 0.0
            )
            results.append(entry)

        emit_metric(
            "candidate_pool_scored",
            float(len(candidates)),
            {"mode": mode, "season": str(season or "latest")},
        )

        payload = {
            "mode": mode,
            "feature_schema_version": self.schema.version,
            "scaler_version": self.scaler_version,
            "candidate_pool": {
                "definition": self.pool_definition,
                "pool_size": len(candidates),
                "top_n": top_n,
                "historical_mode": historical_mode,
                "snapshot_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "after_id": after_id,
                "next_cursor": next_cursor,
            },
            "results": results,
        }
        return payload, vectors
