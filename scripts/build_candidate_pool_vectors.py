#!/usr/bin/env python3
"""Build the candidate pool feature vectors by running the runtime feature pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import RuntimeFeatureBuilder, load_feature_schema
from src.features.pipeline import build_candidate_feature_rows

logger = logging.getLogger(__name__)


def _matrix_from_rows(rows: list[Mapping[str, float]]) -> np.ndarray:
    matrix = np.asarray([list(row.values()) for row in rows], dtype=float)
    mask = np.isnan(matrix)
    if mask.any():
        col_means = np.nanmean(matrix, axis=0)
        col_means[np.isnan(col_means)] = 0.0
        matrix[mask] = np.take(col_means, np.where(mask)[1])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="Season to materialize (defaults to latest available).")
    parser.add_argument("--top-n", type=int, default=30, help="Number of leader players to include in the candidate pool.")
    parser.add_argument("--output", type=Path, default=Path("output/candidate_feature_vectors.json"))
    parser.add_argument("--scaler-output", type=Path, default=Path("json/scaler_params_v1.json"))
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), "INFO"), format="[%(levelname)s] %(message)s")

    schema = load_feature_schema()
    rows, names, ids, metadatas = build_candidate_feature_rows(season=args.season, top_n=args.top_n, schema=schema)

    matrix = _matrix_from_rows(rows)
    scaler = StandardScaler()
    scaler.fit(matrix)
    scaler_params = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}

    builder = RuntimeFeatureBuilder(schema=schema, scaler_params=scaler_params)
    vectors = []
    for row, name, player_id, metadata in zip(rows, names, ids, metadatas):
        vectors.append(
            builder.build_vector(
                player_name=name,
                named_features=row,
                player_id=player_id,
                metadata=metadata,
            ).to_dict()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vectors, indent=2))
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)
    args.scaler_output.write_text(json.dumps(scaler_params, indent=2))

    logger.info("Built %d candidate pool vectors (season=%s top_n=%s)", len(vectors), args.season, args.top_n)
    logger.info("Saved vectors to %s and scaler params to %s", args.output, args.scaler_output)


if __name__ == "__main__":
    main()
