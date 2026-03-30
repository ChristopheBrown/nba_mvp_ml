#!/usr/bin/env python3
"""Build and export a candidate pool ranking using the runtime feature builder."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.preprocessing import StandardScaler

from flask_app.models import ModelHandler
from src.features import CandidatePoolService, load_feature_schema
from src.features.pipeline import build_candidate_feature_rows

DEFAULT_POOL_DIR = Path("data_exporters/candidate_pool")
DEFAULT_SCALER_PATH = Path("json/scaler_params_v1.json")

logger = logging.getLogger(__name__)


def compute_scaler_params(
    season: int | None,
    pool_size: int,
) -> dict[str, Sequence[float]]:
    schema = load_feature_schema()
    rows, *_ = build_candidate_feature_rows(season=season, top_n=pool_size, schema=schema)
    matrix = np.asarray([list(row.values()) for row in rows], dtype=float)
    scaler = StandardScaler()
    scaler.fit(matrix)
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def write_json(path: Path, data: dict[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, help="Season to materialize (defaults to latest available).")
    parser.add_argument("--pool-size", type=int, default=30, help="Size of the candidate pool (default 30).")
    parser.add_argument("--top-n", type=int, default=5, help="Number of ranked players to return (default 5).")
    parser.add_argument(
        "--ranking-output",
        type=Path,
        default=DEFAULT_POOL_DIR / "latest_candidate_pool.json",
        help="Path to write the ranked candidate payload.",
    )
    parser.add_argument(
        "--vectors-output",
        type=Path,
        default=DEFAULT_POOL_DIR / "candidate_feature_vectors.json",
        help="Path to write the normalized feature vectors.",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=DEFAULT_SCALER_PATH,
        help="Path to persist the scaler parameters used for normalization.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), "INFO"), format="[%(levelname)s] %(message)s")

    scaler_params = compute_scaler_params(season=args.season, pool_size=args.pool_size)
    write_json(args.scaler_output, scaler_params)

    handler = ModelHandler()
    service = CandidatePoolService(
        handler=handler,
        schema=load_feature_schema(),
        scaler_params=scaler_params,
    )
    payload, vectors = service.build_candidate_pool(
        season=args.season,
        pool_size=args.pool_size,
        top_n=args.top_n,
        mode="latest",
    )

    write_json(args.ranking_output, payload)
    vector_payload = [vector.to_dict() for vector in vectors]
    write_json(args.vectors_output, {"timestamp": payload["candidate_pool"]["snapshot_timestamp"], "vectors": vector_payload})

    logger.info(
        "Exported candidate pool (%s players) with top_n=%s",
        args.pool_size,
        args.top_n,
    )
    logger.info(
        "Ranking saved to %s, vectors saved to %s, scaler params saved to %s",
        args.ranking_output,
        args.vectors_output,
        args.scaler_output,
    )


if __name__ == "__main__":
    main()
