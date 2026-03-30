#!/usr/bin/env python3
"""Build runtime feature vectors from named feature dictionaries."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features import RuntimeFeatureBuilder, SchemaValidationError, load_feature_schema

logger = logging.getLogger(__name__)


def load_input_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("players") or payload.get("entries") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("Input JSON must be a list of feature dictionaries or contain a 'players' key.")

    if not rows:
        raise ValueError("Input file does not contain any player entries to vectorize.")

    return [dict(row) for row in rows]


def load_scaler_params(path: Path | None) -> Mapping[str, Iterable[float]] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_vectors(
    builder: RuntimeFeatureBuilder,
    rows: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    vectors: list[Mapping[str, Any]] = []

    for row in rows:
        player_name = row.get("player_name") or row.get("name")
        if not player_name:
            raise ValueError("Each entry must include a `player_name` field.")

        features = row.get("features")
        if not isinstance(features, Mapping):
            raise ValueError("Each entry must include a `features` mapping.")

        player_id = row.get("player_id")
        metadata = row.get("metadata")

        try:
            vector = builder.build_vector(
                player_name=player_name,
                named_features=features,
                player_id=player_id,
                metadata=metadata,
            )
        except SchemaValidationError as exc:
            raise SchemaValidationError(f"{player_name}: {exc}") from exc

        vectors.append(vector.to_dict())

    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to the JSON file containing player features.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for the built vectors.")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("json/feature_schema_v1.json"),
        help="Path to the feature schema artifact (default: json/feature_schema_v1.json).",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        help="Optional JSON file containing `mean` and `scale` arrays used for normalization.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for console output.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    schema = load_feature_schema(args.schema)
    scaler_params = load_scaler_params(args.scaler)
    builder = RuntimeFeatureBuilder(schema=schema, scaler_params=scaler_params)

    rows = load_input_rows(args.input)
    vectors = build_vectors(builder, rows)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(vectors, indent=2))

    logger.info("Built %d vectors and wrote them to %s", len(vectors), output_path)


if __name__ == "__main__":
    main()
