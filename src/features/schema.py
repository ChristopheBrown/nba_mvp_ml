"""Feature schema helpers for the NBA MVP runtime.

Tracks the v1 vector order, enforces length checks, and exposes loader helpers for schema artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT_DIR / "json" / "feature_schema_v1.json"


class SchemaValidationError(Exception):
    """Raised when the provided schema artifacts do not align with the expected contract."""


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    vector_order: tuple[str, ...]
    normalized: bool = False
    notes: str = ""

    @property
    def vector_length(self) -> int:
        return len(self.vector_order)


def load_feature_schema(path: Path | str | None = None) -> FeatureSchema:
    """Load the JSON-backed schema artifact, defaulting to the bundled v1 schema."""

    artifact_path = Path(path) if path else SCHEMA_PATH
    if not artifact_path.exists():
        raise SchemaValidationError(f"Schema artifact not found: {artifact_path}")

    with artifact_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    vector_order = payload.get("vector_order")
    if not isinstance(vector_order, Sequence):
        raise SchemaValidationError("Schema artifact missing an ordered `vector_order` list")

    return FeatureSchema(
        version=payload.get("feature_schema_version", "unknown"),
        vector_order=tuple(str(item) for item in vector_order),
        normalized=bool(payload.get("normalized", False)),
        notes=str(payload.get("notes", "")),
    )


def validate_schema_artifact(schema: FeatureSchema, expected_length: int) -> None:
    """Ensure the schema has the expected feature count before runtime scoring."""

    if schema.vector_length != expected_length:
        raise SchemaValidationError(
            f"Schema version {schema.version} declares {schema.vector_length} features but {expected_length} expected"
        )
