from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, ValidationError, field_validator

EXPECTED_FEATURES = 24


class PredictionRequest(BaseModel):
    """Pydantic model for validating incoming prediction requests."""

    features: List[float] = Field(..., description="List of 24 numeric features in model order")

    @field_validator("features")
    @classmethod
    def validate_feature_length(cls, value: List[float]) -> List[float]:
        if len(value) != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} feature values, received {len(value)}"
            )
        return value
