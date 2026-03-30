from __future__ import annotations

from typing import Optional

import numpy as np
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from flask_app.models import ModelHandler
from flask_app.schemas import PredictionRequest
from src.features import CandidatePoolService

api_blueprint = Blueprint("api", __name__)
_model_handler: Optional[ModelHandler] = None


def configure_routes(handler: ModelHandler) -> None:
    global _model_handler
    _model_handler = handler


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    if _model_handler is None:
        return jsonify({"error": "Model handler not initialized."}), 500

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        parsed = PredictionRequest.model_validate(payload)
    except ValidationError as exc:
        return (
            jsonify({
                "error": "Invalid payload.",
                "details": exc.errors(include_url=False, include_context=False),
            }),
            400,
        )

    features = np.array([parsed.features], dtype=np.float32)
    predictions = np.asarray(_model_handler.predict(features))
    return jsonify(
        {
            "predictions": predictions.tolist(),
            "count": len(predictions),
            "model_version": _model_handler.model_version,
        }
    )


@api_blueprint.route("/candidate_pool", methods=["GET"])
def candidate_pool():
    if _model_handler is None:
        return jsonify({"error": "Model handler not initialized."}), 500

    try:
        top_n = int(request.args.get("top_n", 5))
    except ValueError:
        return jsonify({"error": "`top_n` must be an integer."}), 400

    season_param = request.args.get("season")
    season = None
    if season_param:
        try:
            season = int(season_param)
        except ValueError:
            return jsonify({"error": "`season` must be an integer."}), 400
    try:
        pool_size = int(request.args.get("pool_size", 30))
    except ValueError:
        return jsonify({"error": "`pool_size` must be an integer."}), 400

    mode = request.args.get("mode", "latest")
    cursor = request.args.get("cursor")

    service = CandidatePoolService(handler=_model_handler)
    try:
        payload, _ = service.build_candidate_pool(
            season=season,
            pool_size=pool_size,
            top_n=top_n,
            mode=mode,
            after_id=cursor,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    payload["model_version"] = _model_handler.model_version
    return jsonify(payload)
