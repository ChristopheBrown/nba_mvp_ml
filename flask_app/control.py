from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from src.monitoring import emit_metric, get_latest_metrics

logger = logging.getLogger("flask_app.control")
control_bp = Blueprint("control", __name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VENV_PYTHON = PROJECT_ROOT / ".venv312" / "bin" / "python3"
if not VENV_PYTHON.exists():
    VENV_PYTHON = Path("python3")


def _run_script(command: list[str], env: Any | None = None) -> dict[str, Any]:
    logger.info("Launching command: %s", " ".join(command))
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    logger.info("Command finished with %s", result.returncode)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


@control_bp.route("/ui", methods=["GET"])
def serve_ui():
    dist = PROJECT_ROOT / "ui-app" / "dist"
    if dist.exists():
        return send_from_directory(dist, "index.html")
    return (
        jsonify({
            "error": "UI not built",
            "instructions": "cd ui-app && npm install && npm run build",
        }),
        404,
    )


@control_bp.route("/pipeline/vector-builder", methods=["POST"])
def vector_builder():
    json_payload = request.get_json(silent=True) or {}
    input_path = json_payload.get("input", "json/sample_player_features.json")
    output_path = json_payload.get("output", "output/runtime_vectors.json")
    command = [str(VENV_PYTHON), str(SCRIPTS_DIR / "build_runtime_feature_vectors.py"), "--input", input_path, "--output", output_path]

    result = _run_script(command)
    emit_metric(
        "pipeline_vector_builder_run",
        float(result["returncode"] == 0),
        {"input": input_path, "output": output_path},
    )
    return jsonify({"command": command, **result})


@control_bp.route("/pipeline/export-candidate-pool", methods=["POST"])
def candidate_pool_export():
    json_payload = request.get_json(silent=True) or {}
    season = json_payload.get("season")
    pool_size = json_payload.get("pool_size", 30)
    top_n = json_payload.get("top_n", 5)
    mode = json_payload.get("mode", "latest")
    args = [str(VENV_PYTHON), str(SCRIPTS_DIR / "build_candidate_pool_vectors.py"), "--pool-size", str(pool_size), "--top-n", str(top_n), "--output", "data_exporters/candidate_pool/ranking.json", "--vectors-output", "data_exporters/candidate_pool/candidate_feature_vectors.json"]
    if season is not None:
        args.extend(["--season", str(season)])
    result = _run_script(args)

    emit_metric(
        "pipeline_candidate_pool_export",
        float(result["returncode"] == 0),
        {"season": str(season or "latest"), "pool_size": str(pool_size), "top_n": str(top_n)},
    )
    return jsonify({"command": args, **result})


@control_bp.route("/monitoring/metrics", methods=["GET"])
def metrics():
    data = get_latest_metrics()
    return jsonify({"metrics": data})
