# Demo Instructions – MVP Predictor API

This guide shows how to run the packaged `24-nn-1` model locally or via Docker and send a sample request to the `/predict` endpoint.

## 1. Prerequisites
- Python 3.12+ (the packaged artifact relies on newer code-object metadata that ships with 3.12; older interpreters trigger `TypeError: code() takes at most 16 arguments (18 given)` while `torch.load` reads the artifact. On macOS, install via `brew install python@3.12` and use `/usr/local/bin/python3.12`, or install via `pyenv`/distro packages on Linux.)
- Virtualenv (built in to modern Python installs).
- Docker (optional, for containerized runs).

## 2. Local sandbox (Python 3.12)
```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
make demo
```
`make demo` already exports `FLASK_RUN_PORT=5000` and `MVP_MODEL_ARTIFACT_PATH=$(pwd)/mlops/artifacts/24-nn-1`, so the packaged artifact is loaded automatically. If your artifact lives elsewhere, set `MVP_MODEL_ARTIFACT_PATH` to the desired path before running. Wait for Werkzeug to report `Running on http://127.0.0.1:5000` before hitting the endpoint.

## 3. Example `/predict` request
```bash
curl -s -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```
Sample response from the zero-vector request:
```json
{
  "count": 1,
  "predictions": [[0.999840497970581, 0.00015953574620652944]]
}
```

## 4. Run via Docker
*(Recommended when the host Python version < 3.12.)*
```bash
make docker-demo
```
This uses `docker-compose.yml` to build the image, wire gunicorn, and expose `http://localhost:8000`. The compose file mounts `mlops/artifacts/24-nn-1` so the packaged artifact is available without additional configuration.

## 5. Tests
```bash
make test  # wraps pytest -q
```
The suite now includes handler/unit coverage plus a patched integration smoke for `/predict`.

## 6. Troubleshooting
- **Model artifact missing**: ensure `mlops/artifacts/24-nn-1` exists or override `MVP_MLFLOW_MODEL_URI` to point at your registry.
- **Torch missing**: install `torch==2.2.2` (already in `requirements.txt`).
- **Docker memory**: the inference container requires ~2 GB of RAM; allocate accordingly.
- **Python version mismatch**: running the demo with Python 3.11 or older raises `TypeError: code() takes at most 16 arguments (18 given)` as the artifact is deserialized. Create the sandbox with Python 3.12 (see Section 2) so the code objects match.
