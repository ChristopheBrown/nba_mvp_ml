# Demo-Ready Serving Plan (24-nn-1)

## Goal
Allow anyone with this repo to spin up the MVP predictor API locally (or via Docker) using the existing `24-nn-1` model artifact, with tests and documentation verifying the flow.

## Deliverables
1. **Packaged model artifact** under `mlops/artifacts/24-nn-1` (or similar), plus config hooks to point the API at it.
2. **Updated serving code** with a clear entrypoint (`app.py`/`wsgi.py`), input schema validation, and clean error handling.
3. **Working container + local commands** (Dockerfile + `make`/README instructions) to run the API.
4. **Tests** covering model loading + `/predict` smoke.
5. **Documentation** (`docs/demo.md` or README section) describing how to run the demo end-to-end.

## Work Breakdown
1. **Artifact & config alignment**
   - Bring the MLflow bundle from `notebooks/mlruns/5/.../nn-model` into `mlops/artifacts/24-nn-1`.
   - Add `config/settings.py` (Pydantic) reading `MLFLOW_MODEL_URI` or `MODEL_ARTIFACT_PATH` from env, defaulting to the packaged path.
   - Ensure file paths work on any machine (no `/Users/cb/...`).

2. **API clean-up**
   - Create `app.py` exposing `create_app()` for `flask run`/gunicorn.
   - Update `ModelHandler` to load either from MLflow or from the packaged artifact via `mlflow.pyfunc.load_model` or `torch.load`.
   - Add schema validation (24 features, numeric) and better error responses/logging.

3. **Docker & commands**
   - Fix Dockerfile entrypoint (`gunicorn flask_app.app:create_app()` or similar) and copy artifacts/config in.
   - Optional `docker-compose.yml` for easy run (API + optional MLflow UI container).
   - Add `Makefile` targets for `make demo` / `make test` (if time permits) or documented commands otherwise.

4. **Testing**
   - Unit test for model loader/config fallback (mock MLflow).
   - Integration test hitting `/predict` with a fixture payload (mock model or actual artifact depending on speed).

5. **Docs & logging**
   - `docs/demo.md` detailing prerequisites, commands, env vars, sample curl.
   - Update README summary + architecture diagram (if needed).
   - Update `SYSTEM_CHANGES.md` if new OS-level tweaks occur (none expected for this stage).

## Testing Plan
- `pytest tests/test_api_demo.py` (new): ensures `/predict` returns expected structure when model returns deterministic output.
- Existing tests (`tests/test_api.py`) updated or expanded to use local artifact.
- Docker build smoke (`docker build .` + `docker run` curl) documented in `docs/demo.md`.

## Open Questions
- Should demo use the real 24-feature model or a fast mock? (default: real artifact).
- Any preference for CLI tool (`make`, `invoke`, Poetry scripts) for running commands?
- Do we want an optional Streamlit/CLI to visualize predictions in this stage? (default: not yet).
