# NBA MVP ML – Gap Analysis (2026-03-02)

## Scope of this pass
- Clone inspected at commit level present in `/Users/agent/.openclaw/workspace/nba_mvp_ml`.
- External data drop verified at `/Users/agent/Downloads/data` (includes `_raw`, `_processed`, `_html`, `pre-sentiment`, `sentiment`, etc.).
- No live API calls, MLflow tracking runs, or credentialed services were executed per budget / ToS constraints.

## High-level findings
1. **Documentation vs. repo reality** – README describes a complete Mage→MLflow→Flask pipeline, but critical artifacts (Mage projects, MLflow runs, Docker wiring) are either missing or reference hard-coded local paths.
2. **Data + artifacts** – The repo’s `data/` directory only holds sparse placeholder folders. The actual working datasets (especially sentiment-enhanced CSVs) live outside the repo (`~/Downloads/data`), and the code (e.g., `src/analysis.py`) still points to `/Users/cb/...` paths.
3. **Serving/deployment** – Flask blueprint exists, but there is no runnable entrypoint or alignment with the provided Dockerfile (which calls a non-existent `run.py`). Model loading depends on an MLflow registry entry (`models:/24-nn-1/1`) without instructions for reproducing or downloading that artifact.
4. **Automation/real-time story** – No scheduler, CLI, or documentation bridges nightly data pulls, narrative scraping, or retraining. Mage metadata exists but there are no run commands or guaranteed compatibility on a fresh machine.
5. **Testing/observability** – Tests only verify that `nba_api` endpoints respond; nothing protects the scraper, feature engineering, or API contract. No monitoring/logging guidance is provided.

## README instructions mapped to actual files

| README Claim | Reality in Repo |
| --- | --- |
| Mage handles ingestion & preprocessing. | `pipelines/` directory contains Mage scaffolding (`metadata.yaml`, block packages), but no instructions, trigger scripts, or sample `mage start` configs. Absolute paths in `src/analysis.py` still reference the old machine. |
| MLflow manages training + registry. | Notebook `notebooks/p4-01-retraining-model-new-sentiment-5.ipynb` sets `mlflow.set_tracking_uri("sqlite:///mlflow.db")` and registers `mvp_prediction_sentiment-5`, yet there is no `mlruns/` directory, migration scripts, or CLI to recreate `mlflow.db`. |
| Flask API exposes `/predict` using the MLflow model. | `flask_app/routes.py` and `models.py` exist, but app startup relies on `flask run` + `FLASK_APP=flask_app`. Dockerfile instead runs `python run.py` (missing). Input payload contract isn’t documented. |
| Deployment via Docker, future AWS plan. | Dockerfile builds but fails at runtime (missing entrypoint, no environment variables, no MLflow artifacts). No `docker-compose` to stand up MLflow, DB, or scheduler. |
| Data directories as listed (`data/raw`, `data/processed`, `mlruns`, `models`). | Repo only contains `data/raw`, `data/processed`, `teams.json`. Key sentiment CSVs and mage outputs are outside; `mlruns/` and `models/` directories are absent. |

## Detailed gap notes

### 1. Data + configuration
- `src/analysis.py` hard-codes `/Users/cb/src/nba_mvp_ml/...` paths for processed players, teams, and sentiment CSVs. On this machine, the freshest data sits in `~/Downloads/data/{_raw,_processed,sentiment}`. Nothing in the repo references that location or provides a `.env`/config mechanism to point to it.
- Sentiment prompts live in `json/mvp-qualitative_updated.json`, but there’s no script describing how they were executed, rate-limited, or how `sentiment_*` columns are appended to CSVs.
- No documented schema for the inference input expected by the Flask API (dimension count, feature ordering, normalization) even though the README presents a random NumPy array example.
- Hidden files likely in original dev machine (e.g., Mage `.env`, OpenAI key files) aren’t tracked. Need confirmation of required secrets for: OpenAI (LLM scoring), News APIs, nba_api (even if keyless, abide by rate limits), BasketBall Reference scraping (site discourages aggressive scraping—current code uses `time.sleep(2)` but no doc of policy).

### 2. Training & MLflow
- Only notebook-based training flow (`p4-01-...ipynb`). No standalone script/CLI for scheduled retraining.
- Notebook references `_X`, `_y`, `train_dataset`, etc., but their construction cells depend on data paths not included in repo. Reproducibility from scratch is unclear.
- `ModelHandler` defaults to `models:/24-nn-1/1`, but there’s no artifact export, no instruction for running `mlflow models serve`, nor fallback local model packaged in repo.
- Without `mlruns/` directory or `mlflow.db`, experiment history cannot be reconstructed; README implies it exists. Need clarity whether artifacts should be checked in, regenerated, or fetched from remote storage.

### 3. Serving/API layer
- `flask_app/create_app()` is fine for `flask run`, but Dockerfile expects `run.py` (missing). Need either `wsgi.py` or `gunicorn` command plus environment variables for model URIs.
- `/predict` logs input to stdout and converts to PyTorch tensor, but does not validate shape, dtype, or missing features. Error handling is coarse (returns stack trace string). No authentication, rate-limiting, or logging guidance.
- Tests don’t hit the Flask endpoint; they only call `nba_api` clients (`tests/test_api.py`). Need at least smoke tests verifying the API loads and returns predictions with mocked model.

### 4. Mage / data pipelines
- `pipelines/metadata.yaml` shows default Mage template. There’s no README describing pipeline names, triggers, or how to supply variables like `season`.
- `data_loaders` use Mage decorators, but block dependency order is undocumented. Example: `player_loader` expects `data` tuple containing MVP table and HTML text; there’s no block showing how to supply that upstream.
- No exported pipeline runs or `mage_variables` folder to replay historical runs. Recreating the entire ingestion from scratch may require manual steps currently only known to you.

### 5. Containerization & deployment
- Dockerfile copies entire repo and runs `python run.py` (file absent) on port 5000 while README instructs to use `flask run` on 5002. Need unified entrypoint, environment variable injection, and instructions for mounting data/model artifacts.
- No `docker-compose.yml` to run supporting services (MLflow, Postgres/DuckDB, scheduler). README hints at AWS migration, but no IaC or instructions exist.

### 6. Automation / “nightly live system” vision
- No scheduler/cron scripts for nightly ingestion, news scraping, or retraining. README’s “Real-Time Potential” is aspirational only.
- Narrative analysis code (`process_mvp_stories_for_year`) depends on manual iterations over top players and writes back to `/Users/cb/.../data/_processed/...`. Needs automation plan, API budgeting, and ToS compliance notes.
- No monitoring/alerting around pipeline failures or data drift.

### 7. Testing, CI/CD, and quality gates
- Test suite doesn’t cover scrapers, feature engineering, sentiment aggregation, training functions, or Flask routes.
- No linting/formatting config (black/ruff/mypy) or GitHub Actions. README markets “productionization,” but quality gates are missing.

### 8. Compliance, rate limits, and cost awareness (per your request)
- **OpenAI sentiment scoring**: unspecified model; previously GPT-4 (per README). Need to document expected prompt costs, token budgets, and rate limits (OpenAI default: 5 RPM/10 TPM for GPT-4 unless raised). Replay would likely exceed your $15 credit if uncontrolled.
- **nba_api**: built on NBA’s public stats endpoints. Unofficial rate limits ~10 requests per second; there’s a `time.sleep` in scraping but no doc referencing the policy.
- **Basketball-Reference scraping**: their robots.txt discourages heavy automated scraping; current code fetches per player/team with 2-second sleeps, but no mention of compliance or caching.
- **News/narrative sources**: not yet implemented beyond prompts. Need to capture TOS + rate limits once sources (NewsAPI, RSS, social media) are defined.

## Items requiring your follow-up/decisions
1. **Data placement** – Should the `~/Downloads/data` drop be moved or symlinked into the repo (e.g., `data_external/`)? Alternatively, do we formalize a storage bucket path? Need guidance before wiring config.
2. **MLflow artifacts** – Do you want to version `mlruns/`/`mlflow.db` inside the repo, host remotely, or regenerate on demand? This dictates how `ModelHandler` should fetch its model.
3. **Mage usage** – Confirm whether Mage is still the orchestrator of choice. If yes, we need to export/import instructions; if not, we can refactor to a plain Python/Prefect/Dagster pipeline.
4. **Sentiment pipeline** – Confirm the expected data volume (top 5? entire ballot?), desired model (GPT-4 vs GPT-4o mini), and acceptable cost/rate-limit plan.
5. **Deployment target** – For the eventual “demo-able live system,” pick whether you want a Docker Compose local demo first or jump straight to a cloud deployment so I can scope infra work accordingly.

## Next-step suggestions (once you review)
1. **Config hygiene** – Introduce `.env` + `pydantic` config, eliminate hard-coded paths, and document how to point to the external data folder.
2. **Artifact packaging** – Decide on MLflow artifact handling (e.g., bundle latest model into `artifacts/` or script downloads from registry).
3. **Docker/entrypoint fix** – Add `app.py`/`wsgi.py`, update Dockerfile to run `gunicorn flask_app:create_app()` (or similar), include `MLFLOW_MODEL_URI` env var, and document container usage.
4. **Data pipeline README** – Author instructions for running ingestion/sentiment steps with sample command(s) and expected inputs/outputs.
5. **Testing baseline** – Add at least API smoke tests + unit tests for the data loaders (mocking HTTP requests) to catch breaking changes early.

Once you let me know which of these areas matter most, I can turn this into a prioritized execution plan (with or without sub-agents) within the cost constraints you mentioned.
