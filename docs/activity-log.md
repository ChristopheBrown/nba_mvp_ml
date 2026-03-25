# Activity Log – Demo-Ready Serving

| Timestamp (ET) | Actor | Summary |
| --- | --- | --- |
| 2026-03-02 20:25 | Aaron | Created feature branch `feature/demo-ready-serving` and drafted `docs/demo-plan.md` outlining deliverables/work breakdown. |
| 2026-03-02 22:26 | Aaron | Copied the `24-nn-1` MLflow artifact into `mlops/artifacts/24-nn-1` to make it repo-local and ready for the config/serving layer. |
| 2026-03-03 08:20 | Aaron | Added the Pydantic settings layer + updated the Flask app/model handler to load the packaged artifact with schema validation. |
| 2026-03-03 08:35 | Aaron | Replaced the legacy tests with `/predict` smoke + validation coverage, refreshed the Dockerfile, and authored `docs/demo.md` instructions. |
| 2026-03-03 08:50 | Aaron | Installed project dependencies (incl. torch/mlflow) and ran `python3 -m pytest` successfully to confirm the new serving stack. |
| 2026-03-03 09:25 | Aaron | Installed a pyenv-managed Python 3.12.8 + `.venv312`, reinstalled project deps there, and captured a live Flask demo run hitting `/predict`. |
