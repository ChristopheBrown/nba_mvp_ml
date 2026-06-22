# Managerial Brief: MVP Readiness Checklist

## What We’ve Stabilized

- **Feature contract:** The runtime feature builder now uses a locked schema and scaler so the model always receives the 24 numbers it expects. We added template inputs, a validation script, and docs to demonstrate how to build those feature vectors outside of the backend.
- **Pipeline instrumentation:** Our stats and sentiment loaders now feed directly into the runtime builder, keeping per-player metadata and scaler provenance in sync. We log each gating check so reviewers always see the same sequence of validation steps.
- **Candidate pool workflow:** A CLI and HTTP endpoint now produce the deterministic top-30 MVP candidates (by minutes), normalize their feature vectors, and expose the top-N ranking plus metadata so downstream dashboards or gating teams can pull consistent probability snapshots.
- **Documentation & tests:** README + agent checklist entries spell out the tooling, while `tests/test_feature_builder.py`, `tests/test_feature_pipeline.py`, and new `tests/test_candidate_pool.py` guard the core contracts.

## What the Plan Requires Next

- **Nightly exports:** Run `scripts/build_candidate_pool_vectors.py` inside the Python 3.12 sandbox to refresh the scaler, vectors, and the ranked JSON for the top players. Archive these artifacts so the release cadence can always reference the latest snapshot.
- **Endpoint consumption:** Point dashboards or clients at `GET /candidate_pool` with `top_n`/`pool_size` parameters so they receive the same normalized, scored data that the scheduling job produced.
- **Hand-off ready:** Keep the agent checklist and execution plan notes aligned with each run so the next reviewer can see why the instrumentation/candidate-pool updates happened and what to verify.

## Risk Brief

- The MLflow/torch artifact demands Python 3.12; running the candidate pool export or the API outside that environment raises `torch.load` errors. The easiest mitigation is to schedule the export inside the `.venv312` sandbox and document that requirement for operations.
- If we ever rebuild the artifact under a different interpreter, re-export the scaler/schema files so the runtime builder stays compatible.
