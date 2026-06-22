# UI Planning Meeting — Local Pipeline Control & Monitoring

**Date:** 2026-03-30
**Attendees:** Vectoriza (Feature), Slate (Data), Bridge (Backend/API), Echo (Sentiment), Crosscheck (QA/Test), Ledger (Docs/Tracking), Vista (UI/UX)

## Goals
1. Build a lightweight *local* webapp (hosted via `localhost`) that lets operators trigger key pipeline pieces (vectors, scaler refreshes, candidate pool exports) without reopening terminals.
2. Surface the top-N candidate pool, their stats/features, and the latest instrumentation metrics so the team can visually confirm the pipeline is healthy.
3. Keep the UI close to the existing Flask stack: reuse `/predict` and `/candidate_pool` endpoints and add a small dashboard route to wrap them.

## Decisions
- **Stack:** Reuse the Flask backend and add a new blueprint (e.g., `/ui`) that serves a minimal React (or Svelte) SPA built with Vite. The SPA calls the existing API for data and a new control endpoint (`/control/run`) for triggering scripts. Static assets can live under `flask_app/static/ui` and the SPA can be built locally via `npm run build`.
- **Backend controls:** Bridge will expose a protected blueprint that invokes the CLI scripts under `.venv312` (via `subprocess.run`). The control panel will allow users to run `scripts/build_runtime_feature_vectors.py`, `scripts/build_candidate_pool_vectors.py`, and optionally `scripts/player_mvp_pipeline.py`. Results stream back as JSON (status, logs). Crosscheck wants the control buttons to show success/failure plus audit trails.
- **Data views:** Vista will design the UI with the following panes:
  1. **Pipeline control row** (buttons + status badges). Shows last-run timestamps, `stats_rows_loaded`, `sentiment_entries_processed`, `candidate_pool_scored` metrics (consumed from new monitoring logger).
  2. **Candidate table** showing the top-N results from `/candidate_pool`, including `player_name`, `player_id`, main stats (`FGM`, `BPM`, `SRS`), `sentiment_avg`, and `mvp_probability`. Add a sparkline or ranking indicator.
  3. **Feature widget** giving the normalized vector values per player (toggle row to expand). Use charts/labels to highlight high/low values.
  4. **Monitoring timeline** showing metric logs and last scaler version + model version.
- **Texcoord & security:** Slate reminded us to flag when data is stale and show ingestion timestamps; include a freshness indicator derived from the metrics and candidate snapshot timestamp.
- **Docs/Runbooks:** Ledger will capture the UI design in `docs/ui-plan.md` (this document) and keep track of any follow-ups, including accessible treadmill for verifying metrics.

## Next Tasks
1. **Bridge & Crosscheck**: Implement a control blueprint (`/control/run`) that executes the selected scripts inside `.venv312`, streams logs, and emits audit metadata. Add unit tests verifying the endpoint rejects invalid commands and logs metrics.
2. **Vista**: Build the SPA wireframe (React/Vite). Provide `npm run dev` instructions, connect buttons to the control endpoint, and render data from `/candidate_pool`. Add stub for vector detail toggles.
3. **Slate & Echo**: Ensure monitoring metrics and ingestion timestamps are exposed via a new `/metrics/latest` JSON endpoint so the UI can show `stats_rows_loaded`, `sentiment_entries_processed`, `candidate_pool_scored`, and data freshness. Confirm instrumentation docs mention the new endpoint.
4. **Ledger**: Track this meeting and the follow-up tasks in `docs/subagent-feedback.md` plus a UI section in `docs/agent-checklist.md`; include a checklist entry for `npm run dev` instructions and new endpoint references.
5. **Vectoriza**: Provide guidance in the UI detail view on which stats contribute to each MVP candidate’s score; maybe annotate feature highlights (FGM weight, BPM).

## Acceptance Criteria
- Webapp runs locally (Flask hosts SPA or proxy to `npm run dev`) and allows triggering pipeline scripts safely with status feedback.
- Candidate table displays Top-N results, with stats, normalized vector toggles, and metrics for sanity checking.
- Monitoring section surfaces ingestion metrics and pipeline timestamps so users can see freshness.
- All controls/data views are documented in `docs/ui-plan.md`, `docs/agent-checklist.md`, and this meeting note.
- UI user flow is validated by Crosscheck via manual testing and captured in progress logs.
