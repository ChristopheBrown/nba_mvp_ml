# NBA MVP Control Center — Baby Steps Tutorial

This guide walks you through running the local dashboard, exercising the pipeline controls, and reviewing the candidate table so you can give feedback quickly.

## 1. Prepare the environment

```bash
# Activate the Python 3.12 sandbox (you already have .venv312 from earlier work)
source .venv312/bin/activate
pip install -r requirements.txt
```

If you haven’t already, install the Node dependencies for the UI (React + Vite):

```bash
cd ui-app
npm install --legacy-peer-deps
```

Leave this shell open—you’ll use it to build or run the UI.

## 2. Run the Flask server with the control endpoints

```bash
cd ..  # back to nba_mvp_ml
export FLASK_APP=flask_app
flask run
```

This starts the Flask backend on `http://127.0.0.1:5000` with the new control blueprint (`/pipeline/vector-builder`, `/pipeline/export-candidate-pool`, `/monitoring/metrics`, `/ui`).

## 3. Launch the UI in dev mode

In a new terminal (with `.venv312` still active if you need to run backend scripts later):

```bash
cd ui-app
npm run dev
```

Vite’s dev server runs on `http://localhost:5173` and proxies API calls to the Flask host. You’ll see the control panel, metrics feed, candidate table, and log area.

## 4. Trigger pipeline actions via the UI

1. Adjust `Season`, `Pool size`, and `Top N` as desired.
2. Click **Run Vector Builder** to re-run `scripts/build_runtime_feature_vectors.py`. The button displays success/failure and stores logs in the “Recent Actions” column.
3. Click **Export Candidate Pool** to run `scripts/build_candidate_pool_vectors.py`. After completion the candidate table refreshes, and the metrics panel logs the latest `candidate_pool_scored` entry.

## 5. Inspect the candidate table

- The table lists the current top-N MVP candidates with MVP probability, team, and metadata.
- Expand a row to see the partial normalized vector; this is great for checking feature ordering (the UI uses just a preview of the vector, but let me know if you want the full 24 features visible). 
- You can adjust `Top N` and `Pool size` and re-run the export to see how new rankings behave.

## 6. Review instrumentation metrics

- The right-hand “Metrics” card shows the most recent `stats_rows_loaded`, `sentiment_entries_processed`, `sentiment_missing_keys`, and `candidate_pool_scored` values. These come from `src/monitoring.py`, so any script run updates them instantly.

## 7. Optional: Build a static bundle

If you want to serve `/ui` from Flask (without running `npm run dev`), run:

```bash
cd ui-app
npm run build
cp -r dist/* ../flask_app/static/ui/
```

Then hit `http://127.0.0.1:5000/ui` and the dashboard will load the static build instead of the dev server.

## 8. Provide feedback

- Note any UI quirks (buttons, layout, missing metrics).
- Test pipeline buttons in both modes (dev + static) and confirm logs/metrics update.
- Mention if you’d like more detail on feature vectors or additional charts. I’ll iterate from your notes.
