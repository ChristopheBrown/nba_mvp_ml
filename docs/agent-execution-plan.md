# NBA MVP ML Agent Execution Plan and Spec

## Purpose
This document is the implementation spec and execution plan for an agentic system to productionize the NBA MVP project for:

- Local execution on a developer machine
- Optional cloud deployment (AWS)
- User-facing MVP probability/rank experience
- Nightly stats and sentiment refresh
- Robust feature pipeline, model serving, and retraining workflow

This plan preserves current model behavior while enabling future schema evolution.

## Locked Decisions
1. **Model output semantics**: model returns `[p_not_mvp, p_mvp]`; use `p_mvp` as MVP probability.
2. **Candidate pool**: top 30 players by **season-to-date minutes**.
3. **Ranking output**: UI/API allows user-selected top `N` players (default `N=5`).
4. **Sentiment source default (Option A)**: RSS article collection + local LLM scoring.
5. **Inference normalization**: use training-compatible `StandardScaler` parameters; no refit at inference.
6. **Backward compatibility**: keep existing 24-float vector pathway for current API while adding named feature support.
7. **Advanced metric fallback source**: enforce Basketball-Reference as fallback for non-derivable advanced metrics in v1.

## Existing Constraints in Current Repo
- Current `/predict` endpoint accepts `features` array of length 24 and does not build features server-side.
- Runtime feature assembly is not implemented outside notebooks.
- Stats loaders exist in `data_loaders/` and utility logic in `src/analysis.py`.
- Runtime sentiment scripts exist in `scripts/article_collector.py` and `scripts/player_mvp_pipeline.py`.
- Training preprocess logic standardizes features in `src/analysis.py::load_and_preprocess_data()`.

## v1 Feature Schema Contract
`feature_schema_version = v1`

Ordered vector (length 24):

1. `FGM`
2. `BPM`
3. `DRBPct`
4. `DWS`
5. `OBPM`
6. `PER`
7. `TOVPct`
8. `VORP`
9. `WS/48_x`
10. `Rk_opp_pg`
11. `2P%_opp_pg`
12. `DRB_opp_pg`
13. `SRS`
14. `ORtg`
15. `sentiment_1`
16. `sentiment_2`
17. `sentiment_3`
18. `sentiment_5`
19. `sentiment_6`
20. `sentiment_8`
21. `sentiment_13`
22. `sentiment_14`
23. `sentiment_avg`
24. `WS`

Sentiment-specific details:
- Store raw criterion ratings on 0-10 scale.
- Compute `sentiment_avg` as mean of `sentiment_1..sentiment_15`.
- Standardize all selected features (including sentiment) for model inference.

## Inference and Normalization Contract
### Required behavior
- Model-ready input must be standardized with **training scaler params** aligned to `feature_schema_version`.
- Inference must not fit scaler params from live data.
- Feature order must exactly match schema vector order.

### Schema artifacts required with model
Each promoted model artifact must include:
- `feature_schema.json`
- `scaler_params` containing `mean` and `scale` aligned with `vector_order`
- model metadata including model version and schema version

### Startup validation
On model load:
- Validate `len(vector_order) == expected_feature_length`
- Validate scaler dimensions align with vector order
- Validate schema version compatibility between model and serving

Fail fast on mismatch.

## Prediction API Contract (target state)
### Preferred request format
- `feature_schema_version`
- `inputs_normalized` boolean
- `features_by_name` object (preferred), or legacy `features` array

### Ranking response requirements
- Return `mvp_probability` (`p_mvp`) and `mvp_rank`
- Candidate pool metadata:
  - pool definition
  - pool size
  - requested `top_n`
  - snapshot timestamp
- Optional:
  - `not_mvp_probability`
  - model version
  - schema version
  - evidence metadata (`sentiment_last_updated_at`, `articles_count_used`)

Deterministic ranking:
- sort by `mvp_probability` descending
- tie break by `player_id` ascending

### API example (top-N ranking)
Request:
```
{
  "mode": "latest",
  "feature_schema_version": "v1",
  "top_n": 7
}
```

Response:
```
{
  "mode": "latest",
  "feature_schema_version": "v1",
  "candidate_pool": {
    "definition": "top 30 by season-to-date minutes",
    "pool_size": 30,
    "top_n": 7,
    "snapshot_timestamp": "2026-03-25T02:10:00Z"
  },
  "results": [
    {
      "player_id": "201939",
      "player_name": "Stephen Curry",
      "mvp_probability": 0.17,
      "mvp_rank": 1
    }
  ]
}
```

Projection request:
```
{
  "mode": "projection",
  "feature_schema_version": "v1",
  "top_n": 5,
  "projection_horizon": "end_of_season"
}
```

Projection response:
```
{
  "mode": "projection",
  "feature_schema_version": "v1",
  "candidate_pool": {
    "definition": "top 30 by season-to-date minutes",
    "pool_size": 30,
    "top_n": 5,
    "snapshot_timestamp": "2026-03-25T02:10:00Z"
  },
  "projection": {
    "horizon": "end_of_season",
    "method": "minutes-based extrapolation with conservative shrinkage"
  },
  "results": [
    {
      "player_id": "203999",
      "player_name": "Nikola Jokic",
      "mvp_probability": 0.28,
      "mvp_rank": 1
    }
  ]
}
```

## Data and Pipeline Architecture (Strategy A)
### Stats source
- Primary: NBA API nightly ingestion for core stats and minutes.
- Purpose: stable candidate pool extraction and core feature availability.

### Advanced metrics handling
- Keep fallback source for metrics not derivable directly from NBA API in v1.
- Enforce fallback source as **Basketball-Reference** for v1 non-derivable metrics.
- Build a calculator module to derive applicable advanced metrics from base stats.
- Track feature provenance: `computed`, `sourced`, or `unsupported`.

### Sentiment source (Option A)
- `scripts/article_collector.py` for article snapshots.
- `scripts/player_mvp_pipeline.py` for 15 criterion ratings using local LLM.
- Nightly scoring and cache output for candidate pool players.
- Fallback to last-known sentiment when no new articles or scoring fails.

## New Components to Build
1. **Runtime Feature Builder**
   - Suggested path: `src/features/feature_builder.py`
   - Responsibilities:
     - Load stats + sentiment data for player(s)
     - Create named feature records
     - Enforce schema
     - Vectorize to `(N, d)`
     - Apply normalization using stored scaler params

2. **Advanced Metrics Calculator**
   - Suggested module: `src/features/advanced_metrics_calculator.py`
   - Suggested CLI: `scripts/calculate_advanced_metrics.py`
   - Responsibilities:
     - Compute applicable advanced metrics
     - Emit provenance and coverage report

3. **Candidate Pool Service**
   - Extract top 30 by season-to-date minutes
   - Build batch feature matrix
   - Score and rank top N (user-selectable, default 5)

## Execution Workstreams
### WS1: Feature Contract and Artifacts
- Create schema file and artifact format.
- Add loader/validator in serving startup.
- Add compatibility checks.

### WS2: Hybrid Stats Pipeline
- Build/reuse nightly NBA API ingestion.
- Materialize candidate pool by minutes.
- Integrate fallback advanced metric sources.

### WS3: Sentiment Pipeline (Option A)
- Operationalize article collection and local scoring nightly.
- Persist `sentiment_1..sentiment_15`, `sentiment_avg`, update timestamps.
- Implement stale fallback behavior.

### WS4: Runtime Feature Builder
- Build named feature record for each player.
- Vectorize and normalize to current schema version.
- Support both single player and candidate pool batch.

### WS5: API and Ranking
- Add top-N ranking endpoint behavior (user-selectable `N`, default 5).
- Keep legacy endpoint compatibility.
- Return clean probability/rank payload.

### WS6: Testing and QA
- Unit tests for formulas, mapping, normalization, schema checks.
- Integration tests for pool extraction and batch inference.
- Contract tests for schema mismatch and fallback behavior.

### WS7: Documentation and Runbooks
- Update README with feature/normalization contracts.
- Add runbook for nightly jobs, failure handling, and rollback.

## Acceptance Criteria
1. Top 30 by minutes candidate pool produced nightly.
2. Feature builder produces schema-valid, normalized vectors outside notebooks.
3. Batch scoring returns top N ranked probabilities (default N=5) with deterministic ties.
4. Sentiment pipeline produces criterion ratings and aggregate features nightly.
5. Model load validates schema + scaler artifacts and fails clearly on mismatch.
6. Tests cover core transformations and contracts.

## Feature Coverage Matrix (to be filled by agent)
| Feature | Required in v1 | Source now | Target source | Computation | Status | Notes |
|---|---|---|---|---|---|---|
| FGM | yes | | | | | |
| BPM | yes | | | | | |
| DRBPct | yes | | | | | |
| DWS | yes | | | | | |
| OBPM | yes | | | | | |
| PER | yes | | | | | |
| TOVPct | yes | | | | | |
| VORP | yes | | | | | |
| WS/48_x | yes | | | | | |
| Rk_opp_pg | yes | | | | | |
| 2P%_opp_pg | yes | | | | | |
| DRB_opp_pg | yes | | | | | |
| SRS | yes | | | | | |
| ORtg | yes | | | | | |
| sentiment_1 | yes | | | criterion 1 | | |
| sentiment_2 | yes | | | criterion 2 | | |
| sentiment_3 | yes | | | criterion 3 | | |
| sentiment_5 | yes | | | criterion 5 | | |
| sentiment_6 | yes | | | criterion 6 | | |
| sentiment_8 | yes | | | criterion 8 | | |
| sentiment_13 | yes | | | criterion 13 | | |
| sentiment_14 | yes | | | criterion 14 | | |
| sentiment_avg | yes | | | mean of 1..15 | | |
| WS | yes | | | | | |

## Progress Tracking Template
Add entries as PRs merge:

- `YYYY-MM-DD`: Initialized execution plan with locked decisions.
- `YYYY-MM-DD`: WS1 complete (schema/scaler artifacts and validation).
- `YYYY-MM-DD`: WS2 complete (nightly stats + top-30 pool).
- `YYYY-MM-DD`: WS3 complete (nightly sentiment + fallback).
- `YYYY-MM-DD`: WS4 complete (feature builder + vectorizer + normalization).
- `YYYY-MM-DD`: WS5 complete (ranking API and payload updates).
- `YYYY-MM-DD`: WS6 complete (tests and contracts).

## Remaining Clarifications (optional, non-blocking)
- Top-N defaults to 5 unless changed.
- If minutes source changes, candidate pool contract remains top 30 by season-to-date minutes.
