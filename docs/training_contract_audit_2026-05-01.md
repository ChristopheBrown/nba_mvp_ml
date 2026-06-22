# NBA MVP model training contract audit (2026-05-01)

This document captures the recovered training-data contract for the active 24-feature MVP model flow so the findings do not depend on legacy notebook archaeology.

## Summary

On 2026-05-01 we recovered enough evidence to reconstruct the training contract behind the active `24-nn-1` model family and compare it with the current runtime pipeline.

High-confidence findings:

- The active runtime model artifact is still `mlops/artifacts/24-nn-1`.
- The intended model input is a normalized 24-feature vector.
- The original training-style dataset was recovered from Christophe and matches the 24-feature reduction logic in `src/analysis.py` exactly.
- `sentiment_avg` in the training dataset is the average of **all 15** sentiment columns, not only the subset explicitly included in the final 24-feature vector.
- The current runtime is not fully training-compatible. The biggest breaks are:
  - live-fitted scaler regeneration instead of training-derived scaler reuse
  - feature-definition drift, especially `FGM`
  - three trained features currently hard-coded to zero at runtime
  - simplified proxy formulas for advanced metrics that may not match training-time values

## Recovered training dataset artifacts

### Hard-coded dataset paths found in notebooks

These notebook paths reference the training/retraining dataset:

- `/Users/cb/src/nba_mvp_ml/data/processed/by_season/fully_merged/final_stacked_data.csv`
- `/Users/cb/src/nba_mvp_ml/data/_processed/by_season/fully_merged/final_stacked_data.csv`
- `/Users/cb/src/nba_mvp_ml/data/_processed/by_season/fully_merged/player_index_mapping.csv`

### Files received from Christophe on 2026-05-01

Recovered externally and copied into the workspace during analysis:

- `tmp/player_index_mapping_from_christophe.csv`
- `tmp/final_stacked_data_from_christophe.csv`

### Observed dataset properties

From `tmp/final_stacked_data_from_christophe.csv`:

- total columns: `218`
- rows in received file: `295`
- target column: `mvp`
- no `SEASON_ID` column in the dataset itself
- player/season identity is therefore paired externally through `player_index_mapping.csv`

From `tmp/player_index_mapping_from_christophe.csv`:

- total rows: `295`
- seasons covered: `1980` through `2022`
- unique seasons: `43`
- usually `7` rows per season
- `1980` has `2` rows

Interpretation:

- `final_stacked_data.csv` is the model matrix plus labels
- `player_index_mapping.csv` provides row identity (player + season)

## Exact 24-feature training contract

Applying the exact `load_and_preprocess_data(..., remove_excess_features=True)` logic from `src/analysis.py` to the recovered training dataset leaves exactly these 24 features:

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

This exactly matches:

- `json/feature_schema_v1.json`
- the documentation in `docs/agent-execution-plan.md`

## Sentiment column meanings

Recovered from `json/mvp-qualitative.json`:

- `sentiment_1`: Team Success and Playoff Position
- `sentiment_2`: Impact on Winning
- `sentiment_3`: Narratives and Storylines
- `sentiment_4`: Voter Fatigue
- `sentiment_5`: Clutch Performances
- `sentiment_6`: Media and Fan Sentiment
- `sentiment_7`: Leadership and Intangibles
- `sentiment_8`: Historical Significance
- `sentiment_9`: Defying Expectations
- `sentiment_10`: Competitor Context
- `sentiment_11`: Defensive Impact
- `sentiment_12`: Market Size and Visibility
- `sentiment_13`: Postseason Expectations
- `sentiment_14`: Role and Usage
- `sentiment_15`: Player Popularity

### Important sentiment finding

The recovered training dataset confirms:

- `sentiment_avg` = average of `sentiment_1` through `sentiment_15`
- it is **not** the average of only the subset used directly in the 24-feature vector

Example from the recovered dataset:

- row 0 stored `sentiment_avg`: `5.066666666666666`
- average of all 15 sentiments: `5.0666666667`
- average of only selected subset sentiments: `5.25`

## Training-time preprocessing behavior

Recovered from `src/analysis.py`:

- training preprocessing uses `StandardScaler()`
- normalization is applied with `scaler.fit_transform(X)`

Implication:

- inference should use scaler parameters derived from the original training matrix
- inference should **not** fit a new scaler on live candidate-pool rows

## Current runtime mismatches

### 1. Scaler regeneration is incompatible with training

Current runtime path:

- `scripts/build_candidate_pool_vectors.py` builds current candidate rows
- fits a new `StandardScaler()` on those rows
- writes new parameters to `json/scaler_params_v1.json`

Why this breaks compatibility:

- training normalization was fit on the training dataset
- runtime normalization is being fit on the current live pool
- this changes the normalized representation of the same feature values depending on the current candidate set

### 2. Feature-definition drift, especially `FGM`

The recovered training dataset uses season-total style values for at least several core features.

Examples of training means from the recovered dataset:

- `FGM`: `682.172881`
- `BPM`: `6.794237`
- `DRBPct`: `18.906441`
- `DWS`: `4.403616`
- `OBPM`: `5.412881`
- `PER`: `25.356497`
- `TOVPct`: `12.689831`
- `VORP`: `6.171299`
- `WS/48_x`: `0.220315`
- `SRS`: `3.340881`
- `ORtg`: `108.969153`
- `WS`: `12.764746`

Current runtime currently sets:

- `FGM` from `FGM_pg`

That is a direct mismatch with the recovered training dataset, where `FGM` is the season-total column.

This is a high-confidence breaking issue.

### 3. Three schema features are zeroed out at runtime

Current runtime `build_candidate_feature_rows()` sets:

- `Rk_opp_pg = 0.0`
- `2P%_opp_pg = 0.0`
- `DRB_opp_pg = 0.0`

These are real training features present in the recovered dataset and in the 24-feature schema.

This means a portion of the trained signal is currently removed entirely at inference time.

### 4. Proxy feature formulas may not match training values

Current runtime reconstructs several advanced features using simplified formulas, for example:

- `PER = (PTS + REB + AST - TOV) / MIN`
- `ORtg = PTS / FGA * 100`
- `VORP = BPM * MIN / 48`
- `DWS = WS * 0.4`
- `SRS = 0.1 * PER + 0.05 * ORtg + 0.3 * WS/48`

These may be acceptable approximations for demos, but they are not proven to match the exact feature values in the recovered training dataset.

Given that the recovered training file already contains these columns directly, the training dataset should be treated as the contract source of truth.

### 5. Runtime sentiment is only partially compatible

Current runtime sentiment source:

- `json/sample_sentiment_scores.json`

Observed properties:

- only a small number of named players are present in the sample file
- unspecified players fall back to `DEFAULT_SENTIMENT_VALUE = 5.0`
- certain players receive hard-coded postseason narrative overrides

This preserves the `sentiment_avg` formula but likely distorts the training-era distribution.

### 6. Candidate-pool construction is not training-equivalent

Current runtime:

- builds a pool from a weighted heuristic over production/minutes/etc.
- defaults to top `50`
- force-includes certain players

This affects which players are even scored and can create visibly bad rankings even before model-quality issues are addressed.

## Highest-confidence break points

Ranked from strongest to weaker confidence as ranking-breakers:

1. live-fitted scaler regeneration instead of training-derived scaler reuse
2. feature-definition mismatch, especially `FGM` using per-game instead of total
3. `Rk_opp_pg`, `2P%_opp_pg`, and `DRB_opp_pg` hard-coded to zero
4. proxy formulas for advanced metrics instead of recovered training-aligned values
5. artificial/default-heavy sentiment inputs
6. candidate-pool selection distortions

## Recommended fixes

### Immediate

1. Stop regenerating `json/scaler_params_v1.json` from live candidate-pool rows.
2. Rebuild scaler params from the recovered training dataset and preserve them as the serving artifact.
3. Align runtime feature definitions exactly with the recovered training columns.
4. Populate `Rk_opp_pg`, `2P%_opp_pg`, and `DRB_opp_pg` with real values instead of zeros.

### Next

5. Validate or replace proxy formulas for `PER`, `ORtg`, `VORP`, `DWS`, and `SRS` against training-era merged values.
6. Replace sample/default sentiment with a process that reproduces the training-era sentiment shape more faithfully.
7. Revisit candidate-pool selection only after feature/scaler compatibility is restored.

## Source files referenced during the audit

- `src/analysis.py`
- `json/feature_schema_v1.json`
- `json/scaler_params_v1.json`
- `json/mvp-qualitative.json`
- `json/sample_sentiment_scores.json`
- `src/features/pipeline.py`
- `src/features/feature_builder.py`
- `src/features/candidate_pool.py`
- `scripts/build_candidate_pool_vectors.py`
- `notebooks/p3-01-first-model.ipynb`
- `notebooks/p3-03-mlflow-testing.ipynb`
- `notebooks/p3-04-mlflow-sklearn.ipynb`
- `notebooks/p3-05-mlflow-pytorch.ipynb`
- `notebooks/p3-06-mlflow-pytorch-batch_norm.ipynb`
- `notebooks/p3-06-mlflow-pytorch-dropout_layer.ipynb`
- `notebooks/p3-06-mlflow-pytorch-extra_fc_layer.ipynb`
- `notebooks/p3-06-mlflow-pytorch-hidden_layer_size.ipynb`
- `notebooks/p3-07-mlflow-pytorch-optimization.ipynb`
- `notebooks/p4-01-retraining-model-new-sentiment-5.ipynb`
- `tmp/final_stacked_data_from_christophe.csv`
- `tmp/player_index_mapping_from_christophe.csv`
