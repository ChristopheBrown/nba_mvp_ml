# 24-Feature Contract Matrix for MVP Model Cleanup

_Date: 2026-05-01_

This document turns the current 24-feature model into an explicit remediation matrix.

Primary artifacts reviewed:
- `json/feature_schema_v1.json`
- `json/scaler_params_v1.json`
- `src/features/pipeline.py`
- `src/features/feature_builder.py`
- `src/features/candidate_pool.py`
- `src/analysis.py`
- `docs/training_contract_audit_2026-05-01.md`
- `docs/advanced_stats_and_live_data_reference_2026-05-01.md`

## Status labels used
- **correct-ish**: runtime appears broadly aligned with training intent or at least with scaler distribution
- **mismatched**: runtime computes a different semantic variable than training likely used
- **missing/zeroed**: runtime currently hardcodes or effectively removes a trained signal
- **replacement-needed**: runtime feature exists but should be sourced canonically rather than approximated

## Executive summary

The 24-feature runtime is schema-aligned only at the column-name level.

The contract is currently broken in five major ways:

1. some features use the wrong grain
2. some use homemade formulas that do not match the trained stat family
3. three trained opponent-context features are hardcoded to zero
4. sentiment is structurally distorted versus training
5. the scaler artifact in `json/scaler_params_v1.json` appears to be derived from live/runtime data, not original training data

The scaler means are especially revealing. They imply the model expects values like:
- `FGM` mean `6.3543`
- `PER` mean `0.7670`
- `VORP` mean `1682.5998`
- `WS` mean `0.0149`

That combination is internally inconsistent with canonical Basketball Reference semantics for several features. It strongly suggests the current scaler artifact is not a trustworthy training-contract artifact.

---

## Current 24-feature matrix

| Feature | Runtime source now | Runtime definition now | Likely intended training meaning | Status | Why it is risky / broken | Recommended fix |
|---|---|---|---|---|---|---|
| `FGM` | `src/features/pipeline.py` | `FGM_pg = FGM / GP` then mapped into `FGM` | unclear from current scaler, but name is ambiguous; prior audit suggested training dataset used different semantics than runtime | **mismatched** | runtime explicitly feeds per-game under a totals-style name | lock contract explicitly, either rename to `FGM_pg` and retrain, or feed true training-compatible `FGM` |
| `BPM` | `calculate_bpm()` | handmade weighted box score proxy scaled by minute share | canonical BPM or BRef BPM-like field | **replacement-needed** | current formula is not BPM | source canonical BPM directly if available in historical dataset; otherwise exclude/retrain |
| `DRBPct` | `_augment_player_metrics()` | `DREB / (DREB + OREB)` | canonical defensive rebound percentage | **mismatched** | not the BRef/Oliver definition; ignores opponent/team context | compute canonical `DRB%` using glossary formula and on-floor/team-opponent context where possible |
| `DWS` | `_augment_player_metrics()` | `WS * 0.4` | defensive win shares | **replacement-needed** | this is not DWS | source canonical `DWS` from BRef/history or retrain without it |
| `OBPM` | `calculate_bpm()` | `0.1*PTS + 0.5*AST - 0.25*TOV`, scaled by minute share | canonical offensive BPM | **replacement-needed** | this is not OBPM | source canonical `OBPM` or retrain |
| `PER` | `_augment_player_metrics()` | `(PTS + REB + AST - TOV) / MIN` | Hollinger/BRef PER | **replacement-needed** | this is a crude per-minute productivity proxy, not PER | source canonical `PER` or implement full PER calculation with league/team context |
| `TOVPct` | `_augment_player_metrics()` | `TOV / (FGA + 0.44*FTA + TOV)` | turnover percentage | **correct-ish** | formula is basically right, but naming differs from BRef `TOV%` and scaling source still uncertain | keep formula, but document contract clearly and validate against source |
| `VORP` | `_augment_player_metrics()` | `BPM * MIN / 48` | value over replacement player | **replacement-needed** | canonical VORP is not this simple and is season-normalized | source canonical `VORP` or retrain |
| `WS/48_x` | `_augment_player_metrics()` | `WS / (MIN/48)` | win shares per 48 | **mismatched** | mathematically fine only if `WS` is canonical, but runtime `WS` is not canonical | only valid after fixing `WS` source |
| `Rk_opp_pg` | hardcoded in `build_candidate_feature_rows()` | `0.0` | opponent rank per game or team-opponent context from historical merged set | **missing/zeroed** | trained feature receives no signal | rebuild from original historical merge logic or drop/retrain |
| `2P%_opp_pg` | hardcoded | `0.0` | opponent 2P% allowed per game or similar | **missing/zeroed** | trained feature receives no signal | rebuild from historical team/opponent tables or drop/retrain |
| `DRB_opp_pg` | hardcoded | `0.0` | opponent defensive rebounds per game or similar | **missing/zeroed** | trained feature receives no signal | rebuild from historical team/opponent tables or drop/retrain |
| `SRS` | `_augment_player_metrics()` | `0.1*PER + 0.05*ORtg + 0.3*(WS/48)` | simple rating system from team quality context | **replacement-needed** | this is not SRS | source actual team `SRS` and join to players |
| `ORtg` | `_augment_player_metrics()` | `PTS / FGA * 100` | individual offensive rating | **replacement-needed** | not Dean Oliver ORtg; ignores possessions and assists/team context | source canonical `ORtg` or compute properly |
| `sentiment_1` | `json/sample_sentiment_scores.json` or default | sample score / default `5.0` / possible manual override | training sentiment dimension 1 | **mismatched** | tiny sample file, defaults for most players, manual postseason overrides | rebuild sentiment pipeline or freeze training-compatible sentiment artifact |
| `sentiment_2` | same | same | same | **mismatched** | same issue | same fix |
| `sentiment_3` | same | same | same | **mismatched** | same issue | same fix |
| `sentiment_5` | same | same | same | **mismatched** | same issue | same fix |
| `sentiment_6` | same | same | same | **mismatched** | same issue | same fix |
| `sentiment_8` | same | same | same | same | same issue | same fix |
| `sentiment_13` | same | same | same | same | same issue | same fix |
| `sentiment_14` | same | same | same | same | same issue | same fix |
| `sentiment_avg` | computed from all 15 scores, then possibly override-adjusted | mean of sentiment dimensions with postseason override emphasis | average of all 15 training sentiment dimensions | **partially aligned but distorted** | averaging all 15 is good, but defaults + overrides distort distribution | preserve all-15 averaging, replace source data and remove ad hoc emphasis in parity mode |
| `WS` | `calculate_win_shares()` | `(player_pts / team_pts) * (player_min / team_min)` | win shares | **replacement-needed** | this is not Win Shares | source canonical `WS` or retrain |

---

## What is actually closest to salvageable

These are the least broken features conceptually:
- `TOVPct`
- `sentiment_avg` averaging all 15 dimensions, but only after sentiment source is repaired
- possibly `WS/48_x`, but only if `WS` becomes canonical

Everything else in the non-sentiment block is either clearly proxy-derived, zeroed out, or semantically off.

---

## Scaler artifact findings

`json/scaler_params_v1.json` currently contains:
- version: `2026-05-01T17:33:48+00:00`
- 24-element `mean`
- 24-element `scale`

This version timestamp is a major red flag. It looks like a newly generated artifact, not a preserved original training scaler.

Further, the means imply the following expected raw values:
- `FGM = 6.3543`
- `BPM = 33.6953`
- `DRBPct = 0.7987`
- `DWS = 0.0060`
- `OBPM = 25.5012`
- `PER = 0.7670`
- `TOVPct = 0.1156`
- `VORP = 1682.5998`
- `WS/48_x = 0.000301`
- `SRS = 6.6925`
- `ORtg = 132.3138`
- `WS = 0.01493`

A few observations:
- `TOVPct`, `DRBPct`, and `ORtg` are numerically plausible as rates
- `FGM = 6.35` suggests per-game style scale
- `PER = 0.767` is impossible for canonical PER, but plausible for a crude per-minute formula like the current runtime one
- `WS = 0.0149` is nowhere near canonical season Win Shares, but matches a tiny proxy statistic
- `VORP = 1682.6` is absurd for canonical VORP and indicates severe semantic corruption or bad scaling lineage
- `WS/48_x = 0.000301` is impossible if `WS` were canonical and `MIN` were season minutes, but could emerge from the current broken proxy chain

### Conclusion on scaler lineage
The current scaler appears to have been fit on a runtime-generated proxy feature space, not on the original trustworthy training contract.

That means we should not trust it as a source of truth for feature semantics.

---

## Contract conclusions by feature family

### A. Canonical advanced stats that should not be proxied in parity mode
These should come from historical source truth or be fully reimplemented:
- `BPM`
- `DWS`
- `OBPM`
- `PER`
- `VORP`
- `SRS`
- `ORtg`
- `WS`
- `WS/48_x`

### B. Opponent/team context features that are currently absent
These require historical merge reconstruction or explicit backfill logic:
- `Rk_opp_pg`
- `2P%_opp_pg`
- `DRB_opp_pg`

### C. Simple rate features that can be derived locally with confidence
- `TOVPct`
- potentially `DRBPct`, but only if we are explicit that we are using a simplified non-canonical definition

### D. Sentiment features that need a proper data contract
- `sentiment_1`
- `sentiment_2`
- `sentiment_3`
- `sentiment_5`
- `sentiment_6`
- `sentiment_8`
- `sentiment_13`
- `sentiment_14`
- `sentiment_avg`

These need:
- frozen source artifact or reproducible collection pipeline
- consistent player coverage
- explicit defaulting rules
- removal of manual postseason inflation in parity mode

---

## Recommended implementation order

### Phase 1: Restore trustworthy contract artifacts
1. recover or rebuild the original training dataset and exact scaler used for the deployed model
2. preserve those artifacts under versioned paths
3. stop overwriting `json/scaler_params_v1.json` from live/runtime flows

### Phase 2: Reconstruct the historical 24-feature table exactly
4. rebuild the historical merge path that produced:
   - `Rk_opp_pg`
   - `2P%_opp_pg`
   - `DRB_opp_pg`
5. verify exact semantics for every advanced stat column in the historical stacked data

### Phase 3: Split parity mode from live mode
6. add a strict **training-parity mode**
   - only uses fields that match historical contract exactly
   - no manual overrides
   - no proxy substitutions
7. separately add a **live-experimental mode**
   - can use `nba_api`
   - can use estimated metrics and fresh features
   - should not score through the current model unless retrained

### Phase 4: Rebuild serving logic
8. refactor `build_candidate_feature_rows()` so each field is built by a named contract function
9. add schema tests that assert:
   - source type
   - grain
   - null behavior
   - distribution bounds
10. add contract validation against reference snapshots before scoring

---

## Recommended feature disposition for the current model

### Safe to keep with minimal change
- `TOVPct`

### Keep only after upstream repair
- `FGM`
- `DRBPct`
- `WS/48_x`
- `sentiment_avg`

### Must be replaced with canonical source or excluded from parity serving
- `BPM`
- `DWS`
- `OBPM`
- `PER`
- `VORP`
- `SRS`
- `ORtg`
- `WS`

### Must be reconstructed from historical context joins
- `Rk_opp_pg`
- `2P%_opp_pg`
- `DRB_opp_pg`

### Must be rebuilt from a real sentiment contract
- `sentiment_1`
- `sentiment_2`
- `sentiment_3`
- `sentiment_5`
- `sentiment_6`
- `sentiment_8`
- `sentiment_13`
- `sentiment_14`
- `sentiment_avg`

---

## Bottom line

The current 24-feature runtime should not be treated as a faithful serving implementation of the trained model.

It is better understood as a schema-compatible approximation layer with major semantic drift.

The biggest finding from this matrix is not just that a few features are off. It is that most of the advanced-stat backbone of the model is currently proxy-based or missing, and the active scaler artifact appears to belong to that proxy world rather than the original training world.

That means the right cleanup path is:
- recover exact contract artifacts first
- rebuild strict parity mode second
- only then reintroduce live feature generation in a clearly separated mode
