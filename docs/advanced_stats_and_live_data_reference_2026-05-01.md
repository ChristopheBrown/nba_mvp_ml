# Advanced Stats and Live Data Reference for NBA MVP Feature Engineering

_Date: 2026-05-01_

This document captures implementation-oriented notes from Basketball Reference methodology pages plus a practical review of `nba_api` so future functions/scripts can rebuild advanced features reproducibly.

## Why this document exists

The current NBA MVP cleanup effort exposed a broader problem: even when feature names match training, the runtime can still drift semantically if the underlying formulas, stat grains, or source conventions differ. This reference is meant to reduce that drift.

## Core principle

Before adding any advanced stat to a future dataset, define all of the following explicitly:

- source system (`Basketball Reference`, `stats.nba.com`, local derivation, or mixed)
- stat grain (`season totals`, `per game`, `per 36`, `per 100`, `rate`, `possession estimate`)
- formula version
- missing-era fallback rules
- dependencies on team, opponent, and league aggregates
- whether the feature is historical-only, live-computable, or proxy-derived

If these are not fixed, training/serving mismatch is very likely.

---

## 1. Basketball Reference methodology notes

### 1.1 PER
Reference: `https://www.basketball-reference.com/about/per.html`

PER is built in three stages:

1. compute unadjusted PER (`uPER`)
2. adjust for team pace to get adjusted PER (`aPER`)
3. standardize so league average equals `15`

#### Unadjusted PER
Basketball Reference documents:

- `uPER = (1 / MP) * [...]`
- components include `3P`, `AST`, `FG`, `FT`, `TOV`, missed shots, rebounds, steals, blocks, fouls
- it depends on league-level terms:
  - `factor = (2 / 3) - (0.5 * (lg_AST / lg_FG)) / (2 * (lg_FG / lg_FT))`
  - `VOP = lg_PTS / (lg_FGA - lg_ORB + lg_TOV + 0.44 * lg_FTA)`
  - `DRB% = (lg_TRB - lg_ORB) / lg_TRB`

#### Pace adjustment
- `pace adjustment = lg_Pace / team_Pace`
- then `aPER = (pace adjustment) * uPER`

For pre-1973-74 eras where pace cannot be directly computed:
- estimated pace adjustment = `2 * lg_PPG / (team_PPG + opp_PPG)`

#### Final standardization
- compute league average `aPER`, weighted by player minutes
- `PER = aPER * (15 / lg_aPER)`

#### Historical fallback rules documented by Basketball Reference
For earlier eras with missing box score fields:
- zero out `3P`, `TOV`, `BLK`, `STL`
- set `VOP = 1`
- set `DRB% = 0.7`
- set `ORB = 0.3 * TRB`

#### Implementation implications
- PER is not just a player-only formula. It requires league context and team pace context.
- A live PER recomputation pipeline must also ingest season-level league aggregates and team pace inputs.
- If future scripts use PER from another provider, document whether that provider matches Basketball Reference exactly. Do not assume equivalence.

---

### 1.2 Individual Offensive Rating and Defensive Rating
Reference: `https://www.basketball-reference.com/about/ratings.html`

These are Dean Oliver-derived metrics.

#### Offensive Rating (`ORtg`)
Definition:
- points produced per `100` individual possessions consumed
- `ORtg = 100 * (PProd / TotPoss)`

Key intermediate objects:
- `ScPoss`
- `FGxPoss`
- `FTxPoss`
- `TotPoss = ScPoss + FGxPoss + FTxPoss + TOV`
- `PProd`

Important dependencies include:
- player box score stats
- team totals (`Team_FGM`, `Team_AST`, `Team_FTA`, `Team_TOV`, `Team_ORB`, `Team_PTS`, etc.)
- opponent rebounding context
- minute share terms via `qAST`

#### Defensive Rating (`DRtg`)
Definition:
- estimated points allowed per `100` individual possessions faced

Key intermediate objects:
- `Stops = Stops1 + Stops2`
- `Stop% = (Stops * Opponent_MP) / (Team_Possessions * MP)`
- `DRtg = Team_Defensive_Rating + 0.2 * (100 * D_Pts_per_ScPoss * (1 - Stop%) - Team_Defensive_Rating)`

Important dependencies include:
- player `STL`, `BLK`, `DRB`, `PF`, `MP`
- team `DRB`, `STL`, `BLK`, `PF`, possessions
- opponent `FGA`, `FGM`, `ORB`, `TOV`, `FTA`, `FTM`, `PTS`, minutes

#### Important interpretation warning from BRef
- `DRtg` is team-influenced and imperfect for perimeter defense
- box-score-only defensive estimates understate some defender value

#### Implementation implications
- If we want reproducible `ORtg` and `DRtg`, we should either:
  1. pull a canonical version directly from a trusted source, or
  2. recompute from player, team, opponent, and league context in one controlled pipeline
- Ad hoc proxy formulas are dangerous here because these metrics are structurally multilevel.

---

### 1.3 Win Shares
Reference: `https://www.basketball-reference.com/about/ws.html`

Win Shares attempts to allocate team wins across players.

#### Key properties
- total player win shares on a team are roughly equal to that team’s wins
- can be negative
- built from both offensive and defensive components
- `WS = OWS + DWS`

#### Offensive Win Shares, modern era
For `1977-78` onward:
- based on Dean Oliver `points produced` and `offensive possessions`
- `marginal offense = points produced - 0.92 * (league points per possession) * offensive possessions`
- `marginal points per win = 0.32 * (league points per game) * (team pace / league pace)`
- `OWS = marginal offense / marginal points per win`

#### Defensive Win Shares, modern era
For `1973-74` onward:
- based on `Defensive Rating`
- `marginal defense = (player minutes / team minutes) * team defensive possessions * (1.08 * league PPP - DRtg / 100)`
- `DWS = marginal defense / marginal points per win`

#### Older-era fallback logic
Basketball Reference uses separate estimation regimes for:
- `1973-74` to `1976-77` with estimated player turnovers
- pre-`1973-74` offensive approximations using modified points and modified shot attempts
- pre-modern defensive share allocation using proxy weights from minutes/FGA, rebounds, assists, or fouls depending on era

#### Implementation implications
- Win Shares is highly dependency-heavy. It is not a “single formula” stat.
- For MVP work, pulling canonical `WS`, `OWS`, `DWS`, and `WS/48` from a source aligned with training is much safer than partially rebuilding it from incomplete live inputs.
- If we ever rebuild it, the implementation needs explicit era branching and league context tables.

---

### 1.4 Simple Projection System (SPS)
Reference: `https://www.basketball-reference.com/about/projections.html`

SPS is a simple weighted projection method similar to Marcel.

#### Method summary
For a target stat:
- weight prior three seasons `6`, `3`, `1`
- compute weighted player stat sum
- compute weighted expected league-average production for the same minutes
- add `1000` regression-to-mean minutes in denominator and the league-average weighted contribution in numerator
- convert to projected per-36 rate
- apply age adjustment

#### Age adjustment
- if age `< 28`: `(28 - age) * 0.004`
- if age `> 28`: `(28 - age) * 0.002`

For “bad” stats, reverse the age adjustment sign:
- field goals missed
- 3-point field goals missed
- free throws missed
- turnovers
- personal fouls

#### Other notes
- for shooting, project makes and misses rather than attempts directly
- projected attempts are derived from projected makes plus projected misses
- projected points come from projected `FGM`, `3PM`, and `FTM`

#### Implementation implications
- This is useful for forward-looking feature generation when we need lightweight projections for upcoming-season candidates.
- SPS can be a reproducible baseline model for pre-season or sparse-data scenarios.
- We should store it as a transparent baseline, not confuse it with observed season features.

---

### 1.5 Four Factors
Reference: `https://www.basketball-reference.com/about/factors.html`

Dean Oliver’s Four Factors:
- Shooting `40%`
- Turnovers `25%`
- Rebounding `20%`
- Free Throws `15%`

#### Definitions
- `eFG% = (FG + 0.5 * 3P) / FGA`
- `TOV% = TOV / (FGA + 0.44 * FTA + TOV)`
- `ORB% = ORB / (ORB + Opp DRB)`
- `DRB% = DRB / (Opp ORB + DRB)`
- Free throw factor on BRef page: `FT / FGA`

#### Implementation implications
- These are ideal secondary feature families because they are interpretable and relatively easy to reproduce.
- They also help replace brittle proxy features with better-structured rate metrics.
- We should decide carefully whether to store them as player-level, team-level, opponent-level, or rolling-window features.

---

### 1.6 Similarity Scores
Reference: `https://www.basketball-reference.com/about/similar.html`

BRef similarity scores are based on career Win Shares shape and quality, not style.

#### Method summary
- compare players only within compatible position buckets
- rank each player’s season Win Shares from best to worst
- compute a weighted career value using descending weights: best season `1.00`, second `0.95`, third `0.90`, etc.
- compute weighted penalty from absolute season-by-season differences with the same descending weights
- similarity score:
  - `100 * (1 - (2 * penalty / (career_value_a + career_value_b)))`

#### Important caveat
- this is not a playing-style metric
- it is a career-shape and quality metric

#### Implementation implications
- This is useful for candidate-context tooling, not necessarily the core MVP model.
- It could power explainability features such as “career profile is tracking most similarly to ...”
- It requires robust multi-season `WS` history and stable position mapping.

---

### 1.7 FAQ, glossary, and sources pages
References:
- `https://www.basketball-reference.com/about/nba-basketball-faqs.html`
- `https://www.basketball-reference.com/about/glossary.html`
- `https://www.basketball-reference.com/about/sources.html`

#### FAQ usefulness
The FAQ page is mostly general schedule/league info. It is not critical for formulas, but it does confirm current-reference conventions like:
- `82` games in an NBA season
- standard calendar context for current seasons

#### Glossary items we should keep handy
Useful definitions from the glossary for future feature scripts:
- `AST%`
- `BLK%`
- `DRB%`
- `eFG%`
- `Pace`
- `Poss`
- `PProd`
- `TOV%`
- `TRB%`
- `TS%`
- `TSA = FGA + 0.44 * FTA`
- `Usg%`
- `VORP`
- `SRS`
- `WS/48`
- `Award Share`

A few directly relevant formulas from the glossary:
- `TOV% = 100 * TOV / (FGA + 0.44 * FTA + TOV)`
- `eFG% = (FG + 0.5 * 3P) / FGA`
- `TS% = PTS / (2 * TSA)`
- `TSA = FGA + 0.44 * FTA`
- team `Poss` uses the standard possession estimate averaging team and opponent possession estimates
- `Pace = 48 * ((Tm Poss + Opp Poss) / (2 * (Tm MP / 5)))`
- `VORP` is derived from BPM and prorated to an 82-game season
- `SRS` is point differential plus strength of schedule, in points above/below average
- `Award Share = award points / maximum award points`

#### Sources page usefulness
The sources page is most valuable as provenance documentation. It reminds us that Basketball Reference historical data is aggregated and curated from multiple contributors. For this project, the main lesson is:
- when we source historical labels/features from BRef, document that provenance clearly instead of treating it as raw NBA.com data.

---

## 2. `nba_api` review for live feature generation

Primary references:
- `https://github.com/swar/nba_api`
- local clone: `tmp/nba_api_ref`
- package docs tree: `tmp/nba_api_ref/docs`

## 2.1 What `nba_api` is good for

`nba_api` is an API client for NBA.com with strong endpoint documentation.

Repository notes:
- requires `Python 3.10+`
- core dependencies include `requests` and `numpy`
- `pandas` is optional but very useful
- supports custom headers, proxies, and timeouts
- offers both `stats` endpoints and `live` endpoints
- includes static player/team lookup helpers that avoid unnecessary HTTP requests

This matters because it gives us a much cleaner path to reproducible live data retrieval than scraping ad hoc pages.

---

## 2.2 Categories of useful data

### A. Static lookup utilities
From `nba_api.stats.static`:
- `players`
- `teams`

Useful for:
- stable player ID resolution
- team ID resolution
- avoiding brittle name matching in live scripts

This should become the default first step in any live feature script.

### B. Stats endpoints (`stats.nba.com`)
These are strong for season aggregates, splits, dashboards, and box score structured tables.

### C. Live endpoints (`cdn.nba.com` liveData)
These are useful for same-day or in-progress data:
- scoreboard
- live box score
- live play-by-play

This opens the door to nightly refreshes and rolling features without needing page scraping.

---

## 2.3 High-value endpoints for the MVP project

### `LeagueDashPlayerStats`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashplayerstats.md`

Why it matters:
- likely the most important season aggregate endpoint for player features
- supports `MeasureType` values including:
  - `Base`
  - `Advanced`
  - `Misc`
  - `Four Factors`
  - `Scoring`
  - `Opponent`
  - `Usage`
  - `Defense`
- supports `PerMode` values including:
  - `Totals`
  - `PerGame`
  - `Per36`
  - `Per100Possessions`
  - others
- supports season/date filters, context filters, and situational slicing

Key lesson for us:
- when the training contract wants totals, request `PerMode=Totals`
- do not default to per-game or per-36 unless the feature contract explicitly calls for that

### `LeagueDashTeamStats`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashteamstats.md`

Why it matters:
- team-level counterpart needed for many contextual features
- useful for `SRS` alternatives, team quality context, pace, opponent environment, and four-factor context

### `PlayerEstimatedMetrics`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/playerestimatedmetrics.md`

Returns:
- `E_OFF_RATING`
- `E_DEF_RATING`
- `E_NET_RATING`
- `E_AST_RATIO`
- `E_OREB_PCT`
- `E_DREB_PCT`
- `E_REB_PCT`
- `E_TOV_PCT`
- `E_USG_PCT`
- `E_PACE`

Why it matters:
- can provide NBA.com-native estimated metrics instead of homemade proxies
- especially relevant where current runtime logic fabricates rough advanced stat approximations

### `TeamEstimatedMetrics`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/teamestimatedmetrics.md`

Useful for team context analogs:
- `E_OFF_RATING`
- `E_DEF_RATING`
- `E_NET_RATING`
- `E_PACE`
- rebound and turnover percentages

### `LeagueLeaders`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leagueleaders.md`

Useful for:
- ranking sanity checks
- monitoring whether candidate pools omit obvious leaders
- cross-validating totals for points, rebounds, assists, etc.

### `CommonAllPlayers`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/commonallplayers.md`

Useful for:
- current season player universe construction
- roster status and team mapping
- robust candidate-pool joins

### `PlayerGameLog` and `PlayerGameLogs`
Docs:
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/playergamelog.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/playergamelogs.md`

Useful for:
- rolling windows
- recency features
- last `N` games form
- incremental daily refresh pipelines

### `BoxScoreTraditionalV2`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/boxscoretraditionalv2.md`

Useful for:
- game-level player/team raw box stats
- reconstructing rolling totals if needed

### `BoxScoreAdvancedV2`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/boxscoreadvancedv2.md`

Useful for:
- game-level advanced metrics like `OFF_RATING`, `DEF_RATING`, `PACE`, `USG_PCT`, `TS_PCT`, `PIE`
- richer recent-form or matchup-context features

### `BoxScoreFourFactorsV2`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/boxscorefourfactorsv2.md`

Useful for:
- direct four-factor game-level features
- opponent and team factor snapshots

### `LeagueDashPtDefend`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashptdefend.md`

Useful for:
- defender shot suppression context
- optional defense-oriented narrative or matchup features

### `LeagueDashPlayerClutch`
Doc: `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashplayerclutch.md`

Useful for:
- clutch-specific feature experiments
- not necessary for parity with historical training unless the training set used it

### Live endpoints
Docs:
- `tmp/nba_api_ref/docs/nba_api/live/endpoints/scoreboard.md`
- `tmp/nba_api_ref/docs/nba_api/live/endpoints/boxscore.md`
- `tmp/nba_api_ref/docs/nba_api/live/endpoints/playbyplay.md`

Useful for:
- same-day ingest
- game completion monitoring
- rapid nightly feature updates
- future event-derived features if we want to expand beyond season aggregates

---

## 2.4 Practical guidance for future scripts

### Default retrieval strategy
1. resolve player/team IDs with `nba_api.stats.static`
2. use `LeagueDashPlayerStats` and `LeagueDashTeamStats` for season aggregates
3. use `PlayerEstimatedMetrics` and `TeamEstimatedMetrics` when we need estimated pace/rating context directly from NBA.com
4. use game logs and box score endpoints only when building rolling or recency features
5. keep historical BRef-derived features separate from NBA.com-derived features unless the contract explicitly mixes them

### Important caution
The repo docs expose powerful endpoints, but the underlying NBA.com APIs can change. The documentation includes many “last validated” dates from older snapshots. So:
- build wrappers with retry, schema checks, and logging
- persist raw payload snapshots for reproducibility
- validate returned columns before trusting a pipeline run

### Headers / request behavior
The docs explicitly mention support for custom headers, proxy, and timeout settings. We should keep this in mind when operationalizing on hosts that get blocked or throttled.

---

## 3. How these findings improve the NBA MVP data pipeline

This is the part I care about most for the cleanup.

### 3.1 We can define a real feature contract instead of column-name matching
Right now, one of the biggest failures is that a runtime column can share a name with training while representing a different semantic object.

These references give us the raw material to define each feature as:
- exact formula or source endpoint
- unit/grain
- aggregation mode
- season scope
- dependency tables required

That should become the new source of truth for the MVP pipeline.

### 3.2 We should separate canonical sourced stats from derived local stats
Recommended split:

- **Canonical sourced stats**
  - pull directly from BRef or NBA.com if they already exist in the expected form
  - examples: `PER`, `WS`, `DWS`, `VORP`, `ORtg`, `DRtg`, `BPM`, `WS/48`

- **Derived local stats**
  - only compute locally when we have the full dependency graph and explicit formula version
  - examples: Four Factors variants, rolling recent-form features, SPS projections, custom sentiment blends

This reduces silent formula drift.

### 3.3 We can stop using weak homemade proxies where better sources exist
For several advanced metrics, `nba_api` gives us better structured data than rough local approximations.

That means:
- fewer zero-filled trained fields
- less guesswork on pace/rating-style metrics
- lower risk that the model sees unrealistic feature distributions

### 3.4 We can standardize stat grain deliberately
This audit already showed that grain mismatch is dangerous, especially for things like `FGM`.

Using `nba_api` endpoints with explicit `PerMode` lets us lock down whether each feature is:
- `Totals`
- `PerGame`
- `Per36`
- `Per100Possessions`

That is a direct fix path for the current serving/training mismatch problem.

### 3.5 We can build live-refreshable pipelines without re-scraping everything
A practical path emerges:

- use `CommonAllPlayers` to get active player universe
- use `LeagueDashPlayerStats` and `LeagueDashTeamStats` for daily season aggregate refreshes
- use `PlayerGameLogs` plus box score endpoints for incremental updates or recency features
- use live scoreboard/boxscore feeds to know when a day’s games are final

This is much cleaner than relying on ad hoc CSV assembly.

### 3.6 We can document historical-only vs live-computable features
Recommended labels for each feature:
- `historical_source_only`
- `live_api_available`
- `recomputable_from_boxscore_context`
- `proxy_only_do_not_use_for_training_parity`

This would help prevent future confusion about what can be refreshed nightly versus what needs a historical source snapshot.

### 3.7 We can improve explainability and QA
Using reference formulas and documented endpoints makes it easier to build QA checks such as:
- compare local derived `TOV%`, `TS%`, `ORB%`, `DRB%` against source values
- compare totals-vs-rates consistency
- validate candidate pool against league leaderboards
- check that live feature distributions stay within historical training bounds

This would have caught several of the current issues much earlier.

---

## 4. Concrete recommendations for the next cleanup phase

### Highest ROI
1. Create a `feature_contract` document/table for all 24 trained features.
   - include source, formula, grain, dependencies, null rules, and training distribution summary

2. Replace proxy or guessed advanced stats with canonical source pulls wherever possible.

3. Lock `PerMode` and season scope explicitly in every retrieval script.

4. Persist training-derived scaler parameters and stop fitting on live candidate pools.

5. Build a raw-ingest layer that saves exact source snapshots before transformation.

### Strong next additions
6. Add reproducible derived-feature functions for:
   - `eFG%`
   - `TOV%`
   - `TS%`
   - `ORB%`
   - `DRB%`
   - team `Poss`
   - `Pace`

7. Add a lightweight SPS projection module for forward-looking or incomplete-season scenarios.

8. Use `LeagueLeaders` and `CommonAllPlayers` as candidate-pool QA rails so obvious MVP-level players do not disappear from the pool.

### Cautionary note
Do not mix Basketball Reference computed metrics and NBA.com estimated metrics inside a training-parity path unless we explicitly retrain on that mixed-source contract.

---

## 5. Suggested durable reference links

### Basketball Reference methodology
- `https://www.basketball-reference.com/about/per.html`
- `https://www.basketball-reference.com/about/ratings.html`
- `https://www.basketball-reference.com/about/ws.html`
- `https://www.basketball-reference.com/about/projections.html`
- `https://www.basketball-reference.com/about/factors.html`
- `https://www.basketball-reference.com/about/similar.html`
- `https://www.basketball-reference.com/about/nba-basketball-faqs.html`
- `https://www.basketball-reference.com/about/glossary.html`
- `https://www.basketball-reference.com/about/sources.html`

### `nba_api`
- `https://github.com/swar/nba_api`
- `https://pypi.python.org/pypi/nba_api`
- `https://pepy.tech/project/nba-api`
- `https://circleci.com/gh/swar/nba_api`
- `https://github.com/swar/nba_api/blob/master/LICENSE`
- `https://join.slack.com/t/nbaapi/shared_invite/zt-3dc2qtnh0-udQJoSYrQVWaXOF3owVaAw`

### Local cloned docs reviewed
- `tmp/nba_api_ref/docs/table_of_contents.md`
- `tmp/nba_api_ref/docs/package_structure.md`
- `tmp/nba_api_ref/docs/nba_api/stats/examples.md`
- `tmp/nba_api_ref/docs/nba_api/stats/static/players.md`
- `tmp/nba_api_ref/docs/nba_api/stats/static/teams.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashplayerstats.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashteamstats.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/playerestimatedmetrics.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/teamestimatedmetrics.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leagueleaders.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/commonallplayers.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/playergamelog.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/playergamelogs.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/scoreboardv2.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/boxscoretraditionalv2.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/boxscoreadvancedv2.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/boxscorefourfactorsv2.md`
- `tmp/nba_api_ref/docs/nba_api/stats/endpoints/leaguedashptdefend.md`
- `tmp/nba_api_ref/docs/nba_api/live/endpoints/scoreboard.md`
- `tmp/nba_api_ref/docs/nba_api/live/endpoints/boxscore.md`
- `tmp/nba_api_ref/docs/nba_api/live/endpoints/playbyplay.md`

---

## 6. Bottom line for the MVP project

These findings help in three ways:

- they give us authoritative formula references for advanced stats that were previously being treated too loosely
- they give us a realistic live-data acquisition layer through `nba_api`
- they make it possible to redesign the MVP pipeline around a strict, reproducible feature contract instead of “best effort” matching

That should materially reduce training/serving drift, make nightly refreshes feasible, and make future model retraining much safer.
