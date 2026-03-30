# Subagent Feedback

Compiled feedback from the requested subagents (Feature, Data, Backend/API, Sentiment/NLP, QA, and Docs) as of 2026-03-30. Each entry captures priorities, conceptual and code recommendations, risks, and what should be verified before changing code or docs.

## Vectoriza — Feature Engineer
- Prioritize clarifying the MVP feature list (game stats, player context, injury history) and define data quality expectations before touching code.  
- Conceptually, modularize feature transformations (time-window aggregations, opponent adjustments) so experiments can swap components without cascade effects.  
- Code hint: separate extraction, transformation, and validation layers with clear interfaces plus unit tests per layer.  
- Risks: drifting data sources, missing edge cases (e.g., players with limited minutes), and the need to benchmark feature impact before deployment.

## Slate — Data Engineer
- Confirm the key metrics from stats/sentiment feeds (volume, accuracy, freshness) and include schema versioning, timestamps, provenance, and failure indicators to signal when data is stale. Track volume/error rates per source.  
- Architecture idea: treat stats and sentiment ingestion as parallel pipelines with shared monitoring, a validation/enrichment layer, buffering (e.g., Kafka/PubSub), and a metadata store for schema versions plus a health endpoint.  
- Code recommendations: encapsulate validation and enrichment rules, keep ingestion idempotent, instrument structured logs, automate schema regression tests, and expose metrics (ingestion rate, latency, errors).  
- Risks: schema drift, missing timestamps, high-volume spikes overwhelming buffers, and unreliable sentiment APIs; mitigate with monitoring, retries, circuit breakers, and alerts.  
- Verify before coding: definitive schema/version expectations, downstream formatting/latency needs, infrastructure capacity, and monitoring hooks with access to sample payloads and credentials.

## Bridge — Backend/API Engineer
- Define locked contracts for `/candidate_pool` and `/predict` (payloads, filters, confidence metrics) and treat this as a priority before coding.  
- Ensure data freshness/performance: support efficient pagination (cursor/keyset) and guard predict latency with precomputed features or caching; monitor percentiles.  
- Separation of concerns: keep candidate eligibility and scoring in reusable services, not inline in routes.  
- Conceptual/code hints: encapsulate filtering predicates (`isEligible(candidate, criteria)`), return prediction provenance (`{score, features, version}`), and add feature flags around heavy heuristics.  
- Risks: filter changes unexpectedly altering the pool, leaking sensitive data, model version drift—tie responses to version tags.  
- Validate before coding: API spec sign-off, data availability for required fields, operating SLAs, and security/privacy/rate-limit requirements.

## Echo — Sentiment/NLP Engineer
- Start by defining business objectives for sentiment signal (intent vs brand health) so modeling choices and downstream actions are aligned; ensure pipelines cover diverse, labeled language/tones/channels to avoid blind spots.  
- Keep preprocessing (tokenization/normalization) decoupled from modeling, blend lexicon heuristics with supervised models, and instrument metrics (accuracy, calibration, drift) plus logging for monitoring.  
- Risks: overfitting to slang/bias, production feedback loops reinforcing errors, and compliance issues—guard with validations, drift detection, and privacy controls.  
- Validate before coding: confirm use cases/goals, labeled data availability/quality, required latency throughput, and compliance/security constraints.

## Crosscheck — QA/Test Engineer
- Testing priorities: validate stability via unit/integration tests, run performance/regression checks in CI, confirm data contracts, and cover key user flows end-to-end.  
- Automation suggestions: parameterized API contract tests, smoke tests per build, and alerting for flaky tests or coverage drops.  
- Risks: edge-case inputs, schema drift, load-induced race conditions; mitigate with synthetic/fuzz tests and monitoring.  
- Verify before coding: requirements/spec clarity, test data availability, instrumentation/logging readiness, and CI environment parity with production.

## Ledger — Docs/Tracking Engineer
- Documentation priority: keep the feedback document clear about status, action items, decisions, dependencies, and recurring concerns so stakeholders can skim without parsing raw logs.  
- Verification steps: cite each entry’s source (meetings, tickets, subagent output), confirm summaries with subagents, and include a validation checklist (reviewed, linked docs, next check date) per update for auditability.  
- Formatting advice: use consistent headings (Priority, Status, Action Items, Follow-up), bullet lists, bold key decisions, tag entries with the responsible subagent, and include a “Subagent feedback” section to surface their insights or escalations before publishing.
