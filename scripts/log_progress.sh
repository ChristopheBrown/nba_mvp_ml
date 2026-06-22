#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

status=$(git status -sb)
last_commit=$(git log -1 --oneline 2>/dev/null || echo "no commits yet")
current_branch=$(git rev-parse --abbrev-ref HEAD)
files_summary=$(git status -sb | head -n 5)

recent_work="${PROGRESS_RECENT_WORK:-wiring json/feature_schema_v1.json through src/features.RuntimeFeatureBuilder instrumentation, rerunning the deterministic candidate pool exports, and keeping docs/agent-checklist.md synchronized with the runtime gating checkpoints.}"
context_runtime="${PROGRESS_CONTEXT_RUNTIME:-confirm instrumentation by flowing json/feature_schema_v1.json metadata through src/features.RuntimeFeatureBuilder, data_loaders/stats_sentiment_loader, and src/analysis.py::load_and_preprocess_data() so WS4 runtime vectors capture provenance and scaler metadata before the next scoring cadence rolls.}"
context_candidate="${PROGRESS_CONTEXT_CANDIDATE:-rerun scripts/player_mvp_pipeline.py to materialize the nightly top-30 minute leaders, refresh deterministic scoring caches in data_exporters/candidate_pool/, and stage the ranked outputs that feed the gating and API delivery paths documented in docs/agent-execution-plan.md.}"
context_docs="${PROGRESS_CONTEXT_DOCS:-mirror the runtime feature builder and candidate pool verification steps, call out release-ready checkpoints (instrumentation smokes, candidate pool rerun, gating sign-off), and surface the immediate follow-up so the next cron log can pick up where this entry leaves off.}"

cat <<LOG >> progress.log
$(date -u +"%Y-%m-%dT%H:%M:%SZ")
branch: $current_branch
status: $status
last_commit: $last_commit
files_summary:
$files_summary
recent_work: $recent_work
context:
- runtime feature builder: $context_runtime
- candidate pool: $context_candidate
- docs/agent-checklist.md: $context_docs
---
LOG
