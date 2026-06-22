# Codex Handoff Log

This file is the durable implementation log for external coding-agent iterations on the NBA MVP project.

Purpose:
- record what Codex or another LLM IDE changed
- preserve rationale for those changes
- give the NBA MVP subagents a stable artifact to review before weekly facilitation
- make it easier to convert implementation work into follow-up decisions, risks, and agenda items

## Instructions for coding-agent iterations

For each iteration, append a new dated section using this template.

### YYYY-MM-DD HH:MM - <short iteration title>

#### Summary
- 

#### Files changed
- 

#### Behavior changes
- 

#### Tests run and results
- 

#### Open risks / unresolved issues
- 

#### Suggested next actions for subagent review
- 

#### Questions for Christophe / human decisions needed
- 

---

## Review expectation

Before each NBA MVP subagent meeting, review this file alongside:
- `docs/nba_mvp_subagent_agenda.md`
- `docs/nba_mvp_subagent_meeting_log.md`
- the current codebase state, tests, docs, and artifacts

The goal is to let implementation changes feed directly into the next meeting's recommendations.

---

### 2026-06-16 11:04 - Baseline Test Contract Alignment

#### Summary
- Ran the focused baseline suite and aligned three stale test expectations with the current runtime contract.
- Preserved the current behavior that force-included narrative players can extend the requested candidate feature row count.
- Preserved the current sentiment clamp behavior where boosted postseason narrative averages cap at 10.0.
- Preserved the current `/predict` response contract requiring `model_version`.

#### Files changed
- `tests/test_feature_pipeline.py`
- `tests/test_api.py`
- `docs/codex_handoff_log.md`

#### Behavior changes
- Test-only change. Runtime behavior is unchanged.

#### Tests run and results
- Before changes: `.venv312/bin/python -m pytest tests/test_candidate_pool.py tests/test_feature_pipeline.py tests/test_api.py tests/test_api_contract.py` -> 7 passed, 3 failed.
- After changes: `.venv312/bin/python -m pytest tests/test_candidate_pool.py tests/test_feature_pipeline.py tests/test_api.py tests/test_api_contract.py` -> 10 passed.

#### Open risks / unresolved issues
- Force-included player behavior is now reflected in tests, but the API response still needs clearer metadata explaining that the scored pool can exceed the requested candidate limit.
- Sentiment joins still rely on brittle name normalization and have not yet been hardened for accented names.
- `/candidate_pool` provenance still uses request-time metadata without fully exposing artifact freshness.

#### Suggested next actions for subagent review
- Review whether force-included narrative players should continue extending the pool or replace lower-ranked players to preserve exact requested size.
- Add provenance fields to `/candidate_pool` and surface experimental/demo-serving status.
- Centralize Unicode-folded player-name normalization for sentiment and override joins.

#### Questions for Christophe / human decisions needed
- Should narrative priority players extend candidate pools for demo visibility, or should they preserve exact pool sizes by replacing lower-ranked candidates?
