#!/usr/bin/env python3
"""Player MVP snapshot scoring pipeline.

Usage example:
    python player_mvp_pipeline.py \
        --player "Luka Doncic" \
        --snapshot nba_mvp_ml/data/luka_snapshot-week1.json \
        --snapshot nba_mvp_ml/data/luka_snapshot-week2.json \
        --criteria nba_mvp_ml/json/mvp-qualitative.json \
        --output nba_mvp_ml/output/luka-2026-03-06.json

The idea:
1. Load a curated set of narrative snippets for the player (headlines, article text).
2. Run those snippets through a deterministic sentiment classifier (DistilBERT) once per criterion.
3. Convert the positive/negative confidence into a 0-10 rating for each of the 15 prompts.
4. Persist the vector alongside metadata so downstream models can consume it reproducibly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests


@dataclass
class ArticleSnapshot:
    source: str
    title: str
    date: str
    text: str


@dataclass
class CriterionScore:
    id: str
    title: str
    rating: int
    label: str
    score: float
    rationale: str


def read_snapshot_file(path: Path) -> List[ArticleSnapshot]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ArticleSnapshot(**entry) for entry in raw]


def gather_snapshots(paths: List[Path]) -> List[ArticleSnapshot]:
    all_snapshots: List[ArticleSnapshot] = []
    for path in paths:
        all_snapshots.extend(read_snapshot_file(path))
    return all_snapshots


def load_criteria(path: Path) -> Dict[str, Dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_role(path: Optional[Path]) -> str:
    """Load the system role for the evaluator from a JSON file, if provided."""
    if path is None:
        return ""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return str(raw.get("role", "")).strip()
    except FileNotFoundError:
        return ""
    except json.JSONDecodeError:
        return ""


def aggregate_text(snaps: List[ArticleSnapshot]) -> str:
    return "\n\n".join(f"{snap.title}\n{snap.text}" for snap in snaps)


def call_ollama_chat(
    model: str,
    system_prompt: str,
    user_content: str,
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    """Call a local Ollama model via the /api/chat endpoint and return the assistant content."""

    url = base_url.rstrip("/") + "/api/chat"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    resp = requests.post(
        url,
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns {"message": {"role": "...", "content": "..."}, ...}
    return data.get("message", {}).get("content", "")


def parse_rating_and_rationale(text: str) -> (int, str):
    """Extract a 0–10 rating and free-form rationale from the model output.

    The criteria JSON instructs the model to begin with [[<RATING> / 10]].
    We are defensive and support some close variants.
    """

    # Look for [[X / 10]] first
    match = re.search(r"\[\[\s*(\d+)\s*/\s*10\s*\]\]", text)
    if not match:
        # Fallback: plain "X / 10"
        match = re.search(r"\b(\d+)\s*/\s*10\b", text)

    if match:
        try:
            rating = int(match.group(1))
        except ValueError:
            rating = 0
        rating = max(0, min(10, rating))
        rationale = text[match.end() :].strip()
    else:
        # If we can't find a rating, default to 0 and keep the whole text as rationale.
        rating = 0
        rationale = text.strip()

    return rating, rationale


def score_player(
    player: str,
    snaps: List[ArticleSnapshot],
    criteria: Dict[str, Dict[str, str]],
    ollama_model: str,
    system_role: str,
) -> List[CriterionScore]:
    results: List[CriterionScore] = []

    for criterion_id, params in criteria.items():
        # Aggregate and lightly truncate the article text to keep prompts manageable.
        text_blob = aggregate_text(snaps)
        max_chars = 4000
        evidence = text_blob[:max_chars]

        user_prompt = (
            f"You are evaluating the NBA MVP candidacy of {player}.\n\n"
            f"Criterion: {params['title']}\n\n"
            f"Question:\n{params['prompt']}\n\n"
            f"Evidence from recent articles about {player}:\n{evidence}\n\n"
            "Instructions:\n"
            "- Answer specifically about this player and this criterion.\n"
            "- First, provide a single numeric rating from 0 to 10 in the format [[<RATING> / 10]].\n"
            "- Then provide one or two concise sentences explaining your rating.\n"
        )

        content = call_ollama_chat(
            model=ollama_model,
            system_prompt=system_role,
            user_content=user_prompt,
        )
        rating, rationale = parse_rating_and_rationale(content)

        # We no longer have a sentiment label; keep a simple placeholder and normalize rating to [0,1].
        label = "OLLAMA"
        score = rating / 10.0 if rating is not None else 0.0

        results.append(
            CriterionScore(
                id=criterion_id,
                title=params["title"],
                rating=rating,
                label=label,
                score=score,
                rationale=rationale,
            )
        )
    return results


def persist_vector(
    player: str,
    snaps: List[ArticleSnapshot],
    criterion_scores: List[CriterionScore],
    output_path: Path,
):
    payload = {
        "player": player,
        "timestamp": datetime.utcnow().isoformat(),
        "snapshot_sources": [snap.source for snap in snaps],
        "scores": [asdict(score) for score in criterion_scores],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Score a player against the 15 MVP prompts.")
    parser.add_argument("--player", required=True, help="Player name (e.g., 'Luka Doncic').")
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        action="append",
        help="One or more paths to saved article snapshot JSON files. Repeat to combine older + newer weeks.",
    )
    parser.add_argument("--criteria", required=True, type=Path, help="Path to the mvp-qualitative JSON file.")
    # Optional role file for the system prompt (defaults to json/mvp-role.json beside the repo root).
    default_role_path = (
        Path(__file__).resolve().parent.parent / "json" / "mvp-role.json"
    )
    parser.add_argument(
        "--role",
        type=Path,
        default=default_role_path,
        help="Path to JSON file containing a `role` string for the system prompt used with the local LLM.",
    )
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("MVP_OLLAMA_MODEL", "smollm"),
        help="Name of the Ollama model to use (e.g., 'smollm'). "
        "Can also be set via the MVP_OLLAMA_MODEL environment variable.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination JSON for the output vector.")
    args = parser.parse_args()

    snaps = gather_snapshots(args.snapshot)
    criteria = load_criteria(args.criteria)
    system_role = load_role(args.role)
    scores = score_player(args.player, snaps, criteria, args.ollama_model, system_role)
    persist_vector(args.player, snaps, scores, args.output)
    print(f"Saved {len(scores)} criterion scores for {args.player} → {args.output}")


if __name__ == "__main__":
    main()
