#!/usr/bin/env python3
"""Collect narrative articles for an MVP candidate.

This script fetches publicly available RSS feeds from trustworthy outlets,
filters for the specified player, and writes a JSON array consumable by
`player_mvp_pipeline.py` (each entry needs `source`, `title`, `date`, `text`).

Usage:
    pip install feedparser
    python article_collector.py \
        --player "Luka Doncic" \
        --output /tmp/luka-week1.json

If you want more feeds, you can pass a TSV/JSON feed manifest via
`--feeds-file`. Otherwise it uses the built-in list (NYTimes Sports RSS,
AP News, Los Angeles Times sports, Marca English, Heavy.com sports feed).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup


@dataclass
class ArticleMeta:
    source: str
    title: str
    date: str
    text: str


DEFAULT_FEEDS: List[Dict[str, str]] = [
    {
        "source": "NYTimes Sports",
        "url": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
    },
    {
        "source": "AP News Sports",
        "url": "https://apnews.com/hub/college-basketball?outputType=xml",
    },
    {
        "source": "Los Angeles Times Sports",
        "url": "https://www.latimes.com/sports/rss2.0.xml",
    },
    {
        "source": "Marca English",
        "url": "https://e00-marca.uecdn.es/rss/home.xml#contenidos",
    },
    {
        "source": "Heavy.com Sports",
        "url": "https://heavy.com/sports/los-angeles-lakers/rss",
    },
]

PLAYER_REGEX_CACHE: Dict[str, re.Pattern] = {}


def compile_player_regex(player: str) -> re.Pattern:
    key = player.lower()
    if key not in PLAYER_REGEX_CACHE:
        PLAYER_REGEX_CACHE[key] = re.compile(re.escape(player), re.IGNORECASE)
    return PLAYER_REGEX_CACHE[key]


def simplify_text(html_text: str) -> str:
    doc = BeautifulSoup(html_text, "html.parser")
    text = doc.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)


def fetch_feed_entries(feed: Dict[str, str], player: str, limit: int) -> List[ArticleMeta]:
    parsed = feedparser.parse(feed["url"])
    regex = compile_player_regex(player)
    hits: List[ArticleMeta] = []

    for entry in parsed.entries:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", entry.get("description", ""))
        combined_text = f"{title}\n{summary}"
        if regex.search(combined_text):
            published = entry.get("published", entry.get("updated", ""))
            text_source = simplify_text(summary)
            hits.append(
                ArticleMeta(
                    source=feed["source"],
                    title=title,
                    date=published or datetime.utcnow().isoformat(),
                    text=text_source,
                )
            )
        if len(hits) >= limit:
            break
    return hits


def load_external_feeds(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Feed manifest missing: {path}")
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    feeds = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                source, url = parts[0], parts[1]
                feeds.append({"source": source, "url": url})
    return feeds


def collect_articles(
    player: str,
    limit_per_feed: int,
    feeds: Iterable[Dict[str, str]],
) -> List[ArticleMeta]:
    collected: List[ArticleMeta] = []
    for feed in feeds:
        try:
            entries = fetch_feed_entries(feed, player, limit_per_feed)
            collected.extend(entries)
        except Exception as exc:
            print(f"Warning: failed to parse {feed['url']}: {exc}")
    return collected


def persist_articles(articles: List[ArticleMeta], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(article) for article in articles]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch article snapshots for a player.")
    parser.add_argument("--player", required=True, help="Player name (e.g., 'Luka Doncic').")
    parser.add_argument("--output", required=True, type=Path, help="Target JSON file for combined snapshots.")
    parser.add_argument(
        "--feeds-file",
        type=Path,
        help="Optional manifest (JSON or TSV) describing extra feeds (source, url).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Max articles per feed (default: 3).",
    )
    args = parser.parse_args()

    feeds = DEFAULT_FEEDS
    if args.feeds_file:
        feeds = load_external_feeds(args.feeds_file)

    articles = collect_articles(args.player, args.limit, feeds)
    if not articles:
        print("Warning: no matching articles found. Output file will still be created.")
    persist_articles(articles, args.output)
    print(f"Saved {len(articles)} snapshots for {args.player} → {args.output}")


if __name__ == "__main__":
    main()
