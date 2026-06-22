#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
TMP_DIR="$REPO_DIR/tmp"
PLAYER="Luka Doncic"
SNAPSHOT1="$TMP_DIR/${PLAYER// /_}-week1.json"
SNAPSHOT2="$TMP_DIR/${PLAYER// /_}-week2.json"
VECTOR_OUT="$TMP_DIR/${PLAYER// /_}-vector.json"
VENV_DIR="$TMP_DIR/venv"

mkdir -p "$TMP_DIR"

PYTHON_BIN="/usr/local/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install 'numpy<2' feedparser beautifulsoup4 requests

collect_articles() {
  local snapshot="$1"
  echo "Collecting articles for $snapshot..."
  python "$REPO_DIR/scripts/article_collector.py" \
    --player "$PLAYER" \
    --limit 2 \
    --output "$snapshot"
}

collect_articles "$SNAPSHOT1"
collect_articles "$SNAPSHOT2"

echo "Scoring with player_mvp_pipeline..."
python "$REPO_DIR/scripts/player_mvp_pipeline.py" \
  --player "$PLAYER" \
  --snapshot "$SNAPSHOT1" \
  --snapshot "$SNAPSHOT2" \
  --criteria "$REPO_DIR/json/mvp-qualitative.json" \
  --output "$VECTOR_OUT"

cat "$VECTOR_OUT"
deactivate
