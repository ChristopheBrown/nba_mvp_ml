from __future__ import annotations

import numpy as np

import pandas as pd

from src.features.pipeline import (
    FORCE_INCLUDED_PLAYER_NAMES,
    _sentiment_for_player,
    build_candidate_feature_rows,
)
from src.features.schema import load_feature_schema


def test_candidate_feature_rows_match_schema() -> None:
    schema = load_feature_schema()
    requested_top_n = 5
    rows, player_names, player_ids, metadata = build_candidate_feature_rows(top_n=requested_top_n)

    assert len(rows) >= requested_top_n
    assert len(player_names) == len(rows)
    assert len(player_ids) == len(rows)
    assert len(metadata) == len(rows)

    for row in rows:
        assert set(row.keys()) == set(schema.vector_order)
        assert all(not np.isnan(value) for value in row.values())


def test_postseason_narrative_overrides_apply_to_priority_players() -> None:
    luka_values, luka_avg = _sentiment_for_player("Luka Doncic", {})
    jokic_values, jokic_avg = _sentiment_for_player("Nikola Jokic", {})
    sga_values, sga_avg = _sentiment_for_player("Shai Gilgeous-Alexander", {})
    wemby_values, wemby_avg = _sentiment_for_player("Victor Wembanyama", {})

    assert luka_values["sentiment_2"] == 10.0
    assert luka_avg == 10.0
    assert jokic_values["sentiment_1"] == 10.0
    assert jokic_avg == 10.0
    assert sga_values["sentiment_14"] == 9.5
    assert wemby_values["sentiment_3"] == 10.0


def test_force_included_players_extend_minutes_based_pool(monkeypatch) -> None:
    schema = load_feature_schema()

    base_df = pd.DataFrame(
        [
            {
                "PLAYER_ID": "1",
                "PLAYER_FULLNAME": "Player A",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MIN": 3000,
                "PTS": 25,
                "AST": 5,
                "REB": 8,
                "TOV": 3,
                "STL": 1,
                "BLK": 1,
                "GP": 80,
                "FGA": 18,
                "FTA": 6,
                "FGM": 9,
                "OREB": 2,
                "DREB": 6,
            },
            {
                "PLAYER_ID": "2",
                "PLAYER_FULLNAME": "Player B",
                "TEAM_ID": 1,
                "TEAM_ABBREVIATION": "AAA",
                "MIN": 2900,
                "PTS": 24,
                "AST": 6,
                "REB": 7,
                "TOV": 2,
                "STL": 1,
                "BLK": 1,
                "GP": 80,
                "FGA": 17,
                "FTA": 5,
                "FGM": 8,
                "OREB": 1,
                "DREB": 6,
            },
            {
                "PLAYER_ID": "3",
                "PLAYER_FULLNAME": "Victor Wembanyama",
                "TEAM_ID": 2,
                "TEAM_ABBREVIATION": "SAS",
                "MIN": 1200,
                "PTS": 21,
                "AST": 4,
                "REB": 10,
                "TOV": 3,
                "STL": 1,
                "BLK": 3,
                "GP": 40,
                "FGA": 16,
                "FTA": 7,
                "FGM": 8,
                "OREB": 2,
                "DREB": 8,
            },
        ]
    )

    monkeypatch.setattr("src.features.pipeline._load_season_stats", lambda season: base_df.copy())
    monkeypatch.setattr("src.features.pipeline._latest_season", lambda: 2025)
    monkeypatch.setattr("src.features.pipeline.load_sentiment_scores", lambda path=None: {})

    rows, names, ids, metadata = build_candidate_feature_rows(top_n=2, schema=schema)

    assert len(rows) == 3
    assert set(FORCE_INCLUDED_PLAYER_NAMES) >= {"VICTOR WEMBANYAMA"}
    assert "Victor Wembanyama" in names
    assert "3" in ids
    assert any(item["minutes"] == 1200 for item in metadata)
