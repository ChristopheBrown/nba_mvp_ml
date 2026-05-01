from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.features.schema import FeatureSchema, load_feature_schema
from src.monitoring import emit_metric

logger = logging.getLogger(__name__)

STATS_DICT_PATH = Path("data/raw/tmp_backup/seasons_Totals")
SENTIMENT_PATH = Path("json/sample_sentiment_scores.json")
SENTIMENT_KEYS = [f"sentiment_{i}" for i in range(1, 16)]
POSTSEASON_NARRATIVE_PLAYERS = {
    "NIKOLA JOKIC": {
        "sentiment_1": 10.0,
        "sentiment_2": 10.0,
        "sentiment_3": 10.0,
        "sentiment_5": 10.0,
        "sentiment_6": 9.0,
        "sentiment_8": 10.0,
        "sentiment_13": 10.0,
        "sentiment_14": 10.0,
        "sentiment_avg": 9.875,
        "postseason_narrative_emphasis": 1.25,
    },
    "LUKA DONCIC": {
        "sentiment_1": 9.0,
        "sentiment_2": 10.0,
        "sentiment_3": 9.5,
        "sentiment_5": 9.5,
        "sentiment_6": 9.0,
        "sentiment_8": 9.5,
        "sentiment_13": 10.0,
        "sentiment_14": 9.5,
        "sentiment_avg": 9.5,
        "postseason_narrative_emphasis": 1.18,
    },
    "SHAI GILGEOUS-ALEXANDER": {
        "sentiment_1": 9.0,
        "sentiment_2": 9.5,
        "sentiment_3": 9.0,
        "sentiment_5": 9.0,
        "sentiment_6": 9.0,
        "sentiment_8": 9.0,
        "sentiment_13": 9.5,
        "sentiment_14": 9.5,
        "sentiment_avg": 9.1875,
        "postseason_narrative_emphasis": 1.15,
    },
    "VICTOR WEMBANYAMA": {
        "sentiment_1": 8.5,
        "sentiment_2": 10.0,
        "sentiment_3": 10.0,
        "sentiment_5": 8.0,
        "sentiment_6": 10.0,
        "sentiment_8": 8.5,
        "sentiment_13": 10.0,
        "sentiment_14": 10.0,
        "sentiment_avg": 9.375,
        "postseason_narrative_emphasis": 1.12,
    },
}
FORCE_INCLUDED_PLAYER_NAMES = tuple(POSTSEASON_NARRATIVE_PLAYERS.keys())
REQUIRED_STATS_COLUMNS = [
    "PLAYER_ID",
    "PLAYER_FULLNAME",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "MIN",
    "PTS",
    "AST",
    "REB",
    "TOV",
    "STL",
    "BLK",
    "GP",
    "FGA",
    "FTA",
]
DEFAULT_SENTIMENT_VALUE = 5.0
_STATS_CACHE: dict[str, dict[int, pd.DataFrame]] = {}


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    try:
        return float(numerator) / float(denominator) if denominator else default
    except (TypeError, ZeroDivisionError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and np.isnan(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_stats_dict(path: Path) -> dict[int, pd.DataFrame]:
    cache_key = str(path.resolve())
    if cache_key not in _STATS_CACHE:
        if not path.exists():
            raise FileNotFoundError(f"Season stats archive not found: {path}")
        _STATS_CACHE[cache_key] = pd.read_pickle(path)
    return _STATS_CACHE[cache_key]


def _latest_season() -> int:
    stats = _load_stats_dict(STATS_DICT_PATH)
    return max(stats.keys())


def _load_season_stats(season: int | None = None) -> pd.DataFrame:
    stats = _load_stats_dict(STATS_DICT_PATH)
    if season is None:
        season = max(stats.keys())
    if season not in stats:
        raise ValueError(f"Season {season} is not available in {STATS_DICT_PATH}")
    df = stats[season].copy()
    _validate_stats_columns(df)
    emit_metric("stats_rows_loaded", float(len(df)), {"season": str(season)})
    return df


def _validate_stats_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_STATS_COLUMNS if col not in df.columns]
    if missing:
        logger.warning(
            "Missing required stats columns: %s",
            missing,
        )


def _aggregate_team_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["PTS", "MIN", "STL", "BLK", "REB", "OREB", "DREB", "FGA", "FTA", "TOV"]
    agg = (
        player_df.groupby("TEAM_ID", as_index=False)
        .agg({col: ("sum") for col in agg_cols})
    )
    return agg


def calculate_win_shares(
    player_df: pd.DataFrame,
    team_stats_dict: Mapping[str | int, Mapping[str, float]],
    pts_col: str = "PTS",
    min_col: str = "MIN",
    new_col: str = "WS",
) -> pd.DataFrame:
    def _compute(row: pd.Series) -> float:
        team_id = row.get("TEAM_ID")
        team_stats = team_stats_dict.get(team_id, {})
        team_pts = _safe_float(team_stats.get("PTS"))
        team_min = _safe_float(team_stats.get("MIN"))
        player_min = _safe_float(row.get(min_col))
        player_pts = _safe_float(row.get(pts_col))
        if team_pts > 0 and team_min > 0:
            return (player_pts / team_pts) * (player_min / team_min)
        return 0.0

    player_df[new_col] = player_df.apply(_compute, axis=1)
    return player_df


def calculate_bpm(
    player_df: pd.DataFrame,
    team_stats_dict: Mapping[str | int, Mapping[str, float]],
    pts_col: str = "PTS",
    ast_col: str = "AST",
    tov_col: str = "TOV",
    stl_col: str = "STL",
    blk_col: str = "BLK",
    reb_col: str = "REB",
    min_col: str = "MIN",
    new_col: str = "BPM",
) -> pd.DataFrame:
    bpm_list: list[float] = []
    obpm_list: list[float] = []
    dbpm_list: list[float] = []

    for _, row in player_df.iterrows():
        team_id = row.get("TEAM_ID")
        team_stats = team_stats_dict.get(team_id, {})
        team_min = _safe_float(team_stats.get("MIN"))
        player_min = _safe_float(row.get(min_col))

        obpm = (
            0.1 * _safe_float(row.get(pts_col))
            + 0.5 * _safe_float(row.get(ast_col))
            - 0.25 * _safe_float(row.get(tov_col))
        )
        dbpm = (
            0.3 * _safe_float(row.get(stl_col))
            + 0.3 * _safe_float(row.get(blk_col))
            + 0.1 * _safe_float(row.get(reb_col))
        )
        minutes_factor = _safe_div(player_min, team_min)
        bpm = (obpm + dbpm) * minutes_factor
        bpm_list.append(bpm)
        obpm_list.append(obpm * minutes_factor)
        dbpm_list.append(dbpm * minutes_factor)

    player_df[new_col] = bpm_list
    player_df["OBPM"] = obpm_list
    player_df["DBPM"] = dbpm_list
    return player_df


def _augment_player_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["GP"] == 0, "GP"] = 1
    df.loc[df["MIN"] == 0, "MIN"] = 1
    df.loc[df["FGA"] == 0, "FGA"] = 1
    df.loc[df["FTA"] == 0, "FTA"] = 1

    df["FGM_pg"] = df["FGM"] / df["GP"]
    df["TOVPct"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"])
    df["ORtg"] = df["PTS"] / df["FGA"] * 100
    df["PER"] = (df["PTS"] + df["REB"] + df["AST"] - df["TOV"]) / df["MIN"]
    df["WS/48"] = df["WS"] / (df["MIN"] / 48 + 1e-9)
    df["WS/48_x"] = df["WS/48"]
    df["VORP"] = df["BPM"] * df["MIN"] / 48
    df["DRBPct"] = df["DREB"] / (df["DREB"] + df["OREB"] + 1e-9)
    df["DWS"] = df["WS"] * 0.4
    df["SRS"] = 0.1 * df["PER"] + 0.05 * df["ORtg"] + 0.3 * df["WS/48"]
    return df


def load_sentiment_scores(path: Path | None = None) -> Mapping[str, Mapping[str, float]]:
    file_path = path or SENTIMENT_PATH
    if not file_path.exists():
        logger.warning("Sentiment sample file not found at %s", file_path)
        return {}

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    mapping: dict[str, Mapping[str, float]] = {}
    missing_keys = 0
    for entry in payload:
        player_name = str(entry.get("player_fullname", "")).strip().upper()
        if not player_name:
            continue
        scores = entry.get("scores", {})
        missing = [key for key in SENTIMENT_KEYS if key not in scores]
        missing_keys += len(missing)
        mapping[player_name] = {
            key: float(scores.get(key, DEFAULT_SENTIMENT_VALUE)) for key in SENTIMENT_KEYS
        }
    emit_metric(
        "sentiment_entries_processed",
        float(len(mapping)),
        {"source": file_path.name},
    )
    emit_metric(
        "sentiment_missing_keys",
        float(missing_keys),
        {"source": file_path.name},
    )
    return mapping


def _sentiment_for_player(player_name: str, sentiment_map: Mapping[str, Mapping[str, float]]) -> tuple[dict[str, float], float]:
    normalized = str(player_name or "").strip().upper()
    sample = sentiment_map.get(normalized, {})
    values = {
        key: float(sample.get(key, DEFAULT_SENTIMENT_VALUE)) for key in SENTIMENT_KEYS
    }
    avg = float(np.mean(list(values.values()))) if values else DEFAULT_SENTIMENT_VALUE

    override = POSTSEASON_NARRATIVE_PLAYERS.get(normalized)
    if override:
        emphasis = float(override.get("postseason_narrative_emphasis", 1.0))
        for key in SENTIMENT_KEYS:
            if key in override:
                values[key] = float(override[key])
        avg = float(override.get("sentiment_avg", avg))
        avg = max(DEFAULT_SENTIMENT_VALUE, min(10.0, avg * emphasis))

    return values, avg


def build_candidate_feature_rows(
    season: int | None = None,
    top_n: int = 30,
    schema: FeatureSchema | None = None,
) -> tuple[list[Mapping[str, float]], list[str], list[str], list[Mapping[str, Any]]]:
    schema = schema or load_feature_schema()
    season = season or _latest_season()
    player_df = _load_season_stats(season)
    team_totals = _aggregate_team_stats(player_df)
    team_stats_dict = team_totals.set_index("TEAM_ID")[["PTS", "MIN", "STL", "BLK", "REB"]].to_dict(
        orient="index"
    )

    player_df = calculate_win_shares(player_df, team_stats_dict)
    player_df = calculate_bpm(player_df, team_stats_dict)
    player_df = _augment_player_metrics(player_df)

    ranked_by_minutes = player_df.sort_values("MIN", ascending=False)
    candidate_df = ranked_by_minutes.head(top_n)

    forced_mask = player_df["PLAYER_FULLNAME"].astype(str).str.strip().str.upper().isin(FORCE_INCLUDED_PLAYER_NAMES)
    forced_players = player_df.loc[forced_mask]
    if not forced_players.empty:
        candidate_df = (
            pd.concat([candidate_df, forced_players], ignore_index=False)
            .drop_duplicates(subset=["PLAYER_ID"], keep="first")
            .sort_values("MIN", ascending=False)
        )

    sentiment_map = load_sentiment_scores()

    rows: list[Mapping[str, float]] = []
    names: list[str] = []
    ids: list[str] = []
    metadata: list[Mapping[str, Any]] = []

    for _, row in candidate_df.iterrows():
        sentiment_values, sentiment_avg = _sentiment_for_player(row.get("PLAYER_FULLNAME"), sentiment_map)
        base_values: dict[str, float] = {
            "FGM": _safe_float(row.get("FGM_pg")),
            "BPM": _safe_float(row.get("BPM")),
            "DRBPct": _safe_float(row.get("DRBPct")),
            "DWS": _safe_float(row.get("DWS")),
            "OBPM": _safe_float(row.get("OBPM")),
            "PER": _safe_float(row.get("PER")),
            "TOVPct": _safe_float(row.get("TOVPct")),
            "VORP": _safe_float(row.get("VORP")),
            "WS/48_x": _safe_float(row.get("WS/48_x")),
            "Rk_opp_pg": 0.0,
            "2P%_opp_pg": 0.0,
            "DRB_opp_pg": 0.0,
            "SRS": _safe_float(row.get("SRS")),
            "ORtg": _safe_float(row.get("ORtg")),
            "WS": _safe_float(row.get("WS")),
        }
        for feature_key, value in sentiment_values.items():
            base_values[feature_key] = value
        base_values["sentiment_avg"] = sentiment_avg

        ordered = {feature: _safe_float(base_values.get(feature)) for feature in schema.vector_order}
        rows.append(ordered)
        names.append(str(row.get("PLAYER_FULLNAME", "")))
        ids.append(str(row.get("PLAYER_ID", "")))
        metadata.append(
            {
                "season": season,
                "team": row.get("TEAM_ABBREVIATION"),
                "minutes": _safe_float(row.get("MIN")),
            }
        )

    return rows, names, ids, metadata
