"""
database.py — Supabase connection, upsert, and query helpers.

Tables (create via Supabase SQL editor — see schema.sql):
  player_batting    raw BRef+Savant batting stats per player per season
  player_pitching   raw BRef+Savant pitching stats per player per season
  team_philosophy   pre-computed philosophy scores per team per season
  park_factors      park factor results per team per season
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

_CLIENT: Optional[Client] = None

MAX_ROWS = 2000  # safe ceiling above any single season's player pool


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_client() -> Client:
    global _CLIENT
    if _CLIENT is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise EnvironmentError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env"
            )
        _CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client connected to %s", SUPABASE_URL)
    return _CLIENT


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _upsert(table: str, records: list[dict]) -> None:
    if not records:
        return
    client = get_client()
    # Batch in chunks of 500 to stay within Supabase request limits
    for i in range(0, len(records), 500):
        chunk = records[i : i + 500]
        client.table(table).upsert(chunk).execute()
    log.info("Upserted %d rows into %s", len(records), table)


def upsert_batting(df: pd.DataFrame) -> None:
    cols = [
        "key_mlbam", "season", "name", "age", "pa",
        "avg", "obp", "slg", "iso", "k_rate", "bb_rate", "sb", "cs",
        "xwoba", "xba", "hard_hit_pct", "barrel_pct",
        "whiff_pct", "chase_pct", "sprint_speed", "exit_velocity",
        "join_source",
    ]
    present = [c for c in cols if c in df.columns]
    records = df[present].where(df[present].notna(), other=None).to_dict("records")
    _upsert("player_batting", records)


def upsert_pitching(df: pd.DataFrame) -> None:
    cols = [
        "key_mlbam", "season", "name", "age",
        "gs", "bf", "k_rate", "bb_rate",
        "xwoba_allowed", "hard_hit_pct", "barrel_pct",
        "whiff_pct", "fb_velocity", "join_source",
    ]
    present = [c for c in cols if c in df.columns]
    records = df[present].where(df[present].notna(), other=None).to_dict("records")
    _upsert("player_pitching", records)


def upsert_philosophy(team: str, season: int, scores: dict, data_coverage: float) -> None:
    row: dict = {"team": team, "season": season, "data_coverage": data_coverage}
    for code, data in scores.items():
        key = code.lower()
        row[f"{key}_score"]    = data.get("score")
        row[f"{key}_coverage"] = data.get("coverage")
    _upsert("team_philosophy", [row])


def upsert_park_factors(team: str, season: int, pitcher_friendly: float) -> None:
    _upsert("park_factors", [{"team": team, "season": season,
                               "pitcher_friendly": pitcher_friendly}])


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def query_batting(season: int, team: Optional[str] = None) -> pd.DataFrame:
    client = get_client()
    q = client.table("player_batting").select("*").eq("season", season).limit(MAX_ROWS)
    if team:
        q = q.eq("team", team)
    result = q.execute()
    return pd.DataFrame(result.data)


def query_pitching(season: int, team: Optional[str] = None) -> pd.DataFrame:
    client = get_client()
    q = client.table("player_pitching").select("*").eq("season", season).limit(MAX_ROWS)
    if team:
        q = q.eq("team", team)
    result = q.execute()
    return pd.DataFrame(result.data)


def query_philosophy(team: str, season: int) -> Optional[dict]:
    client = get_client()
    result = (
        client.table("team_philosophy")
        .select("*")
        .eq("team", team)
        .eq("season", season)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def season_batting_seeded(season: int) -> bool:
    client = get_client()
    result = (
        client.table("player_batting")
        .select("key_mlbam")
        .eq("season", season)
        .limit(1)
        .execute()
    )
    return len(result.data) > 0


def season_pitching_seeded(season: int) -> bool:
    client = get_client()
    result = (
        client.table("player_pitching")
        .select("key_mlbam")
        .eq("season", season)
        .limit(1)
        .execute()
    )
    return len(result.data) > 0
