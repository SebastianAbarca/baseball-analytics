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

def _clean(record: dict) -> dict:
    """Replace NaN/inf with None so records are JSON-serializable."""
    import math
    return {
        k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
        for k, v in record.items()
    }


def _upsert(table: str, records: list[dict]) -> None:
    if not records:
        return
    client = get_client()
    clean_records = [_clean(r) for r in records]
    # Batch in chunks of 500 to stay within Supabase request limits
    for i in range(0, len(clean_records), 500):
        chunk = clean_records[i : i + 500]
        client.table(table).upsert(chunk).execute()
    log.info("Upserted %d rows into %s", len(records), table)


def upsert_batting(df: pd.DataFrame) -> None:
    cols = [
        "key_mlbam", "season", "name", "age", "pa",
        "avg", "obp", "slg", "iso", "k_rate", "bb_rate", "sb", "cs",
        "xwoba", "xba", "hard_hit_pct", "barrel_pct",
        "whiff_pct", "chase_pct", "sprint_speed", "exit_velocity",
        "join_source",
        "gb_pct", "fps_pct", "pitches_per_pa", "war",
        "zone_swing_pct", "contact_pct",
    ]
    present = [c for c in cols if c in df.columns]
    records = df[present].where(df[present].notna(), other=None).to_dict("records")
    _upsert("player_batting", records)


def upsert_pitching(df: pd.DataFrame) -> None:
    cols = [
        "key_mlbam", "season", "name", "age",
        "g", "gs", "bf", "k_rate", "bb_rate",
        "xwoba_allowed", "hard_hit_pct", "barrel_pct",
        "whiff_pct", "fb_velocity", "join_source",
        "gb_pct", "zone_pct", "csw_pct", "pitches_per_bf", "war", "fip",
        "avg_spin_rate", "arsenal_diversity",
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


def query_seeded_seasons() -> list[int]:
    """Return sorted list of seasons that have data in player_batting."""
    client = get_client()
    result = client.table("player_batting").select("season").execute()
    df = pd.DataFrame(result.data)
    if df.empty:
        return []
    return sorted(df["season"].dropna().unique().astype(int).tolist())


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


def query_debut_seasons() -> dict[int, int]:
    """
    Return {key_mlbam: debut_season} — the earliest season each player
    appears in player_batting.  Used to compute MLB tenure without
    needing service-time data.

    Fetches in pages of 2000 rows to handle the full multi-season table.
    Result is cached to data/processed/debut_seasons.csv and refreshed
    whenever this function is called with force=False (uses cache if it
    exists and is younger than 24 h).
    """
    from pathlib import Path
    import time

    cache = Path(__file__).resolve().parents[1] / "data" / "processed" / "debut_seasons.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)

    # Use cache if fresh (< 24 h)
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        df = pd.read_csv(cache)
        return dict(zip(df["key_mlbam"].astype(int), df["debut_season"].astype(int)))

    client  = get_client()
    page    = 0
    page_sz = 2000
    frames  = []
    while True:
        result = (
            client.table("player_batting")
            .select("key_mlbam,season")
            .range(page * page_sz, (page + 1) * page_sz - 1)
            .execute()
        )
        if not result.data:
            break
        frames.append(pd.DataFrame(result.data))
        if len(result.data) < page_sz:
            break
        page += 1

    if not frames:
        return {}

    df = pd.concat(frames, ignore_index=True)
    df["key_mlbam"] = pd.to_numeric(df["key_mlbam"], errors="coerce")
    df["season"]    = pd.to_numeric(df["season"],    errors="coerce")
    df = df.dropna()
    debut = df.groupby("key_mlbam")["season"].min().reset_index()
    debut.columns = ["key_mlbam", "debut_season"]
    debut.to_csv(cache, index=False)
    log.info("query_debut_seasons: %d unique players cached", len(debut))
    return dict(zip(debut["key_mlbam"].astype(int), debut["debut_season"].astype(int)))
