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
    Return {key_mlbam: debut_season} — the player's actual MLB debut year.

    Primary source: Chadwick Bureau crosswalk (data/reference/chadwick_crosswalk.csv)
    field `mlb_played_first`.  This gives real debut years (Freeman=2010,
    Altuve=2011, etc.) rather than the first year in our DB (which was
    capped at 2021 for all veterans).

    Fallback: earliest season in player_batting for players not found in
    Chadwick (e.g. very recent players whose crosswalk entry is missing).

    Result is cached to data/processed/debut_seasons.csv and refreshed
    whenever this function is called (cache TTL 24 h).

    ── KNOWN LIMITATIONS (Chadwick vs true MLB service time) ────────────────

    This function returns *calendar debut year*, not *MLB service time*.
    The two diverge in meaningful ways for roster construction analysis:

    1. Service time manipulation: Teams routinely delay prospect call-ups
       by 10-15 days to prevent accruing a full year of service, saving a
       year of team control. Chadwick shows the debut year correctly but
       `season − debut_year` overstates service by 1 year for these players.
       (e.g. Kris Bryant 2015, many top prospects since)

    2. IL time doesn't count: A player on 60-day IL accrues no service time.
       Career IL-heavy players (Tommy John surgeries, etc.) have meaningfully
       less service time than calendar years suggest.

    3. Pre-arb / arb / free agent thresholds require true service time days
       (172 days = 1 service year; 3.000 = arb eligible; 6.000 = free agent).
       Calendar years cannot correctly compute these contractual thresholds.

    Impact on current platform:
      - C3 (Youth & Development) and C4 (Veteran Experience) are
        directionally correct but not contractually precise.
      - A future "Roster Control Profile" (pre-arb/arb/FA split per team)
        requires a real service time CSV — this function cannot support it.

    Upgrade path: Replace with a CSV of (key_mlbam, service_time_days, season)
    sourced from Baseball Reference player pages or MLBPA data (~10k rows
    across 12 seasons). Wire into a `query_service_time()` function alongside
    this one and update C3/C4 to use it when available.
    """
    from pathlib import Path
    import time

    cache = Path(__file__).resolve().parents[1] / "data" / "processed" / "debut_seasons.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and (time.time() - cache.stat().st_mtime) < 86400:
        df = pd.read_csv(cache)
        return dict(zip(df["key_mlbam"].astype(int), df["debut_season"].astype(int)))

    # ── Primary: Chadwick mlb_played_first ───────────────────────────────────
    cw_path = Path(__file__).resolve().parents[1] / "data" / "reference" / "chadwick_crosswalk.csv"
    chadwick_map: dict[int, int] = {}
    if cw_path.exists():
        try:
            cw = pd.read_csv(cw_path, low_memory=False,
                             usecols=["key_mlbam", "mlb_played_first"])
            cw["key_mlbam"]        = pd.to_numeric(cw["key_mlbam"],        errors="coerce")
            cw["mlb_played_first"] = pd.to_numeric(cw["mlb_played_first"], errors="coerce")
            cw = cw.dropna(subset=["key_mlbam", "mlb_played_first"])
            chadwick_map = dict(zip(cw["key_mlbam"].astype(int),
                                    cw["mlb_played_first"].astype(int)))
            log.info("Chadwick debut seasons: %d players", len(chadwick_map))
        except Exception as exc:
            log.warning("Chadwick debut load failed: %s", exc)

    # ── Fallback: DB min(season) for any player not in Chadwick ─────────────
    client  = get_client()
    page, page_sz, frames = 0, 2000, []
    while True:
        result = (client.table("player_batting")
                  .select("key_mlbam,season")
                  .range(page * page_sz, (page + 1) * page_sz - 1)
                  .execute())
        if not result.data:
            break
        frames.append(pd.DataFrame(result.data))
        if len(result.data) < page_sz:
            break
        page += 1

    if frames:
        db = pd.concat(frames, ignore_index=True)
        db["key_mlbam"] = pd.to_numeric(db["key_mlbam"], errors="coerce")
        db["season"]    = pd.to_numeric(db["season"],    errors="coerce")
        db = db.dropna()
        db_debut = db.groupby("key_mlbam")["season"].min().to_dict()
        # Merge: Chadwick takes precedence; DB fills gaps
        merged = {**{int(k): int(v) for k, v in db_debut.items()}, **chadwick_map}
    else:
        merged = chadwick_map

    debut = pd.DataFrame(list(merged.items()), columns=["key_mlbam", "debut_season"])
    debut.to_csv(cache, index=False)
    log.info("query_debut_seasons: %d players (Chadwick=%d, DB-fallback=%d)",
             len(debut), len(chadwick_map),
             len(merged) - len(chadwick_map))
    return dict(zip(debut["key_mlbam"].astype(int), debut["debut_season"].astype(int)))
