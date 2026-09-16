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
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from dotenv import load_dotenv

# Estimated-service-time memo, keyed by (season, cache mtime). See
# query_estimated_service_time — the on-disk cache was re-read once per team.
_EST_MEMO: dict[tuple, dict[int, float]] = {}
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")

# Two keys, deliberately. SUPABASE_SERVICE_KEY bypasses Row Level Security on
# every table and belongs only where data is WRITTEN — seeding, uploads, the
# refresh pipeline. SUPABASE_ANON_KEY is the one designed to be handed out; on
# the web tier it reads the portraits bucket and can do nothing else.
#
# The service key used to be the only option, so render.yaml handed an
# RLS-bypassing admin credential to a public, unauthenticated dashboard whose
# entire use of it was downloading one JSON file. Anything that got code
# execution there got the database.
#
# Service key wins when both are present, so a build machine with a full .env
# keeps working unchanged.
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY    = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_KEY = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY

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
                "SUPABASE_URL plus one of SUPABASE_SERVICE_KEY (write) or "
                "SUPABASE_ANON_KEY (read-only) must be set in .env"
            )
        _CLIENT = create_client(SUPABASE_URL, SUPABASE_KEY)
        log.info("Supabase client connected to %s using the %s key",
                 SUPABASE_URL, "service" if SUPABASE_SERVICE_KEY else "anon")
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


# ---------------------------------------------------------------------------
# Estimated Service Time
# ---------------------------------------------------------------------------

# CBA: 172 days of active-roster service = 1 service year
_SERVICE_DAYS_PER_YEAR = 172

# Approximate historical season dates for pre-2015 seasons (no Statcast).
# Used only to compute pre-Statcast era calendar days for veterans.
_PRE_2015_SEASON_DATES: dict[int, tuple[str, str]] = {
    2000: ("2000-04-02", "2000-10-01"), 2001: ("2001-04-01", "2001-10-07"),
    2002: ("2002-04-01", "2002-09-29"), 2003: ("2003-03-30", "2003-09-28"),
    2004: ("2004-04-04", "2004-10-03"), 2005: ("2005-04-03", "2005-10-02"),
    2006: ("2006-04-02", "2006-10-01"), 2007: ("2007-04-01", "2007-09-30"),
    2008: ("2008-03-31", "2008-09-28"), 2009: ("2009-04-05", "2009-10-04"),
    2010: ("2010-04-04", "2010-10-03"), 2011: ("2011-03-31", "2011-09-28"),
    2012: ("2012-03-28", "2012-10-03"), 2013: ("2013-03-31", "2013-09-29"),
    2014: ("2014-03-22", "2014-09-28"),
}


def _fetch_debut_dates(player_ids: list[int]) -> dict[int, str]:
    """
    Fetch exact mlbDebutDate strings ('YYYY-MM-DD') for a list of MLBAM IDs
    from the MLB Stats API.  Batched in groups of 200.  Results cached to
    data/processed/debut_dates.csv (no TTL — debut dates never change).

    Returns {key_mlbam: debut_date_str}.
    """
    import requests
    import time

    cache_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "debut_dates.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    known: dict[int, str] = {}
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            known = dict(zip(df["key_mlbam"].astype(int), df["debut_date"].astype(str)))
        except Exception:
            pass

    missing = [p for p in player_ids if p not in known]
    if not missing:
        return {p: known[p] for p in player_ids if p in known}

    BATCH = 200
    fetched: dict[int, str] = {}
    for i in range(0, len(missing), BATCH):
        batch = missing[i: i + BATCH]
        ids_str = ",".join(str(p) for p in batch)
        try:
            r = requests.get(
                f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
                "&fields=people,id,mlbDebutDate",
                timeout=15,
            )
            for person in r.json().get("people", []):
                pid  = person.get("id")
                date = person.get("mlbDebutDate")
                if pid and date:
                    fetched[int(pid)] = date
        except Exception as exc:
            log.warning("MLB Stats API debut date fetch failed (batch %d): %s", i // BATCH, exc)
        time.sleep(0.2)   # polite rate limit

    # Merge and persist
    known.update(fetched)
    rows = [{"key_mlbam": k, "debut_date": v} for k, v in known.items()]
    pd.DataFrame(rows).to_csv(cache_path, index=False)
    log.info("Debut dates: %d fetched, %d total cached", len(fetched), len(known))

    return {p: known[p] for p in player_ids if p in known}


def query_estimated_service_time(
    player_ids: list[int],
    through_season: int,
) -> dict[int, float]:
    """
    Estimated Service Time (EST) for each player through the end of through_season.

    Method:
      For each player × season from their MLB debut to through_season:
        active_days  = calendar days in that season the player was eligible
                       (capped at season start if they debuted mid-season)
        il_days      = days spent on any IL during that season
        service_days = max(0, min(active_days - il_days, 172))
      EST = sum(service_days) / 172   → expressed as years.days float
            e.g. 5 years 86 days = (5 × 172 + 86) / 172 = 5.5

    CBA reference: 172 active-roster days = 1 service year.
    Thresholds: < 3.0 pre-arb · 3.0–5.999 arb-eligible · ≥ 6.0 free agent.

    Pre-2015 seasons use calendar days with no IL deduction (no cached data).
    2015+ seasons deduct actual IL days from pull_il_data.

    Results are cached to data/processed/est_service_time_{through_season}.csv.
    """
    from datetime import date, datetime
    from ingest import SEASON_DATES, pull_il_data

    # In-process memo of the on-disk cache. A portrait build calls this once
    # per team, so a 30-team season re-read and re-parsed the same CSV thirty
    # times — 0.41s each even on a complete hit. Keyed by (season, mtime) so
    # an external rewrite is still picked up.
    global _EST_MEMO
    cache_path_probe = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed"
        / f"est_service_time_{through_season}.csv"
    )
    try:
        _memo_key = (through_season, cache_path_probe.stat().st_mtime)
    except OSError:
        _memo_key = None
    if _memo_key is not None:
        memo = _EST_MEMO.get(_memo_key)
        if memo is not None and all(p in memo for p in player_ids):
            return {p: memo[p] for p in player_ids}

    cache_path = (
        Path(__file__).resolve().parents[1]
        / "data" / "processed"
        / f"est_service_time_{through_season}.csv"
    )
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            cached = dict(zip(df["key_mlbam"].astype(int), df["est_service_time"].astype(float)))
            # Return only the requested IDs that are cached; compute missing below
            missing_ids = [p for p in player_ids if p not in cached]
            if not missing_ids:
                if _memo_key is not None:
                    _EST_MEMO[_memo_key] = cached
                return {p: cached[p] for p in player_ids if p in cached}
        except Exception:
            cached = {}
            missing_ids = player_ids
    else:
        cached = {}
        missing_ids = player_ids

    # ── Fetch exact debut dates ───────────────────────────────────────────────
    debut_dates = _fetch_debut_dates(missing_ids)

    # ── Build IL lookup: {player_id: {season: [(placed_date, activated_date)]}} ─
    all_seasons = list(dict.fromkeys(
        SEASON_DATES.keys()
    ))  # 2015 onward
    statcast_seasons = [s for s in all_seasons if s <= through_season]

    il_by_player: dict[int, dict[int, list[tuple[date, date]]]] = {}
    for season in statcast_seasons:
        try:
            il_df = pull_il_data(season)
            if il_df.empty:
                continue
            il_df["key_mlbam"] = pd.to_numeric(il_df["key_mlbam"], errors="coerce")
            il_df["date"]      = pd.to_datetime(il_df["date"], errors="coerce")
            il_df = il_df.dropna(subset=["key_mlbam", "date"])

            # Match placed/activated pairs per player
            for pid, grp in il_df.groupby("key_mlbam"):
                pid = int(pid)
                placed_dates:    list[date] = []
                activated_dates: list[date] = []
                for _, row in grp.sort_values("date").iterrows():
                    if row["transaction"] == "placed":
                        placed_dates.append(row["date"].date())
                    elif row["transaction"] == "activated":
                        activated_dates.append(row["date"].date())

                # Pair up: each placed date with the next activated date
                pairs: list[tuple[date, date]] = []
                act_q = list(activated_dates)
                _, s_end_str = SEASON_DATES.get(season, ("", ""))
                s_end = datetime.strptime(s_end_str, "%Y-%m-%d").date() if s_end_str else date(season, 10, 1)

                for p_date in placed_dates:
                    # First activation after this placement
                    matched = next((a for a in act_q if a >= p_date), None)
                    if matched:
                        act_q.remove(matched)
                        pairs.append((p_date, matched))
                    else:
                        # Still on IL at season end
                        pairs.append((p_date, s_end))

                # Unmatched activations: on IL from season start
                s_start_str, _ = SEASON_DATES.get(season, ("", ""))
                s_start = datetime.strptime(s_start_str, "%Y-%m-%d").date() if s_start_str else date(season, 4, 1)
                for a_date in act_q:
                    pairs.append((s_start, a_date))

                il_by_player.setdefault(pid, {})[season] = pairs
        except Exception as exc:
            log.debug("IL data for %d failed: %s", season, exc)

    # ── Compute EST per player ────────────────────────────────────────────────
    all_season_dates = {**_PRE_2015_SEASON_DATES, **SEASON_DATES}
    results: dict[int, float] = {}

    for pid in missing_ids:
        debut_str = debut_dates.get(pid)
        if not debut_str:
            results[pid] = float("nan")
            continue

        try:
            debut_date_obj = datetime.strptime(debut_str, "%Y-%m-%d").date()
        except Exception:
            results[pid] = float("nan")
            continue

        debut_year = debut_date_obj.year
        total_service_days = 0

        for season in range(debut_year, through_season + 1):
            dates = all_season_dates.get(season)
            if not dates:
                continue
            s_start = datetime.strptime(dates[0], "%Y-%m-%d").date()
            s_end   = datetime.strptime(dates[1], "%Y-%m-%d").date()

            # Player's effective start: max of season start and debut date
            eff_start = max(s_start, debut_date_obj)
            if eff_start > s_end:
                continue   # debuted after season ended

            calendar_days = (s_end - eff_start).days

            # 2020 special case: CBA/MLBPA agreement granted all players 1.0 full
            # service year for the 60-game COVID season, regardless of games played
            # or IL time.  Only applies to players who were on an MLB roster in 2020.
            if season == 2020 and debut_date_obj <= s_end:
                total_service_days += _SERVICE_DAYS_PER_YEAR
                continue

            # IL deduction (only for 2015+ seasons where we have data)
            il_days = 0
            if season >= 2015:
                for placed, activated in il_by_player.get(pid, {}).get(season, []):
                    # Clip IL stint to the player's active window this season
                    il_start = max(placed,    eff_start)
                    il_end   = min(activated, s_end)
                    if il_end > il_start:
                        il_days += (il_end - il_start).days

            service_days = max(0, min(calendar_days - il_days, _SERVICE_DAYS_PER_YEAR))
            total_service_days += service_days

        est = total_service_days / _SERVICE_DAYS_PER_YEAR
        results[pid] = round(est, 3)

    # ── Persist and return ───────────────────────────────────────────────────
    merged = {**cached, **results}
    rows = [{"key_mlbam": k, "est_service_time": v}
            for k, v in merged.items() if not (isinstance(v, float) and pd.isna(v))]
    pd.DataFrame(rows).to_csv(cache_path, index=False)
    log.info("Estimated Service Time computed: %d players through %d", len(results), through_season)

    return {p: results.get(p, cached.get(p, float("nan"))) for p in player_ids}
