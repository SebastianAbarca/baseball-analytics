"""
store.py — everything the dashboard reads, from one place.

The dashboard reads Postgres through Supabase's REST API with the ANON key and
nothing else: no portrait files, no league files, no parquet, no Storage
bucket, no building. Row Level Security limits that key to SELECT on the
serving tables (schema_serving.sql), so the dashboard is read-only because of
what its credential can do, not because of a mode it was started in.
scripts/publish.py is the only writer.

Caching is in-process and short-lived. A portrait is ~300 KB and some two
dozen callbacks resolve it per team switch, so it is held in a small LRU; an
entry older than REVALIDATE_S costs one tiny `built_at` query before reuse,
so a publish shows up without a restart and without refetching unchanged rows.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules"))

from database import get_read_client                      # noqa: E402
from team_portrait import PORTRAIT_SCHEMA_VERSION, gate_fingerprint  # noqa: E402

log = logging.getLogger(__name__)

REVALIDATE_S = 600          # reuse a cached row this long before re-checking
PORTRAIT_LRU_MAX = 8        # a few team-seasons; each is ~1 MB parsed
PAGE = 1000                 # PostgREST's default max rows per response

_lock = threading.Lock()
_flights: dict[str, threading.Lock] = {}


def _flight(key: str) -> threading.Lock:
    """
    One lock per cache key, so concurrent misses for the same row make ONE
    request. A page load fires a dozen callbacks at once; without this every
    one of them missed the empty cache together and fetched the same row.
    """
    with _lock:
        return _flights.setdefault(key, threading.Lock())


class _Cache:
    """LRU of key -> (checked_at, built_at, value)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.d: "OrderedDict[str, tuple[float, str, object]]" = OrderedDict()

    def get(self, key):
        with _lock:
            hit = self.d.get(key)
            if hit:
                self.d.move_to_end(key)
            return hit

    def put(self, key, built_at, value):
        with _lock:
            self.d[key] = (time.time(), built_at, value)
            self.d.move_to_end(key)
            while len(self.d) > self.maxsize:
                self.d.popitem(last=False)

    def touch(self, key):
        with _lock:
            if key in self.d:
                _, b, v = self.d[key]
                self.d[key] = (time.time(), b, v)


_portraits = _Cache(PORTRAIT_LRU_MAX)
_leagues = _Cache(48)       # 12 seasons x 3 parts, with room


def _db():
    return get_read_client()


# ---------------------------------------------------------------------------
# Portraits
# ---------------------------------------------------------------------------

# Why a portrait could not be served, for the status banner.
FOUND, MISSING, OUTDATED = "found", "missing", "outdated"


def _fresh(cache: _Cache, key: str, table: str, match: dict):
    """
    Cached value if still valid: within REVALIDATE_S, or unchanged since by
    `built_at`. Returns (hit, value).
    """
    hit = cache.get(key)
    if not hit:
        return False, None
    checked_at, built_at, value = hit
    if time.time() - checked_at < REVALIDATE_S:
        return True, value
    try:
        q = _db().table(table).select("built_at")
        for col, val in match.items():
            q = q.eq(col, val)
        rows = q.limit(1).execute().data
    except Exception as exc:
        # Can't reach the database to revalidate: keep serving what we have
        # rather than fail a page that was working a moment ago.
        log.warning("revalidate %s failed (%s) — serving cached copy", key, exc)
        cache.touch(key)
        return True, value
    if rows and rows[0]["built_at"] == built_at:
        cache.touch(key)
        return True, value
    return False, None


def get_portrait(team: str, season: int) -> tuple[Optional[dict], str]:
    """
    (portrait, status). status is FOUND, MISSING (no row for this team-season)
    or OUTDATED (a row exists, cut by different tag logic or schema than this
    dashboard's code — it will be replaced by the next publish).
    """
    key = f"{team}|{int(season)}"
    with _flight("portrait:" + key):
        return _get_portrait(key, team, int(season))


def _get_portrait(key: str, team: str, season: int) -> tuple[Optional[dict], str]:
    ok, value = _fresh(_portraits, key, "portraits", {"team": team, "season": season})
    if ok:
        return value, FOUND

    rows = (_db().table("portraits")
            .select("data, built_at, schema_version, gate_fingerprint")
            .eq("team", team).eq("season", season)
            .limit(1).execute().data)
    if not rows:
        return None, MISSING
    row = rows[0]
    if (row["schema_version"] != PORTRAIT_SCHEMA_VERSION
            or (row.get("gate_fingerprint") or "") != gate_fingerprint()):
        log.warning("portrait %s is outdated (schema %s, gates %s; this code "
                    "wants %s, %s)", key, row["schema_version"],
                    row.get("gate_fingerprint"), PORTRAIT_SCHEMA_VERSION,
                    gate_fingerprint())
        return None, OUTDATED
    _portraits.put(key, row["built_at"], row["data"])
    return row["data"], FOUND


# ---------------------------------------------------------------------------
# League seasons
# ---------------------------------------------------------------------------

LEAGUE_PARTS = ("identity", "tunnel_average", "pitch_mix")


def get_league(season, part: str) -> dict:
    """
    One part of a season's league_seasons row — identity, tunnel_average or
    pitch_mix — or {} if unpublished. Fetched per part: the year-over-year
    view needs identity for all twelve seasons and nothing else, and the
    pitch-mix part alone is a few hundred KB a season.
    """
    assert part in LEAGUE_PARTS, part
    try:
        season = int(season)
    except (TypeError, ValueError):
        return {}
    key = f"{season}:{part}"
    with _flight("league:" + key):
        ok, value = _fresh(_leagues, key, "league_seasons", {"season": season})
        if ok:
            return value
        try:
            rows = (_db().table("league_seasons").select(f"{part}, built_at")
                    .eq("season", season).limit(1).execute().data)
        except Exception as exc:
            log.warning("league_seasons %s/%s read failed: %s", season, part, exc)
            return {}
        if not rows:
            return {}
        value = rows[0].get(part) or {}
        _leagues.put(key, rows[0]["built_at"], value)
        return value


# ---------------------------------------------------------------------------
# Scout
# ---------------------------------------------------------------------------

_TAGS: tuple[float, dict] | None = None


def scout_tags() -> dict:
    """Tag catalogue, tag -> {kind, family, side, n, dir}. ~94 rows, cached."""
    global _TAGS
    if _TAGS and time.time() - _TAGS[0] < REVALIDATE_S:
        return _TAGS[1]
    try:
        rows = _db().table("scout_tags").select("*").execute().data
        cat = {r["tag"]: {k: r[k] for k in ("kind", "family", "side", "n", "dir")}
               for r in rows}
        _TAGS = (time.time(), cat)
    except Exception as exc:
        log.warning("scout_tags read failed: %s", exc)
        if _TAGS:
            return _TAGS[1]
        return {}
    return cat


def _rpc_all(fn: str, params: dict, cap: int) -> list[dict]:
    """Call a set-returning function, paging past the per-response row cap."""
    out: list[dict] = []
    while len(out) < cap:
        take = min(PAGE, cap - len(out))
        page = _db().rpc(fn, {**params, "p_limit": take,
                              "p_offset": len(out)}).execute().data or []
        out.extend(page)
        if len(page) < take:
            break
    return out


def scout_search(cap: int = 2000, **params) -> tuple[list[dict], int]:
    """(rows, total). Rows are slim; total counts every match, not just cap."""
    rows = _rpc_all("scout_search", params, cap)
    total = rows[0]["total"] if rows else 0
    return rows, total


def scout_name_search(query: str, side: str, limit: int = 40) -> list[dict]:
    return _db().rpc("scout_name_search", {"p_query": query, "p_side": side,
                                           "p_limit": limit}).execute().data or []


def scout_rows(keys: list[str]) -> list[dict]:
    if not keys:
        return []
    return _db().rpc("scout_rows", {"p_keys": list(keys)}).execute().data or []


def scout_comps(key: str, limit: int, min_vol: int,
                require: Optional[list[str]], same_season: bool) -> list[dict]:
    return _db().rpc("scout_comps", {
        "p_key": key, "p_limit": limit, "p_min_vol": min_vol,
        "p_require": require or None, "p_same_season": same_season,
    }).execute().data or []
