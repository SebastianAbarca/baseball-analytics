#!/usr/bin/env python3
"""
pull_rosters.py — every MLB team's 40-man roster on every day of the regular
season, from MLB's Stats API, for the service-time ledger.

Each row is (season, date, team_id, player_id, status): the status code MLB
records for that player that day — A (active), D10/D15/D60 (MLB injured list),
RM (in the minors), and so on. The ledger decides which statuses earn service;
this script only records them, raw.

MLB's historical rosters are complete from 2009 (24-25 active a club); before
that they thin out to a handful of names, so 2009 is the first season pulled.

About 186 days x 30 clubs per season, one request each. Saved per season to
data/raw/rosters/rosters_{season}.parquet, and resumable: a season whose file
exists is skipped, and a run that dies keeps the team-days it finished in a
partial file it picks up from.

    python scripts/pull_rosters.py                  # 2009 through last season
    python scripts/pull_rosters.py --seasons 2024-2026
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("pull_rosters")
API = "https://statsapi.mlb.com/api/v1"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "rosters"
FIRST_SEASON = 2009
RATE = 8.0          # requests per second, all threads together
THREADS = 4

_session = requests.Session()
_next_slot = [0.0]
_slot_lock = threading.Lock()


def _get(path: str, **params) -> dict:
    """GET with a shared rate limit and retries."""
    for attempt in range(5):
        with _slot_lock:
            now = time.monotonic()
            wait = _next_slot[0] - now
            _next_slot[0] = max(_next_slot[0], now) + 1.0 / RATE
        if wait > 0:
            time.sleep(wait)
        try:
            r = _session.get(f"{API}{path}", params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            log.warning("%s %s -> HTTP %d", path, params, r.status_code)
        except requests.RequestException as exc:
            log.warning("%s %s failed: %s", path, params, exc)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{path} {params}: gave up after 5 attempts")


def season_window(season: int) -> tuple[dt.date, dt.date]:
    """Opening Day and the last regular-season day, as MLB records them."""
    s = _get(f"/seasons/{season}", sportId=1)["seasons"][0]
    return (dt.date.fromisoformat(s["regularSeasonStartDate"]),
            dt.date.fromisoformat(s["regularSeasonEndDate"]))


def team_ids(season: int) -> list[int]:
    teams = _get("/teams", sportId=1, season=season)["teams"]
    return sorted(t["id"] for t in teams)


def roster_day(season: int, team: int, day: dt.date) -> list[dict]:
    rows = _get(f"/teams/{team}/roster", rosterType="40Man", date=day.isoformat())
    return [{"season": season, "date": day, "team_id": team,
             "player_id": int(p["person"]["id"]),
             "status": (p.get("status") or {}).get("code")}
            for p in rows.get("roster", [])]


def pull_season(season: int, through: dt.date | None = None) -> Path:
    """
    Pull a season's team-days into rosters_{season}.parquet, adding to what
    the file already holds — so the nightly job calls it with through=
    yesterday and fetches only the new days (30 requests a night). A day
    already in the file is not fetched again; a team-day that came back empty
    is retried next time.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    final = OUT / f"rosters_{season}.parquet"
    partial = OUT / f"rosters_{season}.partial.parquet"
    empty = pd.DataFrame(columns=["season", "date", "team_id", "player_id", "status"])
    have = (pd.read_parquet(partial) if partial.exists()
            else pd.read_parquet(final) if final.exists() else empty)
    done = {(int(t), d) for t, d in zip(have["team_id"], pd.to_datetime(have["date"]).dt.date)}

    start, end = season_window(season)
    last = min(end, through) if through else end
    teams = team_ids(season)
    days = [start + dt.timedelta(n) for n in range((last - start).days + 1)]
    todo = [(t, d) for d in days for t in teams if (t, d) not in done]
    if not todo:
        log.info("%d: current through %s", season, last)
        return final
    log.info("%d: %s to %s, %d clubs, %d team-days to pull (%d already)",
             season, start, last, len(teams), len(todo), len(done))

    frames = [have] if len(have) else []
    batch: list[dict] = []
    with ThreadPoolExecutor(THREADS) as pool:
        for i, rows in enumerate(pool.map(lambda td: roster_day(season, *td), todo), 1):
            batch.extend(rows)
            if i % 600 == 0 or i == len(todo):
                frames.append(pd.DataFrame(batch))
                batch = []
                pd.concat(frames, ignore_index=True).to_parquet(partial, index=False)
                log.info("  %d: %d/%d team-days", season, i, len(todo))
    out = pd.concat(frames, ignore_index=True) if frames else have
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(["date", "team_id", "player_id"])
    out.to_parquet(final, index=False)
    partial.unlink(missing_ok=True)
    log.info("%d: saved %d rows", season, len(out))
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default=f"{FIRST_SEASON}-{dt.date.today().year}")
    ap.add_argument("--through", type=dt.date.fromisoformat, default=None,
                    help="pull only days up to this date (YYYY-MM-DD)")
    args = ap.parse_args()
    a, _, b = args.seasons.partition("-")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    for season in range(int(a), int(b or a) + 1):
        pull_season(season, args.through)
    return 0


if __name__ == "__main__":
    sys.exit(main())
