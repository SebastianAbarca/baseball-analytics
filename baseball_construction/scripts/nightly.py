#!/usr/bin/env python3
"""
nightly.py — rebuild and publish the season in progress, and nothing else.

Run every morning, March to October, by .github/workflows/nightly.yml.

    season in progress?   no  -> exit 0 (off-season, post-season)
                          no, but it is March-October and this year has no
                              SEASON_DATES entry -> exit 1: add the dates
    bootstrap             what a fresh machine lacks: the Chadwick register,
                          the frozen era (data/era -> data/processed), the
                          prior season's Statcast (roster turnover)
    statcast              top up the season to yesterday, and fetch any game
                          MLB lists as completed that the cache lacks
    sources               re-pull its season totals (BRef/Savant, OAA, bWAR, ...)
    seed                  write them to player_batting/player_pitching (strict:
                          refuses to write rows without the pitch aggregates)
    league                rebuild its league caches (tunnel, pitch mix, spin,
                          plate discipline, mechanics, sequencing) and drop its
                          cross-team rows so the portraits rebuild them
    portraits             FORCE-rebuild its 30 portraits + league identity —
                          the database totals changed and are not fingerprinted
    publish               that season's portraits, league data and Scout rows

Finished seasons are not touched: their portraits rank against a pool of
finished seasons that does not move while this one is played (see
ingest.era_pool_seasons and team_portrait.build_fingerprint). When the season
ends it joins that pool, which re-ranks everything once — the yearly finalize,
run by hand where all the data lives (docs/OPERATIONS.md).

Exits non-zero unless the goal was reached: 30 current portraits for the
season, none refused by publish, and every other portrait on disk still
current. So a bad night fails the workflow, and GitHub emails.

    python scripts/nightly.py                          # what the workflow runs
    python scripts/nightly.py --season 2026 --no-publish   # test on a finished season
"""

from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import json
import logging
import shutil
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(ROOT / "modules"))

from ingest import PROCESSED_DIR, RAW_DIR, REF_DIR, SEASON_DATES  # noqa: E402
from ingest import era_pool_seasons, season_state                   # noqa: E402

log = logging.getLogger("nightly")

ERA_DIR = ROOT / "data" / "era"
PORTRAIT_DIR = PROCESSED_DIR / "portraits"
CHADWICK_URL = ("https://raw.githubusercontent.com/chadwickbureau/register/"
                "master/data/people-{x}.csv")
CHADWICK_PARTS = "0123456789abcdef"


def resolve_season(override: int | None, today: dt.date) -> int | None:
    """The season to run, or None when there is nothing to do tonight."""
    if override:
        return override
    finished, in_progress = season_state(today)
    if in_progress:
        return in_progress
    if 3 <= today.month <= 10 and today.year not in SEASON_DATES:
        # A season that is not configured cannot be in progress, so without
        # this the job would quietly do nothing all of next season.
        raise SystemExit(f"nightly: it is {today:%B} {today.year} and "
                         f"SEASON_DATES has no {today.year} entry — add the "
                         "season's first and last regular-season dates in "
                         "modules/ingest.py")
    start = SEASON_DATES.get(today.year, (None,))[0]
    log.info("no season in progress (%s) — nothing to do",
             f"{today.year} starts {start}" if start and start > today.isoformat()
             else f"last finished: {finished[-1] if finished else 'none'}")
    return None


# ---------------------------------------------------------------------------
# Bootstrap — what a machine without this project's data needs
# ---------------------------------------------------------------------------

def _chadwick() -> None:
    """The player register: 16 published CSVs, and the crosswalk that is
    exactly their concatenation (516,081 rows either way when checked)."""
    xwalk = REF_DIR / "chadwick_crosswalk.csv"
    if xwalk.exists():
        return
    REF_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for x in CHADWICK_PARTS:
        path = REF_DIR / f"people-{x}.csv"
        if not path.exists():
            log.info("  chadwick: downloading people-%s.csv", x)
            urllib.request.urlretrieve(CHADWICK_URL.format(x=x), path)
        parts.append(path)
    with open(xwalk, "wb") as out:
        for i, path in enumerate(parts):
            with open(path, "rb") as fh:
                if i:
                    fh.readline()            # every part repeats the header
                shutil.copyfileobj(fh, out)
    log.info("  chadwick: crosswalk built from %d parts", len(parts))


def _era(season: int) -> None:
    """Copy the frozen finished-season roots into data/processed, so this
    machine ranks against byte-identical files to the one that froze them."""
    copied = 0
    for s in era_pool_seasons(season):
        for pat in ("fg_batting_{s}.csv", "fg_pitching_{s}.csv",
                    "cross_team_metrics_v3_{s}.csv"):
            src, dst = ERA_DIR / pat.format(s=s), PROCESSED_DIR / pat.format(s=s)
            if not src.exists():
                if s == season:
                    continue             # the season in progress is not frozen
                raise SystemExit(f"nightly: {src} is missing — run "
                                 "scripts/freeze_era.py where the data lives "
                                 "and commit data/era/")
            if dst.exists() and filecmp.cmp(src, dst, shallow=False):
                continue
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    if copied:
        log.info("  era: %d frozen file(s) copied into data/processed", copied)


def _prior_statcast(season: int) -> None:
    """Last season's pitches, for roster turnover. Finished, so pulled once."""
    if (RAW_DIR / f"statcast_{season - 1}.parquet").exists():
        return
    from ingest import pull_statcast_season
    log.info("  statcast %d: not on this machine — pulling the full season "
             "(once; it is finished)", season - 1)
    df = pull_statcast_season(season - 1, top_up=False)
    log.info("  statcast %d: %d pitches", season - 1, len(df))


def bootstrap(season: int) -> None:
    _chadwick()
    _era(season)
    _prior_statcast(season)


# ---------------------------------------------------------------------------

def _stale_elsewhere(season: int) -> int:
    """Portraits of OTHER seasons on this machine that stopped being current.
    Tonight should not have moved them; if it did, the isolation is broken."""
    from team_portrait import clear_fingerprint_cache, portrait_is_current
    clear_fingerprint_cache()
    n = 0
    for f in PORTRAIT_DIR.glob("*_*.json"):
        if int(f.stem.rsplit("_", 1)[1]) == season:
            continue
        try:
            if not portrait_is_current(json.loads(f.read_text())):
                n += 1
        except Exception:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, help="run this season even if it is not in progress")
    ap.add_argument("--no-publish", action="store_true", help="build, but write nothing to the database")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    for noisy in ("ingest", "tunneling", "team_portrait", "pitch_aggregates", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    season = resolve_season(args.season, dt.date.today())
    if season is None:
        return 0

    import refresh as R
    t0 = time.time()
    steps = []

    def step(name, fn):
        ts = time.time()
        log.info("[%s]", name)
        out = fn()
        steps.append(f"{name} {time.time() - ts:.0f}s")
        return out

    step("bootstrap", lambda: bootstrap(season))
    moved = step("statcast", lambda: R.stage_statcast([season], dry=False))
    finished, _ = season_state()
    if season in finished:
        # Only reachable with --season (testing). A finished season's roots are
        # frozen in data/era and sit in EVERY season's era pool, so re-pulling
        # them would re-rank all 360 portraits — the opposite of a nightly run.
        log.info("[sources, seed] skipped: %d is finished and its roots are frozen", season)
    else:
        step("sources", lambda: R.stage_sources([season], dry=False))

        def seed():
            from seed import seed_season
            from team_portrait import clear_fingerprint_cache
            seed_season(season, force=True)
            clear_fingerprint_cache()          # the fingerprint hashes these rows
        step("seed", seed)
    step("league", lambda: R.stage_league([season], moved, force=False, dry=False))
    problems = step("portraits", lambda: R.stage_portraits([season], jobs=1, dry=False,
                                                          force=True))
    elsewhere = _stale_elsewhere(season)

    refused = 0
    if problems or elsewhere:
        # Act on a failed build BEFORE writing, not after: last night's
        # published season is better than tonight's partial one.
        log.error("[publish] skipped: the build failed — production unchanged")
    elif not args.no_publish:
        def publish():
            import publish as P
            return P.publish_all(ROOT, {season}, P._write_client(), False)
        refused = step("publish", publish)

    log.info("\nnightly %d: %s — %.0fs (%s)", season,
             "OK" if not (problems or refused or elsewhere) else "FAILED",
             time.time() - t0, ", ".join(steps))
    if problems:
        log.error("  %d portrait(s) of %d failed or are still stale", problems, season)
    if refused:
        log.error("  publish refused %d portrait(s)", refused)
    if elsewhere:
        log.error("  %d portrait(s) of OTHER seasons went stale — tonight should "
                  "not touch them", elsewhere)
    return 1 if (problems or refused or elsewhere) else 0


if __name__ == "__main__":
    sys.exit(main())
