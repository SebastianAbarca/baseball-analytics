#!/usr/bin/env python3
"""
refresh.py — one command to bring every derived layer up to date.

A full refresh has always been a sequence run by hand: top up Statcast,
regenerate the tunnel caches, rebuild portraits season by season, regenerate
the tag reference, upload. Every stale-cache bug this project has had came
from that sequence being run partially or out of order — portraits rebuilt
from a half-regenerated pool, thirty-one schema-20 portraits skipped as cache
hits, a tunnel cache built from a Statcast file that was itself three months
stale. The order is not optional, so it should not be retyped each time.

Stages run in dependency order, each feeding the next:

    statcast   top up any in-progress season (finished seasons are immutable)
    sources    re-pull the external feeds — FanGraphs/BRef/Savant aggregates,
               fielding OAA, bWAR, defensive runs, IL — and clear the pools
               built from them                          (opt-in)
    tunnel     regenerate tunnel_league_*.parquet where Statcast moved
    portraits  rebuild whatever the build fingerprint now marks stale,
               and rewrite league_identity_*.json
    tags       regenerate TAGS.md and dashboard/tag_reference.json
    upload     push portraits to Supabase Storage        (opt-in)

Routine refresh (data you own, no external feeds):
    python scripts/refresh.py

EVERYTHING, including re-pulling every external source:
    python scripts/refresh.py --stages statcast,sources,tunnel,portraits,tags

See what would change, touch nothing:
    python scripts/refresh.py --dry-run

One season:            --seasons 2026
Skip the slow part:    --skip portraits
Parallel seasons:      --jobs 4
After a CODE change
rather than a data
change:                --force-tunnel

Every cache here is gated on `if cache.exists()` with no way to ask for a
fresh copy, so refreshing one means removing it and calling its puller again.
`sources` moves each cache aside rather than deleting it and puts it back if
the pull fails, because these feeds do fail — pull_fg_batting has a
three-step fallback chain precisely because FanGraphs answers 403.

Statcast and tunnel run in the parent before any fork, so the workers only
ever read those files. --jobs then rebuilds seasons in parallel; portraits
and league_identity files are per-season, so nothing is shared.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "modules"))

RAW_DIR       = _HERE.parent / "data" / "raw"
TUNNEL_DIR    = _HERE.parent / "modules" / "processed"
PORTRAIT_DIR  = _HERE.parent / "data" / "processed" / "portraits"

FIRST_SEASON, LAST_SEASON = 2015, 2026
DASHBOARD_PORT = 8050

STAGES = ("statcast", "sources", "tunnel", "portraits", "tags", "upload")
# `sources` and `upload` are opt-in: the first re-pulls every external feed
# (slow, network-bound, and the least reliable step), the second publishes.
DEFAULT_STAGES = ("statcast", "tunnel", "portraits", "tags")

log = logging.getLogger("refresh")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def dashboard_is_running(port: int = DASHBOARD_PORT) -> bool:
    """
    Is something listening on the dashboard port?

    The dashboard is a WRITER: on a cache miss it builds the portrait itself
    and saves it. During a rebuild that means it can serve a request against
    half-regenerated pools and write the result, which then looks like a valid
    cache hit forever. That is not hypothetical — it is how a bad HOU_2025
    portrait got created mid-rebuild. So refuse to start while it is up.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_statcast(seasons: list[int], dry: bool) -> list[int]:
    """Top up in-progress seasons. Returns seasons whose data actually moved."""
    from ingest import pull_statcast_season, statcast_gap

    moved = []
    for s in seasons:
        cache = RAW_DIR / f"statcast_{s}.parquet"
        if not cache.exists():
            log.info("  %d: no local Statcast cache — will pull in full", s)
            moved.append(s)
            continue
        if dry:
            # ingest.statcast_gap is the one definition of "behind", so this
            # report cannot drift from what the real run would do.
            gap = statcast_gap(s)
            if gap:
                log.info("  %d: cache ends %s, would top up to %s", s, *gap)
                moved.append(s)
            continue
        before = cache.stat().st_mtime
        pull_statcast_season(s)
        if cache.stat().st_mtime > before:
            log.info("  %d: topped up", s)
            moved.append(s)
    return moved


PROCESSED = _HERE.parent / "data" / "processed"

# External pulls, each cached "if cache.exists(): return it" with no way to
# ask for a fresh copy. Refreshing one means removing its cache and calling
# the puller again — so each entry is (label, filenames, how to re-pull).
def _source_specs():
    import ingest

    def one(fn):
        return lambda s: fn(s)

    return [
        ("fg_batting",   lambda s: [f"fg_batting_{s}.csv"],   one(ingest.pull_fg_batting)),
        ("fg_pitching",  lambda s: [f"fg_pitching_{s}.csv"],  one(ingest.pull_fg_pitching)),
        ("fielding_oaa", lambda s: [f"fielding_oaa_{s}.csv"], one(ingest.pull_fielding_oaa)),
        ("bwar",         lambda s: [f"bwar_bat_{s}.csv", f"bwar_pitch_{s}.csv"],
                         one(ingest.pull_bwar)),
        ("def_runs",     lambda s: [f"def_runs_{s}.csv"],     one(ingest.pull_def_runs)),
        ("il",           lambda s: [f"il_{s}.csv"],
                         lambda s: ingest.pull_il_data(s, force=True)),
    ]

# Caches with no puller of their own: they are rebuilt as a side effect of the
# next portrait build, so refreshing them means deleting them. The history and
# prior pools are built FROM fg_*, so re-pulling the root without clearing
# these would rebuild nothing — the layering trap from repair_metric_units.
def _derived_globs(seasons: list[int]) -> list[Path]:
    out: list[Path] = []
    for s in seasons:
        out += [PROCESSED / f"team_turnover_{s}.csv",
                PROCESSED / f"est_service_time_{s}.csv"]
    out += sorted(PROCESSED.glob("*history*.csv"))
    out += sorted(PROCESSED.glob("*prior*.parquet"))
    return [p for p in out if p.exists()]


def stage_sources(seasons: list[int], dry: bool) -> None:
    """
    Re-pull the external season data: FanGraphs/BRef/Savant aggregates, OAA,
    bWAR, defensive runs, IL stints.

    Opt-in, because it is the slow, network-bound, and least reliable part of
    a refresh — pull_fg_batting alone has a three-step fallback chain because
    FanGraphs returns 403. Each cache is moved aside rather than deleted, and
    restored if its pull fails, so a refresh that dies halfway leaves the data
    it started with instead of a hole.
    """
    specs = _source_specs()
    ok = failed = skipped = 0
    for s in seasons:
        for label, files, pull in specs:
            paths = [PROCESSED / f for f in files(s)]
            if dry:
                have = [p for p in paths if p.exists()]
                log.info("  %d %-13s would re-pull (%s)", s, label,
                         ", ".join(p.name for p in have) or "no cache yet")
                skipped += 1
                continue
            baks = []
            for p in paths:
                if p.exists():
                    b = p.with_suffix(p.suffix + ".bak")
                    p.rename(b)
                    baks.append((p, b))
            try:
                pull(s)
                for _, b in baks:
                    b.unlink(missing_ok=True)
                log.info("  %d %-13s refreshed", s, label)
                ok += 1
            except Exception as e:
                for p, b in baks:          # put back what was there
                    b.rename(p)
                log.warning("  %d %-13s FAILED (%s) — kept existing cache",
                            s, label, str(e)[:80])
                failed += 1

    derived = _derived_globs(seasons)
    if derived:
        log.info("  %s %d derived pool file(s) so they rebuild from the new roots",
                 "would clear" if dry else "clearing", len(derived))
        # The history and prior pools span every season, not just the ones
        # asked for, so touching one season's roots invalidates all of them.
        # That is correct — the pool contains the season that just changed —
        # but it turns "refresh 2026" into a full 12-season rebuild, which is
        # too expensive to discover only when it starts.
        if any("history" in p.name or "prior" in p.name for p in derived):
            log.warning("  note: history/prior pools are cross-season, so this "
                        "forces a FULL portrait rebuild, not just %s",
                        ", ".join(str(s) for s in seasons))
        if not dry:
            for p in derived:
                p.unlink()
    if not dry:
        log.info("  %d refreshed, %d failed", ok, failed)


def stage_tunnel(seasons: list[int], changed: list[int], force: bool, dry: bool) -> int:
    """
    Regenerate tunnel caches whose Statcast is newer than they are.

    These live in modules/processed and feed the portrait build fingerprint,
    so a tunnel cache built from stale Statcast silently poisons every
    portrait that follows it.
    """
    import pandas as pd
    from tunneling import load_league_tunnel

    COLS = ["pitcher", "game_pk", "at_bat_number", "pitch_number", "pitch_type",
            "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az"]
    done = 0
    for s in seasons:
        sc = RAW_DIR / f"statcast_{s}.parquet"
        tn = TUNNEL_DIR / f"tunnel_league_{s}.parquet"
        if not sc.exists():
            continue
        stale = (force or s in changed or not tn.exists()
                 or sc.stat().st_mtime > tn.stat().st_mtime)
        if not stale:
            continue
        if dry:
            log.info("  %d: tunnel cache would be rebuilt", s)
            done += 1
            continue
        df = pd.read_parquet(sc, columns=COLS).dropna(
            subset=["plate_x", "plate_z", "vy0", "ay"])
        agg = load_league_tunnel(s, statcast=df, force=True)
        log.info("  %d: tunnel rebuilt — %d pitchers from %d pitches",
                 s, len(agg), len(df))
        done += 1
    return done


def _rebuild_season(season: int) -> tuple[int, int, float]:
    """Worker: rebuild one season's stale portraits. Returns (season, n, secs)."""
    import build_league_identity as B
    t0 = time.time()
    B.main(season)
    n = len(list(PORTRAIT_DIR.glob(f"*_{season}.json")))
    return season, n, time.time() - t0


def stage_portraits(seasons: list[int], jobs: int, dry: bool) -> None:
    import json
    from team_portrait import portrait_is_current, build_fingerprint

    fp = build_fingerprint()
    log.info("  build fingerprint: %s", fp)

    # Drive off the roster of teams, not off the files on disk. Globbing what
    # exists can only ever find portraits that are present and stale; a
    # MISSING portrait — the most obvious kind of work there is — looks like
    # nothing at all. Deleting one team's file and being told "every portrait
    # is current" is how this was caught.
    from build_league_identity import MLB_TEAMS

    stale_by_season: dict[int, int] = {}
    for s in seasons:
        n = 0
        for team in MLB_TEAMS:
            p = PORTRAIT_DIR / f"{team}_{s}.json"
            if not p.exists():
                n += 1
                continue
            try:
                if not portrait_is_current(json.loads(p.read_text())):
                    n += 1
            except Exception:
                n += 1
        if n:
            stale_by_season[s] = n

    if not stale_by_season:
        log.info("  every portrait is current — nothing to rebuild")
        return
    total = sum(stale_by_season.values())
    log.info("  %d stale portraits across %d seasons: %s",
             total, len(stale_by_season),
             ", ".join(f"{s}({n})" for s, n in sorted(stale_by_season.items())))
    if dry:
        return

    todo = sorted(stale_by_season)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_rebuild_season, s): s for s in todo}
            for f in as_completed(futs):
                s, n, secs = f.result()
                log.info("  %d done — %d portraits in %.0fs", s, n, secs)
    else:
        for s in todo:
            _, n, secs = _rebuild_season(s)
            log.info("  %d done — %d portraits in %.0fs", s, n, secs)


def stage_tags(dry: bool) -> None:
    """
    Regenerate TAGS.md and dashboard/tag_reference.json.

    Must run AFTER portraits: it counts how often each tag fires by reading
    them, so running it first documents the previous build.
    """
    if dry:
        log.info("  would regenerate TAGS.md and tag_reference.json")
        return
    import tag_reference
    tag_reference.build(False)


def stage_upload(dry: bool) -> None:
    import upload_portraits
    upload_portraits.upload_portraits(dry_run=dry)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default=f"{FIRST_SEASON}-{LAST_SEASON}",
                    help="e.g. 2026, or 2015-2026 (default: all)")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--skip", default="", help="stages to drop from --stages")
    ap.add_argument("--jobs", type=int, default=1,
                    help="rebuild this many seasons in parallel")
    ap.add_argument("--force-tunnel", action="store_true",
                    help="rebuild tunnel caches even when Statcast has not moved "
                         "(use after changing tunnelling code)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what each stage would do, change nothing")
    ap.add_argument("--allow-dashboard", action="store_true",
                    help="run even if the dashboard is up (it can write "
                         "portraits mid-rebuild — see dashboard_is_running)")
    args = ap.parse_args()

    # stdout, not the default stderr: stage headers are printed and stage
    # detail is logged, and split across two streams they interleave out of
    # order the moment the output is piped anywhere.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    for noisy in ("ingest", "tunneling", "team_portrait", "pitch_aggregates"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if "-" in args.seasons:
        a, b = args.seasons.split("-")
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(x) for x in args.seasons.split(",")]

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    stages = [s.strip() for s in args.stages.split(",") if s.strip() and s.strip() not in skip]
    bad = set(stages) - set(STAGES)
    if bad:
        print(f"unknown stage(s): {', '.join(sorted(bad))}", file=sys.stderr)
        return 2

    if "portraits" in stages and not args.dry_run and not args.allow_dashboard:
        if dashboard_is_running():
            print(f"\nThe dashboard is listening on :{DASHBOARD_PORT}.\n"
                  "It writes portraits on a cache miss, so a rebuild racing it can\n"
                  "leave a portrait built from half-regenerated pools looking like a\n"
                  "valid cache hit. Stop it first, or pass --allow-dashboard.\n",
                  file=sys.stderr)
            return 1

    t0 = time.time()
    head = "DRY RUN — nothing will change" if args.dry_run else "refresh"
    print(f"{head}  seasons {seasons[0]}–{seasons[-1]}  stages: {', '.join(stages)}\n")

    changed: list[int] = []
    for name in STAGES:
        if name not in stages:
            continue
        ts = time.time()
        print(f"[{name}]")
        if name == "statcast":
            changed = stage_statcast(seasons, args.dry_run)
            if not changed:
                log.info("  all seasons already current")
        elif name == "sources":
            stage_sources(seasons, args.dry_run)
        elif name == "tunnel":
            n = stage_tunnel(seasons, changed, args.force_tunnel, args.dry_run)
            if not n:
                log.info("  all tunnel caches current")
        elif name == "portraits":
            stage_portraits(seasons, max(1, args.jobs), args.dry_run)
        elif name == "tags":
            stage_tags(args.dry_run)
        elif name == "upload":
            stage_upload(args.dry_run)
        print(f"  ({time.time() - ts:.0f}s)\n")

    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
