#!/usr/bin/env python3
"""
refresh.py — one command to bring every derived layer up to date.

A full refresh has always been a sequence run by hand: top up Statcast,
regenerate the tunnel caches, rebuild portraits season by season, regenerate
the tag reference, publish. Every stale-cache bug this project has had came
from that sequence being run partially or out of order — portraits rebuilt
from a half-regenerated pool, thirty-one schema-20 portraits skipped as cache
hits, a tunnel cache built from a Statcast file that was itself three months
stale. The order is not optional, so it should not be retyped each time.

Stages run in dependency order, each feeding the next:

    statcast   top up any in-progress season, then fetch any game MLB lists as
               completed that the cache lacks (every season)
    sources    re-pull the external feeds — FanGraphs/BRef/Savant aggregates,
               fielding OAA, bWAR, defensive runs, IL — and clear the pools
               built from them                          (opt-in)
    seed       write the re-pulled season totals to the database the
               portraits read them from, built from the BRef tables
               `sources` cached — no BRef request of its own
                                                        (opt-in; writes production)
    league     rebuild every league cache (tunnel, pitch mix, spin, plate
               discipline, mechanics, sequencing) where Statcast moved; spin
               also when its method changed, re-pulling Savant's measured
               active spin
    portraits  rebuild whatever the build fingerprint now marks stale,
               and rewrite league_identity_*.json
    tags       regenerate TAGS.md and dashboard/tag_reference.json
    publish    write portraits, league seasons and Scout to the serving
               tables the dashboard reads (scripts/publish.py)

Routine refresh (data you own, no external feeds):
    python scripts/refresh.py

EVERYTHING, including re-pulling every external source:
    python scripts/refresh.py --stages statcast,sources,seed,league,portraits,tags,publish

See what would change, touch nothing:
    python scripts/refresh.py --dry-run

One season:            --seasons 2026
Skip the slow part:    --skip portraits
Parallel seasons:      --jobs 4
After a CODE change
rather than a data
change:                --force-league

Every cache here is gated on `if cache.exists()` with no way to ask for a
fresh copy, so refreshing one means removing it and calling its puller again.
`sources` moves each cache aside rather than deleting it and puts it back if
the pull fails, because these feeds do fail — pull_fg_batting has a
three-step fallback chain precisely because FanGraphs answers 403.

The caching is five layers deep, not four, and the outermost one is not ours:

    pybaseball's HTTP cache (~/.pybaseball/cache, enabled globally)
      -> fg_*.csv, bwar_*, fielding_oaa_*, def_runs_*, il_*
           -> *_history_*.csv
                -> *_prior_*.parquet
                     -> portraits/*.json

Deleting our CSV and re-pulling only reaches the second layer: pybaseball
answers from its own store and writes back the same numbers. `sources`
therefore disables that cache for its duration — see stage_sources.

Statcast and league run in the parent before any fork, so the workers only
ever read those files. --jobs then rebuilds seasons in parallel; portraits
and league_identity files are per-season, so nothing is shared.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))                 # seed.py
sys.path.insert(0, str(_HERE.parent / "modules"))

RAW_DIR       = _HERE.parent / "data" / "raw"
LEAGUE_DIR    = _HERE.parent / "modules" / "processed"
PORTRAIT_DIR  = _HERE.parent / "data" / "processed" / "portraits"

FIRST_SEASON, LAST_SEASON = 2015, 2026

STAGES = ("statcast", "sources", "seed", "league", "portraits", "tags", "publish")
# `sources` is opt-in: it re-pulls every external feed (slow, network-bound,
# and the least reliable step). `seed` is opt-in because it writes the
# production player tables; run it with `sources`, since the portraits read
# season totals from those tables, not from the CSVs `sources` refreshes.
# `publish` is on: a build nobody can read is not finished.
DEFAULT_STAGES = ("statcast", "league", "portraits", "tags", "publish")

log = logging.getLogger("refresh")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_statcast(seasons: list[int], dry: bool) -> list[int]:
    """
    Top up in-progress seasons, then make every season complete against
    MLB's schedule. Returns seasons whose data actually moved.

    The top-up only appends after the last cached date, which cannot recover
    a game missed before it — an opening series outside SEASON_DATES, a day
    a pull dropped. backfill_statcast_games fetches exactly the game_pks
    MLB lists as completed and the cache lacks; a complete season costs one
    schedule lookup (cached once the season is finished) and no Statcast
    request.
    """
    from ingest import (backfill_statcast_games, pull_statcast_season,
                        statcast_gap, statcast_missing_games)

    moved = []
    for s in seasons:
        cache = RAW_DIR / f"statcast_{s}.parquet"
        if not cache.exists():
            log.info("  %d: no local Statcast cache — %s in full", s,
                     "would pull" if dry else "pulling")
            # Pulled HERE, not left for the portrait build to fetch: the tunnel
            # stage needs the file, and a tunnel pool first written mid-build
            # moves the fingerprint under it. On a fresh machine that stamped
            # all 30 portraits stale and failed the night.
            if not dry:
                pull_statcast_season(s)
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

    # Completeness — after the top-up, so only genuinely missing games remain.
    incomplete = []
    for s in seasons:
        if not (RAW_DIR / f"statcast_{s}.parquet").exists():
            continue
        try:
            missing = statcast_missing_games(s)
        except Exception as e:
            log.warning("  %d: could not read MLB's schedule (%s) — completeness "
                        "not checked", s, e)
            continue
        if missing.empty:
            continue
        if dry:
            log.info("  %d: %d game(s) missing, would fetch (%s)", s, len(missing),
                     ", ".join(sorted(set(missing["date"]))))
            moved.append(s)
            continue
        added = backfill_statcast_games(s)
        if added:
            log.info("  %d: %d missing game(s) fetched", s, added)
            moved.append(s)
        if added < len(missing):
            incomplete.append(f"{s} ({len(missing) - added})")
    if incomplete:
        log.warning("  INCOMPLETE vs MLB's schedule: %s — Statcast has not "
                    "published them yet; the next run retries", ", ".join(incomplete))
    if not dry:
        _write_strike_zone_table(seasons)
    return sorted(set(moved))


STRIKE_ZONE_TABLE = _HERE.parent / "dashboard" / "strike_zones.json"


def _write_strike_zone_table(seasons: list[int]) -> None:
    """
    Precompute each season's average strike zone into a small JSON.

    The dashboard draws the zone at the season's measured sz_top/sz_bot rather
    than a fixed constant, and it was reading that straight out of the raw
    Statcast parquet at render time: a 105 MB file resident on the serving
    machine, opened to produce two floats. Serving would have needed the whole
    ~977 MB of raw data for twenty-four numbers.

    Computed here instead, where the raw data already lives, so the web tier
    can hold portraits and nothing else.
    """
    import json
    import pandas as pd

    table = {}
    if STRIKE_ZONE_TABLE.exists():
        try:
            table = json.loads(STRIKE_ZONE_TABLE.read_text())
        except Exception:
            table = {}
    for s in seasons:
        path = RAW_DIR / f"statcast_{s}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["sz_top", "sz_bot"])
        top = pd.to_numeric(df["sz_top"], errors="coerce").mean()
        bot = pd.to_numeric(df["sz_bot"], errors="coerce").mean()
        if pd.notna(top) and pd.notna(bot) and 2.0 < top < 5.0 and 0.5 < bot < 2.5:
            table[str(s)] = {"top": round(float(top), 4),
                             "bot": round(float(bot), 4),
                             "pitches": int(len(df))}
    STRIKE_ZONE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    STRIKE_ZONE_TABLE.write_text(json.dumps(table, indent=1, sort_keys=True))
    log.info("  wrote %s — %d seasons", STRIKE_ZONE_TABLE.name, len(table))


PROCESSED = _HERE.parent / "data" / "processed"

# External pulls, each cached "if cache.exists(): return it" with no way to
# ask for a fresh copy. Refreshing one means removing its cache and calling
# the puller again — so each entry is (label, filenames, how to re-pull).
def _source_specs():
    import ingest

    def one(fn):
        return lambda s: fn(s)

    # bref_range first: the fg_* pulls and the seed build from these tables,
    # so refreshing them here is the only BRef request either makes.
    return [
        ("bref_range",   lambda s: [f"bref_batting_range_{s}.csv",
                                    f"bref_pitching_range_{s}.csv"],
                         lambda s: [ingest.pull_bref_range(s, k)
                                    for k in ("batting", "pitching")]),
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
    """
    Derived files that contain one of `seasons`, and only those.

    It used to clear every history and prior pool on any refresh, which made
    "refresh 2026" a 12-season rebuild. A history pool is named for the season
    range it concatenates (batting_history_2015_2027.csv), so only ranges that
    include a refreshed season are touched; the finished seasons' shared pool
    survives a night of new games. A prior pool holds season s for season
    s+1's projection, so *_prior_{s} goes when s is refreshed.
    """
    import re
    want = set(seasons)
    out: list[Path] = []
    for s in seasons:
        out += [PROCESSED / f"team_turnover_{s}.csv",
                PROCESSED / f"est_service_time_{s}.csv",
                PROCESSED / f"cross_team_metrics_v2_{s}.csv"]
        out += sorted(PROCESSED.glob(f"*_prior_{s}.parquet"))
    for p in sorted(PROCESSED.glob("*_history_*_*.csv")):
        m = re.search(r"_history_(\d{4})_(\d{4})\.csv$", p.name)
        if m and any(int(m.group(1)) <= s <= int(m.group(2)) for s in want):
            out.append(p)
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

    # pybaseball keeps its OWN on-disk HTTP cache, one layer further out than
    # anything else here, and it is enabled globally. Re-running a pull with
    # it on rebuilds our CSV from pybaseball's stored response instead of from
    # the source. That is not theoretical: the first run of this stage rewrote
    # fg_batting_2026 byte-identically, and cross-checking it against Statcast
    # (current to the day) showed the pool covering 98.6% of plate appearances
    # — about two games behind, the age of pybaseball's copy.
    #
    # Disabled for the duration rather than purged: purge() drops all ~9,500
    # entries including every cached Statcast day, which would make the next
    # top-up re-download the season. Disabling is process-local, so the
    # fetches reach the source and nothing else loses its cache.
    from pybaseball import cache as _pb_cache
    was_enabled = _pb_cache.config.enabled
    if not dry and was_enabled:
        _pb_cache.disable()
        log.info("  pybaseball HTTP cache off for this stage "
                 "(it would otherwise serve its own stored responses)")

    ok = failed = 0
    try:
        for s in seasons:
            for label, files, pull in specs:
                paths = [PROCESSED / f for f in files(s)]
                if dry:
                    have = [p for p in paths if p.exists()]
                    log.info("  %d %-13s would re-pull (%s)", s, label,
                             ", ".join(p.name for p in have) or "no cache yet")
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
    finally:
        if not dry and was_enabled:
            _pb_cache.enable()

    derived = _derived_globs(seasons)
    if derived:
        log.info("  %s %d derived pool file(s) so they rebuild from the new roots",
                 "would clear" if dry else "clearing", len(derived))
        # A finished season is in every season's era pool, so refreshing its
        # roots re-ranks all of them — a full rebuild, worth knowing before it
        # starts. The season in progress is only in its own pool.
        from ingest import season_state
        finished, _ = season_state()
        if any(s in finished for s in seasons):
            log.warning("  note: %s finished, and finished seasons are in every "
                        "season's era pool, so this forces a FULL portrait "
                        "rebuild", ", ".join(str(s) for s in seasons if s in finished))
        if not dry:
            for p in derived:
                p.unlink()
    if not dry:
        log.info("  %d refreshed, %d failed", ok, failed)


# Every per-season cache derived from raw Statcast, with its builder. Each is
# read by the portraits and NONE checked its input: the loaders return the
# parquet whenever it exists. Tunnel and spin had stages of their own; the
# other four had nothing, and by the end of 2026 all five non-tunnel caches
# were 92.8 days older than the raw file — pitch_mix held 191,141 pitches of a
# season with 713,868. One stage now covers all six. They are
# all in the build fingerprint (team_portrait._FINGERPRINT_EXTRA_POOLS), so a
# rebuilt cache marks that season's portraits stale.
def _league_caches():
    from pitch_mix import build_pitch_mix, load_league_plate_discipline
    from pitcher_traits import load_league_mechanics, load_league_sequencing
    from spin_efficiency import load_league_spin
    from tunneling import load_league_tunnel
    return [
        ("tunnel_league_{s}.parquet",           load_league_tunnel),
        ("pitch_mix_{s}.parquet",               build_pitch_mix),
        ("spin_league_{s}.parquet",             load_league_spin),
        ("plate_discipline_league_{s}.parquet", load_league_plate_discipline),
        ("mechanics_league_{s}.parquet",        load_league_mechanics),
        ("sequencing_league_{s}.parquet",       load_league_sequencing),
    ]


_TUNNEL_COLS = ["pitcher", "game_pk", "at_bat_number", "pitch_number", "pitch_type",
                "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az"]
# The spin build's input, as its own stage gave it (spin_efficiency, PR #19).
_SPIN_COLS = ["pitcher", "pitch_type", "game_year", "home_team", "release_speed",
              "release_spin_rate", "release_pos_y", "release_extension",
              "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z", "spin_axis"]


def _spin_method_stale(cache: Path) -> bool:
    """A spin cache built by an older SPIN_METHOD_VERSION is stale even when
    Statcast has not moved — the method, not the data, changed."""
    import pandas as pd
    from spin_efficiency import SPIN_METHOD_VERSION
    cached = pd.read_parquet(cache, columns=None)
    return ("spin_method_version" not in cached.columns
            or not (cached["spin_method_version"] == SPIN_METHOD_VERSION).all())


def stage_seed(seasons: list[int], dry: bool) -> None:
    """
    Re-seed player_batting / player_pitching from the sources just pulled.
    The portraits read season totals from these tables whenever a season is
    seeded, so refreshing the CSVs alone changed nothing a portrait reads.
    Strict: refuses to write rows without the Statcast pitch aggregates.
    """
    if dry:
        log.info("  would re-seed player_batting / player_pitching for %s",
                 ", ".join(map(str, seasons)))
        return
    from seed import seed_season
    for s in seasons:
        seed_season(s, force=True)
        log.info("  %d: seeded", s)


def stage_league(seasons: list[int], changed: list[int], force: bool, dry: bool,
                 force_caches: frozenset[str] = frozenset()) -> int:
    """
    Rebuild every league cache whose raw Statcast is newer than it (or that
    is missing, or whose season the statcast stage just moved), and drop the
    season's cross-team rows so the portraits stage rebuilds them from the
    same data. force_caches names caches to rebuild regardless
    ("spin_league" for --force-spin). Returns the number of caches rebuilt.
    """
    from ingest import pull_statcast_season
    from team_portrait import cross_team_path

    done = 0
    for s in seasons:
        sc = RAW_DIR / f"statcast_{s}.parquet"
        if not sc.exists():
            continue
        stale = []
        for pat, build in _league_caches():
            cache = LEAGUE_DIR / pat.format(s=s)
            kind = pat.replace("_{s}.parquet", "")
            if (force or kind in force_caches or s in changed or not cache.exists()
                    or sc.stat().st_mtime > cache.stat().st_mtime
                    or (kind == "spin_league" and _spin_method_stale(cache))):
                stale.append((cache, build))
        if not stale:
            continue
        names = ", ".join(c.name.replace(f"_{s}.parquet", "") for c, _ in stale)
        if dry:
            log.info("  %d: would rebuild %s", s, names)
            done += len(stale)
            continue
        statcast = pull_statcast_season(s, top_up=False)
        for cache, build in stale:
            data = statcast
            if cache.name.startswith("tunnel_league_"):
                # The input the tunnel stage always gave it — located pitches
                # with kinematics. Dropping rows changes which pitches pair,
                # so this keeps tunnel scores identical to before; pairing
                # across a dropped pitch is review item 10, separately.
                data = statcast[_TUNNEL_COLS].dropna(
                    subset=["plate_x", "plate_z", "vy0", "ay"])
            elif cache.name.startswith("spin_league_"):
                data = statcast[[c for c in _SPIN_COLS if c in statcast.columns]]
            out = build(s, statcast=data, force=True)
            if cache.name.startswith("spin_league_") and len(out):
                share = ((out["spin_savant_share"] * out["total_pitches"]).sum()
                         / out["total_pitches"].sum())
                log.info("  %d: spin — %d pitchers, %.0f%% of pitches from Savant's "
                         "measured active spin", s, len(out), 100 * share)
        ct = cross_team_path(s)
        if ct.exists():
            ct.unlink()
        log.info("  %d: rebuilt %s (%d pitches)", s, names, len(statcast))
        done += len(stale)
        del statcast
    return done


def _rebuild_season(season: int, force: bool = False) -> tuple[int, int, float, int]:
    """
    Worker: rebuild one season's stale portraits (all of them if force).
    Returns (season, portraits on disk, secs, failed builds).
    """
    import build_league_identity as B
    t0 = time.time()
    failed = B.main(season, force=force)
    n = len(list(PORTRAIT_DIR.glob(f"*_{season}.json")))
    return season, n, time.time() - t0, failed


def stage_portraits(seasons: list[int], jobs: int, dry: bool,
                    force: bool = False) -> int:
    """
    Rebuild stale portraits (every portrait of `seasons` if force) and their
    league identity. Returns the number of portraits still stale afterwards
    plus failed builds — 0 means the stage did what it was asked.
    """
    import json
    from team_portrait import (portrait_is_current, build_fingerprint,
                               clear_fingerprint_cache)

    # Cross-team rows for every season these portraits rank against, built
    # FIRST: before the workers fork, and before the fingerprints are read,
    # since the rows are pool files. They were a side effect of each season's
    # first portrait, so a full rebuild ranked team construction against
    # whichever seasons happened to finish first.
    if not dry:
        from ingest import era_pool_seasons
        from team_portrait import ensure_cross_team_rows
        need = sorted({p for s in seasons for p in era_pool_seasons(s)})
        built = [s for s in need if (RAW_DIR / f"statcast_{s}.parquet").exists()
                 and ensure_cross_team_rows(s)]
        if built:
            log.info("  cross-team rows built for %s", ", ".join(map(str, built)))
            clear_fingerprint_cache()

    # Per season: each season's portraits hash only the pools they read.
    fp = {s: build_fingerprint(s) for s in seasons}
    log.info("  build fingerprints: %s",
             ", ".join(f"{s}={fp[s]}" for s in seasons))

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
        if force:
            n = len(MLB_TEAMS)
        if n:
            stale_by_season[s] = n

    if not stale_by_season:
        log.info("  every portrait is current — nothing to rebuild")
        return 0
    total = sum(stale_by_season.values())
    log.info("  %d %s portraits across %d seasons: %s",
             total, "forced" if force else "stale", len(stale_by_season),
             ", ".join(f"{s}({n})" for s, n in sorted(stale_by_season.items())))
    if dry:
        return 0

    todo = sorted(stale_by_season)

    failed = 0
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_rebuild_season, s, force): s for s in todo}
            for f in as_completed(futs):
                s, n, secs, nf = f.result()
                failed += nf
                log.info("  %d done — %d portraits in %.0fs%s", s, n, secs,
                         f", {nf} FAILED" if nf else "")
    else:
        for s in todo:
            _, n, secs, nf = _rebuild_season(s, force)
            failed += nf
            log.info("  %d done — %d portraits in %.0fs%s", s, n, secs,
                     f", {nf} FAILED" if nf else "")

    # Did the fingerprint move underneath the rebuild?
    #
    # The history and prior pools feed the fingerprint AND are created as a
    # side effect of building a portrait, so a rebuild that starts with them
    # missing — which is exactly what `sources` leaves behind — writes every
    # portrait stamped with the fingerprint of an empty pool set, and they all
    # read as stale the moment the last pool lands. build_fingerprint is
    # memoised per process, so no worker notices.
    #
    # Seen for real: a full 18-minute rebuild stamped 058b98caea4eb62e while
    # the true value ended up 8c49e074df66b935. The portraits were CORRECT —
    # each was built against properly regenerated pools — but every one of
    # them looked stale, so the next run would have rebuilt the lot again.
    # Did the rebuild actually achieve anything? build_league_identity catches
    # per-team exceptions and carries on, so a season can report "30/30" with
    # portraits it never rewrote. Seen for real: three workers hit Supabase at
    # once at the start of a run, all three got "Server disconnected", and
    # ATH 2015/2016/2017 stayed on the previous build while the run reported
    # success. Counting what is still stale afterwards is the only check that
    # speaks to the goal rather than to the attempt.
    clear_fingerprint_cache()          # re-read the pools the build just wrote
    still = 0
    for s in todo:
        for team in MLB_TEAMS:
            p = PORTRAIT_DIR / f"{team}_{s}.json"
            try:
                if not p.exists() or not portrait_is_current(json.loads(p.read_text())):
                    still += 1
            except Exception:
                still += 1

    after = {s: build_fingerprint(s) for s in todo}
    moved = {s for s in todo if after[s] != fp[s]}
    if still and not moved:
        log.warning(
            "  %d portrait(s) are STILL stale after the rebuild — their builds "
            "failed and the old files were kept. Check the log above for "
            "'FAILED'; a transient database error is the usual cause, and "
            "re-running this stage picks up exactly those.", still)
    if moved:
        log.warning(
            "  fingerprint moved during the rebuild for %s.\n"
            "  The pools these portraits depend on were still being written "
            "when the build started, so they are stamped with a value that is "
            "already out of date. The DATA is fine; the stamp is not.\n"
            "  Run the portraits stage once more to settle it — the pools all "
            "exist now, so the second pass converges:\n"
            "      python scripts/refresh.py --stages portraits --jobs %d",
            ", ".join(f"{s} ({fp[s]} -> {after[s]})" for s in sorted(moved)),
            max(1, jobs))
    return still + failed


def stage_tags(dry: bool) -> None:
    """
    Regenerate the tag reference from the portraits: TAGS.md and
    tag_reference.json.

    Must run AFTER portraits — both are counted out of them, so running first
    documents the previous build.
    """
    if dry:
        log.info("  would regenerate TAGS.md, tag_reference.json")
        return
    import tag_reference
    tag_reference.build(False)


def stage_publish(seasons: list[int], dry: bool) -> None:
    """
    Write the built artefacts to the serving tables. Must run AFTER portraits
    and tags; refuses any portrait that is not current for this machine.
    """
    import publish
    client = publish._write_client()
    root = _HERE.parent
    refused = publish.publish_portraits(root, set(seasons), client, dry, False)
    publish.publish_league(root, set(seasons), client, dry, False)
    publish.publish_scout(root, client, dry)
    if refused:
        log.warning("  %d portrait(s) refused as not current — see above", refused)


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
    ap.add_argument("--force-league", "--force-tunnel", dest="force_league",
                    action="store_true",
                    help="rebuild the league caches even when Statcast has not "
                         "moved (use after changing tunnel/pitch-mix/spin/"
                         "discipline/mechanics/sequencing code)")
    ap.add_argument("--force-spin", action="store_true",
                    help="rebuild only the spin caches, re-pulling Savant's active "
                         "spin, even when nothing has moved")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what each stage would do, change nothing")
    args = ap.parse_args()

    # stdout, not the default stderr: stage headers are printed and stage
    # detail is logged, and split across two streams they interleave out of
    # order the moment the output is piped anywhere.
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    for noisy in ("ingest", "tunneling", "spin_efficiency", "team_portrait", "pitch_aggregates"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if "-" in args.seasons:
        a, b = args.seasons.split("-")
        seasons = list(range(int(a), int(b) + 1))
    else:
        seasons = [int(x) for x in args.seasons.split(",")]

    # `tunnel` and `spin` were separate stages before `league` rebuilt every
    # league cache; both still name it.
    alias = {"tunnel": "league", "spin": "league"}
    skip = {alias.get(s.strip(), s.strip()) for s in args.skip.split(",") if s.strip()}
    stages = [alias.get(s.strip(), s.strip()) for s in args.stages.split(",")
              if s.strip() and alias.get(s.strip(), s.strip()) not in skip]
    bad = set(stages) - set(STAGES)
    if bad:
        print(f"unknown stage(s): {', '.join(sorted(bad))}", file=sys.stderr)
        return 2

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
            if "seed" not in stages:
                log.warning("  sources refreshed without `seed`: the portraits read "
                            "season totals from the database, so they will not see "
                            "these until it is re-seeded (add seed to --stages)")
        elif name == "seed":
            stage_seed(seasons, args.dry_run)
        elif name == "league":
            n = stage_league(seasons, changed, args.force_league, args.dry_run,
                             frozenset({"spin_league"}) if args.force_spin else frozenset())
            if not n:
                log.info("  all league caches current")
        elif name == "portraits":
            stage_portraits(seasons, max(1, args.jobs), args.dry_run)
        elif name == "tags":
            stage_tags(args.dry_run)
        elif name == "publish":
            stage_publish(seasons, args.dry_run)
        print(f"  ({time.time() - ts:.0f}s)\n")

    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
