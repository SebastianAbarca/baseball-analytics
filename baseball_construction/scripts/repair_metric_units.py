"""
repair_metric_units.py — Rewrite the Savant percentile columns in the database
as true Statcast rates, and null the sprint_speed values that are percentiles.

WHY
---
The Savant leaderboard serves several columns as PERCENTILE RANKS (1–100)
rather than as rates or physical units. Measured across the whole database,
every one of them read min 1 / median ~50 / max 100 — `exit_velocity` had a
median of 49 where it should be ~88 mph, and `fb_velocity` 49 where it should
be ~94.

Uniform percentiles are at least monotonic, so anything that merely RANKS them
still ordered correctly. The damage is to absolute thresholds, and `sprint_speed`
is the case that proves it: it is MIXED rather than uniformly converted — ~600
rows (pitchers who batted, 2015–2020) hold a percentile while every other row
holds ft/s. Because the speed tags gate on absolute ft/s, 26% of every
`elite speed` tag in the portraits landed on a pitcher. Jacob deGrom carried it
four seasons running.

ingest.py now fixes this at seed time. This script repairs the rows already
stored, without re-running the full seed: every affected metric is derived from
the LOCAL statcast parquet cache, so no BRef/Savant network pull is needed.

INVARIANT
---------
Every existing row is written — with a true rate where one is computable and
NULL where it is not. Updating only the rows we can compute would leave the
rest holding percentiles, which is precisely the mixed-unit state that caused
the bug in the first place. Unit consistency matters more than coverage.

USAGE
-----
    python repair_metric_units.py --dry-run          # report, change nothing
    python repair_metric_units.py 2023               # one season
    python repair_metric_units.py                    # every cached season
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

import pitch_aggregates as pa                      # noqa: E402
from database import get_client, _upsert           # noqa: E402
from ingest import (SPRINT_SPEED_MIN_FTS,          # noqa: E402
                    SPRINT_SPEED_MAX_FTS)

log = logging.getLogger("repair")

_RAW = _HERE.parent / "data" / "raw"

# DB column ← aggregate column, per table.
_BATTING_FIX = {
    "exit_velocity": "exit_velo_sc",
    "hard_hit_pct":  "hard_hit_sc_pct",
    "barrel_pct":    "barrel_sc_pct",
    "whiff_pct":     "whiff_sc_pct",
    "chase_pct":     "chase_sc_pct",
    "xwoba":         "xwoba_sc",
    "xba":           "xba_sc",
}
_PITCHING_FIX = {
    "hard_hit_pct":  "hard_hit_sc_pct",
    "barrel_pct":    "barrel_sc_pct",
    "whiff_pct":     "whiff_sc_pct",
    "xwoba_allowed": "xwoba_allowed_sc",
    "fb_velocity":   "fb_velo_sc",
}


def _fetch_season(table: str, season: int, cols: str) -> pd.DataFrame:
    client = get_client()
    rows, start, page = [], 0, 1000
    while True:
        r = (client.table(table).select(cols)
             .eq("season", season)
             .range(start, start + page - 1).execute())
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < page:
            break
        start += page
    return pd.DataFrame(rows)


def _repair_table(table: str, season: int, agg: pd.DataFrame,
                  fixes: dict[str, str], extra_cols: str,
                  dry_run: bool) -> None:
    existing = _fetch_season(table, season, f"key_mlbam,{extra_cols}")
    if existing.empty:
        log.info("  %s %d — no rows", table, season)
        return

    merged = existing.merge(agg, on="key_mlbam", how="left")

    payload = pd.DataFrame({"key_mlbam": merged["key_mlbam"],
                            "season": season})
    report = []
    for db_col, agg_col in fixes.items():
        before = pd.to_numeric(merged.get(db_col), errors="coerce")
        after = (pd.to_numeric(merged[agg_col], errors="coerce")
                 if agg_col in merged.columns else pd.Series(np.nan, index=merged.index))
        payload[db_col] = after
        report.append({
            "column":      db_col,
            "before_med":  round(float(before.median()), 3) if before.notna().any() else None,
            "after_med":   round(float(after.median()), 3) if after.notna().any() else None,
            "was_null":    int(before.isna().sum()),
            "now_null":    int(after.isna().sum()),
        })

    # sprint_speed is not derivable from pitch data — only sanitised.
    if "sprint_speed" in merged.columns:
        ss = pd.to_numeric(merged["sprint_speed"], errors="coerce")
        bad = ss.notna() & ~ss.between(SPRINT_SPEED_MIN_FTS, SPRINT_SPEED_MAX_FTS)
        payload["sprint_speed"] = ss.where(~bad)
        report.append({
            "column": "sprint_speed", "before_med": round(float(ss.median()), 3)
            if ss.notna().any() else None,
            "after_med": round(float(ss.where(~bad).median()), 3)
            if (~bad & ss.notna()).any() else None,
            "was_null": int(ss.isna().sum()),
            "now_null": int(ss.isna().sum() + bad.sum()),
        })
        if bad.any():
            log.warning("  %s %d — nulling %d contaminated sprint_speed value(s)",
                        table, season, int(bad.sum()))

    print(f'\n  {table} {season} — {len(payload)} rows')
    print(pd.DataFrame(report).to_string(index=False))

    if dry_run:
        return
    records = payload.where(payload.notna(), other=None).to_dict("records")
    _upsert(table, records)


def purge_derived_pools(dry_run: bool) -> None:
    """
    Delete every cache that carries metric units, so they rebuild through the
    repaired code path.

    This is the trap this script sprang the first time it ran, twice. Repairing
    the stored rows is not enough: team_portrait ranks a player against pools
    cached on disk, and those files keep whatever units they were built with.
    After the first repair the database held true mph while the pool still held
    percentiles, so a 93 mph fastball was ranked against a uniform 1-100 pool
    and came back as the 93rd percentile — `elite velo` fired on 89.8% of
    pitcher-seasons and `soft tosser` on none at all.

    The caches are LAYERED, which is what made the second attempt fail too:

        fg_{batting,pitching}_{season}.csv      <- root; the Savant pull
          -> {batting,pitching}_history_*.csv   <- pooled, built FROM the root
               -> *_prior_*.parquet             <- prior-season slice
                    -> portraits/{TEAM}_{season}.json

    Purging only the pooled layer rebuilds it from the stale root and changes
    nothing. Every level goes, portraits included — they are the layer that
    actually reaches a reader, and PORTRAIT_SCHEMA_VERSION cannot save them
    here. The version detects a change of SHAPE; this is a change of CONTENT
    at the same shape, so a portrait built minutes before a repair and one
    built after are indistinguishable to it. On the third pass of this bug, 31
    portraits carrying the current version but pre-repair units were being
    skipped as cache hits by build_league_identity. Deleting them is the only
    signal that works.

    This makes a repair expensive on purpose: the next portrait build is a
    full one (roughly an hour for 12 seasons). That is the honest cost of
    changing what the numbers mean.

    Nothing is lost. Every one of these is a pure cache gated on
    `if cache.exists()`; pull_fg_* regenerates the root through
    _bref_savant_*_merge, which is the path that drops the Savant percentile
    columns and promotes the Statcast rates.
    """
    processed = _HERE.parent / "data" / "processed"
    stale = (sorted(processed.glob("fg_batting_*.csv"))
             + sorted(processed.glob("fg_pitching_*.csv"))
             + sorted(processed.glob("*history*.csv"))
             + sorted(processed.glob("*prior*.parquet"))
             + sorted((processed / "portraits").glob("*.json")))
    if not stale:
        log.info("No derived pools to purge")
        return
    log.warning("%s %d derived pool file(s) — they will rebuild from the "
                "repaired database on the next portrait build",
                "Would purge" if dry_run else "Purging", len(stale))
    for p in stale:
        log.info("  %s %s", "would remove" if dry_run else "removing", p.name)
        if not dry_run:
            p.unlink()


def main(season: int, dry_run: bool) -> None:
    parquet = _RAW / f"statcast_{season}.parquet"
    if not parquet.exists():
        log.warning("No statcast cache for %d (%s) — skipped", season, parquet.name)
        return

    t0 = time.time()
    log.info("Season %d — loading %s", season, parquet.name)
    sc = pd.read_parquet(parquet)

    # min_pa/min_bf of 1: every stored row must be rewritten, so coverage is
    # maximised. Small samples produce noisy-but-honest rates; the tag gates
    # carry their own sample floors.
    batter_agg = pa._batter_aggregates(sc, min_pa=1)
    pitcher_agg = pa._pitcher_aggregates(sc, min_bf=1)

    _repair_table("player_batting", season, batter_agg, _BATTING_FIX,
                  ",".join(_BATTING_FIX) + ",sprint_speed", dry_run)
    _repair_table("player_pitching", season, pitcher_agg, _PITCHING_FIX,
                  ",".join(_PITCHING_FIX), dry_run)

    log.info("Season %d %s in %.1fs", season,
             "checked" if dry_run else "repaired", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("season", nargs="?", type=int,
                    help="single season; omit to repair every cached season")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    seasons = ([args.season] if args.season
               else sorted(int(p.stem.split("_")[1]) for p in _RAW.glob("statcast_*.parquet")
                           if p.stem.split("_")[1].isdigit()))
    for s in seasons:
        main(s, args.dry_run)
    # Always last: the pools must be dropped AFTER the rows they derive from
    # are correct, or they would simply be rebuilt stale.
    purge_derived_pools(args.dry_run)
