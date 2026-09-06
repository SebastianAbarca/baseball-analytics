"""
rederive_traits.py — Recompute tags on stored portraits, without rebuilding.

WHY
---
Traits are a PURE FUNCTION of inputs the portrait already holds. Changing a
gate — the reliability floor, a percentile threshold, `zone hunter`'s
definition, which tags suppress which — moves no metric, yet used to require
a full rebuild: an hour of re-querying Supabase, re-reading Statcast parquets
and re-ranking percentiles that had not changed. Two of the three rebuilds in
the session that prompted this were exactly that.

Profiling a warm portrait showed why the rebuild is expensive and how little
of it is the analytics: 7.4s per team, of which `query_estimated_service_time`
alone is 2.9s and the actual trait classification ~0.3s.

So portraits now store a `gate_inputs` block per player (team_portrait) and
this script replays it. What it does NOT need: the database, the Statcast
cache, the normalization pools, service time, park factors, tunneling, spin.
It reloads only the five per-season pitcher league frames, which are already
cached as parquet in modules/processed.

WHAT IT REWRITES
----------------
    players[*].traits, .spectrum, .populations, .limited_sample
    team_identity          (trait densities are recomputed from the new tags)
    build_fingerprint      (stamped to the current gates)

Everything else in the portrait is left exactly as built.

USAGE
-----
    python rederive_traits.py --dry-run        # report tag deltas, write nothing
    python rederive_traits.py 2023             # one season
    python rederive_traits.py                  # every season on disk

Follow with build_league_identity.py per season to refresh fingerprints and
neighbours from the new tags (that step is already cache-fast).
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))

import pandas as pd                                    # noqa: E402
import team_portrait as tp                             # noqa: E402
from hitter_traits import build_hitter_traits          # noqa: E402
from pitcher_traits import build_pitcher_traits        # noqa: E402
from reliability import (is_limited_sample,            # noqa: E402
                         LIMITED_SAMPLE_PA, LIMITED_SAMPLE_BF)

log = logging.getLogger("rederive")

_PORTRAITS = _HERE.parent / "data" / "processed" / "portraits"
_POOLS = _HERE.parent / "modules" / "processed"

# The per-season league frames build_pitcher_traits ranks against. Cached as
# parquet by the original build, so re-deriving costs one read per season.
_LEAGUE_FRAMES = {
    "league_mechanics":  "mechanics_league_{season}.parquet",
    "league_sequencing": "sequencing_league_{season}.parquet",
    "league_discipline": "plate_discipline_league_{season}.parquet",
    "league_tunnel":     "tunnel_league_{season}.parquet",
    "league_spin":       "spin_league_{season}.parquet",
}


def _load_league_frames(season: int) -> dict:
    out = {}
    for key, pattern in _LEAGUE_FRAMES.items():
        path = _POOLS / pattern.format(season=season)
        try:
            out[key] = pd.read_parquet(path) if path.exists() else None
        except Exception as exc:
            log.warning("  %s unreadable (%s) — tags resting on it will not fire",
                        path.name, exc)
            out[key] = None
    missing = [k for k, v in out.items() if v is None]
    if missing:
        log.warning("  missing league pools: %s", ", ".join(sorted(missing)))
    return out


def _rederive_portrait(portrait: dict, frames: dict) -> tuple[dict, collections.Counter]:
    """Replay the gate inputs. Returns (portrait, tag delta counter)."""
    delta: collections.Counter = collections.Counter()
    players = portrait.get("players") or {}
    role_pool = ((portrait.get("bullpen_collective") or {}).get("role_pool")) or {}

    for h in players.get("hitters") or []:
        gi = h.get("gate_inputs")
        if not gi:
            continue
        before = {t["tag"] for t in (h.get("traits") or [])}
        traits, spectrum = build_hitter_traits(
            metrics=gi.get("metrics") or {},
            sprint_speed_raw=gi.get("sprint_speed_raw"),
            sb=gi.get("sb") or 0,
            cs=gi.get("cs") or 0,
            opportunities=gi.get("opportunities") or 0,
            attempt_rate_pct=gi.get("attempt_rate_pct"),
            season_metrics=gi.get("season_metrics") or {},
            splits=gi.get("splits"),
            catching=gi.get("catching"),
            advancement=gi.get("advancement"),
            positions=gi.get("positions"),
            pa=gi.get("pa"),
        )
        h["traits"], h["spectrum"] = traits, spectrum
        h["limited_sample"] = is_limited_sample(gi.get("pa"), LIMITED_SAMPLE_PA)
        after = {t["tag"] for t in traits}
        for t in after - before:
            delta[f"+{t}"] += 1
        for t in before - after:
            delta[f"-{t}"] += 1

    for group, is_starter in (("starters", True), ("bullpen_arms", False)):
        for p in players.get(group) or []:
            gi = p.get("gate_inputs")
            if not gi:
                continue
            before = {t["tag"] for t in (p.get("traits") or [])}
            traits = build_pitcher_traits(
                p.get("player_id"),
                frames.get("league_mechanics"),
                frames.get("league_sequencing"),
                frames.get("league_discipline"),
                frames.get("league_tunnel"),
                frames.get("league_spin"),
                arsenal_profile=p.get("arsenal_profile"),
                metrics=gi.get("metrics") or {},
                bf=gi.get("bf"), g=gi.get("g"), gs=gi.get("gs"),
                is_starter=gi.get("is_starter", is_starter),
                role_pool=role_pool,
                leverage=gi.get("leverage"),
                splits=gi.get("splits"),
                tempo=gi.get("tempo"),
                runner_control=gi.get("runner_control"),
            )
            p["traits"] = traits
            p["limited_sample"] = is_limited_sample(gi.get("bf"), LIMITED_SAMPLE_BF)
            after = {t["tag"] for t in traits}
            for t in after - before:
                delta[f"+{t}"] += 1
            for t in before - after:
                delta[f"-{t}"] += 1

    # Team identity is a pure aggregation of the tags, so it has to follow.
    try:
        portrait["team_identity"] = tp._build_team_identity(
            players.get("hitters") or [],
            players.get("starters") or [],
            portrait.get("bullpen_collective") or {},
            portrait.get("philosophy_metrics") or {},
            bullpen_arms=players.get("bullpen_arms") or [],
        )
    except Exception as exc:
        log.warning("  team_identity recompute failed: %s", exc)

    portrait["build_fingerprint"] = tp.build_fingerprint()
    return portrait, delta


def main(season: int, dry_run: bool) -> None:
    files = sorted(_PORTRAITS.glob(f"*_{season}.json"))
    if not files:
        log.warning("No portraits for %d", season)
        return

    t0 = time.time()
    frames = _load_league_frames(season)
    total: collections.Counter = collections.Counter()
    touched = skipped = 0

    for f in files:
        portrait = json.loads(f.read_text())
        players = portrait.get("players") or {}
        if not any((p.get("gate_inputs") for grp in
                    ("hitters", "starters", "bullpen_arms")
                    for p in (players.get(grp) or []))):
            skipped += 1
            continue
        portrait, delta = _rederive_portrait(portrait, frames)
        total.update(delta)
        touched += 1
        if not dry_run:
            tmp = f.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(portrait, default=str))
            tmp.replace(f)

    verb = "would change" if dry_run else "rewrote"
    log.info("Season %d — %s %d portraits in %.1fs (%d skipped: no gate_inputs)",
             season, verb, touched, time.time() - t0, skipped)
    if total:
        print(f"\n  tag changes, season {season}:")
        for tag, n in total.most_common(14):
            print(f"    {tag:<32}{n:>6}")
    else:
        print(f"  season {season}: no tag changed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("season", nargs="?", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seasons = ([args.season] if args.season else
               sorted({int(p.stem.split("_")[1]) for p in _PORTRAITS.glob("*_*.json")}))
    for s in seasons:
        main(s, args.dry_run)
