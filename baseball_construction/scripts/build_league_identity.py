"""
build_league_identity.py — Build team_identity for all 30 teams in a season
and write a league-wide summary JSON for the Compare tab's League Identity Board.

Reuses the same on-disk portrait cache as the dashboard (data/processed/portraits/),
so teams already built at the current PORTRAIT_SCHEMA_VERSION are skipped.

Usage:
    python3 scripts/build_league_identity.py 2023
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "modules"))
sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd

from team_portrait import (build_team_portrait, PORTRAIT_SCHEMA_VERSION,  # noqa: E402
                           TAG_POPULATION, POP_ALL, TAG_KINDS, kind_of,
                           TAG_EXCLUSIVE_GROUPS, portrait_is_current)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict("records")
        return super().default(obj)

MLB_TEAMS = [
    "ATH", "ATL", "AZ", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS",
    "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]

_PORTRAIT_CACHE = _HERE.parent / "data" / "processed" / "portraits"
_PORTRAIT_CACHE.mkdir(parents=True, exist_ok=True)
_LEAGUE_CACHE = _HERE.parent / "data" / "processed" / f"league_identity_{{season}}.json"


def _cache_path(team: str, season: int) -> Path:
    return _PORTRAIT_CACHE / f"{team}_{season}.json"


# ---------------------------------------------------------------------------
# Fingerprint layer — what makes each team THIS team
# ---------------------------------------------------------------------------
# A team's identity is where it DEVIATES from the league, weighted by how
# much of the roster commits to it:
#   baseline  — league mean trait density per tag (equal team weight)
#   deviation — team density − baseline (percentage points of PA/BF share)
#   breadth   — carriers/qualifiers among roster regulars (cohesion: a built
#               identity is broad; one superstar skewing a density is not)

_FP_UNITS = [
    # (identity unit, players key, weight key, qualifying weight floor)
    ("offense",  "hitters",      "pa", 200),
    ("rotation", "starters",     "bf", 100),
    ("bullpen",  "bullpen_arms", "bf", 60),
]
_FP_MIN_DEV = 0.08   # |deviation| floor to count as identity-defining
# A team needs a real eligible pool before a tag can define it. Without this,
# a reliever who is the only arm with tempo data reads as "slow pitcher, 84%
# above league" off a population of one. Either the pool is big enough to mean
# something, or more than one player carries the tag.
_FP_MIN_QUALIFIERS = 3
_FP_MIN_CARRIERS_SMALL_POOL = 2

# ── Per-kind fingerprint ────────────────────────────────────────────────────
# The fingerprint is organised BY KIND (see team_portrait.TAG_KIND) rather
# than as one ranked list. Ranking everything together let an attribute nobody
# chose outrank a skill: HOU 2025's offense led with "left-handed hitter" and
# "right-handed hitter", i.e. it reported that they bat right-handed. Filing
# each tag under its kind keeps that fact — a lineup's handedness IS worth
# knowing — without letting it displace how the team actually plays.
#
# Every kind gets a row, so the card is a fixed skeleton and teams can be read
# against each other line by line. A kind where nothing clears the bar is
# reported as league-typical, which is information rather than a gap. A kind
# with no vocabulary in that unit at all is reported as unmeasured, which is
# a different statement.
#
# Selection is by |z| — deviation over the tag's own cross-team spread —
# because raw deviations are not comparable between tags. The gate is a z
# floor rather than the old winner-take-all top-3: with only three slots the
# floor barely mattered, but per kind it does the work. Measured across 360
# team-seasons, z >= 1.5 with a cap of 2 yields a median 4 tags per unit
# against the 3 the old design showed — the same density, now structured.
_FP_MIN_Z = 1.5
_FP_PER_KIND = 2
_FP_PER_KIND_OVERRIDE: dict[str, int] = {}

_FP_TYPICAL = "league-typical"    # vocabulary exists, team is unremarkable
_FP_UNMEASURED = "not measured"   # kind has no vocabulary for this unit


def _compute_fingerprints(league: dict[str, dict],
                          rosters: dict[str, dict]) -> None:
    """
    Mutates `league`: adds per-team `fingerprint[unit]` lists plus a
    top-level "_baselines" entry (league mean density per tag per unit).
    A team with an empty fingerprint is league-typical — that is itself
    information, not a failure.
    rosters — {team: {unit: [(tag_set, weight, population_set), ...]}}.
    """
    baselines: dict[str, dict[str, float]] = {}
    teams = [t for t in league if not t.startswith("_")]

    for unit, _players_key, _weight_key, w_floor in _FP_UNITS:
        all_tags: set[str] = set()
        for team in teams:
            all_tags |= set((league[team].get(unit) or {})
                            .get("trait_density") or {})
        base: dict[str, float] = {}
        for tag in all_tags:
            vals = [float(((league[t].get(unit) or {}).get("trait_density") or {})
                          .get(tag, 0.0)) for t in teams]
            base[tag] = sum(vals) / len(vals) if vals else 0.0
        baselines[unit] = {k: round(v, 4) for k, v in
                           sorted(base.items(), key=lambda kv: -kv[1])}

        # Spread of each tag ACROSS teams. Raw deviations are not comparable
        # between tags: a catcher tag is measured on ~3 players per team and
        # swings ±0.8, while a lineup-wide tag moves ±0.15. Ranking on raw
        # deviation therefore hands every fingerprint to the smallest
        # populations. Dividing by the tag's own cross-team spread asks the
        # only question that transfers: how unusual is this team FOR THIS TAG.
        sd: dict[str, float] = {}
        for tag in all_tags:
            vals = [float(((league[t].get(unit) or {}).get("trait_density") or {})
                          .get(tag, 0.0)) for t in teams]
            mu = sum(vals) / len(vals)
            sd[tag] = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5

        # Which kinds have any vocabulary at all in this unit — "no vocabulary
        # here" is a different statement from "unremarkable here".
        unit_kinds = {k for k in (kind_of(t) for t in all_tags) if k}

        for team in teams:
            ident = league[team]
            dens = (ident.get(unit) or {}).get("trait_density") or {}
            roster = (rosters.get(team) or {}).get(unit) or []
            devs = []
            for tag in all_tags:
                d = float(dens.get(tag, 0.0)) - base[tag]
                if abs(d) < _FP_MIN_DEV or not sd[tag]:
                    continue
                if abs(d / sd[tag]) < _FP_MIN_Z:
                    continue   # not unusual enough FOR THIS TAG to mean anything
                # Breadth is counted within the tag's ELIGIBLE population —
                # "0 of 13 regulars are good blockers" is meaningless when 11
                # of those 13 never crouch behind the plate.
                pop = TAG_POPULATION.get(tag, POP_ALL)
                qual = [r for r in roster
                        if r[1] >= w_floor and (pop == POP_ALL or pop in r[2])]
                carriers = sum(1 for r in qual if tag in r[0])
                if not carriers:
                    continue   # a fingerprint names what a team HAS
                if (len(qual) < _FP_MIN_QUALIFIERS
                        and carriers < _FP_MIN_CARRIERS_SMALL_POOL):
                    continue   # population of one is coverage, not identity
                devs.append({
                    "tag":        tag,
                    "kind":       kind_of(tag),
                    "density":    round(float(dens.get(tag, 0.0)), 4),
                    "baseline":   round(base[tag], 4),
                    "deviation":  round(d, 4),
                    "z":          round(d / sd[tag], 3),
                    "carriers":   carriers,
                    "qualifiers": len(qual),
                })
            devs.sort(key=lambda x: -abs(x["z"]))

            # File each surviving tag under its kind, most distinctive first,
            # capped so one loud kind cannot fill the card.
            # Of a two-ended axis, keep the end the team actually HAS. Both
            # ends carry density at team level, so a lineup reads as both
            # "right-handed hitter +26" and "left-handed hitter -22" — the same
            # fact, once as a presence and once as an absence. Ranking by |z|
            # alone picked whichever was numerically louder, which named what
            # the team is NOT. Prefer the positive deviation; fall back to |z|
            # when the pair does not straddle zero.
            by_tag = {e["tag"]: e for e in devs}
            dropped: set[str] = set()
            for grp in TAG_EXCLUSIVE_GROUPS:
                members = [by_tag[t] for t in grp if t in by_tag]
                if len(members) < 2:
                    continue
                # A presence beats an absence: "right-handed lineup +26" says
                # what this team IS, "left-handed -22" says what it is not.
                # Only when no member is positive does magnitude decide.
                positives = [m for m in members if m["deviation"] >= 0]
                keep = max(positives or members, key=lambda x: abs(x["z"]))
                dropped |= {m["tag"] for m in members if m["tag"] != keep["tag"]}

            by_kind: dict[str, list] = {}
            for e in devs:
                k = e.get("kind")
                if not k or e["tag"] in dropped:
                    continue
                bucket = by_kind.setdefault(k, [])
                cap = _FP_PER_KIND_OVERRIDE.get(k, _FP_PER_KIND)
                if len(bucket) < cap:
                    bucket.append(e)

            # Every kind gets a row so the card is a fixed skeleton and teams
            # read against each other line by line.
            rows = []
            for k in TAG_KINDS:
                if k not in unit_kinds:
                    rows.append({"kind": k, "state": _FP_UNMEASURED, "tags": []})
                elif by_kind.get(k):
                    rows.append({"kind": k, "state": "defined", "tags": by_kind[k]})
                else:
                    rows.append({"kind": k, "state": _FP_TYPICAL, "tags": []})
            ident.setdefault("fingerprint", {})[unit] = rows

    league["_baselines"] = baselines




def main(season: int, force: bool = False) -> int:
    """
    Build (or reuse) a season's 30 portraits and write its league identity.

    force rebuilds every portrait even when it reads as current. The nightly
    job needs that: the database totals a portrait is built from are not in
    the fingerprint, so a night of new games leaves yesterday's portrait
    looking current. If a forced build fails, the existing portrait is kept
    for the identity and the failure is counted.

    Returns the number of teams whose build failed.
    """
    league: dict[str, dict] = {}
    rosters: dict[str, dict] = {}
    team_players: dict[str, dict] = {}
    t0 = time.time()
    failed = 0

    for i, team in enumerate(MLB_TEAMS, 1):
        cp = _cache_path(team, season)
        portrait = None
        cached = None

        if cp.exists():
            try:
                cached = json.loads(cp.read_text())
                # Schema AND build fingerprint — a portrait can carry the
                # current schema and still have been built against different
                # pools or gates, which is exactly how 31 stale portraits and
                # HOU_2025 were skipped as cache hits during earlier rebuilds.
                if portrait_is_current(cached) and not force:
                    portrait = cached
            except Exception:
                cached = None

        if portrait is None:
            print(f"[{i}/30] Building {team} {season}...", flush=True)
            try:
                portrait = build_team_portrait(team, season)
            except Exception as exc:
                print(f"  FAILED {team}: {exc}", flush=True)
                failed += 1
                if not (force and cached):
                    continue
                portrait = cached          # forced rebuild failed: keep the old one
                print(f"  kept the existing {team} {season} portrait", flush=True)
            if portrait.get("error"):
                print(f"  SKIP {team}: {portrait['error']}", flush=True)
                failed += 1
                continue
            if portrait is not cached:
                try:
                    cp.write_text(json.dumps(portrait, cls=_NumpyEncoder, default=str))
                except Exception as exc:
                    print(f"  WARN: could not cache {team}: {exc}", flush=True)
        else:
            print(f"[{i}/30] {team} {season} — cache hit", flush=True)

        identity = portrait.get("team_identity") or {}
        if identity:
            league[team] = identity
            # Roster tag sets for fingerprint breadth (cohesion) counts
            players = portrait.get("players") or {}
            team_players[team] = players
            rosters[team] = {
                unit: [
                    ({t.get("tag") for t in (pl.get("traits") or [])},
                     float(pl.get(weight_key) or 0),
                     set(pl.get("populations") or [POP_ALL]))
                    for pl in (players.get(players_key) or [])
                ]
                for unit, players_key, weight_key, _ in _FP_UNITS
            }

    _compute_fingerprints(league, rosters)
    # Comps come from the measurements, not the tags — see team_similarity.
    # It lives outside this file so tuning it does not stale every portrait.
    from team_similarity import compute_positions
    compute_positions(league, team_players)

    out_path = Path(str(_LEAGUE_CACHE).format(season=season))
    out_path.write_text(json.dumps(league, indent=2, default=str))
    n_teams = sum(1 for k in league if not k.startswith("_"))
    print(f"\nWrote {n_teams}/30 teams (+fingerprints) to {out_path} "
          f"in {time.time()-t0:.1f}s")
    return failed


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    main(season)
