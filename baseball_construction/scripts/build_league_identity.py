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

import numpy as np
import pandas as pd

from team_portrait import (build_team_portrait, PORTRAIT_SCHEMA_VERSION,  # noqa: E402
                           TAG_POPULATION, POP_ALL)


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
_FP_TOP_N   = 3      # max fingerprint tags per unit


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

        for team in teams:
            ident = league[team]
            dens = (ident.get(unit) or {}).get("trait_density") or {}
            roster = (rosters.get(team) or {}).get(unit) or []
            devs = []
            for tag in all_tags:
                d = float(dens.get(tag, 0.0)) - base[tag]
                if abs(d) < _FP_MIN_DEV or not sd[tag]:
                    continue
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
                    "density":    round(float(dens.get(tag, 0.0)), 4),
                    "baseline":   round(base[tag], 4),
                    "deviation":  round(d, 4),
                    "z":          round(d / sd[tag], 3),
                    "carriers":   carriers,
                    "qualifiers": len(qual),
                })
            devs.sort(key=lambda x: -abs(x["z"]))
            ident.setdefault("fingerprint", {})[unit] = devs[:_FP_TOP_N]

    league["_baselines"] = baselines


_POS_TOP_N = 3   # neighbours reported per unit


def _compute_positions(league: dict[str, dict]) -> None:
    """
    Mutates `league`: adds per-team `neighbours[unit]` and `uniqueness[unit]`.

    This is what replaced the headline labels. A label assigned a team to a
    bucket using a hand-chosen cutoff ("power_share >= 0.40 → Power-Driven
    Lineup"); position says where the team actually sits relative to everyone
    else, with no thresholds at all.

    Each team is a vector of per-tag DEVIATIONS from the league mean (the same
    quantity the fingerprint reports, but over the full tag vocabulary rather
    than the top 3). Similarity is cosine — it asks whether two teams deviate
    in the same DIRECTION, not whether they deviate by the same amount, so a
    mild contact team and an extreme one read as similar in kind.

    `uniqueness` is 1 − (mean cosine similarity to every other team), scaled to
    0–100 across the league: a team nobody resembles scores high. Both are
    descriptive positions, not rankings of quality.
    """
    teams = [t for t in league if not t.startswith("_")]
    baselines = league.get("_baselines") or {}

    for unit, _pk, _wk, _wf in _FP_UNITS:
        base = baselines.get(unit) or {}
        tags = sorted(base)
        if not tags or len(teams) < 2:
            continue

        vecs: dict[str, np.ndarray] = {}
        for t in teams:
            dens = (league[t].get(unit) or {}).get("trait_density") or {}
            vecs[t] = np.array([float(dens.get(tag, 0.0)) - float(base[tag])
                                for tag in tags], dtype=float)

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

        sims = {t: {o: _cos(vecs[t], vecs[o]) for o in teams if o != t}
                for t in teams}
        mean_sim = {t: (sum(s.values()) / len(s)) if s else 0.0
                    for t, s in sims.items()}

        raw = {t: 1.0 - mean_sim[t] for t in teams}
        lo, hi = min(raw.values()), max(raw.values())
        span = (hi - lo) or 1.0

        for t in teams:
            near = sorted(sims[t].items(), key=lambda kv: -kv[1])[:_POS_TOP_N]
            league[t].setdefault("neighbours", {})[unit] = [
                {"team": o, "similarity": round(s, 4)} for o, s in near
            ]
            league[t].setdefault("uniqueness", {})[unit] = {
                "score":    round((raw[t] - lo) / span * 100.0, 1),
                "mean_sim": round(mean_sim[t], 4),
            }


def main(season: int):
    league: dict[str, dict] = {}
    rosters: dict[str, dict] = {}
    t0 = time.time()

    for i, team in enumerate(MLB_TEAMS, 1):
        cp = _cache_path(team, season)
        portrait = None

        if cp.exists():
            try:
                cached = json.loads(cp.read_text())
                if cached.get("schema_version") == PORTRAIT_SCHEMA_VERSION:
                    portrait = cached
            except Exception:
                pass

        if portrait is None:
            print(f"[{i}/30] Building {team} {season}...", flush=True)
            try:
                portrait = build_team_portrait(team, season)
            except Exception as exc:
                print(f"  FAILED {team}: {exc}", flush=True)
                continue
            if portrait.get("error"):
                print(f"  SKIP {team}: {portrait['error']}", flush=True)
                continue
            # Save to disk cache so the dashboard picks it up too
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
    _compute_positions(league)

    out_path = Path(str(_LEAGUE_CACHE).format(season=season))
    out_path.write_text(json.dumps(league, indent=2, default=str))
    n_teams = sum(1 for k in league if not k.startswith("_"))
    print(f"\nWrote {n_teams}/30 teams (+fingerprints) to {out_path} "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    main(season)
