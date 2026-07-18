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

from team_portrait import build_team_portrait, PORTRAIT_SCHEMA_VERSION  # noqa: E402


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
_FP_TOP_N   = 3      # max fingerprint tags per unit


def _compute_fingerprints(league: dict[str, dict],
                          rosters: dict[str, dict]) -> None:
    """
    Mutates `league`: adds per-team `fingerprint[unit]` lists plus a
    top-level "_baselines" entry (league mean density per tag per unit).
    A team with an empty fingerprint is league-typical — that is itself
    information, not a failure.
    rosters — {team: {unit: [(tag_set, weight), ...]}} for breadth counts.
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

        for team in teams:
            ident = league[team]
            dens = (ident.get(unit) or {}).get("trait_density") or {}
            devs = []
            for tag in all_tags:
                d = float(dens.get(tag, 0.0)) - base[tag]
                if abs(d) < _FP_MIN_DEV:
                    continue
                roster = (rosters.get(team) or {}).get(unit) or []
                qual = [(tags, w) for tags, w in roster if w >= w_floor]
                carriers = sum(1 for tags, _ in qual if tag in tags)
                devs.append({
                    "tag":        tag,
                    "density":    round(float(dens.get(tag, 0.0)), 4),
                    "baseline":   round(base[tag], 4),
                    "deviation":  round(d, 4),
                    "carriers":   carriers,
                    "qualifiers": len(qual),
                })
            devs.sort(key=lambda x: -abs(x["deviation"]))
            ident.setdefault("fingerprint", {})[unit] = devs[:_FP_TOP_N]

    league["_baselines"] = baselines


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
                     float(pl.get(weight_key) or 0))
                    for pl in (players.get(players_key) or [])
                ]
                for unit, players_key, weight_key, _ in _FP_UNITS
            }

    _compute_fingerprints(league, rosters)

    out_path = Path(str(_LEAGUE_CACHE).format(season=season))
    out_path.write_text(json.dumps(league, indent=2, default=str))
    n_teams = sum(1 for k in league if not k.startswith("_"))
    print(f"\nWrote {n_teams}/30 teams (+fingerprints) to {out_path} "
          f"in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2023
    main(season)
