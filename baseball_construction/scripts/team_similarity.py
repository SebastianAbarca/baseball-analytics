"""
team_similarity.py — which teams a unit plays like, from the measurements.

A team used to be compared through its tag densities: the share of its
workload carried by players over each tag's line. That throws away most of
what was measured (an 84th-percentile chaser and a league-average one both
count as "no tag") and weights each skill by how many tags happen to describe
it. A unit is now a vector of the measurements themselves:

  for every metric, three numbers over the unit's players, weighted by PA/BF
      mean  — where the unit sits
      hi    — share of the workload at or above the 80th percentile
      lo    — share at or below the 20th
  the tails keep the roster's SHAPE: four sluggers and five slap hitters
  average to the same mean as nine average hitters, but not the same tails.

Each feature is z-scored across the season's 30 teams. Features are grouped
into baseball domains (power, contact, discipline...) and every domain gets
the same total weight however many metrics describe it — equal weights are
the starting point, not a finding. A feature counts for a team only when at
least half the unit's workload was measured for it; similarity is the cosine
over the features both teams have.

Measured on 2015-26 against the tag-density comparison (prototype,
2026-10-24): a unit's own next season lands in its top 3 comps for 68% of
offenses (tags 41%), 65% of rotations (65%) and 55% of bullpens (37%).

Deployment — how the club USED the unit (workload concentration, starts,
leverage) — answers a different question and carries little year to year
(14-21% self-retrieval), so it is a separate comparison, never mixed in.

Not in the portrait build fingerprint (unlike build_league_identity.py), so
tuning the comparison re-runs the league stage only.
"""
from __future__ import annotations

import collections
from typing import Optional

import numpy as np

METHOD = "hybrid-v1"
MIN_COVERAGE = 0.5
TAIL_HI, TAIL_LO = 80.0, 20.0
TOP_N = 3

# domain -> [(source, key)]. Sources: m = gate_inputs.metrics (league
# percentiles), gi = a gate_inputs field, adv = advancement,
# pool:<frame> = the pitcher's row from that league pool.
HITTER_DOMAINS: dict[str, list[tuple[str, str]]] = {
    "power":      [("m", "Barrel_pct_pct"), ("m", "HardHit_pct_pct"), ("m", "ISO_pct"),
                   ("m", "EV_avg_pct")],
    "contact":    [("m", "K_pct_pct"), ("m", "Contact_pct_pct"), ("m", "Whiff_pct_pct")],
    "discipline": [("m", "OSwing_pct_pct"), ("m", "ZSwing_pct_pct"), ("m", "BB_pct_pct"),
                   ("m", "FPS_pct_pct")],
    "direction":  [("m", "Pull_pct_pct"), ("m", "Oppo_pct_pct"), ("m", "BatGB_pct_pct"),
                   ("m", "GapTend_pct_pct")],
    "running":    [("m", "sprint_speed_pct"), ("gi", "attempt_rate_pct"),
                   ("adv", "xbt_pct")],
}
PITCHER_DOMAINS: dict[str, list[tuple[str, str]]] = {
    "stuff":     [("m", "avg_velo_pct"), ("m", "AvgSpinRate_pct"),
                  ("m", "Whiff_pct_pitch_pct"), ("m", "SwStr_per_velo_pct")],
    "outcomes":  [("m", "K_pct_pitch_pct"), ("m", "BB_pct_pitch_pct"),
                  ("m", "GB_pct_pitch_pct"), ("m", "HardHit_allowed_pct")],
    "command":   [("m", "Zone_pct_pct"), ("m", "FirstPitchStrike_pct"), ("m", "CSW_pct_pct")],
    "deception": [("m", "TunnelScore_pct"), ("pool:league_spin", "spin_efficiency_pct"),
                  ("pool:league_mechanics", "extension_pct")],
    "mix":       [("m", "ArsenalDiversity_pct"),
                  ("pool:league_sequencing", "entropy_ratio_pct"),
                  ("pool:league_sequencing", "count_shift_pct")],
}
# unit -> (players key, weight key, domains)
UNITS = {
    "offense":  ("hitters",      "pa", HITTER_DOMAINS),
    "rotation": ("starters",     "bf", PITCHER_DOMAINS),
    "bullpen":  ("bullpen_arms", "bf", PITCHER_DOMAINS),
}


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


def _value(player: dict, src: str, key: str) -> Optional[float]:
    gi = player.get("gate_inputs") or {}
    if src == "m":
        return _num((gi.get("metrics") or {}).get(key))
    if src == "gi":
        return _num(gi.get(key))
    if src == "adv":
        return _num((gi.get("advancement") or {}).get(key))
    if src.startswith("pool:"):
        return _num(((gi.get("pool_rows") or {}).get(src[5:]) or {}).get(key))
    raise ValueError(src)


def style_features(players: list[dict], wkey: str,
                   domains: dict) -> dict[tuple, float]:
    """{(domain, metric, stat): value}; NaN where under MIN_COVERAGE of the
    unit's workload was measured."""
    total = sum(float(p.get(wkey) or 0) for p in players)
    out: dict[tuple, float] = {}
    for dom, items in domains.items():
        for src, key in items:
            have = [(v, w) for p in players
                    if (w := float(p.get(wkey) or 0)) > 0
                    and (v := _value(p, src, key)) is not None]
            ws = sum(w for _, w in have)
            ok = total > 0 and ws / total >= MIN_COVERAGE
            out[(dom, key, "mean")] = sum(v * w for v, w in have) / ws if ok else np.nan
            out[(dom, key, "hi")] = sum(w for v, w in have if v >= TAIL_HI) / ws if ok else np.nan
            out[(dom, key, "lo")] = sum(w for v, w in have if v <= TAIL_LO) / ws if ok else np.nan
    return out


def _wmean(pairs) -> float:
    pairs = [(a, b) for a, b in pairs if a is not None and b > 0]
    ws = sum(b for _, b in pairs)
    return sum(a * b for a, b in pairs) / ws if ws else np.nan


def deployment_features(players: list[dict], unit: str) -> dict[tuple, float]:
    """How the club used the unit: workload concentration and role shape."""
    wkey = UNITS[unit][1]
    ws = sorted((float(p.get(wkey) or 0) for p in players), reverse=True)
    total = sum(ws)
    out: dict[tuple, float] = {}
    if total <= 0:
        return out
    if unit == "offense":
        out[("deployment", "top9_share")] = sum(ws[:9]) / total
        out[("deployment", "bench_share")] = sum(w for w in ws if w < 200) / total
        out[("deployment", "positions_per_player")] = _wmean(
            ((((p.get("gate_inputs") or {}).get("positions") or {}).get("positions_played")),
             float(p.get("pa") or 0)) for p in players)
        skew = []
        for p in players:
            sp = (p.get("gate_inputs") or {}).get("splits") or {}
            lhp = (sp.get("vs_lhp") or {}).get("pa") or 0
            rhp = (sp.get("vs_rhp") or {}).get("pa") or 0
            if lhp + rhp >= 30:
                skew.append((lhp / (lhp + rhp), lhp + rhp))
        if skew:
            mu = _wmean(skew)
            out[("deployment", "platoon_spread")] = _wmean((abs(a - mu), b) for a, b in skew)
    elif unit == "rotation":
        out[("deployment", "top5_share")] = sum(ws[:5]) / total
        out[("deployment", "bf_per_start")] = _wmean(
            (float(p.get("bf") or 0) / p["gs"], float(p["gs"])) for p in players if p.get("gs"))
        out[("deployment", "starters_used")] = float(len(players))
    else:
        out[("deployment", "top3_share")] = sum(ws[:3]) / total
        out[("deployment", "bf_per_game")] = _wmean(
            (float(p.get("bf") or 0) / p["g"], float(p.get("bf") or 0)) for p in players if p.get("g"))
        lev = [(_num((p.get("gate_inputs") or {}).get("leverage")), float(p.get("bf") or 0))
               for p in players]
        mu = _wmean(lev)
        if np.isfinite(mu):
            out[("deployment", "leverage_spread")] = _wmean(
                (None if a is None else abs(a - mu), b) for a, b in lev)
        out[("deployment", "arms_used")] = float(len(players))
    return out


def _standardise(feats: dict[str, dict], budget: bool) -> tuple[list[str], np.ndarray]:
    """Teams x features, z-scored across teams; with `budget`, each domain's
    features share one unit of weight (1/sqrt(n) each, so a domain's squared
    contribution is the same however many metrics it has)."""
    keys = sorted({k for f in feats.values() for k in f})
    teams = sorted(feats)
    X = np.array([[feats[t].get(k, np.nan) for k in keys] for t in teams], float)
    with np.errstate(all="ignore"):
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~(sd > 0)] = np.nan
    Z = (X - mu) / sd
    if budget:
        per = collections.Counter(k[0] for k in keys)
        Z = Z * np.array([1 / np.sqrt(per[k[0]]) for k in keys])
    return teams, Z


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _similarities(teams: list[str], Z: np.ndarray) -> dict[str, dict[str, float]]:
    return {t: {o: _cos(Z[i], Z[j]) for j, o in enumerate(teams) if j != i}
            for i, t in enumerate(teams)}


def _uniqueness(sims: dict[str, dict[str, float]]) -> dict[str, dict]:
    """Closest-comp strength, banded against the unit's own spread (see
    build_league_identity's history: a mean-similarity score was not
    comparable across seasons; the nearest comp is)."""
    closest = {t: max(s.values()) for t, s in sims.items() if s}
    vals = sorted(closest.values())
    if not vals:
        return {}
    q1, q3 = vals[len(vals) // 4], vals[(3 * len(vals)) // 4]
    med = vals[len(vals) // 2]
    order = sorted(closest, key=lambda t: closest[t])
    rank = {t: i for i, t in enumerate(order, 1)}
    out = {}
    for t, s in sims.items():
        c = closest.get(t, 0.0)
        out[t] = {
            "closest": round(c, 4),
            "band": "close" if c >= q3 else "loose" if c <= q1 else "fair",
            "unit_q1": round(q1, 4), "unit_q3": round(q3, 4),
            "unit_median": round(med, 4),
            "rank": rank.get(t), "of": len(order),
            "mean_sim": round(sum(s.values()) / len(s), 4) if s else None,
            "method": METHOD,
        }
    return out


def compute_positions(league: dict[str, dict], players: dict[str, dict]) -> None:
    """
    Mutates `league`: per team, `neighbours[unit]` and `uniqueness[unit]`
    from the style comparison, and `deployment_neighbours[unit]` from the
    deployment one. `players` is {team: portrait["players"]}.
    """
    teams = [t for t in league if not t.startswith("_") and t in players]
    if len(teams) < 2:
        return
    for unit, (grp, wkey, domains) in UNITS.items():
        style = {t: style_features(players[t].get(grp) or [], wkey, domains) for t in teams}
        order, Z = _standardise(style, budget=True)
        sims = _similarities(order, Z)
        uq = _uniqueness(sims)
        dep = {t: deployment_features(players[t].get(grp) or [], unit) for t in teams}
        d_order, DZ = _standardise(dep, budget=False)
        d_sims = _similarities(d_order, DZ)
        for t in teams:
            near = sorted(sims[t].items(), key=lambda kv: -kv[1])[:TOP_N]
            league[t].setdefault("neighbours", {})[unit] = [
                {"team": o, "similarity": round(s, 4)} for o, s in near]
            league[t].setdefault("uniqueness", {})[unit] = uq.get(t, {})
            d_near = sorted(d_sims.get(t, {}).items(), key=lambda kv: -kv[1])[:TOP_N]
            league[t].setdefault("deployment_neighbours", {})[unit] = [
                {"team": o, "similarity": round(s, 4)} for o, s in d_near]
    league["_similarity"] = {
        "method": METHOD, "min_coverage": MIN_COVERAGE, "tails": [TAIL_LO, TAIL_HI],
        "domains": {u: {d: [k for _, k in items] for d, items in doms.items()}
                    for u, (_, _, doms) in UNITS.items()},
    }
