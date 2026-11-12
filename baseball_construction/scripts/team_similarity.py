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
into baseball domains (swing decisions, spray, arm...) and every domain gets
the same total weight however many metrics describe it — equal weights are
the starting point, not a finding. A feature counts for a team only when at
least half the unit's workload was measured for it; similarity is the cosine
over the features both teams have.

Two comparisons, never one score (split-v1, 2026-11-11):

  APPROACH ("plays like")    what the players choose or physically bring —
                             swing decisions, bat-to-ball, spray, tools,
                             running; arm, attack, mix, pitch shape
  OUTCOME ("produces like")  what happened — K/BB, contact quality, power,
                             production; misses, K/BB, contact allowed

The single score they replaced counted K%, BB% and CSW% twice (once as the
result, once through the process that produces it) and leaned on how GOOD a
unit was: its pairwise similarity tracked the gap in xwOBA at rho +0.23 to
+0.37. A raw outcome score is worse (+0.40 to +0.56) — it mostly pairs
equally good teams — so the outcome comparison is taken on SHAPE: every
outcome feature is regressed on the unit's quality (xwOBA, or xwOBA allowed)
across the season's teams and the residual is compared; quality itself is
reported beside the comps as `level`, not voted on.

Measured on 2015-26 (prototype, 2026-11-11), a unit's own next season in its
top 3 comps (random 10%), and how much similarity tracks quality:

                  offense   rotation   bullpen   tracks quality
  single score      81%       72%        62%      +0.23 .. +0.37
  approach          75%       72%        62%      +0.07 .. +0.13
  outcome (shape)   64%       47%        35%      -0.07 .. -0.05

The two lists share 13-23% of their top-3 comps, so each says something the
other does not. Pitch shape (fastball ride and run, separation from the out
pitch) is what lifts pitching approach from 59% / 54% to 72% / 62%.

Deployment — how the club USED the unit (workload concentration, starts,
leverage) — answers a third question and carries little year to year
(14-21% self-retrieval), so it is a separate comparison, never mixed in.

Not in the portrait build fingerprint (unlike build_league_identity.py), so
tuning the comparison re-runs the league stage only.
"""
from __future__ import annotations

import collections
from typing import Optional

import numpy as np

METHOD = "split-v1"   # explanations do not change the comps, so no bump
MIN_COVERAGE = 0.5
TAIL_HI, TAIL_LO = 80.0, 20.0
TOP_N = 3

# domain -> [(source, key)]. Sources: m = gate_inputs.metrics (league
# percentiles), gi = a gate_inputs field, adv = advancement,
# pool:<frame> = the pitcher's row from that league pool, shape = the
# pitcher's pitch shape (pitch_shape below), ranked among the season's arms.
#
# Team comps: approach and outcome (see the module docstring).
H_APPROACH: dict[str, list[tuple[str, str]]] = {
    "swing decisions": [("m", "OSwing_pct_pct"), ("m", "ZSwing_pct_pct"),
                        ("m", "FPS_pct_pct"), ("m", "PitchesPerPA_pct")],
    "bat-to-ball":     [("m", "Contact_pct_pct"), ("m", "Whiff_pct_pct")],
    "spray":           [("m", "Pull_pct_pct"), ("m", "Oppo_pct_pct"), ("m", "BatGB_pct_pct"),
                        ("m", "GapTend_pct_pct")],
    "tools":           [("m", "EV_avg_pct"), ("m", "sprint_speed_pct")],
    "running":         [("gi", "attempt_rate_pct"), ("adv", "xbt_pct")],
}
H_OUTCOME: dict[str, list[tuple[str, str]]] = {
    "strikeouts and walks": [("m", "K_pct_pct"), ("m", "BB_pct_pct")],
    "contact quality":      [("m", "Barrel_pct_pct"), ("m", "HardHit_pct_pct")],
    "power":                [("m", "ISO_pct"), ("m", "HR_FB_pct_pct"), ("m", "XB_pct_pct")],
    "production":           [("m", "xwOBA_pct"), ("m", "xBA_pct"), ("m", "OBP_SLG_gap_pct")],
}
P_APPROACH: dict[str, list[tuple[str, str]]] = {
    "arm":    [("m", "avg_velo_pct"), ("m", "AvgSpinRate_pct"),
               ("pool:league_spin", "spin_efficiency_pct"),
               ("pool:league_mechanics", "extension_pct")],
    "attack": [("m", "Zone_pct_pct"), ("m", "FirstPitchStrike_pct")],
    "mix":    [("m", "ArsenalDiversity_pct"), ("pool:league_sequencing", "entropy_ratio_pct"),
               ("pool:league_sequencing", "count_shift_pct"), ("m", "TunnelScore_pct")],
    "shape":  [("shape", "fb_ivb"), ("shape", "fb_run"), ("shape", "vert_sep"),
               ("shape", "horiz_sep")],
}
P_OUTCOME: dict[str, list[tuple[str, str]]] = {
    "misses":               [("m", "Whiff_pct_pitch_pct"), ("m", "CSW_pct_pct"),
                             ("m", "SwStr_per_velo_pct")],
    "strikeouts and walks": [("m", "K_pct_pitch_pct"), ("m", "BB_pct_pitch_pct")],
    "contact allowed":      [("m", "GB_pct_pitch_pct"), ("m", "HardHit_allowed_pct"),
                             ("m", "Barrel_allowed_pct"), ("m", "xwOBA_allowed_pct")],
}
# The quality an outcome comparison is taken net of, per unit (an outcome
# key; it is reported as `level` and dropped from the outcome cosine).
LEVEL = {"offense": "xwOBA_pct", "rotation": "xwOBA_allowed_pct",
         "bullpen": "xwOBA_allowed_pct"}

# Player comps (scout_comps) still use one combined vector over these; their
# split is the player side of the same change.
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
# What a HIGH value of each feature means for a team, and what a low one
# means — (name, high, low). Percentile directions follow the rank-invert
# tables in team_portrait (K% and walks/hard contact allowed are inverted:
# a high percentile is FEWER of them); V26 checks the two agree, so a phrase
# cannot say the opposite of the number behind it.
PHRASES: dict[str, tuple[str, str, str]] = {
    # hitters
    "Barrel_pct_pct":   ("barrels", "more barrels", "fewer barrels"),
    "HardHit_pct_pct":  ("hard contact", "more hard contact", "less hard contact"),
    "ISO_pct":          ("isolated power", "more isolated power", "less isolated power"),
    "EV_avg_pct":       ("exit velocity", "higher exit velocity", "lower exit velocity"),
    "K_pct_pct":        ("strikeouts", "fewer strikeouts", "more strikeouts"),
    "Contact_pct_pct":  ("contact", "more contact", "less contact"),
    "Whiff_pct_pct":    ("whiffs", "more whiffs", "fewer whiffs"),
    "OSwing_pct_pct":   ("chasing", "more chasing", "less chasing"),
    "ZSwing_pct_pct":   ("zone swings", "more zone swings", "fewer zone swings"),
    "BB_pct_pct":       ("walks", "more walks", "fewer walks"),
    "FPS_pct_pct":      ("first-pitch swings", "more first-pitch swings", "fewer first-pitch swings"),
    "Pull_pct_pct":     ("pulled balls", "more pulled balls", "fewer pulled balls"),
    "Oppo_pct_pct":     ("opposite-field balls", "more opposite-field balls", "fewer opposite-field balls"),
    "BatGB_pct_pct":    ("ground balls", "more ground balls", "fewer ground balls"),
    "GapTend_pct_pct":  ("gap hits", "more gap hits", "fewer gap hits"),
    "sprint_speed_pct": ("speed", "faster runners", "slower runners"),
    "attempt_rate_pct": ("steal attempts", "more steal attempts", "fewer steal attempts"),
    "xbt_pct":          ("extra-base attempts", "more extra-base attempts", "fewer extra-base attempts"),
    "PitchesPerPA_pct": ("pitches per PA", "longer at-bats", "shorter at-bats"),
    "HR_FB_pct_pct":    ("home runs per fly ball", "more home runs per fly ball",
                         "fewer home runs per fly ball"),
    "XB_pct_pct":       ("extra-base hits", "more extra-base hits", "fewer extra-base hits"),
    "xwOBA_pct":        ("expected wOBA", "higher expected wOBA", "lower expected wOBA"),
    "xBA_pct":          ("expected average", "higher expected average", "lower expected average"),
    "OBP_SLG_gap_pct":  ("on-base vs power", "on-base over power", "power over on-base"),
    # pitchers
    "avg_velo_pct":         ("velocity", "more velocity", "less velocity"),
    "AvgSpinRate_pct":      ("spin", "more spin", "less spin"),
    "Whiff_pct_pitch_pct":  ("whiffs", "more whiffs", "fewer whiffs"),
    "SwStr_per_velo_pct":   ("whiffs for their velocity", "more whiffs than velocity predicts",
                             "fewer whiffs than velocity predicts"),
    "K_pct_pitch_pct":      ("strikeouts", "more strikeouts", "fewer strikeouts"),
    "BB_pct_pitch_pct":     ("walks", "fewer walks", "more walks"),
    "GB_pct_pitch_pct":     ("ground balls", "more ground balls", "fewer ground balls"),
    "HardHit_allowed_pct":  ("hard contact allowed", "less hard contact allowed",
                             "more hard contact allowed"),
    "Barrel_allowed_pct":   ("barrels allowed", "fewer barrels allowed", "more barrels allowed"),
    "xwOBA_allowed_pct":    ("damage allowed", "less damage allowed", "more damage allowed"),
    "fb_ivb":               ("fastball ride", "more fastball ride", "flatter, sinking fastballs"),
    "fb_run":               ("fastball run", "more arm-side run", "less arm-side run"),
    "vert_sep":             ("vertical separation", "more vertical separation off the fastball",
                             "less vertical separation off the fastball"),
    "horiz_sep":            ("horizontal separation", "more horizontal separation off the fastball",
                             "less horizontal separation off the fastball"),
    "Zone_pct_pct":         ("zone rate", "more pitches in the zone", "fewer pitches in the zone"),
    "FirstPitchStrike_pct": ("first-pitch strikes", "more first-pitch strikes", "fewer first-pitch strikes"),
    "CSW_pct_pct":          ("called and swinging strikes", "more called and swinging strikes",
                             "fewer called and swinging strikes"),
    "TunnelScore_pct":      ("tunneling", "better tunneling", "weaker tunneling"),
    "spin_efficiency_pct":  ("spin efficiency", "more efficient spin", "less efficient spin"),
    "extension_pct":        ("extension", "longer extension", "shorter extension"),
    "ArsenalDiversity_pct": ("arsenal depth", "deeper arsenals", "shallower arsenals"),
    "entropy_ratio_pct":    ("sequencing", "less predictable sequencing", "more predictable sequencing"),
    "count_shift_pct":      ("count-dependent mixing", "more count-dependent mixes", "steadier mixes"),
    # deployment
    "top9_share":           ("lineup stability", "a more fixed lineup", "a more rotated lineup"),
    "bench_share":          ("bench time", "more bench playing time", "less bench playing time"),
    "positions_per_player": ("position switching", "more position switching", "less position switching"),
    "platoon_spread":       ("platooning", "more platooning", "less platooning"),
    "top5_share":           ("rotation stability", "a more stable rotation", "more starters in the mix"),
    "bf_per_start":         ("start length", "longer starts", "shorter starts"),
    "starters_used":        ("starters used", "more starters used", "fewer starters used"),
    "top3_share":           ("bullpen concentration", "a more top-heavy bullpen", "a more spread-out bullpen"),
    "bf_per_game":          ("relief outing length", "longer relief outings", "shorter relief outings"),
    "leverage_spread":      ("leverage roles", "sharper leverage roles", "flatter leverage roles"),
    "arms_used":            ("relievers used", "more relievers used", "fewer relievers used"),
}
# Explanations name at most this many shared traits, one per domain, each
# standing out by at least MIN_Z for both teams.
EXPLAIN_N = 3
MIN_Z = 0.5

# unit -> (players key, weight key, approach domains, outcome domains)
UNITS = {
    "offense":  ("hitters",      "pa", H_APPROACH, H_OUTCOME),
    "rotation": ("starters",     "bf", P_APPROACH, P_OUTCOME),
    "bullpen":  ("bullpen_arms", "bf", P_APPROACH, P_OUTCOME),
}


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None


# ---------------------------------------------------------------------------
# Pitch shape — from the arsenal profile every portrait already stores
# ---------------------------------------------------------------------------
# Movement is in inches, arm side positive for both hands, from the primary
# fastball and the out pitch (pitch_mix.build_arsenal_profile). The full
# arsenal (every pitch type, pitch_mix_{s}.parquet) retrieved better in the
# prototype (rotation 73%, bullpen 65%, against 72% / 62%) but needs the season's pitch
# mix where the league stage runs and a portrait field to rank it against;
# that waits for the next schema bump. Ranked among the season's arms on the
# 30 portraits, so a feature reads as a percentile like every other.
SHAPE_KEYS = ("fb_ivb", "fb_run", "vert_sep", "horiz_sep")


def pitch_shape(player: dict) -> dict[str, float]:
    """{fb_ivb, fb_run, vert_sep, horiz_sep} in inches, for what is known:
    the fastball's ride and arm-side run, and how far the out pitch sits
    from it vertically and horizontally."""
    a = player.get("arsenal_profile") or {}
    fb, op = a.get("fastball") or {}, a.get("out_pitch") or {}
    sign = 1.0 if str(player.get("handedness") or "R")[:1].upper() == "L" else -1.0
    out: dict[str, float] = {}
    fb_ivb, fb_hb = _num(fb.get("avg_ivb")), _num(fb.get("avg_hb"))
    if fb_ivb is not None and fb_hb is not None:
        out["fb_ivb"], out["fb_run"] = fb_ivb * 12, sign * fb_hb * 12
        op_ivb, op_hb = _num(op.get("avg_ivb")), _num(op.get("avg_hb"))
        if op_ivb is not None and op_hb is not None:
            out["vert_sep"] = (fb_ivb - op_ivb) * 12
            out["horiz_sep"] = sign * (fb_hb - op_hb) * 12
    return out


def shape_percentiles(players: list[dict]) -> dict[int, dict[str, float]]:
    """{player_id: {shape key: percentile 0-100}}, each key ranked (average
    ties) among the given pitchers who have it."""
    raw: dict[int, dict[str, float]] = {}
    for p in players:
        pid = p.get("player_id")
        if pid is not None and int(pid) not in raw:
            raw[int(pid)] = pitch_shape(p)
    out: dict[int, dict[str, float]] = {pid: {} for pid in raw}
    for key in SHAPE_KEYS:
        have = [(v[key], pid) for pid, v in raw.items() if key in v]
        if not have:
            continue
        vals = np.array([v for v, _ in have])
        lo = np.searchsorted(np.sort(vals), vals, "left")
        hi = np.searchsorted(np.sort(vals), vals, "right")
        for (_, pid), a, b in zip(have, lo, hi):
            out[pid][key] = float(round((a + b) / 2 / len(vals) * 100, 2))
    return out


def _value(player: dict, src: str, key: str,
           shape: Optional[dict] = None) -> Optional[float]:
    if src == "shape":
        return _num(((shape or {}).get(int(player.get("player_id") or -1)) or {}).get(key))
    gi = player.get("gate_inputs") or {}
    if src == "m":
        # Career-to-date talent (talent.py, schema 28) where it exists — the
        # same stat, era-ranked, from more than one season's sample.
        t = _num((gi.get("talent_metrics") or {}).get(key))
        return t if t is not None else _num((gi.get("metrics") or {}).get(key))
    if src == "gi":
        return _num(gi.get(key))
    if src == "adv":
        return _num((gi.get("advancement") or {}).get(key))
    if src.startswith("pool:"):
        return _num(((gi.get("pool_rows") or {}).get(src[5:]) or {}).get(key))
    raise ValueError(src)


# ---------------------------------------------------------------------------
# Player comps — the same measurements, one player at a time
# ---------------------------------------------------------------------------
# A player-season's comp vector: each metric's era percentile (its talent
# twin where there is one — see _value), centred on the
# league midpoint ((pct - 50) / 50, so -1..1), every domain sharing one unit of
# weight. Era percentiles because comps reach across seasons. Release
# geometry is left out — like handedness it is an attribute, offered as a
# filter rather than a vote. scout_comps (SQL) takes the cosine of these.
#
# Measured against the tag vectors it replaced (prototype, 2026-10-30; 600
# player-seasons per side, 300+ PA/BF): a player's own next season ranks at a
# median 17th of 2,932 hitters (tags: 122nd) and 14th of 2,061 pitchers
# (63rd), and lands in the top 8 comps 38% / 44% of the time (13% / 22%).
_PLAYER_EXCLUDE = {"extension_pct"}     # release geometry: an attribute
PLAYER_DOMAINS = {
    "H": HITTER_DOMAINS,
    "P": {d: [(s, k) for s, k in items if k not in _PLAYER_EXCLUDE]
          for d, items in PITCHER_DOMAINS.items()},
}


def player_vector(player: dict, side: str) -> dict[str, float]:
    """{metric key: weighted, centred percentile} for one player profile;
    metrics the player has no reading for are left out."""
    out: dict[str, float] = {}
    for dom, items in PLAYER_DOMAINS[side].items():
        w = 1 / np.sqrt(len(items))
        for src, key in items:
            v = _value(player, src, key)
            if v is not None:
                out[key] = float(round((v - 50.0) / 50.0 * w, 5))
    return out


# A metric is named as shared only where both players stand at least this far
# from the midpoint, as a centred percentile: 0.3 = the 65th / 35th.
PLAYER_MIN_DEV = 0.3


def explain_players(a: dict, b: dict, side: str) -> dict:
    """
    Why two player-seasons are comps, from their player_vector()s: up to
    EXPLAIN_N metrics where both stand out the same way (one per domain,
    ranked by what they add to the cosine), and the metric that most pulls
    them apart. Same phrases as the team comps.
    Returns {"shared": [phrase, ...], "differ": metric name or None}.
    """
    rows = []
    for dom, items in PLAYER_DOMAINS[side].items():
        w = 1 / np.sqrt(len(items))
        for _, key in items:
            if key in a and key in b and key in PHRASES:
                rows.append((a[key] * b[key], key, dom, a[key] / w, b[key] / w))
    shared, used = [], set()
    for c, key, dom, za, zb in sorted(rows, key=lambda r: -r[0]):
        if len(shared) >= EXPLAIN_N or c <= 0:
            break
        if dom in used or min(abs(za), abs(zb)) < PLAYER_MIN_DEV:
            continue
        _, hi, lo = PHRASES[key]
        shared.append(hi if za > 0 else lo)
        used.add(dom)
    worst = min(rows, key=lambda r: r[0], default=None)
    differ = PHRASES[worst[1]][0] if worst is not None and worst[0] < 0 else None
    return {"shared": shared, "differ": differ}


def style_features(players: list[dict], wkey: str, domains: dict,
                   shape: Optional[dict] = None) -> dict[tuple, float]:
    """{(domain, metric, stat): value}; NaN where under MIN_COVERAGE of the
    unit's workload was measured. `shape` is shape_percentiles() of the
    season's arms, for the shape domain."""
    total = sum(float(p.get(wkey) or 0) for p in players)
    out: dict[tuple, float] = {}
    for dom, items in domains.items():
        for src, key in items:
            have = [(v, w) for p in players
                    if (w := float(p.get(wkey) or 0)) > 0
                    and (v := _value(p, src, key, shape)) is not None]
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


def _standardise(feats: dict[str, dict], budget: bool):
    """(teams, keys, Z, w): teams x features z-scored across teams, and the
    per-feature weight. With `budget`, each domain's features share one unit
    of weight (1/sqrt(n) each, so a domain's squared contribution is the same
    however many metrics it has); similarity is computed on Z * w."""
    keys = sorted({k for f in feats.values() for k in f})
    teams = sorted(feats)
    X = np.array([[feats[t].get(k, np.nan) for k in keys] for t in teams], float)
    with np.errstate(all="ignore"):
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~(sd > 0)] = np.nan
    Z = (X - mu) / sd
    w = np.ones(len(keys))
    if budget:
        per = collections.Counter(k[0] for k in keys)
        w = np.array([1 / np.sqrt(per[k[0]]) for k in keys])
    return teams, keys, Z, w


def _net_of_level(keys: list, Z: np.ndarray, level: str) -> np.ndarray:
    """Z with every feature's linear dependence on the unit's quality (the
    `level` metric's team mean) removed across teams and re-standardised,
    and the level metric's own features blanked: the outcome comparison is
    of how a unit produces, not how much. Unchanged if level is unmeasured."""
    li = next((i for i, k in enumerate(keys) if k[1] == level and k[2] == "mean"), None)
    if li is None:
        return Z
    q, out = Z[:, li].copy(), Z.copy()
    for j, k in enumerate(keys):
        if k[1] == level:
            out[:, j] = np.nan
            continue
        m = np.isfinite(q) & np.isfinite(Z[:, j])
        if m.sum() < 5:
            continue
        slope, icept = np.polyfit(q[m], Z[m, j], 1)
        r = Z[:, j] - (slope * q + icept)
        sd = np.nanstd(r)
        out[:, j] = r / sd if sd > 0 else r
    return out


def explain(keys: list, w: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """
    Why two units are close: the metrics on which BOTH stand out the same
    way (|z| >= MIN_Z for each), ranked by what they add to the similarity,
    one per domain; and the metric that most pulls them apart. Reads each
    metric's team MEAN — the tails move the score but have no plain-words
    reading. `a`, `b` are unweighted z rows. Returns
    {"shared": [phrase, ...], "differ": metric name or None}.
    """
    rows = []
    for i, k in enumerate(keys):
        if len(k) == 3 and k[2] != "mean":
            continue
        metric, za, zb = k[1], a[i], b[i]
        if metric in PHRASES and np.isfinite(za) and np.isfinite(zb):
            rows.append((za * zb * w[i] ** 2, metric, k[0], za, zb))
    shared, used = [], set()
    for c, metric, dom, za, zb in sorted(rows, key=lambda r: -r[0]):
        if len(shared) >= EXPLAIN_N or c <= 0:
            break
        if min(abs(za), abs(zb)) < MIN_Z or (dom in used and dom != "deployment"):
            continue
        _, hi, lo = PHRASES[metric]
        shared.append(hi if za > 0 else lo)
        used.add(dom)
    worst = min(rows, key=lambda r: r[0], default=None)
    differ = PHRASES[worst[1]][0] if worst is not None and worst[0] < 0 else None
    return {"shared": shared, "differ": differ}


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


def _comps(order: list, keys: list, Z: np.ndarray, w: np.ndarray,
           sims: dict, t: str, **extra) -> list[dict]:
    i = order.index(t)
    near = sorted(sims.get(t, {}).items(), key=lambda kv: -kv[1])[:TOP_N]
    return [{"team": o, "similarity": round(s, 4),
             **{k: v.get(o) for k, v in extra.items()},
             **explain(keys, w, Z[i], Z[order.index(o)])} for o, s in near]


def compute_positions(league: dict[str, dict], players: dict[str, dict]) -> None:
    """
    Mutates `league`, per team and unit:
      neighbours            approach comps ("plays like"), and
      uniqueness            how close the nearest of them is
      outcome_neighbours    outcome comps net of quality ("produces like"),
                            each carrying the comp's `level`
      level                 the unit's own quality: the LEVEL metric's
                            workload-weighted mean percentile
      deployment_neighbours usage comps
    `players` is {team: portrait["players"]}.
    """
    teams = [t for t in league if not t.startswith("_") and t in players]
    if len(teams) < 2:
        return
    arms = [p for t in teams for g in ("starters", "bullpen_arms")
            for p in players[t].get(g) or []]
    shape = shape_percentiles(arms)
    for unit, (grp, wkey, approach, outcome) in UNITS.items():
        roster = {t: players[t].get(grp) or [] for t in teams}
        a_feats = {t: style_features(roster[t], wkey, approach, shape) for t in teams}
        order, keys, Z, w = _standardise(a_feats, budget=True)
        sims = _similarities(order, Z * w)
        uq = _uniqueness(sims)

        o_feats = {t: style_features(roster[t], wkey, outcome, shape) for t in teams}
        o_order, o_keys, OZ, ow = _standardise(o_feats, budget=True)
        OZ = _net_of_level(o_keys, OZ, LEVEL[unit])
        o_sims = _similarities(o_order, OZ * ow)
        lvl = {t: (None if not np.isfinite(v := next(
                   (x for k, x in f.items() if k[1] == LEVEL[unit] and k[2] == "mean"), np.nan))
                   else round(float(v), 1)) for t, f in o_feats.items()}

        dep = {t: deployment_features(roster[t], unit) for t in teams}
        d_order, d_keys, DZ, dw = _standardise(dep, budget=False)
        d_sims = _similarities(d_order, DZ * dw)
        for t in teams:
            league[t].setdefault("neighbours", {})[unit] = _comps(order, keys, Z, w, sims, t)
            league[t].setdefault("uniqueness", {})[unit] = uq.get(t, {})
            league[t].setdefault("outcome_neighbours", {})[unit] = _comps(
                o_order, o_keys, OZ, ow, o_sims, t, level=lvl)
            league[t].setdefault("level", {})[unit] = lvl.get(t)
            league[t].setdefault("deployment_neighbours", {})[unit] = _comps(
                d_order, d_keys, DZ, dw, d_sims, t)
    league["_similarity"] = {
        "method": METHOD, "min_coverage": MIN_COVERAGE, "tails": [TAIL_LO, TAIL_HI],
        "domains": {u: {d: [k for _, k in items] for d, items in doms.items()}
                    for u, (_, _, doms, _) in UNITS.items()},
        "outcome_domains": {u: {d: [k for _, k in items] for d, items in doms.items()}
                            for u, (_, _, _, doms) in UNITS.items()},
        "level": LEVEL,
    }
