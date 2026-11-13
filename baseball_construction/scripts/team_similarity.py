"""
team_similarity.py — which teams a unit plays like, from the measurements.

A team used to be compared through its tag densities: the share of its
workload carried by players over each tag's line. That throws away most of
what was measured and weights each skill by how many tags happen to describe
it. A unit is now its measurements: for every metric, the workload-weighted
(PA/BF) mean of its players' percentiles, z-scored across the season's 30
teams. A metric counts for a team only when at least half the unit's
workload was measured for it.

Two scores, read off one picture (dots-v1, 2026-11-13). Lay two units side by
side on every metric — a dot each, in league standard deviations — and the
similarity is how close the dots sit:

  APPROACH   what the players choose or physically bring: swing decisions,
             bat-to-ball, spray, tools, running; arm, attack, mix, pitch shape
  RESULTS    what happened: strikeouts and walks, contact quality, power,
             production; misses (whiffs, called strikes, fouls per swing),
             strikeouts and walks, contact allowed

  section gap   mean |z_a - z_b| over the section's scored metrics
  similarity    0-100: the share of the season's 435 unit pairs whose gap is
                larger, so the two sections share a scale and add
  total         approach + results (0-200), which ranks the comps

Every metric counts once and equally. Rules, each a decision with its reason:
  - contact and whiffs are one number at team level (rho -1.00): whiffs out
  - exit velocity is shown, not scored: it and hard contact are rho +0.96,
    and hard contact (results) is the one that counts
  - expected wOBA / average (allowed) are shown, not scored: they are the
    unit's quality, which pulled results comps toward equally good teams
  - offense results are scored after removing each offense's own mean over
    the rows where higher is better (not OBP-vs-power), so the gap compares
    how a lineup produces, not how much; pitching keeps the plain rows,
    where centring cost 6-9 points of repeatability
  - CSW was two opposite styles added together (called strikes and whiffs,
    rho -0.42 per pitcher), so misses are whiffs, called strikes and fouls
    per swing; whiffs above what velocity predicts repeated plain whiffs

Measured on 2015-26 (prototype, 2026-11-13): a unit's own next season in its
top 3 comps (random 10%), and how much pair similarity tracks the gap in
quality (Spearman with -|xwOBA gap|):

                 offense      rotation     bullpen
  approach      66%  +0.06   67%  +0.09   51%  +0.08
  results       55%  +0.09   54%  +0.46   41%  +0.38
  total         71%  +0.09   72%  +0.35   53%  +0.29

The cosine it replaced repeated slightly better (81/72/62%) but could not be
read: a similarity of 0.43 said nothing, and nothing on screen showed which
metrics made it. Here every point of the score is a gap you can see.

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

METHOD = "dots-v1"   # explanations do not change the comps, so no bump
MIN_COVERAGE = 0.5
TOP_N = 3

# domain -> [(source, key)]. Sources: m = gate_inputs.metrics (league
# percentiles), gi = a gate_inputs field, adv = advancement,
# pool:<frame> = the pitcher's row from that league pool, shape = the
# pitcher's pitch shape (pitch_shape below), ranked among the season's arms.
H_APPROACH: dict[str, list[tuple[str, str]]] = {
    "swing decisions": [("m", "OSwing_pct_pct"), ("m", "ZSwing_pct_pct"),
                        ("m", "FPS_pct_pct"), ("m", "PitchesPerPA_pct")],
    "bat-to-ball":     [("m", "Contact_pct_pct")],
    "spray":           [("m", "Pull_pct_pct"), ("m", "Oppo_pct_pct"), ("m", "BatGB_pct_pct"),
                        ("m", "GapTend_pct_pct")],
    "tools":           [("m", "EV_avg_pct"), ("m", "sprint_speed_pct")],
    "running":         [("gi", "attempt_rate_pct"), ("adv", "xbt_pct")],
}
H_RESULTS: dict[str, list[tuple[str, str]]] = {
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
P_RESULTS: dict[str, list[tuple[str, str]]] = {
    "misses":               [("m", "Whiff_pct_pitch_pct"),
                             ("pool:league_discipline", "called_strike_pct"),
                             ("pool:league_discipline", "foul_swing_pct")],
    "strikeouts and walks": [("m", "K_pct_pitch_pct"), ("m", "BB_pct_pitch_pct")],
    "contact allowed":      [("m", "GB_pct_pitch_pct"), ("m", "HardHit_allowed_pct"),
                             ("m", "Barrel_allowed_pct"), ("m", "xwOBA_allowed_pct")],
}
# Drawn on the comparison but not in the score (see the module docstring).
SHOW_ONLY = frozenset({"EV_avg_pct", "xwOBA_pct", "xBA_pct", "xwOBA_allowed_pct"})
# Results rows with no better/worse direction: never centred.
STYLE = frozenset({"OBP_SLG_gap_pct", "GB_pct_pitch_pct"})
# unit -> sections scored after removing the unit's own mean over the
# directional rows (every scored percentile there already points high =
# better; K%, walks and contact allowed are inverted in the contracts).
CENTRED = {"offense": frozenset({"results"})}
# The unit's quality, for the check that comps do not just pair equal teams.
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
    "called_strike_pct":    ("called strikes", "more called strikes", "fewer called strikes"),
    "foul_swing_pct":       ("fouls per swing", "more fouls per swing", "fewer fouls per swing"),
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
# The comparison's row labels, read left to right as less -> more. Where a
# high percentile means FEWER of something the label says so, so right of
# centre is always the high percentile.
LABELS: dict[str, str] = {
    "K_pct_pct": "avoids strikeouts", "BB_pct_pitch_pct": "avoids walks",
    "HardHit_allowed_pct": "avoids hard contact", "Barrel_allowed_pct": "avoids barrels",
    "xwOBA_allowed_pct": "limits expected wOBA", "OBP_SLG_gap_pct": "on-base over power",
}
# Explanations name at most this many shared traits, one per domain, each
# standing out by at least MIN_Z for both teams.
EXPLAIN_N = 3
MIN_Z = 0.5

# unit -> (players key, weight key, approach domains, results domains)
UNITS = {
    "offense":  ("hitters",      "pa", H_APPROACH, H_RESULTS),
    "rotation": ("starters",     "bf", P_APPROACH, P_RESULTS),
    "bullpen":  ("bullpen_arms", "bf", P_APPROACH, P_RESULTS),
}
SECTIONS = ("approach", "results")


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
    """{(domain, metric, "mean"): workload-weighted mean percentile}; NaN
    where under MIN_COVERAGE of the unit's workload was measured. `shape` is
    shape_percentiles() of the season's arms, for the shape domain. (The
    hybrid-v1 tails — share of workload above the 80th / below the 20th —
    went with the cosine: on the comparison they had no dot to draw.)"""
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


def comparison_rows(unit: str) -> list[dict]:
    """The rows of a unit's comparison, in drawing order: section, domain,
    metric key, label, whether it is scored and whether it is centred."""
    _, _, approach, results = UNITS[unit]
    rows = []
    for section, doms in zip(SECTIONS, (approach, results)):
        for dom, items in doms.items():
            for src, key in items:
                scored = key not in SHOW_ONLY
                rows.append({"section": section, "domain": dom, "src": src, "key": key,
                             "label": LABELS.get(key, PHRASES[key][0]), "scored": scored,
                             "centred": scored and key not in STYLE
                                        and section in CENTRED.get(unit, ())})
    return rows


def unit_profiles(players: dict[str, list], unit: str, shape: Optional[dict] = None):
    """(teams, rows, Z): every team's z-score on every comparison row,
    across the given teams. `players` is {team: that unit's players}."""
    _, wkey, _, _ = UNITS[unit]
    rows = comparison_rows(unit)
    doms: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        doms[r["domain"] + "|" + r["section"]].append((r["src"], r["key"]))
    feats = {t: style_features(pl, wkey, doms, shape) for t, pl in players.items()}
    teams, keys, Z, _ = _standardise(feats, budget=False)
    col = {(k[0].split("|")[1], k[1]): i for i, k in enumerate(keys)}
    Z = Z[:, [col[(r["section"], r["key"])] for r in rows]]
    return teams, rows, Z


def scored_values(rows: list[dict], Z: np.ndarray) -> np.ndarray:
    """Z as scored: show-only rows blanked, centred rows less each team's own
    mean over them."""
    S = Z.copy()
    c = [j for j, r in enumerate(rows) if r["centred"]]
    if c:
        with np.errstate(all="ignore"):
            S[:, c] = S[:, c] - np.nanmean(S[:, c], axis=1, keepdims=True)
    S[:, [j for j, r in enumerate(rows) if not r["scored"]]] = np.nan
    return S


def section_gap(rows: list[dict], a: np.ndarray, b: np.ndarray, section: str) -> float:
    """Mean |a - b| over the section's scored rows both units have; inf when
    they share none."""
    idx = [j for j, r in enumerate(rows) if r["section"] == section]
    d = np.abs(a[idx] - b[idx])
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size else np.inf


def explain_dots(rows: list[dict], za: np.ndarray, zb: np.ndarray) -> dict:
    """
    Why two units are comps, off the comparison itself: up to EXPLAIN_N
    scored metrics where both stand out the same way (|z| >= MIN_Z for each),
    closest first, one per domain; and the scored metric they sit furthest
    apart on. `za`, `zb` are the drawn (unscored) z rows.
    Returns {"shared": [phrase, ...], "differ": label or None}.
    """
    have = [(abs(za[j] - zb[j]), j) for j, r in enumerate(rows)
            if r["scored"] and np.isfinite(za[j]) and np.isfinite(zb[j])]
    shared, used = [], set()
    for _, j in sorted(have):
        r = rows[j]
        if len(shared) >= EXPLAIN_N:
            break
        if r["domain"] in used or za[j] * zb[j] <= 0 or min(abs(za[j]), abs(zb[j])) < MIN_Z:
            continue
        _, hi, lo = PHRASES[r["key"]]
        shared.append(hi if za[j] > 0 else lo)
        used.add(r["domain"])
    far = max(have, default=None)
    return {"shared": shared, "differ": rows[far[1]]["label"] if far else None}


def compute_positions(league: dict[str, dict], players: dict[str, dict]) -> None:
    """
    Mutates `league`, per team and unit:
      profile               its z-score on every comparison row (the rows
                            are league["_similarity"]["rows"][unit])
      comps                 every other team as [team, approach, results]
                            (each 0-100), best total first
      neighbours            the TOP_N of those, with why (explain_dots)
      uniqueness            how close the nearest comp is, banded
      deployment_neighbours usage comps (cosine, unchanged)
    `players` is {team: portrait["players"]}.
    """
    teams = [t for t in league if not t.startswith("_") and t in players]
    if len(teams) < 2:
        return
    arms = [p for t in teams for g in ("starters", "bullpen_arms")
            for p in players[t].get(g) or []]
    shape = shape_percentiles(arms)
    rows_out = {}
    for unit, (grp, _, _, _) in UNITS.items():
        order, rows, Z = unit_profiles({t: players[t].get(grp) or [] for t in teams}, unit, shape)
        rows_out[unit] = [{k: r[k] for k in ("section", "domain", "key", "label", "scored", "centred")}
                          for r in rows]
        S = scored_values(rows, Z)
        n = len(order)
        sims = {}
        for section in SECTIONS:
            G = np.full((n, n), np.inf)
            for i in range(n):
                for j in range(i + 1, n):
                    G[i, j] = G[j, i] = section_gap(rows, S[i], S[j], section)
            pairs = np.sort(G[np.triu_indices(n, 1)])
            # share of the season's pairs farther apart, so 100 = the closest pair
            sims[section] = np.rint(100 * (1 - np.searchsorted(pairs, G, "left") / len(pairs)))
        total = sims["approach"] + sims["results"]
        uq = _uniqueness({t: {o: float(total[i, j]) for j, o in enumerate(order) if j != i}
                          for i, t in enumerate(order)})
        dep = {t: deployment_features(players[t].get(grp) or [], unit) for t in teams}
        d_order, d_keys, DZ, dw = _standardise(dep, budget=False)
        d_sims = _similarities(d_order, DZ * dw)
        for i, t in enumerate(order):
            ranked = sorted((j for j in range(n) if j != i), key=lambda j: -total[i, j])
            comps = [{"team": order[j], "approach": int(sims["approach"][i, j]),
                      "results": int(sims["results"][i, j]), "total": int(total[i, j])}
                     for j in ranked]
            # Compact, as [team, approach, results]: every season's file is
            # read for the year-over-year view, and the total is their sum.
            league[t].setdefault("profile", {})[unit] = [
                None if not np.isfinite(v) else round(float(v), 2) for v in Z[i]]
            league[t].setdefault("comps", {})[unit] = [
                [c["team"], c["approach"], c["results"]] for c in comps]
            league[t].setdefault("neighbours", {})[unit] = [
                {**c, **explain_dots(rows, Z[i], Z[order.index(c["team"])])} for c in comps[:TOP_N]]
            league[t].setdefault("uniqueness", {})[unit] = uq.get(t, {})
            d_near = sorted(d_sims.get(t, {}).items(), key=lambda kv: -kv[1])[:TOP_N]
            di = d_order.index(t)
            league[t].setdefault("deployment_neighbours", {})[unit] = [
                {"team": o, "similarity": round(s, 4),
                 **explain(d_keys, dw, DZ[di], DZ[d_order.index(o)])} for o, s in d_near]
    league["_similarity"] = {
        "method": METHOD, "min_coverage": MIN_COVERAGE, "rows": rows_out,
        "scale": "approach and results are each the share (0-100) of the season's "
                 "unit pairs whose average gap is larger; total = approach + results",
    }
