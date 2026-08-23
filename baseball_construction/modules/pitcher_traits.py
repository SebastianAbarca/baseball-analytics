"""
pitcher_traits.py — Trait tags for pitchers: evidence-based, independent,
non-exclusive.

Philosophy: no boxes. Each trait is tagged independently when the evidence
clears a league-percentile gate, and a pitcher's identity is the SET of tags
he carries. A league-average pitcher may carry zero tags in a family — that
is informative, not a classification failure. Team shape emerges from
BF-weighted tag density across a staff, not from a winner-take-all label.

Trait record format (shared by all families):
    {
        "tag":      "sidearm",            # short kebab/space label
        "family":   "mechanics",          # mechanics | sequencing | deception | arsenal | outcome
        "value":    17.4,                 # raw metric behind the tag
        "pct":      3.5,                  # league percentile of that metric (0-100), None if N/A
        "evidence": "arm angle 17.4° (3rd pct of 684 pitchers)",
    }

Families (build order):
    1. mechanics  — arm slot, release extension, release width   [THIS FILE, live]
    2. sequencing — transition entropy, count-shifting, FPS      [next]
    3. deception  — tunneler, invisible-ball, spin traits        [next]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from reliability import filter_traits, LIMITED_SAMPLE_BF  # noqa: E402,F401

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
PROCESSED_DIR = _HERE / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Minimum pitches for a stable mechanics read
MIN_MECHANICS_PITCHES = 150

# ---------------------------------------------------------------------------
# Mechanics thresholds — set from the 2023 league distribution (684 pitchers):
#   arm_angle pct:  2%≈8°  10%≈22.5°  50%≈39.4°  90%≈52.3°  95%≈55.4°  98%≈59°
#   extension pct: 10%≈5.89  50%≈6.44  90%≈6.90
# Verified against known slots: Kershaw 59° (over-the-top), Cimber 17°
# (sidearm), Ryan Thompson −8° (submarine).
# The broad middle (~25°–52°) is a normal three-quarters slot → NO tag.
# ---------------------------------------------------------------------------
SUBMARINE_MAX_ANGLE    = 10.0   # below → submarine (angle can be negative)
SIDEARM_MAX_ANGLE      = 25.0   # 10–25° → sidearm (bottom ~15%)
OVER_THE_TOP_MIN_ANGLE = 55.0   # ≥ 55° → over-the-top (top ~5%)

DEEP_EXTENSION_PCT  = 90.0      # release_extension percentile ≥ → deep extension
SHORT_EXTENSION_PCT = 10.0      # ≤ → short extension
WIDE_RELEASE_PCT    = 90.0      # |release_pos_x| percentile within handedness ≥ → wide release


# ---------------------------------------------------------------------------
# League mechanics pool
# ---------------------------------------------------------------------------

def load_league_mechanics(
    season: int,
    statcast: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    League-wide per-pitcher mechanics pool, cached to parquet.

    Columns: pitcher, p_throws, n_pitches,
             arm_angle, extension, rel_x, rel_z, rel_x_abs,
             arm_angle_pct, extension_pct, rel_x_abs_pct (within handedness)
    """
    cache = PROCESSED_DIR / f"mechanics_league_{season}.parquet"
    if cache.exists() and not force:
        log.info("mechanics_league %d: loading from cache", season)
        return pd.read_parquet(cache)

    if statcast is None:
        from ingest import pull_statcast_season
        statcast = pull_statcast_season(season)

    log.info("mechanics_league %d: building from Statcast…", season)
    sc = statcast[["pitcher", "p_throws", "arm_angle",
                   "release_extension", "release_pos_x", "release_pos_z"]]

    per = sc.groupby(["pitcher", "p_throws"]).agg(
        n_pitches=("arm_angle", "size"),
        arm_angle=("arm_angle", "mean"),
        extension=("release_extension", "mean"),
        rel_x=("release_pos_x", "mean"),
        rel_z=("release_pos_z", "mean"),
    ).reset_index()

    per = per[(per["n_pitches"] >= MIN_MECHANICS_PITCHES)
              & per["arm_angle"].notna()].copy()
    per["rel_x_abs"] = per["rel_x"].abs()

    per["arm_angle_pct"] = per["arm_angle"].rank(pct=True) * 100.0
    per["extension_pct"] = per["extension"].rank(pct=True) * 100.0
    # Release width ranked within handedness — L and R live on opposite sides
    per["rel_x_abs_pct"] = (
        per.groupby("p_throws")["rel_x_abs"].rank(pct=True) * 100.0
    )

    per = per.astype({"arm_angle": float, "extension": float,
                      "rel_x": float, "rel_z": float, "rel_x_abs": float})
    per.to_parquet(cache, index=False)
    log.info("mechanics_league %d: saved %d pitchers", season, len(per))
    return per


# ---------------------------------------------------------------------------
# Mechanics traits
# ---------------------------------------------------------------------------

def _trait(tag: str, family: str, value: float, pct: Optional[float],
           evidence: str) -> dict:
    return {
        "tag":      tag,
        "family":   family,
        "value":    round(float(value), 2) if value is not None else None,
        "pct":      round(float(pct), 1) if pct is not None else None,
        "evidence": evidence,
    }


def mechanics_traits(row: pd.Series) -> list[dict]:
    """
    Mechanics trait tags for one pitcher (a row from load_league_mechanics).

    A normal three-quarters slot with average extension gets zero tags.
    """
    traits: list[dict] = []

    angle = row.get("arm_angle")
    if angle is not None and not pd.isna(angle):
        a, apct = float(angle), row.get("arm_angle_pct")
        if a < SUBMARINE_MAX_ANGLE:
            traits.append(_trait("submarine", "mechanics", a, apct,
                                 f"arm angle {a:.0f}° — below horizontal-plus buffer"))
        elif a < SIDEARM_MAX_ANGLE:
            traits.append(_trait("sidearm", "mechanics", a, apct,
                                 f"arm angle {a:.0f}° (bottom ~15% of slots)"))
        elif a >= OVER_THE_TOP_MIN_ANGLE:
            traits.append(_trait("over-the-top", "mechanics", a, apct,
                                 f"arm angle {a:.0f}° (top ~5% of slots)"))

    ext, ext_pct = row.get("extension"), row.get("extension_pct")
    if ext is not None and not pd.isna(ext) and ext_pct is not None:
        if ext_pct >= DEEP_EXTENSION_PCT:
            traits.append(_trait("deep extension", "mechanics", ext, ext_pct,
                                 f"{ext:.2f} ft release extension ({ext_pct:.0f}th pct) — "
                                 "perceived velo plays up"))
        elif ext_pct <= SHORT_EXTENSION_PCT:
            traits.append(_trait("short extension", "mechanics", ext, ext_pct,
                                 f"{ext:.2f} ft release extension ({ext_pct:.0f}th pct) — "
                                 "ball travels farther to the plate"))

    rx_pct = row.get("rel_x_abs_pct")
    if rx_pct is not None and not pd.isna(rx_pct) and rx_pct >= WIDE_RELEASE_PCT:
        traits.append(_trait("wide release", "mechanics", row["rel_x_abs"], rx_pct,
                             f"release {row['rel_x_abs']:.2f} ft off-center "
                             f"({rx_pct:.0f}th pct for handedness) — cross-fire angle"))

    return traits


# ---------------------------------------------------------------------------
# Sequencing — how the arsenal is deployed
# ---------------------------------------------------------------------------

# Minimum pitches for stable transition statistics
MIN_SEQUENCING_PITCHES = 300

# Behavioral tags gate at the distribution tails (top/bottom decile)
UNPREDICTABLE_PCT   = 90.0   # entropy ratio ≥ → "unpredictable"
PATTERNED_PCT       = 10.0   # entropy ratio ≤ → "patterned"
COUNT_SHIFTER_PCT   = 90.0   # ahead-vs-behind mix distance ≥ → "count-shifter"
ONE_LOOK_PCT        = 10.0   # ≤ → "steady mix" (same mix in any count)
FP_ATTACKER_PCT     = 90.0   # first-pitch strike rate ≥ → "first-pitch attacker"

# Count groups for the count-shift comparison (balls, strikes)
_AHEAD_COUNTS  = {(0, 1), (0, 2), (1, 2)}
_BEHIND_COUNTS = {(1, 0), (2, 0), (3, 0), (2, 1), (3, 1)}


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _pitcher_sequencing(g: pd.DataFrame) -> Optional[dict]:
    """
    Sequencing metrics for one pitcher's season of pitches (pre-sorted).

    entropy_ratio — H(next pitch | current pitch) / H(pitch mix overall).
        1.0 = knowing the current pitch tells you nothing (unpredictable);
        low = strong pitch-to-pitch patterns. The ratio (not raw entropy)
        controls for arsenal size, so a two-pitch closer isn't automatically
        "patterned" just for having two pitches.
    count_shift — total-variation distance between the pitch mix when ahead
        (0-1, 0-2, 1-2) vs behind (1-0, 2-0, 3-0, 2-1, 3-1).
        0 = identical mix in any count; 1 = completely different arsenal.
    """
    types = g["pitch_type"]
    mix = types.value_counts(normalize=True)
    h_mix = _entropy(mix.to_numpy())
    if h_mix <= 0:
        return None  # single-pitch season — sequencing undefined

    # Conditional entropy over within-AB transitions
    same_ab = (g["game_pk"] == g["game_pk"].shift()) & \
              (g["at_bat_number"] == g["at_bat_number"].shift())
    pairs = pd.DataFrame({"prev": types.shift(), "cur": types})[same_ab].dropna()
    if len(pairs) < MIN_SEQUENCING_PITCHES // 2:
        return None
    h_cond = 0.0
    for _, sub in pairs.groupby("prev")["cur"]:
        w = len(sub) / len(pairs)
        h_cond += w * _entropy(sub.value_counts(normalize=True).to_numpy())
    entropy_ratio = h_cond / h_mix

    # Ahead-vs-behind mix distance
    counts = list(zip(g["balls"], g["strikes"]))
    ahead  = types[[c in _AHEAD_COUNTS for c in counts]]
    behind = types[[c in _BEHIND_COUNTS for c in counts]]
    count_shift = None
    if len(ahead) >= 50 and len(behind) >= 50:
        pa = ahead.value_counts(normalize=True)
        pb = behind.value_counts(normalize=True)
        all_t = pa.index.union(pb.index)
        count_shift = float(
            0.5 * sum(abs(pa.get(t, 0.0) - pb.get(t, 0.0)) for t in all_t)
        )

    return {"entropy_ratio": entropy_ratio, "count_shift": count_shift,
            "n_transitions": len(pairs)}


def load_league_sequencing(
    season: int,
    statcast: Optional[pd.DataFrame] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    League-wide per-pitcher sequencing pool, cached to parquet.

    Columns: pitcher, n_pitches, n_transitions,
             entropy_ratio, count_shift,
             entropy_ratio_pct, count_shift_pct
    """
    cache = PROCESSED_DIR / f"sequencing_league_{season}.parquet"
    if cache.exists() and not force:
        log.info("sequencing_league %d: loading from cache", season)
        return pd.read_parquet(cache)

    if statcast is None:
        from ingest import pull_statcast_season
        statcast = pull_statcast_season(season)

    log.info("sequencing_league %d: building from Statcast…", season)
    sc = statcast[["pitcher", "game_pk", "at_bat_number", "pitch_number",
                   "pitch_type", "balls", "strikes"]].copy()
    sc = sc[sc["pitch_type"].notna() & (sc["pitch_type"] != "")
            & ~sc["pitch_type"].isin({"PO", "EP"})]
    sc = sc.sort_values(["pitcher", "game_pk", "at_bat_number", "pitch_number"])

    rows = []
    for pid, g in sc.groupby("pitcher"):
        if len(g) < MIN_SEQUENCING_PITCHES:
            continue
        m = _pitcher_sequencing(g)
        if m is not None:
            rows.append({"pitcher": int(pid), "n_pitches": len(g), **m})

    pool = pd.DataFrame(rows)
    if pool.empty:
        log.warning("sequencing_league %d: no qualifying pitchers", season)
        return pool
    pool["entropy_ratio_pct"] = pool["entropy_ratio"].rank(pct=True) * 100.0
    pool["count_shift_pct"]   = pool["count_shift"].rank(pct=True) * 100.0

    pool.to_parquet(cache, index=False)
    log.info("sequencing_league %d: saved %d pitchers", season, len(pool))
    return pool


def sequencing_traits(
    row: pd.Series,
    discipline_row: Optional[pd.Series] = None,
) -> list[dict]:
    """
    Sequencing trait tags for one pitcher: pool row from
    load_league_sequencing, plus (optionally) the pitcher's row from
    pitch_mix.load_league_plate_discipline for the first-pitch tag.
    """
    traits: list[dict] = []

    er, er_pct = row.get("entropy_ratio"), row.get("entropy_ratio_pct")
    if er_pct is not None and not pd.isna(er_pct):
        if er_pct >= UNPREDICTABLE_PCT:
            traits.append(_trait("unpredictable", "sequencing", er, er_pct,
                                 f"pitch-to-pitch entropy ratio {er:.2f} "
                                 f"({er_pct:.0f}th pct) — the current pitch "
                                 "tells you almost nothing about the next"))
        elif er_pct <= PATTERNED_PCT:
            traits.append(_trait("patterned", "sequencing", er, er_pct,
                                 f"pitch-to-pitch entropy ratio {er:.2f} "
                                 f"({er_pct:.0f}th pct) — strong repeatable "
                                 "sequences"))

    cs, cs_pct = row.get("count_shift"), row.get("count_shift_pct")
    if cs is not None and not pd.isna(cs) and cs_pct is not None and not pd.isna(cs_pct):
        if cs_pct >= COUNT_SHIFTER_PCT:
            traits.append(_trait("count-shifter", "sequencing", cs, cs_pct,
                                 f"pitch mix moves {cs:.0%} between ahead and "
                                 f"behind counts ({cs_pct:.0f}th pct) — a "
                                 "different pitcher at 0-2 than 2-0"))
        elif cs_pct <= ONE_LOOK_PCT:
            traits.append(_trait("steady mix", "sequencing", cs, cs_pct,
                                 f"pitch mix moves only {cs:.0%} between ahead "
                                 f"and behind counts ({cs_pct:.0f}th pct) — "
                                 "same arsenal in any count"))

    if discipline_row is not None:
        fps, fps_pct = discipline_row.get("fps_rate"), discipline_row.get("first_pitch_strike_pct")
        if fps_pct is not None and not pd.isna(fps_pct) and fps_pct >= FP_ATTACKER_PCT:
            traits.append(_trait("first-pitch attacker", "sequencing", fps, fps_pct,
                                 f"{fps:.0%} first-pitch strikes "
                                 f"({fps_pct:.0f}th pct)"))

    return traits


# ---------------------------------------------------------------------------
# Deception — how the stuff plays beyond its raw quality
# Consolidates the league pools built in tunneling.py / spin_efficiency.py /
# pitch_mix.py into trait tags.
# ---------------------------------------------------------------------------

TUNNELER_PCT       = 75.0   # league tunnel score ≥ → "tunneler" (matches the
                            # arsenal approach gate in pitch_mix._approach)
INVISIBLE_BALL_PCT = 90.0   # velocity-adjusted whiff residual ≥ → "invisible ball"
HIGH_SPIN_EFF_PCT  = 90.0   # spin efficiency ≥ → "high spin efficiency"
GYRO_HEAVY_PCT     = 10.0   # spin efficiency ≤ → "gyro-heavy" (bullet spin)


def deception_traits(
    tunnel_row:     Optional[pd.Series] = None,
    spin_row:       Optional[pd.Series] = None,
    discipline_row: Optional[pd.Series] = None,
) -> list[dict]:
    """
    Deception trait tags for one pitcher from the league tunnel, spin, and
    plate-discipline pools.
    """
    traits: list[dict] = []

    if tunnel_row is not None:
        tp = tunnel_row.get("tunnel_score_pct")
        if tp is not None and not pd.isna(tp) and float(tp) >= TUNNELER_PCT:
            raw = tunnel_row.get("tunnel_score_weighted")
            traits.append(_trait("tunneler", "deception",
                                 raw if raw is not None and not pd.isna(raw) else tp, tp,
                                 f"league tunnel score {tp:.0f}th pct — release "
                                 "paths converge, plate locations diverge"))

    if discipline_row is not None:
        sp = discipline_row.get("swstr_per_velo_pct")
        if sp is not None and not pd.isna(sp) and float(sp) >= INVISIBLE_BALL_PCT:
            resid = discipline_row.get("swstr_resid")
            traits.append(_trait("invisible ball", "deception", resid, sp,
                                 f"whiff rate {resid:+.1%} above what velocity "
                                 f"predicts ({sp:.0f}th pct) — misses bats the "
                                 "radar gun can't explain"))

    if spin_row is not None:
        ep = spin_row.get("spin_efficiency_pct")
        if ep is not None and not pd.isna(ep):
            eff = spin_row.get("spin_efficiency_weighted")
            if float(ep) >= HIGH_SPIN_EFF_PCT:
                traits.append(_trait("high spin efficiency", "deception", eff, ep,
                                     f"spin efficiency {ep:.0f}th pct — movement "
                                     "extracts nearly all of the raw spin"))
            # `gyro-heavy` retired: P(slider out pitch | gyro-heavy) = 0.79 —
            # gyro spin IS slider spin, so the tag mostly restated the pitch.

    return traits


# ---------------------------------------------------------------------------
# Arsenal — what they throw (reformats pitch_mix.build_arsenal_profile)
# ---------------------------------------------------------------------------

ELITE_VELO_PCT  = 90.0
PLUS_VELO_PCT   = 75.0
SOFT_TOSSER_PCT = 10.0


def arsenal_traits(
    arsenal_profile: Optional[dict],
    metrics: Optional[dict] = None,
    is_starter: Optional[bool] = None,
) -> list[dict]:
    """
    Arsenal trait tags from the weapon-based profile already computed by
    pitch_mix.build_arsenal_profile, plus velocity tiers from the pitcher's
    league percentile metrics.
    """
    traits: list[dict] = []
    ap = arsenal_profile or {}

    if ap.get("knuckleballer"):
        kn_usage = (ap.get("out_pitch") or {}).get("usage_pct") or 0
        traits.append(_trait("knuckleballer", "arsenal", kn_usage * 100, None,
                             f"knuckleball {kn_usage:.0%} of pitches"))
        # A knuckleballer's other arsenal descriptors are noise — stop here.
        return traits

    fb = ap.get("fastball") or {}
    family, fb_usage = fb.get("family"), fb.get("usage_pct") or 0
    if family == "sinker":
        traits.append(_trait("sinker-baller", "arsenal", fb_usage * 100, None,
                             f"sinker-led fastball family ({fb_usage:.0%} usage)"))
    elif family == "4-seam" and (fb.get("avg_ivb") or 0) >= 1.4:
        traits.append(_trait("ride four-seam", "arsenal", fb["avg_ivb"], None,
                             f"4-seam with {fb['avg_ivb']:.2f} ft IVB — "
                             "carry through the top of the zone"))
    elif family == "cutter":
        traits.append(_trait("cutter-primary", "arsenal", fb_usage * 100, None,
                             f"cutter-led fastball family ({fb_usage:.0%} usage)"))

    out = ap.get("out_pitch") or {}
    if out.get("pitch_type"):
        try:
            from pitch_mix import PITCH_LABEL
            label = PITCH_LABEL.get(out["pitch_type"], out["pitch_type"])
        except Exception:
            label = out["pitch_type"]
        whiff = out.get("whiff_pct")
        traits.append(_trait(f"{label.lower()} out pitch", "arsenal",
                             (out.get("usage_pct") or 0) * 100, None,
                             f"{label} put-away pitch"
                             + (f" — {whiff:.0%} whiffs" if whiff is not None else "")))

    depth = ap.get("depth")
    n_meaningful = 2 + len(ap.get("supporting") or [])
    # 4 pitches at ≥5% is just a normal modern starter (league mean) —
    # "deep arsenal" marks genuine kitchen-sink mixes only.
    if depth == "multi-pitch" and n_meaningful >= 5:
        traits.append(_trait("deep arsenal", "arsenal", n_meaningful, None,
                             f"{n_meaningful} pitches thrown ≥5% of the time"))
    elif depth in ("one-pitch", "two-pitch") and is_starter:
        # Starters only: a two-pitch RELIEVER is the default reliever —
        # unremarkable. A two-pitch starter (Greene, Steele) is notable.
        traits.append(_trait("two-pitch", "arsenal", n_meaningful, None,
                             "lives on two (or fewer) pitches as a starter"))

    if metrics:
        vp = metrics.get("avg_velo_pct")
        if vp is not None and not pd.isna(vp):
            if vp >= ELITE_VELO_PCT:
                traits.append(_trait("elite velo", "arsenal", vp, vp,
                                     f"average velocity {vp:.0f}th pct"))
            elif vp >= PLUS_VELO_PCT:
                traits.append(_trait("plus velo", "arsenal", vp, vp,
                                     f"average velocity {vp:.0f}th pct"))
            elif vp <= SOFT_TOSSER_PCT:
                traits.append(_trait("soft tosser", "arsenal", vp, vp,
                                     f"average velocity {vp:.0f}th pct"))

    return traits


# ---------------------------------------------------------------------------
# Outcome — what the results say (from league-percentile metrics)
# ---------------------------------------------------------------------------

# Tempo (savant empty-bases median seconds; higher pct = slower)
QUICK_PITCHER_PCT = 15.0
SLOW_PITCHER_PCT  = 85.0

# Runner control (mid-PA advances allowed; base-state derived)
CONTROLS_RUNNERS_PCT      = 15.0   # advance-rate percentile ≤ → tag
CONTROLS_RUNNERS_MIN_OPPS = 100

BAT_MISSER_PCT        = 80.0
PITCH_TO_CONTACT_PCT  = 20.0
GROUND_BALLER_PCT     = 75.0
FLY_BALL_PRONE_PCT    = 10.0
CONTACT_SUPPRESS_PCT  = 80.0
COMMAND_ELITE_PCT     = 90.0
COMMAND_PLUS_PCT      = 75.0
WALK_PRONE_PCT        = 15.0

# Ace badge: composite mean of the dominance keys plus ≥2 independently
# elite traits — same spirit as pitcher_archetypes.compute_ace_tier but
# computed directly from metrics (no archetype score needed).
ACE_KEYS            = ["K_pct_pct", "BB_pct_pct", "HardHit_allowed_pct",
                       "SwStr_pct_pct", "GB_pct_pct", "CSW_pct_pct"]
ACE_COMPOSITE_MIN   = 70.0
ACE_ELITE_TRAIT_PCT = 80.0
ACE_MIN_ELITE       = 2


def outcome_traits(metrics: Optional[dict]) -> list[dict]:
    """Outcome trait tags from the per-pitcher league-percentile metrics."""
    traits: list[dict] = []
    m = metrics or {}

    def _get(key):
        v = m.get(key)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    k = _get("K_pct_pct")
    if k is not None:
        if k >= BAT_MISSER_PCT:
            traits.append(_trait("bat-misser", "outcome", k, k,
                                 f"strikeout rate {k:.0f}th pct"))
        elif k <= PITCH_TO_CONTACT_PCT:
            traits.append(_trait("pitch-to-contact", "outcome", k, k,
                                 f"strikeout rate {k:.0f}th pct — outs come "
                                 "from the defense"))

    gb = _get("GB_pct_pct")
    if gb is not None:
        if gb >= GROUND_BALLER_PCT:
            traits.append(_trait("ground-baller", "outcome", gb, gb,
                                 f"ground-ball rate {gb:.0f}th pct"))
        elif gb <= FLY_BALL_PRONE_PCT:
            traits.append(_trait("fly-ball prone", "outcome", gb, gb,
                                 f"ground-ball rate {gb:.0f}th pct — lives in "
                                 "the air"))

    hh = _get("HardHit_allowed_pct")
    if hh is not None and hh >= CONTACT_SUPPRESS_PCT:
        br = _get("Barrel_allowed_pct")
        traits.append(_trait("contact suppressor", "outcome", hh, hh,
                             f"hard-hit suppression {hh:.0f}th pct"
                             + (f", barrel suppression {br:.0f}th" if br is not None else "")))

    bb = _get("BB_pct_pct")   # inverted: higher = fewer walks
    if bb is not None:
        if bb >= COMMAND_ELITE_PCT:
            traits.append(_trait("elite command", "outcome", bb, bb,
                                 f"walk avoidance {bb:.0f}th pct"))
        elif bb >= COMMAND_PLUS_PCT:
            traits.append(_trait("plus command", "outcome", bb, bb,
                                 f"walk avoidance {bb:.0f}th pct"))
        elif bb <= WALK_PRONE_PCT:
            traits.append(_trait("walk prone", "outcome", bb, bb,
                                 f"walk avoidance {bb:.0f}th pct"))

    # Ace badge — elite across the board, not one loud number
    vals = {key: _get(key) for key in ACE_KEYS}
    present = {key: v for key, v in vals.items() if v is not None}
    # `ace` retired: an evaluative verdict rather than a description, and it
    # fired on 10.8% of relievers against 8.2% of starters — more "aces" in
    # bullpens than rotations. The dominance evidence lives in the individual
    # outcome tags it was composited from.
    return traits


# ---------------------------------------------------------------------------
# Role — usage shape (workload, outing length, deployment)
# ---------------------------------------------------------------------------

# Percentile gates applied within the season's starter/reliever pools
WORKHORSE_BF_PCT       = 85.0   # starter BF among starters
SHORT_OUTING_BFGS_PCT  = 15.0   # BF-per-start among starters (min GS)
HEAVY_USAGE_G_PCT      = 85.0   # appearances among relievers
MULTI_INNING_BFG_PCT   = 85.0   # BF-per-appearance among relievers
ROLE_MIN_GS            = 10     # starts needed to judge outing length
ROLE_MIN_G_RELIEF      = 15     # appearances needed to judge reliever usage
SWINGMAN_MIN_GS        = 3
SWINGMAN_MIN_RELIEF_G  = 5

# Platoon gates (raw wOBA-allowed gaps; opposite-hand advantage is normal)
PLATOON_MIN_BF        = 50
PLATOON_VULN_GAP      = 0.060   # opp-hand hitters this far better = exploitable
REVERSE_SPLIT_GAP     = 0.040   # better vs opp-hand than own = reverse
MATCHUP_DEPLOY_SHARE  = 0.60    # same-hand BF share (league ~0.50) = sheltered usage


def role_traits(
    bf: Optional[float],
    g: Optional[float],
    gs: Optional[float],
    is_starter: bool,
    role_pool: Optional[dict] = None,
    leverage: Optional[float] = None,
) -> list[dict]:
    """
    Usage-shape tags from bf/g/gs vs the season's starter/reliever pools.
    role_pool carries the pool thresholds (computed once per portrait build):
      starter_bf_hi, starter_bfgs_lo, reliever_g_hi, reliever_bfg_hi,
      reliever_lev_hi, reliever_lev_lo
    leverage — the pitcher's mean |ΔWE| per PA (empirical leverage exposure);
    high/low tags fire for relievers only (starters pitch every situation).
    """
    traits: list[dict] = []
    rp = role_pool or {}
    bf = float(bf) if bf else 0.0
    g  = float(g)  if g  else 0.0
    gs = float(gs) if gs else 0.0

    relief_g = max(g - gs, 0)
    is_swingman = gs >= SWINGMAN_MIN_GS and relief_g >= SWINGMAN_MIN_RELIEF_G
    if is_swingman:
        traits.append(_trait("swingman", "role", gs, None,
                             f"{gs:.0f} starts + {relief_g:.0f} relief "
                             "appearances — fills whatever role is open"))

    if is_starter:
        hi = rp.get("starter_bf_hi")
        workhorse = hi is not None and bf >= hi
        if workhorse:
            traits.append(_trait("workhorse", "role", bf, None,
                                 f"{bf:.0f} BF — top of the league in workload"))
        lo = rp.get("starter_bfgs_lo")
        # Suppressed for workhorses: a high-volume starter with short average
        # outings (Javier '23) is a workhorse story, not a 5-and-dive one.
        if (not workhorse and lo is not None and gs >= ROLE_MIN_GS
                and gs > 0 and (bf / gs) <= lo):
            traits.append(_trait("short-outing starter", "role", bf / gs, None,
                                 f"{bf/gs:.1f} BF per start — rarely faces a "
                                 "lineup the third time"))
    else:
        hi_g = rp.get("reliever_g_hi")
        if hi_g is not None and g >= max(hi_g, ROLE_MIN_G_RELIEF):
            traits.append(_trait("heavy usage", "role", g, None,
                                 f"{g:.0f} appearances — an every-other-day arm"))
        hi_bfg = rp.get("reliever_bfg_hi")
        # A swingman is multi-inning by definition — swingman absorbs the tag
        if (not is_swingman and hi_bfg is not None and g >= ROLE_MIN_G_RELIEF
                and g > 0 and (bf / g) >= hi_bfg):
            traits.append(_trait("multi-inning reliever", "role", bf / g, None,
                                 f"{bf/g:.1f} BF per appearance — length out "
                                 "of the pen"))

        # Leverage exposure — how much win probability rides on his PAs
        if leverage is not None:
            lev_hi, lev_lo = rp.get("reliever_lev_hi"), rp.get("reliever_lev_lo")
            if lev_hi is not None and leverage >= lev_hi:
                traits.append(_trait("high-leverage arm", "role", leverage, None,
                                     f"{leverage*100:.1f}% avg win-prob swing "
                                     "per PA — trusted with the tight spots"))
            elif lev_lo is not None and leverage <= lev_lo:
                traits.append(_trait("mop-up duty", "role", leverage, None,
                                     f"{leverage*100:.1f}% avg win-prob swing "
                                     "per PA — used when the game is decided"))

    return traits


def platoon_traits(splits: Optional[dict]) -> list[dict]:
    """
    Pitcher platoon tags from statcast splits:
      platoon-vulnerable — opp-hand hitters far better (exploitable)
      reverse split      — own-hand hitters hit him harder (unusual)
      platoon specialist — sheltered usage: same-hand BF share well above 50%
    """
    traits: list[dict] = []
    sp = splits or {}
    throws = sp.get("p_throws")
    if throws not in ("L", "R"):
        return traits

    # Handedness is an ATTRIBUTE — meaningless alone, but a staff's L/R mix is
    # a real roster-construction fact that was previously invisible.
    traits.append(_trait(
        "left-handed pitcher" if throws == "L" else "right-handed pitcher",
        "handedness", None, None,
        f"throws {'left' if throws == 'L' else 'right'}-handed"))

    same_lbl = "vs_lhh" if throws == "L" else "vs_rhh"
    opp_lbl  = "vs_rhh" if throws == "L" else "vs_lhh"
    same, opp = sp.get(same_lbl) or {}, sp.get(opp_lbl) or {}
    if (same.get("pa", 0) >= PLATOON_MIN_BF and opp.get("pa", 0) >= PLATOON_MIN_BF):
        w_same, w_opp = same.get("woba"), opp.get("woba")
        if w_same is not None and w_opp is not None:
            gap = float(w_opp) - float(w_same)   # positive = normal direction
            if gap >= PLATOON_VULN_GAP:
                traits.append(_trait("platoon-vulnerable", "platoon", gap, None,
                                     f"wOBA allowed {w_opp:.3f} vs opp-hand, "
                                     f"{w_same:.3f} vs same — a {gap:.3f} gap"))
            elif -gap >= REVERSE_SPLIT_GAP:
                traits.append(_trait("reverse split", "platoon", gap, None,
                                     f"wOBA allowed {w_same:.3f} vs same-hand, "
                                     f"{w_opp:.3f} vs opp — backwards splits"))

    share = sp.get("same_hand_bf_share")
    if share is not None and float(share) >= MATCHUP_DEPLOY_SHARE:
        traits.append(_trait("platoon specialist", "platoon", share, None,
                             f"{float(share):.0%} of BF vs same-hand hitters — "
                             "deployed for the matchup"))

    return traits


# ---------------------------------------------------------------------------
# Entry point — per-pitcher trait set (families accumulate here)
# ---------------------------------------------------------------------------

def _pool_row(pool: Optional[pd.DataFrame], pitcher_id: int) -> Optional[pd.Series]:
    if pool is None or pool.empty or "pitcher" not in pool.columns:
        return None
    row = pool[pool["pitcher"] == pitcher_id]
    return row.iloc[0] if not row.empty else None


def build_pitcher_traits(
    pitcher_id: int,
    league_mechanics:  Optional[pd.DataFrame] = None,
    league_sequencing: Optional[pd.DataFrame] = None,
    league_discipline: Optional[pd.DataFrame] = None,
    league_tunnel:     Optional[pd.DataFrame] = None,
    league_spin:       Optional[pd.DataFrame] = None,
    arsenal_profile:   Optional[dict] = None,
    metrics:           Optional[dict] = None,
    bf:                Optional[float] = None,
    g:                 Optional[float] = None,
    gs:                Optional[float] = None,
    is_starter:        Optional[bool] = None,
    role_pool:         Optional[dict] = None,
    leverage:          Optional[float] = None,
    splits:            Optional[dict] = None,
    tempo:             Optional[dict] = None,
    runner_control:    Optional[dict] = None,
) -> list[dict]:
    """
    Full trait set for one pitcher-season:
    mechanics + sequencing + deception + arsenal + outcome + role + platoon.
    """
    traits: list[dict] = []

    mech_row = _pool_row(league_mechanics, pitcher_id)
    if mech_row is not None:
        traits.extend(mechanics_traits(mech_row))

    seq_row = _pool_row(league_sequencing, pitcher_id)
    disc_row = _pool_row(league_discipline, pitcher_id)
    if seq_row is not None:
        traits.extend(sequencing_traits(seq_row, disc_row))
    elif disc_row is not None:
        # No sequencing sample but discipline data exists — FPS tag still valid
        traits.extend(sequencing_traits(pd.Series(dtype=object), disc_row))

    traits.extend(deception_traits(
        tunnel_row=_pool_row(league_tunnel, pitcher_id),
        spin_row=_pool_row(league_spin, pitcher_id),
        discipline_row=disc_row,
    ))

    if arsenal_profile is not None or metrics is not None:
        traits.extend(arsenal_traits(arsenal_profile, metrics,
                                     is_starter=is_starter))
    if metrics is not None:
        traits.extend(outcome_traits(metrics))
    if is_starter is not None:
        traits.extend(role_traits(bf, g, gs, is_starter, role_pool, leverage))
    if splits is not None:
        traits.extend(platoon_traits(splits))

    # ── tempo (mechanics) — savant empty-bases median seconds ──────────────
    t = tempo or {}
    tp = t.get("tempo_pct")
    if tp is not None and not (isinstance(tp, float) and np.isnan(tp)):
        secs = t.get("tempo_empty_s")
        if float(tp) <= QUICK_PITCHER_PCT:
            traits.append(_trait("quick pitcher", "sequencing", secs, tp,
                                 f"{secs:.1f}s between pitches, bases empty — "
                                 "among the fastest workers"))
        elif float(tp) >= SLOW_PITCHER_PCT:
            traits.append(_trait("slow pitcher", "sequencing", secs, tp,
                                 f"{secs:.1f}s between pitches, bases empty — "
                                 "among the slowest workers"))

    # ── runner control (outcome) — mid-PA advances allowed ─────────────────
    rc = runner_control or {}
    rcp = rc.get("advance_pct")
    if (rcp is not None and not (isinstance(rcp, float) and np.isnan(rcp))
            and rc.get("opps", 0) >= CONTROLS_RUNNERS_MIN_OPPS
            and float(rcp) <= CONTROLS_RUNNERS_PCT):
        rate = rc.get("advance_rate") or 0
        traits.append(_trait("controls runners", "running-game", rate, rcp,
                             f"runners moved mid-PA on just {rate:.1%} of "
                             f"{rc.get('opps', 0)} chances (steals, WP and PB "
                             "included)"))

    # ── reliability gate ──────────────────────────────────────────────────
    # Mirrors the hitter side: batters faced is the sample every rate-based
    # pitcher tag rests on. Arsenal, mechanics and role tags carry their own
    # pitch-count or appearance floors and are absent from TAG_EVIDENCE.
    traits = filter_traits(traits, bf)

    return traits
