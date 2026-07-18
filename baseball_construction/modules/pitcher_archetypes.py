"""
pitcher_archetypes.py — Starter and bullpen classification.

Starter types P1–P4 are scored independently; the type with the highest
score among those meeting all qualifying thresholds wins.

Velocity routing rules (applied before scoring):
  avg_velo > 65th AND K% > 65th → eligible for P1 only (not P2)
  K% > 60th                     → not eligible for P3
  K% > 65th                     → not eligible for P4

Classification:
  P1 — Power              (high stuff; Ks and contact-suppression scored, not gated)
  P2 — Craft Strikeout    (miss bats through deception, not velocity)
  P3 — Ground Ball Craftsman (weak contact, command-driven, low K)
  P4 — Stuff to Contact   (above-avg stuff → ground balls, not Ks)
  P5 — Power Contact      (elite GB% + above-avg velo + above-avg K; Framber Valdez type)

Bullpen:
  Collective profile scored 0–100 on 7 dimensions vs league bullpens.

All inputs are pre-normalised percentile metrics (0–100).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from philosophy import display_confidence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold / ceiling constants (percentile space, 0–100)
# ---------------------------------------------------------------------------

# P1 — Power: minimum thresholds
# Renamed from "Power Ace" — this archetype is about *style* (lives off velocity/stuff),
# not a quality tier. "Ace"-level standing is now a separate cross-archetype modifier
# (see compute_ace_tier) so elite pitchers keep their style label instead of being
# forced into a binary "Ace or not" slot.
#
# K% hard floor removed (was 58.0): a league-wide audit found a cluster of hard-throwing
# starters (e.g. Jack Leiter, Michael Kopech, Carlos Rodón's '23 return, Shane Baz,
# Bryce Miller, Luis Patiño — avg_velo_pct 65-86 but K_pct_pct only ~20-56) who were
# excluded from BOTH this archetype (K floor) and Craft Strikeout (velo ceiling),
# landing in "Unclassified" despite sharing an obvious, nameable trait: they live by
# velocity/stuff, whether or not it's missing bats yet. K% remains a heavily-weighted
# *scored* component (_score_p1), so true bat-missing aces still rise to the top —
# this just stops "stuff outpacing results" arms from falling through the cracks.
P1_MIN = {"avg_velo_pct": 60.0}

# P2 — Craft Strikeout: min thresholds + velocity CEILING
# SwStr_pct is scored but NOT a hard gate (it can be missing from DB).
# K_pct is the primary gate; if K% is very high (>=65) SwStr absence is excused.
P2_MIN          = {"K_pct_pct": 55.0}   # SwStr removed from hard gate
P2_SWSTR_MIN    = 55.0                   # soft floor: if present, must be >= this OR K% >= 65
P2_VELO_CEILING = 65.0   # avg_velo_pct must be AT OR BELOW this

# P3 — Ground Ball Craftsman: min thresholds + K% CEILING
# GB floor lowered to 35 to capture command-based contact managers (finesse types)
# who have below-average-but-not-extreme ground-ball rates.
# HardHit_allowed_pct is already Statcast-inverted (higher = better suppressor);
# it drives the score rather than a binary ceiling gate.
P3_MIN      = {"GB_pct_pct": 35.0}
P3_K_CEILING = 65.0   # K_pct_pct must be AT OR BELOW — GB craftsmen are not K pitchers
P3_K_CEILING_SOFT = 75.0   # raised ceiling when GB% is elite (see P3_ELITE_GB below)
P3_ELITE_GB = 85.0         # GB% at/above this excuses a borderline-high K% —
                           # an elite-groundball trait shouldn't be hidden behind
                           # a razor-thin miss on the K ceiling (e.g. K%=65.3 vs 65.0)

# P4 — Stuff to Contact: min thresholds + K% CEILING
P4_MIN       = {"avg_velo_pct": 55.0, "GB_pct_pct": 38.0}
P4_K_CEILING = 65.0      # K_pct_pct must be AT OR BELOW

# P5 — Power Contact: elite GB% + above-avg velo + above-avg K (Framber Valdez type)
# These pitchers get Ks alongside extreme ground balls — not a "craftsman" per se.
# Higher GB floor (65th) distinguishes from P4; no K ceiling unlike P3/P4.
P5_MIN = {"avg_velo_pct": 55.0, "GB_pct_pct": 65.0, "K_pct_pct": 55.0}

# P6 — Finesse Control: exceptional command + low K + below-avg velo (Kyle Hendricks type)
# Pitchers who survive through precision location, not stuff or groundballs.
# The single hard gate is command (BB_pct_pct is inverted: higher = fewer walks = better).
P6_MIN         = {"BB_pct_pct": 70.0}   # must have excellent command (top-30% low-walk rate)
P6_K_CEILING   = 50.0                   # below-average K rate (not a strikeout pitcher)
P6_VELO_CEILING = 60.0                  # below-average velocity (not a power pitcher)

# Platoon vulnerability threshold (raw wOBA difference, not percentile)
PLATOON_VULN_THRESHOLD = 0.040

# Arsenal depth thresholds
DEEP_ARSENAL_COUNT     = 3   # 3+ pitch types with above-avg run value
ONE_DIMENSIONAL_COUNT  = 1

# Confidence ceiling for pitcher archetypes
PITCHER_CONFIDENCE_CEILING = 0.35

# "Ace" tier — a cross-archetype quality badge (see compute_ace_tier).
# Archetype answers HOW a pitcher gets outs; Ace answers whether they're
# elite at it. Both Verlander-prime (Power) and peak-Maddux (Finesse Control)
# were aces via opposite styles — Ace is layered on top, not a competing type.
ACE_SCORE_THRESHOLD     = 78.0   # winning archetype's composite score floor
ACE_DOMINANCE_KEYS      = [
    "K_pct_pct", "BB_pct_pct", "HardHit_allowed_pct",
    "SwStr_pct_pct", "GB_pct_pct", "CSW_pct_pct",
]
ACE_ELITE_TRAIT_PCT     = 80.0   # a trait counts as "elite" at/above this percentile
ACE_MIN_ELITE_TRAITS    = 2      # need 2+ elite traits — guards against one big number
                                 # propping up an otherwise-average season into "Ace"


# ---------------------------------------------------------------------------
# Generic score helper (same as philosophy.py pattern)
# ---------------------------------------------------------------------------

def _weighted_score(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> tuple[Optional[float], float]:
    """
    Weighted mean of available metrics. Returns (score, coverage).
    Coverage = fraction of weight with data.
    """
    total_w = sum(weights.values())
    avail_w = 0.0
    wsum    = 0.0
    for metric, weight in weights.items():
        val = metrics.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        wsum    += float(val) * weight
        avail_w += weight
    if avail_w == 0:
        return None, 0.0
    return wsum / avail_w, avail_w / total_w


def _meets_thresholds(
    metrics: dict[str, float],
    minimums: dict[str, float],
) -> tuple[bool, float]:
    """
    Returns (passes, mean_margin_above_threshold / 100).
    Fails if any required metric is missing.
    """
    margins = []
    for metric, threshold in minimums.items():
        val = metrics.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return False, 0.0
        margins.append((float(val) - threshold) / 100.0)
    if not margins:
        return False, 0.0
    return all(m >= 0 for m in margins), float(np.mean(margins))


def _below_ceiling(
    metrics: dict[str, float],
    metric: str,
    ceiling: float,
) -> bool:
    """Return True if metric <= ceiling (or metric is missing → allow)."""
    val = metrics.get(metric)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return True  # conservative: don't disqualify for missing data
    return float(val) <= ceiling


def _blend_confidence(margin: float, score: Optional[float]) -> float:
    """
    Blend gate-clearance margin with overall fit quality.

    `margin` (mean distance above the qualifying threshold, /100) answers
    "how comfortably did this pitcher clear the minimum bar to be considered
    this archetype at all?" It does NOT capture how good a fit they are
    overall — a pitcher can clear a single threshold by a mile while ranking
    near the bottom of the league on other traits that define the archetype
    (e.g. a "Ground Ball Craftsman" who easily clears the GB% floor but is a
    2nd-percentile contact-suppressor, despite HardHit_allowed_pct being a
    quarter of that archetype's score).

    `score` is the weighted-composite fit (0-100, ~50 = league average).
    We fold in how far above/below average that composite sits so that
    "comfortably cleared the gate" + "below-average overall fit" doesn't
    still read as ~100% confidence.
    """
    quality = 0.0 if score is None else max(-0.5, (score - 50.0) / 100.0)
    return max(0.0, 0.5 * margin + 0.5 * quality)


# ---------------------------------------------------------------------------
# P1 — Power
# ---------------------------------------------------------------------------

def _score_p1(metrics: dict[str, float]) -> Optional[float]:
    weights = {
        "stuff_composite_pct":  0.30,  # pre-computed: (velo + spin + spin_eff) blend
        "K_pct_pct":            0.25,
        "SwStr_pct_pct":        0.20,
        "CSW_pct_pct":          0.15,
        "HardHit_allowed_pct":  0.10,  # Statcast rank: higher = better at preventing hard contact
    }
    score, _ = _weighted_score(metrics, weights)
    return score


def classify_p1(metrics: dict[str, float]) -> Optional[dict]:
    """
    Power — lives off velocity/stuff (avg_velo_pct >= 60th).
    No ceiling constraints. K%, SwStr%, CSW%, and contact-suppression are
    *scored* (see _score_p1) rather than gated, so this archetype spans
    everything from "stuff that's translating into dominance" to "big arm,
    results still developing" — both are the same style, differentiated by
    score/confidence rather than forced into different labels.
    """
    passes, margin = _meets_thresholds(metrics, P1_MIN)
    if not passes:
        return None

    score = _score_p1(metrics)
    if score is None:
        return None

    return {
        "type":               "Power",
        "type_code":          "P1",
        "score":              score,
        "raw_confidence":     _blend_confidence(margin, score),
        "display_confidence": display_confidence(_blend_confidence(margin, score), PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# P2 — Craft Strikeout
# ---------------------------------------------------------------------------

def _score_p2(metrics: dict[str, float]) -> Optional[float]:
    weights = {
        "SpinEfficiency_pct":   0.25,
        "TunnelScore_pct":      0.25,
        "SwStr_per_velo_pct":   0.20,  # whiff rate adjusted for velocity
        "K_pct_pct":            0.15,
        "FirstPitchStrike_pct": 0.15,
    }
    score, _ = _weighted_score(metrics, weights)
    return score


def classify_p2(metrics: dict[str, float]) -> Optional[dict]:
    """
    Craft Strikeout — K% >= 55th AND velo <= 65th.
    SwStr% is scored but not a hard gate (can be missing from DB).
    If SwStr% IS present, it must be >= 55 UNLESS K% >= 65 (high-K excuses low/missing SwStr).
    """
    # Velocity ceiling check
    if not _below_ceiling(metrics, "avg_velo_pct", P2_VELO_CEILING):
        return None

    # K% hard gate
    passes, margin = _meets_thresholds(metrics, P2_MIN)
    if not passes:
        return None

    # Soft SwStr check: if present, must be >= 55 unless K% >= 65
    swstr_val = metrics.get("SwStr_pct_pct")
    if swstr_val is not None and not (isinstance(swstr_val, float) and np.isnan(swstr_val)):
        if swstr_val < P2_SWSTR_MIN and metrics.get("K_pct_pct", 0) < 65.0:
            return None  # Both SwStr and K% are below threshold

    score = _score_p2(metrics)
    if score is None:
        return None

    return {
        "type":               "Craft Strikeout",
        "type_code":          "P2",
        "score":              score,
        "raw_confidence":     _blend_confidence(margin, score),
        "display_confidence": display_confidence(_blend_confidence(margin, score), PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# P3 — Ground Ball Craftsman
# ---------------------------------------------------------------------------

def _score_p3(metrics: dict[str, float]) -> Optional[float]:
    weights = {
        "GB_pct_pct":          0.35,
        "HardHit_allowed_pct": 0.25,  # Statcast rank: higher = better at preventing hard contact
        "Barrel_allowed_pct":  0.20,  # Statcast rank: higher = better at preventing barrels
        "BB_pct_pct":          0.10,  # (already inverted: higher = fewer walks)
        "K_inv_pct":           0.10,  # (100 - K_pct_pct) — low K confirms GB type
    }
    score, _ = _weighted_score(metrics, weights)
    return score


def classify_p3(metrics: dict[str, float]) -> Optional[dict]:
    """
    Ground Ball Craftsman — GB% >= 40th, K% <= 65th.

    HardHit_allowed_pct is Statcast's already-inverted rank (higher = better at
    preventing hard contact).  We use it directly in scoring rather than gating
    on a ceiling — the scoring naturally rewards strong contact suppression.

    K% ceiling has a soft override: a pitcher with an elite (>=85th pct) GB%
    is allowed up to P3_K_CEILING_SOFT instead of the normal ceiling. Without
    this, a pitcher whose defining trait is a top-7% groundball rate could be
    excluded from this archetype by missing the K ceiling by a fraction of a
    percentile point — and end up labeled by a trait that isn't their game at
    all (e.g. "Craft Strikeout"). Mirrors the soft-floor pattern used in P2.
    """
    # K% ceiling — ground-ball craftsmen are not high-strikeout pitchers,
    # unless their groundball rate is so elite it defines them regardless.
    gb_val = metrics.get("GB_pct_pct")
    k_ceiling = P3_K_CEILING
    if gb_val is not None and not (isinstance(gb_val, float) and np.isnan(gb_val)):
        if float(gb_val) >= P3_ELITE_GB:
            k_ceiling = P3_K_CEILING_SOFT
    if not _below_ceiling(metrics, "K_pct_pct", k_ceiling):
        return None

    passes, margin = _meets_thresholds(metrics, P3_MIN)
    if not passes:
        return None

    score = _score_p3(metrics)
    if score is None:
        return None

    return {
        "type":               "Ground Ball Craftsman",
        "type_code":          "P3",
        "score":              score,
        "raw_confidence":     _blend_confidence(margin, score),
        "display_confidence": display_confidence(_blend_confidence(margin, score), PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# P4 — Stuff to Contact
# ---------------------------------------------------------------------------

def _score_p4(metrics: dict[str, float]) -> Optional[float]:
    weights = {
        "avg_velo_pct":        0.25,
        "GB_pct_pct":          0.25,
        "HardHit_allowed_pct": 0.25,  # Statcast rank: higher = better at preventing hard contact
        "K_pct_pct":           0.15,  # moderate — not a strikeout pitcher
        "Barrel_allowed_pct":  0.10,  # Statcast rank: higher = better at preventing barrels
    }
    score, _ = _weighted_score(metrics, weights)
    return score


def classify_p4(metrics: dict[str, float]) -> Optional[dict]:
    """
    Stuff to Contact — above-avg velo + GB%, K% <= 65th.
    """
    if not _below_ceiling(metrics, "K_pct_pct", P4_K_CEILING):
        return None

    passes, margin = _meets_thresholds(metrics, P4_MIN)
    if not passes:
        return None

    score = _score_p4(metrics)
    if score is None:
        return None

    return {
        "type":               "Stuff to Contact",
        "type_code":          "P4",
        "score":              score,
        "raw_confidence":     _blend_confidence(margin, score),
        "display_confidence": display_confidence(_blend_confidence(margin, score), PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# P5 — Power Contact
# ---------------------------------------------------------------------------

def _score_p5(metrics: dict[str, float]) -> Optional[float]:
    weights = {
        "GB_pct_pct":          0.35,  # defining trait — elite ground-ball rate
        "avg_velo_pct":        0.25,  # velocity powers the sinker
        "K_pct_pct":           0.20,  # above-avg Ks distinguish from P3 craftsmen
        "HardHit_allowed_pct": 0.20,  # quality of contact suppressed
    }
    score, _ = _weighted_score(metrics, weights)
    return score


def classify_p5(metrics: dict[str, float]) -> Optional[dict]:
    """
    Power Contact — elite GB% (>=65th) + above-avg velo + above-avg K%.
    Captures sinker-ballers like Framber Valdez who get Ks alongside
    extreme ground balls.  No K% ceiling (unlike P3/P4).
    """
    passes, margin = _meets_thresholds(metrics, P5_MIN)
    if not passes:
        return None

    score = _score_p5(metrics)
    if score is None:
        return None

    return {
        "type":               "Power Contact",
        "type_code":          "P5",
        "score":              score,
        "raw_confidence":     _blend_confidence(margin, score),
        "display_confidence": display_confidence(_blend_confidence(margin, score), PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# P6 — Finesse Control
# ---------------------------------------------------------------------------

def _score_p6(metrics: dict[str, float]) -> Optional[float]:
    weights = {
        "BB_pct_pct":          0.40,  # command is the defining trait (inverted: high = few walks)
        "Zone_pct_pct":        0.25,  # zone rate as secondary command signal
        "HardHit_allowed_pct": 0.20,  # suppresses hard contact without Ks
        "K_inv_pct":           0.15,  # low K confirms the style (K_inv = 100 - K_pct_pct)
    }
    score, _ = _weighted_score(metrics, weights)
    return score


def classify_p6(metrics: dict[str, float]) -> Optional[dict]:
    """
    Finesse Control — exceptional command (BB_pct_pct >= 70) + low K + below-avg velo.
    Captures Kyle Hendricks / Tommy Milone types who survive through precision location.
    Not eligible if K or velo are above-average (those pitchers fit P1–P5 better).
    """
    # Velocity ceiling — power pitchers are not P6
    if not _below_ceiling(metrics, "avg_velo_pct", P6_VELO_CEILING):
        return None

    # K ceiling — strikeout pitchers are not P6
    if not _below_ceiling(metrics, "K_pct_pct", P6_K_CEILING):
        return None

    # Command hard gate — must have truly excellent walk avoidance
    passes, margin = _meets_thresholds(metrics, P6_MIN)
    if not passes:
        return None

    score = _score_p6(metrics)
    if score is None:
        return None

    return {
        "type":               "Finesse Control",
        "type_code":          "P6",
        "score":              score,
        "raw_confidence":     _blend_confidence(margin, score),
        "display_confidence": display_confidence(_blend_confidence(margin, score), PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# Primary classification
# ---------------------------------------------------------------------------

def classify_starter(metrics: dict[str, float]) -> dict:
    """
    Classify a starter into one primary archetype.

    Evaluates all four types; returns the qualifying type with the highest
    composite score. Velocity routing rules are enforced inside each
    classify_* function via ceiling checks.

    Falls back to 'Unclassified' if no type qualifies.
    """
    candidates = {}

    p1 = classify_p1(metrics)
    if p1:
        candidates["P1"] = p1

    # P2 not eligible if avg_velo > 65th AND K% > 65th (both route to P1)
    if not (
        metrics.get("avg_velo_pct", 0) > 65
        and metrics.get("K_pct_pct",  0) > 65
    ):
        p2 = classify_p2(metrics)
        if p2:
            candidates["P2"] = p2

    p3 = classify_p3(metrics)
    if p3:
        candidates["P3"] = p3

    p4 = classify_p4(metrics)
    if p4:
        candidates["P4"] = p4

    p5 = classify_p5(metrics)
    if p5:
        candidates["P5"] = p5

    p6 = classify_p6(metrics)
    if p6:
        candidates["P6"] = p6

    if not candidates:
        return {
            "type":               "Unclassified",
            "type_code":          "U0",
            "score":              None,
            "raw_confidence":     0.0,
            "display_confidence": 0,
        }

    # Highest score among qualifying types wins
    winner = max(candidates.values(), key=lambda x: x["score"])
    return winner


# ---------------------------------------------------------------------------
# Starter modifiers
# ---------------------------------------------------------------------------

def compute_platoon_vulnerability(
    woba_vs_same: float,
    woba_vs_opposite: float,
) -> dict:
    """
    Platoon vulnerability.

    For RHP: woba_vs_LHH - woba_vs_RHH  (opposite - same handedness)
    For LHP: woba_vs_RHH - woba_vs_LHH

    Caller is responsible for providing (same-hand wOBA, opposite-hand wOBA).

    Returns:
        vulnerable — True if difference > 0.040
        difference — raw wOBA gap
    """
    diff = woba_vs_opposite - woba_vs_same
    return {
        "vulnerable":  diff > PLATOON_VULN_THRESHOLD,
        "difference":  diff,
    }


def compute_command_profile(metrics: dict[str, float]) -> str:
    """
    Command profile: 'Elite' | 'Poor' | 'Average'.

    BB_pct_pct is INVERTED: higher = fewer walks = better command.
    Zone_pct_pct used as a secondary signal but NOT required for elite —
    power pitchers (Cole, Verlander) have low BB% without high zone%.

    Elite: BB_pct_pct ≥ 80th  (very few walks — top 20% historically)
    Good:  BB_pct_pct ≥ 65th  (above-average command)
    Poor:  BB_pct_pct < 30th  (lots of walks)
    """
    bb = metrics.get("BB_pct_pct")

    if bb is None:
        return "Average"
    if bb >= 80:
        return "Elite"
    if bb >= 65:
        return "Good"
    if bb < 30:
        return "Poor"
    return "Average"


def compute_ace_tier(primary: Optional[dict], metrics: dict[str, float]) -> dict:
    """
    'Ace' tier — cross-archetype quality badge, layered on top of the
    style archetype rather than competing with it.

    Why a modifier and not its own archetype: classify_starter picks exactly
    ONE winning style per pitcher. If "Ace" were its own archetype, an elite
    pitcher would be forced into a binary "Ace or [style]" choice — losing
    the description of HOW they dominate (Verlander-prime via swing-and-miss
    Power, peak-Maddux via Finesse Control command — both aces, opposite
    styles). Layering Ace on top preserves both signals: "Power · Ace" tells
    you everything; "Power" alone or "Ace" alone tells you only half.

    Requires BOTH:
      - the winning archetype's composite score sits in the elite range
        (>= ACE_SCORE_THRESHOLD), AND
      - at least ACE_MIN_ELITE_TRAITS individual traits are independently
        elite (>= ACE_ELITE_TRAIT_PCT) — guards against one big number
        (e.g. a velocity outlier) propping an otherwise-average season into
        "Ace" standing.

    Returns {"tier": "Ace" | None, "elite_traits": [...]}
    """
    if primary is None or primary.get("score") is None:
        return {"tier": None, "elite_traits": []}

    score = primary["score"]
    elite_traits = [
        k for k in ACE_DOMINANCE_KEYS
        if metrics.get(k) is not None
        and not (isinstance(metrics[k], float) and np.isnan(metrics[k]))
        and metrics[k] >= ACE_ELITE_TRAIT_PCT
    ]

    is_ace = score >= ACE_SCORE_THRESHOLD and len(elite_traits) >= ACE_MIN_ELITE_TRAITS
    return {
        "tier":         "Ace" if is_ace else None,
        "elite_traits": elite_traits,
    }


def compute_arsenal_depth(
    pitch_type_run_values: dict[str, float],
    usage_threshold: float = 10.0,
    league_mean_rv: float = 0.0,
) -> dict:
    """
    Arsenal depth — count pitch types thrown > 10% with above-average run value.

    Args:
        pitch_type_run_values: {pitch_type: run_value_per_100}
                               run_value > league_mean = above average
        usage_threshold:  minimum usage % to count a pitch (default 10%)
        league_mean_rv:   league-average run value benchmark

    Returns:
        above_avg_count — int
        label           — 'Deep Arsenal' | 'One Dimensional' | 'Standard'
        pitch_breakdown — {pitch_type: above_avg bool}
    """
    breakdown = {}
    count = 0
    for pt, rv in pitch_type_run_values.items():
        above = rv > league_mean_rv
        breakdown[pt] = above
        if above:
            count += 1

    if count >= DEEP_ARSENAL_COUNT:
        label = "Deep Arsenal"
    elif count <= ONE_DIMENSIONAL_COUNT:
        label = "One Dimensional"
    else:
        label = "Standard"

    return {
        "above_avg_count": count,
        "label":           label,
        "pitch_breakdown": breakdown,
    }


# ---------------------------------------------------------------------------
# Full starter profile
# ---------------------------------------------------------------------------

def build_starter_profile(
    player_id:              int,
    metrics:                dict[str, float],
    handedness:             str = "R",
    woba_vs_same:           Optional[float] = None,
    woba_vs_opposite:       Optional[float] = None,
    pitch_type_run_values:  Optional[dict[str, float]] = None,
    season:                 Optional[int] = None,
    pitch_mix_df=           None,
) -> dict:
    """
    Full starter classification for one pitcher × season.

    Args:
        player_id:             MLBAM ID
        metrics:               pre-normalised metric dict (0–100 percentiles)
        handedness:            'R' or 'L'
        woba_vs_same:          wOBA allowed vs same-hand batters (raw)
        woba_vs_opposite:      wOBA allowed vs opposite-hand batters (raw)
        pitch_type_run_values: {pitch_type: run_value_per_100} for arsenal depth
        season:                season year — used to load pitch mix for arsenal profile
        pitch_mix_df:          pre-loaded pitch_mix DataFrame (avoids reload per-pitcher)

    Returns:
        {
            player_id, handedness,
            primary:         outcome-based archetype dict (P1-P6, kept for compatibility)
            arsenal_profile: weapon-based profile (fastball, out_pitch, approach, depth…)
            modifiers:       {platoon, command, arsenal, ace}
        }
    """
    primary = classify_starter(metrics)

    # Platoon vulnerability
    if woba_vs_same is not None and woba_vs_opposite is not None:
        platoon = compute_platoon_vulnerability(woba_vs_same, woba_vs_opposite)
    else:
        platoon = {"vulnerable": None, "difference": None}

    command = compute_command_profile(metrics)
    ace     = compute_ace_tier(primary, metrics)

    if pitch_type_run_values is not None:
        arsenal = compute_arsenal_depth(pitch_type_run_values)
    else:
        arsenal = {"above_avg_count": None, "label": None, "pitch_breakdown": {}}

    # Weapon-based arsenal profile (Tier 2 — from Statcast pitch-type aggregation)
    arsenal_profile = None
    if season is not None:
        try:
            from pitch_mix import build_arsenal_profile, load_pitch_mix
            pm = pitch_mix_df if pitch_mix_df is not None else load_pitch_mix(season)
            arsenal_profile = build_arsenal_profile(
                player_id, season, pm,
                tunnel_pct=metrics.get("TunnelScore_pct"),
            )
            if arsenal_profile is not None:
                # Don't store the raw DataFrame inside the portrait — too large
                arsenal_profile = {k: v for k, v in arsenal_profile.items()
                                   if k != "pitch_rows"}
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "arsenal_profile failed for %s season %s: %s", player_id, season, exc
            )

    return {
        "player_id":       player_id,
        "handedness":      handedness,
        "primary":         primary,
        "arsenal_profile": arsenal_profile,
        "modifiers": {
            "platoon": platoon,
            "command": command,
            "arsenal": arsenal,
            "ace":     ace,
        },
    }


# ---------------------------------------------------------------------------
# Bullpen collective profile
# ---------------------------------------------------------------------------

def build_bullpen_profile(
    team:           str,
    season:         int,
    metrics:        dict[str, float],
) -> dict:
    """
    Bullpen collective profile — 7 dimensions, each 0–100 vs league.

    Expected metric keys (pre-normalised 0–100):
        velocity_pct        — mean fastball velo, all relievers
        pitch_identity_pct  — dominant pitch type by usage %
        attack_pct          — Zone% across all relievers
        K_pct_pct           — strikeout rate (K-driven out mechanism)
        GB_pct_pct          — ground ball rate (GB-driven out mechanism)
        HardHit_inv_pct     — 100 - HardHit% allowed
        Barrel_inv_pct      — 100 - Barrel% allowed
        LevWPA_conc_pct     — WPA concentration (one arm vs committee)
        Platoon_bal_pct     — LHP/RHP ratio + split effectiveness

    Returns scores dict plus derived fields:
        out_mechanism       — 'Strikeout-driven' | 'Ground-ball-driven' | 'Mixed'
        leverage_structure  — 'One-arm dominant' | 'Committee'
    """
    dimension_weights = {
        "velocity":           {"avg_velo_pct":        1.0},
        "attack_philosophy":  {"Zone_pct_pct":        1.0},
        # HardHit_allowed_pct / Barrel_allowed_pct are Statcast ranks: higher = better suppressor
        "damage_prevention":  {"HardHit_allowed_pct": 0.50, "Barrel_allowed_pct": 0.50},
        "platoon_balance":    {"Platoon_bal_pct":      1.0},
        "leverage_structure": {"LevWPA_conc_pct":      1.0},
        # out_mechanism scored separately (K vs GB comparison)
        "K_out_mechanism":    {"K_pct_pct":            1.0},
        "GB_out_mechanism":   {"GB_pct_pct":           1.0},
    }

    scores: dict[str, Optional[float]] = {}
    for dim, weights in dimension_weights.items():
        s, _ = _weighted_score(metrics, weights)
        scores[dim] = s

    # Derive out mechanism label
    k_score  = scores.get("K_out_mechanism")
    gb_score = scores.get("GB_out_mechanism")
    if k_score is not None and gb_score is not None:
        if k_score > gb_score + 15:
            out_mechanism = "Strikeout-driven"
        elif gb_score > k_score + 15:
            out_mechanism = "Ground-ball-driven"
        else:
            out_mechanism = "Mixed"
    else:
        out_mechanism = None

    # Leverage structure label
    lev = scores.get("leverage_structure")
    leverage_structure = (
        "One-arm dominant" if lev is not None and lev > 65
        else "Committee" if lev is not None and lev <= 65
        else None
    )

    return {
        "team":               team,
        "season":             season,
        "scores":             scores,
        "out_mechanism":      out_mechanism,
        "leverage_structure": leverage_structure,
    }


# ---------------------------------------------------------------------------
# Batch: list of profiles → DataFrame
# ---------------------------------------------------------------------------

def starters_to_dataframe(profiles: list[dict]) -> pd.DataFrame:
    """
    Flatten a list of starter profiles to one row per player.
    """
    rows = []
    for p in profiles:
        pri = p["primary"]
        mod = p["modifiers"]
        rows.append({
            "player_id":          p["player_id"],
            "handedness":         p["handedness"],
            "type":               pri["type"],
            "type_code":          pri["type_code"],
            "score":              pri.get("score"),
            "display_confidence": pri.get("display_confidence"),
            "platoon_vulnerable": mod["platoon"]["vulnerable"],
            "platoon_diff":       mod["platoon"]["difference"],
            "command":            mod["command"],
            "arsenal_label":      mod["arsenal"]["label"],
            "arsenal_count":      mod["arsenal"]["above_avg_count"],
        })
    return pd.DataFrame(rows)
