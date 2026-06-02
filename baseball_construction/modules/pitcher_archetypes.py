"""
pitcher_archetypes.py — Starter and bullpen classification.

Starter types P1–P4 are scored independently; the type with the highest
score among those meeting all qualifying thresholds wins.

Velocity routing rules (applied before scoring):
  avg_velo > 65th AND K% > 65th → eligible for P1 only (not P2)
  K% > 60th                     → not eligible for P3
  K% > 65th                     → not eligible for P4

Classification:
  P1 — Power Ace          (high stuff + miss bats)
  P2 — Craft Strikeout    (miss bats through deception, not velocity)
  P3 — Ground Ball Craftsman (weak contact, command-driven, low K)
  P4 — Stuff to Contact   (above-avg stuff → ground balls, not Ks)
  P5 — Power Sinker       (elite GB% + above-avg velo + above-avg K; Framber Valdez type)

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

# P1 — Power Ace: minimum thresholds
# SwStr is used in scoring but NOT a hard gate — K% already confirms miss-bat ability.
# K gate lowered to 58 (above-average) to capture developing power pitchers with above-avg velo;
# the scoring function still heavily rewards elite K% and SwStr, keeping true aces at the top.
P1_MIN = {"avg_velo_pct": 60.0, "K_pct_pct": 58.0}

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

# P4 — Stuff to Contact: min thresholds + K% CEILING
P4_MIN       = {"avg_velo_pct": 55.0, "GB_pct_pct": 38.0}
P4_K_CEILING = 65.0      # K_pct_pct must be AT OR BELOW

# P5 — Power Sinker: elite GB% + above-avg velo + above-avg K (Framber Valdez type)
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


# ---------------------------------------------------------------------------
# P1 — Power Ace
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
    Power Ace — must meet all P1_MIN thresholds.
    No ceiling constraints.
    """
    passes, margin = _meets_thresholds(metrics, P1_MIN)
    if not passes:
        return None

    score = _score_p1(metrics)
    if score is None:
        return None

    return {
        "type":               "Power Ace",
        "type_code":          "P1",
        "score":              score,
        "raw_confidence":     margin,
        "display_confidence": display_confidence(margin, PITCHER_CONFIDENCE_CEILING),
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
        "raw_confidence":     margin,
        "display_confidence": display_confidence(margin, PITCHER_CONFIDENCE_CEILING),
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
    """
    # K% ceiling — ground-ball craftsmen are not high-strikeout pitchers
    if not _below_ceiling(metrics, "K_pct_pct", P3_K_CEILING):
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
        "raw_confidence":     margin,
        "display_confidence": display_confidence(margin, PITCHER_CONFIDENCE_CEILING),
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
        "raw_confidence":     margin,
        "display_confidence": display_confidence(margin, PITCHER_CONFIDENCE_CEILING),
    }


# ---------------------------------------------------------------------------
# P5 — Power Sinker
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
    Power Sinker — elite GB% (>=65th) + above-avg velo + above-avg K%.
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
        "type":               "Power Sinker",
        "type_code":          "P5",
        "score":              score,
        "raw_confidence":     margin,
        "display_confidence": display_confidence(margin, PITCHER_CONFIDENCE_CEILING),
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
        "raw_confidence":     margin,
        "display_confidence": display_confidence(margin, PITCHER_CONFIDENCE_CEILING),
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

    Returns:
        {
            player_id, handedness,
            primary: archetype dict,
            modifiers: {platoon, command, arsenal},
        }
    """
    primary = classify_starter(metrics)

    # Platoon vulnerability
    if woba_vs_same is not None and woba_vs_opposite is not None:
        platoon = compute_platoon_vulnerability(woba_vs_same, woba_vs_opposite)
    else:
        platoon = {"vulnerable": None, "difference": None}

    command = compute_command_profile(metrics)

    if pitch_type_run_values is not None:
        arsenal = compute_arsenal_depth(pitch_type_run_values)
    else:
        arsenal = {"above_avg_count": None, "label": None, "pitch_breakdown": {}}

    return {
        "player_id":  player_id,
        "handedness": handedness,
        "primary":    primary,
        "modifiers": {
            "platoon": platoon,
            "command": command,
            "arsenal": arsenal,
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
