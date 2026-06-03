"""
hitter_archetypes.py — Classify hitters into primary archetypes with modifiers.

Classification order (most restrictive first):
  1. Complete Hitter  — all 5 thresholds met
  2. Three True Outcomes — all 3 thresholds met; if display_confidence < 30 → spectrum
  3. Spectrum → Contact (< 45) | Balanced (45–55) | Power (> 55)
     Balanced confidence = proximity to 50 (closer = more balanced = higher)

Modifiers (independent of primary type):
  • Free Swinger  — all 3 conditions met
  • Speed         — sprint_speed hard threshold + any behavioral gate
  • Disruptive    — net positive SB run value + high attempt rate
  • Chaotic       — net negative SB run value + high attempt rate

Inputs: pre-normalised metric dict (0–100 percentiles) plus raw values
        for hard thresholds (sprint_speed in ft/sec, raw SB/CS counts).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — thresholds in percentile space (0–100)
# ---------------------------------------------------------------------------

COMPLETE_THRESHOLDS: dict[str, float] = {
    "xwOBA":       70.0,   # overall expected offensive quality (replaces wRC+)
    "OBP":         70.0,   # on-base ability
    "ISO":         70.0,   # real power
    "BB_pct":      65.0,   # plate discipline
    "Barrel_pct":  65.0,   # consistent hard contact (replaces ZContact%)
    "Contact_pct": 55.0,   # ability to make contact (the Schwarber gate)
}

TTO_THRESHOLDS: dict[str, float] = {
    "K_pct_raw": 55.0,   # above average K rate (non-inverted pct — higher = more Ks)
    "BB_pct":    65.0,   # clearly above average walks
    "ISO":       65.0,   # clearly above average power
}

FREE_SWINGER_THRESHOLDS: dict[str, float] = {
    "FPS_pct":    65.0,   # first-pitch swing % > 65th pct
    "OSwing_pct": 60.0,   # chase rate > 60th pct
    "BB_inv_pct": 65.0,   # walk rate < 35th pct → caller inverts → inv pct > 65
}

SPRINT_SPEED_HARD_THRESHOLD = 28.0   # ft/sec

# Disruptiveness — note: raw values required (not percentiles)
DISRUPTIVE_MIN_ATTEMPTS = 8          # prorated by games played

# Spectrum grey zone
GREY_ZONE_LOW  = 45.0
GREY_ZONE_HIGH = 55.0

# Confidence ceiling for spectrum types
SPECTRUM_CONFIDENCE_CEILING = 1.0

# ---------------------------------------------------------------------------
# Imports from shared modules
# ---------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from philosophy import display_confidence


# ---------------------------------------------------------------------------
# Step 1 — Threshold checks
# ---------------------------------------------------------------------------

def _check_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, float, dict[str, float]]:
    """
    Check whether all thresholds are met. Return (passes, raw_confidence, margins).

    passes         — True if every metric meets its threshold
    raw_confidence — mean margin above threshold across all required metrics,
                     divided by 100 to bring into [0, 1] range
    margins        — {metric: (observed - threshold) / 100}
    """
    margins = {}
    missing = []

    for metric, threshold in thresholds.items():
        val = metrics.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing.append(metric)
            continue
        margins[metric] = (float(val) - threshold) / 100.0

    if missing:
        log.debug("_check_thresholds: missing metrics %s — treating as failed", missing)
        return False, 0.0, margins

    passes         = all(m >= 0 for m in margins.values())
    raw_confidence = np.mean(list(margins.values())) if margins else 0.0
    return passes, raw_confidence, margins


# ---------------------------------------------------------------------------
# Step 2 — Complete Hitter
# ---------------------------------------------------------------------------

def classify_complete(metrics: dict[str, float]) -> Optional[dict]:
    """
    Type 1 — Complete Hitter.

    All 5 COMPLETE_THRESHOLDS must be met.
    Returns archetype dict or None if not Complete.
    """
    passes, raw_conf, margins = _check_thresholds(metrics, COMPLETE_THRESHOLDS)
    if not passes:
        return None

    disp_conf = display_confidence(raw_conf, ceiling=0.35)
    return {
        "type":               "Complete Hitter",
        "type_code":          "T1",
        "raw_confidence":     raw_conf,
        "display_confidence": disp_conf,
        "spectrum_score":     None,
        "grey_zone":          False,
        "margins":            margins,
    }


# ---------------------------------------------------------------------------
# Step 3 — Three True Outcomes
# ---------------------------------------------------------------------------

def classify_tto(metrics: dict[str, float]) -> Optional[dict]:
    """
    Type 2 — Three True Outcomes.

    All 3 TTO_THRESHOLDS must be met.
    If display_confidence < 30 → route to spectrum (return None).
    """
    passes, raw_conf, margins = _check_thresholds(metrics, TTO_THRESHOLDS)
    if not passes:
        return None

    disp_conf = display_confidence(raw_conf, ceiling=0.35)
    if disp_conf < 30:
        log.debug(
            "classify_tto: thresholds met but display_confidence=%d < 30 "
            "→ routing to spectrum", disp_conf
        )
        return None

    return {
        "type":               "Three True Outcomes",
        "type_code":          "T2",
        "raw_confidence":     raw_conf,
        "display_confidence": disp_conf,
        "spectrum_score":     None,
        "grey_zone":          False,
        "margins":            margins,
    }


# ---------------------------------------------------------------------------
# Step 4 — Spectrum score
# ---------------------------------------------------------------------------

def compute_spectrum(metrics: dict[str, float]) -> Optional[float]:
    """
    Compute the power-contact spectrum score 0–100.

      0   = extreme contact (Luis Arraez)
      50  = balanced middle
      100 = extreme power (Joey Gallo)

    Returns None if required metrics are unavailable.
    """
    power_weights   = {"ISO_pct": 0.30, "HR_FB_pct": 0.25, "Barrel_pct": 0.25, "EV_90_pct": 0.20}
    # K_pct_raw is the non-inverted K% percentile (higher = more strikeouts).
    # The spectrum code inverts it below (100 - val) so higher raw K% → lower contact score.
    contact_weights = {"Contact_pct": 0.25, "K_pct_raw": 0.20, "AVG_pct": 0.20, "OBP_ISO_gap_pct": 0.10}

    # Power score: weighted sum of power metrics (missing → 0 weight)
    power_sum = power_w = 0.0
    for k, w in power_weights.items():
        v = metrics.get(k)
        if v is not None and not np.isnan(float(v)):
            power_sum += float(v) * w
            power_w   += w

    # Contact score: K_pct_raw is inverted here (lower raw K% = more contact)
    contact_sum = contact_w = 0.0
    for k, w in contact_weights.items():
        v = metrics.get(k)
        if v is None or np.isnan(float(v)):
            continue
        val = (100.0 - float(v)) if k == "K_pct_raw" else float(v)
        contact_sum += val * w
        contact_w   += w

    if power_w == 0 and contact_w == 0:
        return None

    # Normalise each component to its available weight so they're comparable
    power_score   = power_sum   / power_w   if power_w   > 0 else 50.0
    contact_score = contact_sum / contact_w if contact_w > 0 else 50.0

    spectrum = (power_score / (power_score + contact_score + 0.001)) * 100.0
    return float(spectrum)


def classify_spectrum(metrics: dict[str, float]) -> dict:
    """
    Types 3 (Contact), 4 (Balanced), and 5 (Power) via spectrum score.

    Grey zone (45–55) → Balanced, with confidence = proximity to 50.
    Outside grey zone → Contact (< 45) or Power (> 55).

    Always returns a result — this is the fallback classification.
    """
    spectrum = compute_spectrum(metrics)

    if spectrum is None:
        return {
            "type":               "Undetermined",
            "type_code":          "U0",
            "raw_confidence":     0.0,
            "display_confidence": 0,
            "spectrum_score":     None,
            "grey_zone":          True,
            "margins":            {},
        }

    grey_zone = GREY_ZONE_LOW <= spectrum <= GREY_ZONE_HIGH

    if grey_zone:
        # Balanced: confidence = proximity to 50 (closer = more balanced = higher confidence)
        raw_conf  = 1.0 - abs(spectrum - 50.0) / (GREY_ZONE_HIGH - 50.0)
        disp_conf = display_confidence(raw_conf, ceiling=SPECTRUM_CONFIDENCE_CEILING)
        return {
            "type":               "Balanced",
            "type_code":          "T4",
            "raw_confidence":     raw_conf,
            "display_confidence": disp_conf,
            "spectrum_score":     spectrum,
            "grey_zone":          True,
            "margins":            {},
        }

    if spectrum < GREY_ZONE_LOW:
        archetype = "Contact"
        type_code = "T3"
    else:
        archetype = "Power"
        type_code = "T5"

    # Confidence = distance from the grey zone boundary
    boundary = GREY_ZONE_LOW if spectrum < GREY_ZONE_LOW else GREY_ZONE_HIGH
    raw_conf  = abs(spectrum - boundary) / (50.0 - (GREY_ZONE_HIGH - 50.0))
    raw_conf  = min(raw_conf, 1.0)
    disp_conf = display_confidence(raw_conf, ceiling=SPECTRUM_CONFIDENCE_CEILING)

    return {
        "type":               archetype,
        "type_code":          type_code,
        "raw_confidence":     raw_conf,
        "display_confidence": disp_conf,
        "spectrum_score":     spectrum,
        "grey_zone":          False,
        "margins":            {},
    }


# ---------------------------------------------------------------------------
# Step 5 — Primary classification (ordered)
# ---------------------------------------------------------------------------

def classify_primary(metrics: dict[str, float]) -> dict:
    """
    Classify a hitter into one primary archetype.

    Order:
      1. Complete Hitter (most restrictive)
      2. Three True Outcomes (routes to spectrum if confidence < 30)
      3. Spectrum (Pure Contact or Pure Power)
    """
    result = classify_complete(metrics)
    if result is not None:
        return result

    result = classify_tto(metrics)
    if result is not None:
        return result

    return classify_spectrum(metrics)


# ---------------------------------------------------------------------------
# Step 6 — Modifiers
# ---------------------------------------------------------------------------

def compute_free_swinger(metrics: dict[str, float]) -> bool:
    """
    Free Swinger flag — all 3 conditions must be met:
      FPS_pct    >= 65th pct  (first-pitch swing)
      OSwing_pct >= 60th pct  (chase rate)
      BB_pct_inv >= 35th pct  (low walk rate — caller supplies inverted pct)
    """
    for metric, threshold in FREE_SWINGER_THRESHOLDS.items():
        val = metrics.get(metric)
        if val is None or float(val) < threshold:
            return False
    return True


def compute_speed_modifier(
    metrics: dict[str, float],
    sprint_speed_raw: Optional[float] = None,
) -> Optional[str]:
    """
    Speed modifier.

    Requires:
      sprint_speed_raw >= 28.0 ft/sec  (hard threshold on raw value)
      AND any one of:
        SB_efficiency_pct >= 50  (net positive run value — pct rank)
        XBT_pct           >= 55  (extra bases taken %)
        HP_to_1B_inv_pct  >= 25  (low home-to-first time — caller inverts)

    Returns 'Speed' or None.
    """
    # Hard threshold on raw sprint speed
    speed = sprint_speed_raw or metrics.get("sprint_speed_raw")
    if speed is None or float(speed) < SPRINT_SPEED_HARD_THRESHOLD:
        return None

    gates = {
        "SB_efficiency_pct": 50.0,
        "XBT_pct":           55.0,
        "HP_to_1B_inv_pct":  25.0,
    }
    for gate_metric, gate_threshold in gates.items():
        val = metrics.get(gate_metric)
        if val is not None and float(val) >= gate_threshold:
            return "Speed"

    return None


def compute_disruptiveness(
    sb:           int,
    cs:           int,
    opportunities:int,
    games:        int,
    season_games: int = 162,
) -> dict:
    """
    Disruptive / Chaotic baserunning modifier.

    Args:
        sb:            stolen bases
        cs:            caught stealing
        opportunities: times on base (1B + walks + HBP)
        games:         games played so far
        season_games:  full season length (default 162)

    Returns dict with keys:
        modifier    — 'Disruptive' | 'Chaotic' | None
        efficiency  — run value of SB activity
        sb_pct      — success rate
        attempt_rate— attempts / opportunities
        reason      — explanation if no modifier
    """
    attempts     = sb + cs
    min_attempts = max(2, int(DISRUPTIVE_MIN_ATTEMPTS * (games / season_games)))

    if attempts < min_attempts:
        return {
            "modifier":     None,
            "reason":       "insufficient_sample",
            "efficiency":   None,
            "sb_pct":       None,
            "attempt_rate": None,
        }

    efficiency   = (sb * 0.20) + (cs * -0.45)
    sb_pct       = sb / attempts
    attempt_rate = attempts / max(opportunities, 1)

    if efficiency > 0.5 and sb_pct > 0.72 and attempt_rate > 0.40:
        modifier = "Disruptive"
    elif efficiency < -0.5 and sb_pct < 0.65 and attempt_rate > 0.40:
        modifier = "Chaotic"
    else:
        modifier = None

    return {
        "modifier":     modifier,
        "reason":       None,
        "efficiency":   efficiency,
        "sb_pct":       sb_pct,
        "attempt_rate": attempt_rate,
    }


# ---------------------------------------------------------------------------
# Step 7 — Full hitter profile
# ---------------------------------------------------------------------------

def build_hitter_profile(
    player_id:     int,
    metrics:       dict[str, float],
    sprint_speed_raw: Optional[float] = None,
    sb:            int = 0,
    cs:            int = 0,
    opportunities: int = 0,
    games:         int = 162,
    season_games:  int = 162,
) -> dict:
    """
    Full hitter classification for one player × season.

    Args:
        player_id:       MLBAM ID
        metrics:         pre-normalised metric dict (0–100 percentiles)
        sprint_speed_raw: raw sprint speed in ft/sec (for Speed modifier)
        sb, cs:          stolen base counts (for Disruptiveness modifier)
        opportunities:   times on base (for attempt_rate)
        games:           games played (for prorated min attempts)
        season_games:    full season length

    Returns:
        {
            player_id,
            primary:    archetype dict (type, type_code, confidence, …),
            modifiers:  {free_swinger, speed, disruptiveness},
        }
    """
    primary          = classify_primary(metrics)
    free_swinger     = compute_free_swinger(metrics)
    speed_mod        = compute_speed_modifier(metrics, sprint_speed_raw)
    disruptiveness   = compute_disruptiveness(sb, cs, opportunities, games, season_games)

    return {
        "player_id": player_id,
        "primary":   primary,
        "modifiers": {
            "free_swinger":    free_swinger,
            "speed":           speed_mod,
            "disruptiveness":  disruptiveness,
        },
    }


# ---------------------------------------------------------------------------
# Batch: list of players → DataFrame
# ---------------------------------------------------------------------------

def profiles_to_dataframe(profiles: list[dict]) -> pd.DataFrame:
    """
    Convert list of hitter profiles to a flat DataFrame.

    One row per player. Columns:
      player_id, type, type_code, display_confidence, spectrum_score,
      grey_zone, free_swinger, speed_modifier, disruptive_modifier
    """
    rows = []
    for p in profiles:
        pri = p["primary"]
        mod = p["modifiers"]
        rows.append({
            "player_id":          p["player_id"],
            "type":               pri["type"],
            "type_code":          pri["type_code"],
            "display_confidence": pri["display_confidence"],
            "spectrum_score":     pri.get("spectrum_score"),
            "grey_zone":          pri.get("grey_zone", False),
            "free_swinger":       mod["free_swinger"],
            "speed_modifier":     mod["speed"],
            "disruptive_modifier":mod["disruptiveness"]["modifier"],
        })
    return pd.DataFrame(rows)
