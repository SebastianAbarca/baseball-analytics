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
    "xwOBA":    70.0,   # overall expected offensive quality
    "OBP":      70.0,   # on-base ability
    "ISO":      70.0,   # real power
    "BB_pct":   65.0,   # plate discipline
    "Barrel_pct": 65.0, # consistent hard contact quality
    "AVG":      75.0,   # batting average — output is king. Replaces Contact_pct:
                        # Contact% measures process (swing contact rate) but doesn't
                        # guarantee useful contact. AVG measures actual results.
}

# K% hard ceiling — true TTO strikeout rates are incompatible with "complete"
# regardless of AVG or other metrics. Raw fraction (0.30 = 30%).
COMPLETE_K_CEILING = 0.30

TTO_THRESHOLDS: dict[str, float] = {
    "K_pct_raw": 55.0,   # above average K rate (non-inverted pct — higher = more Ks)
    "BB_pct":    65.0,   # clearly above average walks
    "ISO":       65.0,   # clearly above average power
}

# Aggressive modifier — high FPS AND high chase rate (2-gate, no BB requirement)
AGGRESSIVE_THRESHOLDS: dict[str, float] = {
    "FPS_pct":    65.0,   # first-pitch swing % > 65th pct
    "OSwing_pct": 60.0,   # chase rate > 60th pct
}

# Speed tiers (raw ft/sec thresholds)
SPEED_ELITE_THRESHOLD   = 29.0   # ft/sec — top ~10%
SPEED_FAST_THRESHOLD    = 28.0   # ft/sec — top ~35%
SPEED_AVERAGE_THRESHOLD = 26.5   # ft/sec — above slow

# Lucky / Unlucky (LuckDelta = xwOBA − wOBA percentile, positive = unlucky)
LUCKY_THRESHOLD   = 25.0   # bottom 25% luck delta → luckier than expected
UNLUCKY_THRESHOLD = 75.0   # top 25% luck delta → unluckier than expected

# Contact Quality (composite of Barrel_pct + HardHit_pct percentiles)
CONTACT_QUALITY_PLUS_THRESHOLD   = 70.0  # top 30% → "Plus Contact"
CONTACT_QUALITY_WEAK_THRESHOLD   = 30.0  # bottom 30% → "Weak Contact"

# Plate Discipline (composite: BB_pct weighted 60%, OSwing inv 40%)
DISCIPLINE_ELITE_THRESHOLD       = 75.0  # top 25% → "Elite Discipline"
DISCIPLINE_PATIENT_THRESHOLD     = 55.0  # top 45% → "Disciplined"
DISCIPLINE_FREE_SWINGER_THRESHOLD= 35.0  # bottom 35% → "Free Swinger" (swings freely, low discipline)

# Table Setter
TABLE_SETTER_OBP_THRESHOLD = 65.0   # OBP percentile — above-average on-base
TABLE_SETTER_ISO_CEILING   = 45.0   # ISO percentile — not a power threat

# Disruptiveness — note: raw values required (not percentiles)
DISRUPTIVE_MIN_ATTEMPTS = 8          # prorated by games played

# Gap Hitter (extra-base contact, gap-zone spray)
GAP_HITTER_XB_THRESHOLD   = 60.0   # xb_pct ≥ 60th pct (doubles+triples rate)
GAP_HITTER_GAP_THRESHOLD  = 55.0   # gap_pct ≥ 55th pct (BIP in gap zones)

# Plus Power (above-average ISO for a contact/balanced player — fallback when Gap Hitter doesn't fire)
PLUS_POWER_ISO_THRESHOLD  = 60.0   # ISO_pct ≥ 60th pct (above-average power for a contact hitter)

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
    min_present: int | None = None,
) -> tuple[bool, float, dict[str, float]]:
    """
    Check whether all present thresholds are met.

    passes         — True if all present metrics meet their thresholds AND
                     min_present gates have data (or min_present is None →
                     original strict behaviour: any missing = fail)
    raw_confidence — mean margin above threshold across present metrics / 100
    margins        — {metric: (observed - threshold) / 100}

    min_present:
      None  → strict mode: any missing metric fails the check (original behaviour)
      int N → flexible mode: at least N metrics must be present; missing ones are
              skipped rather than failing. Use for Statcast-dependent gates (Complete
              Hitter) where early seasons / low-PA players lack xwOBA / Barrel%.
    """
    margins = {}
    missing = []

    for metric, threshold in thresholds.items():
        val = metrics.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            missing.append(metric)
            continue
        margins[metric] = (float(val) - threshold) / 100.0

    n_present = len(margins)

    if min_present is None:
        # Original strict mode — any missing fails
        if missing:
            log.debug("_check_thresholds: missing metrics %s — treating as failed", missing)
            return False, 0.0, margins
    else:
        # Flexible mode — require at least min_present gates to have data
        if n_present < min_present:
            log.debug(
                "_check_thresholds: only %d/%d metrics present (min %d) — failing",
                n_present, len(thresholds), min_present,
            )
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

    All 6 COMPLETE_THRESHOLDS must be met, including AVG ≥ 75th pct.
    AVG replaces the old Contact_pct gate — output is king. Contact%
    measures process (swing contact rate) but a high contact rate with
    weak contact (Grisham) doesn't make a Complete Hitter. AVG does.

    Additional ceiling: K% > 30% raw disqualifies regardless of AVG,
    keeping extreme TTO strikeout rates out of the Complete Hitter label.
    """
    # min_present=4: require at least 4 of 6 gates to have data.
    # xwOBA and Barrel% are missing for < 300 PA and in 2015-2016 Statcast gaps —
    # don't auto-fail for data absence, but all present gates must still pass.
    # Confidence is naturally lower when fewer metrics are available.
    passes, raw_conf, margins = _check_thresholds(metrics, COMPLETE_THRESHOLDS, min_present=4)
    if not passes:
        return None

    # K% hard ceiling: > 30% raw K rate disqualifies
    k_raw = metrics.get("k_rate_raw")
    if k_raw is not None and float(k_raw) > COMPLETE_K_CEILING:
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
    # Power score: 5 real metrics, all confirmed present in our data pipeline.
    #   ISO_pct    — always available (BRef/FG); primary raw power signal
    #   Barrel_pct — Statcast pct rank (300+ PA); hard contact quality
    #   HardHit_pct— Statcast pct rank (300+ PA); hard contact rate
    #   HR_FB_pct  — computed from raw Statcast bb_type (min 20 BIP); power purity
    #   EV_avg     — Statcast avg exit velo pct rank (300+ PA), pool-ranked by us
    # HR_FB_pct and EV_90_pct from the original spec were never in our data.
    power_weights   = {
        "ISO_pct":    0.35,
        "Barrel_pct": 0.25,
        "HardHit_pct":0.20,
        "HR_FB_pct":  0.10,
        "EV_avg":     0.10,
    }
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

def compute_aggressive(metrics: dict[str, float]) -> bool:
    """
    Aggressive modifier — high FPS AND high chase rate.

    Players who attack early in counts and expand the zone.
    Good hitters (Tucker, Alvarez) can have this; it doesn't imply poor discipline.

    Gate: FPS_pct >= 65 AND OSwing_pct >= 60.
    """
    for metric, threshold in AGGRESSIVE_THRESHOLDS.items():
        val = metrics.get(metric)
        if val is None or float(val) < threshold:
            return False
    return True


def compute_speed_tier(sprint_speed_raw: Optional[float]) -> Optional[str]:
    """
    Speed tier based on raw sprint speed (ft/sec).

    Returns 'Elite' | 'Fast' | 'Average' | 'Slow' | None (if no data).

    Thresholds (raw ft/sec, season-stable):
      Elite:   >= 29.0  (top ~10%)
      Fast:    >= 28.0  (top ~35%)
      Average: >= 26.5  (middle ~40%)
      Slow:     < 26.5  (bottom ~25%)
    """
    if sprint_speed_raw is None:
        return None
    speed = float(sprint_speed_raw)
    if speed >= SPEED_ELITE_THRESHOLD:
        return "Elite"
    if speed >= SPEED_FAST_THRESHOLD:
        return "Fast"
    if speed >= SPEED_AVERAGE_THRESHOLD:
        return "Average"
    return "Slow"


def compute_lucky_unlucky(metrics: dict[str, float]) -> Optional[str]:
    """
    Lucky / Unlucky modifier based on xwOBA − wOBA percentile rank.

    LuckDelta = xwOBA − wOBA (from est_woba_minus_woba_diff in FG batting CSV).
    After cross-player normalization, higher percentile = more unlucky.

    Returns 'Lucky' | 'Unlucky' | None.
    Only fires for players with Statcast coverage (LuckDelta available).
    """
    luck = metrics.get("LuckDelta")
    if luck is None:
        return None
    if float(luck) <= LUCKY_THRESHOLD:
        return "Lucky"
    if float(luck) >= UNLUCKY_THRESHOLD:
        return "Unlucky"
    return None


def compute_contact_quality(metrics: dict[str, float]) -> Optional[str]:
    """
    Contact Quality tier based on Barrel% and Hard Hit% percentile ranks.

    Composite = 0.5 * Barrel_pct + 0.5 * HardHit_pct
    Falls back to whichever is available if only one is present.

    Returns 'Plus Contact' | 'Weak Contact' | None (average or no data).
    """
    barrel   = metrics.get("Barrel_pct")
    hard_hit = metrics.get("HardHit_pct")

    if barrel is None and hard_hit is None:
        return None

    vals = [v for v in [barrel, hard_hit] if v is not None]
    composite = float(np.mean(vals))

    if composite >= CONTACT_QUALITY_PLUS_THRESHOLD:
        return "Plus Contact"
    if composite <= CONTACT_QUALITY_WEAK_THRESHOLD:
        return "Weak Contact"
    return None


def compute_plate_discipline(metrics: dict[str, float]) -> Optional[str]:
    """
    Plate Discipline grade — composite of walk rate and chase rate.

    Score = BB_pct_pct × 0.60 + (100 − OSwing_pct) × 0.40
    OSwing falls back to zone swing / pitches per PA signals if unavailable.

    Returns 'Elite Discipline' | 'Disciplined' | 'Free Swinger' | None (average).
    Skips Average to reduce label clutter.
    """
    bb_pct   = metrics.get("BB_pct")     # already percentile-ranked
    oswing   = metrics.get("OSwing_pct") # higher = chases more (bad)

    if bb_pct is None:
        return None

    if oswing is not None:
        score = float(bb_pct) * 0.60 + (100.0 - float(oswing)) * 0.40
        if score >= DISCIPLINE_ELITE_THRESHOLD:
            return "Elite Discipline"
        if score >= DISCIPLINE_PATIENT_THRESHOLD:
            return "Disciplined"
        if score <= DISCIPLINE_FREE_SWINGER_THRESHOLD:
            return "Free Swinger"
        return None
    else:
        # Chase rate unavailable — walk rate only.
        # Cap at "Disciplined": can't confirm Elite without knowing chase behavior.
        score = float(bb_pct)
        if score >= DISCIPLINE_PATIENT_THRESHOLD:
            return "Disciplined"
        if score <= DISCIPLINE_FREE_SWINGER_THRESHOLD:
            return "Free Swinger"
        return None


def compute_table_setter(
    metrics:          dict[str, float],
    sprint_speed_raw: Optional[float] = None,
) -> bool:
    """
    Table Setter modifier — high OBP + speed + low power.

    Gates:
      OBP_pct   >= 65  (above-average on-base ability)
      speed tier Fast or Elite (>= 28.0 ft/sec)
      ISO_pct   <= 45  (not a power threat — limits to true table setters)

    Captures: Myles Straw, Tommy Edman, early-career Altuve types.
    A Complete Hitter would have ISO > 45 and thus NOT be a Table Setter.
    """
    obp = metrics.get("OBP")   # stripped from OBP_pct after normalization
    iso = metrics.get("ISO")   # stripped from ISO_pct after normalization

    if obp is None or iso is None:
        return False
    if float(obp) < TABLE_SETTER_OBP_THRESHOLD:
        return False
    if float(iso) > TABLE_SETTER_ISO_CEILING:
        return False

    speed = compute_speed_tier(sprint_speed_raw)
    return speed in ("Fast", "Elite")


def compute_plus_power(metrics: dict[str, float]) -> bool:
    """
    Plus Power modifier — above-average ISO in a contact or balanced hitter.

    Fires only for Contact (T3) and Balanced (T4) players; Power (T5) and
    Complete Hitter (T1) already imply power. This tag surfaces the contact
    hitters who also have real thump — Bogaerts, Goldschmidt, early Freeman.

    Gate: ISO_pct ≥ 65th pct

    Caller is responsible for checking primary archetype before applying
    (suppressed for T1, T2, T5 in build_hitter_profile).
    """
    iso = metrics.get("ISO")   # stripped from ISO_pct after normalization
    if iso is None:
        return False
    return float(iso) >= PLUS_POWER_ISO_THRESHOLD


def compute_gap_hitter(metrics: dict[str, float]) -> bool:
    """
    Gap Hitter modifier — player whose extra-base production comes from
    hitting the ball into the gaps.

    Gates (percentile-ranked 0–100 against the season pool):
      XB_pct      >= 60th pct  — above-average doubles+triples rate per BIP
      GapTend_pct >= 55th pct  — above-average fraction of BIP in gap zones

    HR/FB ceiling removed: spray angle + doubles rate already describe the
    behavior. A contact player with high gap tendency who also has a modest
    HR/FB rate is still a gap hitter. The ceiling was creating artificial
    mutual exclusivity with Plus Power.

    When Gap Hitter fires, Plus Power is suppressed in build_hitter_profile
    (Gap Hitter is more specific — spray-confirmed).

    Requires Statcast spray chart data (hc_x/hc_y); returns False if unavailable.
    """
    xb  = metrics.get("XB_pct")
    gap = metrics.get("GapTend_pct")
    if xb is None or gap is None:
        return False
    return float(xb) >= GAP_HITTER_XB_THRESHOLD and float(gap) >= GAP_HITTER_GAP_THRESHOLD


def compute_disruptiveness(
    sb:                int,
    cs:                int,
    opportunities:     int,
    games:             int,
    season_games:      int = 162,
    attempt_rate_pct:  Optional[float] = None,
) -> dict:
    """
    Disruptive / Chaotic baserunning modifier.

    Args:
        sb:               stolen bases
        cs:               caught stealing
        opportunities:    approximate times on base (OBP * PA or 1B+BB+HBP)
        games:            games played so far
        season_games:     full season length (default 162)
        attempt_rate_pct: cross-player percentile rank of attempt_rate (0–100).
                          When provided, gates at >= 25 (top 75% of active
                          baserunners). When None, uses raw fallback (> 0.05).

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

    # Rate gate: top 50% of active baserunners (pct >= 50) or raw fallback
    if attempt_rate_pct is not None:
        rate_passes = attempt_rate_pct >= 50.0
    else:
        rate_passes = attempt_rate > 0.08

    if efficiency > 0.5 and sb_pct > 0.72 and rate_passes:
        modifier = "Disruptive"
    elif efficiency < -0.5 and sb_pct < 0.65 and rate_passes:
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
    player_id:        int,
    metrics:          dict[str, float],
    sprint_speed_raw: Optional[float] = None,
    sb:               int = 0,
    cs:               int = 0,
    opportunities:    int = 0,
    games:            int = 162,
    season_games:     int = 162,
    attempt_rate_pct: Optional[float] = None,
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
    primary      = classify_primary(metrics)
    primary_type = primary.get("type", "")
    is_complete  = primary_type == "Complete Hitter"
    is_tto       = primary_type == "Three True Outcomes"
    is_contact   = primary_type in ("Contact", "Balanced")

    aggressive       = compute_aggressive(metrics)
    gap_hitter       = compute_gap_hitter(metrics)
    # Plus Power fires for Contact/Balanced only, and is suppressed when Gap Hitter
    # fires — Gap Hitter is the spray-confirmed, more specific label.
    plus_power       = compute_plus_power(metrics) if (is_contact and not gap_hitter) else False
    speed_tier       = compute_speed_tier(sprint_speed_raw)
    lucky_unlucky    = compute_lucky_unlucky(metrics)
    contact_quality  = compute_contact_quality(metrics)
    plate_discipline = compute_plate_discipline(metrics)
    table_setter     = compute_table_setter(metrics, sprint_speed_raw)
    disruptiveness   = compute_disruptiveness(sb, cs, opportunities, games, season_games,
                                              attempt_rate_pct=attempt_rate_pct)

    # ── Archetype-based modifier suppression ─────────────────────────────────
    # Tags implied by the archetype's own gates are dropped — only surprising
    # or additional information survives.

    if is_complete:
        # T1 gates: Barrel% ≥ 65, Contact% ≥ 55, BB% ≥ 65
        # Implied → suppress Gap Hitter, Plus Contact, Disciplined
        # Keep: speed, Lucky/Unlucky, Aggressive, Elite Discipline, Disruptive
        gap_hitter = False
        if contact_quality == "Plus Contact":
            contact_quality = None
        if plate_discipline == "Disciplined":
            plate_discipline = None
        # Elite Discipline (≥ 75th) exceeds the BB% ≥ 65 floor → keep it.

    elif is_tto:
        # T2 gates: K% ≥ 55 (no contact implied), BB% ≥ 65 (discipline implied)
        # Implied → suppress Weak Contact, Disciplined
        # Keep: speed, Lucky/Unlucky, Elite Discipline, Aggressive, Disruptive
        if contact_quality == "Weak Contact":
            contact_quality = None
        if plate_discipline == "Disciplined":
            plate_discipline = None

    return {
        "player_id": player_id,
        "primary":   primary,
        "modifiers": {
            "aggressive":       aggressive,
            "gap_hitter":       gap_hitter,
            "plus_power":       plus_power,
            "speed":            speed_tier,
            "lucky_unlucky":    lucky_unlucky,
            "contact_quality":  contact_quality,
            "plate_discipline": plate_discipline,
            "table_setter":     table_setter,
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
            "aggressive":         mod.get("aggressive", False),
            "gap_hitter":         mod.get("gap_hitter", False),
            "speed_tier":         mod["speed"],
            "disruptive_modifier":mod["disruptiveness"]["modifier"],
        })
    return pd.DataFrame(rows)
