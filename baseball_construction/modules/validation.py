"""
validation.py — validation tests against known outputs.

Each test is self-contained and returns (passed: bool, detail: str).
run_all() executes every test and prints a summary.

Tests:
  V1  Temporal mode determination — known PA/games → expected modes
  V2  Reliability weighting — known sample/threshold → expected weight
  V3  Historical smoothing — 3-year weighted avg matches hand calculation
  V4  Philosophy scoring — known metric dict → expected score and primary
  V5  Hitter archetype — Complete Hitter and TTO routing
  V6  Pitcher archetype — P1 Power and P3 GB Craftsman routing
  V7  End-to-end portrait — builds without error on cached test data
  V8  Pitch mix aggregation — build_pitch_mix produces expected schema
  V9  Arsenal profile — build_arsenal_profile classifies correctly
  V10 Service time estimation — CBA thresholds and IL deduction logic
  V11 Spin efficiency pipeline — physics, Savant precedence, within-type ranks
  V12 Park factors — build_park_profile returns normalized pitcher_friendly
  V13 Spin efficiency vs Savant — physics estimate ranks and scales like measured
  V14 Metric rank directions — every metric declares one; outcome gates monotonic
  V15 Batted-ball direction — pull/oppo relative to the batter, boundaries
  V16 Tag evidence coverage — every tag gated or exempt; the gate refuses thin samples
  V17 Season completeness — missing games vs the schedule, the finished rule, opener dates
  V18 Regular season only — raw caches and BRef totals hold no other games
  V19 Team stints — team PA/BF reconcile; a traded player splits by team
  V20 League caches current — rebuilt and fingerprinted alike, none older than raw
  V21 Sprint speed units — ft/s from Savant's leaderboard, not percentile ranks
  V22 Complete reads and rosters — every DB row read; every 30+ PA hitter on his portrait
  V23 Pitch outcomes and pairing — foul tip = whiff; adjacent pitches; sequencing ratio <= 1
  V24 Steal-attempt ranks league-wide — one rank per player-season; non-runners rank low
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

log = logging.getLogger(__name__)

Result = tuple[bool, str]

# What a check needs beyond the code: --offline (the pull-request checks)
# skips these instead of failing on a runner that has none of them. The
# rest need nothing, or one cached week of Statcast (2023-06-01..07).
NEEDS = {
    "V7":  "the database",
    "V9":  "a full season of Statcast (2023)",
    "V13": "a full season of Statcast (2025) and Savant's active spin",
    "V18": "this machine's season data",
    "V20": "this machine's league caches",
    "V21": "this machine's fg_batting roots",
    "V22": "the database",
}


# ---------------------------------------------------------------------------
# V1 — Temporal mode determination
# ---------------------------------------------------------------------------

def v1_temporal_modes() -> Result:
    """
    Known (PA, games) pairs → expected mode strings.
    Tests all three branches including boundary conditions.
    """
    from temporal import determine_mode, blend_weights

    cases = [
        # (PA, games, expected_mode)
        (0,   0,  "historical"),   # brand-new player
        (14,  14, "historical"),   # just below both thresholds
        (15,  15, "early"),        # exactly at threshold
        (50,  20, "early"),        # mid-early
        (199, 49, "early"),        # just below current threshold
        (200, 50, "current"),      # exactly at current threshold
        (500, 130,"current"),      # full season
    ]

    errors = []
    for pa, games, expected in cases:
        got = determine_mode(pa, games)
        if got != expected:
            errors.append(f"PA={pa} games={games}: expected={expected} got={got}")

    # Also verify blend weights at boundary and midpoint
    hist_w, curr_w = blend_weights(15)   # progress = 0 → curr_w = 0
    if not np.isclose(curr_w, 0.0):
        errors.append(f"blend_weights(15): curr_w should be 0.0, got {curr_w:.4f}")

    hist_w, curr_w = blend_weights(50)   # progress = 1 → curr_w = 0.50
    if not np.isclose(curr_w, 0.50):
        errors.append(f"blend_weights(50): curr_w should be 0.50, got {curr_w:.4f}")

    # Midpoint: games=32 → progress = (32-15)/(50-15) = 17/35 ≈ 0.4857
    hist_w, curr_w = blend_weights(32)
    expected_curr = (17 / 35) * 0.50
    if not np.isclose(curr_w, expected_curr, atol=1e-6):
        errors.append(
            f"blend_weights(32): curr_w expected {expected_curr:.6f} got {curr_w:.6f}"
        )

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V2 — Reliability weighting
# ---------------------------------------------------------------------------

def v2_reliability_weighting() -> Result:
    """
    Known sample sizes → expected reliability and weighted values.
    """
    from ingest import compute_reliability, apply_reliability_weight

    errors = []

    # K_pct threshold = 60
    r = compute_reliability(30, "K_pct")
    if not np.isclose(r, 0.5):
        errors.append(f"K_pct sample=30: expected 0.5, got {r:.4f}")

    r = compute_reliability(60, "K_pct")
    if not np.isclose(r, 1.0):
        errors.append(f"K_pct sample=60: expected 1.0, got {r:.4f}")

    r = compute_reliability(120, "K_pct")
    if not np.isclose(r, 1.0):   # capped at 1.0
        errors.append(f"K_pct sample=120: expected 1.0 (capped), got {r:.4f}")

    # Unknown metric → 0.0
    r = compute_reliability(500, "nonexistent_metric")
    if not np.isclose(r, 0.0):
        errors.append(f"unknown metric: expected 0.0, got {r:.4f}")

    # Weighted value: observed=0.80, reliability=0.5, mean=0.50
    # → 0.5*0.80 + 0.5*0.50 = 0.40 + 0.25 = 0.65
    w = apply_reliability_weight(0.80, 0.5, league_mean=0.50)
    if not np.isclose(w, 0.65):
        errors.append(f"apply_reliability_weight: expected 0.65, got {w:.4f}")

    # Full reliability: observed=0.80, reliability=1.0 → return observed unchanged
    w = apply_reliability_weight(0.80, 1.0, league_mean=0.50)
    if not np.isclose(w, 0.80):
        errors.append(f"apply_reliability_weight full: expected 0.80, got {w:.4f}")

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V3 — Historical smoothing
# ---------------------------------------------------------------------------

def v3_historical_smoothing() -> Result:
    """
    3-year weighted average matches hand calculation.
    Tests missing-season reweighting too.
    """
    from temporal import compute_historical_smooth, HISTORICAL_WEIGHTS

    errors = []

    # Full data: ref=2023 → uses 2023(lag0), 2022(lag1), 2021(lag2)
    season_values = {2023: 0.70, 2022: 0.60, 2021: 0.50}
    # Expected: (0.70*0.50 + 0.60*0.30 + 0.50*0.20) / 1.0 = 0.35+0.18+0.10 = 0.63
    result = compute_historical_smooth(season_values, reference_season=2023)
    if result is None or not np.isclose(result, 0.63, atol=1e-6):
        errors.append(f"full 3-year: expected 0.63, got {result}")

    # Missing lag-1 (2022 absent): reweight lag0 and lag2
    # available weight = 0.50 + 0.20 = 0.70
    # weighted_sum = 0.70*0.50 + 0.50*0.20 = 0.35 + 0.10 = 0.45
    # result = 0.45 / 0.70 = 0.6428...
    season_values_gap = {2023: 0.70, 2021: 0.50}
    result = compute_historical_smooth(season_values_gap, reference_season=2023)
    expected_gap = (0.70 * 0.50 + 0.50 * 0.20) / (0.50 + 0.20)
    if result is None or not np.isclose(result, expected_gap, atol=1e-6):
        errors.append(f"missing lag-1: expected {expected_gap:.6f}, got {result}")

    # All missing → None
    result = compute_historical_smooth({}, reference_season=2023)
    if result is not None:
        errors.append(f"all missing: expected None, got {result}")

    # Single season: only lag0 present → result = that value
    result = compute_historical_smooth({2023: 0.55}, reference_season=2023)
    if result is None or not np.isclose(result, 0.55, atol=1e-6):
        errors.append(f"single season: expected 0.55, got {result}")

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V4 — Philosophy scoring
# ---------------------------------------------------------------------------

def v4_philosophy_scoring() -> Result:
    """
    Known metric dict → verify A1 (TTO) score and primary philosophy selection.

    A TTO-heavy team (high ISO, K%, BB%, HR/FB) should score high on A1
    and low on A2 (Contact and Pressure).
    Primary offense philosophy should be A1.
    """
    from philosophy import build_philosophy_summary, score_philosophy, PHILOSOPHY_DEFS

    errors = []

    # Construct a TTO-heavy metric profile
    metrics = {
        # A1 weights: ISO_pct(0.30), BB_pct_pct(0.25), K_pct_pct(0.20),
        #             HR_FB_pct(0.15), Sprint_inv_pct(0.10)
        "ISO_pct":         85.0,
        "BB_pct_pct":      80.0,
        "K_pct_pct":       75.0,
        "HR_FB_pct":       80.0,
        "Sprint_inv_pct":  70.0,
        # A2 — contact profile (deliberately low)
        "K_inv_pct":       25.0,    # high K% → low contact inv
        "OBP_SLG_gap_pct": 30.0,
        "PitchesPerPA_pct":40.0,
        "SprintSpeed_pct": 30.0,
        "Contact_pct_pct": 25.0,
    }

    # A1 score should be a weighted mean of [85,80,75,80,70]
    # = (85*0.30 + 80*0.25 + 75*0.20 + 80*0.15 + 70*0.10) / 1.0
    # = 25.5 + 20.0 + 15.0 + 12.0 + 7.0 = 79.5
    a1_weights = PHILOSOPHY_DEFS["A1"]["weights"]
    a1_score, a1_cov = score_philosophy(metrics, a1_weights)
    if a1_score is None or not np.isclose(a1_score, 79.5, atol=0.5):
        errors.append(f"A1 score: expected ~79.5, got {a1_score}")
    if not np.isclose(a1_cov, 1.0):
        errors.append(f"A1 coverage: expected 1.0, got {a1_cov:.3f}")

    # A2 score should be low (contact team opposite)
    a2_weights = PHILOSOPHY_DEFS["A2"]["weights"]
    a2_score, _ = score_philosophy(metrics, a2_weights)
    if a2_score is None or a2_score > 40.0:
        errors.append(f"A2 score: expected < 40.0 for TTO team, got {a2_score}")

    # Full summary — primary offense should be A1
    summary = build_philosophy_summary("TST", 2023, metrics)
    offense_conf = summary["confidences"]["offense"]
    if offense_conf["primary"] != "A1":
        errors.append(
            f"offense primary: expected A1, got {offense_conf['primary']}"
        )

    # Confidence should be positive (A1 score >> A2 score)
    if offense_conf["display_confidence"] is None or offense_conf["display_confidence"] <= 0:
        errors.append(f"display_confidence should be > 0, got {offense_conf['display_confidence']}")

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V5 — Hitter archetypes
# ---------------------------------------------------------------------------

def v5_hitter_archetypes() -> Result:
    """
    Test Complete Hitter, TTO → spectrum routing, and spectrum classification.
    Metric keys match current COMPLETE_THRESHOLDS (xwOBA, AVG, Barrel_pct, etc.)
    and TTO_THRESHOLDS (K_pct_raw, BB_pct, ISO).
    """
    from hitter_archetypes import (
        classify_complete, classify_tto, classify_spectrum,
        classify_primary, build_hitter_profile,
        COMPLETE_THRESHOLDS, TTO_THRESHOLDS,
    )

    errors = []

    # ── Complete Hitter: all thresholds comfortably met ───────────────────
    # Current COMPLETE_THRESHOLDS: xwOBA≥70, OBP≥70, ISO≥70, BB_pct≥65,
    #                              Barrel_pct≥65, AVG≥75
    # K% raw ceiling: > 0.30 disqualifies
    complete_metrics = {
        "xwOBA":      80.0,
        "OBP":        80.0,
        "ISO":        75.0,
        "BB_pct":     70.0,
        "Barrel_pct": 70.0,
        "AVG":        78.0,
        "k_rate_raw": 0.20,   # well below 0.30 ceiling
    }
    result = classify_complete(complete_metrics)
    if result is None:
        errors.append("Complete Hitter: expected match, got None")
    elif result["type"] != "Complete Hitter":
        errors.append(f"Complete Hitter: got type={result['type']}")

    # ── Complete Hitter fails when AVG below threshold ─────────────────────
    low_avg = {**complete_metrics, "AVG": 60.0}   # below 75 threshold
    result = classify_complete(low_avg)
    if result is not None:
        errors.append("Complete Hitter with AVG=60 should fail, but passed")

    # ── Complete Hitter fails when K% exceeds hard ceiling ────────────────
    high_k = {**complete_metrics, "k_rate_raw": 0.35}   # above 0.30 ceiling
    result = classify_complete(high_k)
    if result is not None:
        errors.append("Complete Hitter with k_rate_raw=0.35 should fail (K% ceiling), but passed")

    # ── TTO: thresholds comfortably met for display_confidence >= 30 ────────
    # Current TTO_THRESHOLDS: K_pct_raw≥55, BB_pct≥65, ISO≥65.
    # display_confidence = min(int(raw_conf/0.35 * 100), 100) must be >= 30,
    # requiring raw_conf >= 0.105.  Use values well above thresholds:
    # margins: K=(85-55)/100=0.30, BB=(85-65)/100=0.20, ISO=(85-65)/100=0.20
    # raw_conf = mean(0.30,0.20,0.20) = 0.233 → disp_conf=66 ≥ 30 ✓
    tto_metrics = {
        "K_pct_raw": 85.0,   # threshold 55 — well above
        "BB_pct":    85.0,   # threshold 65 — well above
        "ISO":       85.0,   # threshold 65 — well above
    }
    result = classify_tto(tto_metrics)
    if result is None:
        errors.append("TTO: expected match, got None")
    elif result["type"] != "Three True Outcomes":
        errors.append(f"TTO type: expected 'Three True Outcomes', got {result['type']}")

    # ── TTO fails when K% below threshold ─────────────────────────────────
    low_k_tto = {**tto_metrics, "K_pct_raw": 40.0}   # below 55 threshold
    result = classify_tto(low_k_tto)
    if result is not None:
        errors.append("TTO with K_pct_raw=40 should fail (below threshold), but passed")

    # ── Spectrum: pure power profile ──────────────────────────────────────
    power_metrics = {
        "ISO_pct":       90.0,
        "HR_FB_pct":     85.0,
        "Barrel_pct":    88.0,
        "EV_90_pct":     80.0,
        "Contact_pct":   20.0,
        "K_pct":         80.0,
        "AVG_pct":       25.0,
        "OBP_ISO_gap_pct": 30.0,
    }
    result = classify_spectrum(power_metrics)
    # Spectrum type may be "Power" or a sub-label depending on score ranges
    if result is None:
        errors.append("Spectrum power profile: expected a result, got None")
    elif result.get("spectrum_score") is None or result["spectrum_score"] < 50:
        errors.append(f"Spectrum score expected > 50 for power profile, got {result.get('spectrum_score')}")

    # ── classify_primary returns a valid result for any profile ──────────
    result = classify_primary(complete_metrics)
    if result is None:
        errors.append("classify_primary returned None for valid complete profile")
    elif result.get("type") not in {"Complete Hitter", "Three True Outcomes",
                                     "Power", "Contact", "Balanced", "Undetermined"}:
        errors.append(f"classify_primary returned unexpected type: {result.get('type')}")

    # ── build_hitter_profile returns expected structure ───────────────────
    profile = build_hitter_profile(12345, complete_metrics)
    for key in ("player_id", "primary", "modifiers"):
        if key not in profile:
            errors.append(f"build_hitter_profile missing key: {key}")
    # Verify modifier keys match current implementation
    expected_mods = {"aggressive", "gap_hitter", "plus_power", "speed",
                     "lucky_unlucky", "contact_quality", "plate_discipline",
                     "table_setter", "disruptiveness"}
    actual_mods = set(profile.get("modifiers", {}).keys())
    missing_mods = expected_mods - actual_mods
    if missing_mods:
        errors.append(f"modifiers missing keys: {missing_mods}")

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V6 — Pitcher archetypes
# ---------------------------------------------------------------------------

def v6_pitcher_archetypes() -> Result:
    """
    Test P1 Power qualification, P3 GB Craftsman ceiling enforcement,
    and classify_starter ordering.
    """
    from pitcher_archetypes import (
        classify_p1, classify_p2, classify_p3, classify_p4,
        classify_starter, build_starter_profile,
        P1_MIN, P2_VELO_CEILING, P3_K_CEILING,
    )

    errors = []

    # ── P1 Power: all minimums met ────────────────────────────────────
    p1_metrics = {
        "avg_velo_pct":    70.0,   # >= 60
        "K_pct_pct":       72.0,   # >= 65
        "Whiff_pct_pct":   68.0,   # >= 65
        "stuff_composite_pct": 75.0,
        "CSW_pct_pct":     65.0,
        "HardHit_inv_pct": 60.0,
    }
    result = classify_p1(p1_metrics)
    if result is None:
        errors.append("P1: expected match, got None")
    elif result["type_code"] != "P1":
        errors.append(f"P1 type_code: expected P1, got {result['type_code']}")

    # ── P1 ("Power") is gated on velocity only — K% is scored, not gated ───
    # (A league-wide audit found hard-throwers with modest K% — e.g. Jack
    # Leiter, Michael Kopech — were falling into "Unclassified" because P1
    # required K% >= 58 AND P2's velo ceiling excluded them too. K% remains
    # heavily weighted in _score_p1, so true bat-missing aces still rank
    # highest; this just keeps "big arm, results still developing" inside
    # the same style label instead of in no-man's-land.)
    p1_low_k = {**p1_metrics, "K_pct_pct": 30.0}   # well below old 58 floor
    result = classify_p1(p1_low_k)
    if result is None:
        errors.append("P1 with low K_pct_pct but qualifying velo should still match (K% is scored, not gated), got None")
    elif result["type_code"] != "P1":
        errors.append(f"P1 (low-K, qualifying velo) type_code: expected P1, got {result['type_code']}")

    # ── P1 fails if velocity minimum not met ───────────────────────────────
    p1_fail = {**p1_metrics, "avg_velo_pct": 50.0}   # below 60
    result = classify_p1(p1_fail)
    if result is not None:
        errors.append("P1 with avg_velo_pct=50 should fail (below P1_MIN velocity floor), but passed")

    # ── P2 blocked when velo > ceiling ────────────────────────────────────
    p2_highvelo = {
        "avg_velo_pct":  70.0,   # above P2_VELO_CEILING (65)
        "K_pct_pct":     60.0,   # >= 55
        "Whiff_pct_pct": 58.0,   # >= 55
    }
    result = classify_p2(p2_highvelo)
    if result is not None:
        errors.append("P2 with avg_velo > ceiling should fail, but passed")

    p2_valid = {**p2_highvelo, "avg_velo_pct": 60.0}   # at or below ceiling
    result = classify_p2(p2_valid)
    if result is None:
        errors.append("P2 with avg_velo <= ceiling should pass, got None")

    # ── P3 blocked if K% above ceiling ────────────────────────────────────
    # P3_K_CEILING = 65.0 — K_pct_pct must be AT OR BELOW 65.
    # Use 66.0 to be strictly above the ceiling.
    p3_too_k = {
        "GB_pct_pct":          65.0,   # >= 50 (passes minimum)
        "K_pct_pct":           66.0,   # strictly above P3_K_CEILING (65) → blocked
        "HardHit_allowed_pct": 40.0,   # below hard ceiling
    }
    result = classify_p3(p3_too_k)
    if result is not None:
        errors.append("P3 with K_pct_pct=66 (above ceiling 65) should fail, but passed")

    p3_valid = {
        "GB_pct_pct":          65.0,
        "K_pct_pct":           55.0,   # at or below 60
        "HardHit_allowed_pct": 45.0,   # at or below 50
    }
    result = classify_p3(p3_valid)
    if result is None:
        errors.append("P3 valid profile should pass, got None")
    elif result["type_code"] != "P3":
        errors.append(f"P3 type_code: expected P3, got {result['type_code']}")

    # ── classify_starter: P1 wins when both P1 and P3 qualify ─────────────
    both_p1_p3 = {
        "avg_velo_pct":        70.0,
        "K_pct_pct":           72.0,
        "Whiff_pct_pct":       68.0,
        "stuff_composite_pct": 75.0,
        "CSW_pct_pct":         65.0,
        "HardHit_inv_pct":     60.0,
        "GB_pct_pct":          60.0,   # also passes P3 minimum
        "HardHit_allowed_pct": 45.0,   # below P3 ceiling
    }
    result = classify_starter(both_p1_p3)
    if result["type_code"] != "P1":
        errors.append(
            f"classify_starter ordering: P1 should win, got {result['type_code']}"
        )

    # ── build_starter_profile returns expected structure ──────────────────
    profile = build_starter_profile(99999, p1_metrics)
    for key in ("player_id", "primary", "modifiers"):
        if key not in profile:
            errors.append(f"build_starter_profile missing key: {key}")
    for mod in ("platoon", "command", "arsenal"):
        if mod not in profile.get("modifiers", {}):
            errors.append(f"starter modifiers missing key: {mod}")

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V7 — End-to-end portrait
# ---------------------------------------------------------------------------

def v7_end_to_end_portrait() -> Result:
    """
    Build portraits for HOU, NYY, and ATL using cached 2023 test data.
    Verify structure, no exceptions, and portraits_to_dataframe output.
    """
    from ingest import pull_statcast_range
    from team_portrait import build_team_portrait, portraits_to_dataframe

    errors = []

    try:
        sc = pull_statcast_range("2023-06-01", "2023-06-07", label="test")
    except Exception as exc:
        return False, f"Failed to load cached statcast: {exc}"

    REQUIRED_KEYS = {
        "team", "season", "players", "team_metrics",
        "park", "philosophy", "temporal", "spin",
        "tunneling", "data_coverage",
    }

    portraits = []
    for team in ("HOU", "NYY", "ATL"):
        try:
            portrait = build_team_portrait(team, 2023, statcast=sc)
        except Exception as exc:
            errors.append(f"build_team_portrait({team}) raised: {exc}")
            continue

        # Check top-level keys
        missing = REQUIRED_KEYS - set(portrait.keys())
        if missing:
            errors.append(f"{team}: missing keys {missing}")

        # Check players sub-keys (bullpen_collective is team-level since schema 14)
        players = portrait.get("players", {})
        for pk in ("hitters", "starters", "bullpen_arms"):
            if pk not in players:
                errors.append(f"{team}: players missing {pk}")
        if "bullpen_collective" not in portrait:
            errors.append(f"{team}: missing bullpen_collective")

        # Check temporal is a dict with mode
        temporal = portrait.get("temporal", {})
        if "mode" not in temporal:
            errors.append(f"{team}: temporal missing 'mode'")
        elif temporal["mode"] not in ("historical", "early", "current"):
            errors.append(f"{team}: unexpected temporal mode {temporal['mode']}")

        # With only 7 games, mode should be historical
        if temporal.get("mode") != "historical":
            errors.append(
                f"{team}: 7-game slice should be historical, got {temporal.get('mode')}"
            )

        # Check philosophy structure
        philosophy = portrait.get("philosophy", {})
        for ph_key in ("scores", "confidences"):
            if ph_key not in philosophy:
                errors.append(f"{team}: philosophy missing {ph_key}")
        if "scores" in philosophy and len(philosophy["scores"]) != 12:
            errors.append(
                f"{team}: expected 12 philosophy scores, got {len(philosophy['scores'])}"
            )

        # data_coverage should be a float in [0, 1]
        cov = portrait.get("data_coverage")
        if not isinstance(cov, float) or not (0.0 <= cov <= 1.0):
            errors.append(f"{team}: data_coverage={cov} should be float in [0,1]")

        portraits.append(portrait)

    if not portraits:
        return False, "No portraits built — " + "; ".join(errors)

    # portraits_to_dataframe
    try:
        df = portraits_to_dataframe(portraits)
        if len(df) != len(portraits):
            errors.append(f"DataFrame rows: expected {len(portraits)}, got {len(df)}")
        required_cols = {"team", "season", "data_coverage", "temporal_mode", "hitter_count"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            errors.append(f"DataFrame missing columns: {missing_cols}")
    except Exception as exc:
        errors.append(f"portraits_to_dataframe raised: {exc}")

    passed = len(errors) == 0
    detail = (
        f"{len(portraits)} portraits built, DataFrame {len(df) if portraits else 0} rows"
        if passed
        else "; ".join(errors)
    )
    return passed, detail


# ---------------------------------------------------------------------------
# V8 — Pitch mix aggregation schema
# ---------------------------------------------------------------------------

def v8_pitch_mix_schema() -> Result:
    """
    Verify the pitch mix aggregation logic produces the expected
    per-pitcher×pitch_type schema using the cached 2023 test Statcast slice.

    Runs build_pitch_mix itself on the cached week (into a temp cache), then
    validates column schema, sensible value ranges, and PITCH_FAMILY mapping.
    """
    import pandas as pd
    import numpy as _np
    from ingest import pull_statcast_range
    from pitch_mix import PITCH_FAMILY, FASTBALL_CODES

    errors = []

    try:
        sc = pull_statcast_range("2023-06-01", "2023-06-07", label="test")
    except Exception as exc:
        return False, f"Failed to load cached statcast: {exc}"

    # The real build_pitch_mix, not a copy of its steps: a copy re-states the
    # classification it was meant to check. Cached to a temp file, so no
    # season's pitch-mix cache is touched.
    import tempfile
    from pathlib import Path as _Path
    import pitch_mix as _pm
    tmp = _Path(tempfile.mkdtemp()) / "pitch_mix_test.parquet"
    real_path, _pm._cache_path = _pm._cache_path, (lambda season: tmp)
    try:
        agg = _pm.build_pitch_mix(-2023, force=True, statcast=sc)
    finally:
        _pm._cache_path = real_path

    if agg.empty:
        return False, "No pitch-type rows aggregated from test data"

    # Column presence
    required = {"pitcher", "pitch_type", "n", "usage_pct", "avg_velo", "avg_ivb", "avg_hb",
                "swstr_pct", "whiff_per_swing", "csw_pct"}
    missing = required - set(agg.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, "; ".join(errors)

    # Value ranges
    if not agg["usage_pct"].between(0, 1).all():
        errors.append("usage_pct outside [0,1]")
    velo = agg["avg_velo"].dropna()
    # Floor of 40 accommodates position players pitching in blowouts
    if len(velo) > 0 and not (velo.between(40, 115).all()):
        errors.append(f"avg_velo outside [40,115]: {velo.min():.1f}–{velo.max():.1f}")
    for col in ("swstr_pct", "whiff_per_swing", "csw_pct"):
        v = agg[col].dropna()
        if len(v) > 0 and not v.between(0, 1).all():
            errors.append(f"{col} outside [0,1]")
    # A whiff is a subset of swings and of pitches, so per swing >= per pitch.
    both = agg[["swstr_pct", "whiff_per_swing"]].dropna()
    if (both["whiff_per_swing"] < both["swstr_pct"] - 1e-9).any():
        errors.append("whiff_per_swing below swstr_pct")

    # PITCH_FAMILY mapping for fastball codes
    for code in FASTBALL_CODES:
        if code in PITCH_FAMILY and PITCH_FAMILY[code] != "fastball":
            errors.append(f"PITCH_FAMILY[{code}] = '{PITCH_FAMILY[code]}', expected 'fastball'")

    passed = len(errors) == 0
    detail = (
        f"{len(agg)} pitch-type rows, {agg['pitcher'].nunique()} pitchers, "
        f"{agg['pitch_type'].nunique()} types"
        if passed else "; ".join(errors)
    )
    return passed, detail


# ---------------------------------------------------------------------------
# V9 — Arsenal profile classification
# ---------------------------------------------------------------------------

def v9_arsenal_profile() -> Result:
    """
    Verify build_arsenal_profile classifies fastball family and out-pitch
    correctly using the cached 2023 pitch_mix parquet (or the full-season
    cache if available). Tests structure, approach label membership,
    and display string format for up to 5 pitchers.
    """
    import pandas as pd
    from pathlib import Path
    from pitch_mix import load_pitch_mix, build_arsenal_profile

    errors = []

    VALID_APPROACHES = {"fastball-first", "secondary-led", "balanced", "tunnel-dependent", "knuckleball"}
    VALID_FB_FAMILIES = {"4-seam", "sinker", "cutter", "multi-FB", None}

    # Use cached pitch_mix for 2023 if available; otherwise skip gracefully
    try:
        pm = load_pitch_mix(2023)
    except Exception as exc:
        return False, f"load_pitch_mix(2023) raised: {exc} — rebuild pitch_mix cache first"

    if pm.empty:
        return False, "pitch_mix cache is empty for 2023 — rebuild with build_pitch_mix(2023)"

    # Test against the 5 highest-volume pitchers
    top_pitchers = (
        pm.groupby("pitcher")["n"].sum()
        .nlargest(5)
        .index.tolist()
    )

    for pid in top_pitchers:
        try:
            profile = build_arsenal_profile(pid, 2023, pitch_mix_df=pm)
        except Exception as exc:
            errors.append(f"pitcher {pid}: build_arsenal_profile raised {exc}")
            continue

        # Required top-level keys
        for key in ("fastball", "out_pitch", "approach", "display", "depth"):
            if key not in profile:
                errors.append(f"pitcher {pid}: arsenal_profile missing key '{key}'")

        if errors:
            continue

        # Approach must be a known value
        approach = profile.get("approach")
        if approach not in VALID_APPROACHES:
            errors.append(f"pitcher {pid}: unknown approach '{approach}'")

        # display must be a non-empty string
        display = profile.get("display", "")
        if not isinstance(display, str) or not display.strip():
            errors.append(f"pitcher {pid}: display label empty or wrong type")

        # fastball family must be known or None
        fb = profile.get("fastball") or {}
        if fb.get("family") not in VALID_FB_FAMILIES:
            errors.append(f"pitcher {pid}: unknown fastball family '{fb.get('family')}'")

    passed = len(errors) == 0
    detail = (
        f"Tested {len(top_pitchers)} top-volume pitchers, all profiles valid"
        if passed else "; ".join(errors)
    )
    return passed, detail


# ---------------------------------------------------------------------------
# V10 — Service time estimation (CBA thresholds + IL deduction)
# ---------------------------------------------------------------------------

def v10_service_time() -> Result:
    """
    MLB service time (service_time.py):
      1. years.days arithmetic: '5.143' is 5*172 + 143 days and 5 whole years;
         the list's position codes read as pitcher / non-pitcher / two-way
      2. same-named players separate by continuity: a veteran's chain and a
         rookie whose total exceeds what his seasons could hold
      3. on the real lists (where present): Verlander 20 and Kershaw 17 years
         through 2025; no matched player's total falls, or grows past a season
         plus CHAIN_SLACK, between consecutive lists
      4. each season with a list pair and a roster ledger: last season's list
         plus this season's ledger days gives the same WHOLE years as this
         season's list for 96%+ of players, and never misses by more than one
         — the ledger's job wherever a player is off a list, and all of 2026
         (98.8% overall; 2024-25 run 97-98%, where up-and-down players come
         out 14-20 days short, likely MLB's archive updating status late)
    The calendar estimate this replaced asserted that anyone who debuted by
    2015 had 6+ years by 2023 — false for every player who went up and down.
    """
    import pandas as pd
    import service_time as st

    errors = []
    if st.mls_days("5.143") != 5 * 172 + 143 or st.service_years(5 * 172 + 143) != 5:
        errors.append("years.days arithmetic")
    want = {"rhp": True, "rhp-s": True, "rhp-c": True, "c": False, "c-1b": False,
            "ss-2b": False, "dh-rhp": None}
    got = {k: st.is_pitcher_pos(k) for k in want}
    if got != want:
        errors.append(f"position codes {got}")

    # 2. two pitchers named alike: 1 debuted 2010, 2 debuted 2018
    info = pd.DataFrame({"mlb_played_first": [2010, 2018], "mlb_played_last": [2020, 2025],
                         "mlb_pitcher": [True, True]}, index=[1, 2])
    res = pd.DataFrame({
        "season": [2015, 2019, 2019], "days": [900, 1500, 200], "pitcher": [True] * 3,
        "key_mlbam": [None] * 3, "method": ["ambiguous"] * 3, "candidates": [[1, 2]] * 3,
        "name_key": [("smith", "joe")] * 3, "all_cands": [(1, 2)] * 3})
    st._resolve_by_continuity(res, info, set())
    if list(res["key_mlbam"]) != [1, 1, 2]:
        errors.append(f"continuity split same-named players as {list(res['key_mlbam'])}")

    detail_real = "synthetic checks only: no MLS lists on this machine"
    if any(st.MLS_DIR.glob("mls_thru_*.csv")):
        m = st.match_mls_ids()
        through = st.service_days_through(2025, m)
        for pid, name, years in ((434378, "Verlander", 20), (477132, "Kershaw", 17)):
            got_y = st.service_years(through.get(pid, 0))
            if got_y != years:
                errors.append(f"{name} through 2025: {got_y} years, list says {years}")
        ok = m.dropna(subset=["key_mlbam", "days"])
        w = ok.pivot(index="key_mlbam", columns="season", values="days")
        breaks = sum(int((((w[a + 1] - w[a]).dropna() < 0)
                          | ((w[a + 1] - w[a]).dropna() > 172 + st.CHAIN_SLACK)).sum())
                     for a in w.columns if a + 1 in w.columns)
        if breaks:
            errors.append(f"{breaks} consecutive-list chain breaks")
        r = st.reconcile(m)
        listed = ok.drop_duplicates(["season", "key_mlbam"]).set_index(["season", "key_mlbam"])["days"]
        seasons = []
        for season, g in r.groupby("season"):
            true = g.apply(lambda x: listed[(x.season, x.key_mlbam)], axis=1)
            est = true - g["mls_increase"] + g["ledger_days"]
            miss = (true // 172) - (est // 172)
            share = float((miss == 0).mean())
            seasons.append(f"{season} {share:.0%}")
            if share < 0.96:
                errors.append(f"{season}: whole years agree for only {share:.1%}")
            if miss.abs().max() > 1:
                errors.append(f"{season}: a whole-year count off by {int(miss.abs().max())}")
        resolved = float(m.loc[m["days"].fillna(0) > 0, "key_mlbam"].notna().mean())
        detail_real = (f"{resolved:.1%} of list rows matched; whole years from the "
                       f"ledger agree: {', '.join(seasons)}")

    passed = not errors
    return passed, (detail_real if passed else "; ".join(errors))

# ---------------------------------------------------------------------------
# V11 — Spin efficiency pipeline
# ---------------------------------------------------------------------------

def v11_spin_efficiency() -> Result:
    """
    Verify the spin efficiency pipeline on cached 2023 Statcast (offline —
    Savant values are injected, never fetched):
      1. compute_spin_efficiency yields values in [0, 1] for most pitches
      2. the physics is sane: four-seamers mostly active, sliders mostly gyro
      3. attach_savant_active_spin replaces a row's value and marks its source
      4. percentiles are ranked within pitch type, not pooled
      5. build_pitcher_spin_profile returns one row per pitcher
    """
    import pandas as pd
    from ingest import pull_statcast_range
    from spin_efficiency import (
        compute_spin_efficiency,
        aggregate_pitcher_pitch_type,
        attach_savant_active_spin,
        build_pitcher_spin_profile,
    )

    errors = []

    try:
        sc = pull_statcast_range("2023-06-01", "2023-06-07", label="test")
    except Exception as exc:
        return False, f"Failed to load cached statcast: {exc}"

    # Step 1 — pitch level
    try:
        df_se = compute_spin_efficiency(sc)
    except Exception as exc:
        return False, f"compute_spin_efficiency raised: {exc}"
    eff = df_se["spin_efficiency"]
    if eff.notna().mean() < 0.9:
        errors.append(f"only {eff.notna().mean():.0%} of pitches got a spin_efficiency")
    if not eff.dropna().between(0, 1).all():
        errors.append("spin_efficiency values outside [0, 1]")

    # Step 2 — physics sanity (Savant measured medians: FF ≈ 0.91, SL ≈ 0.31)
    med = df_se.groupby("pitch_type")["spin_efficiency"].median()
    if not med.get("FF", 0) > 0.8:
        errors.append(f"FF median spin efficiency {med.get('FF')} — expected > 0.8")
    if not med.get("SL", 1) < 0.5:
        errors.append(f"SL median spin efficiency {med.get('SL')} — expected < 0.5")

    # Step 3 — Savant values take precedence where present
    agg = aggregate_pitcher_pitch_type(df_se)
    if agg.empty:
        return False, "aggregate_pitcher_pitch_type returned empty DataFrame"
    probe = agg.iloc[0]
    fake = pd.DataFrame({"pitcher": [probe["pitcher"]],
                         "pitch_type": [probe["pitch_type"]],
                         "active_spin": [0.123]})
    attached = attach_savant_active_spin(agg, fake)
    hit = attached[(attached["pitcher"] == probe["pitcher"])
                   & (attached["pitch_type"] == probe["pitch_type"])].iloc[0]
    if hit["spin_efficiency_raw"] != 0.123 or hit["spin_source"] != "savant":
        errors.append("attach_savant_active_spin did not replace the physics value")
    if (attached["spin_source"] == "savant").sum() != 1:
        errors.append("attach_savant_active_spin touched rows Savant did not cover")

    # Step 4/5 — full pipeline, offline
    empty = pd.DataFrame(columns=["pitcher", "pitch_type", "active_spin"])
    try:
        pt_level, p_level = build_pitcher_spin_profile(sc, active_spin=empty)
    except Exception as exc:
        return False, f"build_pitcher_spin_profile raised: {exc}"

    top = pt_level.groupby("pitch_type")["spin_efficiency_pct"].max()
    if not (top.round(6) == 100).all():
        errors.append("spin_efficiency_pct is not ranked within each pitch type")
    if p_level.empty:
        errors.append("pitcher_level spin profile is empty")
    else:
        for col in ("spin_efficiency_weighted", "spin_efficiency_pct", "spin_savant_share"):
            if col not in p_level.columns:
                errors.append(f"pitcher_level missing '{col}'")
        if p_level["pitcher"].nunique() != len(p_level):
            errors.append("pitcher_level has duplicate pitchers")

    passed = len(errors) == 0
    detail = (
        f"Pipeline OK — {len(sc)} pitches → {len(pt_level)} pitcher×type rows → "
        f"{len(p_level)} pitchers; FF median {med.get('FF', float('nan')):.2f}, "
        f"SL median {med.get('SL', float('nan')):.2f}"
        if passed else "; ".join(errors)
    )
    return passed, detail


# ---------------------------------------------------------------------------
# V13 — Spin efficiency against Savant's measured active spin
# ---------------------------------------------------------------------------

# Floors sit a few points under what the estimator reached on 2024–26
# (FF 0.92, SI 0.84, CU 0.93, SL 0.83, ST 0.76, FC 0.89, CH 0.59, FS 0.65).
# CH and FS are low for any movement-based method: seam-shifted wake moves
# those pitches without spin.
V13_MIN_SPEARMAN = {"FF": 0.85, "SI": 0.75, "CU": 0.85, "SL": 0.75,
                    "ST": 0.65, "FC": 0.80, "CH": 0.50, "FS": 0.50}
V13_LEVEL_BAND = (0.85, 1.15)   # median(physics) / median(Savant), per type
V13_SEASON = 2025


def v13_spin_vs_savant() -> Result:
    """
    The physics estimate (used before 2020, and for rows Savant omits) must
    rank pitchers like Savant's measured active spin and sit on its scale:
      1. Spearman per pitch type ≥ V13_MIN_SPEARMAN
      2. median level ratio per pitch type inside V13_LEVEL_BAND
    Uses one full season of cached Statcast and Savant's leaderboard (cached
    after the first pull).
    """
    import pandas as pd
    from scipy.stats import spearmanr
    from ingest import pull_statcast_season
    from spin_efficiency import (
        aggregate_pitcher_pitch_type,
        compute_spin_efficiency,
        load_savant_active_spin,
    )

    try:
        sc = pull_statcast_season(V13_SEASON, top_up=False)
        savant = load_savant_active_spin(V13_SEASON)
    except Exception as exc:
        return False, f"inputs unavailable: {exc}"

    agg = aggregate_pitcher_pitch_type(compute_spin_efficiency(sc))
    agg = agg[agg["pitch_count"] >= 50]
    j = agg.merge(savant, on=["pitcher", "pitch_type"])

    errors, parts = [], []
    lo, hi = V13_LEVEL_BAND
    for ptype, floor in V13_MIN_SPEARMAN.items():
        x = j[j["pitch_type"] == ptype]
        if len(x) < 30:
            errors.append(f"{ptype}: only {len(x)} rows matched Savant")
            continue
        rho = spearmanr(x["spin_efficiency_raw"], x["active_spin"])[0]
        level = x["spin_efficiency_raw"].median() / x["active_spin"].median()
        parts.append(f"{ptype} ρ={rho:.2f} lvl={level:.2f}")
        if rho < floor:
            errors.append(f"{ptype}: Spearman {rho:.2f} < {floor}")
        if not lo <= level <= hi:
            errors.append(f"{ptype}: level {level:.2f} outside [{lo}, {hi}]")

    passed = len(errors) == 0
    return passed, ("; ".join(parts) if passed else "; ".join(errors))


# ---------------------------------------------------------------------------
# V12 — Park factors
# ---------------------------------------------------------------------------

def v12_park_factors() -> Result:
    """
    Verify build_park_profile on cached 2023 Statcast:
      1. Returns a non-empty DataFrame with required columns
      2. pitcher_friendly is in [0, 100] for all teams
      3. Target team 'HOU' is present
      4. Higher woba_pf → lower pitcher_friendly (inversion correct)
    """
    import pandas as pd
    from ingest import pull_statcast_range
    from park_effects import build_park_profile

    errors = []

    try:
        sc = pull_statcast_range("2023-06-01", "2023-06-07", label="test")
    except Exception as exc:
        return False, f"Failed to load cached statcast: {exc}"

    try:
        parks = build_park_profile(sc, 2023)
    except Exception as exc:
        return False, f"build_park_profile raised: {exc}"

    if parks.empty:
        return False, "build_park_profile returned empty DataFrame"

    # Required columns
    required = {"team", "woba_pf", "pitcher_friendly"}
    missing = required - set(parks.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")
        return False, "; ".join(errors)

    # pitcher_friendly in [0, 100]
    pf = parks["pitcher_friendly"]
    if not pf.between(0, 100).all():
        errors.append(f"pitcher_friendly out of [0,100]: min={pf.min():.1f} max={pf.max():.1f}")

    # HOU should be present (7 days of June 2023 data covers multiple parks)
    if "HOU" not in parks["team"].values and len(parks) < 5:
        errors.append(f"Surprisingly few teams in park profile: {parks['team'].tolist()}")

    # Inversion: higher raw woba_pf (hitter-friendly) → lower pitcher_friendly
    if len(parks) >= 2:
        parks_sorted = parks.sort_values("woba_pf")
        most_hitter = parks_sorted.iloc[-1]["pitcher_friendly"]
        most_pitcher = parks_sorted.iloc[0]["pitcher_friendly"]
        if most_hitter >= most_pitcher:
            errors.append(
                f"Inversion error: most hitter-friendly park has pitcher_friendly={most_hitter:.1f} "
                f">= most pitcher-friendly={most_pitcher:.1f}"
            )

    passed = len(errors) == 0
    detail = (
        f"{len(parks)} teams — pitcher_friendly range {pf.min():.1f}–{pf.max():.1f}"
        if passed else "; ".join(errors)
    )
    return passed, detail


# ---------------------------------------------------------------------------
# V14 — Metric rank direction
# ---------------------------------------------------------------------------

def v14_metric_directions() -> Result:
    """
    Every ranked metric declares its direction, and the outcome gates read
    those directions the right way round:
      1. every BATTING_COL_MAP / PITCHING_COL_MAP target is in its
         *_RANK_INVERT table (no metric falls back to a default)
      2. allowing less hard contact, fewer barrels or fewer walks never lowers
         the pitcher's percentile (monotonic in the good direction)
      3. `contact suppressor` goes to the low-hard-hit end of a pool and
         `walk prone` to the high-walk end — never the reverse
    """
    import pandas as pd
    import team_portrait as tp
    from pitcher_traits import outcome_traits

    errors = []
    for cmap, table, label in ((tp.BATTING_COL_MAP, tp.BATTING_RANK_INVERT, "batting"),
                               (tp.PITCHING_COL_MAP, tp.PITCHING_RANK_INVERT, "pitching")):
        undeclared = sorted(set(cmap.values()) - set(table))
        if undeclared:
            errors.append(f"{label} metrics with no declared direction: {undeclared}")

    # A pool of 101 pitchers whose rates allowed rise together, raw rates as
    # ingest stores them (hard_hit_pct 0.20-0.60, not Savant percentiles).
    n = 101
    hh = np.linspace(0.20, 0.60, n)
    pool = pd.DataFrame({
        "key_mlbam":    np.arange(n),
        "hard_hit_pct": hh,
        "barrel_pct":   hh / 5,
        "bb_rate":      np.linspace(0.03, 0.15, n),
    })
    ranked = tp._normalize_pitching(pool, pool.copy())
    for col in ("HardHit_allowed_pct", "Barrel_allowed_pct", "BB_pct_pitch_pct"):
        if col not in ranked.columns:
            errors.append(f"{col} not produced")
            continue
        # raw rate allowed rises down the frame, so the pct must fall
        if not ranked[col].is_monotonic_decreasing:
            errors.append(f"{col} rises with the rate allowed")

    def tags_of(i):
        r = ranked.iloc[i]
        m = {"HardHit_allowed_pct": r.get("HardHit_allowed_pct"),
             "Barrel_allowed_pct": r.get("Barrel_allowed_pct"),
             "BB_pct_pct": r.get("BB_pct_pitch_pct")}
        return {t["tag"] for t in outcome_traits(m)}

    best, worst = tags_of(0), tags_of(n - 1)
    if "contact suppressor" not in best:
        errors.append("lowest hard-hit rate allowed is not a contact suppressor")
    if "contact suppressor" in worst:
        errors.append("highest hard-hit rate allowed is a contact suppressor")
    if "walk prone" not in worst or "walk prone" in best:
        errors.append("walk prone is not on the high-walk end")

    passed = not errors
    detail = (f"{len(tp.BATTING_RANK_INVERT)} batting + {len(tp.PITCHING_RANK_INVERT)} "
              f"pitching directions declared; hard-hit {hh[0]:.0%} → "
              f"{ranked['HardHit_allowed_pct'].iloc[0]:.0f}th pct, "
              f"{hh[-1]:.0%} → {ranked['HardHit_allowed_pct'].iloc[-1]:.0f}th"
              if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V15 — Batted-ball direction
# ---------------------------------------------------------------------------

def v15_spray_direction() -> Result:
    """
    Pull and oppo are relative to the batter, and point the right way:
      1. synthetic balls to left field are pull for a right-handed batter and
         oppo for a left-handed one; right field the reverse
      2. boundaries: 15° is center, 15.1° pull gap, 30.1° pull line, and the
         per-player shares agree with the chart zones on the same balls
      3. on real pitches (cached June 2023 week) both hands pull their ground
         balls more than they hit them the other way — the most reliable
         directional fact in batted-ball data
    """
    import pandas as pd
    from pitch_aggregates import (SPRAY_HOME_X, SPRAY_HOME_Y, compute_batter_spray_stats,
                                  pull_angle, spray_angle, spray_zone)

    errors = []

    def ball(angle_deg: float, r: float = 150.0) -> tuple[float, float]:
        t = np.radians(angle_deg)       # + is right field
        return SPRAY_HOME_X + r * np.sin(t), SPRAY_HOME_Y - r * np.cos(t)

    rows = []
    for batter, stand, angle in [(1, "R", -35), (2, "L", -35), (3, "R", 35), (4, "L", 35)]:
        x, y = ball(angle)
        rows += [dict(batter=batter, stand=stand, hc_x=x, hc_y=y, bb_type="line_drive",
                      events="single")] * 20
    stats = compute_batter_spray_stats(pd.DataFrame(rows), min_bip=1).set_index("key_mlbam")
    want = {1: (1.0, 0.0), 2: (0.0, 1.0), 3: (0.0, 1.0), 4: (1.0, 0.0)}
    for b, (pull, oppo) in want.items():
        got = (stats.loc[b, "pull_pct"], stats.loc[b, "oppo_pct"])
        if got != (pull, oppo):
            errors.append(f"batter {b}: pull/oppo {got}, want {(pull, oppo)}")

    angles = pd.Series([15.0, 15.1, 30.0, 30.1, -15.0, -15.1, -30.1])
    zones = spray_zone(pull_angle(angles, pd.Series(["L"] * len(angles)))).tolist()
    expect = ["center", "pull_gap", "pull_gap", "pull_line", "center", "oppo_gap", "oppo_line"]
    if zones != expect:
        errors.append(f"boundary zones {zones}, want {expect}")

    try:
        from ingest import pull_statcast_range
        sc = pull_statcast_range("2023-06-01", "2023-06-07", label="test")
        gb = sc[(sc["bb_type"] == "ground_ball") & sc["hc_x"].notna()]
        pa = pull_angle(spray_angle(gb["hc_x"], gb["hc_y"]), gb["stand"])
        for hand in ("R", "L"):
            p = pa[gb["stand"] == hand]
            pull, oppo = (p > 15).mean(), (p < -15).mean()
            if not pull > oppo:
                errors.append(f"{hand}HB ground balls pulled {pull:.0%} vs oppo {oppo:.0%}")
        real = (f"; June 2023 grounders pulled R {(pa[gb['stand'] == 'R'] > 15).mean():.0%}"
                f" / L {(pa[gb['stand'] == 'L'] > 15).mean():.0%}")
    except Exception as exc:
        errors.append(f"real-data check failed: {exc}")
        real = ""

    passed = not errors
    detail = (f"synthetic L/R pull/oppo and boundaries correct{real}"
              if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V16 — Tag evidence coverage
# ---------------------------------------------------------------------------

def v16_tag_evidence() -> Result:
    """
    Every tag the portraits can emit is either gated on a metric or exempt
    for a stated reason (reliability.validate_tag_evidence), and the gate
    actually bites: a 20-BF pitcher with the worst walk and ground-ball rates
    in the pool loses `walk prone` and `fly-ball prone`, and keeps them at
    200 BF.
    """
    import team_portrait as tp
    from pitcher_traits import outcome_traits
    from reliability import TAG_EVIDENCE, UNGATED_TAGS, filter_traits, validate_tag_evidence

    errors = list(validate_tag_evidence(tp.TAG_KIND))

    raw = outcome_traits({"BB_pct_pct": 2.0, "GB_pct_pct": 2.0})
    emitted = {t["tag"] for t in raw}
    want = {"walk prone", "fly-ball prone"}
    if not want <= emitted:
        errors.append(f"test pitcher should emit {sorted(want)}, got {sorted(emitted)}")
    small = {t["tag"] for t in filter_traits(raw, 20)}
    large = {t["tag"] for t in filter_traits(raw, 200)}
    if small & want:
        errors.append(f"20 BF still carries {sorted(small & want)}")
    if not want <= large:
        errors.append(f"200 BF lost {sorted(want - large)}")

    passed = not errors
    detail = (f"{len(tp.TAG_KIND)} tags: {len(TAG_EVIDENCE)} gated, {len(UNGATED_TAGS)} "
              "exempt with a reason; walk prone / fly-ball prone refused at 20 BF"
              if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V17 — Season completeness
# ---------------------------------------------------------------------------

def v17_season_completeness() -> Result:
    """
    Offline (MLB's schedule is stubbed):
      1. statcast_missing_games finds a game missing from the START of a
         season and one from the END — the two the date table used to lose —
         ignores today's, and reports nothing for a complete cache
      2. a season is still in progress the morning after its last game (so
         the nightly job collects it) and finished the day after that
      3. SEASON_DATES starts no later than the international openers it had
         missed (2019, 2025 Tokyo; 2026 Mar 25) and 2024 ends Sep 30
    """
    import datetime as dt
    import pandas as pd
    import ingest

    errors = []
    today = dt.date.today()
    sched = pd.DataFrame({
        "game_pk": [1, 2, 3, 4],
        "date": ["2025-03-18", "2025-06-01", "2025-09-28", today.isoformat()],
    })
    sched["official_date"] = sched["date"]
    real = ingest.regular_season_games
    ingest.regular_season_games = lambda season: sched
    try:
        cache = pd.DataFrame({"game_pk": [2]})
        missing = sorted(ingest.statcast_missing_games(2025, cache)["game_pk"])
        if missing != [1, 3]:
            errors.append(f"missing games {missing}, want [1, 3] (today's excluded)")
        full = pd.DataFrame({"game_pk": [1, 2, 3]})
        if not ingest.statcast_missing_games(2025, full).empty:
            errors.append("a complete cache reported missing games")
    finally:
        ingest.regular_season_games = real

    end = dt.date.fromisoformat(ingest.SEASON_DATES[2026][1])
    _, ip_next = ingest.season_state(end + dt.timedelta(days=1))
    fin_after, _ = ingest.season_state(end + dt.timedelta(days=2))
    if ip_next != 2026:
        errors.append("the morning after the last game no longer counts as in progress")
    if 2026 not in fin_after:
        errors.append("two days after the last game the season is not finished")

    for season, first in [(2019, "2019-03-20"), (2025, "2025-03-18"), (2026, "2026-03-25")]:
        if ingest.SEASON_DATES[season][0] > first:
            errors.append(f"{season} starts {ingest.SEASON_DATES[season][0]}, after {first}")
    if ingest.SEASON_DATES[2024][1] < "2024-09-30":
        errors.append("2024 ends before the Sep 30 doubleheader")

    passed = not errors
    detail = ("missing-game detection, finished rule and opener dates correct"
              if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V18 — Regular season only
# ---------------------------------------------------------------------------

# Statcast's terminal-pitch count is not an official PA count (it misses a few
# PAs per player-season and counts the odd non-PA event), so a little slack;
# a postseason leak is 20-50 PA for a regular, far beyond it.
_PA_SLACK = 5
# A leak is systemic: it hits every regular on a playoff roster, about a
# hundred players a season before the fix. Statcast's own gaps hit one or two
# (Ellsbury 2016: 626 PA vs 618, two games missing from the feed), and only
# before 2019. So a season fails when this many players run over the slack;
# fewer are reported as Statcast gaps and pass.
_LEAK_MIN_PLAYERS = 5


def v18_regular_season_only() -> Result:
    """
    On whatever data this machine has:
      1. every raw Statcast cache holds regular-season pitches only
      2. no player's season total in fg_batting / fg_pitching exceeds the PA
         or BF Statcast counts for him in that season's regular-season games
         by more than _PA_SLACK for _LEAK_MIN_PLAYERS or more players — the
         signature of postseason games leaking into the BRef totals (Soto
         2022: 716 PA vs 664 regular season). Fewer are named as gaps.
    Seasons without local files are skipped and named in the detail.
    """
    import pandas as pd
    from ingest import PROCESSED_DIR, RAW_DIR, SEASON_DATES

    errors, gaps, checked, skipped = [], [], [], []
    for season in sorted(SEASON_DATES):
        raw = RAW_DIR / f"statcast_{season}.parquet"
        if not raw.exists():
            skipped.append(season)
            continue
        sc = pd.read_parquet(raw, columns=["game_type", "batter", "pitcher", "events"])
        other = sc["game_type"][sc["game_type"] != "R"].value_counts().to_dict()
        if other:
            errors.append(f"{season} raw cache holds non-regular-season pitches {other}")
        from pitch_aggregates import _NON_PA_EVENTS
        ends = sc[(sc["game_type"] == "R") & sc["events"].notna()
                  & ~sc["events"].isin(_NON_PA_EVENTS)]
        counts = {"batter": ends.groupby("batter").size(),
                  "pitcher": ends.groupby("pitcher").size()}
        for fname, col, who in ((f"fg_batting_{season}.csv", "PA", "batter"),
                                (f"fg_pitching_{season}.csv", "BF", "pitcher")):
            path = PROCESSED_DIR / fname
            if not path.exists():
                continue
            fg = pd.read_csv(path, low_memory=False)
            if col not in fg.columns or "key_mlbam" not in fg.columns:
                continue
            fg = fg[["key_mlbam", col]].dropna()
            sc_n = fg["key_mlbam"].astype(int).map(counts[who]).fillna(0)
            fg = fg.assign(sc=sc_n, gap=fg[col].astype(float) - sc_n)
            over = fg[fg["gap"] > _PA_SLACK].sort_values("gap")
            if len(over) >= _LEAK_MIN_PLAYERS:
                worst = over.iloc[-1]
                errors.append(f"{fname}: {len(over)} player(s) with more {col} than "
                              f"regular-season Statcast (e.g. {int(worst['key_mlbam'])}: "
                              f"{int(worst[col])} vs {int(worst['sc'])})")
            else:
                gaps += [f"{season} {int(r.key_mlbam)} {int(getattr(r, col))} vs "
                         f"{int(r.sc)} {col}" for r in over.itertuples()]
        checked.append(season)

    passed = not errors
    detail = (f"seasons checked {checked}" + (f"; no local data {skipped}" if skipped else "")
              + (f"; Statcast gaps, not leaks: {', '.join(gaps)}" if gaps else "")
              if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V19 — Team stints
# ---------------------------------------------------------------------------

def v19_team_stints() -> Result:
    """
    A team gets what it got, not the player's season:
      1. on cached pitches (June 2023 week), every team's batting stints sum
         to the team's plate appearances and its pitching stints to its
         batters faced, every team-game has exactly one starter, and each
         player's stints sum to the player's total
      2. a synthetic player traded between two teams: each team's row carries
         its own PA and WAR, the season total sits in pa_total / war_total,
         both list the two teams, and a stint under MIN_HITTER_PA goes to
         also-appeared rather than the tagged roster
    """
    import pandas as pd
    import team_portrait as tp
    from ingest import pull_statcast_range
    from pitch_aggregates import _NON_PA_EVENTS, compute_stints

    errors = []
    sc = pull_statcast_range("2023-06-01", "2023-06-07", label="test")
    sc = sc[sc["game_type"] == "R"] if "game_type" in sc.columns else sc
    bat, pit = compute_stints(sc, -2023)       # negative key: never collides with a season
    ends = sc[sc["events"].notna() & ~sc["events"].isin(_NON_PA_EVENTS)]
    top = ends["inning_topbot"] == "Top"
    team_pa = pd.Series(np.where(top, ends["away_team"], ends["home_team"])).value_counts()
    team_bf = pd.Series(np.where(top, ends["home_team"], ends["away_team"])).value_counts()
    # Compared as plain dicts: Statcast's nullable Int64 vs value_counts'
    # int64 makes Series.equals False on identical numbers.
    def same(a: pd.Series, b: pd.Series) -> bool:
        return {k: int(v) for k, v in a.items()} == {k: int(v) for k, v in b.items()}

    if not same(bat.groupby("team")["pa"].sum(), team_pa):
        errors.append("batting stints do not sum to team PA")
    if not same(pit.groupby("team")["bf"].sum(), team_bf):
        errors.append("pitching stints do not sum to team BF")
    games = sc.groupby(["game_pk"]).size().shape[0]
    if int(pit["gs"].sum()) != 2 * games:
        errors.append(f"{int(pit['gs'].sum())} starts for {games} games (want 2 per game)")
    if not same(bat.groupby("batter")["pa"].sum(), ends.groupby("batter").size()):
        errors.append("a player's stints do not sum to the player's PA")

    stints = pd.DataFrame({"batter": [1, 1, 2], "team": ["AAA", "BBB", "AAA"],
                           "pa": [400, 25, 300], "g": [100, 8, 80]})
    war = pd.DataFrame({"key_mlbam": [1, 1, 2], "team": ["AAA", "BBB", "AAA"],
                        "war": [3.0, 0.2, 1.0]})
    rows = pd.DataFrame({"key_mlbam": [1, 2], "pa": [425, 300], "war": [3.2, 1.0]})
    a_keep, a_cameo = tp._team_stints_batting(rows, stints, war, "AAA")
    b_keep, b_cameo = tp._team_stints_batting(rows[rows.key_mlbam == 1], stints, war, "BBB")
    r = a_keep.set_index("key_mlbam").loc[1]
    if (r["pa"], r["pa_total"], r["war"], r["war_total"]) != (400, 425, 3.0, 3.2):
        errors.append(f"AAA row: pa {r['pa']}/{r['pa_total']} war {r['war']}/{r['war_total']}")
    if r["season_teams"] != ["AAA", "BBB"]:
        errors.append(f"season_teams {r['season_teams']}")
    if not b_keep.empty or list(b_cameo["key_mlbam"]) != [1] or int(b_cameo["pa"].iloc[0]) != 25:
        errors.append("a 25-PA stint should be also-appeared, not tagged")
    if a_keep.set_index("key_mlbam").loc[2, "season_teams"] is not None:
        errors.append("a one-team player should carry no season_teams")

    passed = not errors
    detail = (f"{len(team_pa)} teams reconcile PA/BF/starts on cached pitches; "
              "synthetic trade splits PA and WAR by team" if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V20 — League caches current
# ---------------------------------------------------------------------------

def v20_league_caches_current() -> Result:
    """
      1. refresh.stage_league rebuilds exactly the caches the build fingerprint
         hashes — one list cannot gain a cache the other lacks
      2. on this machine's data, no league cache is older than its season's
         raw Statcast (2026's five were 92.8 days older when this was added)
    """
    import importlib
    import team_portrait as tp
    from ingest import RAW_DIR, SEASON_DATES

    sys.path.insert(0, str(_HERE.parent / "scripts"))
    refresh = importlib.import_module("refresh")
    errors = []
    rebuilt = {pat for pat, _ in refresh._league_caches()}
    hashed = {pat for _, pat in tp._FINGERPRINT_EXTRA_POOLS}
    if rebuilt != hashed:
        errors.append(f"refresh rebuilds {sorted(rebuilt - hashed)} not fingerprinted; "
                      f"fingerprinted {sorted(hashed - rebuilt)} not rebuilt")

    stale, checked = [], 0
    for season in sorted(SEASON_DATES):
        raw = RAW_DIR / f"statcast_{season}.parquet"
        if not raw.exists():
            continue
        for directory, pat in tp._FINGERPRINT_EXTRA_POOLS:
            cache = directory / pat.format(s=season)
            if not cache.exists():
                continue
            checked += 1
            days = (raw.stat().st_mtime - cache.stat().st_mtime) / 86400
            if days > 0:
                stale.append(f"{cache.name} ({days:.1f} d)")
    if stale:
        errors.append("older than raw Statcast: " + ", ".join(stale[:8])
                      + (f" … +{len(stale) - 8}" if len(stale) > 8 else ""))

    passed = not errors
    detail = (f"{len(hashed)} caches rebuilt and fingerprinted alike; "
              f"{checked} on disk, none older than raw" if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V21 — Sprint speed units
# ---------------------------------------------------------------------------

# Savant's sprint-speed leaderboard, ft/s. Soto 2022 is the case the
# percentile table got wrong: rank 32, which passed as an elite 32.0 ft/s.
_SPRINT_REFERENCE = {(665742, 2022): 26.6,   # Juan Soto
                     (677951, 2022): 30.3,   # Bobby Witt Jr.
                     (621439, 2022): 29.1}   # Byron Buxton


def v21_sprint_speed_units() -> Result:
    """
    sprint_speed in each fg_batting root is a measurement in ft/s, not a
    percentile: whole numbers are a minority (a rank is always whole; about
    9% of real speeds are), the median sits in the 26-28.5 ft/s band, most
    qualified hitters have one, and the reference players match Savant.
    """
    import pandas as pd
    from ingest import PROCESSED_DIR, SEASON_DATES

    errors, checked = [], []
    for season in sorted(SEASON_DATES):
        path = PROCESSED_DIR / f"fg_batting_{season}.csv"
        if not path.exists():
            continue
        fg = pd.read_csv(path, low_memory=False)
        if "sprint_speed" not in fg.columns:
            errors.append(f"{season}: no sprint_speed column")
            continue
        ss = pd.to_numeric(fg["sprint_speed"], errors="coerce")
        have = ss.dropna()
        whole = float((have % 1 == 0).mean()) if len(have) else 1.0
        if len(have) < 0.5 * len(fg):
            errors.append(f"{season}: only {len(have)}/{len(fg)} hitters have a speed")
        if whole > 0.5:
            errors.append(f"{season}: {whole:.0%} of speeds are whole numbers — a percentile")
        if len(have) and not 26.0 <= have.median() <= 28.5:
            errors.append(f"{season}: median {have.median():.1f} ft/s")
        for (pid, s), ref in _SPRINT_REFERENCE.items():
            if s != season or "key_mlbam" not in fg.columns:
                continue
            got = ss[pd.to_numeric(fg["key_mlbam"], errors="coerce") == pid]
            if len(got) and abs(float(got.iloc[0]) - ref) > 0.3:
                errors.append(f"{season} {pid}: {float(got.iloc[0]):.1f} ft/s, Savant {ref}")
        checked.append(season)

    passed = not errors and bool(checked)
    detail = (f"ft/s in seasons {checked}; references match Savant" if passed
              else "; ".join(errors) or "no fg_batting roots on this machine")
    return passed, detail


# ---------------------------------------------------------------------------
# V22 — Complete reads and rosters
# ---------------------------------------------------------------------------

def v22_complete_rosters() -> Result:
    """
    Nothing is lost between the database and a portrait:
      1. query_batting / query_pitching return every row the table holds for
         each seeded season — the API caps a request at 1,000 rows, and the
         unpaged read lost 86-204 regular hitters a season from 2015-2019
         and 2021 (Votto, Walker, Voit on their 2021 clubs)
      2. every hitter with MIN_HITTER_PA for a team in regular-season Statcast
         is on that team's portrait, in hitters or also-appeared
    Each season's reconciliation runs where both its raw cache and portraits
    exist; 2025 alone passed both before the fix, since it is under the cap.
    """
    import json
    import pandas as pd
    import team_portrait as tp
    sys.path.insert(0, str(_HERE.parent / "scripts"))
    from build_league_identity import MLB_TEAMS
    from database import count_rows, query_batting, query_pitching
    from ingest import PROCESSED_DIR, RAW_DIR, SEASON_DATES
    from pitch_aggregates import compute_stints

    errors, checked = [], []
    portraits = PROCESSED_DIR / "portraits"
    for season in sorted(SEASON_DATES):
        for table, query in (("player_batting", query_batting),
                             ("player_pitching", query_pitching)):
            n = count_rows(table, season)
            got = len(query(season))
            if got != n:
                errors.append(f"{table} {season}: read {got} of {n} rows")

        raw = RAW_DIR / f"statcast_{season}.parquet"
        if not raw.exists() or not (portraits / f"NYY_{season}.json").exists():
            continue
        sc = pd.read_parquet(raw, columns=["game_pk", "at_bat_number", "pitch_number",
                                           "inning", "inning_topbot", "home_team",
                                           "away_team", "batter", "pitcher", "events",
                                           "stand", "p_throws", "delta_home_win_exp"])
        bat, _ = compute_stints(sc, season)
        missing = []
        for team in MLB_TEAMS:
            f = portraits / f"{team}_{season}.json"
            if not f.exists():
                continue
            p = json.loads(f.read_text())["players"]
            on = {h.get("player_id") for g in ("hitters", "also_appeared")
                  for h in p.get(g, [])}
            need = bat[(bat["team"] == team) & (bat["pa"] >= tp.MIN_HITTER_PA)]
            missing += [f"{team} {int(b)}" for b in need["batter"] if int(b) not in on]
        if missing:
            errors.append(f"{season}: {len(missing)} hitter(s) with {tp.MIN_HITTER_PA}+ "
                          f"PA missing from their portraits (e.g. {', '.join(missing[:3])})")
        checked.append(season)

    passed = not errors
    detail = (f"every row read; rosters complete in {checked}" if passed
              else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# V23 — Pitch outcomes and pairing
# ---------------------------------------------------------------------------

def v23_pitch_outcomes_pairing() -> Result:
    """
    Offline, on synthetic pitches:
      1. a foul tip is a whiff (Savant's definition) in the shared sets, and
         build_pitch_mix gives 150 foul tips + 150 balls swstr 0.5 and whiffs
         per swing 1.0 — the review's case read csw 1.0 off an all-pitches
         denominator under the name whiff_pct
      2. a tunnel pair needs back-to-back pitches: pitches 1 and 3 around an
         untracked pitch 2 make no pair
      3. the sequencing ratio uses one transition sample: a pitcher who opens
         every PA with FF and follows 50/50 FF/SL scores 1.0 (was 1.23), and
         no ratio exceeds 1 on random sequences
    """
    import tempfile
    from pathlib import Path as _Path
    import pandas as pd
    import pitch_mix as pm
    from pitch_aggregates import CSW_EVENTS, SWING_EVENTS, WHIFF_EVENTS
    from pitcher_traits import _pitcher_sequencing
    from tunneling import build_pitch_pairs

    errors = []

    # 1. outcomes
    if not ("foul_tip" in WHIFF_EVENTS and "foul_tip" in SWING_EVENTS
            and "foul_tip" in CSW_EVENTS):
        errors.append("foul_tip is not a whiff, swing and CSW event")
    n = 300
    sc = pd.DataFrame({
        "pitcher": 1, "game_year": 2099, "p_throws": "R", "pitch_type": "SL",
        "release_speed": 85.0, "pfx_z": 0.1, "pfx_x": 0.3, "release_spin_rate": 2400.0,
        "spin_axis": 90.0, "bb_type": None,
        "description": ["foul_tip"] * (n // 2) + ["ball"] * (n // 2),
    })
    tmp = _Path(tempfile.mkdtemp()) / "pm.parquet"
    real, pm._cache_path = pm._cache_path, (lambda season: tmp)
    try:
        row = pm.build_pitch_mix(2099, force=True, statcast=sc).iloc[0]
    finally:
        pm._cache_path = real
    if abs(row["swstr_pct"] - 0.5) > 1e-9 or abs(row["whiff_per_swing"] - 1.0) > 1e-9:
        errors.append(f"150 foul tips + 150 balls: swstr {row['swstr_pct']}, "
                      f"whiff/swing {row['whiff_per_swing']} (want 0.5, 1.0)")

    # 2. adjacency
    three = pd.DataFrame({
        "pitcher": 1, "game_pk": 1, "at_bat_number": 1, "pitch_number": [1, 2, 3],
        "pitch_type": ["FF", "SL", "CH"], "tunnel_x": [0.0, None, 0.1],
        "tunnel_z": [5.0, None, 5.1], "plate_x": [0.0, None, 0.5],
        "plate_z": [2.5, None, 2.0],
    })
    if len(build_pitch_pairs(three)):
        errors.append("tunnel pair made across an untracked pitch")

    # 3. sequencing
    def _seq(seconds):
        rows = []
        for ab, second in enumerate(seconds):
            rows += [(1, ab, 1, "FF", 0, 0), (1, ab, 2, second, 0, 1)]
        return pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number",
                                           "pitch_type", "balls", "strikes"])
    m = _pitcher_sequencing(_seq(["FF", "SL"] * 100))
    if m is None or abs(m["entropy_ratio"] - 1.0) > 1e-9:
        errors.append(f"FF-then-FF/SL ratio {m and m['entropy_ratio']} (want 1.0)")
    rng = np.random.default_rng(23)
    seq = pd.DataFrame({"game_pk": 1, "at_bat_number": np.repeat(np.arange(200), 4),
                        "pitch_number": np.tile(np.arange(1, 5), 200),
                        "pitch_type": rng.choice(["FF", "SL", "CH", "CU"], 800,
                                                 p=[.5, .25, .15, .1]),
                        "balls": 0, "strikes": 0})
    r = _pitcher_sequencing(seq)
    if r is None or r["entropy_ratio"] > 1 + 1e-9:
        errors.append(f"random-sequence ratio {r and r['entropy_ratio']} above 1")
    gap = _seq(["SL"] * 200)
    gap.loc[gap["pitch_number"] == 2, "pitch_number"] = 3      # pitch 2 untyped
    if _pitcher_sequencing(gap) is not None:
        errors.append("sequencing paired pitches 1 and 3")

    passed = not errors
    return passed, ("foul tip = whiff; pairs need adjacent pitches; ratio <= 1"
                    if passed else "; ".join(errors))


# ---------------------------------------------------------------------------
# V24 — Steal-attempt ranks league-wide
# ---------------------------------------------------------------------------

def v24_league_steal_ranks() -> Result:
    """
    1. synthetic league: a hitter with 0 attempts and plenty of times on base
       ranks in the bottom 15% (low steal attempts); a hitter under
       STEAL_RANK_MIN_ON_BASE times on base gets no rank; the busiest runner
       ranks top
    2. saved portraits: each player-season carries ONE attempt rank on every
       team it appears for — it was ranked within each roster, so 27 traded
       2025 hitters had a different rank on each club
    """
    import json
    import pandas as pd
    import team_portrait as tp
    from ingest import PROCESSED_DIR

    errors = []
    n = 100
    league = pd.DataFrame({
        "key_mlbam": range(1, n + 1),
        "pa": [500] * n, "obp": [0.320] * n,                  # 160 times on base
        "sb": [0] * 20 + list(range(1, n - 19)), "cs": [0] * n,
    })
    league.loc[league.key_mlbam == 100, ["pa", "obp"]] = [60, 0.300]   # 18 on base
    pcts = tp._league_attempt_pcts(league)
    if not pcts.get(1, 100) <= 15:
        errors.append(f"0 attempts on 160 times on base ranks {pcts.get(1)}, not bottom 15%")
    if 100 in pcts.index:
        errors.append("hitter under the times-on-base floor was ranked")
    if pcts.idxmax() != 99:
        errors.append(f"busiest runner is not the top rank ({pcts.idxmax()})")

    split, checked = [], 0
    for season in range(2015, 2027):
        ranks: dict = {}
        for f in sorted((PROCESSED_DIR / "portraits").glob(f"*_{season}.json")):
            for h in json.loads(f.read_text())["players"].get("hitters", []):
                pct = (h.get("gate_inputs") or {}).get("attempt_rate_pct")
                ranks.setdefault(h["player_id"], set()).add(
                    None if pct is None else round(float(pct), 6))
        split += [f"{season} {pid}" for pid, v in ranks.items() if len(v) > 1]
        checked += bool(ranks)
    if split:
        errors.append(f"{len(split)} player-season(s) ranked differently on different "
                      f"teams (e.g. {', '.join(split[:3])})")

    passed = not errors
    return passed, ((f"league-wide ranks; one per player-season across {checked} seasons"
                     if checked else "synthetic league only: no portraits on this machine")
                    if passed else "; ".join(errors))


# ---------------------------------------------------------------------------
# V25 — Tag states and framing
# ---------------------------------------------------------------------------

def v25_tag_states_and_framing() -> Result:
    """
    Offline, synthetic:
      1. every tag in the vocabulary belongs to exactly one tag family on each
         side (out-pitch tags by suffix), so no tag can fire without moving a
         family to `present`
      2. a catcher with edge pitches but no framing percentile reads
         `insufficient_sample` for framing, a non-catcher `not_applicable`, a
         player who never ran `not_applicable` for steal success, and a
         reliever `not_applicable` for starter workload
      3. framing credits the catcher, not the pitch: on synthetic pitches
         called by one rule except that one catcher gets every edge pitch
         called a strike, he ranks first and the others are not all above
         expected
    """
    import numpy as np
    import pandas as pd
    import team_portrait as tp
    import pitch_aggregates as pa

    errors = []
    h_fam = [t for tags in tp.HITTER_TAG_FAMILIES.values() for t in tags]
    p_fam = [t for tags in tp.PITCHER_TAG_FAMILIES.values() for t in tags]
    for side, fam in (("hitter", h_fam), ("pitcher", p_fam)):
        dup = sorted({t for t in fam if fam.count(t) > 1})
        if dup:
            errors.append(f"{side} tags in two families: {dup}")
    covered = set(h_fam) | set(p_fam)
    loose = sorted(t for t in tp.TAG_KIND if t not in covered and not t.endswith(" out pitch"))
    if loose:
        errors.append(f"tags in no family: {loose}")
    if covered - set(tp.TAG_KIND):
        errors.append(f"family tags not in the vocabulary: {sorted(covered - set(tp.TAG_KIND))}")

    st = tp.hitter_tag_states([], {"pa": 400, "sb": 0, "cs": 0,
                                   "catching": {"framing_taken": 120},
                                   "positions": {"pos_detail": {"C": 1.0}}})
    if st.get("framing") != "insufficient_sample":
        errors.append(f"unranked catcher: framing = {st.get('framing')}")
    if st.get("steal success") != "not_applicable":
        errors.append(f"never ran: steal success = {st.get('steal success')}")
    st = tp.hitter_tag_states([], {"pa": 400, "positions": {"pos_detail": {"SS": 1.0}}})
    if st.get("framing") != "not_applicable":
        errors.append(f"shortstop: framing = {st.get('framing')}")
    st = tp.pitcher_tag_states([], {"bf": 250, "pool_rows": {}}, False, None)
    if st.get("starter workload") != "not_applicable":
        errors.append(f"reliever: starter workload = {st.get('starter workload')}")

    rng = np.random.default_rng(7)
    n = 60000
    df = pd.DataFrame({
        "fielder_2": rng.integers(1, 9, n),
        "plate_x": rng.uniform(-1.6, 1.6, n),
        "plate_z": rng.uniform(1.0, 4.0, n),
        "sz_top": 3.4, "sz_bot": 1.6,
        "stand": rng.choice(["L", "R"], n),
        "balls": rng.integers(0, 4, n), "strikes": rng.integers(0, 3, n),
    })
    zn = (df["plate_z"] - 1.6) / 1.8
    inside = (df["plate_x"].abs() <= pa.ZONE_HALF_WIDTH) & (zn >= 0) & (zn <= 1)
    cs = rng.random(n) < np.where(inside, 0.85, 0.08)
    edge = (((df["plate_x"].abs() - pa.ZONE_HALF_WIDTH).abs() <= 0.5)
            | ((zn - 1).abs() <= 0.2) | (zn.abs() <= 0.2))
    cs = np.where((df["fielder_2"] == 1) & edge, True, cs)
    df["description"] = np.where(cs, "called_strike", "ball")
    saved = dict(pa._FRAMING_CACHE)
    try:
        pa._FRAMING_CACHE.pop(-1, None)
        fr = pa.compute_catcher_framing(df, -1, min_taken=100).set_index("catcher")
    finally:
        pa._FRAMING_CACHE.clear()
        pa._FRAMING_CACHE.update(saved)
    if fr["framing_above"].idxmax() != 1:
        errors.append(f"strike-getter ranked {fr['framing_pct'].get(1)}th pct")
    rest = fr.drop(index=1)["framing_above"]
    if (rest > 0).all():
        errors.append(f"every other catcher above expected: {rest.round(1).to_dict()}")

    passed = not errors
    detail = (f"{len(h_fam)} hitter + {len(p_fam)} pitcher tags in families; "
              "unranked catcher / non-catcher / never ran / reliever states; "
              f"synthetic framer {fr.loc[1, 'framing_above']:+.1f} per 100, "
              f"others {rest.min():+.1f}..{rest.max():+.1f}"
              if passed else "; ".join(errors))
    return passed, detail


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("V1",  "Temporal mode determination",     v1_temporal_modes),
    ("V2",  "Reliability weighting",           v2_reliability_weighting),
    ("V3",  "Historical smoothing",            v3_historical_smoothing),
    ("V4",  "Philosophy scoring",              v4_philosophy_scoring),
    ("V5",  "Hitter archetypes",               v5_hitter_archetypes),
    ("V6",  "Pitcher archetypes",              v6_pitcher_archetypes),
    ("V7",  "End-to-end portrait",             v7_end_to_end_portrait),
    ("V8",  "Pitch mix aggregation schema",    v8_pitch_mix_schema),
    ("V9",  "Arsenal profile classification",  v9_arsenal_profile),
    ("V10", "Service time estimation",         v10_service_time),
    ("V11", "Spin efficiency pipeline",        v11_spin_efficiency),
    ("V12", "Park factors",                    v12_park_factors),
    ("V13", "Spin efficiency vs Savant",       v13_spin_vs_savant),
    ("V14", "Metric rank directions",          v14_metric_directions),
    ("V15", "Batted-ball direction",           v15_spray_direction),
    ("V16", "Tag evidence coverage",           v16_tag_evidence),
    ("V17", "Season completeness",             v17_season_completeness),
    ("V18", "Regular season only",             v18_regular_season_only),
    ("V19", "Team stints",                     v19_team_stints),
    ("V20", "League caches current",           v20_league_caches_current),
    ("V21", "Sprint speed units",              v21_sprint_speed_units),
    ("V22", "Complete reads and rosters",      v22_complete_rosters),
    ("V23", "Pitch outcomes and pairing",      v23_pitch_outcomes_pairing),
    ("V24", "Steal-attempt ranks league-wide", v24_league_steal_ranks),
    ("V25", "Tag states and framing",          v25_tag_states_and_framing),
]


def run_all(verbose: bool = True, offline: bool = False) -> bool:
    """
    Run every validation test and print a summary. With offline, a test in
    NEEDS is reported SKIP — not run, not counted — so the pull-request
    checks pass or fail on what a bare runner can actually test.
    Returns True if no test that ran failed.
    """
    results = []
    for code, name, fn in TESTS:
        if offline and code in NEEDS:
            print(f"SKIP  {code}  {name}")
            if verbose:
                print(f"      needs {NEEDS[code]}")
            continue
        try:
            passed, detail = fn()
        except Exception as exc:
            passed = False
            detail = f"EXCEPTION: {exc}"
        results.append((code, name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"{status}  {code}  {name}")
        if verbose or not passed:
            print(f"      {detail}")

    all_pass = all(r[2] for r in results)
    n_pass = sum(1 for r in results if r[2])
    skipped = len(TESTS) - len(results)
    print()
    print(f"{'ALL PASS' if all_pass else 'FAILURES PRESENT'}  {n_pass}/{len(results)} tests passed"
          + (f"  ({skipped} skipped offline)" if skipped else ""))
    return all_pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the validation suite.")
    ap.add_argument("--offline", action="store_true",
                    help="skip checks that need the database or full-season data "
                         "(the pull-request checks)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    ok = run_all(verbose=True, offline=args.offline)
    sys.exit(0 if ok else 1)
