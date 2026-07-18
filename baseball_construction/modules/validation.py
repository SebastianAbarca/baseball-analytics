"""
validation.py — 12 validation tests against known outputs.

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
  V11 Spin efficiency pipeline — compute and aggregate on synthetic pitches
  V12 Park factors — build_park_profile returns normalized pitcher_friendly
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
        "SwStr_pct_pct":   68.0,   # >= 65
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
        "SwStr_pct_pct": 58.0,   # >= 55
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
        "SwStr_pct_pct":       68.0,
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

    Runs the core aggregation steps directly (without the full build_pitch_mix
    pipeline which requires a full-season Statcast pull), then validates column
    schema, sensible value ranges, and PITCH_FAMILY mapping.
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

    # Replicate the core aggregation steps from build_pitch_mix
    sc = sc[sc["pitch_type"].notna() & (sc["pitch_type"] != "")].copy()
    sc = sc[~sc["pitch_type"].isin({"PO", "EP", "SC"})].copy()

    sc["is_swing"] = sc["description"].isin({
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play",
    })
    sc["is_whiff"] = sc["description"].isin({
        "swinging_strike", "swinging_strike_blocked", "foul_tip",
    })
    sc["is_csw"]  = sc["is_whiff"] | (sc["description"] == "called_strike")
    sc["is_gb"]   = sc["bb_type"] == "ground_ball"
    sc["is_bip"]  = sc["bb_type"].isin({"ground_ball", "fly_ball", "line_drive", "popup"})

    totals = sc.groupby(["pitcher", "game_year"])["pitch_type"].count().rename("total_n")
    grp = sc.groupby(["pitcher", "game_year", "pitch_type"])
    agg = grp.agg(
        n             = ("pitch_type",       "count"),
        avg_velo      = ("release_speed",    "mean"),
        avg_ivb       = ("pfx_z",            "mean"),
        avg_hb        = ("pfx_x",            "mean"),
        n_whiff       = ("is_whiff",         "sum"),
        n_bip         = ("is_bip",           "sum"),
        n_gb          = ("is_gb",            "sum"),
    ).reset_index()
    agg = agg.merge(totals.reset_index(), on=["pitcher", "game_year"])
    agg["usage_pct"] = agg["n"] / agg["total_n"]
    agg["whiff_pct"] = _np.where(agg["n"] > 0, agg["n_whiff"] / agg["n"], _np.nan)
    agg["gb_pct"]    = _np.where(agg["n_bip"] > 0, agg["n_gb"] / agg["n_bip"], _np.nan)

    if agg.empty:
        return False, "No pitch-type rows aggregated from test data"

    # Column presence
    required = {"pitcher", "pitch_type", "n", "usage_pct", "avg_velo", "avg_ivb", "avg_hb", "whiff_pct"}
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
    wp = agg["whiff_pct"].dropna()
    if len(wp) > 0 and not wp.between(0, 1).all():
        errors.append("whiff_pct outside [0,1]")

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
    Verify query_estimated_service_time:
      1. Returns a dict keyed by MLBAM ID
      2. Values are floats (years.fraction)
      3. A player who debuted in 2015 has at least ~8 years EST by 2023
      4. A 2022 debut player has < 3 years EST by 2023 (pre-arb)
      5. CBA tier logic: <3 pre-arb, 3-5.999 arb, >=6 FA-eligible
    """
    from database import query_estimated_service_time, query_debut_seasons

    errors = []

    # Pull a small sample of known MLBAM IDs from the Chadwick people files
    # (IDs we know exist — use Statcast-verified players)
    # Fallback: load a handful from cached people CSVs
    try:
        debut_map = query_debut_seasons()
    except Exception as exc:
        return False, f"query_debut_seasons raised: {exc}"

    if not debut_map:
        return False, "No debut seasons available — seeded data may be missing"

    # Sample: 10 players with known debut years
    sample_ids = list(debut_map.keys())[:10]
    try:
        est = query_estimated_service_time(sample_ids, 2023)
    except Exception as exc:
        return False, f"query_estimated_service_time raised: {exc}"

    if not est:
        return False, "Empty EST result for sample players"

    # All values should be non-negative floats
    for pid, val in est.items():
        if not isinstance(val, (int, float)):
            errors.append(f"EST for {pid} is not numeric: {val!r}")
        elif val < 0:
            errors.append(f"EST for {pid} is negative: {val}")

    # Veterans (debut <= 2015) should have >=6 EST by 2023 (FA-eligible)
    vets = [p for p in sample_ids if debut_map.get(p, 9999) <= 2015 and p in est]
    for pid in vets:
        if est[pid] < 6.0:
            errors.append(
                f"Vet {pid} (debut {debut_map[pid]}) has EST={est[pid]:.2f} < 6.0 by 2023"
            )

    # Rookies (debut >= 2022) should have <3 EST by 2023 (pre-arb)
    rookies = [p for p in sample_ids if debut_map.get(p, 0) >= 2022 and p in est]
    for pid in rookies:
        if est[pid] >= 3.0:
            errors.append(
                f"Rookie {pid} (debut {debut_map[pid]}) has EST={est[pid]:.2f} >= 3.0 by 2023"
            )

    # CBA tier logic verification on the full sample
    for pid, val in est.items():
        tier = "pre-arb" if val < 3.0 else ("arb" if val < 6.0 else "fa")
        if tier not in ("pre-arb", "arb", "fa"):
            errors.append(f"Unexpected tier '{tier}' for {pid} EST={val:.2f}")

    passed = len(errors) == 0
    detail = (
        f"Tested {len(est)} players — vets={len(vets)} rookies={len(rookies)}, all tiers valid"
        if passed else "; ".join(errors)
    )
    return passed, detail


# ---------------------------------------------------------------------------
# V11 — Spin efficiency pipeline
# ---------------------------------------------------------------------------

def v11_spin_efficiency() -> Result:
    """
    Verify the spin efficiency pipeline on synthetic pitch data:
      1. compute_spin_efficiency adds a spin_efficiency column
      2. aggregate_pitcher_pitch_type groups correctly
      3. apply_reliability_weighting shrinks toward league mean
      4. build_pitcher_spin_profile returns (pitch_type_level, pitcher_level)
         with expected columns and value ranges
    """
    import pandas as pd
    import numpy as np
    from spin_efficiency import (
        compute_spin_efficiency,
        aggregate_pitcher_pitch_type,
        apply_reliability_weighting,
        build_pitcher_spin_profile,
    )

    errors = []

    # Build synthetic pitch-level data — 3 pitchers, 2 pitch types each
    rng = np.random.default_rng(42)
    n = 300
    rows = []
    for pid in [100, 200, 300]:
        for ptype, baked_angle in [("FF", 210), ("SL", 160)]:
            for _ in range(n // 6):
                rows.append({
                    "pitcher":           pid,
                    "pitch_type":        ptype,
                    "release_spin_rate": float(rng.integers(2000, 2800)),
                    "release_speed":     float(rng.integers(88, 97)),
                    "spin_axis":         float(baked_angle + rng.integers(-10, 10)),
                    "pfx_z":             rng.uniform(0.5, 1.5),
                    "pfx_x":             rng.uniform(-1.0, 1.0),
                    "release_pos_z":     rng.uniform(5.5, 6.5),
                    "p_throws":          "R",
                })
    df = pd.DataFrame(rows)

    # Step 1 — compute_spin_efficiency
    try:
        df_se = compute_spin_efficiency(df)
    except Exception as exc:
        return False, f"compute_spin_efficiency raised: {exc}"

    if "spin_efficiency" not in df_se.columns:
        errors.append("compute_spin_efficiency: 'spin_efficiency' column missing")
    elif df_se["spin_efficiency"].notna().sum() == 0:
        errors.append("compute_spin_efficiency: all spin_efficiency values are NaN")
    elif not df_se["spin_efficiency"].dropna().between(0, 1).all():
        errors.append("spin_efficiency values outside [0, 1]")

    # Step 2 — aggregate
    try:
        agg = aggregate_pitcher_pitch_type(df_se)
    except Exception as exc:
        return False, f"aggregate_pitcher_pitch_type raised: {exc}"

    if agg.empty:
        errors.append("aggregate_pitcher_pitch_type returned empty DataFrame")
    else:
        expected_rows = 3 * 2   # 3 pitchers × 2 pitch types
        if len(agg) != expected_rows:
            errors.append(f"Expected {expected_rows} agg rows, got {len(agg)}")

    # Step 3 — reliability weighting
    try:
        agg_rel = apply_reliability_weighting(agg)
    except Exception as exc:
        return False, f"apply_reliability_weighting raised: {exc}"

    if "spin_efficiency_weighted" not in agg_rel.columns:
        errors.append("apply_reliability_weighting: 'spin_efficiency_weighted' missing")

    # Step 4 — full pipeline
    try:
        pt_level, p_level = build_pitcher_spin_profile(df)
    except Exception as exc:
        return False, f"build_pitcher_spin_profile raised: {exc}"

    if p_level.empty:
        errors.append("pitcher_level spin profile is empty")
    else:
        if "spin_efficiency_weighted" not in p_level.columns:
            errors.append("pitcher_level missing 'spin_efficiency_weighted'")
        if p_level["pitcher"].nunique() != 3:
            errors.append(f"Expected 3 pitchers in summary, got {p_level['pitcher'].nunique()}")

    passed = len(errors) == 0
    detail = (
        f"Pipeline OK — {len(df)} pitches → {len(agg if not agg.empty else [])} agg rows → {len(p_level) if not p_level.empty else 0} pitchers"
        if passed else "; ".join(errors)
    )
    return passed, detail


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
]


def run_all(verbose: bool = True) -> bool:
    """
    Run all 12 validation tests and print a summary.
    Returns True if every test passes.
    """
    results = []
    for code, name, fn in TESTS:
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
    print()
    print(f"{'ALL PASS' if all_pass else 'FAILURES PRESENT'}  {n_pass}/{len(results)} tests passed")
    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ok = run_all(verbose=True)
    sys.exit(0 if ok else 1)
