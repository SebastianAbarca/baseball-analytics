"""
validation.py — 7 validation tests against known outputs.

Each test is self-contained and returns (passed: bool, detail: str).
run_all() executes every test and prints a summary.

Tests:
  V1  Temporal mode determination — known PA/games → expected modes
  V2  Reliability weighting — known sample/threshold → expected weight
  V3  Historical smoothing — 3-year weighted avg matches hand calculation
  V4  Philosophy scoring — known metric dict → expected score and primary
  V5  Hitter archetype — Complete Hitter and TTO routing
  V6  Pitcher archetype — P1 Power Ace and P3 GB Craftsman routing
  V7  End-to-end portrait — builds without error on cached test data
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
    """
    from hitter_archetypes import (
        classify_complete, classify_tto, classify_spectrum,
        classify_primary, build_hitter_profile,
        COMPLETE_THRESHOLDS, TTO_THRESHOLDS,
    )

    errors = []

    # ── Complete Hitter: all 5 thresholds comfortably met ─────────────────
    complete_metrics = {
        "wRC_plus":     85.0,   # threshold 75
        "OBP":          80.0,   # threshold 70
        "ISO":          75.0,   # threshold 70
        "BB_pct":       70.0,   # threshold 65
        "ZContact_pct": 65.0,   # threshold 55
    }
    result = classify_complete(complete_metrics)
    if result is None:
        errors.append("Complete Hitter: expected match, got None")
    elif result["type"] != "Complete Hitter":
        errors.append(f"Complete Hitter: got type={result['type']}")

    # ── Complete Hitter fails if any threshold missed ──────────────────────
    incomplete = {**complete_metrics, "ZContact_pct": 40.0}   # below 55
    result = classify_complete(incomplete)
    if result is not None:
        errors.append("Complete Hitter with ZContact=40 should fail, but passed")

    # ── TTO: thresholds met, display_confidence must be >= 30 ────────────
    tto_metrics = {
        "K_pct":   75.0,   # threshold 50
        "BB_pct":  80.0,   # threshold 65
        "ISO":     80.0,   # threshold 65
    }
    result = classify_tto(tto_metrics)
    if result is None:
        errors.append("TTO: expected match, got None")
    elif result["type"] != "Three True Outcomes":
        errors.append(f"TTO type: expected 'Three True Outcomes', got {result['type']}")

    # ── TTO barely passing → low display_confidence → routes to spectrum ──
    # margins: K=0.25, BB=0.15, ISO=0.15; raw_conf = mean = 0.1833
    # display_confidence = min(int(0.1833/0.35 * 100), 100) = 52 → >= 30, stays TTO
    # To force < 30: need raw_conf < 0.105 → margins avg < 0.105
    # Use K=51 (margin=1/100=0.01), BB=66 (0.01), ISO=66 (0.01) → raw=0.01/3 ≈ 0.01 → disp=2 → routes spectrum
    barely_tto = {"K_pct": 51.0, "BB_pct": 66.0, "ISO": 66.0}
    result = classify_tto(barely_tto)
    if result is not None:
        errors.append(
            f"Low-confidence TTO should route to spectrum (return None), but got type={result['type']}"
        )

    # ── Spectrum: pure power profile ──────────────────────────────────────
    power_metrics = {
        "ISO_pct":       90.0,
        "HR_FB_pct":     85.0,
        "Barrel_pct":    88.0,
        "EV_90_pct":     80.0,
        "Contact_pct":   20.0,   # low contact
        "K_pct":         80.0,   # high K
        "AVG_pct":       25.0,
        "OBP_ISO_gap_pct":30.0,
    }
    result = classify_spectrum(power_metrics)
    if result["type"] != "Pure Power":
        errors.append(f"Spectrum power profile: expected 'Pure Power', got {result['type']}")
    if result["spectrum_score"] is None or result["spectrum_score"] < 60:
        errors.append(f"Spectrum score expected > 60, got {result.get('spectrum_score')}")

    # ── classify_primary respects ordering (Complete > TTO > Spectrum) ────
    # Profile meets Complete Hitter AND would be TTO — should return Complete
    overlapping = {
        "wRC_plus":     85.0, "OBP": 80.0, "ISO": 75.0,
        "BB_pct":       70.0, "ZContact_pct": 65.0,
        "K_pct":        75.0,
    }
    result = classify_primary(overlapping)
    if result["type"] != "Complete Hitter":
        errors.append(
            f"Primary ordering: Complete should win over TTO, got {result['type']}"
        )

    # ── build_hitter_profile returns expected structure ───────────────────
    profile = build_hitter_profile(12345, complete_metrics)
    for key in ("player_id", "primary", "modifiers"):
        if key not in profile:
            errors.append(f"build_hitter_profile missing key: {key}")
    for mod in ("free_swinger", "speed", "disruptiveness"):
        if mod not in profile.get("modifiers", {}):
            errors.append(f"modifiers missing key: {mod}")

    passed = len(errors) == 0
    detail = "all cases correct" if passed else "; ".join(errors)
    return passed, detail


# ---------------------------------------------------------------------------
# V6 — Pitcher archetypes
# ---------------------------------------------------------------------------

def v6_pitcher_archetypes() -> Result:
    """
    Test P1 Power Ace qualification, P3 GB Craftsman ceiling enforcement,
    and classify_starter ordering.
    """
    from pitcher_archetypes import (
        classify_p1, classify_p2, classify_p3, classify_p4,
        classify_starter, build_starter_profile,
        P1_MIN, P2_VELO_CEILING, P3_HARD_CEILING, P3_K_CEILING,
    )

    errors = []

    # ── P1 Power Ace: all minimums met ────────────────────────────────────
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

    # ── P1 fails if one minimum not met ────────────────────────────────────
    p1_fail = {**p1_metrics, "K_pct_pct": 60.0}   # below 65
    result = classify_p1(p1_fail)
    if result is not None:
        errors.append("P1 with K_pct_pct=60 should fail, but passed")

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

    # ── P3 blocked if K% or HardHit above ceilings ─────────────────────
    p3_too_k = {
        "GB_pct_pct":          65.0,   # >= 50
        "K_pct_pct":           65.0,   # above P3_K_CEILING (60) → blocked
        "HardHit_allowed_pct": 40.0,   # below hard ceiling
    }
    result = classify_p3(p3_too_k)
    if result is not None:
        errors.append("P3 with K_pct > ceiling should fail, but passed")

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

        # Check players sub-keys
        players = portrait.get("players", {})
        for pk in ("hitters", "starters", "bullpen_profile"):
            if pk not in players:
                errors.append(f"{team}: players missing {pk}")

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
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("V1", "Temporal mode determination",     v1_temporal_modes),
    ("V2", "Reliability weighting",           v2_reliability_weighting),
    ("V3", "Historical smoothing",            v3_historical_smoothing),
    ("V4", "Philosophy scoring",              v4_philosophy_scoring),
    ("V5", "Hitter archetypes",               v5_hitter_archetypes),
    ("V6", "Pitcher archetypes",              v6_pitcher_archetypes),
    ("V7", "End-to-end portrait",             v7_end_to_end_portrait),
]


def run_all(verbose: bool = True) -> bool:
    """
    Run all 7 validation tests and print a summary.
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
