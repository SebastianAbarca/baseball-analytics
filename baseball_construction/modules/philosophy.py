"""
philosophy.py — All 11 team philosophy scores across 3 dimensions.

Inputs:  a flat dict of pre-normalized metrics (each 0–100 percentile).
         Missing metrics are handled by reweighting to available coverage.
Outputs: scores (0–100), confidences (0–100), primary philosophy per dimension.

DIMENSION A — Offensive Philosophy  (A1–A4)
DIMENSION B — Pitching Philosophy   (B1–B4)
DIMENSION C — Roster Construction   (C1–C4)

All metric keys ending in _pct are assumed already normalized 0–100.
Inversion is handled upstream by normalize_percentile(invert=True).
The only exception: a few keys carry an explicit _inv suffix where the
caller must supply the inverted percentile (see PHILOSOPHY_DEFS).

Confidence:
  raw        = (top_score - second_score) / 100
  display    = min(int((raw / 0.35) × 100), 100)
  hybrid     = True if top AND second both score > 60
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Philosophy definitions
# Each value is {metric_key: weight}.
# Weights sum to 1.0.
# Metric keys must be pre-normalised 0–100 by the caller.
# ---------------------------------------------------------------------------

PHILOSOPHY_DEFS: dict[str, dict] = {

    # ── DIMENSION A — OFFENSIVE PHILOSOPHY ──────────────────────────────

    "A1": {
        "name":      "Three True Outcomes",
        "dimension": "offense",
        "weights": {
            "ISO_pct":        0.30,
            "BB_pct_pct":     0.25,
            "K_pct_pct":      0.20,   # higher K% = more TTO (not inverted)
            "HR_FB_pct":      0.15,
            "Sprint_inv_pct": 0.10,   # low speed confirms TTO (caller inverts)
        },
    },

    "A2": {
        "name":      "Contact and Pressure",
        "dimension": "offense",
        "weights": {
            "K_inv_pct":          0.20,   # low K% (caller inverts)
            "OBP_SLG_gap_pct":    0.15,   # OBP outperforms SLG
            "PitchesPerPA_pct":   0.15,
            "SprintSpeed_pct":    0.15,
            "Contact_pct_pct":    0.15,
            "TeamXB_pct":         0.10,   # doubles+triples rate (gap contact pressure)
            "TeamGap_pct":        0.10,   # fraction of BIP in gap zones
        },
    },

    "A3": {
        "name":      "Aggressive Early Count",
        "dimension": "offense",
        "weights": {
            "FPS_pct_pct":         0.35,   # first-pitch swing %
            "PitchesPerPA_inv_pct":0.25,   # low pitches/PA (caller inverts)
            "ZSwing_pct_pct":      0.20,
            "BB_inv_pct":          0.20,   # low BB% (caller inverts)
        },
    },

    "A4": {
        "name":      "Lineup Power",
        "dimension": "offense",
        "weights": {
            "TeamHR_pct":            0.30,  # team HR total rank vs all 30 teams
            "TeamSLG_pct":           0.25,  # PA-weighted team SLG rank
            "PowerContributors_pct": 0.25,  # fraction of PA from above-median-ISO batters
            "TeamBarrel_pct":        0.20,  # PA-weighted team barrel rate rank
        },
    },

    # ── DIMENSION B — PITCHING PHILOSOPHY ───────────────────────────────

    "B1": {
        "name":      "Stuff Dominant",
        "dimension": "pitching",
        "weights": {
            "TeamK_pct_pct":    0.30,
            "AvgVelo_pct":      0.25,
            "SwStr_pct_pct":    0.25,
            "AvgSpinRate_pct":  0.20,
        },
    },

    "B2": {
        "name":      "Command and Contact Management",
        "dimension": "pitching",
        "weights": {
            "BB_pitch_inv_pct":  0.30,   # low BB% (caller inverts)
            "Zone_pct_pct":      0.25,
            "GB_pct_pct":        0.25,
            "TeamDefense_pct":   0.20,   # OAA or DRS
        },
    },

    "B3": {
        "name":      "Pitch Design and Analytics Driven",
        "dimension": "pitching",
        "weights": {
            "SpinEfficiency_pct":      0.25,
            "ArsenalDiversity_pct":    0.20,
            "PlatoonOptimization_pct": 0.20,
            "OpenerUsage_pct":         0.15,   # binary scaled 0/100
            "CSW_pct_pct":             0.20,
        },
    },

    "B4": {
        "name":      "Defensive Infrastructure",
        "dimension": "pitching",
        "weights": {
            "OAA_pct":                0.35,
            "DRS_pct":                0.25,
            "GB_pitch_pct":           0.25,
            "ParkPitcherFriendly_pct":0.15,
        },
    },

    # ── DIMENSION C — ROSTER CONSTRUCTION ───────────────────────────────

    "C1": {
        "name":      "bWAR Distribution",
        "dimension": "roster",
        "weights": {
            "WAR_concentration_pct": 0.35,   # top-3 share of total bWAR
            "WAR_variance_inv_pct":  0.35,   # spread of bWAR across roster (inverted)
            "RosterFloor_pct":       0.30,   # fraction of players with positive bWAR
        },
    },

    "C2": {
        "name":      "Roster Continuity",
        "dimension": "roster",
        "weights": {
            "CoreRetention_pct": 1.0,   # fraction of prior-year players retained
        },
    },

    "C3": {
        "name":      "Youth and Development",
        "dimension": "roster",
        "weights": {
            "AvgTenure_inv_pct":  0.50,  # PA-weighted avg MLB tenure, inverted (lower = more developmental)
            "NewPlayerShare_pct": 0.50,  # share of players with <= 2 years in MLB
        },
    },

    "C4": {
        "name":      "Veteran Experience",
        "dimension": "roster",
        "weights": {
            "AvgTenure_pct":   0.50,  # PA-weighted avg MLB tenure (higher = more experienced)
            "VeteranShare_pct":0.50,  # share of players with >= 5 years in MLB
        },
    },
}

# Dimension groupings for confidence / primary philosophy logic
DIMENSIONS: dict[str, list[str]] = {
    "offense": ["A1", "A2", "A3", "A4"],
    "pitching": ["B1", "B2", "B3", "B4"],
    "roster":   ["C1", "C2", "C3", "C4"],
}


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_philosophy(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> tuple[Optional[float], float]:
    """
    Weighted mean of available metrics.

    Args:
        metrics: {metric_key: value_0_to_100}  (NaN / None = missing)
        weights: {metric_key: weight}

    Returns:
        (score, coverage)
        score    — weighted mean 0–100, or None if no metrics available
        coverage — fraction of weight that had data (1.0 = all metrics present)
    """
    total_weight = sum(weights.values())
    available_weight = 0.0
    weighted_sum = 0.0

    for metric, weight in weights.items():
        val = metrics.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        weighted_sum    += float(val) * weight
        available_weight += weight

    if available_weight == 0.0:
        return None, 0.0

    # Reweight to available coverage so scores stay 0–100
    score    = weighted_sum / available_weight
    coverage = available_weight / total_weight
    return score, coverage


def dimension_label(score: float | None) -> str:
    """
    Convert a 0–100 dimension score into a descriptive identity label.

    Thresholds are calibrated to the multi-year historical pool so that
    "Strong" means genuinely above the Statcast-era league average.
    """
    if score is None:  return "No Data"
    if score >= 85:    return "Defining"
    if score >= 70:    return "Strong"
    if score >= 55:    return "Notable"
    if score >= 35:    return "Moderate"
    return "Low"


def compute_all_philosophies(
    metrics: dict[str, float],
) -> dict[str, dict]:
    """
    Score all 11 philosophies for one team.

    Args:
        metrics: flat dict of pre-normalized metric values (0–100 each).
                 Missing keys / NaN values are handled gracefully.

    Returns dict keyed by philosophy code (A1…C4):
        score       — 0–100 (None if no data)
        coverage    — 0.0–1.0 (fraction of weights populated)
        name        — philosophy label
        label       — Defining | Strong | Notable | Moderate | Low | No Data
        dimension   — offense | pitching | roster
    """
    results = {}
    for code, defn in PHILOSOPHY_DEFS.items():
        score, coverage = score_philosophy(metrics, defn["weights"])
        results[code] = {
            "score":     score,
            "coverage":  coverage,
            "name":      defn["name"],
            "label":     dimension_label(score),
            "dimension": defn["dimension"],
        }
        if score is not None:
            log.debug(
                "%s (%s): score=%.1f  coverage=%.0f%%  label=%s",
                code, defn["name"], score, coverage * 100, results[code]["label"],
            )
        else:
            log.debug("%s (%s): no data", code, defn["name"])
    return results


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

CONFIDENCE_CEILING = 0.35


def display_confidence(raw: float, ceiling: float = CONFIDENCE_CEILING) -> int:
    """
    Convert raw confidence (top - second) / 100 to a 0–100 display value.

    Raw of 0.35 or above → 100%. Rescaled linearly below that.
    """
    return min(int((raw / ceiling) * 100), 100)


def compute_dimension_confidence(
    scores: dict[str, dict],
    dimension: str,
) -> dict:
    """
    For one dimension, find:
      - primary philosophy (highest score)
      - secondary philosophy (second highest)
      - raw confidence gap
      - display confidence (0–100)
      - hybrid flag (both top and second > 60)

    Returns dict with keys:
      primary, secondary, top_score, second_score,
      raw_confidence, display_confidence, hybrid
    """
    codes = DIMENSIONS[dimension]
    scored = [
        (code, scores[code]["score"])
        for code in codes
        if scores[code]["score"] is not None
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return {
            "primary": None, "secondary": None,
            "top_score": None, "second_score": None,
            "raw_confidence": None, "display_confidence": None,
            "hybrid": False,
        }

    top_code,  top_score   = scored[0]
    sec_code,  sec_score   = scored[1] if len(scored) > 1 else (None, None)
    raw_conf = (top_score - sec_score) / 100.0 if sec_score is not None else 1.0

    return {
        "primary":            top_code,
        "secondary":          sec_code,
        "top_score":          top_score,
        "second_score":       sec_score,
        "raw_confidence":     raw_conf,
        "display_confidence": display_confidence(raw_conf),
        "hybrid":             bool(
            top_score > 60 and sec_score is not None and sec_score > 60
        ),
    }


def compute_all_confidences(scores: dict[str, dict]) -> dict[str, dict]:
    """
    Compute confidence for all three dimensions.

    Returns dict keyed by dimension name.
    """
    return {
        dim: compute_dimension_confidence(scores, dim)
        for dim in DIMENSIONS
    }


# ---------------------------------------------------------------------------
# Team philosophy summary
# ---------------------------------------------------------------------------

def build_philosophy_summary(
    team: str,
    season: int,
    metrics: dict[str, float],
) -> dict:
    """
    Full philosophy profile for one team × season.

    Args:
        team:    team abbreviation (e.g. 'NYY')
        season:  season year
        metrics: pre-normalized metric dict (0–100 each)

    Returns:
        {
          team, season,
          scores:      {A1…C4: {score, coverage, name, dimension}},
          confidences: {offense: {...}, pitching: {...}, roster: {...}},
        }
    """
    scores      = compute_all_philosophies(metrics)
    confidences = compute_all_confidences(scores)

    return {
        "team":        team,
        "season":      season,
        "scores":      scores,
        "confidences": confidences,
    }


# ---------------------------------------------------------------------------
# Batch: DataFrame of teams → DataFrame of scores
# ---------------------------------------------------------------------------

def scores_to_dataframe(summaries: list[dict]) -> pd.DataFrame:
    """
    Convert a list of team philosophy summaries to a flat DataFrame.

    One row per team × season. Columns:
      team, season,
      A1…C4 (scores),
      A1_cov…C4_cov (coverage),
      offense_primary, offense_display_confidence, offense_hybrid,
      pitching_primary, pitching_display_confidence, pitching_hybrid,
      roster_primary,  roster_display_confidence,  roster_hybrid,
    """
    rows = []
    for s in summaries:
        row: dict = {"team": s["team"], "season": s["season"]}
        for code, info in s["scores"].items():
            row[code]            = info["score"]
            row[f"{code}_cov"]   = info["coverage"]
        for dim, conf in s["confidences"].items():
            row[f"{dim}_primary"]     = conf["primary"]
            row[f"{dim}_secondary"]   = conf["secondary"]
            row[f"{dim}_conf"]        = conf["display_confidence"]
            row[f"{dim}_hybrid"]      = conf["hybrid"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric assembly helpers
# (partial — team-level aggregation from Savant player data)
# ---------------------------------------------------------------------------

def aggregate_team_offensive_metrics(
    fg_batting: pd.DataFrame,
    statcast_batting: pd.DataFrame,
    team_col: str = "Team",
) -> pd.DataFrame:
    """
    Aggregate player-level batting metrics to team level and normalize.

    fg_batting:      output of pull_fg_batting() — one row per player
    statcast_batting: Statcast batter percentile ranks (player_id = MLBAM)
    team_col:        column name for team in fg_batting

    Returns one row per team with pre-normalized metric columns.
    Key columns produced (already 0–100 percentiles):
      K_pct_pct, K_inv_pct, BB_pct_pct, BB_inv_pct,
      ISO_pct, SprintSpeed_pct, Sprint_inv_pct,
      Contact_pct_pct (proxied via whiff_percent inverted),
      FPS_pct_pct (not available from Savant — left None),
      PitchesPerPA_pct, PitchesPerPA_inv_pct (not available — left None),
    """
    # Savant percentile ranks already have player_id = MLBAM
    bat = statcast_batting.copy()
    if "player_id" in bat.columns:
        bat = bat.rename(columns={"player_id": "key_mlbam"})

    # Need team affiliation — join from fg_batting if it has IDfg → key_mlbam
    if "key_mlbam" in fg_batting.columns and team_col in fg_batting.columns:
        team_map = fg_batting[["key_mlbam", team_col, "PA"]].copy()
        team_map["key_mlbam"] = pd.to_numeric(team_map["key_mlbam"], errors="coerce")
        bat["key_mlbam"] = pd.to_numeric(bat["key_mlbam"], errors="coerce")
        bat = bat.merge(team_map, on="key_mlbam", how="left")
    else:
        log.warning(
            "aggregate_team_offensive_metrics: team column '%s' or key_mlbam "
            "not in fg_batting — team aggregation skipped", team_col
        )
        return pd.DataFrame()

    bat = bat.dropna(subset=[team_col])
    if bat.empty:
        return pd.DataFrame()

    # PA-weight all averages
    pa = bat["PA"].fillna(1)

    def wm(col: str) -> "pd.Series":
        return bat.groupby(team_col).apply(
            lambda g: np.average(
                g[col].fillna(g[col].median()),
                weights=g["PA"].fillna(1),
            ),
            include_groups=False,
        )

    metrics_raw = pd.DataFrame(index=bat[team_col].unique())

    savant_map = {
        "k_percent":          "team_K_pct",
        "bb_percent":         "team_BB_pct",
        "hard_hit_percent":   "team_HardHit_pct",
        "sprint_speed":       "team_SprintSpeed",
        "whiff_percent":      "team_Whiff_pct",
        "xwoba":              "team_xwOBA",
        "brl_percent":        "team_Barrel_pct",
    }
    for src, dst in savant_map.items():
        if src in bat.columns:
            metrics_raw[dst] = wm(src)

    # Normalize within this team pool
    from ingest import normalize_percentile

    out = pd.DataFrame({"team": metrics_raw.index})
    out = out.reset_index(drop=True)
    metrics_raw = metrics_raw.reset_index(drop=True)
    out = pd.concat([out, metrics_raw], axis=1)

    if "team_K_pct" in out.columns:
        out["K_pct_pct"]  = normalize_percentile(out["team_K_pct"])          # higher K = higher score
        out["K_inv_pct"]  = normalize_percentile(out["team_K_pct"],  invert=True)
    if "team_BB_pct" in out.columns:
        out["BB_pct_pct"] = normalize_percentile(out["team_BB_pct"])
        out["BB_inv_pct"] = normalize_percentile(out["team_BB_pct"], invert=True)
    if "team_SprintSpeed" in out.columns:
        out["SprintSpeed_pct"] = normalize_percentile(out["team_SprintSpeed"])
        out["Sprint_inv_pct"]  = normalize_percentile(out["team_SprintSpeed"], invert=True)
    if "team_Whiff_pct" in out.columns:
        out["Contact_pct_pct"] = normalize_percentile(out["team_Whiff_pct"], invert=True)
        out["SwStr_pct_pct"]   = normalize_percentile(out["team_Whiff_pct"])
    if "team_xwOBA" in out.columns:
        out["xwOBA_pct"] = normalize_percentile(out["team_xwOBA"])

    return out


def aggregate_team_pitching_metrics(
    statcast_pitching: pd.DataFrame,
    park_profile: pd.DataFrame,
    spin_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate player-level pitching metrics + park/spin data to team level.

    statcast_pitching: output of pull_fg_pitching() (Savant fallback)
    park_profile:      output of build_park_profile()
    spin_profile:      output of build_pitcher_spin_profile() pitcher level

    Returns one row per team with pre-normalized metric columns.
    """
    from ingest import normalize_percentile

    # Savant pitching: player_id = MLBAM — need team affiliation
    # Without a roster lookup we can't reliably assign team here.
    # Return park_profile columns which already have team-level data.
    if park_profile.empty:
        log.warning("aggregate_team_pitching_metrics: empty park_profile")
        return pd.DataFrame()

    out = park_profile[["team", "pitcher_friendly", "hr_pf_pct"]].copy()
    out = out.rename(columns={
        "pitcher_friendly": "ParkPitcherFriendly_pct",
        "hr_pf_pct":        "HRPark_pct",
    })

    # K%, BB%, velo, spin from Savant pitching (if team column available)
    pit = statcast_pitching.copy()
    if "player_id" in pit.columns:
        pit = pit.rename(columns={"player_id": "key_mlbam"})

    # Savant pitching percentile ranks columns of interest
    savant_pit_map = {
        "k_percent":       "team_K_pitch",
        "bb_percent":      "team_BB_pitch",
        "hard_hit_percent":"team_HardHit_allowed",
        "whiff_percent":   "team_SwStr",
        "xwoba":           "team_xwOBA_allowed",
        "fb_velocity":     "team_AvgVelo",
        "fb_spin":         "team_AvgSpinRate",
    }

    # Without reliable team affiliation for individual pitchers, flag as gap
    log.info(
        "aggregate_team_pitching_metrics: pitcher→team mapping requires "
        "roster lookup (mlbstatsapi). Returning park columns only for now."
    )

    return out
