"""
park_effects.py — Park factors computed from Statcast event data.

Method: wOBA-based and HR-based park factors using plate appearance outcomes
extracted from pitch-level Statcast data.

Standard two-component formula:
  home_rate  = wOBA (or HR/PA) for all PAs played at a given park
  road_rate  = wOBA (or HR/PA) for all away PAs by the same team
  raw_PF     = (home_rate / road_rate).  Neutral park = 1.0.

Pitcher-friendly interpretation (used in B4 — Defensive Infrastructure):
  lower wOBA park factor = more pitcher-friendly = higher ParkPitcherFriendly_pct

Outputs (one row per team × season):
  team, season,
  woba_pf          — wOBA park factor  (neutral = 1.0)
  hr_pf            — HR park factor    (neutral = 1.0)
  pitcher_friendly — wOBA PF inverted, 0–100 percentile (high = pitcher-friendly)
  hr_pf_pct        — HR PF, 0–100 (high = HR-friendly)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ingest import normalize_percentile

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regression toward mean: single-season park factors are noisy.
# Apply a standard 1/3 regression toward 1.0.
REGRESSION_WEIGHT = 1 / 3

# Minimum PAs at home and on the road to include a team in output
MIN_HOME_PA = 200
MIN_ROAD_PA = 200


# ---------------------------------------------------------------------------
# Step 1 — Extract PA-level outcomes from pitch-level Statcast
# ---------------------------------------------------------------------------

def extract_pa_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce pitch-level Statcast to one row per plate appearance.

    Keeps only the terminal pitch of each PA (the pitch where the
    at-bat ends: events is not null and woba_denom == 1).

    Returns columns:
      game_pk, at_bat_number, batter, pitcher,
      home_team, away_team,
      events, woba_value, woba_denom,
      is_hr
    """
    required = {
        "game_pk", "at_bat_number", "batter", "pitcher",
        "home_team", "away_team",
        "woba_value", "woba_denom", "events",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"extract_pa_outcomes: missing columns {missing}")

    # Terminal pitch of each PA = last pitch with a recorded event
    pa = df[df["events"].notna()].copy()

    # Keep only scoreable PAs (woba_denom tracks which PAs count toward wOBA)
    pa = pa[pa["woba_denom"] == 1].copy()

    pa["is_hr"] = (pa["events"] == "home_run").astype(int)

    cols = [
        "game_pk", "at_bat_number", "batter", "pitcher",
        "home_team", "away_team",
        "events", "woba_value", "woba_denom", "is_hr",
    ]
    pa = pa[cols].drop_duplicates(subset=["game_pk", "at_bat_number"]).copy()

    log.info("extract_pa_outcomes — %d PA outcomes from %d pitches", len(pa), len(df))
    return pa


# ---------------------------------------------------------------------------
# Step 2 — Compute home and road splits per team
# ---------------------------------------------------------------------------

def _split_home_road(pa: pd.DataFrame) -> pd.DataFrame:
    """
    For each PA, create two rows:
      one tagging the home team as the park host
      one tagging the visiting team as the road team

    Returns long-form DataFrame with columns:
      team, context (home|road), woba_value, is_hr
    """
    # Home PAs — batters playing in their home park
    home = pa[["home_team", "woba_value", "is_hr"]].copy()
    home = home.rename(columns={"home_team": "team"})
    home["context"] = "home"

    # Road PAs — batters playing away (at another team's park)
    road = pa[["away_team", "woba_value", "is_hr"]].copy()
    road = road.rename(columns={"away_team": "team"})
    road["context"] = "road"

    long = pd.concat([home, road], ignore_index=True)
    return long


def compute_raw_park_factors(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wOBA and HR park factors for each team.

    Args:
        pa: output of extract_pa_outcomes()

    Returns one row per team:
      team, home_pa, road_pa,
      home_woba, road_woba, league_woba,
      home_hr_rate, road_hr_rate,
      woba_pf, hr_pf
    """
    long = _split_home_road(pa)

    split = long.groupby(["team", "context"]).agg(
        pa_count  =("woba_value", "count"),
        woba_sum  =("woba_value", "sum"),
        hr_count  =("is_hr",      "sum"),
    ).reset_index()

    home = split[split["context"] == "home"].rename(columns={
        "pa_count": "home_pa",
        "woba_sum": "home_woba_sum",
        "hr_count": "home_hr",
    }).drop(columns="context")

    road = split[split["context"] == "road"].rename(columns={
        "pa_count": "road_pa",
        "woba_sum": "road_woba_sum",
        "hr_count": "road_hr",
    }).drop(columns="context")

    teams = home.merge(road, on="team", how="inner")

    # Filter minimum sample
    teams = teams[
        (teams["home_pa"] >= MIN_HOME_PA)
        & (teams["road_pa"] >= MIN_ROAD_PA)
    ].copy()

    teams["home_woba"]    = teams["home_woba_sum"] / teams["home_pa"]
    teams["road_woba"]    = teams["road_woba_sum"] / teams["road_pa"]
    teams["home_hr_rate"] = teams["home_hr"]  / teams["home_pa"]
    teams["road_hr_rate"] = teams["road_hr"]  / teams["road_pa"]

    # League averages (weighted by PA)
    total_pa    = long["woba_value"].notna().sum()
    league_woba = long["woba_value"].sum() / total_pa
    league_hr   = long["is_hr"].sum()      / total_pa

    teams["league_woba"]    = league_woba
    teams["league_hr_rate"] = league_hr

    # Raw park factors — avoid division by zero on road_rate
    safe_road_woba = teams["road_woba"].replace(0, np.nan)
    safe_road_hr   = teams["road_hr_rate"].replace(0, np.nan)

    teams["woba_pf_raw"] = teams["home_woba"] / safe_road_woba
    teams["hr_pf_raw"]   = teams["home_hr_rate"] / safe_road_hr

    log.info(
        "compute_raw_park_factors — %d teams | league_woba=%.3f | league_hr_rate=%.4f",
        len(teams), league_woba, league_hr,
    )
    return teams


# ---------------------------------------------------------------------------
# Step 3 — Regression toward 1.0
# ---------------------------------------------------------------------------

def apply_regression(teams: pd.DataFrame, weight: float = REGRESSION_WEIGHT) -> pd.DataFrame:
    """
    Regress raw park factors toward the neutral value of 1.0.

    Reduces noise from single-season samples.
    regressed_PF = (1 - weight) × raw_PF + weight × 1.0

    Adds: woba_pf, hr_pf  (regressed versions; raw preserved as *_raw)
    """
    teams = teams.copy()
    teams["woba_pf"] = (1 - weight) * teams["woba_pf_raw"] + weight * 1.0
    teams["hr_pf"]   = (1 - weight) * teams["hr_pf_raw"]   + weight * 1.0
    log.info(
        "apply_regression — woba_pf range: %.3f – %.3f | hr_pf range: %.3f – %.3f",
        teams["woba_pf"].min(), teams["woba_pf"].max(),
        teams["hr_pf"].min(),   teams["hr_pf"].max(),
    )
    return teams


# ---------------------------------------------------------------------------
# Step 4 — Percentile normalization
# ---------------------------------------------------------------------------

def normalize_park_factors(teams: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank park factors within the team pool.

    pitcher_friendly (0–100): high = suppresses offense.
      woba_pf inverted so lower PF = higher percentile.

    hr_pf_pct (0–100): high = HR-friendly park.
      Not inverted.

    Adds: pitcher_friendly, hr_pf_pct
    """
    teams = teams.copy()

    # Pitcher-friendly: invert woba_pf (lower offense = more pitcher-friendly)
    teams["pitcher_friendly"] = normalize_percentile(teams["woba_pf"], invert=True)

    # HR factor: not inverted (higher hr_pf = more HR-friendly)
    teams["hr_pf_pct"] = normalize_percentile(teams["hr_pf"], invert=False)

    log.info(
        "normalize_park_factors — pitcher_friendly range: %.1f – %.1f",
        teams["pitcher_friendly"].min(), teams["pitcher_friendly"].max(),
    )
    return teams


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_park_profile(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Full park effects pipeline: pitch-level Statcast → team park factors.

    Args:
        df:     pitch-level Statcast DataFrame
        season: season year (attached to output for multi-season use)

    Returns one row per team with:
      team, season, home_pa, road_pa,
      home_woba, road_woba, league_woba,
      woba_pf_raw, woba_pf,
      hr_pf_raw, hr_pf,
      pitcher_friendly, hr_pf_pct
    """
    pa    = extract_pa_outcomes(df)
    teams = compute_raw_park_factors(pa)
    teams = apply_regression(teams)
    teams = normalize_park_factors(teams)
    teams["season"] = season

    col_order = [
        "team", "season", "home_pa", "road_pa",
        "home_woba", "road_woba", "league_woba",
        "woba_pf_raw", "woba_pf",
        "hr_pf_raw", "hr_pf",
        "pitcher_friendly", "hr_pf_pct",
    ]
    col_order = [c for c in col_order if c in teams.columns]
    return teams[col_order].sort_values("woba_pf").reset_index(drop=True)
