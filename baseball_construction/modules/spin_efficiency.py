"""
spin_efficiency.py — Magnus-force spin efficiency per pitcher × pitch type.

Formula from Nathan (2008):
  theoretical_max_movement = MAGNUS_K × spin_rate / speed × 12
  spin_efficiency = actual_movement / theoretical_max_movement, clipped [0, 1]

Pipeline:
  1. compute_spin_efficiency(df)           — pitch-level
  2. aggregate_pitcher_pitch_type(df)      — group to pitcher × pitch_type
  3. apply_reliability_weighting(agg)      — shrink toward league mean
  4. normalize_spin_efficiency(agg)        — percentile 0–100
  5. build_pitcher_spin_profile(df)        — full pipeline in one call
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ingest import (
    STABILIZATION,
    apply_reliability_weight,
    compute_reliability,
    normalize_percentile,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGNUS_K = 0.00153  # Nathan 2008 empirical constant

# Pitch types to exclude from spin efficiency analysis
# (knuckleballs and eephus have near-zero intentional spin)
EXCLUDE_PITCH_TYPES = {"KN", "EP", "PO", "IN", "FO"}

# Minimum pitches to include a pitcher × pitch_type row in any output
MIN_PITCHES_TO_INCLUDE = 10

# ---------------------------------------------------------------------------
# Step 1 — Pitch-level computation
# ---------------------------------------------------------------------------

def compute_spin_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute spin efficiency for every pitch row.

    Adds columns:
      movement_mag        — actual movement magnitude (same units as pfx_x/pfx_z)
      theoretical_max_mov — max movement a given spin rate could produce
      spin_efficiency     — ratio clipped to [0, 1]

    Rows with missing pfx_x, pfx_z, release_spin_rate, or release_speed
    receive NaN spin_efficiency.
    """
    df = df.copy()

    required = ["pfx_x", "pfx_z", "release_spin_rate", "release_speed"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"compute_spin_efficiency: missing columns {missing_cols}")

    df["movement_mag"] = np.sqrt(df["pfx_x"] ** 2 + df["pfx_z"] ** 2)

    # Theoretical maximum: spin fully converted to Magnus-force movement
    # release_speed in mph, release_spin_rate in rpm
    df["theoretical_max_mov"] = (
        MAGNUS_K * df["release_spin_rate"] / df["release_speed"] * 12
    )

    # Zero or null denominator → NaN rather than inf
    bad_denom = df["theoretical_max_mov"].isna() | (df["theoretical_max_mov"] <= 0)
    df.loc[bad_denom, "theoretical_max_mov"] = np.nan

    df["spin_efficiency"] = (
        (df["movement_mag"] / df["theoretical_max_mov"]).clip(0, 1)
    )

    # Exclude pitch types where spin efficiency is not meaningful
    if "pitch_type" in df.columns:
        df.loc[df["pitch_type"].isin(EXCLUDE_PITCH_TYPES), "spin_efficiency"] = np.nan

    valid = df["spin_efficiency"].notna().sum()
    log.info(
        "compute_spin_efficiency — %d pitches with valid spin_efficiency / %d total",
        valid, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 — Aggregate to pitcher × pitch_type
# ---------------------------------------------------------------------------

def aggregate_pitcher_pitch_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level data to pitcher × pitch_type.

    Input df must already have spin_efficiency column (run compute_spin_efficiency first).

    Returns columns:
      pitcher, pitch_type, pitch_count,
      spin_efficiency_raw,   (mean of pitch-level values)
      spin_rate_mean,
      velo_mean,
      movement_mag_mean,
      pfx_x_mean,           (signed horizontal break — positive = arm side)
      pfx_z_mean,           (signed vertical break)
      spin_axis_mean,
    """
    needed = {"spin_efficiency", "pitcher", "pitch_type"}
    if not needed.issubset(df.columns):
        raise ValueError(f"aggregate_pitcher_pitch_type: need columns {needed}")

    valid = df[df["spin_efficiency"].notna()].copy()

    agg_cols = {
        "pitch_count":         ("spin_efficiency",   "count"),
        "spin_efficiency_raw": ("spin_efficiency",   "mean"),
        "spin_rate_mean":      ("release_spin_rate", "mean"),
        "velo_mean":           ("release_speed",     "mean"),
        "movement_mag_mean":   ("movement_mag",      "mean"),
    }

    # Optional columns — include if present
    if "pfx_x" in valid.columns:
        agg_cols["pfx_x_mean"] = ("pfx_x", "mean")
    if "pfx_z" in valid.columns:
        agg_cols["pfx_z_mean"] = ("pfx_z", "mean")
    if "spin_axis" in valid.columns:
        agg_cols["spin_axis_mean"] = ("spin_axis", "mean")

    agg = valid.groupby(["pitcher", "pitch_type"]).agg(**agg_cols).reset_index()

    # Drop rows below minimum threshold (too sparse to be meaningful)
    before = len(agg)
    agg = agg[agg["pitch_count"] >= MIN_PITCHES_TO_INCLUDE].copy()
    log.info(
        "aggregate_pitcher_pitch_type — %d pitcher×pitch_type rows "
        "(%d dropped below min=%d pitches)",
        len(agg), before - len(agg), MIN_PITCHES_TO_INCLUDE,
    )
    return agg


# ---------------------------------------------------------------------------
# Step 3 — Reliability weighting
# ---------------------------------------------------------------------------

def apply_reliability_weighting(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Shrink spin_efficiency_raw toward the league mean based on sample size.

    reliability = min(1.0, pitch_count / 200)
    weighted    = (reliability × raw) + ((1 - reliability) × league_mean)

    Adds columns:
      reliability             — 0.0–1.0
      league_mean_spin_eff    — population mean used for shrinkage
      spin_efficiency_weighted
    """
    agg = agg.copy()

    # League mean computed across all rows in this aggregated pool
    league_mean = agg["spin_efficiency_raw"].mean()
    agg["league_mean_spin_eff"] = league_mean

    agg["reliability"] = agg["pitch_count"].apply(
        lambda n: compute_reliability(int(n), "spin_efficiency")
    )

    agg["spin_efficiency_weighted"] = agg.apply(
        lambda row: apply_reliability_weight(
            row["spin_efficiency_raw"],
            row["reliability"],
            league_mean,
        ),
        axis=1,
    )

    log.info(
        "apply_reliability_weighting — league_mean=%.3f | "
        "reliability: min=%.2f  median=%.2f  max=%.2f",
        league_mean,
        agg["reliability"].min(),
        agg["reliability"].median(),
        agg["reliability"].max(),
    )
    return agg


# ---------------------------------------------------------------------------
# Step 4 — Percentile normalization
# ---------------------------------------------------------------------------

def normalize_spin_efficiency(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Percentile-rank spin_efficiency_weighted within the pitcher × pitch_type pool.

    Higher spin efficiency = higher percentile (not inverted).
    Adds column: spin_efficiency_pct (0–100).
    """
    agg = agg.copy()
    agg["spin_efficiency_pct"] = normalize_percentile(
        agg["spin_efficiency_weighted"], invert=False
    )
    log.info(
        "normalize_spin_efficiency — pct range: %.1f – %.1f",
        agg["spin_efficiency_pct"].min(),
        agg["spin_efficiency_pct"].max(),
    )
    return agg


# ---------------------------------------------------------------------------
# Step 5 — Pitcher-level rollup (across pitch types)
# ---------------------------------------------------------------------------

def build_pitcher_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up from pitcher × pitch_type to pitcher-level.

    Weights each pitch type by pitch_count.
    Returns one row per pitcher with:
      spin_efficiency_weighted (weighted mean)
      spin_efficiency_pct      (re-computed percentile at pitcher level)
      dominant_pitch_type      (highest-count pitch type)
      pitch_types              (comma-separated list)
      total_pitches
    """
    def weighted_mean(group: pd.DataFrame) -> pd.Series:
        w = group["pitch_count"]
        wm = np.average(group["spin_efficiency_weighted"], weights=w)
        dominant = group.loc[group["pitch_count"].idxmax(), "pitch_type"]
        types = ",".join(sorted(group["pitch_type"].tolist()))
        return pd.Series({
            "spin_efficiency_weighted": wm,
            "dominant_pitch_type":      dominant,
            "pitch_types":              types,
            "total_pitches":            w.sum(),
        })

    summary = agg.groupby("pitcher").apply(weighted_mean, include_groups=False).reset_index()
    summary["spin_efficiency_pct"] = normalize_percentile(
        summary["spin_efficiency_weighted"], invert=False
    )
    log.info("build_pitcher_summary — %d pitchers", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_pitcher_spin_profile(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete spin efficiency pipeline.

    Args:
        df: pitch-level Statcast DataFrame

    Returns:
        (pitch_type_level, pitcher_level)
        pitch_type_level — one row per pitcher × pitch_type
        pitcher_level    — one row per pitcher (weighted rollup)
    """
    df = compute_spin_efficiency(df)
    agg = aggregate_pitcher_pitch_type(df)
    agg = apply_reliability_weighting(agg)
    agg = normalize_spin_efficiency(agg)
    summary = build_pitcher_summary(agg)
    return agg, summary


def load_league_spin(
    season: int,
    statcast: pd.DataFrame | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    League-wide pitcher spin-efficiency summary for one season, cached to
    parquet. spin_efficiency_pct here is ranked against ALL pitchers in the
    season — the per-team pipeline only ranks within one team's ~30 arms,
    which is not valid for league-percentile claims (archetype scoring).
    """
    _here = Path(__file__).parent
    processed = _here / "processed"
    processed.mkdir(exist_ok=True)
    cache = processed / f"spin_league_{season}.parquet"
    if cache.exists() and not force:
        log.info("spin_league %d: loading from cache", season)
        return pd.read_parquet(cache)

    if statcast is None:
        from ingest import pull_statcast_season
        statcast = pull_statcast_season(season)

    log.info("spin_league %d: building from Statcast (%d pitches)…",
             season, len(statcast))
    _, summary = build_pitcher_spin_profile(statcast)
    if not summary.empty:
        summary.to_parquet(cache, index=False)
        log.info("spin_league %d: saved %d pitchers to cache", season, len(summary))
    return summary
