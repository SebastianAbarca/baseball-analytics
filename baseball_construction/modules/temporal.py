"""
temporal.py — Three-mode temporal processing applied universally.

Three modes, determined per player/team:
  historical — fewer than 15 PA or 15 games played
               use 3-year weighted average of prior seasons
  early      — 15–199 PA or 15–49 games played
               blend historical + current; current weight ramps 0→50%
  current    — 200+ PA or 50+ games played
               use current season directly (upstream reliability weighting applies)

Historical smoothing weights:
  lag 0 (most recent prior season) = 0.50
  lag 1 (two seasons ago)          = 0.30
  lag 2 (three seasons ago)        = 0.20

All functions operate on scalar metric values.
Call apply_temporal_mode() per metric, or process_player_metrics() for a full dict.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HISTORICAL_WEIGHTS: dict[int, float] = {
    0: 0.50,   # most recent prior season
    1: 0.30,   # two seasons ago
    2: 0.20,   # three seasons ago
}

# Mode thresholds — hitters use PA; pitchers use BF (caller maps)
PA_HISTORICAL_THRESHOLD  = 15
PA_EARLY_THRESHOLD       = 200
GAMES_HISTORICAL_THRESHOLD = 15
GAMES_EARLY_THRESHOLD      = 50

# Early-mode max current weight (current weight ramps from 0 → this)
MAX_CURRENT_WEIGHT = 0.50


# ---------------------------------------------------------------------------
# Step 1 — Mode determination
# ---------------------------------------------------------------------------

def determine_mode(current_pa: int, games_played: int) -> str:
    """
    Determine temporal mode for a player or team.

    For hitters: current_pa = plate appearances.
    For pitchers: current_pa = batters faced (caller responsibility).
    For teams:    current_pa = team plate appearances.

    Returns: 'historical' | 'early' | 'current'
    """
    if current_pa < PA_HISTORICAL_THRESHOLD or games_played < GAMES_HISTORICAL_THRESHOLD:
        return "historical"
    if current_pa < PA_EARLY_THRESHOLD or games_played < GAMES_EARLY_THRESHOLD:
        return "early"
    return "current"


# ---------------------------------------------------------------------------
# Step 2 — Early-mode blend weights
# ---------------------------------------------------------------------------

def blend_weights(games_played: int) -> tuple[float, float]:
    """
    Compute (historical_weight, current_weight) for early mode.

    Progress ramps linearly from 0 at game 15 to 1.0 at game 50.
    current_weight = progress × MAX_CURRENT_WEIGHT (max 0.50)
    historical_weight = 1 - current_weight

    Returns: (historical_weight, current_weight)
    """
    progress = min(
        (games_played - GAMES_HISTORICAL_THRESHOLD)
        / (GAMES_EARLY_THRESHOLD - GAMES_HISTORICAL_THRESHOLD),
        1.0,
    )
    current_weight    = progress * MAX_CURRENT_WEIGHT
    historical_weight = 1.0 - current_weight
    return historical_weight, current_weight


# ---------------------------------------------------------------------------
# Step 3 — Historical smoothing
# ---------------------------------------------------------------------------

def compute_historical_smooth(
    season_values: dict[int, Optional[float]],
    reference_season: int,
) -> Optional[float]:
    """
    3-year weighted average of prior seasons.

    Args:
        season_values:    {season_year: metric_value}
                          None values are treated as missing.
        reference_season: the most recent completed prior season
                          (e.g. if current season is 2024, reference = 2023)

    Returns: weighted mean, or None if no historical data.

    Missing seasons are handled by reweighting available lags.
    """
    weighted_sum  = 0.0
    total_weight  = 0.0

    for lag, weight in HISTORICAL_WEIGHTS.items():
        season = reference_season - lag
        val    = season_values.get(season)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        weighted_sum += float(val) * weight
        total_weight += weight

    if total_weight == 0.0:
        return None

    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Step 4 — Apply temporal mode to a single metric value
# ---------------------------------------------------------------------------

def apply_temporal_mode(
    mode:              str,
    current_value:     Optional[float],
    historical_value:  Optional[float],
    games_played:      int,
) -> Optional[float]:
    """
    Return the temporally-adjusted value for one metric.

    Args:
        mode:             'historical' | 'early' | 'current'
        current_value:    current season metric (post reliability-weighting)
        historical_value: output of compute_historical_smooth()
        games_played:     games played this season (for blend weight in early mode)

    Returns:
        Scalar metric value, or None if neither source is available.
    """
    if mode == "historical":
        return historical_value

    if mode == "current":
        # Fall back to historical if current is missing (e.g. new player)
        if current_value is None or (isinstance(current_value, float) and np.isnan(current_value)):
            return historical_value
        return current_value

    # early mode
    hist_w, curr_w = blend_weights(games_played)

    has_current    = current_value    is not None and not np.isnan(float(current_value or 0))
    has_historical = historical_value is not None and not np.isnan(float(historical_value or 0))

    if has_current and has_historical:
        return hist_w * historical_value + curr_w * current_value
    if has_current:
        return current_value      # no history → trust current only
    if has_historical:
        return historical_value   # no current data → use history
    return None


# ---------------------------------------------------------------------------
# Step 5 — Delta (current vs historical expectation)
# ---------------------------------------------------------------------------

def compute_delta(
    current_value:    Optional[float],
    historical_value: Optional[float],
) -> Optional[float]:
    """
    Delta = current_value - historical_value.

    Positive → performing above historical expectation (breakout signal).
    Negative → performing below historical expectation (decline signal).
    None     → insufficient data to compute.
    """
    if current_value is None or historical_value is None:
        return None
    if np.isnan(float(current_value)) or np.isnan(float(historical_value)):
        return None
    return float(current_value) - float(historical_value)


# ---------------------------------------------------------------------------
# Step 6 — Process a full dict of metrics for one player/team
# ---------------------------------------------------------------------------

def process_metrics(
    current_pa:        int,
    games_played:      int,
    current_metrics:   dict[str, Optional[float]],
    historical_metrics:dict[str, Optional[float]],
    reference_season:  Optional[int] = None,
    season_history:    Optional[dict[str, dict[int, float]]] = None,
) -> dict[str, dict]:
    """
    Apply temporal processing to every metric in a dict.

    Supports two calling patterns:

    Pattern A — pre-computed historical (historical_metrics already smoothed):
        Pass historical_metrics directly.

    Pattern B — multi-season history (season_history provided):
        season_history = {metric_key: {2023: val, 2022: val, 2021: val}}
        reference_season must be provided.
        historical_metrics is computed internally and can be empty {}.

    Args:
        current_pa:         plate appearances (or BF for pitchers)
        games_played:       games played this season
        current_metrics:    {metric: current_value}  (0–100 percentile or raw)
        historical_metrics: {metric: smoothed_historical_value}  OR {}
        reference_season:   most recent completed prior season year
        season_history:     {metric: {year: value}} for pattern B

    Returns:
        {
          metric_key: {
            mode:             'historical' | 'early' | 'current',
            current:          value,
            historical:       smoothed value,
            adjusted:         temporally-adjusted value,
            delta:            current - historical,
            hist_weight:      weight given to historical (early mode only),
            curr_weight:      weight given to current   (early mode only),
          }
        }
    """
    mode = determine_mode(current_pa, games_played)
    hist_w, curr_w = blend_weights(games_played) if mode == "early" else (1.0, 0.0)

    # Resolve historical per metric
    resolved_hist: dict[str, Optional[float]] = dict(historical_metrics)
    if season_history and reference_season:
        for metric, year_vals in season_history.items():
            if metric not in resolved_hist or resolved_hist[metric] is None:
                resolved_hist[metric] = compute_historical_smooth(year_vals, reference_season)

    all_metrics = set(current_metrics) | set(resolved_hist)
    result: dict[str, dict] = {}

    for metric in all_metrics:
        cur = current_metrics.get(metric)
        hist = resolved_hist.get(metric)
        adj  = apply_temporal_mode(mode, cur, hist, games_played)
        delta = compute_delta(cur, hist)
        result[metric] = {
            "mode":        mode,
            "current":     cur,
            "historical":  hist,
            "adjusted":    adj,
            "delta":       delta,
            "hist_weight": hist_w if mode == "early" else (1.0 if mode == "historical" else 0.0),
            "curr_weight": curr_w if mode == "early" else (0.0 if mode == "historical" else 1.0),
        }

    log.debug(
        "process_metrics — mode=%s | PA=%d | games=%d | metrics=%d | "
        "hist_w=%.2f curr_w=%.2f",
        mode, current_pa, games_played, len(result), hist_w, curr_w,
    )
    return result


# ---------------------------------------------------------------------------
# Step 7 — Batch: DataFrame of players → temporally-adjusted DataFrame
# ---------------------------------------------------------------------------

def apply_temporal_to_df(
    df:              pd.DataFrame,
    pa_col:          str,
    games_col:       str,
    metric_cols:     list[str],
    hist_suffix:     str = "_hist",
    adjusted_suffix: str = "_adj",
    delta_suffix:    str = "_delta",
) -> pd.DataFrame:
    """
    Apply temporal mode to every metric column in a DataFrame.

    For each metric in metric_cols:
      - current value column:    metric
      - historical value column: metric + hist_suffix  (must exist)
    Adds:
      - metric + adjusted_suffix
      - metric + delta_suffix
      - 'mode' column

    Args:
        df:           player/team DataFrame
        pa_col:       column name for PA count (or BF for pitchers)
        games_col:    column name for games played
        metric_cols:  list of metric columns to process
        hist_suffix:  suffix of historical columns  (default: '_hist')
        adjusted_suffix: suffix for adjusted output (default: '_adj')
        delta_suffix:    suffix for delta output    (default: '_delta')
    """
    df = df.copy()

    def _row_mode(row: pd.Series) -> str:
        return determine_mode(int(row[pa_col] or 0), int(row[games_col] or 0))

    df["mode"] = df.apply(_row_mode, axis=1)

    def _row_weights(row: pd.Series) -> tuple[float, float]:
        return blend_weights(int(row[games_col] or 0))

    for metric in metric_cols:
        hist_col = metric + hist_suffix
        if hist_col not in df.columns:
            log.warning(
                "apply_temporal_to_df: historical column '%s' not found — skipping %s",
                hist_col, metric,
            )
            continue

        adj_col   = metric + adjusted_suffix
        delta_col = metric + delta_suffix

        def _adj(row: pd.Series, m: str = metric, hc: str = hist_col) -> Optional[float]:
            return apply_temporal_mode(
                row["mode"],
                row.get(m),
                row.get(hc),
                int(row[games_col] or 0),
            )

        def _delta(row: pd.Series, m: str = metric, hc: str = hist_col) -> Optional[float]:
            return compute_delta(row.get(m), row.get(hc))

        df[adj_col]   = df.apply(_adj,   axis=1)
        df[delta_col] = df.apply(_delta, axis=1)

    return df
