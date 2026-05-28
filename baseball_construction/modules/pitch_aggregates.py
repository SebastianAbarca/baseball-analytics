"""
pitch_aggregates.py — Per-player aggregates computed from Statcast pitch data.

Fills gaps that BRef/Savant endpoints don't expose:
  batters  — gb_pct, fps_pct, pitches_per_pa
  pitchers — gb_pct, zone_pct, csw_pct, pitches_per_bf

All metrics are raw rates (0–1 scale); normalization happens downstream
in team_portrait._normalize_batting / _normalize_pitching.

Uses the cached Statcast parquet (pull_statcast_season) — no extra API calls.
In-process cache (_CACHE) prevents double-computation when batting and pitching
merge functions both call compute_pitch_aggregates in the same seed run.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event-type sets
# ---------------------------------------------------------------------------

_SWUNG: frozenset[str] = frozenset({
    "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "foul_bunt", "missed_bunt",
})

_CSW: frozenset[str] = frozenset({
    "called_strike", "swinging_strike", "swinging_strike_blocked",
})

_SWING_MISS: frozenset[str] = frozenset({
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
})

# ---------------------------------------------------------------------------
# In-process cache keyed by season
# ---------------------------------------------------------------------------

_CACHE: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _batter_aggregates(sc: pd.DataFrame, min_pa: int = 50) -> pd.DataFrame:
    """
    Per-batter aggregates from pitch-level Statcast.

    Returns DataFrame with columns:
        key_mlbam, gb_pct, fps_pct, pitches_per_pa, zone_swing_pct, contact_pct
    """
    df = sc[["batter", "game_pk", "at_bat_number", "pitch_number",
             "description", "bb_type", "zone"]].copy()

    # Ground ball flag
    df["_is_gb"]   = df["bb_type"] == "ground_ball"
    df["_has_bip"] = df["bb_type"].notna()

    # First-pitch swing flag
    df["_is_fp"]    = df["pitch_number"] == 1
    df["_fp_swung"] = df["_is_fp"] & df["description"].isin(_SWUNG)

    # Zone swing% — pitches in Statcast zones 1–9 (strike zone)
    df["_in_zone"]    = pd.to_numeric(df["zone"], errors="coerce").between(1, 9)
    df["_zone_swing"] = df["_in_zone"] & df["description"].isin(_SWUNG)

    # Contact rate — swings that don't miss
    df["_is_swing"]      = df["description"].isin(_SWUNG)
    df["_is_swing_miss"] = df["description"].isin(_SWING_MISS)

    # Aggregate counting stats per batter
    agg = df.groupby("batter", sort=False).agg(
        gb_n          = ("_is_gb",        "sum"),
        bip_n         = ("_has_bip",      "sum"),
        fp_swung_n    = ("_fp_swung",     "sum"),
        fp_n          = ("_is_fp",        "sum"),
        total_pitches = ("pitch_number",  "count"),
        zone_swing_n  = ("_zone_swing",   "sum"),
        zone_pitch_n  = ("_in_zone",      "sum"),
        swing_n       = ("_is_swing",     "sum"),
        swing_miss_n  = ("_is_swing_miss","sum"),
    )

    # PA count = distinct (game_pk, at_bat_number) per batter
    pa_counts = (
        df.drop_duplicates(["batter", "game_pk", "at_bat_number"])
        .groupby("batter", sort=False)
        .size()
        .rename("pa_n")
    )
    agg = agg.join(pa_counts).fillna({"pa_n": 0})

    agg = agg[agg["pa_n"] >= min_pa].copy()

    agg["gb_pct"]         = np.where(agg["bip_n"] > 0,       agg["gb_n"] / agg["bip_n"],                              np.nan)
    agg["fps_pct"]        = np.where(agg["fp_n"] > 0,        agg["fp_swung_n"] / agg["fp_n"],                         np.nan)
    agg["pitches_per_pa"] = np.where(agg["pa_n"] > 0,        agg["total_pitches"] / agg["pa_n"],                      np.nan)
    agg["zone_swing_pct"] = np.where(agg["zone_pitch_n"] > 0, agg["zone_swing_n"] / agg["zone_pitch_n"],              np.nan)
    agg["contact_pct"]    = np.where(agg["swing_n"] > 0,     (agg["swing_n"] - agg["swing_miss_n"]) / agg["swing_n"], np.nan)

    result = agg[["gb_pct", "fps_pct", "pitches_per_pa", "zone_swing_pct", "contact_pct"]].reset_index()
    result = result.rename(columns={"batter": "key_mlbam"})
    log.info("Batter pitch aggregates — %d players (min_pa=%d)", len(result), min_pa)
    return result


def _arsenal_entropy(sc: pd.DataFrame) -> pd.Series:
    """Shannon entropy of pitch_type distribution per pitcher (arsenal diversity)."""
    valid = sc[sc["pitch_type"].notna() & (sc["pitch_type"] != "")]
    pt = valid.groupby(["pitcher", "pitch_type"]).size()
    totals = pt.groupby("pitcher").sum()
    probs  = pt / totals
    entropy = (-probs * np.log(probs + 1e-9)).groupby("pitcher").sum()
    return entropy.rename("arsenal_diversity")


def _pitcher_aggregates(sc: pd.DataFrame, min_bf: int = 30) -> pd.DataFrame:
    """
    Per-pitcher aggregates from pitch-level Statcast.

    Returns DataFrame with columns:
        key_mlbam, gb_pct, zone_pct, csw_pct, pitches_per_bf,
        avg_spin_rate, arsenal_diversity
    """
    df = sc[["pitcher", "game_pk", "at_bat_number", "batter",
             "pitch_number", "description", "bb_type",
             "plate_x", "plate_z", "sz_top", "sz_bot",
             "release_spin_rate", "pitch_type"]].copy()

    # Ground ball flag
    df["_is_gb"]   = df["bb_type"] == "ground_ball"
    df["_has_bip"] = df["bb_type"].notna()

    # Zone flag — use per-pitch sz_top/sz_bot, fall back to league averages
    plate_x = pd.to_numeric(df["plate_x"], errors="coerce")
    plate_z = pd.to_numeric(df["plate_z"], errors="coerce")
    sz_top  = pd.to_numeric(df["sz_top"], errors="coerce").fillna(3.5)
    sz_bot  = pd.to_numeric(df["sz_bot"], errors="coerce").fillna(1.5)
    df["_in_zone"]      = (plate_x.abs() <= 0.83) & (plate_z >= sz_bot) & (plate_z <= sz_top)
    df["_has_location"] = plate_x.notna() & plate_z.notna()

    # CSW flag
    df["_is_csw"] = df["description"].isin(_CSW)

    # Spin rate
    df["_spin"]     = pd.to_numeric(df["release_spin_rate"], errors="coerce")
    df["_has_spin"] = df["_spin"].notna()

    agg = df.groupby("pitcher", sort=False).agg(
        gb_n          = ("_is_gb",        "sum"),
        bip_n         = ("_has_bip",      "sum"),
        zone_n        = ("_in_zone",      "sum"),
        location_n    = ("_has_location", "sum"),
        csw_n         = ("_is_csw",       "sum"),
        total_pitches = ("pitch_number",  "count"),
        spin_sum      = ("_spin",         "sum"),
        spin_n        = ("_has_spin",     "sum"),
    )

    # BF count = distinct (game_pk, at_bat_number, batter) per pitcher
    bf_counts = (
        df.drop_duplicates(["pitcher", "game_pk", "at_bat_number", "batter"])
        .groupby("pitcher", sort=False)
        .size()
        .rename("bf_n")
    )
    agg = agg.join(bf_counts).fillna({"bf_n": 0})

    agg = agg[agg["bf_n"] >= min_bf].copy()

    agg["gb_pct"]         = np.where(agg["bip_n"]      > 0, agg["gb_n"]    / agg["bip_n"],                   np.nan)
    agg["zone_pct"]       = np.where(agg["location_n"] > 0, agg["zone_n"]  / agg["location_n"],               np.nan)
    agg["csw_pct"]        = np.where(agg["total_pitches"] > 0, agg["csw_n"] / agg["total_pitches"],           np.nan)
    agg["pitches_per_bf"] = np.where(agg["bf_n"]        > 0, agg["total_pitches"] / agg["bf_n"],              np.nan)
    agg["avg_spin_rate"]  = np.where(agg["spin_n"]      > 0, agg["spin_sum"] / agg["spin_n"],                 np.nan)

    # Arsenal diversity — Shannon entropy of pitch mix
    entropy = _arsenal_entropy(df)
    agg = agg.join(entropy, how="left")

    result = agg[["gb_pct", "zone_pct", "csw_pct", "pitches_per_bf",
                  "avg_spin_rate", "arsenal_diversity"]].reset_index()
    result = result.rename(columns={"pitcher": "key_mlbam"})
    log.info("Pitcher pitch aggregates — %d pitchers (min_bf=%d)", len(result), min_bf)
    return result


# ---------------------------------------------------------------------------
# Opener usage (all 30 teams from full Statcast)
# ---------------------------------------------------------------------------

def compute_platoon_optimization(statcast: pd.DataFrame) -> pd.Series:
    """
    For each pitching team, fraction of PAs where pitcher and batter share handedness
    (same-hand matchups: RHP vs RHH, LHP vs LHH).

    Higher = team creates more same-hand matchups for its pitchers.
    Returns Series indexed by team abbreviation, values in [0, 1].
    """
    sc = statcast[["pitcher", "batter", "p_throws", "stand",
                   "inning_topbot", "home_team", "away_team",
                   "game_pk", "at_bat_number"]].copy()

    # Only count one row per plate appearance (avoid per-pitch duplication)
    sc = sc.drop_duplicates(["game_pk", "at_bat_number"])

    sc["pitching_team"] = np.where(
        sc["inning_topbot"] == "Top",
        sc["home_team"],
        sc["away_team"],
    )

    sc = sc.dropna(subset=["p_throws", "stand", "pitching_team"])
    sc["same_hand"] = sc["p_throws"] == sc["stand"]

    stats = sc.groupby("pitching_team").agg(
        same_hand_pa=("same_hand", "sum"),
        total_pa=("same_hand", "count"),
    )
    result = (stats["same_hand_pa"] / stats["total_pa"]).rename("platoon_pct")
    log.info("Platoon optimization computed — %d teams", len(result))
    return result


def compute_opener_usage(statcast: pd.DataFrame) -> pd.Series:
    """
    For each team, fraction of games where the first pitcher faced ≤ 9 batters (opener).
    Returns Series indexed by team abbreviation, values in [0, 1].
    """
    sc = statcast[["game_pk", "pitcher", "batter", "at_bat_number",
                   "inning_topbot", "home_team", "away_team", "pitch_number"]].copy()

    # First pitcher per (game_pk, side) = pitcher who threw pitch number 1
    first = (
        sc[sc["pitch_number"] == 1]
        .drop_duplicates(["game_pk", "inning_topbot"])
        [["game_pk", "inning_topbot", "pitcher", "home_team", "away_team"]]
    )

    # Count distinct batters faced per (game_pk, pitcher) to detect short outings
    bf_per_game = (
        sc.drop_duplicates(["game_pk", "pitcher", "at_bat_number", "batter"])
        .groupby(["game_pk", "pitcher"])
        .size()
        .reset_index(name="game_bf")
    )

    first = first.merge(bf_per_game, on=["game_pk", "pitcher"], how="left")
    first["is_opener"] = first["game_bf"].fillna(0) <= 9  # ≤ 9 BF ≈ 3 innings

    # Pitching team: when inning_topbot == 'Top', away team bats → home team pitches
    first["pitching_team"] = np.where(
        first["inning_topbot"] == "Top",
        first["home_team"],
        first["away_team"],
    )

    stats = first.groupby("pitching_team").agg(
        opener_games=("is_opener", "sum"),
        total_games=("is_opener", "count"),
    )
    result = (stats["opener_games"] / stats["total_games"]).rename("opener_pct")
    log.info("Opener usage computed — %d teams", len(result))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pitch_aggregates(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (batter_agg, pitcher_agg) for the given season.

    Both DataFrames have key_mlbam as the join key.
    Results are cached in-process so a seed run that calls this for both
    batting and pitching only reads the Statcast parquet once.
    """
    if season in _CACHE:
        return _CACHE[season]

    from ingest import pull_statcast_season
    log.info("Computing pitch aggregates for season %d", season)
    sc = pull_statcast_season(season)

    batter_agg  = _batter_aggregates(sc)
    pitcher_agg = _pitcher_aggregates(sc)

    _CACHE[season] = (batter_agg, pitcher_agg)
    return batter_agg, pitcher_agg
