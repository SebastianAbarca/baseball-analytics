"""
ingest.py — Data pull, cache, normalize, reliability scoring.

All data is cached to parquet/CSV on first pull. Subsequent calls
load from cache — never re-pull if cache exists.

Master join key: key_mlbam (MLBAM ID). Never join on player name strings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pybaseball

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REF_DIR       = ROOT / "data" / "reference"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stabilization thresholds
# ---------------------------------------------------------------------------

STABILIZATION: dict[str, int] = {
    # Hitter metrics
    "K_pct":           60,
    "BB_pct":         120,
    "SwStr_pct":      150,
    "Chase_pct":      100,
    "Contact_pct":    150,
    "ZContact_pct":   150,
    "xwOBA":          200,
    "EV_90":          200,
    "HardHit_pct":    200,
    "Barrel_pct":     300,
    "xBA":            200,
    "BABIP":          800,
    "HR_per_FB":      300,
    "ISO":            300,
    "wRC_plus":       300,
    "OBP":            300,
    "FPS_pct":        150,
    "OSwing_pct":     100,
    "PitchesPerPA":   150,
    # Pitcher metrics
    "spin_rate":      200,
    "spin_efficiency":200,
    "tunnel_score":   150,
    "K_pct_pitch":    150,
    "BB_pct_pitch":   150,
    "GB_pct":         200,
    "HardHit_allowed":200,
    "Barrel_allowed": 200,
    "FIP":            150,
    # Catcher
    "framing":       1000,
    # Movement
    "sprint_speed":    50,
    "SB_attempts":      8,
}

# Default league-mean shrinkage target (percentile space: 0–1 scale)
LEAGUE_MEAN_DEFAULT: float = 0.50

# ---------------------------------------------------------------------------
# Season date helpers
# ---------------------------------------------------------------------------

SEASON_DATES: dict[int, tuple[str, str]] = {
    2015: ("2015-04-05", "2015-10-04"),
    2016: ("2016-04-03", "2016-10-02"),
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-28", "2019-09-29"),
    2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-10-04"),  # end date projected; Statcast pulls up to today
}

# ---------------------------------------------------------------------------
# Chadwick crosswalk
# ---------------------------------------------------------------------------

def load_chadwick() -> pd.DataFrame:
    """
    Load the Chadwick Bureau player register.

    Returns columns: key_mlbam, key_fangraphs, key_bbref,
                     name_first, name_last.
    """
    path = REF_DIR / "chadwick_crosswalk.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Chadwick crosswalk not found at {path}. "
            "Download from https://github.com/chadwickbureau/register"
        )

    cols = ["key_mlbam", "key_fangraphs", "key_bbref", "name_first", "name_last"]
    df = pd.read_csv(path, usecols=cols, low_memory=False)

    # Coerce ID columns to nullable int where possible
    for col in ("key_mlbam", "key_fangraphs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Only keep rows that have at least one usable ID
    df = df[df["key_mlbam"].notna() | df["key_fangraphs"].notna()].copy()

    log.info("Chadwick loaded — %d rows with at least one ID", len(df))
    return df


# ---------------------------------------------------------------------------
# Statcast pull / cache
# ---------------------------------------------------------------------------

def pull_statcast_season(season: int) -> pd.DataFrame:
    """
    Pull full-season Statcast pitch data.

    Caches to data/raw/statcast_{season}.parquet.
    Loads from cache on subsequent calls — never re-pulls.
    If the cache file is corrupt (e.g. from a failed prior download),
    deletes it and re-pulls.
    """
    cache = RAW_DIR / f"statcast_{season}.parquet"
    if cache.exists():
        try:
            log.info("Loading statcast_%d from cache: %s", season, cache)
            return pd.read_parquet(cache)
        except Exception as e:
            log.warning(
                "Cache file %s is corrupt (%s) — deleting and re-pulling", cache, e
            )
            cache.unlink()

    if season not in SEASON_DATES:
        raise ValueError(f"Season {season} not in SEASON_DATES — add dates manually.")

    start, end = SEASON_DATES[season]
    log.info("Pulling Statcast %d  (%s → %s) — this may take several minutes", season, start, end)
    pybaseball.cache.enable()
    df = pybaseball.statcast(start_dt=start, end_dt=end, verbose=True)
    df.to_parquet(cache, index=False)
    log.info("Saved %d rows to %s", len(df), cache)
    return df


def pull_statcast_range(start: str, end: str, label: str = "test") -> pd.DataFrame:
    """
    Pull Statcast for an arbitrary date range.

    Caches to data/raw/statcast_{label}.parquet.
    """
    cache = RAW_DIR / f"statcast_{label}.parquet"
    if cache.exists():
        try:
            log.info("Loading statcast_%s from cache: %s", label, cache)
            return pd.read_parquet(cache)
        except Exception as e:
            log.warning("Cache file %s is corrupt (%s) — deleting and re-pulling", cache, e)
            cache.unlink()

    log.info("Pulling Statcast %s → %s", start, end)
    pybaseball.cache.enable()
    df = pybaseball.statcast(start_dt=start, end_dt=end, verbose=True)
    df.to_parquet(cache, index=False)
    log.info("Saved %d rows to %s", len(df), cache)
    return df


# ---------------------------------------------------------------------------
# FanGraphs batting / pitching
# ---------------------------------------------------------------------------

def _savant_batting_aggregate(season: int, min_pa: int = 100) -> pd.DataFrame:
    """
    Assemble a batting aggregate from Baseball Savant endpoints.

    Uses statcast_batter_expected_stats + statcast_batter_percentile_ranks.
    player_id in these endpoints IS the MLBAM ID — no crosswalk join needed.
    """
    expected = pybaseball.statcast_batter_expected_stats(season, minPA=min_pa)
    pct_ranks = pybaseball.statcast_batter_percentile_ranks(season)

    expected = expected.rename(columns={"player_id": "key_mlbam"})
    pct_ranks = pct_ranks.rename(columns={"player_id": "key_mlbam"})

    # Split 'last_name, first_name' into Name
    if "last_name, first_name" in expected.columns:
        expected["Name"] = expected["last_name, first_name"].str.replace(
            r"^(.*?),\s*(.*)$", r"\2 \1", regex=True
        )

    df = expected.merge(
        pct_ranks.drop(columns=["player_name", "year"], errors="ignore"),
        on="key_mlbam",
        how="left",
    )
    # The join_source is trivially 'savant' for all rows (player_id = MLBAM)
    df["join_source"] = "savant"
    return df


def _savant_pitching_aggregate(season: int, min_pa: int = 30) -> pd.DataFrame:
    """
    Assemble a pitching aggregate from Baseball Savant endpoints.

    Uses statcast_pitcher_expected_stats + statcast_pitcher_percentile_ranks.
    player_id IS the MLBAM ID.
    """
    expected = pybaseball.statcast_pitcher_expected_stats(season, minPA=min_pa)
    pct_ranks = pybaseball.statcast_pitcher_percentile_ranks(season)

    expected = expected.rename(columns={"player_id": "key_mlbam"})
    pct_ranks = pct_ranks.rename(columns={"player_id": "key_mlbam"})

    if "last_name, first_name" in expected.columns:
        expected["Name"] = expected["last_name, first_name"].str.replace(
            r"^(.*?),\s*(.*)$", r"\2 \1", regex=True
        )

    df = expected.merge(
        pct_ranks.drop(columns=["player_name", "year"], errors="ignore"),
        on="key_mlbam",
        how="left",
    )
    df["join_source"] = "savant"
    return df


def _bref_savant_batting_merge(season: int, min_pa: int = 100) -> pd.DataFrame:
    """
    Merge Baseball Reference batting (traditional stats) with Baseball Savant
    (Statcast advanced metrics) using MLBAM ID as the join key.

    BRef provides clean raw rates (K%, BB%, ISO, OBP, AVG, SB, CS, Age).
    Savant provides Statcast metrics (xwOBA, Barrel%, Hard Hit%, whiff%, sprint speed).
    k_percent/bb_percent are dropped from Savant because BRef's raw rates are better.
    """
    log.info("Pulling Baseball Reference batting %d", season)
    bref = pybaseball.batting_stats_bref(season)
    bref = bref.rename(columns={"mlbID": "key_mlbam", "BA": "AVG"})
    bref["key_mlbam"] = pd.to_numeric(bref["key_mlbam"], errors="coerce")
    bref = bref[bref["key_mlbam"].notna() & (bref["PA"] >= min_pa)].copy()

    # Compute raw rate columns (will be percentile-ranked later by normalize_df).
    # Use K_rate/BB_rate to avoid collision with the normalization column naming
    # convention (BATTING_COL_MAP keys ending in _pct get misread as percentiles).
    bf_safe = bref["PA"].replace(0, np.nan)
    bref["K_rate"]  = bref["SO"] / bf_safe
    bref["BB_rate"] = bref["BB"] / bf_safe
    bref["ISO"]     = bref["SLG"] - bref["AVG"]

    log.info("BRef batting %d — %d qualifying rows", season, len(bref))

    # Savant Statcast columns (xwOBA, Barrel%, Hard Hit%, whiff%, sprint speed, etc.)
    savant = _savant_batting_aggregate(season, min_pa=min_pa)
    # Drop columns BRef covers with cleaner raw rates, and duplicates
    drop_savant = [
        "k_percent", "bb_percent",   # percentile ranks — use BRef raw rates instead
        "xiso", "xobp", "ba",        # BRef provides actual OBP/ISO/AVG
        "pa",                        # BRef has PA directly (avoids duplicate column after rename)
        "slg",                       # Savant has lowercase slg; BRef has SLG — drop to avoid dupe
        "Name", "join_source",
    ]
    savant = savant.drop(columns=[c for c in drop_savant if c in savant.columns], errors="ignore")

    df = bref.merge(savant, on="key_mlbam", how="left")
    df["join_source"] = "bref_savant"
    log.info("BRef+Savant batting %d — %d rows after merge", season, len(df))
    return df


def _bref_savant_pitching_merge(season: int, min_pa: int = 30) -> pd.DataFrame:
    """
    Merge Baseball Reference pitching (traditional stats) with Baseball Savant
    (Statcast metrics) using MLBAM ID as the join key.

    BRef provides GS (crucial for starter identification), BF, K%, BB%, Age.
    Savant provides fb_velocity, whiff%, hard_hit%, xwOBA_allowed, Barrel_allowed.
    """
    log.info("Pulling Baseball Reference pitching %d", season)
    bref = pybaseball.pitching_stats_bref(season)
    bref = bref.rename(columns={"mlbID": "key_mlbam"})
    bref["key_mlbam"] = pd.to_numeric(bref["key_mlbam"], errors="coerce")
    bref = bref[bref["key_mlbam"].notna() & (bref["BF"] >= min_pa)].copy()

    bf_safe = bref["BF"].replace(0, np.nan)
    bref["K_rate_pitch"]  = bref["SO"] / bf_safe
    bref["BB_rate_pitch"] = bref["BB"] / bf_safe

    log.info("BRef pitching %d — %d qualifying rows", season, len(bref))

    savant = _savant_pitching_aggregate(season, min_pa=min_pa)
    drop_savant = [
        "k_percent", "bb_percent",  # replaced by BRef raw rates
        "pa",                        # BRef has BF directly
        "Name", "join_source",
    ]
    savant = savant.drop(columns=[c for c in drop_savant if c in savant.columns], errors="ignore")

    df = bref.merge(savant, on="key_mlbam", how="left")
    df["join_source"] = "bref_savant"
    log.info("BRef+Savant pitching %d — %d rows after merge", season, len(df))
    return df


def pull_fg_batting(season: int, qual: int = 100) -> pd.DataFrame:
    """
    Pull season batting aggregates.

    Fallback chain:
      1. FanGraphs (blocked as of 2024+ with 403)
      2. Baseball Reference + Savant merge (BRef raw rates + Statcast advanced)
      3. Baseball Savant aggregates alone

    Caches to data/processed/fg_batting_{season}.csv.
    """
    cache = PROCESSED_DIR / f"fg_batting_{season}.csv"
    if cache.exists():
        log.info("Loading fg_batting_%d from cache", season)
        return pd.read_csv(cache, low_memory=False)

    # ── 1. Try FanGraphs ─────────────────────────────────────────────────────
    try:
        log.info("Pulling FanGraphs batting %d (qual=%d)", season, qual)
        df = pybaseball.batting_stats(season, season, qual=qual)
        log.info("FanGraphs batting %d — %d rows", season, len(df))
    except Exception as e:
        log.warning("FanGraphs batting unavailable (%s) — trying Baseball Reference", e)
        # ── 2. Try BRef + Savant merge ────────────────────────────────────────
        try:
            df = _bref_savant_batting_merge(season, min_pa=qual)
        except Exception as e2:
            log.warning("BRef batting unavailable (%s) — falling back to Savant only", e2)
            # ── 3. Savant alone ───────────────────────────────────────────────
            df = _savant_batting_aggregate(season, min_pa=qual)
            log.info("Savant batting %d — %d rows", season, len(df))

    df.to_csv(cache, index=False)
    log.info("Saved %d rows to %s", len(df), cache)
    return df


def pull_fg_pitching(season: int, qual: int = 30) -> pd.DataFrame:
    """
    Pull season pitching aggregates.

    Fallback chain:
      1. FanGraphs
      2. Baseball Reference + Savant merge
      3. Baseball Savant alone

    Caches to data/processed/fg_pitching_{season}.csv.
    """
    cache = PROCESSED_DIR / f"fg_pitching_{season}.csv"
    if cache.exists():
        log.info("Loading fg_pitching_%d from cache", season)
        return pd.read_csv(cache, low_memory=False)

    # ── 1. Try FanGraphs ─────────────────────────────────────────────────────
    try:
        log.info("Pulling FanGraphs pitching %d (qual=%d)", season, qual)
        df = pybaseball.pitching_stats(season, season, qual=qual)
        log.info("FanGraphs pitching %d — %d rows", season, len(df))
    except Exception as e:
        log.warning("FanGraphs pitching unavailable (%s) — trying Baseball Reference", e)
        # ── 2. Try BRef + Savant merge ────────────────────────────────────────
        try:
            df = _bref_savant_pitching_merge(season, min_pa=qual)
        except Exception as e2:
            log.warning("BRef pitching unavailable (%s) — falling back to Savant only", e2)
            # ── 3. Savant alone ───────────────────────────────────────────────
            df = _savant_pitching_aggregate(season, min_pa=qual)
            log.info("Savant pitching %d — %d rows", season, len(df))

    df.to_csv(cache, index=False)
    log.info("Saved %d rows to %s", len(df), cache)
    return df


# ---------------------------------------------------------------------------
# MLBAM ID attachment
# ---------------------------------------------------------------------------

def attach_mlbam_ids(df: pd.DataFrame, chadwick: pd.DataFrame) -> pd.DataFrame:
    """
    Join df to Chadwick to ensure every row has a valid MLBAM ID.

    Tries:
      1. Direct join on IDfg → key_fangraphs (FanGraphs ID)
      2. Name-based fallback on name_first + name_last

    Adds:
      key_mlbam     — MLBAM ID (NaN if join failed)
      join_source   — 'fangraphs_id' | 'name_fallback' | 'failed'
    """
    # Standardise Chadwick IDs
    cw = chadwick.copy()
    cw["key_fangraphs"] = pd.to_numeric(cw["key_fangraphs"], errors="coerce")
    cw["key_mlbam"]     = pd.to_numeric(cw["key_mlbam"],     errors="coerce")

    # ---- Pass 1: FanGraphs ID join ----------------------------------------
    if "IDfg" in df.columns:
        df["IDfg_num"] = pd.to_numeric(df["IDfg"], errors="coerce")
        fg_map = (
            cw[cw["key_fangraphs"].notna()]
            .drop_duplicates("key_fangraphs")
            [["key_fangraphs", "key_mlbam"]]
        )
        df = df.merge(
            fg_map.rename(columns={"key_fangraphs": "IDfg_num", "key_mlbam": "key_mlbam_fg"}),
            on="IDfg_num",
            how="left",
        )
        df["key_mlbam"]   = df.get("key_mlbam_fg")
        df["join_source"] = np.where(df["key_mlbam"].notna(), "fangraphs_id", "failed")
        df.drop(columns=["IDfg_num", "key_mlbam_fg"], errors="ignore", inplace=True)
    else:
        df["key_mlbam"]   = np.nan
        df["join_source"] = "failed"

    # ---- Pass 2: name fallback for unjoined rows --------------------------
    needs_name = df["join_source"] == "failed"
    if needs_name.any() and "Name" in df.columns:
        # Build name lookup from Chadwick
        cw["full_name"] = (
            cw["name_first"].fillna("").str.strip()
            + " "
            + cw["name_last"].fillna("").str.strip()
        ).str.lower()
        name_map = (
            cw[cw["key_mlbam"].notna()]
            .drop_duplicates("full_name")
            [["full_name", "key_mlbam"]]
        )
        df.loc[needs_name, "_name_lookup"] = df.loc[needs_name, "Name"].str.lower()
        df = df.merge(
            name_map.rename(columns={"full_name": "_name_lookup", "key_mlbam": "key_mlbam_name"}),
            on="_name_lookup",
            how="left",
        )
        recovered = df["key_mlbam"].isna() & df["key_mlbam_name"].notna()
        df.loc[recovered, "key_mlbam"]   = df.loc[recovered, "key_mlbam_name"]
        df.loc[recovered, "join_source"] = "name_fallback"
        df.drop(columns=["_name_lookup", "key_mlbam_name"], errors="ignore", inplace=True)

    # ---- Report ------------------------------------------------------------
    n_total   = len(df)
    n_ok      = df["key_mlbam"].notna().sum()
    n_failed  = n_total - n_ok
    log.info(
        "MLBAM join — total: %d | success: %d (%.1f%%) | failed: %d",
        n_total, n_ok, 100 * n_ok / max(n_total, 1), n_failed,
    )
    if n_failed:
        log.warning(
            "Failed rows (sample): %s",
            df[df["key_mlbam"].isna()][["Name"] if "Name" in df.columns else []].head(5).to_dict("records"),
        )

    return df


# ---------------------------------------------------------------------------
# Reliability scoring
# ---------------------------------------------------------------------------

def compute_reliability(sample_size: int, metric: str) -> float:
    """
    reliability = min(1.0, sample_size / threshold)

    Returns 0.0 if metric not in STABILIZATION.
    """
    threshold = STABILIZATION.get(metric)
    if threshold is None:
        log.debug("Metric '%s' not in STABILIZATION — returning 0.0", metric)
        return 0.0
    return min(1.0, sample_size / threshold)


def apply_reliability_weight(
    observed: float,
    reliability: float,
    league_mean: float = LEAGUE_MEAN_DEFAULT,
) -> float:
    """
    Shrink observed value toward league_mean by (1 - reliability).

    weighted = (reliability × observed) + ((1 - reliability) × league_mean)
    """
    return (reliability * observed) + ((1.0 - reliability) * league_mean)


# ---------------------------------------------------------------------------
# Percentile normalization
# ---------------------------------------------------------------------------

def normalize_percentile(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Rank each value within series as a percentile 0–100.

    invert=True for metrics where lower is better (K%, BB allowed, etc.).
    NaN values remain NaN.
    """
    ranked = series.rank(pct=True, na_option="keep") * 100
    if invert:
        ranked = 100 - ranked
    return ranked


# ---------------------------------------------------------------------------
# Convenience: normalize multiple columns in a DataFrame
# ---------------------------------------------------------------------------

INVERT_METRICS: set[str] = {
    "K_pct", "BB_pct_pitch", "HardHit_allowed", "Barrel_allowed",
    "Chase_pct", "BB_pct_allowed",
}


def normalize_df(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Add a '{col}_pct' column for each col in columns.

    Inverts automatically for known lower-is-better metrics.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            log.warning("normalize_df: column '%s' not found — skipping", col)
            continue
        invert = col in INVERT_METRICS
        out[f"{col}_pct"] = normalize_percentile(out[col], invert=invert)
    return out
